"""Archive full-val outputs and compare paired official HD scores."""

import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from common import ROOT, STORE, atomic_json, read_jsonl, sha256, utc_now, validate_scores


def paired_interval(difference, groups, repeats=10000, seed=42):
    """Resample source videos; retain query weighting within each resample."""
    _, inverse = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inverse)
    sums = np.stack([np.bincount(inverse, weights=difference[:, col]) for col in range(difference.shape[1])], axis=1)
    rng = np.random.default_rng(seed)
    bootstrap = []
    for begin in range(0, repeats, 250):
        indices = rng.integers(0, len(sizes), size=(min(250, repeats - begin), len(sizes)))
        bootstrap.append(sums[indices].sum(axis=1) / sizes[indices].sum(axis=1, keepdims=True))
    samples = np.concatenate(bootstrap)
    bounds = np.percentile(samples, [2.5, 97.5], axis=0)
    return {
        "delta_percentage_points": dict(zip(["mAP", "Hit1"], difference.mean(axis=0).tolist())),
        "ci95_percentage_points": dict(zip(["mAP", "Hit1"], bounds.T.tolist())),
        "query_map_win_tie_loss": [
            int(np.sum(difference[:, 0] > 1e-10)),
            int(np.sum(np.abs(difference[:, 0]) <= 1e-10)),
            int(np.sum(difference[:, 0] < -1e-10)),
        ],
    }


def archive_run(name):
    source = STORE / "runs" / name
    destination = ROOT / "results" / name
    completion = json.loads((source / "completion.json").read_text())
    assert completion["status"] == "complete" and completion["completed_queries"] == 1550
    report = json.loads((source / "evaluation_report.json").read_text())
    assert report["queries"] == 1550
    assert sha256(source / "predictions.jsonl") == completion["predictions_sha256"]
    assert report["predictions_sha256"] == completion["predictions_sha256"]
    destination.mkdir(parents=True, exist_ok=True)
    filenames = [
        "predictions.jsonl",
        "raw_predictions.jsonl",
        "metrics.json",
        "evaluation_report.json",
        "runtime.json",
        "completion.json",
        "binding.json",
    ]
    receipt = {}
    for name in filenames:
        original = source / name
        if original.exists():
            target = destination / name
            if target.exists():
                assert sha256(target) == sha256(original), f"Refusing to replace changed artifact: {target}"
            else:
                shutil.copy2(original, target)
            receipt[name] = sha256(original)
    shutil.copytree(source / "source_snapshot", destination / "source_snapshot", dirs_exist_ok=True)
    atomic_json(destination / "archive_receipt.json", {"source": str(source), "sha256": receipt})
    return source / "predictions.jsonl", report["metrics"], completion


def main():
    protocol = json.loads((ROOT / "configs/protocol_v1.json").read_text())
    audit = json.loads((ROOT / "manifests/data_audit.json").read_text())
    assert sha256(protocol["labels_path"]) == audit["label_sha256"]
    for filename, digest in audit["evaluator_hashes"].items():
        assert sha256(Path(protocol["evaluator_root"]) / "standalone_eval" / filename) == digest
    sys.path.insert(0, protocol["evaluator_root"])
    from standalone_eval.eval import compute_ap_from_tuple, eval_submission, mk_gt_scores

    labels = read_jsonl(protocol["labels_path"])
    manifest = read_jsonl(ROOT / "manifests/val_inference.jsonl")
    assert [r["qid"] for r in labels] == [r["qid"] for r in manifest]
    assert len(labels) == 1550
    all_results, values = {}, {}
    paths = {}
    for name in ["clip_val", "vtg_gpt_val"]:
        path, metrics, completion = archive_run(name)
        paths[name] = path
        all_results[name] = {"metrics": metrics, "completion": completion, "prediction_sha256": sha256(path)}

    merit = Path((os.environ.get("MERIT_ARTIFACT_ROOT", "artifacts/merit") + "/runs/merit_qvh_main_v1-15f5f6e4ee00"))
    paths["merit_historical_val"] = merit / "val_predictions.jsonl"
    historical = json.loads((merit / "val_report.json").read_text())
    historical_audit = json.loads((merit / "data_audit.json").read_text())
    assert historical_audit["val_annotations_sha256"] == audit["label_sha256"]
    all_results["merit_historical_val"] = {
        "metrics": historical["official_metrics"],
        "prediction_sha256": sha256(paths["merit_historical_val"]),
        "status_counts": historical["status_counts"],
        "source_report": str(merit / "val_report.json"),
        "source_report_sha256": sha256(merit / "val_report.json"),
        "historical_predictions_reused": True,
        "identity_check": "qid + vid + historical validation-label SHA256 (query text not stored in predictions)",
        "source_data_audit_sha256": sha256(merit / "data_audit.json"),
    }
    for name, path in paths.items():
        rows = read_jsonl(path)
        predictions = {r["qid"]: r for r in rows}
        assert len(rows) == len(predictions) == len(labels)
        assert set(predictions) == {r["qid"] for r in labels}
        per_query = []
        for row in labels:
            prediction = predictions[row["qid"]]
            assert prediction["vid"] == row["vid"]
            if name != "merit_historical_val":
                assert prediction["query"] == row["query"]
            scores = np.array(prediction["pred_saliency_scores"])
            validate_scores(scores.tolist(), int(row["duration"] / 2))
            binary = (mk_gt_scores(row) >= 4).astype(float)
            average_ap = np.mean([compute_ap_from_tuple((0, w, binary[:, w], scores))[2] for w in range(3)])
            hit = binary[int(np.argmax(scores))].max()
            per_query.append([100 * average_ap, 100 * hit])
        values[name] = np.asarray(per_query)
        rounded = np.round(values[name].mean(axis=0), 2)
        expected = all_results[name]["metrics"]["HL-min-VeryGood"]
        np.testing.assert_allclose(rounded, [expected["HL-mAP"], expected["HL-Hit1"]], atol=1e-9)
        if name == "merit_historical_val":
            hd = [{k: r[k] for k in ["qid", "vid", "pred_saliency_scores"]} for r in rows]
            assert eval_submission(hd, labels, verbose=False, match_number=True) == historical["official_metrics"]
        all_results[name]["unrounded_VeryGood"] = dict(zip(["mAP", "Hit1"], values[name].mean(axis=0).tolist()))

    groups = np.array([r["vid"].rsplit("_", 2)[0] for r in labels])
    pairs = {}
    for first, second in [
        ("vtg_gpt_val", "clip_val"),
        ("clip_val", "merit_historical_val"),
        ("vtg_gpt_val", "merit_historical_val"),
    ]:
        pairs[f"{first}_minus_{second}"] = paired_interval(values[first] - values[second], groups)

    traces = [json.loads(p.read_text()) for p in sorted((STORE / "cache/clip_center_v1").glob("*/trace.json"))]
    assert {t["vid"] for t in traces} == {r["vid"] for r in labels}
    clip_audit = {
        "videos": len(traces),
        "sampled_frames": sum(t["frame_count"] for t in traces),
        "terminal_centers_clamped": sum(t["terminal_centers_clamped"] for t in traces),
        "videos_with_terminal_clamp": sum(t["terminal_centers_clamped"] > 0 for t in traces),
        "video_feature_encoding_seconds_sum": sum(t["encode_seconds"] for t in traces),
        "decode_preprocess_worker_seconds_sum": sum(t["decode_preprocess_seconds"] for t in traces),
        "traces_sha256": {t["vid"]: sha256(STORE / "cache/clip_center_v1" / t["vid"] / "trace.json") for t in traces},
    }
    atomic_json(ROOT / "results/clip_media_audit.json", clip_audit)
    summary = {
        "created_at": utc_now(),
        "split": "val",
        "queries": len(labels),
        "video_segments": len({r["vid"] for r in labels}),
        "source_videos": len(set(groups)),
        "label_sha256": audit["label_sha256"],
        "evaluator_hashes": audit["evaluator_hashes"],
        "results": all_results,
        "paired_comparisons": pairs,
        "uncertainty": {
            "unit": "source-video cluster",
            "repeats": 10000,
            "seed": 42,
            "interval": "95% percentile; exploratory, no multiple-comparison adjustment",
        },
        "clip_media_audit": {k: v for k, v in clip_audit.items() if k != "traces_sha256"},
    }
    atomic_json(ROOT / "results/summary.json", summary)
    with (ROOT / "results/paired_per_query.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["qid", "vid", "source_video"] + [f"{name}_{metric}" for name in values for metric in ["mAP", "Hit1"]]
        )
        for index, row in enumerate(labels):
            writer.writerow(
                [row["qid"], row["vid"], groups[index]] + [v for name in values for v in values[name][index]]
            )
    print(
        json.dumps(
            {
                "queries": len(labels),
                "methods": {n: r["metrics"]["brief"] for n, r in all_results.items()},
                "paired": pairs,
                "clip_media": summary["clip_media_audit"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
