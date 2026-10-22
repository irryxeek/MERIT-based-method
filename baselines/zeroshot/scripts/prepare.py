"""Audit released inputs and make label-free inference manifests."""

import ast
import json
import os
import subprocess
import zipfile
from pathlib import Path

from common import ROOT, STORE, atomic_json, read_jsonl, sha256, utc_now, validate_manifest, write_jsonl


def main():
    source = ROOT / "third_party/VTG-GPT"
    evaluator = Path(os.environ.get("QVH_EVALUATOR_ROOT", "third_party/moment_detr"))
    labels = evaluator / "data/highlight_val_release.jsonl"
    video_root = Path((os.environ.get("REPRO_DATA_ROOT", "data") + "/QVHighlights/videos_full/videos"))
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != "9d6606140edb36c89ea5ce96a0dc5e50361d108a":
        raise ValueError("Unexpected VTG-GPT revision")
    gt = read_jsonl(labels)
    published = read_jsonl(source / "data/qvhighlights/query/val.jsonl")
    lookup = {r["qid"]: r for r in published}
    assert len(gt) == len(lookup) == len(published) == 1550
    assert {r["qid"] for r in gt} == set(lookup)
    assert sha256(labels) == sha256(source / "data/qvhighlights/gt/highlight_val_release.jsonl")
    caption_root = STORE / "cache/vtg_gpt_captions_val"
    caption_root.mkdir(parents=True, exist_ok=True)
    rows, tails, caption_hashes = [], [], {}
    archive = source / "data/qvhighlights/caption/val.zip"
    with zipfile.ZipFile(archive) as zipped:
        members = {Path(n).stem: n for n in zipped.namelist() if n.endswith(".jsonl") and not n.startswith("__MACOSX/")}
        for row in gt:
            upstream = lookup[row["qid"]]
            assert all(row[k] == upstream[k] for k in ["qid", "vid", "query", "duration"])
            assert all(isinstance(q, str) for q in upstream["rephrased_query"])
            caption_path = caption_root / (row["vid"] + ".jsonl")
            data = zipped.read(members[row["vid"]])
            if caption_path.exists():
                assert caption_path.read_bytes() == data
            else:
                caption_path.write_bytes(data)
            captions = read_jsonl(caption_path)
            count = int(row["duration"] / 2)
            assert len(captions) in (count, count - 1)
            assert [c["moment"] for c in captions] == list(range(len(captions)))
            assert all(isinstance(c["description"], str) and c["description"] for c in captions)
            if len(captions) != count:
                tails.append({"qid": row["qid"], "vid": row["vid"], "captions": len(captions), "clips": count})
            video_path = video_root / (row["vid"] + ".mp4")
            assert video_path.is_file(), video_path
            rows.append(
                {
                    **{k: row[k] for k in ["qid", "vid", "query", "duration"]},
                    "video_path": str(video_path),
                    "caption_path": str(caption_path),
                    "rephrased_query": upstream["rephrased_query"],
                }
            )
            caption_hashes[row["vid"]] = sha256(caption_path)
    assert len(tails) == 13
    validate_manifest(rows)
    manifest = ROOT / "manifests/val_inference.jsonl"
    write_jsonl(manifest, rows)
    # Deterministic smoke: first four records plus the first short-caption record.
    smoke_qids = {r["qid"] for r in rows[:4]} | {tails[0]["qid"]}
    write_jsonl(ROOT / "manifests/smoke_inference.jsonl", [r for r in rows if r["qid"] in smoke_qids])
    equivalent = {}
    for name in ["eval.py", "utils.py"]:
        functions = []
        for directory in [source / "standalone_eval", evaluator / "standalone_eval"]:
            tree = ast.parse((directory / name).read_text())
            functions.append(
                {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}
            )
        for key in [
            "eval_highlight",
            "compute_hl_hit1",
            "compute_hl_ap",
            "compute_ap_from_tuple",
            "mk_gt_scores",
            "get_ap",
        ]:
            if key in functions[0]:
                equivalent[key] = functions[0][key] == functions[1].get(key)
    assert all(equivalent.values())
    audit = {
        "created_at": utc_now(),
        "queries": len(rows),
        "videos": len(caption_hashes),
        "upstream_commit": commit,
        "manifest_sha256": sha256(manifest),
        "label_path": str(labels),
        "label_sha256": sha256(labels),
        "caption_archive_sha256": sha256(archive),
        "caption_hashes": caption_hashes,
        "missing_tail_clips": tails,
        "official_hd_functions_identical": equivalent,
        "evaluator_hashes": {n: sha256(evaluator / "standalone_eval" / n) for n in ["eval.py", "utils.py"]},
        "upstream_scoring_sha256": sha256(source / "infer_qvhighlights.py"),
    }
    atomic_json(ROOT / "manifests/data_audit.json", audit)
    protocol = {
        "schema": "qvh_zeroshot_v1",
        "split": "val",
        "expected_queries": 1550,
        "evaluator_root": str(evaluator),
        "labels_path": str(labels),
        "manifest": str(manifest),
        "data_audit_sha256": sha256(ROOT / "manifests/data_audit.json"),
        "primary_metrics": ["HL-min-VeryGood-mAP", "HL-min-VeryGood-Hit1"],
        "vtg_gpt": {
            "upstream_commit": commit,
            "model": "sentence-transformers/paraphrase-distilroberta-base-v2",
            "revision": "1972a507973d92e15f4f7cd9457d9fcf43ea117f",
            "local_model": str(STORE / "models/paraphrase-distilroberta-base-v2"),
            "num_bins": 10,
            "top_k": 8,
            "gap": 6,
            "nms_threshold": 0.95,
            "dtype": "float32",
            "use_released_captions_and_rephrased_queries": True,
            "missing_tail_policy": "pad one absent clip with zero, equivalent to official HD evaluator",
            "failure_policy": "stop and retain checkpoint; never silently exclude queries",
        },
    }
    atomic_json(ROOT / "configs/protocol_v1.json", protocol)
    print(
        json.dumps(
            {
                "queries": len(rows),
                "videos": len(caption_hashes),
                "short_caption_records": len(tails),
                "manifest": str(manifest),
            }
        )
    )


if __name__ == "__main__":
    main()
