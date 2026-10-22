"""Run the pinned, unmodified author's scorer on released intermediates."""

import argparse
import fcntl
import importlib.metadata
import json
import os
import sys
import time
from collections import OrderedDict

from common import ROOT, STORE, atomic_json, pad_vtg_tail, read_jsonl, sha256, utc_now, validate_manifest, write_jsonl


class CachedEncoder:
    """Reuse identical caption encodings; call the same model with identical arguments."""

    def __init__(self, model, capacity=16):
        self.model = model
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def encode(self, text, **kwargs):
        key = (tuple(text) if isinstance(text, list) else text, tuple(sorted(kwargs.items())))
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        value = self.model.encode(text, **kwargs)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = ROOT / "configs/protocol_v1.json"
    protocol = json.loads(protocol_path.read_text())
    config = protocol["vtg_gpt"]
    audit = json.loads((ROOT / "manifests/data_audit.json").read_text())
    source = ROOT / "third_party/VTG-GPT"
    assert sha256(source / "infer_qvhighlights.py") == audit["upstream_scoring_sha256"]
    manifest = ROOT / "manifests" / ("smoke_inference.jsonl" if args.smoke else "val_inference.jsonl")
    rows = read_jsonl(manifest)
    validate_manifest(rows)
    if not args.smoke:
        assert sha256(manifest) == audit["manifest_sha256"]
        gate = json.loads((STORE / "runs/vtg_gpt_smoke/completion.json").read_text())
        assert gate["status"] == "complete" and gate["cache_equivalence_verified"]
    run = STORE / "runs" / ("vtg_gpt_smoke" if args.smoke else "vtg_gpt_val")
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    binding = {
        "protocol_sha256": sha256(protocol_path),
        "manifest_sha256": sha256(manifest),
        "runner_sha256": sha256(__file__),
        "common_sha256": sha256(ROOT / "scripts/common.py"),
    }
    if (run / "binding.json").exists():
        assert json.loads((run / "binding.json").read_text()) == binding, "Frozen run changed"
    else:
        atomic_json(run / "binding.json", binding)
    started = time.time()
    invocation = utc_now()
    status = {
        "status": "loading_model",
        "invocation_started": invocation,
        "planned_queries": len(rows),
        "completed_queries": 0,
    }
    atomic_json(run / "progress.json", status)
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, str(source))
    from infer_qvhighlights import VTG_GPT

    model = SentenceTransformer(
        config["local_model"], device="cuda:0", local_files_only=True, model_kwargs={"attn_implementation": "eager"}
    )
    model.eval()
    cache = CachedEncoder(model)
    scorer = VTG_GPT.__new__(VTG_GPT)
    for key in ["num_bins", "top_k", "gap", "nms_threshold"]:
        setattr(scorer, key, config[key])
    scorer.similarity_model = cache
    package_names = ["torch", "transformers", "sentence-transformers", "numpy", "scikit-learn"]
    runtime = {
        "python": sys.version,
        "packages": {n: importlib.metadata.version(n) for n in package_names},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0),
        "max_seq_length": model.max_seq_length,
        "model_dtype": str(next(model.parameters()).dtype),
        "model_load_seconds": time.time() - started,
        "model_files_sha256": {
            str(p.relative_to(config["local_model"])): sha256(p)
            for p in sorted(__import__("pathlib").Path(config["local_model"]).rglob("*"))
            if p.is_file() and ".cache" not in p.parts
        },
    }
    atomic_json(run / "runtime.json", runtime)
    outputs = []
    for idx, row in enumerate(rows):
        record_path = run / "queries" / f"{row['qid']}.json"
        if record_path.exists():
            record = json.loads(record_path.read_text())
            assert record["qid"] == row["qid"] and record["status"] == "ok"
            outputs.append(record)
            continue
        t0 = time.time()
        try:
            assert sha256(row["caption_path"]) == audit["caption_hashes"][row["vid"]]
            captions = [c["description"] for c in read_jsonl(row["caption_path"])]
            with torch.inference_mode():
                raw = scorer.locate_span(
                    row["qid"], row["vid"], row["query"], captions, row["rephrased_query"], gt_span_list=None
                )
            # The inference process never loads GT labels.
            raw.pop("relevant_windows", None)
            raw.pop("span_index_list", None)
            scores = pad_vtg_tail(raw["pred_saliency_scores"], int(row["duration"] / 2))
            if args.smoke:
                scorer.similarity_model = model
                with torch.inference_mode():
                    uncached = scorer.locate_span(
                        row["qid"], row["vid"], row["query"], captions, row["rephrased_query"], gt_span_list=None
                    )
                np.testing.assert_array_equal(raw["pred_saliency_scores"], uncached["pred_saliency_scores"])
                np.testing.assert_array_equal(raw["pred_relevant_windows"], uncached["pred_relevant_windows"])
                scorer.similarity_model = cache
            torch.cuda.synchronize()
            record = {
                "qid": row["qid"],
                "vid": row["vid"],
                "status": "ok",
                "finished_at": utc_now(),
                "elapsed_seconds": time.time() - t0,
                "raw_prediction": raw,
                "prediction": {**raw, "pred_saliency_scores": scores},
                "tail_zero_padding": len(scores) - len(raw["pred_saliency_scores"]),
            }
            atomic_json(record_path, record)
            outputs.append(record)
        except Exception as exc:
            atomic_json(
                run / "failures" / f"{row['qid']}_{int(time.time())}.json",
                {"qid": row["qid"], "error_type": type(exc).__name__, "error": str(exc), "time": utc_now()},
            )
            atomic_json(
                run / "progress.json",
                {**status, "status": "failed", "failed_qid": row["qid"], "completed_queries": len(outputs)},
            )
            raise
        status.update(
            status="running",
            completed_queries=len(outputs),
            updated_at=utc_now(),
            invocation_elapsed_seconds=time.time() - started,
            cache_hits=cache.hits,
            cache_misses=cache.misses,
        )
        atomic_json(run / "progress.json", status)
        if args.smoke or (idx + 1) % 50 == 0:
            print(json.dumps(status), flush=True)
    write_jsonl(run / "predictions.jsonl", [r["prediction"] for r in outputs])
    write_jsonl(run / "raw_predictions.jsonl", [r["raw_prediction"] for r in outputs])
    complete = {
        **status,
        "status": "complete",
        "completed_queries": len(outputs),
        "finished_at": utc_now(),
        "cache_equivalence_verified": args.smoke or gate["cache_equivalence_verified"],
        "tail_zero_padded_queries": sum(r["tail_zero_padding"] > 0 for r in outputs),
        "scoring_elapsed_seconds_sum": sum(r["elapsed_seconds"] for r in outputs),
        "invocation_elapsed_seconds": time.time() - started,
        "predictions_sha256": sha256(run / "predictions.jsonl"),
        "failures": 0,
    }
    atomic_json(run / "completion.json", complete)
    atomic_json(run / "progress.json", complete)
    print(json.dumps(complete), flush=True)


if __name__ == "__main__":
    main()
