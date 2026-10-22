"""Audit saved M0/D0/H1 output relationships without reading labels or calling models."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from merit_qvh.core import digest, file_sha256, read_jsonl, write_json


def audit(root):
    rows = read_jsonl(root / "data/manifests/explore.jsonl")
    expected = {row["qid"]: row for row in rows}
    completed, failures = [], []
    statuses = {g: Counter() for g in ("M0", "D0", "H1")}
    for path in sorted((root / "queries").glob("*.json")):
        result = json.loads(path.read_text())
        row = expected[result["qid"]]
        n = int(row["duration"] / 2)
        if result["qid"] in completed:
            raise ValueError("Duplicate query output")
        completed.append(result["qid"])
        for group, maximum in (("M0", 100), ("D0", 4), ("H1", 4)):
            output = result[group]
            assert output["qid"] == row["qid"] and output["vid"] == row["vid"]
            scores = output["pred_saliency_scores"]
            assert len(scores) == n
            assert all(type(s) in (int, float) and math.isfinite(s) and 0 <= s <= maximum for s in scores)
            statuses[group][output["status"]] += 1
            if output["status"] == "failed":
                assert scores == [0.0] * n
                failures.append({"qid": row["qid"], "group": group, "error": output["error"]})
        assert result["retrieval_trace_sha256"] == digest(result["M0"].get("rounds", []))
        if result["D0"]["status"] == "ok":
            hashes = {}
            for batch in result["D0"]["batches"]:
                for i, sha in zip(batch["ids"], batch["frame_hashes"], strict=True):
                    assert i not in hashes
                    hashes[i] = sha
            assert set(hashes) == set(range(n))
        if result["H1"]["status"] == "ok":
            assert result["D0"]["status"] == "ok"
            h1 = result["H1"]
            updated = h1["updated_clip_ids"]
            assert updated == sorted(set(updated))
            memory = json.loads((root / "legacy/memories" / (row["vid"] + ".json")).read_text())
            selected = result["M0"]["selected_clip_ids"]
            mapped = sorted(
                {
                    i
                    for anchor in selected
                    for i in range(n)
                    if 2 * i < memory["clips"][anchor]["end"]
                    and 2 * (i + 1) > memory["clips"][anchor]["start"]
                }
            )
            assert updated == mapped
            assert updated == [i for batch in h1["batches"] for i in batch["ids"]]
            for i in set(range(n)) - set(updated):
                assert h1["pred_saliency_scores"][i] == result["D0"]["pred_saliency_scores"][i]
            for batch in h1["batches"]:
                assert 0 < len(batch["ids"]) <= 24
                assert batch["frame_hashes"] == [hashes[i] for i in batch["ids"]]
                assert [s["score"] for s in batch["response"]["clip_scores"]] == [
                    h1["pred_saliency_scores"][i] for i in batch["ids"]
                ]
    return {
        "planned": len(rows),
        "completed": len(completed),
        "missing_qids": sorted(set(expected) - set(completed)),
        "status_counts": {g: dict(c) for g, c in statuses.items()},
        "failures": failures,
        "invariants": "passed",
        "labels_read": False,
        "model_calls": 0,
        "auditor_sha256": file_sha256(__file__),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run_root)
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False))
