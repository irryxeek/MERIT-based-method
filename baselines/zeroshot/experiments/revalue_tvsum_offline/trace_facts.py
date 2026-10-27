"""Supplement the finished offline audit with every raw fact's source chain."""

import argparse
import ast
from pathlib import Path

from audit import compact, install_offline_guard, read, sha, write_new


def main(parent, output):
    install_offline_guard()
    audit = read(output / "audit.json")
    assert sha(output / "audit.json") == read(output / "preparation_receipt.json")["audit_sha256"]
    before = read(output / "before_receipts.json")
    manifest = read(parent / "manifest.json")
    prompt_path = parent / "source_snapshot/experiments/revalue/prompts.py"
    prompt_node = next(n for n in ast.parse(prompt_path.read_text()).body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "FACTS" for t in n.targets))
    prompt = ast.literal_eval(prompt_node.value)
    lineages = []
    for slot, vids in enumerate(manifest["shards"]):
        folder = parent / f"shard_{slot}"
        keys = {a["key"]: a["number"] for a in read(folder / "ledger.json")["attempts"]}
        for vid in vids:
            record_path = folder / "records" / f"{vid}.json"
            assert sha(record_path) == before[str(record_path)]["sha256"]
            record = read(record_path)
            reconstructed = {}
            for candidate in record["candidate_ids"]:
                key = f"{vid}/facts/{candidate}"
                raw_path = folder / "attempts" / f"{keys[key]:05d}.json"
                assert sha(raw_path) == before[str(raw_path)]["sha256"]
                raw = read(raw_path)
                trace = record["candidate_frames"][str(candidate)]
                frames = [{"id": f"s{f['frame_index']}", "pts": f["frame_timestamps"]} for f in trace["frames"]]
                assert raw["specification"]["text"] == prompt.format(frames=compact(frames))
                assert raw["specification"]["evidence"] == [trace]
                ids = []
                for index, value in enumerate(raw["parsed"]["facts"]):
                    identity = f"c{candidate}_f{index}"
                    assert set(value["frames"]) <= {f["id"] for f in frames}
                    reconstructed[identity] = {"id": identity, "candidate": candidate, **value}
                    ids.append(identity)
                for index, text in enumerate(raw["parsed"]["unknown"]):
                    identity = f"c{candidate}_u{index}"
                    reconstructed[identity] = {"id": identity, "candidate": candidate,
                                               "kind": "unknown", "text": text, "frames": []}
                    ids.append(identity)
                lineages.append({"vid": vid, "candidate": candidate, "source_attempt": str(raw_path.relative_to(parent)),
                    "source_attempt_sha256": sha(raw_path), "frame_map": frames,
                    "candidate_tensor_sha256": trace["model_rgb_sha256"], "fact_ids": ids,
                    "prompt_is_frozen_query_free_fact_template": True})
            assert reconstructed == record["facts"]
    assert len(lineages) == 50 and sum(len(x["fact_ids"]) for x in lineages) == 448
    files = {
        "experiments/revalue_crossdataset/core.py": {"one_video"},
        "experiments/revalue/schema.py": {"validate_task"},
        "experiments/revalue/comparison.py": {"validate_comparison", "rerank"},
        "experiments/revalue/memory.py": {"build_memory"},
        "experiments/revalue/backend.py": {"citation_schema"},
    }
    locations = []
    for name, functions in files.items():
        path = parent / "source_snapshot" / name
        assert sha(path) == manifest["source_sha256"][name]
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name in functions:
                locations.append({"snapshot_file": name, "sha256": sha(path), "function": node.name,
                                  "start_line": node.lineno, "end_line": node.end_lineno})
    write_new(output / "fact_lineage.json", {"marker": audit["marker"], "audit_sha256": sha(output / "audit.json"),
        "code_sha256": sha(Path(__file__)), "candidate_fact_requests": 50, "matched_fact_records": 448,
        "lineages": lineages, "source_locations": locations, "new_model_calls": 0, "labels_read": False,
        "scope": "byte/source-chain validation; does not establish visual factual accuracy"})
    print("Matched 448 fact records to all 50 original raw fact requests; zero model calls.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.parent.resolve(), args.output.resolve())
