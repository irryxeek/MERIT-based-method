"""Read-only R2 trajectory audit, frozen-ranker replay and source-frame review sheets."""

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def edges(arm):
    return {tuple(sorted((e["a"], e["b"]))): e for e in arm["ranking"]["edges"]}


def normalized(text):
    return " ".join(re.findall(r"\w+", text.lower()))


def memory_audit(record, arm):
    old = record["memories"]["E_memory"]
    new = record["memories"][arm]
    lost = sorted(set(old["retained_fact_ids"]) - set(new["retained_fact_ids"]))
    before, after = edges(record["arms"]["E_memory"]), edges(record["arms"][arm])
    transitions = Counter(f"{e['state']}->{after[p]['state']}" for p, e in before.items())
    observations = []
    for ob in record["observations"][arm]:
        trace = {f["id"]: f for f in ob["trace"]["frames"]}
        target = record["candidate_frames"][str(ob["action"]["candidate"])]
        facts = []
        for fact in ob["result"]["facts"]:
            pts = [trace[i]["pts"][0] for i in fact["frames"]]
            # Source PTS at a requested clip boundary may precede it by <1 frame.
            tolerance = max(f["pts"][1] - f["pts"][0] for f in trace.values())
            positions = [target["start"] - tolerance <= t < target["end"] for t in pts]
            facts.append({
                **fact, "cited_pts": pts,
                "citation_position": "inside" if all(positions) else "mixed" if any(positions) else "outside",
                "exact_query_echo": normalized(fact["text"]) == normalized(record["query"]),
            })
        observations.append({"action": ob["action"], "facts": facts, "unknown": ob["result"]["unknown"]})
    return {
        "tokens_before": old["tokens"], "tokens_after": new["tokens"], "lost_old_fact_ids": lost,
        "lost_old_facts": {k: record["facts"][k] for k in lost},
        "pair_transitions": dict(transitions), "observations": observations,
        "memory_before": old, "memory_after": new,
        "comparisons_before": record["arms"]["E_memory"]["comparisons"],
        "comparisons_after": record["arms"][arm]["comparisons"],
    }


def review_sheet(row, record, output):
    import decord
    import numpy as np
    from PIL import Image, ImageDraw

    reader = decord.VideoReader(row["video_path"], num_threads=1)
    wanted, panels, checked = {}, [], []
    for candidate, trace in record["candidate_frames"].items():
        for f in (trace["frames"][0], trace["frames"][-1]):
            i = f["frame_index"]
            wanted[i] = None
            panels.append((i, f"old c{candidate} s{i} {f['frame_timestamps'][0]:.2f}s"))
    for arm in ("E_uniform", "E_targeted"):
        for ob in record["observations"][arm]:
            trace = {f["id"]: f for f in ob["trace"]["frames"]}
            cited = sorted({i for fact in ob["result"]["facts"] for i in fact["frames"]}, key=lambda i: trace[i]["frame_index"])
            for identity in cited:
                f = trace[identity]
                wanted[f["frame_index"]] = f["source_rgb_sha256"]
                checked.append(f)
            if cited:
                for k in sorted(set(np.linspace(0, len(cited) - 1, min(4, len(cited))).round().astype(int))):
                    f = trace[cited[k]]
                    panels.append((f["frame_index"], f"{arm[2:]} c{ob['action']['candidate']} {f['id']} {f['pts'][0]:.2f}s"))
    indices = sorted(wanted)
    pixels = reader.get_batch(indices).asnumpy()
    images = dict(zip(indices, pixels, strict=True))
    for i, expected in wanted.items():
        if expected and hashlib.sha256(images[i].tobytes()).hexdigest() != expected:
            raise ValueError(f"Source pixel drift: {row['qid']} {i}")
    for f in checked:
        if np.asarray(reader.get_frame_timestamp(f["frame_index"])).tolist() != f["pts"]:
            raise ValueError("Source timestamp drift")
    width, height = 248, 168
    canvas = Image.new("RGB", (width * 4, math.ceil(len(panels) / 4) * height + 48), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), f"qid={row['qid']}  assistant review only; original source pixels verified", fill="black")
    draw.text((8, 22), row["query"][:125], fill="black")
    for number, (index, label) in enumerate(panels):
        x, y = (number % 4) * width, (number // 4) * height + 48
        im = Image.fromarray(images[index])
        im.thumbnail((width - 4, height - 26))
        canvas.paste(im, (x + 2, y + 24))
        draw.text((x + 2, y + 4), label, fill="black")
    path = output / f"{row['qid']}.jpg"
    canvas.save(path, quality=92)
    return {"path": str(path), "sha256": sha(path), "cited_frame_checks": len(checked), "unique_decoded_frames": len(indices)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    run = args.run.resolve()
    sys.path.insert(0, str(run / "source_snapshot/experiments/revalue"))
    from comparison import rerank
    from io_utils import fixed_twenty

    evaluation_path = run / "R2/results/evaluation.json"
    if read(run / "complete.json")["evaluation_sha256"] != sha(evaluation_path):
        raise ValueError("Completed evaluation changed")
    rows = [json.loads(line) for line in (run / "inference.jsonl").read_text().splitlines()]
    controls = {row["qid"] for row in fixed_twenty(rows)}
    all_states, preferences, eligibility = {}, {}, {}
    replays, details, hashes, cases, tie_conflicts = 0, [], {}, [], []
    for row in rows:
        path = run / "R2/records" / f"{row['qid']}.json"
        record = read(path)
        hashes[str(row["qid"])] = sha(path)
        summary = {"qid": row["qid"], "query": row["query"], "R1_control": row["qid"] in controls, "arms": {}}
        for arm, value in record["arms"].items():
            scores, ranking = rerank(record["original"], record["candidate_ids"], value["comparisons"])
            if scores != value["scores"] or ranking != value["ranking"]:
                raise ValueError(f"Rank replay mismatch: {row['qid']}/{arm}")
            replays += 1
            states = Counter(e["state"] for e in ranking["edges"])
            all_states.setdefault(arm, Counter()).update(states)
            preferences.setdefault(arm, Counter()).update(c["result"]["preference"] for c in value["comparisons"])
            eligibility.setdefault(arm, Counter()).update(c["result"][s + "_eligibility"] for c in value["comparisons"] for s in ("a", "b"))
            for c in value["comparisons"]:
                result = c["result"]
                if result["preference"] == "tie" and {result["a_eligibility"], result["b_eligibility"]} == {"satisfied", "violated"}:
                    tie_conflicts.append({"qid": row["qid"], "arm": arm, "a": c["a"], "b": c["b"], "result": result})
            summary["arms"][arm] = {"states": dict(states), "scores_equal_B0": scores == record["original"]}
        observed = bool(record.get("observations", {}).get("E_uniform"))
        summary["observed"] = observed
        if observed or row["qid"] in controls:
            case = {"qid": row["qid"], "query": row["query"], "task": record["task"], "planning": record.get("planning"), "candidate_frames": record["candidate_frames"], "observed": observed}
            if observed:
                case["audit"] = {arm: memory_audit(record, arm) for arm in ("E_uniform", "E_targeted")}
                if args.frames:
                    case["source_review"] = review_sheet(row, record, args.output)
            else:
                case["comparisons"] = {arm: value["comparisons"] for arm, value in record["arms"].items()}
            write(args.output / f"case_{row['qid']}.json", case)
            cases.append({"qid": row["qid"], "observed": observed, "R1_control": row["qid"] in controls})
        details.append(summary)
    result = {
        "at": datetime.now(timezone.utc).isoformat(), "run": str(run), "evaluation_sha256": sha(evaluation_path),
        "records_sha256": hashes, "rank_replays_passed": replays,
        "comparison_states": {k: dict(v) for k, v in all_states.items()},
        "directed_preferences": {k: dict(v) for k, v in preferences.items()},
        "eligibility": {k: dict(v) for k, v in eligibility.items()},
        "tie_with_satisfied_and_violated": tie_conflicts,
        "queries": details, "review_cases": cases, "new_model_calls": 0,
        "human_review": "not_reviewed", "script_sha256": sha(__file__),
    }
    write(args.output / "audit.json", result)
    print(json.dumps({k: result[k] for k in ("rank_replays_passed", "comparison_states", "directed_preferences", "eligibility", "new_model_calls")}, indent=2))


if __name__ == "__main__":
    main()
