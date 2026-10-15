"""Fixed real request replay; execution benchmark, never prediction-quality evidence."""

import _bootstrap  # noqa: F401
import base64
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from videoarm.qvhighlights.backend import LocalBackend
from videoarm.qvhighlights.common import ARTIFACTS, atomic_json, digest, load_manifest
from videoarm.qvhighlights.contracts import clip_scores, windows
from videoarm.qvhighlights.media import visual_message
from videoarm.qvhighlights.pipeline import parse_action, parse_events


def main():
    root = ARTIFACTS.parents[1] / "qvh_data"
    output = root / "probes/concurrency_m1_v1"
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ARTIFACTS / "configs/default_v2.json").read_text())
    manifest = load_manifest(ARTIFACTS / "manifests/smoke8.jsonl")
    # Register by identifiers and stage, before replay output or timing is observed.
    qids = sorted(r["qid"] for r in manifest)[:2]
    specifications = [
        ("mr", qids[0], s) for s in ("overview", "controller_0", "tool_0", "final")
    ] + [("mr", qids[1], s) for s in ("overview", "controller_0")]
    specifications += [
        ("hd", q, s) for q in qids for s in ("initialize_0", "initialize_24", "initialize_48")
    ]
    cases = []
    for task, qid, stage in specifications:
        name = "dev_v2_mr_smoke8_M1" if task == "mr" else "dev_v1_hd_smoke8_B0"
        db = sqlite3.connect(f"file:{root / 'runs' / name / 'state.sqlite'}?mode=ro", uri=True)
        request = json.loads(db.execute(
            "SELECT request FROM requests WHERE qid=? AND stage=? AND attempt=0", (qid, stage)
        ).fetchone()[0])
        db.close()
        frames = []
        for f in request["frames"]:
            encoded = base64.b64encode(Path(f["path"]).read_bytes()).decode()
            if digest(encoded) != f["jpeg_sha256"]:
                raise ValueError("benchmark recorded pixel hash drift")
            frames.append({**f, "url": "data:image/jpeg;base64," + encoded})
        messages = [
            {"role": "system", "content": request["system"]},
            visual_message(request["text"], frames),
        ]
        context = json.loads(request["text"].split("\n", 1)[0])
        cases.append({"task": task, "qid": qid, "stage": stage, "source": name,
                      "request": request, "messages": messages, "context": context,
                      "request_sha256": digest([messages, request["max_tokens"], config["seed"]])})
    order = [1, 4, 2, 2, 4, 1]  # Balance warm-cache/order effects; two repeats per setting.
    atomic_json(output / "registered.json", {
        "cases": [{k: v for k, v in c.items() if k not in {"messages", "request", "context"}} for c in cases],
        "concurrency_order": order, "max_actual_requests": len(cases) * len(order),
        "same_recorded_pixels_messages_model_generation": True,
        "scope": "request replay, not end-to-end query or algorithm-effect benchmark",
    })

    def execute(case):
        backend = LocalBackend(config)
        started = time.monotonic()
        try:
            response = backend.complete(case["messages"], case["request"]["max_tokens"], 300)
            elapsed = time.monotonic() - started
            error = None
            try:
                if response["finish_reason"] != "stop":
                    raise ValueError("nonterminal/truncated response")
                parsed = json.loads(response["raw"])
                task, stage = case["task"], case["stage"]
                duration = case["context"]["duration"]
                if task == "hd":
                    ids = json.loads(case["request"]["text"].split("Expected IDs: ", 1)[1].split("\n", 1)[0])
                    clip_scores(parsed, ids)
                elif stage.startswith("controller"):
                    parse_action(parsed, duration, task)
                elif stage == "final":
                    windows(parsed["pred_relevant_windows"], duration, allow_empty=False, canonicalize=True)
                elif stage == "overview":
                    parse_events(parsed)
                else:
                    interval = json.loads(case["request"]["text"].rsplit("\n", 1)[1])["requested_interval"]
                    parse_events(parsed, clip="Clip Analyzer: check" in case["request"]["text"],
                                 duration=duration, start=interval[0], end=interval[1])
            except (ValueError, TypeError, KeyError) as e:
                error = str(e)
            return {"qid": case["qid"], "task": case["task"], "stage": case["stage"],
                    "request_sha256": case["request_sha256"], "elapsed": elapsed,
                    "response": response, "raw_sha256": digest(response["raw"]),
                    "contract_error": error, "http_error": None}
        except Exception as e:
            return {"qid": case["qid"], "task": case["task"], "stage": case["stage"],
                    "request_sha256": case["request_sha256"], "elapsed": time.monotonic()-started,
                    "http_error": f"{type(e).__name__}: {e}"}
        finally:
            backend.session.close()

    rounds = []
    for iteration, concurrency in enumerate(order):
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(execute, cases))
        elapsed = time.monotonic()-started
        value = {"round": iteration, "concurrency": concurrency, "wall_seconds": elapsed,
                 "requests": len(results), "results": results}
        atomic_json(output / f"round_{iteration}_c{concurrency}.json", value)
        rounds.append(value)
        print(json.dumps({k: v for k, v in value.items() if k != "results"}), flush=True)
    summary = {}
    reference = {r["request_sha256"]: r.get("raw_sha256") for r in rounds[0]["results"]}
    for c in sorted(set(order)):
        selected = [r for r in rounds if r["concurrency"] == c]
        rows = [x for r in selected for x in r["results"]]
        wall = sum(r["wall_seconds"] for r in selected)
        summary[str(c)] = {
            "requests": len(rows), "wall_seconds": wall,
            "requests_per_second": len(rows)/wall,
            "http_errors": sum(bool(r.get("http_error")) for r in rows),
            "contract_errors": sum(bool(r.get("contract_error")) for r in rows),
            "exact_raw_agreement_with_first_serial": sum(r.get("raw_sha256") == reference[r["request_sha256"]] for r in rows),
            "latency_p50_p95_seconds": np.quantile([r["elapsed"] for r in rows], [.5,.95]).tolist(),
            "input_tokens": sum(r.get("response", {}).get("usage", {}).get("prompt_tokens", 0) for r in rows),
            "output_tokens": sum(r.get("response", {}).get("usage", {}).get("completion_tokens", 0) for r in rows),
        }
    eligible = [c for c in summary if summary[c]["http_errors"] == 0
                and summary[c]["contract_errors"] <= summary['1']["contract_errors"]]
    if not eligible:
        raise RuntimeError("no stable concurrency setting; replay artifacts retained")
    selected = max(eligible, key=lambda c: summary[c]["requests_per_second"])
    for c in summary:
        summary[c]["speedup_over_serial"] = summary[c]["requests_per_second"] / summary['1']["requests_per_second"]
    report = {"registered": str(output / 'registered.json'), "summary": summary,
              "selected_query_workers": int(selected), "max_num_seqs": 4,
              "actual_requests": len(order)*len(cases),
              "selection_rule": "maximum measured throughput among HTTP-error-free settings with no additional contract failures; never ground-truth score selection",
              "output_note": "temperature/seed fixed; batch scheduling can change exact outputs, agreement is reported",
              "scope": "fixed real request replay; no autonomous-quality or complete-query speedup claim"}
    atomic_json(ARTIFACTS / 'checks/concurrency_benchmark_m1.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
