"""Render the preregistered source frames and collect paired descriptions; no model calls."""

import argparse
import hashlib
import math
import textwrap
from collections import Counter
from pathlib import Path

from frame_observer import FRAME_PROMPT, grounded_description, has_visible_claim
from io_utils import legacy, parents, read_json, sha256, utc_now, write_json
from r2_observer import read_frames, source_timeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--qid", type=int, help="Render one completed query while others are running")
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = read_json(run / "manifest.json")
    parent = Path(manifest["parent_run"])
    reference = Path(manifest["reference_probe"])
    binding = read_json(parent / "binding.json")
    rows = legacy.read_jsonl(run / "inference.jsonl")
    if args.qid is not None:
        rows = [row for row in rows if row["qid"] == args.qid]
    import torch
    from PIL import Image, ImageDraw, ImageFont

    torch.set_num_threads(2)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    output = run / "review"
    output.mkdir(exist_ok=True)
    for row in rows:
        qid = row["qid"]
        original = read_json(parent / "R2/records" / f"{qid}.json")
        baseline, _ = parents(binding, row)
        reader, timestamps, _ = source_timeline(row, baseline)
        for arm in ("E_uniform", "E_targeted"):
            entries, images = [], []
            for number, old in enumerate(original["observations"][arm]):
                neutral = read_json(reference / "observations" / f"{qid}_{arm}_{number}.json")
                video, trace = read_frames(reader, timestamps, [f["frame_index"] for f in old["trace"]["frames"]], row, old["action"], baseline["media"]["resized_height_width"], arm)
                if trace != old["trace"]:
                    raise ValueError("Review pixels differ from inference pixels")
                for position, frame in enumerate(trace["frames"]):
                    if position not in manifest["review_positions"] and frame["id"] not in manifest["known_error_frames"].get(str(qid), []):
                        continue
                    path = run / "frames" / f"{qid}_{arm}_{number}_{position:02d}.json"
                    record = read_json(path)
                    if record["proof"]["frames"] != [frame]:
                        raise ValueError("Still observation points at another frame")
                    pixels = video["video"][position]
                    if hashlib.sha256(pixels.tobytes()).hexdigest() != frame["rgb_sha256"]:
                        raise ValueError("Review/source hash mismatch")
                    entries.append({
                        "qid": qid, "arm": arm, "number": number, "position": position,
                        "candidate": old["action"]["candidate"], "frame": frame, "record_sha256": sha256(path),
                        "status": record["status"], "new": record.get("result"),
                        "original_facts_citing_frame": [f for f in old["result"]["facts"] if frame["id"] in f["frames"]],
                        "neutral_facts_citing_frame": [f for f in neutral["result"]["facts"] if frame["id"] in f["frames"]],
                        "assistant_review": "pending", "independent_human_review": "not_reviewed",
                    })
                    images.append(Image.fromarray(pixels))
            width, height, columns = 360, 330, 4
            sheet = Image.new("RGB", (width * columns, height * math.ceil(len(entries) / columns)), "white")
            draw = ImageDraw.Draw(sheet)
            for index, (entry, image) in enumerate(zip(entries, images, strict=True)):
                x, y = index % columns * width, index // columns * height
                title = f"q{qid} {arm} c{entry['candidate']} {entry['frame']['id']}"
                draw.text((x + 5, y + 3), title, fill="black", font=font)
                draw.text((x + 5, y + 22), f"PTS {entry['frame']['pts'][0]:.3f}s / position {entry['position']}", fill="black", font=font)
                image.thumbnail((350, 195))
                sheet.paste(image, (x + 5, y + 43))
                description = (entry["new"] or {}).get("visible", "INVALID")
                uncertainty = (entry["new"] or {}).get("uncertainty", "")
                text = description + (" [?] " + uncertainty if uncertainty else "")
                draw.multiline_text((x + 5, y + 241), "\n".join(textwrap.wrap(text, 45)), fill="black", font=font, spacing=2)
            sheet.save(output / f"{qid}_{arm}.jpg", quality=90)
            write_json(output / f"{qid}_{arm}.json", {"at": utc_now(), "entries": entries, "source_pixels_verified": True, "review_script_sha256": sha256(__file__)})
        del reader
    if args.qid is None:
        records = [read_json(p) for p in sorted((run / "frames").glob("*.json"))]
        ledger = read_json(run / "probe/ledger.json")
        raws = [read_json(run / "probe/attempts" / f"{a['number']:05d}.json") for a in ledger["attempts"]]
        if any(FRAME_PROMPT not in raw["specification"]["rendered_prompt"] for raw in raws):
            raise ValueError("Rendered prompt differs from frozen observer")
        scopes = Counter(f["grounding"]["scope"] for r in records for f in r.get("grounded_facts", {}).values())
        derived = []
        for record in records:
            if record["status"] != "complete":
                continue
            name = f"obs{record['number']}_p{record['position']}_c{record['candidate']}"
            facts = grounded_description(record["result"], record["proof"], record["candidate"], record["target_interval"], name)
            derived.append({"key": record["key"], "grounded_facts": facts, "has_visible_claim": has_visible_claim(record["result"]["visible"])})
        write_json(output / "guarded_facts.json", {
            "post_inference_repair": "punctuation_only_visible_fields_are_not_visual_facts",
            "source_sha256": sha256(Path(__file__).with_name('frame_observer.py')),
            "new_model_calls": 0, "original_records_modified": False, "records": derived,
        })
        unique = {(r["qid"], r["proof"]["frames"][0]["frame_index"]) for r in records}
        summary = {
            "records": len(records), "statuses": dict(Counter(r["status"] for r in records)),
            "scopes": dict(scopes), "distinct_qid_source_frames": len(unique),
            "post_inference_visible_abstentions": sum(not r["has_visible_claim"] for r in derived),
            "visible_at_character_cap": sum(len(r.get("result", {}).get("visible", "")) == 160 for r in records),
            "uncertainty_at_character_cap": sum(len(r.get("result", {}).get("uncertainty", "")) == 80 for r in records),
            "input_tokens": sum(r.get("prompt_tokens", 0) for r in raws),
            "output_tokens": sum(r.get("output_tokens", 0) for r in raws),
            "selected_review_frame_occurrences": sum(len(read_json(p)["entries"]) for p in output.glob('*_E_*.json')),
            "semantic_accuracy": "not_independently_evaluated", "new_hd_metrics": False,
        }
        write_json(output / "summary.json", summary)
        print(summary)


if __name__ == "__main__":
    main()
