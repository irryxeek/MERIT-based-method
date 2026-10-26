"""Render source-verified, caption-free sheets for the completed output probe."""

import argparse
import hashlib
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from io_utils import read_json, sha256, utc_now, write_json
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    completion = read_json(run / "complete.json")
    for name, expected in completion["receipts"].items():
        if sha256(run / name) != expected:
            raise ValueError("Completed probe changed")
    pairs = read_json(run / "review_pairs.json")
    by_query = defaultdict(list)
    for number, pair in enumerate(pairs, 1):
        by_query[pair["qid"]].append((number, pair))
    output = run / "review"
    output.mkdir(exist_ok=True)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    receipts = []
    for qid, entries in by_query.items():
        width, height, columns = 380, 290, 4
        canvas = Image.new("RGB", (width * columns, height * math.ceil(len(entries) / columns)), "white")
        draw = ImageDraw.Draw(canvas)
        for position, (number, pair) in enumerate(entries):
            source = Image.open(pair["review_image"]).convert("RGB")
            if hashlib.sha256(np.array(source).tobytes()).hexdigest() != pair["frame"]["rgb_sha256"]:
                raise ValueError("Review image pixels do not match inference source")
            x, y = position % columns * width, position // columns * height
            draw.text((x + 5, y + 3), f"#{number} q{qid} {pair['frame']['id']} {pair['frame']['pts'][0]:.3f}s", font=font, fill="black")
            source.thumbnail((370, 255))
            canvas.paste(source, (x + 5, y + 29))
        path = output / f"{qid}.jpg"
        canvas.save(path, quality=94)
        receipts.append({"qid": qid, "image": str(path), "sha256": sha256(path), "pair_numbers": [n for n, _ in entries]})
    write_json(output / "render_receipt.json", {"at": utc_now(), "source_sha256": sha256(__file__), "pairs_sha256": sha256(run / "review_pairs.json"), "source_pixels_verified": True, "sheets": receipts, "human_review_count": 0})
    for number, pair in enumerate(pairs, 1):
        print(f"#{number} {pair['key']} {pair['frame']['id']}")
        print("OLD:", pair["before"])
        print("NEW:", pair["after"])


if __name__ == "__main__":
    main()
