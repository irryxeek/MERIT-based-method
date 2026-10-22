"""Verify effective image grids and API token counts, not just CLI argument names."""

import argparse
import base64
import io
import json
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor

from merit_qvh.client import CachedClient, redact_images
from merit_qvh.core import write_json


def audit(config, directory, endpoint):
    processors = {
        "old_explicit_cap": AutoProcessor.from_pretrained(
            config["vlm"]["weight_path"],
            local_files_only=True,
            size={"shortest_edge": 65536, "longest_edge": 200704},
        ),
        "new_explicit_cap": AutoProcessor.from_pretrained(
            config["vlm"]["weight_path"],
            local_files_only=True,
            size={"shortest_edge": 65536, "longest_edge": 1048576},
        ),
        "old_max_pixels_argument": AutoProcessor.from_pretrained(
            config["vlm"]["weight_path"], local_files_only=True, max_pixels=200704
        ),
    }
    report = {
        "effective_sizes": {k: p.image_processor.size for k, p in processors.items()},
        "synthetic": [],
        "real_dimensions_checked": [],
    }
    for size in ((672, 448), (448, 300)):
        image = Image.new("RGB", size)
        grids = {
            k: p.image_processor(images=[image], return_tensors="pt")["image_grid_thw"].tolist()
            for k, p in processors.items()
        }
        buff = io.BytesIO()
        image.save(buff, format="JPEG", quality=90)
        content = [
            {"type": "text", "text": "Describe this image briefly."},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buff.getvalue()).decode()},
            },
        ]
        payload = {
            "model": config["vlm"]["model"],
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "seed": 42,
            "max_tokens": 1,
        }
        processor = processors["new_explicit_cap"]
        text = processor.apply_chat_template(payload["messages"], tokenize=False, add_generation_prompt=True)
        expected = processor(text=[text], images=[image], return_tensors="pt")["input_ids"].shape[-1]
        client = CachedClient(endpoint, config["vlm"], directory, "processor-audit")
        response = client._post(payload)
        record = {
            "size": size,
            "grids": grids,
            "expected_prompt_tokens": expected,
            "response": response,
            "request": redact_images(payload),
        }
        report["synthetic"].append(record)
        write_json(directory / "processor_audit.json", report)
        if response["usage"]["prompt_tokens"] != expected:
            raise ValueError("Actual server token count differs from explicit-cap processor")
    # Prove the preflight's former unbounded cap and the explicit cap have the
    # same grids on every image size actually used in the HD preflight.
    sizes = set()
    for path in directory.glob("hd_frames/*/*.jpg"):
        with Image.open(path) as image:
            sizes.add(image.size)
    for size in sorted(sizes):
        image = Image.new("RGB", size)
        grids = {
            k: p.image_processor(images=[image], return_tensors="pt")["image_grid_thw"].tolist()
            for k, p in processors.items()
            if k != "old_explicit_cap"
        }
        if grids["new_explicit_cap"] != grids["old_max_pixels_argument"]:
            raise ValueError("Preflight image grid changed under corrected processor cap")
        report["real_dimensions_checked"].append({"size": size, "grid": grids["new_explicit_cap"]})
    report.update(
        status="passed",
        note="max_pixels alone was ineffective in installed Transformers; explicit size verified",
    )
    write_json(directory / "processor_audit.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8037/v1")
    args = parser.parse_args()
    audit(json.loads(args.config.read_text()), args.directory, args.endpoint)
