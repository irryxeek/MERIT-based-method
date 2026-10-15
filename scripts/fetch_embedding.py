"""Fetch the pinned embedding weights; verify a Qwen ModelScope mirror by HF SHA-256.

Small configuration/tokenizer files use Hugging Face. Weight ranges can resume from
an interrupted HF download, but are accepted only after matching the pinned HF LFS hash.
"""

import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path

import requests
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub._local_folder import write_download_metadata

REPO = Path(__file__).resolve().parents[1]
config = json.loads((REPO / "configs/main_v1.json").read_text())["embedding"]
root = Path(config["weight_path"])
revision = config["revision"]
info = HfApi().model_info(config["model"], revision=revision, files_metadata=True)
assert info.sha == revision
snapshot_download(config["model"], revision=revision, local_dir=str(root), ignore_patterns=["*.safetensors"])


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


provenance_path = root.parent / "embedding_source.json"
previous = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
sources = previous.get("files", {}) if previous.get("canonical_revision") == revision else {}
for sibling in info.siblings:
    name = sibling.rfilename
    if not name.endswith(".safetensors"):
        continue
    expected = sibling.lfs.sha256
    size = sibling.size
    target = root / name
    if target.exists() and target.stat().st_size == size and sha256(target) == expected:
        sources.setdefault(
            name,
            {
                "sha256": expected,
                "size": size,
                "source": "existing local bytes verified against the pinned HF SHA-256",
            },
        )
        print(json.dumps({"file": name, "status": "already_verified"}), flush=True)
        continue
    candidates = list((root / ".cache/huggingface/download").glob("*." + expected + ".incomplete"))
    temporary = candidates[0] if candidates else root / (name + ".incomplete")
    offset = temporary.stat().st_size if temporary.exists() else 0
    if offset > size:
        raise ValueError("Existing partial weight exceeds the expected size")
    part_root = root / ".verified_mirror_parts" / name
    part_root.mkdir(parents=True, exist_ok=True)
    url = "https://modelscope.cn/models/Qwen/Qwen3-Embedding-4B/resolve/master/" + name
    block_size = 32 * 1024 * 1024
    ranges = [(start, min(size, start + block_size)) for start in range(offset, size, block_size)]

    def download(bounds, part_root=part_root, url=url, size=size, name=name):
        start, end = bounds
        path = part_root / f"{start}-{end}.part"
        if path.exists() and path.stat().st_size == end - start:
            return path
        for attempt in range(4):
            try:
                # Range in the URL also avoids any intermediary caching the wrong range.
                with requests.get(
                    url + f"?merit_range={start}-{end}",
                    headers={"Range": f"bytes={start}-{end - 1}"},
                    stream=True,
                    timeout=(20, 90),
                ) as response:
                    response.raise_for_status()
                    if (
                        response.status_code != 206
                        or response.headers.get("Content-Range") != f"bytes {start}-{end - 1}/{size}"
                    ):
                        raise ValueError("Mirror did not honor the requested range")
                    with path.open("wb") as handle:
                        for block in response.iter_content(1024 * 1024):
                            handle.write(block)
                if path.stat().st_size != end - start:
                    raise ValueError("Truncated range")
                return path
            except (requests.RequestException, OSError, ValueError) as error:
                if attempt == 3:
                    raise
                print(
                    json.dumps(
                        {
                            "file": name,
                            "range": [start, end],
                            "retry": attempt + 1,
                            "error": type(error).__name__,
                        }
                    ),
                    flush=True,
                )
        raise RuntimeError("Unreachable download state")

    print(
        json.dumps({"file": name, "resume_bytes": offset, "total_bytes": size, "ranges": len(ranges)}),
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        parts = []
        for index, path in enumerate(pool.map(download, ranges)):
            parts.append(path)
            if index % 8 == 7 or index == len(ranges) - 1:
                print(
                    json.dumps({"file": name, "ranges_complete": index + 1, "ranges_total": len(ranges)}),
                    flush=True,
                )
    with temporary.open("ab") as handle:
        for path in parts:
            with path.open("rb") as part:
                for block in iter(lambda: part.read(1024 * 1024), b""):
                    handle.write(block)
    if temporary.stat().st_size != size or sha256(temporary) != expected:
        raise ValueError("Mirror bytes do not match the pinned Hugging Face revision")
    os.replace(temporary, target)
    # Standard HF cache metadata records content identity; actual transport is recorded below.
    write_download_metadata(root, name, revision, expected)
    sources[name] = {"sha256": expected, "size": size, "source": url, "resumed_hf_prefix_bytes": offset}
    print(json.dumps({"file": name, "status": "sha256_verified"}), flush=True)

record = {
    "model": config["model"],
    "canonical_revision": revision,
    "timestamp": time.time(),
    "files": sources,
}
(root.parent / "embedding_source.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({"status": "complete", "root": str(root)}), flush=True)
