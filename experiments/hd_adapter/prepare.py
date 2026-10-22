"""Deterministic, source-disjoint development data and immutable run snapshots."""

import json
import shutil
from collections import defaultdict
from pathlib import Path

from merit_qvh.cli import inference_row, put_manifest
from merit_qvh.core import digest, file_sha256, read_jsonl, validate_manifest, write_json

REPO = Path(__file__).resolve().parents[2]


def source_id(row):
    return row["vid"].rsplit("_", 2)[0]


def select_rows(train, excluded, count=100, seed=42):
    groups = defaultdict(list)
    for row in train:
        if source_id(row) not in excluded:
            groups[source_id(row)].append(row)
    sources = sorted(groups, key=lambda s: digest([seed, "merit_hd_v2_source", s]))
    if len(sources) < 2 * count:
        raise ValueError("Insufficient unseen train source videos")
    selected = [
        min(groups[s], key=lambda r: digest([seed, "query", r["qid"], r["vid"]]))
        for s in sources[: 2 * count]
    ]
    return {"explore": selected[:count], "confirm": selected[count:]}


def prepare_data(config):
    directory = Path(config["hd"]["data_root"])
    train_path = Path(config["annotation_root"]) / "highlight_train_release.jsonl"
    excluded, inputs = set(), {}
    for name in config["hd"]["exclusion_manifests"]:
        path = Path(name)
        rows = read_jsonl(path) if path.suffix == ".jsonl" else json.loads(path.read_text())
        if not isinstance(rows, list) or not all("vid" in r for r in rows):
            raise ValueError(f"Unexpected exclusion manifest: {path}")
        excluded.update(source_id(row) for row in rows)
        inputs[str(path)] = {"sha256": file_sha256(path), "queries": len(rows)}
    selections = select_rows(read_jsonl(train_path), excluded)
    audit = {
        "seed": 42,
        "train_sha256": file_sha256(train_path),
        "exclusion_inputs": inputs,
        "excluded_source_ids": sorted(excluded),
        "selection": "SHA256-ranked sources; one SHA256-ranked query per source; explore first 100, confirm next 100",
        "source_overlap": 0,
        "ground_truth_used_for_selection": False,
        "splits": {},
    }
    for split, truth in selections.items():
        rows = validate_manifest([inference_row(row, config["video_root"]) for row in truth])
        for row in rows:
            if not Path(row["video_path"]).is_file():
                raise FileNotFoundError(row["video_path"])
        put_manifest(directory / "manifests" / (split + ".jsonl"), rows)
        put_manifest(directory / "ground_truth" / (split + ".jsonl"), truth)
        audit["splits"][split] = {
            "qids": [row["qid"] for row in rows],
            "sources": len({source_id(r) for r in rows}),
            "sha256": file_sha256(directory / "manifests" / (split + ".jsonl")),
        }
    path = directory / "data_audit.json"
    if path.exists() and json.loads(path.read_text()) != audit:
        raise ValueError("Data preparation changed; use a new directory")
    write_json(path, audit)
    return audit


def source_files():
    return sorted(
        list((REPO / "src/merit_qvh").glob("*.py")) + list((REPO / "experiments/hd_adapter").glob("*.py"))
    )


def validate_config(config):
    expected = {
        "image_longest_edge": 672,
        "jpeg_quality": 90,
        "batch_clips": 24,
        "max_tokens": 2048,
        "minimum": 0,
        "maximum": 4,
        "max_queries": 100,
        "confirm_queries_reserved": 100,
        "automatic_confirmation": False,
        "seed": 42,
    }
    for key, value in expected.items():
        if config["hd"].get(key) != value:
            raise ValueError(f"Unsupported HD protocol setting: {key}")
    if config["hd"]["workers"] not in (1, 2, 4):
        raise ValueError("Unsupported worker count")


def freeze(config):
    validate_config(config)
    data = Path(config["hd"]["data_root"])
    source = {str(p.relative_to(REPO)): file_sha256(p) for p in source_files()}
    inputs = {str(p.relative_to(data)): file_sha256(p) for p in data.rglob("*") if p.is_file()}
    protocol = {"config": config, "source_sha256": source, "data_sha256": inputs}
    protocol_id = digest(protocol)
    root = Path(config["hd"]["runs_root"]) / ("merit_hd_v2-" + protocol_id[:12])
    root.mkdir(parents=True, exist_ok=True)
    path = root / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Protocol drift")
    write_json(path, protocol)
    for name, expected in source.items():
        target = root / "source_snapshot" / name
        if target.exists() and file_sha256(target) != expected:
            raise ValueError("Source snapshot drift")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(REPO / name, target)
    for name, expected in inputs.items():
        target = root / "data" / name
        if target.exists() and file_sha256(target) != expected:
            raise ValueError("Data snapshot drift")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(data / name, target)
    # Legacy revision validation resolves the upstream checkout relative to its
    # module. A read-only reference lets the frozen source snapshot execute alone.
    upstream = root / "source_snapshot/third_party/merit"
    upstream.parent.mkdir(parents=True, exist_ok=True)
    if not upstream.exists():
        upstream.symlink_to(REPO / "third_party/merit", target_is_directory=True)
    return protocol_id, root


def verify(root):
    root = Path(root)
    protocol = json.loads((root / "protocol.json").read_text())
    validate_config(protocol["config"])
    for name, expected in protocol["source_sha256"].items():
        if file_sha256(REPO / name) != expected:
            raise ValueError(f"Source drift: {name}; do not alter a running experiment")
    for name, expected in protocol["data_sha256"].items():
        if file_sha256(root / "data" / name) != expected:
            raise ValueError(f"Manifest/GT drift: {name}")
    return protocol["config"], digest(protocol)
