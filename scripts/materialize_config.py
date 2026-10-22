"""Expand a reviewed JSON template into a new machine-local configuration."""

import argparse
import json
import os
import re
from pathlib import Path

VARIABLE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")


def expand(value, environment):
    if isinstance(value, dict):
        return {key: expand(item, environment) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item, environment) for item in value]
    if not isinstance(value, str):
        return value

    def replacement(match):
        name = match.group(1)
        if not environment.get(name):
            raise ValueError(f"Set {name} before materializing this configuration")
        return environment[name]

    return VARIABLE.sub(replacement, value)


def materialize(source, destination, environment):
    result = expand(json.loads(Path(source).read_text()), environment)
    if isinstance(result, dict) and isinstance(result.get("hd"), dict):
        devices = [item.strip() for item in environment.get("REPRO_GPU_UUIDS", "").split(",") if item.strip()]
        server = result["hd"].get("server", {})
        if devices:
            if len(devices) != server["tensor_parallel_size"] or len(set(devices)) != len(devices):
                raise ValueError("REPRO_GPU_UUIDS must contain one unique UUID per tensor-parallel rank")
            if any(not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", item) for item in devices):
                raise ValueError("Expected full NVIDIA GPU UUIDs")
            server["gpu_uuids"] = devices
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(materialize(args.template, args.output, os.environ))


if __name__ == "__main__":
    main()
