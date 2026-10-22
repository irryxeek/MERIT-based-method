"""Check published source/report hashes, source coverage and local Markdown links."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
SOURCE_SUFFIXES = {".py", ".sh", ".md", ".txt", ".LICENSE"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(merit_source=None, zeroshot_source=None):
    reports = json.loads((ROOT / "docs/report_publication_manifest.json").read_text())
    code = json.loads((ROOT / "docs/code_publication_manifest.json").read_text())
    sources = {"MERIT": merit_source, "QVHighlights-ZeroShot": zeroshot_source}
    errors, checked, source_checked = [], set(), 0
    entries = reports["files"] + reports["indexes"] + code["files"]
    for entry in entries:
        target = ROOT / entry["destination"]
        checked.add(entry["destination"])
        if not target.is_file() or digest(target) != entry["publication_sha256"]:
            errors.append("Published file changed/missing: " + entry["destination"])
        name, relative = entry["source"].split(":", 1)
        if sources.get(name) is not None:
            original = sources[name] / relative
            if not original.is_file() or digest(original) != entry["source_sha256"]:
                errors.append("Research source changed/missing: " + entry["source"])
            source_checked += 1
    for entry in code["publication_only"]:
        target = ROOT / entry["destination"]
        checked.add(entry["destination"])
        if not target.is_file() or digest(target) != entry["publication_sha256"]:
            errors.append("Publication helper changed/missing: " + entry["destination"])
    recorded = {entry["source"] for entry in entries}
    for name, source in sources.items():
        if source is None:
            continue
        folders = ["scripts", "tests", "experiments"] + (["src"] if name == "MERIT" else [])
        for folder in folders:
            for path in (source / folder).rglob("*"):
                relative = path.relative_to(source)
                if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
                    continue
                if any(part in IGNORED_PARTS or part.startswith(".") for part in relative.parts):
                    continue
                key = name + ":" + relative.as_posix()
                if key not in recorded:
                    errors.append("Unpublished source: " + key)
        for path in (source / "docs").glob("*.md"):
            if path.name == "README.md":
                continue
            key = name + ":" + path.relative_to(source).as_posix()
            if key not in recorded:
                errors.append("Unpublished report: " + key)
    links_checked = 0
    markdown = {ROOT / name for name in checked if name.endswith(".md")}
    markdown.update((ROOT / "docs").glob("*.md"))
    for path in sorted(markdown):
        content = re.sub(r"```.*?```", "", path.read_text(), flags=re.DOTALL)
        for match in re.finditer(r"(?<!!)\[[^\]\n]*\]\(([^)\n]+)\)", content):
            target = match.group(1).split(' "', 1)[0].strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            target = unquote(target.split("#", 1)[0])
            links_checked += 1
            if not (path.parent / target).exists():
                errors.append(f"Broken link in {path.relative_to(ROOT)}: {target}")
    return {
        "passed": not errors,
        "published_files_checked": len(checked),
        "source_files_checked": source_checked,
        "local_links_checked": links_checked,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merit-source", type=Path)
    parser.add_argument("--zeroshot-source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.merit_source, args.zeroshot_source)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(text)
    print(text, end="")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
