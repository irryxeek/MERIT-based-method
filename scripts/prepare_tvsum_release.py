"""Extract the official TVSum package without changing media or annotations."""

import argparse
import gzip
import json
import shutil
import stat
import tarfile
import zipfile
from pathlib import Path

from download_qfvs_videos import digest, now, write_json


def member_path(root, name):
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Archive member escapes destination: {name}")
    return path


def copy_member(source, destination, size):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".extracting")
    with source, temporary.open("wb") as output:
        shutil.copyfileobj(source, output, 8 * 1024 * 1024)
    if temporary.stat().st_size != size:
        raise ValueError(f"Wrong extracted size: {destination}")
    if destination.exists():
        if digest(destination) != digest(temporary):
            raise ValueError(f"Existing release file differs: {destination}")
        temporary.unlink()
    else:
        temporary.replace(destination)
    return {"path": str(destination), "bytes": size, "sha256": digest(destination)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    archive = root / "archives/tvsum50_ver_1_1.tgz"
    receipt = json.loads((root / "assets_complete.json").read_text())["assets"][0]
    if digest(archive) != receipt["sha256"]:
        raise ValueError("Archive differs from its download receipt")
    # Reading through EOF verifies the gzip footer, including its CRC.
    with gzip.open(archive, "rb") as source:
        while source.read(8 * 1024 * 1024):
            pass
    extracted, expanded = [], []
    release = root / "official_release"
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        if sum(m.size for m in members) > 10 * 1024**3:
            raise ValueError("Unexpected tar expanded size")
        for member in members:
            path = member_path(release, member.name)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                extracted.append(copy_member(source.extractfile(member), path, member.size))
            else:
                raise ValueError("Only regular files and directories are supported")
    for record in extracted:
        path = Path(record["path"])
        if path.suffix != ".zip":
            continue
        with zipfile.ZipFile(path) as source:
            members = source.infolist()
            if sum(m.file_size for m in members) > 10 * 1024**3:
                raise ValueError("Unexpected ZIP expanded size")
            for member in members:
                destination = member_path(path.parent, member.filename)
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError("ZIP symlinks are unsupported")
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    item = copy_member(source.open(member), destination, member.file_size)
                    item["zip_crc32"] = f"{member.CRC:08x}"
                    item["archive"] = str(path)
                    expanded.append(item)
    result = {
        "status": "extracted_crc_and_hash_verified",
        "created_at": now(),
        "archive": receipt,
        "gzip_crc_verified": True,
        "zip_members_crc_verified": True,
        "outer_files": extracted,
        "files": expanded,
        "media_or_annotations_modified": False,
    }
    write_json(root / "release_manifest.json", result)
    print(json.dumps({"status": result["status"], "files": len(expanded)}), flush=True)


if __name__ == "__main__":
    main()
