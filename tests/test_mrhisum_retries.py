"""Protect frozen downloads and verify recovery accounting without network access."""

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
RETRY = importlib.import_module("retry_mrhisum_downloads")


def record(number, status="source_unavailable", error="Video unavailable"):
    return {
        "video_id": f"video_{number}",
        "youtube_id": f"test{number:07d}",
        "status": status,
        "error": error,
        "attempts": 1,
        "annotated_seconds": 120,
    }


def test_selection_skips_successes_and_explicit_access_restrictions():
    rows = [
        record(1, "downloaded_verified"),
        record(2, error="Private video"),
        record(3, error="This video has been removed"),
        record(4, error="Sign in to confirm your age"),
        record(5),
        record(6, error="Sign in to confirm you’re not a bot"),
        record(7, error="TimeoutExpired"),
        record(8, error="Unavailable with a VPN or proxy"),
        record(9),
    ]
    records = {r["video_id"]: r for r in rows}
    items = [{key: r[key] for key in ("video_id", "youtube_id", "annotated_seconds")} for r in rows]
    selected, skipped = RETRY.select_items({"videos": items}, records)
    assert [r["video_id"] for r in selected] == ["video_7", "video_6", "video_8", "video_5", "video_9"]
    assert [r["video_id"] for r in skipped] == ["video_2", "video_3", "video_4"]


def test_merge_preserves_first_pass_and_counts_recovery_once():
    original = {"video_1": record(1, "downloaded_verified"), "video_2": record(2)}
    before = json.dumps(original, sort_keys=True)
    recovered = record(2, "downloaded_verified")
    merged = RETRY.merge_records(original, [recovered], {"video_2"})
    assert json.dumps(original, sort_keys=True) == before
    assert len(merged) == 2
    assert all(r["status"] == "downloaded_verified" for r in merged.values())


@pytest.mark.parametrize("case", ["previous_success", "wrong_source", "outside_scope"])
def test_merge_rejects_scope_or_source_changes(case):
    original = {"video_1": record(1, "downloaded_verified"), "video_2": record(2)}
    recovered = record(2, "downloaded_verified")
    scope = {"video_2"}
    if case == "previous_success":
        recovered = record(1, "downloaded_verified")
        scope.add("video_1")
    elif case == "wrong_source":
        recovered["youtube_id"] = "changed_id_"
    else:
        scope.clear()
    with pytest.raises(ValueError):
        RETRY.merge_records(original, [recovered], scope)


def test_retry_resumes_without_changing_baseline_or_repeating_download(tmp_path, monkeypatch):
    baseline = tmp_path / "raw"
    (baseline / "receipts").mkdir(parents=True)
    (baseline / "videos").mkdir()
    video = baseline / "videos" / "video_1.mp4"
    video.write_bytes(b"existing")
    rows = [record(1, "downloaded_verified"), record(2, error="TimeoutExpired")]
    rows[0].update(path=str(video), bytes=video.stat().st_size)
    config = {
        "root": str(baseline),
        "scope": "test",
        "workers": 1,
        "minimum_free_gib": 0,
        "maximum_download_gib": 1,
        "maximum_consecutive_download_errors": 8,
        "videos": [{key: r[key] for key in ("video_id", "youtube_id", "annotated_seconds")} for r in rows],
    }
    RETRY.write_json(baseline / "download_manifest.json", config)
    RETRY.write_json(baseline / "complete.json", {"status": "queue_finished"})
    for r in rows:
        RETRY.write_json(baseline / "receipts" / f"{r['video_id']}.json", r)
    before = {str(p): p.read_bytes() for p in baseline.rglob("*.json")}
    root = tmp_path / "retry"
    RETRY.prepare(root, baseline)
    calls = []

    def fake_download(directory, item, retry_config):
        calls.append(item["video_id"])
        assert retry_config["max_attempts_per_video"] == 1
        path = directory / "videos" / f"{item['video_id']}.mp4"
        path.write_bytes(b"recovered")
        result = {**record(2, "downloaded_verified"), "path": str(path), "bytes": path.stat().st_size}
        RETRY.write_json(directory / "receipts" / f"{item['video_id']}.json", result)
        return result

    monkeypatch.setattr(RETRY, "process_one", fake_download)
    monkeypatch.setattr(RETRY.time, "sleep", lambda _: None)
    RETRY.run(root)
    RETRY.run(root)
    assert calls == ["video_2"]
    assert all(Path(path).read_bytes() == contents for path, contents in before.items())
    status = json.loads((root / "availability.json").read_text())
    assert status["downloaded"] == 2 and status["newly_downloaded"] == 1
    assert status["status"] == "queue_finished"
    assert video.read_bytes() == b"existing"
