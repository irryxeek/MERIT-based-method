"""Regressions for selecting media compatible with the existing inference decoder."""

import importlib.util
import sys
from pathlib import Path

import pytest
import yt_dlp

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("mrhisum_downloader", SCRIPTS / "download_mrhisum_videos.py")
DOWNLOADER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOWNLOADER)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ("This video is unavailable", "source_unavailable"),
        ("This video is not available", "source_unavailable"),
        ("Unavailable with a VPN or proxy", "network_restricted"),
        ("Sign in to confirm you’re not a bot", "authentication_required"),
        ("TimeoutExpired: timed out after 300 seconds", "source_or_download_error"),
    ],
)
def test_source_errors_do_not_trigger_network_failure_circuit(error, expected):
    assert DOWNLOADER.classify_failure(error) == expected


def test_mp4_selection_prefers_decodable_avc_over_av1():
    # Both streams use MP4. Container filtering alone previously chose AV1.
    formats = [
        {"format_id": "140", "ext": "m4a", "vcodec": "none", "acodec": "mp4a.40.2", "abr": 128},
        {
            "format_id": "135",
            "ext": "mp4",
            "vcodec": "avc1.4d401e",
            "acodec": "none",
            "height": 480,
            "width": 640,
            "tbr": 600,
        },
        {
            "format_id": "397",
            "ext": "mp4",
            "vcodec": "av01.0.04M.08",
            "acodec": "none",
            "height": 480,
            "width": 640,
            "tbr": 1000,
        },
    ]
    for item in formats:
        item["url"] = f"https://example.invalid/{item['format_id']}"
    with yt_dlp.YoutubeDL({"format": DOWNLOADER.FORMAT, "quiet": True, "no_warnings": True}) as downloader:
        selected = downloader.process_ie_result(
            {"id": "codec-regression", "title": "Offline format selection", "formats": formats},
            download=False,
        )
    assert selected["format_id"] == "135+140"
    assert selected["requested_formats"][0]["vcodec"].startswith("avc1")
