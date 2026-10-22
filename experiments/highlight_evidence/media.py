"""Decode a planned evidence view at exact original PTS, using existing media I/O."""

import base64
import io

from hd_adapter.media import FrameReader
from PIL import Image

from merit_qvh.core import digest


def decode_view(path, info, view):
    """No playback-speed simulation, duplicate frames, interpolation or extrapolation."""
    pts = view["pts"]
    if not pts or len(pts) > 24 or pts != sorted(set(pts)):
        raise ValueError("One view requires 1-24 distinct ordered source frames")
    valid = set(info["pts"])
    if any(t not in valid or not view["start"] <= t < view["end"] for t in pts):
        raise ValueError("Requested evidence is not a real PTS in this interval")
    reader = FrameReader(path, info)
    try:
        frames = []
        for t in pts:
            picture = reader.at(t)
            picture.thumbnail((672, 672), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            picture.save(buffer, format="JPEG", quality=90)
            encoded = base64.b64encode(buffer.getvalue()).decode()
            frames.append(
                {
                    "image_id": digest([info["media_sha256"], t]),
                    "pts": t,
                    "jpeg_sha256": digest(encoded),
                    "url": "data:image/jpeg;base64," + encoded,
                }
            )
        return frames
    finally:
        reader.close()
