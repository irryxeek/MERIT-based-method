#!/usr/bin/env bash
set -euo pipefail

DOWNLOAD_PID="${DOWNLOAD_PID:-61026}"
EXPECTED_BYTES=143734787897
ROOT=/path/to/workspace/LongVT
ARCHIVE=/path/to/storage/longvt-repro/benchmark_data/QVHighlights/archive/qvhilights_videos.tar.gz
CONTROL="${ARCHIVE}.aria2"
PY=/path/to/storage/longvt-repro/env/inference/bin/python

while kill -0 "${DOWNLOAD_PID}" 2>/dev/null; do
    sleep 60
done

if [[ -e "${CONTROL}" ]]; then
    printf 'Download process exited but aria2 control file remains: %s\n' "${CONTROL}" >&2
    exit 1
fi

actual_bytes=$(stat -c '%s' "${ARCHIVE}")
if [[ "${actual_bytes}" -ne "${EXPECTED_BYTES}" ]]; then
    printf 'Archive size mismatch: expected %s, got %s\n' "${EXPECTED_BYTES}" "${actual_bytes}" >&2
    exit 1
fi

gzip -t "${ARCHIVE}"

cd "${ROOT}"
"${PY}" reproduction/qvhighlights/scripts/prepare_from_archive.py \
    --archive "${ARCHIVE}" \
    --candidates reproduction/qvhighlights/manifests/candidate_queue.json \
    --config reproduction/qvhighlights/configs/pilot.json \
    --media-dir /path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_official \
    --manifest-dir reproduction/qvhighlights/manifests
