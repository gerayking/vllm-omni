#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
MODEL="${MODEL:-openbmb/MiniCPM-o-4_5}"
VIDEO_MME_METADATA="${VIDEO_MME_METADATA:?set VIDEO_MME_METADATA to the official parquet or JSON file}"
VIDEO_MME_VIDEO_DIR="${VIDEO_MME_VIDEO_DIR:?set VIDEO_MME_VIDEO_DIR to extracted MP4 files}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/artifacts/minicpmo_ascend/video_mme}"

exec "${PYTHON}" -m benchmarks.competition.minicpmo_ascend.video_mme \
    --metadata "${VIDEO_MME_METADATA}" \
    --video-dir "${VIDEO_MME_VIDEO_DIR}" \
    --model "${MODEL}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
