#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL="${MODEL:-openbmb/MiniCPM-o-4_5}"
BENCH_BIN="${BENCH_BIN:-${ROOT_DIR}/.venv/bin/vllm-omni}"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
SAMPLE_ROOT="${SAMPLE_ROOT:?set SAMPLE_ROOT to the generated public benchmark sample directory}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the public evaluation data directory}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/artifacts/minicpmo_ascend/public-proxy-benchmark-v1}"
NUM_WARMUPS="${NUM_WARMUPS:-3}"
SOURCE_REVISION="${SOURCE_REVISION:-$(git -C "${ROOT_DIR}" rev-parse HEAD)}"

mkdir -p "${OUTPUT_ROOT}"

DAILY_OMNI_QA_JSON="${SAMPLE_ROOT}/daily_omni/qa.json" \
DAILY_OMNI_VIDEO_DIR="${DATA_ROOT}/Daily-Omni/Videos" \
DAILY_OMNI_INPUT_MODE=all \
DAILY_OMNI_PACK_MODE=minicpm-interleave \
NUM_PROMPTS=100 \
NUM_WARMUPS="${NUM_WARMUPS}" \
MAX_CONCURRENCY=1 \
MODEL="${MODEL}" \
BENCH_BIN="${BENCH_BIN}" \
OUTPUT_DIR="${OUTPUT_ROOT}/daily_omni" \
bash "${ROOT_DIR}/benchmarks/competition/minicpmo_ascend/run_daily_omni.sh" --disable-shuffle

for locale in en zh; do
    SEED_TTS_ROOT="${SAMPLE_ROOT}/seed_tts" \
    SEED_TTS_LOCALE="${locale}" \
    NUM_PROMPTS=100 \
    NUM_WARMUPS="${NUM_WARMUPS}" \
    MAX_CONCURRENCY=1 \
    OUTPUT_LEN=128 \
    WER_EVAL=0 \
    MODEL="${MODEL}" \
    BENCH_BIN="${BENCH_BIN}" \
    OUTPUT_DIR="${OUTPUT_ROOT}/seed_tts_${locale}" \
    bash "${ROOT_DIR}/benchmarks/competition/minicpmo_ascend/run_seed_tts.sh" --disable-shuffle
done

PYTHON="${PYTHON}" \
MODEL="${MODEL}" \
VIDEO_MME_METADATA="${SAMPLE_ROOT}/video_mme/metadata.json" \
VIDEO_MME_VIDEO_DIR="${DATA_ROOT}/Video-MME/extracted/data" \
OUTPUT_DIR="${OUTPUT_ROOT}/video_mme" \
bash "${ROOT_DIR}/benchmarks/competition/minicpmo_ascend/run_video_mme.sh" \
    --num-questions 100 --concurrency 1

cp "${SAMPLE_ROOT}/manifest.json" "${OUTPUT_ROOT}/sample_manifest.json"
"${PYTHON}" -m benchmarks.competition.minicpmo_ascend.summarize_public_benchmark \
    --output-root "${OUTPUT_ROOT}" \
    --source-revision "${SOURCE_REVISION}"
echo "Public proxy benchmark completed: ${OUTPUT_ROOT}"
