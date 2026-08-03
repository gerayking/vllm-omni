#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL="${MODEL:-openbmb/MiniCPM-o-4_5}"
HOST="${HOST:-localhost}"
PORT="${PORT:-8099}"
SEED_TTS_ROOT="${SEED_TTS_ROOT:?set SEED_TTS_ROOT to the public seed-tts-eval directory}"
SEED_TTS_LOCALE="${SEED_TTS_LOCALE:-en}"
NUM_PROMPTS="${NUM_PROMPTS:-20}"
NUM_WARMUPS="${NUM_WARMUPS:-2}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
WER_EVAL="${WER_EVAL:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/artifacts/minicpmo_ascend/seed_tts_${SEED_TTS_LOCALE}}"
BENCH_BIN="${BENCH_BIN:-${ROOT_DIR}/.venv/bin/vllm-omni}"

mkdir -p "${OUTPUT_DIR}"
command=(
    "${BENCH_BIN}" bench serve --omni
    --backend openai-chat-omni
    --endpoint /v1/chat/completions
    --host "${HOST}"
    --port "${PORT}"
    --model "${MODEL}"
    --trust-remote-code
    --dataset-name seed-tts
    --dataset-path "${SEED_TTS_ROOT}"
    --seed-tts-root "${SEED_TTS_ROOT}"
    --seed-tts-locale "${SEED_TTS_LOCALE}"
    --seed-tts-file-ref-audio
    --num-prompts "${NUM_PROMPTS}"
    --num-warmups "${NUM_WARMUPS}"
    --max-concurrency "${MAX_CONCURRENCY}"
    --output-len "${OUTPUT_LEN}"
    --request-rate inf
    --temperature 0
    --percentile-metrics ttft,e2el,audio_rtf,audio_ttfp,audio_duration,audio_underrun
    --extra-body '{"modalities":["text","audio"],"chat_template_kwargs":{"enable_thinking":false,"use_tts_template":true}}'
    --save-result
    --result-dir "${OUTPUT_DIR}"
)
if [[ "${WER_EVAL}" == "1" ]]; then
    command+=(--seed-tts-wer-eval --seed-tts-wer-save-items)
fi

printf '%q ' "${command[@]}" | tee "${OUTPUT_DIR}/command.txt"
printf '\n' | tee -a "${OUTPUT_DIR}/command.txt"
"${command[@]}" 2>&1 | tee "${OUTPUT_DIR}/run.log"
