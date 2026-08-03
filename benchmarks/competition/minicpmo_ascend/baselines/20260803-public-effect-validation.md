# 2026-08-03 public effect validation

This record captures public-data proxy checks for Daily-Omni, Seed-TTS, and
Video-MME on the Ascend 910C development machine. These measurements are not
official competition scores.

## Starter Kit status

The competition Toolkit page was checked again on 2026-08-03. Its two
container addresses are explicitly described as placeholder examples, the old
`oiac-toolkit-v1.0.tar.gz` host does not resolve, and both listed registries
reject anonymous image access. The official Starter Kit, benchmark subsets,
evaluator, and submission schema therefore remain unavailable. This run uses
the upstream public benchmark releases and preserves that distinction.

## Environment

- Branch: `baseline/minicpmo-ascend-v0.25.0-20260803`.
- Upstream integration: `63fdc8e7` from `upstream/minicpm-challenge`.
- Hardware: one physical Ascend 910C card exposing logical devices 0 and 1.
- Runtime: Python 3.12.13, PyTorch 2.10.0, torch-npu 2.10.0, CANN
  9.1.0-beta.3.
- Model: `/workspace/user_data/models/MiniCPM-o-4_5`.
- Service: three-stage vLLM-Omni deployment, local OpenAI-compatible endpoint,
  local media access enabled, MiniCPM interleaved AV strings enabled.

## Daily-Omni

The public `liarliar/Daily-Omni` release was downloaded at dataset revision
`bf5a6ee4c829510b7a14869e3104efeafc330b07`. The local release contains 1,197
questions and 684 videos. SHA-256 values:

- `qa.json`: `3210a45d42424c7d57c1b40a0b9aa2708fc02fab2364bf01fd7d16e1242e146b`
- `Videos.tar`: `5b42ca809bf1bfdc0894509755d5d5b2210994fa2e4c7149e3ce3db310db4afc`

The deterministic first 100 questions were evaluated with all audio/video
inputs, MiniCPM interleaving, temperature 0, output length 16, and concurrency
2. All 100 requests completed and all answers parsed.

| Metric | Result |
| --- | ---: |
| Accuracy | 71/100 (71.00%) |
| 30-second videos | 39/51 (76.47%) |
| 60-second videos | 32/49 (65.31%) |
| Median E2E | 11.882 s |
| P99 E2E | 17.251 s |
| Run duration | 592.31 s |

## Seed-TTS

The public `zhaochenyang20/seed-tts-eval` reorganization was used with the
standard prompt audio and metadata: 1,088 English rows, 2,020 Chinese rows,
1,007 English prompt WAV files, and 1,010 Chinese prompt WAV files. All 2,017
required prompt files were validated as real audio rather than Git LFS pointer
files.

An English five-item generation smoke test completed 5/5 requests and produced
19.8 seconds of valid audio. Streaming continuity was 100%, audio underrun was
zero, median audio TTFP was 1.480 seconds, median E2E was 2.249 seconds, and
median audio RTF was 0.64. Whisper-large-v3 evaluated all five outputs under
the public Seed-TTS English protocol; mean and median WER were both 0.0000,
with no request, PCM capture, or ASR failures.

A Chinese five-item generation smoke test also completed 5/5 requests and
produced 21.88 seconds of valid audio. Streaming continuity was 100%, audio
underrun was zero, median audio TTFP was 1.559 seconds, median E2E was 2.698
seconds, and median audio RTF was 0.61.

A subsequent five-item run using Paraformer-zh under the public Seed-TTS
Chinese protocol evaluated all five outputs. Mean WER was 0.0091 and median WER
was 0.0000, with no request, PCM capture, or ASR failures. WavLM similarity and
UTMOS are intentionally not reported because their public checkpoints are not
guaranteed to match the unreleased competition evaluator.

## Video-MME

The public `lmms-eval/Video-MME` metadata contains 2,700 questions. The full
video payload is approximately 101 GB and split into 20 archives. At this
validation point, 296 videos from completed public archives were available, so
the deterministic first 100 resolvable questions were evaluated using the
official subtitles-free multiple-choice prompt. This is a partial public-set
check, not a full Video-MME score.

All 100 requests completed and all answers parsed.

| Metric | Result |
| --- | ---: |
| Accuracy | 79/100 (79.00%) |
| Knowledge | 57/72 (79.17%) |
| Film & TV | 22/28 (78.57%) |
| Duration bucket | 100 short-video questions |

## Reproduction and limitations

Commands are documented in the parent README and saved with each ignored raw
artifact under `artifacts/minicpmo_ascend/public-eval/`. Install public
evaluator dependencies with:

```bash
.venv/bin/python -m pip install -r \
  benchmarks/competition/minicpmo_ascend/requirements-eval.txt
```

Before a formal submission, replace every public subset and evaluator with the
official Starter Kit versions, run the complete three-suite effect gate against
the official baseline, and retain raw outputs and exact environment metadata.
