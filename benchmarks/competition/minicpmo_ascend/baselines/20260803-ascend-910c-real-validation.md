# 2026-08-03 Ascend 910C real-machine baseline validation

This record captures the pre-submission validation of the competition baseline.
The measurements below are local proxy results, not an official competition
score.

## Source and environment

- Upstream base: `f05babab841e7374d0f4c98f5d289940f7118c49`
  (`upstream/minicpm-challenge`, fetched 2026-08-03).
- Official image reference: `quay.io/ascend/vllm-omni:v0.25.0-a3`.
- Resolved arm64 image digest:
  `sha256:6c22113f72a276bb4662d361680e14240189e8d01d31521cbef565f71dd0a078`.
- The registry stopped allowing anonymous layer downloads during validation, so
  the final run used an isolated environment reconstructed from the source
  revisions embedded in the image rather than claiming an official-container
  run.
- vLLM source: `e5588e49bc2642670116664a7fc4096e27adb179`.
- vLLM Ascend source: `8092d3f66599ce07cd0aca2bcc99d14b8a9192f8`.
- Python 3.12.13, PyTorch 2.10.0, torch-npu 2.10.0,
  triton-ascend 3.2.1, and stepaudio2-minicpmo 0.1.1.
- Host CANN: 9.1.0-beta.3.
- Hardware: one physical Ascend 910C card exposing logical devices 0 and 1,
  each with 64 GiB HBM.
- Model: complete MiniCPM-o-4_5 checkpoint downloaded through ModelScope;
  checkpoint weight size reported by the loader was 17.46 GiB.

Before server startup, tensor allocation and arithmetic passed independently
on both logical NPU devices.

## Service and multimodal validation

The three-stage deployment in
`vllm_omni/deploy/minicpmo_4_5_ascend_910c_1card.yaml` initialized on the real
device. Stage 0 loaded 16.8394 GiB of weights, stage 1 loaded 0.6422 GiB, and
stage 2 loaded the Token2Wav stack. The API completed model warmup, returned
HTTP 200 from `/health`, and served all five deterministic smoke cases:

| Case | Input | Output | Result |
| --- | --- | --- | --- |
| `text_only` | text | text | PASS |
| `text_audio` | text | text and audio | PASS |
| `image_audio` | image | text and audio | PASS |
| `audio_audio` | audio | text and audio | PASS |
| `video_audio` | video | text and audio | PASS |

All generated audio passed the suite's WAV structure, non-empty PCM, and
stream-completion checks.

## Reduced proxy benchmark

The submission tooling was exercised with two measured requests per point and
one warmup. These small samples verify the path and artifact schema; they are
not statistically meaningful competition measurements.

| Mode | Concurrency | OK/failed | First text p50 (s) | First audio p50 (s) | E2E p50 (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| text | 1 | 2/0 | 0.088 | - | 1.517 |
| text | 2 | 2/0 | 7.468 | - | 10.571 |
| text plus audio | 1 | 2/0 | 0.783 | 1.691 | 2.234 |
| text plus audio | 2 | 2/0 | 0.817 | 2.500 | 3.450 |

A separate text-plus-audio stability point at concurrency 1 passed 2/2
requests with 0 failures. The machine-readable correctness gate passed with
an empty failure list. Raw local artifacts were written to
`artifacts/minicpmo_ascend/baseline-20260803-real/` and are intentionally not
committed because they include generated media and machine-specific paths.

## Regression validation

- Competition and profile tools: 11 tests passed.
- Scheduler, engine-output compatibility, Code2Wav batching, CosyVoice2 NPU,
  StepAudio2 Token2Wav, and NPU worker compatibility: 65 tests passed.
- Ruff checks passed for every changed Python file.
- Python bytecode compilation, Bash syntax, YAML parsing, and
  `git diff --cached --check` passed.

## Formal submission blockers

The organizer starter kit URL was unavailable from the development machine at
validation time. Consequently, the official evaluator schema, exact benchmark
subsets, and score weights remain unresolved. A formal submission still needs
official Daily-Omni, TTS-Seed, and Video-MME effect results, the official
RTF/TTFT/TTFP run, and the required stable Demo recording. This branch is a
tested, reproducible baseline for those steps; it is not represented as a
platform submission.
