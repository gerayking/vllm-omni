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

## Expanded proxy benchmark

The low-sample path check was superseded by an expanded run pinned to commit
`69d29b78feeedd08ef3dd66986a05895ca7b0323`. Each primary configuration used
three warmups followed by 30 measured requests. The matrix covered text and
text-plus-audio output at concurrency 1, 2, and 4, for 180 measured requests.

| Mode | C | OK/failed | First text p50/p95 (s) | First audio p50/p95 (s) | E2E p50/p95 (s) | Req/s | Audio s/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| text | 1 | 30/0 | 0.086/0.091 | -/- | 1.521/1.532 | 0.658 | - |
| text | 2 | 30/0 | 0.104/0.113 | -/- | 1.564/1.878 | 1.209 | - |
| text | 4 | 30/0 | 0.133/0.177 | -/- | 1.623/1.699 | 2.249 | - |
| text plus audio | 1 | 30/0 | 0.772/0.778 | 1.702/1.714 | 3.834/3.924 | 0.257 | 1.914 |
| text plus audio | 2 | 30/0 | 0.810/0.829 | 2.589/2.740 | 6.404/6.456 | 0.311 | 2.310 |
| text plus audio | 4 | 30/0 | 0.824/0.875 | 2.774/3.320 | 8.287/9.684 | 0.468 | 3.410 |

A separate text-plus-audio concurrency-4 stability run used three warmups and
200 measured requests. It completed 200/200 requests with no failures in
471.806 seconds: first text p50/p95 was 0.788/1.176 seconds, first audio was
2.962/3.712 seconds, E2E was 9.311/11.271 seconds, request throughput was 0.424
requests/s, and generated-audio throughput was 3.113 audio seconds/s.

Across the primary and stability runs, all 380 measured requests succeeded.
All 290 measured audio responses were complete and non-empty, and no adjacent
duplicate audio chunks were detected. The machine-readable correctness gate
passed with an empty failure list. Peak aggregate HBM reported by the resource
sampler was 108,899 MiB during stability.

Raw local artifacts are under
`artifacts/minicpmo_ascend/baseline-69d29b78-expanded/` and are intentionally
ignored because they contain generated media and machine-specific paths. Key
SHA-256 values:

- Primary benchmark JSON: `ca45c3cc36a12146952a0891c90682c8a9e3852230695623361c0bee8b348997`
- Stability benchmark JSON: `9277295da5a9b88ddf5935482aefa20a65e08ba993836c4b28aa23c45eca2e65`
- Generated baseline report: `219f6aea4f5b0de57ba4f31da86a366dbd95443061dfb564e02a3ae45cbe27c3`

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
