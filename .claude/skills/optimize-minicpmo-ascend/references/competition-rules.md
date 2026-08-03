# Competition Rules Snapshot

Snapshot date: 2026-08-03 (China Standard Time)

Primary source: https://ascend.openbmb.cn/competition

This snapshot records the published rules used to prepare this baseline. The
official site, organizer announcements, Feishu group, starter kit, and final
evaluation document remain authoritative when they differ from this file.

## Selected Track

This repository targets Track 1, sub-track B: MiniCPM-o 4.5 inference with
vLLM-Omni on Ascend.

- Official evaluation hardware: one physical Ascend 910C card.
- Official A3 image: `quay.io/ascend/vllm-omni:v0.25.0-a3`.
- A2 reference image: `quay.io/ascend/vllm-omni:v0.25.0`.
- Upstream contribution target: the vLLM-Omni `minicpm-challenge` branch.
- Submissions close on 2026-08-17 (China Standard Time).
- Each team may submit at most three times per day.

The organizer evaluates the llama.cpp-omni and vLLM-Omni sub-tracks against
their own official baselines. Results are not compared across frameworks.

## Admission Gates

A submission enters performance ranking only after both gates pass.

### Accuracy and capability

The optimized result may lose no more than 2 percentage points relative to the
official vLLM-Omni baseline. The published benchmark families are:

- Daily-Omni.
- TTS-Seed.
- Video-MME.

The organizer still controls the exact dataset revisions, subsets, request
format, evaluation scripts, and aggregation. A submission fails this gate if
it cannot complete a benchmark, produces abnormal output, materially changes
model behavior, or loses a core capability.

### Demo usability

The optimized service must connect to the designated vLLM-Omni Demo and run a
stable end-to-end interaction. Validation covers service startup, Demo
connection, audio/video/text input, complete output, continuous streamed
speech, the full official interaction flow, and sustained stability. Passing
benchmarks without a working Demo is insufficient.

## Performance Metrics

After admission, sub-track B is ranked using three lower-is-better metrics:

- Per-chunk RTF: compute time for an audio chunk divided by that chunk's audio
  duration.
- TTFT: time from request receipt to the first valid output token.
- TTFP: time from request receipt to the first usable audio packet/chunk.

The official TTFT boundary, preprocessing treatment, valid-token definition,
RTF aggregation, normalization, score formula, and metric weights are not yet
published. Do not invent a composite score. Local first-text and first-audio
timings are proxy evidence only until the official evaluator is available.

## Required Submission Materials

The published final deliverables are:

1. Complete adaptation and optimization code and vLLM-Omni configuration.
2. Service, benchmark, and Demo startup scripts plus dependency/environment
   files.
3. Complete Daily-Omni, TTS-Seed, and Video-MME results, including commands,
   parameters, raw output, and summaries.
4. A performance report containing RTF, TTFT, TTFP, environment, data, run
   count, statistics, before/after comparison, resource usage, and anomalies.
5. A runnable Demo, usage and access instructions, core interaction flow, and
   a recorded demonstration video.
6. Bottleneck analysis, optimization details, per-change performance effects,
   capability retention, key technical notes, and complete reproduction steps.

The organizer reproduces submissions in the unified environment. Missing
files, unstable execution, or materially inconsistent reproduced results may
invalidate a score.

## Current Starter-Kit Status

The competition frontend advertises:

`https://oiac.openbmb.org/toolkit/oiac-toolkit-v1.0.tar.gz`

On 2026-08-03 the hostname returned no DNS answer from the HiDevLab machine,
so the artifact could not be downloaded or checksummed. The frontend also says
that benchmark subsets, exact evaluation scripts, score weights, and final
package format remain subject to the starter kit and final evaluation
document. Keep those manifest fields unresolved until an authoritative
artifact becomes accessible.

## Rules That Remain Dynamic

Recheck these items before every scored run or upload:

- Starter-kit URL, version, checksum, and package schema.
- Official model revision and model packaging rules.
- Daily-Omni, TTS-Seed, and Video-MME revisions and subsets.
- Request schema, concurrency, warmup, run count, timeouts, and statistics.
- Exact RTF, TTFT, and TTFP definitions, normalization, weights, and score.
- Demo document and endpoint.
- Driver, firmware, CANN, Python, PyTorch, torch-npu, vLLM, and vLLM-Ascend
  versions inside the official image.
- Network, cache, quantization, runtime, and package-size restrictions.
- Submission endpoint status and organizer announcements.

## Refresh Checklist

Before a formal run:

1. Re-fetch the official competition page and announcements.
2. Check the team Feishu group and submission guide.
3. Download the current starter kit and record its SHA256.
4. Resolve every required field in `environment_manifest.yaml`.
5. Run all three official accuracy suites and the official Demo gate.
6. Run the unprofiled official performance evaluator in a clean environment.
7. Preserve raw output, logs, resource samples, exact commands, and Git SHA.
