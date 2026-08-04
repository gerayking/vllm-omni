# MiniCPM-o 4.5 Ascend Competition Suite

This directory provides the reproducible local proxy suite for Track 1,
vLLM-Omni sub-track B. The organizer now publishes the single-card Ascend 910C
target, the `v0.25.0-a3` image, the 2-percentage-point accuracy-loss gate, the
Demo gate, and the RTF/TTFT/TTFP performance objectives. Exact evaluator
statistics, score weights, benchmark subsets, and package schema still depend
on the unavailable starter kit and final evaluation document.

The local suite is a pre-submission gate, not an implementation of the
official score. Its first-text and first-audio timings are explicitly proxy
signals for TTFT and TTFP.

## Official admission and submission gates

Before uploading a package, all of the following must be present:

- Daily-Omni, TTS-Seed, and Video-MME results against the official baseline,
  with no more than 2 percentage points of accuracy loss.
- A stable end-to-end run through the designated vLLM-Omni Demo, including
  audio, video, text, and continuous streamed speech, plus a demonstration
  recording.
- RTF, TTFT, and TTFP results from the official evaluator, with raw output,
  exact commands, environment, run count, statistics, before/after comparison,
  resource use, and anomaly notes.
- Complete code, deploy configuration, service/benchmark/Demo scripts,
  dependencies, optimization analysis, and reproduction instructions.

See `.claude/skills/optimize-minicpmo-ascend/references/competition-rules.md`
and `environment_manifest.yaml` for the dated rule snapshot. Do not upload a
proxy-only package as a formal scored submission.

## 1. Resolve the environment

Update `environment_manifest.yaml` from the newest official announcement.
Every `UNRESOLVED` value must be resolved before a formal run. Capture the
actual machine separately:

```bash
.venv/bin/python -m benchmarks.competition.minicpmo_ascend.collect_environment \
  --output artifacts/minicpmo_ascend/environment.json \
  --starter-kit /path/to/starter-kit.tar.gz \
  --model-path /path/to/MiniCPM-o-4_5 \
  --model-manifest /path/to/model-sha256.txt
```

The collector records the Git SHA and dirty diff, package versions, CANN
version files, physical-card and logical-chip inventory from `npu-smi`, and
checksums of supplied manifests/artifacts. It deliberately does not hash a
multi-gigabyte model tree implicitly. When the model directory has no revision
metadata, create a relative-path manifest explicitly:

```bash
cd /path/to/MiniCPM-o-4_5
find . -type f -print0 | sort -z | xargs -0 sha256sum \
  > /path/to/model-sha256.txt
```

## 2. Start the server

```bash
MODEL=/path/to/MiniCPM-o-4_5 \
MODEL_REVISION=<fixed-revision> \
bash benchmarks/competition/minicpmo_ascend/start_server.sh
```

The default deployment is
`vllm_omni/deploy/minicpmo_4_5_ascend_910c_1card.yaml`. It uses both logical
chips exposed by one physical Ascend 910C card; device IDs `0` and `1` do not
mean two cards. Override `DEPLOY_CONFIG` only with a checked-in candidate
configuration.

## 3. Run the gated proxy suite

Provide a deterministic local video. Image and audio fixtures are generated
offline by the suite.

```bash
VIDEO_INPUT=/path/to/official-or-local-fixture.mp4 \
MODEL=/path/to/MiniCPM-o-4_5 \
MODEL_PATH=/path/to/MiniCPM-o-4_5 \
MODEL_MANIFEST=/path/to/model-sha256.txt \
bash benchmarks/competition/minicpmo_ascend/run_suite.sh \
  --concurrency 1 2 4 --num-requests 20 --warmups 2
```

The command runs multimodal smoke validation first, then separate text-only
and text-plus-audio benchmarks, a text-plus-audio stability run, raw NPU/host
resource collection, the machine-readable correctness gate, and a baseline
report with an artifact checksum manifest. A failed, timed-out, truncated,
empty, or invalid-audio request is excluded from metrics and makes the command
fail.

Raw per-request output includes first SSE event, first text, first audio,
audio chunk arrival/inter-chunk times, E2E, finish reasons, error details, WAV
format, chunk hashes, and reconstructed audio. Results are labeled
`local_proxy`; no unofficial composite score is emitted.

Install the optional public benchmark evaluator dependencies before running the
three effect checks below:

```bash
.venv/bin/python -m pip install -r \
  benchmarks/competition/minicpmo_ascend/requirements-eval.txt
```

## 4. Build and run the fixed public proxy benchmark

Build the versioned four-workload sample set with seed 42. The default 400
samples contain 100 Daily-Omni questions, 100 English Seed-TTS rows, 100
Chinese Seed-TTS rows, and 100 Video-MME questions. Daily-Omni is balanced by
video duration before category, Seed-TTS is balanced by target-text length
quartile, and Video-MME is balanced by duration before domain. Media inputs are
unique within each workload where the public data permits it.

```bash
/workspace/minicpmo-npu-venv/bin/python -m \
  benchmarks.competition.minicpmo_ascend.build_public_benchmark \
  --daily-qa /data/Daily-Omni/qa.json \
  --daily-video-dir /data/Daily-Omni/Videos \
  --seed-tts-root /data/seed-tts-eval \
  --video-mme-metadata /data/Video-MME/videomme/test-00000-of-00001.parquet \
  --video-mme-video-dir /data/Video-MME/extracted/data \
  --output-dir /data/public-proxy-benchmark-v1 \
  --seed 42
```

Start the server with local media access covering `/data` and MiniCPM
interleaved AV strings enabled, then run all four workloads at concurrency 1:

```bash
SAMPLE_ROOT=/data/public-proxy-benchmark-v1 \
DATA_ROOT=/data \
MODEL=/path/to/MiniCPM-o-4_5 \
BENCH_BIN=/path/to/vllm-omni \
PYTHON=/path/to/python \
SOURCE_REVISION=<tested-source-revision> \
bash benchmarks/competition/minicpmo_ascend/run_public_proxy_benchmark.sh
```

This suite is a deterministic public-data proxy, not an official competition
benchmark or score. Keep the generated `manifest.json`, `summary.json`, and
`report.md` with every result set. The 100-row Seed-TTS passes measure
generation success, streaming continuity, TTFP, E2E, RTF, and throughput; run
the separate ASR evaluator when WER is required.

## 5. Run the Daily-Omni proxy effect check

```bash
DAILY_OMNI_QA_JSON=/data/Daily-Omni/qa.json \
DAILY_OMNI_VIDEO_DIR=/data/Daily-Omni/Videos \
DAILY_OMNI_INPUT_MODE=all \
DAILY_OMNI_PACK_MODE=minicpm-interleave \
bash benchmarks/competition/minicpmo_ascend/run_daily_omni.sh
```

This requests text only and disables thinking so A-D answer extraction is
stable. Start MiniCPM-o with `--interleave-mm-strings` when using the default
`minicpm-interleave` pack mode. Replace this public benchmark with the exact
competition subset and protocol when the organizer releases them.

## 6. Run the public Seed-TTS effect check

Use the standard `en/meta.lst` or `zh/meta.lst` split from the public
`seed-tts-eval` release. Start the server with `ALLOWED_LOCAL_MEDIA_PATH`
covering the dataset root because reference WAV files are sent as `file://`
URLs.

```bash
SEED_TTS_ROOT=/data/seed-tts-eval \
SEED_TTS_LOCALE=en \
MODEL=/path/to/MiniCPM-o-4_5 \
bash benchmarks/competition/minicpmo_ascend/run_seed_tts.sh
```

Set `WER_EVAL=1` to run the public Seed-TTS ASR/WER evaluator. Its official
English protocol loads Whisper-large-v3; the Chinese protocol loads
Paraformer-zh. Keep WavLM SIM and UTMOS results labeled as proxies because
their public checkpoints are not identical to the organizer's unreleased
competition evaluator.

## 7. Run the public Video-MME effect check

Download and extract the public Video-MME release, then point the runner at
the official parquet metadata and MP4 directory. Large local videos are sent
as `file://` URLs, so start the server with `ALLOWED_LOCAL_MEDIA_PATH` covering
the dataset root.

```bash
VIDEO_MME_METADATA=/data/Video-MME/videomme/test-00000-of-00001.parquet \
VIDEO_MME_VIDEO_DIR=/data/Video-MME/videos \
MODEL=/path/to/MiniCPM-o-4_5 \
bash benchmarks/competition/minicpmo_ascend/run_video_mme.sh \
  --durations short medium long --concurrency 1
```

Use `--max-videos 1` or `--num-questions 3` for a deterministic smoke run.
The runner emits raw request records, an accuracy breakdown, and the grouped
JSON structure documented by Video-MME. It evaluates the public,
subtitles-free protocol; it is not a substitute for an unreleased competition
subset or score.

## 8. Capture an NPU profile

Keep profiling separate from score measurements. The profile runner generates
a temporary deploy config, starts a clean server, warms it outside the capture
window, profiles a fixed request, stops the server, and emits a unified JSON
and Markdown summary:

```bash
MODEL=/path/to/MiniCPM-o-4_5 \
PROFILE_ID=stage2-baseline \
PROFILE_STAGES=2 \
bash benchmarks/competition/minicpmo_ascend/run_profile.sh
```

Use a unique `PROFILE_ID` for every capture; the runner refuses to mix a new
capture into a non-empty artifact directory.

Profile all stages only when stage ownership is unclear:

```bash
PROFILE_ID=all-stages-baseline PROFILE_STAGES=0,1,2 \
bash benchmarks/competition/minicpmo_ascend/run_profile.sh
```

Artifacts are written under
`artifacts/minicpmo_ascend/profiles/<profile-id>/`. Compare the same workload,
stage selection, profiler configuration, and environment with:

```bash
.venv/bin/python -m benchmarks.competition.minicpmo_ascend.profile_analysis compare \
  artifacts/minicpmo_ascend/profiles/baseline/profile_analysis.json \
  artifacts/minicpmo_ascend/profiles/candidate/profile_analysis.json \
  --output artifacts/minicpmo_ascend/profiles/comparison.json
```

Profiler timing is diagnostic evidence and must never replace the unprofiled
baseline/candidate benchmark.

## Formal-run rules

- Use clean server restarts and record warmup separately from measurements.
- Keep profiler runs separate from score runs.
- Never compare text-only and text-plus-audio as the same workload.
- Preserve raw JSON, server logs, resource samples, WAV artifacts, exact
  commands, Git diff, model manifest, and starter-kit checksum.
- Do not report a formal score until the starter kit and official evaluator
  are available, all submission-critical `UNRESOLVED` manifest fields are
  resolved, and the three official effect suites plus Demo gate pass.
