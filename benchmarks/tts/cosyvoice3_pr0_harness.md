# CosyVoice3 PR0 Harness

This harness standardizes the benchmark, quality, and profile evidence required
before changing the CosyVoice3 flow-matching path.

The script does not start the serving process. Start a CosyVoice3 server first,
then run the commands below from the repository root.

## Benchmark

```bash
MODEL_DIR=/path/to/Fun-CosyVoice3-0.5B-2512
TESTSET=/path/to/seeden/testset_seed0_n100
RUN=benchmark_results/cosyvoice3_pr0/$(date +%Y%m%d-%H%M%S)

python benchmarks/tts/cosyvoice3_pr0_harness.py bench \
  --model "$MODEL_DIR" \
  --tokenizer "$MODEL_DIR/CosyVoice-BlankEN" \
  --dataset-path "$TESTSET" \
  --result-dir "$RUN/bench" \
  --concurrency 4 8 \
  --num-prompts 100
```

Add `--quality-eval --save-quality-items` to run Seed-TTS WER/SIM/UTMOS
evaluation through `vllm bench serve`.

## Audio Sanity

```bash
python benchmarks/tts/cosyvoice3_pr0_harness.py audio-sanity \
  --wav generated_0.wav generated_1.wav \
  --output-json "$RUN/quality/audio_sanity.json"
```

The output reports duration, finite values, RMS, peak absolute value,
non-silence, and clipping ratio.

## Mel Diff

Compare deterministic mel captures from a baseline and candidate run:

```bash
python benchmarks/tts/cosyvoice3_pr0_harness.py mel-compare \
  --reference "$RUN/mel/baseline.npy" \
  --candidate "$RUN/mel/candidate.npy" \
  --mean-abs-threshold 0.01 \
  --max-abs-threshold 0.1 \
  --output-json "$RUN/quality/mel_diff.json"
```

The capture step is intentionally separate so future optimization PRs can run
the baseline and candidate at different commits while using the same comparison
schema.

## Profiling

Create a profiler-enabled deploy config:

```bash
python benchmarks/tts/cosyvoice3_pr0_harness.py profile-deploy \
  --input vllm_omni/deploy/cosyvoice3.yaml \
  --output "$RUN/cosyvoice3_profile.yaml" \
  --torch-profiler-dir "$RUN/torch_profiles" \
  --stages 0 1
```

Generate the explicit profile commands:

```bash
python benchmarks/tts/cosyvoice3_pr0_harness.py profile-commands \
  --run-dir "$RUN/profile_c4_n20" \
  --model "$MODEL_DIR" \
  --tokenizer "$MODEL_DIR/CosyVoice-BlankEN" \
  --dataset-path "$TESTSET" \
  --output "$RUN/profile_c4_n20/run_profile.sh"
```

Run the generated script after starting the server with the profiler deploy
config.

## Report

Summarize benchmark JSONs and optional quality/profile JSONs into a PR-ready
report:

```bash
python benchmarks/tts/cosyvoice3_pr0_harness.py summarize \
  --run-id "$(basename "$RUN")" \
  --model "$MODEL_DIR" \
  --dataset-path "$TESTSET" \
  --artifact-root "$RUN" \
  --benchmark-json "$RUN/bench/c4/seeden_c4_n100.json" "$RUN/bench/c8/seeden_c8_n100.json" \
  --mel-diff-json "$RUN/quality/mel_diff.json" \
  --profile-summary-json "$RUN/profile_c4_n20/profile_summary.json" \
  --output-json "$RUN/summary.json" \
  --output-md "$RUN/REPORT.md"
```

Do not commit generated `benchmark_results` artifacts. Include their paths and
the report summary in the PR description.
