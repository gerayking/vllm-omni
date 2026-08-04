# Public Proxy Benchmark v1 - 2026-08-04

This report records a deterministic public-data proxy run. It is not an
official competition benchmark or score. The official Starter Kit, evaluator
subsets, score weights, and final package schema were not available for this
run.

## Run identity

- Tested source revision: `69d29b78feeedd08ef3dd66986a05895ca7b0323`
- Sample seed: `42`
- Hardware: one physical Ascend 910C, logical devices 0 and 1
- Concurrency: 1 for every workload
- Requests: 400/400 completed, 0 failed
- Raw artifacts: `artifacts/minicpmo_ascend/public-proxy-benchmark-v1-baseline-69d29b78`

## Fixed sample protocol

| Workload | Public source rows available | Selected | Stratification |
| --- | ---: | ---: | --- |
| Daily-Omni | 1,197 | 100 | 50 each for 30s/60s, then parent category; one question per video |
| Seed-TTS English | 1,088 | 100 | 25 per target-text length quartile; unique prompt preferred |
| Seed-TTS Chinese | 2,020 | 100 | 25 per target-text length quartile; unique prompt preferred |
| Video-MME | 1,380 questions / 460 videos | 100 | 34 long, 33 medium, 33 short, then six domains; one question per video |

## Results

| Workload | Effect result | Throughput | P50 first response | P50 E2E | P99 E2E |
| --- | --- | ---: | ---: | ---: | ---: |
| Daily-Omni | 73/100 (73.00%) | 0.1462 req/s | 5,557.30 ms | 5,623.38 ms | 9,545.29 ms |
| Seed-TTS English | 100/100 generated | 1.6696 audio s/s | 1,489.78 ms TTFP | 2,611.86 ms | 3,457.90 ms |
| Seed-TTS Chinese | 100/100 generated | 1.7262 audio s/s | 1,563.03 ms TTFP | 2,876.16 ms | 3,881.47 ms |
| Video-MME | 65/100 (65.00%) | 0.1137 req/s | 5.71 s | 5.75 s | 27.54 s |

Seed-TTS English generated 446.96 seconds of audio with median RTF 0.6040.
Seed-TTS Chinese generated 501.08 seconds with median RTF 0.5844. Both runs
had zero p99 audio underrun and a 100% streaming-continuity proxy rate.

The separate five-sample public Whisper-large-v3 English check evaluated 5/5
generated clips with mean WER 0.0000 and median WER 0.0000. The separate
five-sample public Paraformer-zh Chinese check evaluated 5/5 clips with mean
WER 0.0091 and median WER 0.0000. WER was not rerun for either 100-sample
generation pass; those passes cover generation and performance rather than
ASR quality.

## Accuracy breakdown

Daily-Omni:

| Duration | Correct | Total | Accuracy |
| --- | ---: | ---: | ---: |
| 30s | 35 | 50 | 70.00% |
| 60s | 38 | 50 | 76.00% |

Video-MME:

| Duration | Correct | Total | Accuracy |
| --- | ---: | ---: | ---: |
| Long | 16 | 34 | 47.06% |
| Medium | 27 | 33 | 81.82% |
| Short | 22 | 33 | 66.67% |

## Interpretation

The suite now covers the basic public data paths for all three organizer-named
effect families: Daily-Omni, Seed-TTS in English and Chinese, and Video-MME.
Seed-TTS is counted as two workloads because language-specific generation and
ASR quality paths differ. These results remain proxy evidence until the
organizer publishes the formal Starter Kit and exact evaluation protocol.
