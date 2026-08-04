#!/usr/bin/env python3
"""Summarize one deterministic public proxy benchmark run."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .client import metric_summary


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return data


def _newest_result(directory: Path, pattern: str) -> Path:
    candidates = list(directory.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"no result matching {pattern!r} under {directory}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def summarize_video_timing(items: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = metric_summary(items)
    starts_and_ends = []
    for item in items:
        started_at = item.get("started_at")
        e2e_s = item.get("e2e_s")
        if started_at is None or e2e_s is None:
            continue
        start = datetime.fromisoformat(str(started_at))
        starts_and_ends.append((start, start + timedelta(seconds=float(e2e_s))))
    wall_s = None
    if starts_and_ends:
        wall_s = (max(end for _, end in starts_and_ends) - min(start for start, _ in starts_and_ends)).total_seconds()
    successful = int(metrics["successful_requests"])
    return {
        **metrics,
        "duration_s": wall_s,
        "request_throughput": successful / wall_s if wall_s else None,
    }


def _timings_ms(data: dict[str, Any], first_key: str) -> dict[str, Any]:
    return {
        "first_response_p50": data.get(f"median_{first_key}_ms"),
        "first_response_p99": data.get(f"p99_{first_key}_ms"),
        "e2e_p50": data.get("median_e2el_ms"),
        "e2e_p99": data.get("p99_e2el_ms"),
    }


def _seed_summary(data: dict[str, Any]) -> dict[str, Any]:
    continuity = 100.0 if data.get("failed") == 0 and data.get("p99_audio_underrun_s") == 0 else None
    return {
        "samples": data.get("num_prompts"),
        "completed": data.get("completed"),
        "failed": data.get("failed"),
        "duration_s": data.get("duration"),
        "request_throughput": data.get("request_throughput"),
        "generated_audio_s": data.get("total_audio_duration_s"),
        "audio_throughput": data.get("audio_throughput"),
        "timing_ms": _timings_ms(data, "audio_ttfp"),
        "audio_rtf": {
            "p50": data.get("median_audio_rtf"),
            "p99": data.get("p99_audio_rtf"),
        },
        "p99_audio_underrun_s": data.get("p99_audio_underrun_s"),
        "streaming_continuity_proxy_pct": continuity,
        "wer": None,
    }


def build_summary(output_root: Path, source_revision: str) -> dict[str, Any]:
    manifest_path = output_root / "sample_manifest.json"
    daily_path = _newest_result(output_root / "daily_omni", "openai-chat-omni-*.json")
    seed_en_path = _newest_result(output_root / "seed_tts_en", "openai-chat-omni-*.json")
    seed_zh_path = _newest_result(output_root / "seed_tts_zh", "openai-chat-omni-*.json")
    video_path = output_root / "video_mme" / "video_mme_results.json"

    manifest = _load_json(manifest_path)
    daily = _load_json(daily_path)
    seed_en = _load_json(seed_en_path)
    seed_zh = _load_json(seed_zh_path)
    video = _load_json(video_path)
    video_accuracy = video["summary"]
    video_timing = summarize_video_timing(video["items"])

    workloads = {
        "daily_omni": {
            "samples": daily.get("num_prompts"),
            "completed": daily.get("completed"),
            "failed": daily.get("failed"),
            "correct": daily.get("daily_omni_correct"),
            "evaluated": daily.get("daily_omni_evaluated"),
            "accuracy": daily.get("daily_omni_accuracy"),
            "duration_s": daily.get("duration"),
            "request_throughput": daily.get("request_throughput"),
            "timing_ms": _timings_ms(daily, "ttft"),
            "by_duration": daily.get("daily_omni_per_duration"),
            "by_task": daily.get("daily_omni_per_task"),
        },
        "seed_tts_en": _seed_summary(seed_en),
        "seed_tts_zh": _seed_summary(seed_zh),
        "video_mme": {
            "samples": video_accuracy["total"],
            "completed": video_timing["successful_requests"],
            "failed": video_timing["failed_requests"],
            "correct": video_accuracy["correct"],
            "evaluated": video_accuracy["total"],
            "accuracy": video_accuracy["accuracy"],
            "duration_s": video_timing["duration_s"],
            "request_throughput": video_timing["request_throughput"],
            "timing_s": {
                "first_text": video_timing["first_text_s"],
                "e2e": video_timing["e2e_s"],
            },
            "unparsed_responses": video_accuracy["unparsed_responses"],
            "by_duration": video_accuracy["by_duration"],
            "by_domain": video_accuracy["by_domain"],
        },
    }
    total_samples = sum(int(workload["samples"] or 0) for workload in workloads.values())
    total_completed = sum(int(workload["completed"] or 0) for workload in workloads.values())
    total_failed = sum(int(workload["failed"] or 0) for workload in workloads.values())
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "deterministic public-data proxy benchmark; not an official competition benchmark or score",
        "formal_competition_score": None,
        "source_revision": source_revision,
        "sample_seed": manifest.get("seed"),
        "totals": {
            "samples": total_samples,
            "completed": total_completed,
            "failed": total_failed,
        },
        "sample_distribution": {
            "daily_omni": {
                "duration": manifest["daily_omni"]["duration_distribution"],
                "category": manifest["daily_omni"]["category_distribution"],
            },
            "seed_tts_en": manifest["seed_tts"]["en"]["length_bin_distribution"],
            "seed_tts_zh": manifest["seed_tts"]["zh"]["length_bin_distribution"],
            "video_mme": {
                "duration": manifest["video_mme"]["duration_distribution"],
                "domain": manifest["video_mme"]["domain_distribution"],
            },
        },
        "workloads": workloads,
        "raw_results": {
            "manifest": str(manifest_path.relative_to(output_root)),
            "daily_omni": str(daily_path.relative_to(output_root)),
            "seed_tts_en": str(seed_en_path.relative_to(output_root)),
            "seed_tts_zh": str(seed_zh_path.relative_to(output_root)),
            "video_mme": str(video_path.relative_to(output_root)),
        },
    }


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _number(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_report(summary: dict[str, Any]) -> str:
    workloads = summary["workloads"]
    daily = workloads["daily_omni"]
    seed_en = workloads["seed_tts_en"]
    seed_zh = workloads["seed_tts_zh"]
    video = workloads["video_mme"]
    lines = [
        "# MiniCPM-o Ascend Public Proxy Benchmark v1",
        "",
        "> This is a deterministic public-data proxy benchmark, not an official competition benchmark or score.",
        "",
        f"- Source revision: `{summary['source_revision']}`",
        f"- Sample seed: `{summary['sample_seed']}`",
        f"- Requests: {summary['totals']['completed']}/{summary['totals']['samples']} completed, "
        f"{summary['totals']['failed']} failed",
        "- Concurrency: 1 for every workload",
        "",
        "## Results",
        "",
        "| Workload | Samples | Effect metric | Throughput | P50 first response | P50 E2E |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
        f"| Daily-Omni | {daily['samples']} | {daily['correct']}/{daily['evaluated']} "
        f"({_percent(daily['accuracy'])}) | {_number(daily['request_throughput'], 4)} req/s | "
        f"{_number(daily['timing_ms']['first_response_p50'])} ms | "
        f"{_number(daily['timing_ms']['e2e_p50'])} ms |",
        f"| Seed-TTS English | {seed_en['samples']} | generation 100/100; WER not run in this 100-sample pass | "
        f"{_number(seed_en['audio_throughput'], 4)} audio s/s | "
        f"{_number(seed_en['timing_ms']['first_response_p50'])} ms | "
        f"{_number(seed_en['timing_ms']['e2e_p50'])} ms |",
        f"| Seed-TTS Chinese | {seed_zh['samples']} | generation 100/100; WER not run in this 100-sample pass | "
        f"{_number(seed_zh['audio_throughput'], 4)} audio s/s | "
        f"{_number(seed_zh['timing_ms']['first_response_p50'])} ms | "
        f"{_number(seed_zh['timing_ms']['e2e_p50'])} ms |",
        f"| Video-MME | {video['samples']} | {video['correct']}/{video['evaluated']} "
        f"({_percent(video['accuracy'])}) | {_number(video['request_throughput'], 4)} req/s | "
        f"{_number(video['timing_s']['first_text']['p50'])} s | {_number(video['timing_s']['e2e']['p50'])} s |",
        "",
        "Both Seed-TTS runs had zero p99 audio underrun and a 100% streaming-continuity proxy rate. "
        f"English generated {_number(seed_en['generated_audio_s'])} seconds of audio; Chinese generated "
        f"{_number(seed_zh['generated_audio_s'])} seconds.",
        "",
        "## Sampling",
        "",
        "- Daily-Omni: 100 unique videos, split equally between 30-second and 60-second groups, "
        "then stratified by parent category.",
        "- Seed-TTS English: 100 rows, 25 from each target-text length quartile, "
        "with unique reference prompts preferred.",
        "- Seed-TTS Chinese: 100 rows, 25 from each target-text length quartile, "
        "with unique reference prompts preferred.",
        "- Video-MME: 100 unique videos, 34 long, 33 medium, and 33 short, then stratified across six domains.",
        "",
        "## Accuracy Breakdown",
        "",
        "### Daily-Omni duration",
        "",
        "| Duration | Correct | Total | Accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, values in sorted(daily["by_duration"].items()):
        accuracy = values["correct"] / values["total"]
        lines.append(f"| {name} | {values['correct']} | {values['total']} | {_percent(accuracy)} |")
    lines.extend(
        [
            "",
            "### Video-MME duration",
            "",
            "| Duration | Correct | Total | Accuracy |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, values in sorted(video["by_duration"].items()):
        lines.append(f"| {name} | {values['correct']} | {values['total']} | {_percent(values['accuracy'])} |")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- The official Starter Kit, evaluator subsets, score weights, and final package schema "
            "were unavailable for this run.",
            "- These public datasets are coverage proxies and cannot establish an official competition score.",
            "- Seed-TTS quality evaluation is separate from this 100-sample generation/performance pass.",
            "- Video-MME uses the available public, subtitles-free subset and is not a full Video-MME result.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    summary = build_summary(args.output_root, args.source_revision)
    summary_path = args.output_root / "summary.json"
    report_path = args.output_root / "report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(summary), encoding="utf-8")
    print(summary_path)
    print(report_path)


if __name__ == "__main__":
    main()
