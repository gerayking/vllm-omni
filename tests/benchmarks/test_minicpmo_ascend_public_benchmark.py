from pathlib import Path

import pytest

from benchmarks.competition.minicpmo_ascend.build_public_benchmark import (
    _length_bins,
    _load_seed_rows,
    select_hierarchical,
    select_stratified,
)
from benchmarks.competition.minicpmo_ascend.summarize_public_benchmark import (
    render_report,
    summarize_video_timing,
)


def test_select_stratified_is_deterministic_balanced_and_unique() -> None:
    rows = [
        {"id": f"{group}-{index}", "media": f"media-{index}", "group": group}
        for group in ("a", "b", "c")
        for index in range(8)
    ]
    first = select_stratified(
        rows,
        count=6,
        seed=42,
        stratum_key=lambda row: row["group"],
        unique_key=lambda row: row["media"],
    )
    second = select_stratified(
        rows,
        count=6,
        seed=42,
        stratum_key=lambda row: row["group"],
        unique_key=lambda row: row["media"],
    )
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len({row["media"] for row in first}) == 6
    assert {row["group"] for row in first} == {"a", "b", "c"}


def test_seed_rows_filter_missing_audio_and_support_length_bins(tmp_path: Path) -> None:
    locale_root = tmp_path / "en"
    prompt_root = locale_root / "prompt-wavs"
    prompt_root.mkdir(parents=True)
    (prompt_root / "present.wav").write_bytes(b"RIFF")
    (locale_root / "meta.lst").write_text(
        "one|prompt|prompt-wavs/present.wav|short\n"
        "two|prompt|prompt-wavs/missing.wav|this row is unavailable\n"
        "three|prompt|prompt-wavs/present.wav|a substantially longer target sentence\n",
        encoding="utf-8",
    )
    rows = _load_seed_rows(tmp_path, "en")
    thresholds, length_bin = _length_bins(rows)
    assert [row["utterance_id"] for row in rows] == ["one", "three"]
    assert len(thresholds) == 3
    assert length_bin(rows[0]) < length_bin(rows[1])


def test_select_hierarchical_balances_primary_groups_first() -> None:
    rows = [
        {"id": f"{duration}-{domain}-{index}", "duration": duration, "domain": domain}
        for duration in ("short", "long")
        for domain in ("knowledge", "sports", "film")
        for index in range(4)
    ]
    selected = select_hierarchical(
        rows,
        count=10,
        seed=42,
        primary_key=lambda row: row["duration"],
        secondary_key=lambda row: row["domain"],
        unique_key=lambda row: row["id"],
    )
    assert sum(row["duration"] == "short" for row in selected) == 5
    assert sum(row["duration"] == "long" for row in selected) == 5
    assert {row["domain"] for row in selected} == {"knowledge", "sports", "film"}


def test_summarize_video_timing_includes_percentiles_and_wall_throughput() -> None:
    items = [
        {
            "success": True,
            "started_at": "2026-08-04T00:00:00+00:00",
            "first_event_s": 1.0,
            "first_text_s": 1.5,
            "first_audio_s": None,
            "e2e_s": 2.0,
            "audio_chunk_intervals_s": [],
        },
        {
            "success": True,
            "started_at": "2026-08-04T00:00:02+00:00",
            "first_event_s": 2.0,
            "first_text_s": 2.5,
            "first_audio_s": None,
            "e2e_s": 3.0,
            "audio_chunk_intervals_s": [],
        },
    ]
    summary = summarize_video_timing(items)
    assert summary["duration_s"] == 5.0
    assert summary["request_throughput"] == 0.4
    assert summary["first_text_s"]["p50"] == 2.0
    assert summary["e2e_s"]["p99"] == pytest.approx(2.99)


def test_render_report_labels_proxy_scope() -> None:
    summary = {
        "source_revision": "abc123",
        "sample_seed": 42,
        "totals": {"samples": 4, "completed": 4, "failed": 0},
        "workloads": {
            "daily_omni": {
                "samples": 1,
                "correct": 1,
                "evaluated": 1,
                "accuracy": 1.0,
                "request_throughput": 1.0,
                "timing_ms": {"first_response_p50": 1.0, "e2e_p50": 2.0},
                "by_duration": {"30s": {"correct": 1, "total": 1}},
            },
            "seed_tts_en": {
                "samples": 1,
                "audio_throughput": 1.0,
                "generated_audio_s": 2.0,
                "timing_ms": {"first_response_p50": 1.0, "e2e_p50": 2.0},
            },
            "seed_tts_zh": {
                "samples": 1,
                "audio_throughput": 1.0,
                "generated_audio_s": 2.0,
                "timing_ms": {"first_response_p50": 1.0, "e2e_p50": 2.0},
            },
            "video_mme": {
                "samples": 1,
                "correct": 1,
                "evaluated": 1,
                "accuracy": 1.0,
                "request_throughput": 1.0,
                "timing_s": {"first_text": {"p50": 1.0}, "e2e": {"p50": 2.0}},
                "by_duration": {"short": {"correct": 1, "total": 1, "accuracy": 1.0}},
            },
        },
    }
    report = render_report(summary)
    assert "not an official competition benchmark or score" in report
    assert "WER not run in this 100-sample pass" in report
