"""Tests for the CosyVoice3 PR0 benchmark and quality harness."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "benchmarks" / "tts"))
import cosyvoice3_pr0_harness as harness


def test_build_seed_tts_bench_command_uses_fixed_dataset_and_quality_flags(tmp_path: Path) -> None:
    cmd = harness.build_seed_tts_bench_command(
        host="127.0.0.1",
        port=8091,
        model="/models/cosyvoice3",
        tokenizer="/models/cosyvoice3/CosyVoice-BlankEN",
        dataset_path="/data/seeden100",
        result_dir=tmp_path,
        result_filename="c4.json",
        concurrency=4,
        num_prompts=100,
        quality_eval=True,
        save_quality_items=True,
        ready_check_timeout_sec=0,
    )

    assert cmd[:4] == ["vllm", "bench", "serve", "--omni"]
    assert cmd[cmd.index("--backend") + 1] == "openai-audio-speech"
    assert cmd[cmd.index("--endpoint") + 1] == "/v1/audio/speech"
    assert cmd[cmd.index("--dataset-name") + 1] == "seed-tts"
    assert cmd[cmd.index("--dataset-path") + 1] == "/data/seeden100"
    assert cmd[cmd.index("--max-concurrency") + 1] == "4"
    assert cmd[cmd.index("--request-rate") + 1] == "inf"
    assert "--disable-shuffle" in cmd
    assert "--seed-tts-inline-ref-audio" in cmd
    assert "--seed-tts-wer-eval" in cmd
    assert "--seed-tts-wer-save-items" in cmd
    assert cmd[cmd.index("--ready-check-timeout-sec") + 1] == "0"


def test_summarize_benchmark_result_normalizes_latency_and_quality_fields() -> None:
    summary = harness.summarize_benchmark_result(
        {
            "completed": 100,
            "failed": 0,
            "duration": 10.0,
            "request_throughput": 10.0,
            "audio_throughput": 43.0,
            "mean_e2el_ms": 101.0,
            "median_e2el_ms": 90.0,
            "p99_e2el_ms": 150.0,
            "mean_audio_ttfp_ms": 20.0,
            "median_audio_ttfp_ms": 18.0,
            "p99_audio_ttfp_ms": 30.0,
            "mean_audio_rtf": 0.8,
            "median_audio_rtf": 0.7,
            "p99_audio_rtf": 1.2,
            "mean_audio_underrun_s": 0.05,
            "median_audio_underrun_s": 0.0,
            "p99_audio_underrun_s": 0.2,
            "mean_audio_duration_s": 4.0,
            "total_audio_duration_s": 400.0,
            "seed_tts_content_error_mean": 0.12,
            "seed_tts_sim_mean": 0.81,
            "seed_tts_utmos_mean": 3.7,
        },
        label="c4_quality",
        source=Path("/tmp/c4.json"),
    )

    assert summary["label"] == "c4_quality"
    assert summary["completed"] == 100
    assert summary["e2el_ms"]["mean"] == 101.0
    assert summary["e2el_ms"]["p50"] == 90.0
    assert summary["e2el_ms"]["p90"] is None
    assert summary["e2el_ms"]["p99"] == 150.0
    assert summary["audio_ttfp_ms"]["p50"] == 18.0
    assert summary["audio_rtf"]["mean"] == 0.8
    assert summary["audio_underrun_s"]["p99"] == 0.2
    assert summary["quality"]["wer_or_cer_mean"] == 0.12
    assert summary["quality"]["speaker_similarity_mean"] == 0.81
    assert summary["quality"]["utmos_mean"] == 3.7
    assert summary["source"] == "/tmp/c4.json"


def test_audio_sanity_detects_duration_non_silence_and_clipping() -> None:
    samples = np.array([0.0, 0.1, -0.2, 1.0, -1.0], dtype=np.float32)

    sanity = harness.compute_audio_sanity(samples, sample_rate=5, clip_threshold=0.999, silence_rms_threshold=0.01)

    assert sanity["sample_rate"] == 5
    assert sanity["num_samples"] == 5
    assert sanity["duration_s"] == 1.0
    assert sanity["finite"] is True
    assert sanity["non_silent"] is True
    assert sanity["clipped_samples"] == 2
    assert sanity["clipped_ratio"] == 0.4


def test_compare_mel_arrays_reports_threshold_pass_fail() -> None:
    reference = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    candidate = reference + np.array([[0.0, 0.01], [0.02, 0.03]], dtype=np.float32)

    result = harness.compare_mel_arrays(reference, candidate, mean_abs_threshold=0.02, max_abs_threshold=0.04)

    assert result["shape_match"] is True
    assert result["finite"] is True
    assert result["mean_abs"] <= 0.02
    assert result["max_abs"] <= 0.04
    assert result["passed"] is True


def test_make_profile_deploy_adds_profiler_config_to_selected_stages(tmp_path: Path) -> None:
    deploy = {
        "stages": [
            {"stage_id": 0, "max_num_seqs": 8},
            {"stage_id": 1, "max_num_seqs": 4},
        ]
    }
    src = tmp_path / "deploy.yaml"
    dst = tmp_path / "profile.yaml"
    src.write_text(yaml.safe_dump(deploy), encoding="utf-8")

    harness.make_profile_deploy(
        input_yaml=src,
        output_yaml=dst,
        torch_profiler_dir=tmp_path / "torch_profiles",
        stages={1},
    )

    out = yaml.safe_load(dst.read_text(encoding="utf-8"))
    assert "profiler_config" not in out["stages"][0]
    assert out["stages"][1]["profiler_config"]["profiler"] == "torch"
    assert out["stages"][1]["profiler_config"]["torch_profiler_dir"] == str(tmp_path / "torch_profiles")


def test_render_report_contains_required_sections(tmp_path: Path) -> None:
    report = harness.render_markdown_report(
        run_id="run-a",
        model="/models/cosyvoice3",
        dataset_path="/data/seeden100",
        benchmark_summaries=[
            {
                "label": "c4",
                "completed": 100,
                "failed": 0,
                "request_throughput": 1.0,
                "audio_throughput": 4.0,
                "e2el_ms": {"mean": 10.0, "p50": 9.0, "p90": None, "p99": 20.0},
                "audio_ttfp_ms": {"mean": 2.0, "p50": 1.0, "p90": None, "p99": 3.0},
                "audio_rtf": {"mean": 0.8, "p50": 0.7, "p90": None, "p99": 1.1},
                "audio_underrun_s": {"mean": 0.1, "p50": 0.0, "p90": None, "p99": 0.3},
                "quality": {"wer_or_cer_mean": 0.12, "speaker_similarity_mean": 0.8, "utmos_mean": 3.6},
                "source": str(tmp_path / "c4.json"),
            }
        ],
        audio_sanity=[],
        mel_diff=None,
        profile_summary=None,
        artifact_root=tmp_path,
    )

    assert "## Purpose" in report
    assert "## Performance Result" in report
    assert "## Precision / Quality Result" in report
    assert "## Artifact Index" in report
    assert "c4" in report
