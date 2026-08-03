# SPDX-License-Identifier: Apache-2.0

import base64
import io
import json
import subprocess
import sys
import wave
from pathlib import Path

import pytest

from benchmarks.competition.minicpmo_ascend.client import (
    WavAccumulator,
    build_payload,
    media_url,
    metric_summary,
    percentile,
)
from benchmarks.competition.minicpmo_ascend.report import _resource_peaks, _write_manifest
from benchmarks.competition.minicpmo_ascend.video_mme import (
    build_official_results,
    extract_choice,
    summarize,
)


def _wav_chunk(samples: list[int], sample_rate: int = 24000) -> str:
    pcm = b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_wav_accumulator_tracks_format_continuity_and_duplicates(tmp_path: Path) -> None:
    audio = WavAccumulator()
    audio.append(_wav_chunk([1, 2, 3]))
    audio.append(_wav_chunk([1, 2, 3]))

    metadata = audio.metadata()
    assert metadata["chunk_count"] == 2
    assert metadata["pcm_bytes"] == 12
    assert metadata["sample_rate_hz"] == 24000
    assert metadata["adjacent_duplicate_chunks"] == 1
    assert metadata["boundary_jump_abs_pcm16"] == [2]

    path = tmp_path / "joined.wav"
    audio.write(path)
    with wave.open(str(path), "rb") as joined:
        assert joined.getnframes() == 6
        assert joined.getframerate() == 24000


def test_wav_accumulator_rejects_format_change() -> None:
    audio = WavAccumulator()
    audio.append(_wav_chunk([1, 2], sample_rate=24000))
    with pytest.raises(ValueError, match="audio format changed"):
        audio.append(_wav_chunk([3, 4], sample_rate=16000))


def test_payload_and_metrics_keep_modes_and_failures_separate() -> None:
    payload = build_payload(
        model="model",
        prompt="hello",
        input_modality="text",
        media=None,
        with_audio=False,
        seed=42,
        thinker_max_tokens=8,
        talker_max_tokens=16,
    )
    assert payload["modalities"] == ["text"]
    assert payload["chat_template_kwargs"] == {"use_tts_template": False}
    assert payload["sampling_params_list"][0]["max_tokens"] == 8
    assert payload["sampling_params_list"][1]["max_tokens"] == 16

    records = [
        {"success": True, "first_text_s": 1.0, "e2e_s": 2.0},
        {"success": True, "first_text_s": 3.0, "e2e_s": 4.0},
        {"success": False, "first_text_s": 0.01, "e2e_s": 0.02},
    ]
    summary = metric_summary(records)
    assert summary["successful_requests"] == 2
    assert summary["failed_requests"] == 1
    assert summary["first_text_s"]["p50"] == 2.0
    assert summary["e2e_s"]["mean"] == 3.0
    assert percentile([], 0.95) is None


def test_media_url_preserves_explicit_file_uri(tmp_path: Path) -> None:
    uri = (tmp_path / "large.mp4").as_uri()
    assert media_url(uri, "video") == uri


def test_video_mme_choice_and_summary_match_official_shape() -> None:
    assert extract_choice("C") == "C"
    assert extract_choice("Answer: D. Something") == "D"
    assert extract_choice("I cannot tell") is None
    items = [
        {
            "video_id": "001",
            "question_id": "001-1",
            "duration": "short",
            "domain": "Knowledge",
            "sub_category": "History",
            "task_type": "Counting",
            "question": "Q?",
            "options": ["A. One", "B. Two", "C. Three", "D. Four"],
            "gold": "C",
            "response": "C",
            "predicted": "C",
            "correct": True,
            "success": True,
        },
        {
            "video_id": "001",
            "question_id": "001-2",
            "duration": "short",
            "domain": "Knowledge",
            "sub_category": "History",
            "task_type": "Counting",
            "question": "Q2?",
            "options": ["A", "B", "C", "D"],
            "gold": "A",
            "response": "B",
            "predicted": "B",
            "correct": False,
            "success": True,
        },
    ]
    summary = summarize(items)
    assert summary["accuracy"] == 0.5
    assert summary["by_duration"]["short"] == {"correct": 1, "total": 2, "accuracy": 0.5}
    official = build_official_results(items)
    assert official[0]["video_id"] == "001"
    assert [question["response"] for question in official[0]["questions"]] == ["C", "B"]


def test_correctness_gate_writes_failure_when_smoke_artifact_is_missing(tmp_path: Path) -> None:
    output = tmp_path / "gate.json"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.competition.minicpmo_ascend.correctness_gate",
            "--smoke-results",
            str(tmp_path / "missing.json"),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert process.returncode == 1
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["passed"] is False
    assert any("cannot load smoke results" in failure for failure in result["failures"])


def test_resource_peaks_aggregate_single_card_chips(tmp_path: Path) -> None:
    resources = {
        "samples": [
            {
                "host_memory_bytes": {"MemTotal": 1000, "MemAvailable": 400},
                "npu_smi": {
                    "stdout": "\n".join(
                        [
                            "| 0 0 | 0000:9D:00.0 | 72 0 / 0 12000 / 65536 |",
                            "| 0 1 | 0000:9F:00.0 | 81 0 / 0 13000 / 65536 |",
                        ]
                    )
                },
            }
        ]
    }
    path = tmp_path / "resources.json"
    path.write_text(json.dumps(resources), encoding="utf-8")

    assert _resource_peaks(path) == {
        "aicore_percent": 81,
        "aggregate_hbm_mib": 25000,
        "host_memory_bytes": 600,
        "aggregate_hbm_delta_mib": 0,
        "host_memory_delta_bytes": 0,
    }


def test_artifact_manifest_is_stable_and_excludes_itself(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.txt").write_text("b\n", encoding="utf-8")
    output = tmp_path / "artifact_manifest.sha256"

    _write_manifest(tmp_path, output)
    first = output.read_text(encoding="utf-8")
    _write_manifest(tmp_path, output)

    assert output.read_text(encoding="utf-8") == first
    assert "artifact_manifest.sha256" not in first
    assert "  a.txt" in first
    assert "  nested/b.txt" in first
