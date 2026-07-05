#!/usr/bin/env python3
"""CosyVoice3 PR0 benchmark, quality, and profile harness.

This script intentionally avoids starting services implicitly. It provides
repeatable command builders and small local analyzers so each CosyVoice3
optimization PR can produce comparable performance and quality evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


DEFAULT_RESULT_ROOT = Path("benchmark_results/cosyvoice3_pr0")
DEFAULT_DATASET_PATH = Path("benchmark_results/seeden_official_streaming_c4/testset_seed0_n100")
DEFAULT_PERCENTILE_METRICS = "ttft,e2el,audio_rtf,audio_ttfp,audio_duration,audio_underrun"


def build_seed_tts_bench_command(
    *,
    host: str,
    port: int,
    model: str,
    tokenizer: str | None,
    dataset_path: str | Path,
    result_dir: str | Path,
    result_filename: str,
    concurrency: int,
    num_prompts: int,
    quality_eval: bool = False,
    save_quality_items: bool = False,
    num_warmups: int = 0,
    hf_output_len: int = 256,
    ready_check_timeout_sec: int | None = None,
    vllm_bin: str = "vllm",
    locale: str = "en",
) -> list[str]:
    """Build the canonical fixed SeedEN ``vllm bench serve --omni`` command."""
    cmd = [
        vllm_bin,
        "bench",
        "serve",
        "--omni",
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        str(model),
    ]
    if tokenizer:
        cmd.extend(["--tokenizer", str(tokenizer)])
    cmd.extend(
        [
            "--backend",
            "openai-audio-speech",
            "--endpoint",
            "/v1/audio/speech",
            "--dataset-name",
            "seed-tts",
            "--dataset-path",
            str(dataset_path),
            "--seed-tts-locale",
            locale,
            "--seed-tts-inline-ref-audio",
            "--disable-shuffle",
            "--num-prompts",
            str(num_prompts),
            "--num-warmups",
            str(num_warmups),
            "--max-concurrency",
            str(concurrency),
            "--request-rate",
            "inf",
            "--percentile-metrics",
            DEFAULT_PERCENTILE_METRICS,
            "--hf-output-len",
            str(hf_output_len),
            "--save-result",
            "--result-dir",
            str(result_dir),
            "--result-filename",
            result_filename,
        ]
    )
    if ready_check_timeout_sec is not None:
        cmd.extend(["--ready-check-timeout-sec", str(int(ready_check_timeout_sec))])
    if quality_eval:
        cmd.append("--seed-tts-wer-eval")
    if save_quality_items:
        cmd.append("--seed-tts-wer-save-items")
    return cmd


def _metric_block(data: dict[str, Any], prefix: str, *, unit_suffix: str = "") -> dict[str, float | None]:
    mean_key = f"mean_{prefix}{unit_suffix}"
    median_key = f"median_{prefix}{unit_suffix}"
    p90_key = f"p90_{prefix}{unit_suffix}"
    p99_key = f"p99_{prefix}{unit_suffix}"
    return {
        "mean": _maybe_float(data.get(mean_key)),
        "p50": _maybe_float(data.get(median_key)),
        "p90": _maybe_float(data.get(p90_key)),
        "p99": _maybe_float(data.get(p99_key)),
    }


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def summarize_benchmark_result(data: dict[str, Any], *, label: str, source: Path | None = None) -> dict[str, Any]:
    """Normalize vLLM benchmark JSON into a stable PR0 summary schema."""
    return {
        "label": label,
        "source": str(source) if source is not None else None,
        "completed": int(data.get("completed", 0) or 0),
        "failed": int(data.get("failed", 0) or 0),
        "duration_s": _maybe_float(data.get("duration")),
        "request_throughput": _maybe_float(data.get("request_throughput")),
        "audio_throughput": _maybe_float(data.get("audio_throughput")),
        "total_audio_duration_s": _maybe_float(data.get("total_audio_duration_s")),
        "mean_audio_duration_s": _maybe_float(data.get("mean_audio_duration_s")),
        "e2el_ms": _metric_block(data, "e2el", unit_suffix="_ms"),
        "audio_ttfp_ms": _metric_block(data, "audio_ttfp", unit_suffix="_ms"),
        "audio_rtf": _metric_block(data, "audio_rtf"),
        "audio_underrun_s": _metric_block(data, "audio_underrun", unit_suffix="_s"),
        "quality": {
            "wer_or_cer_mean": _maybe_float(data.get("seed_tts_content_error_mean")),
            "speaker_similarity_mean": _maybe_float(data.get("seed_tts_sim_mean")),
            "utmos_mean": _maybe_float(data.get("seed_tts_utmos_mean")),
            "evaluated": int(data.get("seed_tts_content_evaluated", 0) or 0),
            "setup_error": data.get("seed_tts_eval_setup_error"),
        },
    }


def compute_audio_sanity(
    samples: np.ndarray,
    *,
    sample_rate: int,
    clip_threshold: float = 0.999,
    silence_rms_threshold: float = 1e-4,
) -> dict[str, Any]:
    """Compute finite, duration, silence, and clipping checks for generated audio."""
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim > 1:
        arr = np.mean(arr, axis=1)
    finite = bool(np.isfinite(arr).all())
    num_samples = int(arr.size)
    duration_s = float(num_samples / sample_rate) if sample_rate > 0 else 0.0
    if num_samples == 0 or not finite:
        rms = 0.0
        peak_abs = 0.0
        clipped = 0
    else:
        rms = float(np.sqrt(np.mean(np.square(arr, dtype=np.float64))))
        peak_abs = float(np.max(np.abs(arr)))
        clipped = int(np.count_nonzero(np.abs(arr) >= float(clip_threshold)))
    return {
        "sample_rate": int(sample_rate),
        "num_samples": num_samples,
        "duration_s": duration_s,
        "finite": finite,
        "rms": rms,
        "peak_abs": peak_abs,
        "non_silent": bool(rms >= float(silence_rms_threshold)),
        "clipped_samples": clipped,
        "clipped_ratio": float(clipped / num_samples) if num_samples else 0.0,
    }


def compare_mel_arrays(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    mean_abs_threshold: float,
    max_abs_threshold: float,
) -> dict[str, Any]:
    """Compare deterministic mel tensors captured from baseline and candidate runs."""
    ref = np.asarray(reference, dtype=np.float32)
    cand = np.asarray(candidate, dtype=np.float32)
    shape_match = ref.shape == cand.shape
    finite = bool(np.isfinite(ref).all() and np.isfinite(cand).all())
    if not shape_match or not finite:
        mean_abs = None
        max_abs = None
    else:
        diff = np.abs(ref - cand)
        mean_abs = float(np.mean(diff))
        max_abs = float(np.max(diff)) if diff.size else 0.0
    passed = (
        shape_match
        and finite
        and mean_abs is not None
        and max_abs is not None
        and mean_abs <= float(mean_abs_threshold)
        and max_abs <= float(max_abs_threshold)
    )
    return {
        "shape_match": shape_match,
        "reference_shape": list(ref.shape),
        "candidate_shape": list(cand.shape),
        "finite": finite,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "mean_abs_threshold": float(mean_abs_threshold),
        "max_abs_threshold": float(max_abs_threshold),
        "passed": bool(passed),
    }


def make_profile_deploy(
    *,
    input_yaml: Path,
    output_yaml: Path,
    torch_profiler_dir: Path,
    stages: set[int],
) -> None:
    """Write a deploy YAML with torch profiler enabled for selected stages."""
    data = yaml.safe_load(input_yaml.read_text(encoding="utf-8"))
    for stage in data.get("stages", []):
        stage_id = int(stage.get("stage_id", -1))
        if stage_id not in stages:
            continue
        stage["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(torch_profiler_dir),
            "torch_profiler_record_shapes": False,
            "torch_profiler_with_stack": False,
            "torch_profiler_with_memory": False,
            "torch_profiler_dump_cuda_time_total": True,
        }
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def render_profile_commands(
    *,
    run_dir: Path,
    model: str,
    tokenizer: str | None,
    dataset_path: str | Path,
    host: str,
    port: int,
    concurrency: int = 4,
    num_prompts: int = 20,
) -> str:
    """Render manual profile commands with explicit stage start/stop controls."""
    tokenizer_line = f'  --tokenizer "{tokenizer}" \\\n' if tokenizer else ""
    return f"""#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="{run_dir}"
MODEL="{model}"
DATASET="{dataset_path}"
HOST="{host}"
PORT="{port}"

mkdir -p "$RUN_DIR/bench"

curl -sf "http://$HOST:$PORT/health"
curl -sS -X POST "http://$HOST:$PORT/start_profile" \\
  -H 'Content-Type: application/json' \\
  -d '{{"stages":[0,1]}}'

vllm bench serve --omni \\
  --host "$HOST" --port "$PORT" \\
  --model "$MODEL" \\
{tokenizer_line}  --backend openai-audio-speech \\
  --endpoint /v1/audio/speech \\
  --dataset-name seed-tts \\
  --dataset-path "$DATASET" \\
  --seed-tts-locale en \\
  --seed-tts-inline-ref-audio \\
  --disable-shuffle \\
  --num-prompts {num_prompts} --num-warmups 0 \\
  --max-concurrency {concurrency} --request-rate inf \\
  --percentile-metrics {DEFAULT_PERCENTILE_METRICS} \\
  --hf-output-len 256 \\
  --save-result --result-dir "$RUN_DIR/bench" \\
  --result-filename "profile_c{concurrency}_n{num_prompts}.json"

curl -sS -X POST "http://$HOST:$PORT/stop_profile" \\
  -H 'Content-Type: application/json' \\
  -d '{{"stages":[0,1]}}'
"""


def render_markdown_report(
    *,
    run_id: str,
    model: str,
    dataset_path: str,
    benchmark_summaries: list[dict[str, Any]],
    audio_sanity: list[dict[str, Any]],
    mel_diff: dict[str, Any] | None,
    profile_summary: dict[str, Any] | None,
    artifact_root: Path,
) -> str:
    """Render a PR-description-ready report for PR0 and future optimization PRs."""
    lines = [
        f"# CosyVoice3 PR0 Benchmark and Quality Report: {run_id}",
        "",
        "## Purpose",
        "",
        "Establish repeatable CosyVoice3 flow-matching benchmark, quality, and profile gates.",
        "",
        "## Test Plan",
        "",
        f"- Model: `{model}`",
        f"- Dataset: `{dataset_path}`",
        f"- Artifact root: `{artifact_root}`",
        "- Fixed SeedEN ordering with `--disable-shuffle`",
        "- Required performance runs: c4 and c8, 100 prompts, request-rate `inf`",
        "- Required quality gates: mel diff, audio sanity, WER/CER, speaker similarity",
        "",
        "## Performance Result",
        "",
        "| Label | Completed | Failed | Req/s | Audio s/s | E2EL mean/p50/p90/p99 ms | TTFP mean/p50/p90/p99 ms | RTF mean/p50/p90/p99 | Underrun mean/p50/p90/p99 s |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for item in benchmark_summaries:
        lines.append(
            "| {label} | {completed} | {failed} | {req} | {audio} | {e2e} | {ttfp} | {rtf} | {underrun} |".format(
                label=item.get("label"),
                completed=item.get("completed"),
                failed=item.get("failed"),
                req=_fmt(item.get("request_throughput")),
                audio=_fmt(item.get("audio_throughput")),
                e2e=_fmt_block(item.get("e2el_ms", {})),
                ttfp=_fmt_block(item.get("audio_ttfp_ms", {})),
                rtf=_fmt_block(item.get("audio_rtf", {})),
                underrun=_fmt_block(item.get("audio_underrun_s", {})),
            )
        )
    lines.extend(["", "## Precision / Quality Result", ""])
    if benchmark_summaries:
        lines.extend(
            [
                "| Label | WER/CER mean | Speaker similarity mean | UTMOS mean | Quality setup error |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for item in benchmark_summaries:
            q = item.get("quality", {})
            lines.append(
                f"| {item.get('label')} | {_fmt(q.get('wer_or_cer_mean'))} | "
                f"{_fmt(q.get('speaker_similarity_mean'))} | {_fmt(q.get('utmos_mean'))} | "
                f"{q.get('setup_error') or ''} |"
            )
    if audio_sanity:
        lines.extend(["", "### Audio Sanity", "", "```json", json.dumps(audio_sanity, indent=2), "```"])
    if mel_diff is not None:
        lines.extend(["", "### Mel Diff", "", "```json", json.dumps(mel_diff, indent=2), "```"])
    if profile_summary is not None:
        lines.extend(["", "## Profile Result", "", "```json", json.dumps(profile_summary, indent=2), "```"])
    lines.extend(["", "## Artifact Index", ""])
    for item in benchmark_summaries:
        if item.get("source"):
            lines.append(f"- `{item['source']}`")
    lines.append(f"- `{artifact_root}`")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    num = _maybe_float(value)
    return "n/a" if num is None else f"{num:.3f}"


def _fmt_block(block: dict[str, Any]) -> str:
    return "/".join(_fmt(block.get(key)) for key in ("mean", "p50", "p90", "p99"))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        data = np.load(path)
        key = "mel" if "mel" in data else data.files[0]
        return np.asarray(data[key])
    if suffix == ".json":
        return np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=np.float32)
    raise ValueError(f"Unsupported array format: {path}")


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def _cmd_bench(args: argparse.Namespace) -> int:
    result_dir = Path(args.result_dir)
    for concurrency in args.concurrency:
        filename = args.result_filename or f"seeden_c{concurrency}_n{args.num_prompts}.json"
        cmd = build_seed_tts_bench_command(
            host=args.host,
            port=args.port,
            model=args.model,
            tokenizer=args.tokenizer,
            dataset_path=args.dataset_path,
            result_dir=result_dir / f"c{concurrency}",
            result_filename=filename,
            concurrency=concurrency,
            num_prompts=args.num_prompts,
            quality_eval=args.quality_eval,
            save_quality_items=args.save_quality_items,
            num_warmups=args.num_warmups,
            hf_output_len=args.hf_output_len,
            ready_check_timeout_sec=args.ready_check_timeout_sec,
            vllm_bin=args.vllm_bin,
        )
        _run(cmd, dry_run=args.dry_run)
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    summaries = [
        summarize_benchmark_result(_load_json(path), label=path.stem, source=path) for path in args.benchmark_json
    ]
    mel_diff = _load_json(args.mel_diff_json) if args.mel_diff_json else None
    profile_summary = _load_json(args.profile_summary_json) if args.profile_summary_json else None
    report = render_markdown_report(
        run_id=args.run_id,
        model=args.model,
        dataset_path=args.dataset_path,
        benchmark_summaries=summaries,
        audio_sanity=[],
        mel_diff=mel_diff,
        profile_summary=profile_summary,
        artifact_root=args.artifact_root,
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")
    _write_json(args.output_json, {"benchmarks": summaries, "mel_diff": mel_diff, "profile": profile_summary})
    return 0


def _cmd_audio_sanity(args: argparse.Namespace) -> int:
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - dependency may be optional in minimal envs
        raise SystemExit("soundfile is required for audio-sanity") from exc
    results = []
    for wav in args.wav:
        samples, sample_rate = sf.read(wav, always_2d=False)
        row = compute_audio_sanity(
            samples,
            sample_rate=int(sample_rate),
            clip_threshold=args.clip_threshold,
            silence_rms_threshold=args.silence_rms_threshold,
        )
        row["path"] = str(wav)
        results.append(row)
    _write_json(args.output_json, results)
    return 0


def _cmd_mel_compare(args: argparse.Namespace) -> int:
    result = compare_mel_arrays(
        _load_array(args.reference),
        _load_array(args.candidate),
        mean_abs_threshold=args.mean_abs_threshold,
        max_abs_threshold=args.max_abs_threshold,
    )
    _write_json(args.output_json, result)
    return 0 if result["passed"] or args.no_fail else 1


def _cmd_profile_deploy(args: argparse.Namespace) -> int:
    make_profile_deploy(
        input_yaml=args.input,
        output_yaml=args.output,
        torch_profiler_dir=args.torch_profiler_dir,
        stages=set(args.stages),
    )
    return 0


def _cmd_profile_commands(args: argparse.Namespace) -> int:
    script = render_profile_commands(
        run_dir=args.run_dir,
        model=args.model,
        tokenizer=args.tokenizer,
        dataset_path=args.dataset_path,
        host=args.host,
        port=args.port,
        concurrency=args.concurrency,
        num_prompts=args.num_prompts,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(script, encoding="utf-8")
    args.output.chmod(0o755)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("bench", help="Run or print fixed SeedEN c4/c8 benchmark commands")
    bench.add_argument("--host", default="127.0.0.1")
    bench.add_argument("--port", type=int, default=8091)
    bench.add_argument("--model", required=True)
    bench.add_argument("--tokenizer")
    bench.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    bench.add_argument("--result-dir", default=str(DEFAULT_RESULT_ROOT / "bench"))
    bench.add_argument("--result-filename")
    bench.add_argument("--concurrency", type=int, nargs="+", default=[4, 8])
    bench.add_argument("--num-prompts", type=int, default=100)
    bench.add_argument("--num-warmups", type=int, default=0)
    bench.add_argument("--hf-output-len", type=int, default=256)
    bench.add_argument("--quality-eval", action="store_true")
    bench.add_argument("--save-quality-items", action="store_true")
    bench.add_argument("--ready-check-timeout-sec", type=int)
    bench.add_argument("--vllm-bin", default="vllm")
    bench.add_argument("--dry-run", action="store_true")
    bench.set_defaults(func=_cmd_bench)

    summarize = sub.add_parser("summarize", help="Summarize benchmark JSONs and render a PR report")
    summarize.add_argument("--run-id", required=True)
    summarize.add_argument("--model", required=True)
    summarize.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    summarize.add_argument("--artifact-root", type=Path, required=True)
    summarize.add_argument("--benchmark-json", type=Path, nargs="+", required=True)
    summarize.add_argument("--mel-diff-json", type=Path)
    summarize.add_argument("--profile-summary-json", type=Path)
    summarize.add_argument("--output-json", type=Path, required=True)
    summarize.add_argument("--output-md", type=Path, required=True)
    summarize.set_defaults(func=_cmd_summarize)

    audio = sub.add_parser("audio-sanity", help="Validate generated WAV files")
    audio.add_argument("--wav", type=Path, nargs="+", required=True)
    audio.add_argument("--output-json", type=Path, required=True)
    audio.add_argument("--clip-threshold", type=float, default=0.999)
    audio.add_argument("--silence-rms-threshold", type=float, default=1e-4)
    audio.set_defaults(func=_cmd_audio_sanity)

    mel = sub.add_parser("mel-compare", help="Compare captured baseline/candidate mel arrays")
    mel.add_argument("--reference", type=Path, required=True)
    mel.add_argument("--candidate", type=Path, required=True)
    mel.add_argument("--output-json", type=Path, required=True)
    mel.add_argument("--mean-abs-threshold", type=float, required=True)
    mel.add_argument("--max-abs-threshold", type=float, required=True)
    mel.add_argument("--no-fail", action="store_true")
    mel.set_defaults(func=_cmd_mel_compare)

    prof = sub.add_parser("profile-deploy", help="Create a deploy YAML with torch profiler enabled")
    prof.add_argument("--input", type=Path, required=True)
    prof.add_argument("--output", type=Path, required=True)
    prof.add_argument("--torch-profiler-dir", type=Path, required=True)
    prof.add_argument("--stages", type=int, nargs="+", default=[0, 1])
    prof.set_defaults(func=_cmd_profile_deploy)

    prof_cmd = sub.add_parser("profile-commands", help="Write manual profile start/bench/stop commands")
    prof_cmd.add_argument("--run-dir", type=Path, required=True)
    prof_cmd.add_argument("--model", required=True)
    prof_cmd.add_argument("--tokenizer")
    prof_cmd.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    prof_cmd.add_argument("--host", default="127.0.0.1")
    prof_cmd.add_argument("--port", type=int, default=8091)
    prof_cmd.add_argument("--concurrency", type=int, default=4)
    prof_cmd.add_argument("--num-prompts", type=int, default=20)
    prof_cmd.add_argument("--output", type=Path, required=True)
    prof_cmd.set_defaults(func=_cmd_profile_commands)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
