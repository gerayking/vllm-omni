#!/usr/bin/env python3
"""Build a deterministic four-workload public proxy benchmark sample set."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], Hashable]) -> dict[str, int]:
    return dict(sorted(Counter(str(key(row)) for row in rows).items()))


def select_stratified(
    rows: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    stratum_key: Callable[[dict[str, Any]], Hashable],
    unique_key: Callable[[dict[str, Any]], Hashable],
) -> list[dict[str, Any]]:
    """Round-robin over shuffled strata while preferring unique media inputs."""
    if count < 1:
        raise ValueError("sample count must be positive")
    rng = random.Random(seed)
    groups: dict[Hashable, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[stratum_key(row)].append(row)
    group_keys = sorted(groups, key=str)
    rng.shuffle(group_keys)
    for values in groups.values():
        rng.shuffle(values)

    selected: list[dict[str, Any]] = []
    seen: set[Hashable] = set()
    while len(selected) < count:
        progressed = False
        for group_key in group_keys:
            values = groups[group_key]
            while values:
                candidate = values.pop()
                identity = unique_key(candidate)
                if identity in seen:
                    continue
                selected.append(candidate)
                seen.add(identity)
                progressed = True
                break
            if len(selected) == count:
                break
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError(f"requested {count} unique samples, but only selected {len(selected)}")
    return selected


def select_hierarchical(
    rows: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    primary_key: Callable[[dict[str, Any]], Hashable],
    secondary_key: Callable[[dict[str, Any]], Hashable],
    unique_key: Callable[[dict[str, Any]], Hashable],
) -> list[dict[str, Any]]:
    """Balance primary groups first, then round-robin secondary strata."""
    primary_groups: dict[Hashable, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        primary_groups[primary_key(row)].append(row)
    keys = sorted(primary_groups, key=str)
    base, remainder = divmod(count, len(keys))
    selected_groups: dict[Hashable, list[dict[str, Any]]] = {}
    for index, key in enumerate(keys):
        group_count = base + int(index < remainder)
        selected_groups[key] = select_stratified(
            primary_groups[key],
            count=group_count,
            seed=seed + index,
            stratum_key=secondary_key,
            unique_key=unique_key,
        )

    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        for key in keys:
            if selected_groups[key]:
                selected.append(selected_groups[key].pop(0))
    return selected


def _load_seed_rows(root: Path, locale: str) -> list[dict[str, Any]]:
    meta = root / locale / "meta.lst"
    rows = []
    for index, line in enumerate(meta.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        parts = line.split("|", 3)
        if len(parts) != 4:
            raise ValueError(f"invalid Seed-TTS metadata at {meta}:{index + 1}")
        utterance_id, prompt_text, prompt_wav, target_text = parts
        prompt_path = root / locale / prompt_wav
        if not prompt_path.is_file():
            continue
        rows.append(
            {
                "utterance_id": utterance_id,
                "prompt_text": prompt_text,
                "prompt_wav": prompt_wav,
                "target_text": target_text,
                "target_length": len(target_text),
                "source_line": line,
            }
        )
    return rows


def _length_bins(rows: list[dict[str, Any]]) -> tuple[list[int], Callable[[dict[str, Any]], int]]:
    lengths = sorted(int(row["target_length"]) for row in rows)
    thresholds = [lengths[len(lengths) * numerator // 4] for numerator in (1, 2, 3)]

    def bin_for(row: dict[str, Any]) -> int:
        return bisect.bisect_right(thresholds, int(row["target_length"]))

    return thresholds, bin_for


def _ensure_prompt_link(output_root: Path, source_root: Path, locale: str) -> None:
    locale_output = output_root / "seed_tts" / locale
    locale_output.mkdir(parents=True, exist_ok=True)
    link = locale_output / "prompt-wavs"
    target = (source_root / locale / "prompt-wavs").resolve()
    if link.is_symlink():
        if link.resolve() != target:
            raise ValueError(f"existing prompt link points elsewhere: {link}")
        return
    if link.exists():
        raise ValueError(f"prompt link path already exists: {link}")
    link.symlink_to(target, target_is_directory=True)


def _daily_video_path(root: Path, video_id: str) -> Path | None:
    candidates = [root / video_id / f"{video_id}_video.mp4", root / f"{video_id}.mp4"]
    return next((path for path in candidates if path.is_file()), None)


def build(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    daily_rows = json.loads(args.daily_qa.read_text(encoding="utf-8"))
    daily_available = [
        row
        for row in daily_rows
        if _daily_video_path(args.daily_video_dir, str(row.get("video_id", ""))) is not None
    ]
    daily_selected = select_hierarchical(
        daily_available,
        count=args.daily_count,
        seed=args.seed,
        primary_key=lambda row: row.get("video_duration"),
        secondary_key=lambda row: row.get("content_parent_category"),
        unique_key=lambda row: str(row.get("video_id")),
    )
    daily_output = args.output_dir / "daily_omni" / "qa.json"
    daily_output.parent.mkdir(parents=True, exist_ok=True)
    daily_output.write_text(json.dumps(daily_selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    seed_manifest: dict[str, Any] = {}
    for locale, count, offset in (
        ("en", args.seed_en_count, 1000),
        ("zh", args.seed_zh_count, 2000),
    ):
        rows = _load_seed_rows(args.seed_tts_root, locale)
        thresholds, length_bin = _length_bins(rows)
        selected = select_stratified(
            rows,
            count=count,
            seed=args.seed + offset,
            stratum_key=length_bin,
            unique_key=lambda row: str(row["prompt_wav"]),
        )
        _ensure_prompt_link(args.output_dir, args.seed_tts_root, locale)
        meta_output = args.output_dir / "seed_tts" / locale / "meta.lst"
        meta_output.write_text("\n".join(str(row["source_line"]) for row in selected) + "\n", encoding="utf-8")
        seed_manifest[locale] = {
            "source_rows": len(rows),
            "selected_rows": len(selected),
            "selection": "target-text-length quartiles; unique reference prompt preferred",
            "length_thresholds": thresholds,
            "length_bin_distribution": _distribution(selected, length_bin),
            "target_length_min": min(int(row["target_length"]) for row in selected),
            "target_length_max": max(int(row["target_length"]) for row in selected),
            "utterance_ids": [row["utterance_id"] for row in selected],
            "metadata": str(meta_output),
            "metadata_sha256": _sha256(meta_output),
        }

    import pyarrow.parquet as pq

    video_rows = pq.read_table(args.video_mme_metadata).to_pylist()
    video_files = {path.stem for path in args.video_mme_video_dir.glob("*.mp4")}
    video_available = [
        row
        for row in video_rows
        if str(row.get("videoID", "")) in video_files or str(row.get("video_id", "")) in video_files
    ]
    video_selected = select_hierarchical(
        video_available,
        count=args.video_mme_count,
        seed=args.seed + 3000,
        primary_key=lambda row: row.get("duration"),
        secondary_key=lambda row: row.get("domain"),
        unique_key=lambda row: str(row.get("videoID") or row.get("video_id")),
    )
    video_output = args.output_dir / "video_mme" / "metadata.json"
    video_output.parent.mkdir(parents=True, exist_ok=True)
    video_output.write_text(json.dumps(video_selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "scope": "deterministic public-data proxy benchmark; not an official competition benchmark",
        "formal_competition_score": None,
        "seed": args.seed,
        "total_selected_samples": len(daily_selected) + args.seed_en_count + args.seed_zh_count + len(video_selected),
        "daily_omni": {
            "source_rows": len(daily_rows),
            "available_rows": len(daily_available),
            "selected_rows": len(daily_selected),
            "selection": "equal duration allocation, then parent-category strata; one question per video",
            "duration_distribution": _distribution(daily_selected, lambda row: row.get("video_duration")),
            "category_distribution": _distribution(daily_selected, lambda row: row.get("content_parent_category")),
            "video_ids": [row["video_id"] for row in daily_selected],
            "metadata": str(daily_output),
            "metadata_sha256": _sha256(daily_output),
        },
        "seed_tts": seed_manifest,
        "video_mme": {
            "source_rows": len(video_rows),
            "available_rows": len(video_available),
            "available_videos": len(video_files),
            "selected_rows": len(video_selected),
            "selection": "equal duration allocation, then domain strata; one question per video",
            "duration_distribution": _distribution(video_selected, lambda row: row.get("duration")),
            "domain_distribution": _distribution(video_selected, lambda row: row.get("domain")),
            "question_ids": [row["question_id"] for row in video_selected],
            "metadata": str(video_output),
            "metadata_sha256": _sha256(video_output),
        },
    }
    manifest_output = args.output_dir / "manifest.json"
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(manifest_output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-qa", type=Path, required=True)
    parser.add_argument("--daily-video-dir", type=Path, required=True)
    parser.add_argument("--seed-tts-root", type=Path, required=True)
    parser.add_argument("--video-mme-metadata", type=Path, required=True)
    parser.add_argument("--video-mme-video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--daily-count", type=int, default=100)
    parser.add_argument("--seed-en-count", type=int, default=100)
    parser.add_argument("--seed-zh-count", type=int, default=100)
    parser.add_argument("--video-mme-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
