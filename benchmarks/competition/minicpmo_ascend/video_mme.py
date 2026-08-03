#!/usr/bin/env python3
"""Evaluate an OpenAI-compatible MiniCPM-o service on public Video-MME."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .client import build_payload, run_stream_request


def extract_choice(text: str | None) -> str | None:
    if not text:
        return None
    candidate = str(text).strip()
    direct = re.match(r"(?i)^\s*(?:answer\s*[:：]?\s*)?\(?([A-D])\)?(?:[\s.\):：]|$)", candidate)
    if direct:
        return direct.group(1).upper()
    fallback = re.search(r"(?i)(?:answer|option|choice)\s*(?:is|:)\s*\(?([A-D])\)?", candidate)
    return fallback.group(1).upper() if fallback else None


def _flatten_official_json(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for video in data:
        for question in video.get("questions", []):
            rows.append({**video, **question})
    return rows


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Video-MME metadata must contain a JSON list")
    if data and "questions" in data[0]:
        return _flatten_official_json(data)
    return data


def index_videos(root: Path) -> dict[str, Path]:
    videos: dict[str, Path] = {}
    for path in sorted(root.rglob("*.mp4")):
        videos.setdefault(path.stem, path.resolve())
        videos.setdefault(path.name, path.resolve())
    return videos


def find_video(row: dict[str, Any], videos: dict[str, Path]) -> Path | None:
    for key in ("video_id", "videoID", "videoId"):
        value = str(row.get(key, "")).strip()
        if value in videos:
            return videos[value]
        if f"{value}.mp4" in videos:
            return videos[f"{value}.mp4"]
    return None


def build_prompt(row: dict[str, Any]) -> str:
    options = row.get("options") or row.get("choice") or row.get("Choice") or []
    if isinstance(options, str):
        options = [part.strip() for part in options.splitlines() if part.strip()]
    lines = [
        "Select the best answer to the following multiple-choice question based on the video.",
        "Respond with only the letter (A, B, C, or D) of the correct option.",
        str(row.get("question") or row.get("Question") or "").strip(),
        *(str(option).strip() for option in options),
        "The best answer is:",
    ]
    return "\n".join(line for line in lines if line)


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    def bucket(field: str) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            groups[str(item.get(field) or "unknown")].append(item)
        return {
            key: {
                "correct": sum(bool(item["correct"]) for item in values),
                "total": len(values),
                "accuracy": sum(bool(item["correct"]) for item in values) / len(values),
            }
            for key, values in sorted(groups.items())
        }

    total = len(items)
    correct = sum(bool(item["correct"]) for item in items)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
        "request_failures": sum(not item["success"] for item in items),
        "unparsed_responses": sum(item["predicted"] is None for item in items),
        "by_duration": bucket("duration"),
        "by_domain": bucket("domain"),
        "by_sub_category": bucket("sub_category"),
        "by_task_type": bucket("task_type"),
    }


def build_official_results(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in items:
        video_id = str(item["video_id"])
        video = grouped.setdefault(
            video_id,
            {
                "video_id": video_id,
                "duration": item.get("duration"),
                "domain": item.get("domain"),
                "sub_category": item.get("sub_category"),
                "questions": [],
            },
        )
        video["questions"].append(
            {
                "question_id": item.get("question_id"),
                "task_type": item.get("task_type"),
                "question": item.get("question"),
                "options": item.get("options"),
                "answer": item.get("gold"),
                "response": item.get("response", ""),
            }
        )
    return list(grouped.values())


async def _main(args: argparse.Namespace) -> int:
    rows = load_rows(args.metadata)
    if args.durations:
        allowed = set(args.durations)
        rows = [row for row in rows if row.get("duration") in allowed]
    videos = index_videos(args.video_dir)
    if not videos:
        raise SystemExit(f"no MP4 files found under {args.video_dir}")
    if args.skip_missing_videos:
        rows = [row for row in rows if find_video(row, videos) is not None]
    if args.max_videos is not None:
        selected: set[str] = set()
        kept = []
        for row in rows:
            video_id = str(row.get("video_id"))
            if video_id not in selected and len(selected) >= args.max_videos:
                continue
            selected.add(video_id)
            kept.append(row)
        rows = kept
    if args.num_questions is not None:
        rows = rows[: args.num_questions]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(args.timeout)

    async with httpx.AsyncClient(timeout=timeout, headers={"Authorization": "Bearer EMPTY"}) as client:

        async def one(index: int, row: dict[str, Any]) -> dict[str, Any]:
            video = find_video(row, videos)
            base = {
                "index": index,
                "video_id": str(row.get("video_id", "")),
                "question_id": row.get("question_id"),
                "duration": row.get("duration"),
                "domain": row.get("domain"),
                "sub_category": row.get("sub_category"),
                "task_type": row.get("task_type"),
                "question": row.get("question"),
                "options": row.get("options"),
                "gold": str(row.get("answer", "")).strip().upper(),
            }
            if video is None:
                return {
                    **base,
                    "success": False,
                    "response": "",
                    "predicted": None,
                    "correct": False,
                    "errors": ["video file not found"],
                }
            payload = build_payload(
                model=args.model,
                prompt=build_prompt(row),
                input_modality="video",
                media=video.as_uri(),
                with_audio=False,
                seed=args.seed,
                thinker_max_tokens=args.max_tokens,
                talker_max_tokens=args.max_tokens,
            )
            payload["chat_template_kwargs"]["enable_thinking"] = False
            async with semaphore:
                record = await run_stream_request(
                    client,
                    endpoint=f"{args.base_url.rstrip('/')}/chat/completions",
                    payload=payload,
                    request_name=str(row.get("question_id") or index),
                    input_modality="video",
                    with_audio=False,
                )
            response = record.get("text", "")
            predicted = extract_choice(response)
            item = {
                **base,
                **record,
                "video_path": str(video),
                "response": response,
                "predicted": predicted,
                "correct": predicted == base["gold"],
            }
            print(
                f"[{index + 1}/{len(rows)}] {base['question_id']}: "
                f"{'OK' if item['success'] else 'FAIL'} pred={predicted} gold={base['gold']}",
                flush=True,
            )
            return item

        items = await asyncio.gather(*(one(index, row) for index, row in enumerate(rows)))

    result = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "scope": "public Video-MME, subtitles-free",
        "formal_competition_score": None,
        "command": [sys.executable, *sys.argv],
        "summary": summarize(items),
        "items": items,
    }
    (args.output_dir / "video_mme_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "official_results.json").write_text(
        json.dumps(build_official_results(items), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 1 if result["summary"]["request_failures"] else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:8099/v1")
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument("--durations", nargs="+", choices=["short", "medium", "long"])
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--num-questions", type=int)
    parser.add_argument("--skip-missing-videos", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    raise SystemExit(asyncio.run(_main(parser.parse_args())))


if __name__ == "__main__":
    main()
