"""Assemble LLM input by aligning ASR segments and PPT OCR text in 10-min buckets.

Two flavors:
  - assemble_bucketed(): for new lectures (segments captured during ASR, in-memory)
  - assemble_flat():     for old lectures (only joined transcript text)

Important: transcript segments are NEVER persisted. They are produced by the
transcriber, used here at prompt-build time, and discarded. The DB stores only
the joined transcript string. This avoids doubling text storage.
"""

from __future__ import annotations

import json
from collections import defaultdict

from src.ai.ppt_dedup import clean_ppt_text, dedup_text_subset

BUCKET_SIZE_SEC = 600  # 10 minutes


def _format_timestamp(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# Synthetic board rows are stored with page_num >= BOARD_PAGE_NUM_BASE (10000)
# because the blackboard is sampled from the video, not a slide deck.  They
# must not be labelled as PPT pages: "页 10000" is meaningless to the model and
# invites it to treat the number as a slide index.
_BOARD_PAGE_NUM_BASE = 10000


def _page_label(page_num, ts: str) -> str:
    """Human label for one page row — board frames vs real slides."""
    try:
        num = int(page_num)
    except (TypeError, ValueError):
        num = 0
    if num >= _BOARD_PAGE_NUM_BASE:
        return f"[板书 @ {ts}]"
    return f"[页 {num} @ {ts}]" if num else f"[@ {ts}]"



def assemble_bucketed(
    transcript_segments: list[dict], ppt_pages: list[dict],
) -> str:
    asr_by_bucket = defaultdict(list)
    for seg in transcript_segments or []:
        bucket = int(seg.get("start_ms", 0)) // 1000 // BUCKET_SIZE_SEC
        text = (seg.get("text") or "").strip()
        if text:
            asr_by_bucket[bucket].append(text)

    ppt_by_bucket = defaultdict(list)
    for page in ppt_pages or []:
        bucket = int(page.get("created_sec", 0)) // BUCKET_SIZE_SEC
        ppt_by_bucket[bucket].append(page)

    all_buckets = sorted(set(asr_by_bucket.keys()) | set(ppt_by_bucket.keys()))
    if not all_buckets:
        return ""

    out = []
    for b in all_buckets:
        start = b * BUCKET_SIZE_SEC
        end = (b + 1) * BUCKET_SIZE_SEC
        out.append(
            f"\n=== 时间段 {_format_timestamp(start)} – "
            f"{_format_timestamp(end)} ===\n"
        )

        asr_text = " ".join(asr_by_bucket.get(b, []))
        if asr_text:
            out.append("【音频转录】")
            out.append(asr_text)
            out.append("")

        pages = ppt_by_bucket.get(b, [])
        if pages:
            out.append("【课件/板书 文字识别】")
            for p in pages:
                ts = _format_timestamp(int(p["created_sec"]))
                tag = _page_label(p.get("page_num", ""), ts)
                text = clean_ppt_text(p.get("text") or "").strip()
                if text:
                    out.append(f"{tag}\n{text}")
            out.append("")

    return "\n".join(out).strip()


def assemble_flat(transcript: str, ppt_pages: list[dict]) -> str:
    parts = []
    if transcript and transcript.strip():
        parts.append("【音频转录（无时间轴）】")
        parts.append(transcript.strip())
        parts.append("")
    if ppt_pages:
        parts.append("【课件/板书 文字识别（按出现顺序）】")
        sorted_pages = sorted(
            ppt_pages, key=lambda p: int(p.get("created_sec", 0))
        )
        for p in sorted_pages:
            ts = _format_timestamp(int(p.get("created_sec", 0)))
            tag = _page_label(p.get("page_num", ""), ts)
            text = (p.get("text") or "").strip()
            if text:
                parts.append(f"{tag}\n{text}")
        parts.append("")
    return "\n".join(parts).strip()


def assemble(
    transcript: str,
    transcript_segments: list[dict] | None,
    ppt_pages: list[dict] | None,
) -> tuple[str, str]:
    """Single entry point used by main.py.

    Args:
        transcript: joined transcript string (always available; persisted in DB).
        transcript_segments: in-memory list of {start_ms, end_ms, text}, or None
            for re-summarized old lectures.
        ppt_pages: list of {created_sec, page_num, text}; usually loaded from
            DB's ppt_ocr JSON column. Can be empty for transition cases.

    Returns:
        (assembled_text, mode) where mode is 'bucketed' or 'flat'.
    """
    pages = dedup_text_subset(ppt_pages or [])
    if transcript_segments:
        return assemble_bucketed(transcript_segments, pages), "bucketed"
    return assemble_flat(transcript or "", pages), "flat"

