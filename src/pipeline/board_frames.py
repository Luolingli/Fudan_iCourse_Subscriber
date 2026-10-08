"""Blackboard recovery from video frames — the pipeline path for
board-writing lectures whose platform screenshot feed is empty or
placeholder-only.

泛函分析 9.21/9.28 failure mode: the CI egress (WebVPN) receives a
recording whose *audio* track is dead (full duration, zero speech) while
its *video* track carries the real blackboard, and the platform PPT feed
yields only a 1-frame "operation guide" placeholder.  The blackboard is
still fully recoverable from the video frames — this module automates the
manual 664140 recovery of 2026-09-29.

Flow: stream the lecture video through ``ffmpeg`` (same reconnect ladder as
AudioDownloader; ``-vf fps=1/INTERVAL`` frame sampling) → dHash per frame →
garbage catalog + pairwise dedup (capped at ``BOARD_MAX_PAGES``) → OCR
survivors through the scheduler's OCR pool → persist as synthetic
``ppt_pages`` rows with ``page_num >= 10000`` (the platform feed enumerates
page_num from 1, so the two spaces never collide) and
``pptimgurl = 'videoframe:<sec>'``.  ``get_done_ppt_pages`` surfaces them
exactly like platform slides, so the PPT-only summarization path is
unchanged.

Idempotent across runs: pages already done/invalid are not re-OCR'd, and
a lecture whose board pages are all processed short-circuits without
re-downloading the video.  ``force_reset_lecture`` flips the rows back to
pending and the next pass re-OCR's from freshly sampled frames.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from concurrent.futures import as_completed
from typing import TYPE_CHECKING

from src.ai.ocr import ocr_image_text
from src.ai.ppt_dedup import (
    clean_ppt_text,
    compute_dhash,
    dedup_dhash,
    is_invalid_page,
    match_garbage,
)
from src.runtime import config

if TYPE_CHECKING:
    from src.api.icourse import ICourseClient
    from src.data.database import Database
    from src.runtime.reporter import Reporter
    from src.runtime.scheduler import Scheduler

# Synthetic board rows start here; platform page_nums enumerate from 1.
BOARD_PAGE_NUM_BASE = 10000

# Wall-clock cap for one ffmpeg frame-sampling pass (decode is ~3-6 min
# for a 3-hour 1080p lecture on CI; 1h is generous headroom).
_FFMPEG_TIMEOUT = 3600


def _board_rows(db: "Database", sub_id: str) -> tuple[int, int, int]:
    """(done rows, pending rows, total board rows) with page_num >= base."""
    states = db.get_board_page_states(sub_id, BOARD_PAGE_NUM_BASE)
    done = sum(1 for _pn, st in states if st == "done")
    pending = sum(1 for _pn, st in states if st == "pending")
    return done, pending, len(states)


def _sample_frames_ffmpeg(video_source: str, frame_dir: str, interval: int,
                          reporter: "Reporter | None",
                          headers: str = "") -> list[str]:
    """Stream (or read) into ffmpeg and emit 1 jpg per ``interval`` s.

    HTTP(S) sources get the same reconnect ladder as AudioDownloader —
    the WebVPN path is intermittently cut mid-stream at ~40 % of a full
    lecture, and a plain requests download of the whole mp4 dies there.
    Sampling straight through ffmpeg survives the cuts and simply yields
    fewer frames on a bad network (graceful degradation).
    """
    pattern = os.path.join(frame_dir, "f_%05d.jpg")
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if headers:
        cmd += ["-headers", headers]
    if video_source.startswith(("http://", "https://")):
        cmd += ["-reconnect", "1", "-reconnect_streamed", "1",
                "-reconnect_delay_max", "10",
                "-reconnect_on_network_error", "1",
                "-reconnect_on_http_error", "4xx,5xx"]
    cmd += ["-i", video_source, "-an",
            "-vf", f"fps=1/{interval}", "-q:v", "2", pattern]
    if reporter:
        reporter.info(
            f"    [Board] ffmpeg sampling 1 frame / {interval}s ..."
        )
    proc = subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        timeout=_FFMPEG_TIMEOUT,
    )
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="replace")[-400:]
        raise RuntimeError(
            f"ffmpeg frame sampling failed (rc={proc.returncode}): {tail}"
        )
    return sorted(
        os.path.join(frame_dir, n) for n in os.listdir(frame_dir)
        if n.endswith(".jpg")
    )


def _sample_frames_pyav(video_source: str, frame_dir: str, interval: int,
                        reporter: "Reporter | None",
                        headers: str = "") -> list[str]:
    """Fallback sampler via PyAV for environments without the ffmpeg
    binary (local dev).  Emits one jpg per >=interval seconds of media
    time, same numbering convention as the ffmpeg path."""
    import av
    if reporter:
        reporter.info(
            f"    [Board] PyAV sampling 1 frame / {interval}s ..."
        )
    os.makedirs(frame_dir, exist_ok=True)
    kwargs = {}
    if headers and video_source.startswith(("http://", "https://")):
        kwargs["headers"] = {
            kv.split("=", 1)[0].strip(): kv.split("=", 1)[1].strip()
            for kv in headers.replace("\r\n", "\n").split("\n")
            if "=" in kv
        }
    out: list[str] = []
    next_sec = 0.0
    idx = 1
    with av.open(video_source, **kwargs) as container:
        for frame in container.decode(video=0):
            t = frame.time if frame.time is not None else idx * 0.04
            if t < next_sec:
                continue
            img = frame.to_image()
            img.save(os.path.join(frame_dir, f"f_{idx:05d}.jpg"),
                     quality=92)
            out.append(os.path.join(frame_dir, f"f_{idx:05d}.jpg"))
            idx += 1
            next_sec = t + interval
    return out


def _sample_frames(video_source: str, frame_dir: str, interval: int,
                   reporter: "Reporter | None",
                   headers: str = "") -> list[str]:
    os.makedirs(frame_dir, exist_ok=True)
    if shutil.which("ffmpeg"):
        return _sample_frames_ffmpeg(video_source, frame_dir, interval,
                                     reporter, headers)
    return _sample_frames_pyav(video_source, frame_dir, interval,
                               reporter, headers)


def extract_board_pages(
    client: "ICourseClient",
    db: "Database",
    scheduler: "Scheduler",
    reporter: "Reporter | None",
    course_id: str,
    sub_id: str,
    video_url: str,
    workdir: str,
) -> int:
    """Sample the lecture video (streamed through ffmpeg with reconnect)
    and OCR its blackboard frames.

    Returns the number of board pages (page_num >= ``BOARD_PAGE_NUM_BASE``)
    in a terminal state afterwards (done + invalid).  0 means nothing was
    recovered; any hard failure (ffmpeg, stream) propagates — the caller
    treats the whole attempt as degraded.
    """
    sub_id = str(sub_id)
    done, pending, total_rows = _board_rows(db, sub_id)
    if total_rows and not pending and done:
        if reporter:
            reporter.info(
                f"    [Board] {total_rows} board rows already processed "
                f"({done} done), skipping extraction."
            )
        return done
    if total_rows and not pending and not done:
        # Previous attempt ended with nothing usable (all failed/invalid or
        # empty-text "done") — e.g. a force_reset race where the platform
        # PPT pipeline "fetched" our videoframe pseudo-URLs and marked them
        # failed.  Re-arm and redo instead of stranding the lecture.
        rearmed = db.reset_board_pages(sub_id, BOARD_PAGE_NUM_BASE)
        if reporter and rearmed:
            reporter.info(
                f"    [Board] re-armed {rearmed} empty/failed rows "
                f"for re-OCR."
            )

    frame_dir = os.path.join(workdir, f"{sub_id}_frames")
    mp4 = os.path.join(workdir, f"{sub_id}_board.mp4")
    interval = max(5, int(config.BOARD_FRAME_INTERVAL))
    try:
        # Best-effort download first; then sample from the FILE whenever it's
        # usable. A complete file (server-declared length reached) gives 100 %
        # board coverage; even an incomplete-but-substantial file decodes
        # offline to far more frames than a live stream that WebVPN cuts at
        # ~41 % (and the stream path is additionally prone to sporadic
        # 403/rate-limit deaths after burst traffic — 2026-10-08). Only fall
        # back to streaming when the download yielded nothing worth decoding.
        frames = None
        try:
            path, complete = client.download_video_resumable(video_url, mp4)
        except Exception as e:
            if reporter:
                reporter.info(f"    [Board] download failed ({type(e).__name__}) "
                              f"— streaming sample instead")
            path, complete = mp4, False
        size = os.path.getsize(path) if os.path.exists(path) else 0
        # Offline sampling is only worth trusting on a COMPLETE file: these
        # mp4s keep the moov index at the tail, so a partial download (even
        # 85 % of it) decodes ZERO frames (v3: rc=183, empty harvest) while
        # the streaming path — decode-as-you-receive, no index needed —
        # reliably yields frames from the same flaky link. A partial file
        # therefore only accelerates sampling if it happens to be faststart
        # (harvest catches that); otherwise fall through to streaming.
        if complete and size > 20 * 1024 * 1024:
            if reporter:
                reporter.info(
                    f"    [Board] offline sample from downloaded file "
                    f"({size / 1e6:.0f} MB"
                    + (", complete)" if complete else ", partial — best effort)")
                )
            try:
                frames = _sample_frames(path, frame_dir, interval, reporter, "")
            except Exception as e:
                # Decoding a truncated file exits non-zero AT the cut point,
                # but every jpg emitted before it is real — harvest them.
                if reporter:
                    reporter.info(
                        f"    [Board] offline decode stopped early "
                        f"({type(e).__name__}) — harvesting partial frames"
                    )
                frames = sorted(
                    os.path.join(frame_dir, n)
                    for n in os.listdir(frame_dir)
                    if n.endswith(".jpg")
                ) if os.path.isdir(frame_dir) else []
        if not frames:
            vpn_url, headers = client.get_stream_params(video_url)
            frames = _sample_frames(vpn_url, frame_dir, interval, reporter,
                                    headers)
        if not frames:
            return 0
        if reporter:
            reporter.info(f"    [Board] {len(frames)} frames sampled.")

        # dHash every frame; drop known-garbage screens, then collapse
        # near-identical consecutive blackboard states.  (dedup_dhash
        # returns the DROPPED indices within the list passed to it.)
        hashes = []
        for path in frames:
            with open(path, "rb") as f:
                hashes.append(compute_dhash(f.read()))
        garbage = set(match_garbage(hashes))
        keep_idx = [i for i in range(len(frames)) if i not in garbage]
        dropped = set(dedup_dhash(
            [hashes[i] for i in keep_idx],
            max_survivors=int(config.BOARD_MAX_PAGES),
        ))
        ocr_idx = [keep_idx[i] for i in range(len(keep_idx))
                   if i not in dropped]
        if reporter:
            reporter.info(
                f"    [Board] {len(ocr_idx)} pages kept after dedup "
                f"({len(garbage)} garbage)."
            )

        # Register synthetic rows, then OCR the still-pending ones.
        items = []
        for i in ocr_idx:
            sec = i * interval
            items.append({
                "page_num": BOARD_PAGE_NUM_BASE + i,
                "created_sec": sec,
                "pptimgurl": f"videoframe:{sec}",
            })
        db.insert_ppt_pages_pending(sub_id, items)
        pend_nums = {
            int(p["page_num"])
            for p in db.get_pending_ppt_pages(sub_id)
            if int(p["page_num"]) >= BOARD_PAGE_NUM_BASE
        }

        def _ocr_worker(page_num: int, path: str) -> tuple[int, str]:
            try:
                with open(path, "rb") as f:
                    text = ocr_image_text(f.read())
            except Exception as e:
                print(f"      board page {page_num}: OCR error "
                      f"{type(e).__name__}: {e}", flush=True)
                db.update_ppt_page(sub_id, page_num, None, "failed")
                return page_num, "failed"
            cleaned = clean_ppt_text(text)
            status = "invalid" if is_invalid_page(cleaned) else "done"
            db.update_ppt_page(sub_id, page_num, cleaned, status)
            return page_num, status

        futs = [
            scheduler.submit_ocr(
                _ocr_worker, BOARD_PAGE_NUM_BASE + i, frames[i],
            )
            for i in ocr_idx
            if BOARD_PAGE_NUM_BASE + i in pend_nums
        ]
        done_ct = invalid_ct = failed_ct = 0
        for fut in as_completed(futs):
            try:
                _pn, status = fut.result()
            except Exception:
                failed_ct += 1
                continue
            if status == "done":
                done_ct += 1
            elif status == "invalid":
                invalid_ct += 1
            else:
                failed_ct += 1
        if reporter:
            reporter.info(
                f"    [Board] OCR done: {done_ct} kept, "
                f"{invalid_ct} invalid, {failed_ct} failed."
            )
        done, _pending, _total = _board_rows(db, sub_id)
        return done
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)
        if os.path.exists(mp4):
            try:
                os.remove(mp4)
            except OSError:
                pass
