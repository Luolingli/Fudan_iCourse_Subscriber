"""iCourse Subscriber — top-level orchestration.

The runtime is split across cooperating components — Scheduler (pools +
resource monitor), Reporter (centralised logging), PPTPipeline, LectureRunner,
AudioDownloader.  This file does only orchestration:

  1. Build all components.
  2. Login + enumerate.
  3. Drive LectureRunner across the queued lectures.
  4. Email + bookkeeping.
  5. Shutdown.

Anything more interesting belongs in one of ``src/*`` modules.
"""

import datetime
import time
import traceback

from src.runtime import config
from src.data.database import Database
from src.api.emailer import Emailer
from src.api.icourse import ICourseClient, fetch_ppt_image
from src.ai.ocr import ocr_image_text
from src.pipeline.lecture_runner import LectureRunner
from src.runtime.reporter import Reporter
from src.runtime.scheduler import Scheduler
from src.ai.summarizer import Summarizer
from src.ai.transcriber import Transcriber
from src.api.webvpn import WebVPNSession


def login_with_retry(max_attempts: int = 10) -> WebVPNSession:
    """Login to WebVPN + iCourse CAS, retrying on transient failures.

    The iCourse CAS step (authenticate_icourse) has its own inner retry
    loop for transient redirect-chain hiccups; this outer loop only runs
    when those inner retries are exhausted, which generally means the
    WebVPN session itself needs a fresh login.  5 attempts handles the
    long tail of times when CAS rejects multiple fresh sessions in a
    row before letting one through.
    """
    for attempt in range(max_attempts):
        try:
            vpn = WebVPNSession()
            print(f"\n[Login] WebVPN (attempt {attempt + 1}/{max_attempts})...")
            vpn.login()
            print("[Login] iCourse CAS...")
            vpn.authenticate_icourse()
            return vpn
        except Exception as e:
            if attempt < max_attempts - 1:
                print(f"  Failed: {type(e).__name__}: {e}; retrying...")
                time.sleep(5)
            else:
                raise


def _check_session(client: ICourseClient) -> None:
    """Verify WebVPN session; re-login in place if expired.

    Mutates ``client`` so background workers holding the same instance
    automatically pick up refreshed cookies.
    """
    if client.check_alive():
        return
    print("[Session] WebVPN session expired, re-logging in...")
    client.vpn = login_with_retry()
    client._userinfo = None


def _enumerate_lectures(client: ICourseClient, db: Database,
                        reporter: Reporter) -> list[tuple[str, str, dict]]:
    """Sync, fast: list every (course_id, course_title, lecture) we'll
    process this run.  Done up-front so the prefetch loop can see across
    course boundaries when picking the "next" lecture."""
    out: list[tuple[str, str, dict]] = []
    for course_id in config.COURSE_IDS:
        try:
            _check_session(client)
            detail = client.get_course_detail(course_id)
            course_title = detail["title"]
            teacher = detail["teacher"]
            lectures = detail["lectures"]
            playback_count = sum(1 for l in lectures if l.get("has_playback"))
            reporter.course_header(
                course_id, course_title, teacher,
                total=len(lectures), playback=playback_count,
            )
            db.upsert_course(course_id, course_title, teacher)

            # School system sometimes lists duplicate lectures; dedup the
            # raw list so the same logic produces the same outcome each run.
            # When a sub_title appears more than once, keep the first one
            # that has playback; if none have playback, keep the first.
            seen_sub_titles: dict[str, dict] = {}
            deduped = []
            for lec in lectures:
                title = lec.get("sub_title", "")
                if not title:
                    deduped.append(lec)
                    continue
                existing = seen_sub_titles.get(title)
                if existing is None:
                    seen_sub_titles[title] = lec
                    deduped.append(lec)
                elif not existing.get("has_playback") and lec.get("has_playback"):
                    # replace the earlier no-playback entry in place so the
                    # processing order stays chronological
                    deduped[deduped.index(existing)] = lec
                    seen_sub_titles[title] = lec
                    reporter.course_dedup_skip(title, existing["sub_id"])
                else:
                    reporter.course_dedup_skip(title, lec["sub_id"])
            lectures = deduped

            known_processed = db.get_processed_sub_ids(course_id)
            abandoned = db.get_abandoned_sub_ids(course_id)
            new_lectures = [
                lec for lec in lectures
                if lec.get("has_playback")
                and str(lec["sub_id"]) not in known_processed
                and str(lec["sub_id"]) not in abandoned
            ]
            unprocessed = db.get_unprocessed_lectures(course_id)
            new_ids = {str(lec["sub_id"]) for lec in new_lectures}
            retry_only = [
                {"sub_id": u["sub_id"], "sub_title": u["sub_title"],
                 "date": u["date"]}
                for u in unprocessed if u["sub_id"] not in new_ids
            ]
            new_lectures.extend(retry_only)
            reporter.course_new_count(len(new_lectures))
            if not new_lectures:
                continue

            for lecture in new_lectures:
                sub_id = str(lecture["sub_id"])
                db.insert_lecture(
                    sub_id, course_id,
                    lecture.get("sub_title", ""),
                    lecture.get("date", ""),
                )
                out.append((course_id, course_title, lecture))
        except Exception:
            reporter.course_enumeration_error(course_id)
            traceback.print_exc()
    return out


def _drive_lectures(client: ICourseClient, db: Database,
                    scheduler: Scheduler, transcriber: Transcriber,
                    summarizer: Summarizer, reporter: Reporter,
                    all_lectures: list[tuple[str, str, dict]],
                    email_items: list) -> None:
    """Phase 2: run each lecture through LectureRunner.

    Pre-schedules the first lecture's prefetch (audio + images) before
    entering the loop; subsequent prefetches are kicked off from inside
    each LectureRunner.run via ``next_info``.
    """
    if not all_lectures:
        return

    runner = LectureRunner(
        client, db, scheduler, transcriber, summarizer, reporter,
    )

    first_course, _, first_lec = all_lectures[0]
    runner.prefetch_first(first_course, str(first_lec["sub_id"]))

    for i, (course_id, course_title, lecture) in enumerate(all_lectures):
        sub_id = str(lecture["sub_id"])
        next_info: tuple[str, str] | None = None
        if i + 1 < len(all_lectures):
            next_course, _, next_lec = all_lectures[i + 1]
            next_info = (next_course, str(next_lec["sub_id"]))

        _check_session(client)
        try:
            summary = runner.run(
                course_id, course_title, lecture, next_info=next_info,
            )
            if summary:
                email_items.append({
                    "sub_id": sub_id,
                    "course_title": course_title,
                    "sub_title": lecture.get("sub_title", sub_id),
                    "date": lecture.get("date", ""),
                    "summary": summary,
                })
        except Exception:
            reporter.lecture_error(sub_id)
            traceback.print_exc()
        finally:
            # Belt-and-braces: drop any lingering prefetch entry for this
            # lecture so we don't leak bytes if the runner crashed before
            # PPTPipeline.submit released the cache.
            scheduler.image_cache.discard(sub_id)
            scheduler.audio_downloader.release(sub_id)


def _send_email(emailer: Emailer | None, db: Database, reporter: Reporter,
                email_items: list) -> None:
    """Append any previously-processed-but-unsent lectures, then send."""
    unsent = db.get_unsent_lectures()
    if unsent:
        seen_sub_ids = {item["sub_id"] for item in email_items}
        for row in unsent:
            if row["sub_id"] not in seen_sub_ids:
                email_items.append({
                    "sub_id": row["sub_id"],
                    "course_title": row["course_title"],
                    "sub_title": row["sub_title"],
                    "date": row["date"],
                    "summary": row["summary"],
                })
        reporter.email_recovered_unsent(len(unsent))

    if not (emailer and email_items):
        return
    try:
        reporter.email_summary(len(email_items))
        if emailer.send(email_items):
            db.mark_emailed_batch([item["sub_id"] for item in email_items])
        else:
            reporter.email_failed()
    except Exception:
        reporter.info("[Email] Failed to send:")
        traceback.print_exc()


def _crawl_semester_catalog(client: ICourseClient, db: Database,
                            reporter: Reporter) -> None:
    """Auto-discover every available semester and refresh ``all_courses``.

    Walks every page of get-course-list for each discovered term and
    replaces the term's catalog in one transaction.  No longer requires
    the ``CRAWL_TERM`` secret — the API tells us what terms exist.
    """
    reporter.info("Discovering available semesters from API...")
    try:
        _check_session(client)
        terms = client.discover_terms()
    except Exception as e:
        reporter.crawl_courses_failed("discovery", e)
        return

    if not terms:
        reporter.info("No semesters found via API discovery.")
        return

    reporter.info(f"Found {len(terms)} semester(s): "
                  f"{', '.join(t['name'] for t in terms)}")

    for term_info in terms:
        code = term_info["code"]
        name = term_info["name"]
        expected = term_info["count"]
        reporter.crawl_courses_start(name)
        t0 = time.time()
        try:
            _check_session(client)
            rows = client.list_semester_courses(code)
            if not rows:
                reporter.info(f"  Term {name}: API returned 0 courses, skipping.")
                continue
            # Pass the human-readable term name (not the API code) to the
            # DB so the frontend displays "2025-20262" instead of "25".
            deleted, upserted = db.upsert_all_courses_for_term(name, rows)
            reporter.crawl_courses_done(
                name, len(rows), deleted, upserted, time.time() - t0,
            )
        except Exception as e:
            reporter.crawl_courses_failed(name, e)
        reporter.info(f"  ({code}) → {expected} API courses, "
                      f"{len(rows)} fetched")

    reporter.info("Semester catalog crawl complete.")


def _probe_ppt_feed(client, sid: str, row: dict | None = None) -> None:
    """Read-only forensics for the PPT screenshot feed of one lecture.

    Answers "does this lecture have board/board-slide content the summary is
    silently missing?" — the question behind 泛函分析 686005 (2026-10-21:
    `PPT pages: 1931 total` yet `PPT pipeline: 0 done, 1931 dedup'd`, i.e.
    the whole feed was thrown away and the summary came from audio alone).

    Prints one line per stage so the loss can be attributed rather than
    guessed at:
      [ppt  ] how many pages the platform offers + their time cadence
      [ppt  ] dHash variety BEFORE dedup (distinct hashes, near-dup rate)
      [ppt  ] how many each dedup stage would drop (garbage catalog /
              pairwise at the live thresholds) and how many survive OCR
      [ppt  ] how many OCR'd pages the DATABASE already keeps for it
    Nothing is written: no DB rows, no OCR, no downloads beyond the feed
    metadata and a sample of images capped by ``sample``.
    """
    import statistics
    try:
        from src.ai.ppt_dedup import (
            compute_dhash, match_garbage, dedup_dhash,
        )
        from src.ai.ppt_dedup_config import (
            DHASH_THRESHOLD, DHASH_MAX_SURVIVORS,
        )

        items = client.get_ppt_list(
            str((row or {}).get("course_id") or ""), sid)
        if not items:
            print("  [ppt  ] feed EMPTY — no screenshot rows at all",
                  flush=True)
            return
        secs = [int(i.get("created_sec") or 0) for i in items]
        span = secs[-1] - secs[0] if len(secs) > 1 else 0
        gaps = [b - a for a, b in zip(secs, secs[1:]) if b > a]
        med_gap = statistics.median(gaps) if gaps else 0
        print(f"  [ppt  ] pages={len(items)} span={span}s "
              f"median_gap={med_gap:.1f}s "
              f"cadence≈{'5s camera frame' if 4 <= med_gap <= 6 else 'slide changes'}",
              flush=True)

        # Sample evenly so a 1931-page feed still answers quickly.
        sample_n = min(60, len(items))
        step = max(1, len(items) // sample_n)
        sample = items[::step][:sample_n]
        dhashes, fetched = [], 0
        for it in sample:
            img = fetch_ppt_image(client, it, max_attempts=2, timeout=30)
            if img is None:
                continue
            fetched += 1
            dhashes.append(compute_dhash(img))
        if not dhashes:
            print(f"  [ppt  ] sampled {len(sample)} pages, none fetchable",
                  flush=True)
            return
        distinct = len(set(h for h in dhashes if h))
        print(f"  [ppt  ] sampled={len(sample)} fetched={fetched} "
              f"distinct_dhash={distinct}/{fetched} "
              f"({distinct / max(fetched, 1):.0%} unique)", flush=True)

        garbage = match_garbage(dhashes)
        survivors = [h for i, h in enumerate(dhashes) if i not in set(garbage)]
        dropped = dedup_dhash(survivors)
        kept = len(survivors) - len(dropped)
        print(f"  [ppt  ] drop: garbage_catalog={len(garbage)} "
              f"pairwise@{DHASH_THRESHOLD}={len(dropped)} "
              f"-> survivors={kept}/{fetched} "
              f"(cap {DHASH_MAX_SURVIVORS})  => extrapolated full-feed "
              f"kept≈{round(kept / max(fetched, 1) * len(items))}/{len(items)}",
              flush=True)

        # THE decisive question: is the discarded material actually CONTENT?
        # A blackboard fills the frame with low-contrast dark pixels, which
        # collapses dHash toward all-zero — indistinguishable from a
        # no-signal/desktop splash by hash alone.  OCR a handful of the very
        # pages dedup threw away and judge the text the SAME way the pipeline
        # does (clean_ppt_text + is_invalid_page), with one addition: the
        # 686005 feed OCRs to the classroom DESKTOP and the platform's own
        # "录像指南" guide page, i.e. the screenshot job captured the PC
        # screen while this course is taught on a physical blackboard.  That
        # text is real characters, and the catalog deliberately leaves two
        # related entries commented out as too generic, so screen-capture
        # markers are called noise HERE — in the probe only.  A diagnostic
        # must not change the pipeline's classifier.
        SCREEN_CAPTURE_MARKERS = (
            "录像指南", "cfdfudan", "此电脑", "player", "potplayer",
            "课程录制", "demoasp", "yuyue", "u盘", "回收站",
        )

        def _is_screen_capture(text: str) -> bool:
            low = text.lower()
            return any(m in low for m in SCREEN_CAPTURE_MARKERS)

        from src.ai.ppt_dedup import clean_ppt_text, is_invalid_page
        ocr_n = min(8, len(sample))
        ocr_idx = list(range(ocr_n))
        meaningful = junk = 0
        snippets: list[str] = []
        for i in ocr_idx:
            img = fetch_ppt_image(client, sample[i], max_attempts=2,
                                  timeout=30)
            if img is None:
                continue
            try:
                text = clean_ppt_text(ocr_image_text(img))
            except Exception as e:
                print(f"  [ppt  ] OCR sample #{i} error "
                      f"{type(e).__name__}: {str(e)[:60]}", flush=True)
                continue
            text = text or ""
            if (len(text.strip()) < 12 or is_invalid_page(text)
                    or _is_screen_capture(text)):
                junk += 1
            else:
                meaningful += 1
                snippets.append(" ".join(text.split())[:70])
        judged = meaningful + junk
        print(f"  [ppt  ] OCR on discarded pages: {judged} judged, "
              f"{meaningful} carry NON-JUNK text, {junk} are noise "
              f"(desktop/guide/placeholder)", flush=True)
        for s in snippets[:4]:
            print(f"  [ppt  ]   content: {s}", flush=True)
        if meaningful:
            print("  [ppt  ] VERDICT: discarded frames hold real content — "
                  "the drop filters ARE eating lecture material", flush=True)
        elif judged:
            print("  [ppt  ] VERDICT: discarded frames are ALL noise "
                  "(desktop / 录像指南 / placeholder). The feed itself has no "
                  "board content — nothing is being lost by dropping them",
                  flush=True)

        if kept == 0:
            print("  [ppt  ] NOTE: dedup drops EVERY feed page — the "
                  "screenshot job captured the PC screen, not slides",
                  flush=True)
        elif med_gap and 4 <= med_gap <= 6 and kept / max(fetched, 1) < 0.1:
            print("  [ppt  ] NOTE: camera-cadence feed ("
                  f"{med_gap:.0f}s/page) keeps only "
                  f"{kept / max(fetched, 1):.0%} of sampled pages",
                  flush=True)
    except Exception as e:
        print(f"  [ppt  ] probe failed: {type(e).__name__}: "
              f"{str(e)[:110]}", flush=True)


def _probe_video_frames(client, cid: str, sid: str, date: str = "",
                        interval: int = 600, max_ocr: int = 8,
                        ) -> None:
    """Read-only: sample the LECTURE VIDEO and OCR its frames.

    Why this exists: the PPT-feed probe proved 686005's screenshot feed is the
    classroom PC screen (desktop + 录像指南 guide), so the platform never
    captured any blackboard for this course.  That leaves exactly one possible
    home for board content — the video itself — and the board-recovery path
    only runs when ASR FAILS (686005 transcribed 26,349 chars, so it never
    ran).  This probe pulls frames straight from the video and OCRs them, so
    "the video has a blackboard we never look at" becomes a measured fact
    instead of an assumption.

    Each frame is classified the way the pipeline would see it:
      board   — real text that is NOT a screen-capture artifact
      screen  — desktop / 录像指南 / player chrome (what the feed showed)
      empty   — (near) no OCR text at all
    Frames are discarded and nothing is persisted: read-only by construction.
    """
    import os
    import shutil
    import tempfile
    from src.ai.ppt_dedup import clean_ppt_text as _clean
    from src.pipeline.board_frames import _sample_frames
    # Screen-capture markers only CLASSIFY here; the production catalog keeps
    # two of these commented out as too generic, and a probe must not change
    # the pipeline it measures.
    _SCREEN = ("录像指南", "cfdfudan", "此电脑", "player", "potplayer",
               "课程录制", "demoasp", "yuyue", "u盘", "回收站", "资源管理器")
    tmpdir = None
    try:
        urls = client.get_video_url_candidates(cid, sid, date=date or None)
        if not urls:
            print("  [vid  ] no video URL", flush=True)
            return
        vpn_url, headers = client.get_stream_params(urls[0])
        # get_stream_params already returns ffmpeg-format header TEXT
        # ("Cookie: ...\r\nUser-Agent: ...\r\n"), not a dict — pass it
        # through exactly as board_frames.extract_board_pages does.
        tmpdir = tempfile.mkdtemp(prefix="diagframes_")
        try:
            frames = _sample_frames(vpn_url, tmpdir, interval, None,
                                    headers or "")
        except Exception as e:
            print(f"  [vid  ] sampling failed: {type(e).__name__}: "
                  f"{str(e)[:110]}", flush=True)
            return
        if not frames:
            print("  [vid  ] sampler produced no frames", flush=True)
            return
        print(f"  [vid  ] sampled {len(frames)} frame(s) "
              f"(1 per {interval}s of media)", flush=True)
        # Spread the OCR samples across the whole lecture rather than its
        # first minutes: a board lecture can open with 20 min of talking.
        if len(frames) <= max_ocr:
            pick = frames
        else:
            pick = [frames[round(i * (len(frames) - 1) / (max_ocr - 1))]
                    for i in range(max_ocr)]
        counts = {"board": 0, "screen": 0, "empty": 0}
        for path in pick:
            try:
                with open(path, "rb") as f:
                    raw = f.read()
                text = (_clean(ocr_image_text(raw)) or "").strip()
            except Exception as e:
                print(f"  [vid  ] {os.path.basename(path)} OCR error "
                      f"{type(e).__name__}: {str(e)[:60]}", flush=True)
                continue
            low = text.lower()
            if len(text) < 12:
                kind = "empty"
            elif any(m in low for m in _SCREEN):
                kind = "screen"
            else:
                kind = "board"
            counts[kind] += 1
            print(f"  [vid  ] {os.path.basename(path):12s} {kind:6s} "
                  f"chars={len(text):4d} :: "
                  f"{' '.join(text.split())[:80]}", flush=True)
        total = sum(counts.values())
        print(f"  [vid  ] classified {total}: board={counts['board']} "
              f"screen={counts['screen']} empty={counts['empty']}",
              flush=True)
        if counts["board"]:
            print("  [vid  ] VERDICT: the VIDEO carries board text that no "
                  "summary ever sees — the board path only runs when ASR "
                  "fails, so lectures like this silently lose their board",
                  flush=True)
        elif counts["screen"]:
            print("  [vid  ] VERDICT: the video is a SCREEN recording "
                  "(desktop/guide visible), not a blackboard capture — there "
                  "is no board content to OCR in this lecture", flush=True)
        elif total:
            print("  [vid  ] VERDICT: sampled frames carry no OCR text "
                  "(dark/talking-head view) — inconclusive for this position "
                  "set; try another interval before concluding", flush=True)
        print("  [vid  ] (read-only: frames discarded, nothing persisted)",
              flush=True)
    except Exception as e:
        print(f"  [vid  ] probe failed: {type(e).__name__}: "
              f"{str(e)[:110]}", flush=True)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _run_source_diag(client, db, reporter) -> None:
    """Read-only forensics sweep for DIAG_SOURCE_IDS entries. A plain
    sub_id probes that one lecture; a ``c:<course_id>`` entry sweeps the
    WHOLE course — one line per lecture with (a) official-transcript
    density (chars/s — the platform's own ASR view), (b) the first video
    variant's head-window envelope (max/median/speech_frac) and syllable
    AM spread. Reading down the course sorts the noise floor from the
    speech without any lecture selection bias: electrical hum shows
    am_spread≈0.05, buried-but-present speech still shows ≥0.3."""
    import json
    from src.ai.transcriber import probe_audio
    targets: list[tuple[str, str, str]] = []
    for entry in config.DIAG_SOURCE_IDS:
        if str(entry).startswith("c:"):
            cid = str(entry)[2:].strip()
            for row in db.list_lecture_digests(cid):
                targets.append((cid, str(row["sub_id"]),
                                str(row.get("date") or "")))
        else:
            row = db.get_lecture(str(entry)) or {}
            targets.append((str(row.get("course_id") or ""),
                            str(entry), str(row.get("date") or "")))
    def _audit(sid, row):
        """Per-lecture content inventory, straight from the local DB.

        Answers "which lecture is missing its board?" without touching the
        network: slides are ppt_pages with page_num < 10000, board frames are
        page_num >= 10000 (synthetic rows sampled from the video).
        """
        pages = db.get_done_ppt_pages(sid) or []
        slides, boards = [], []
        for pg in pages:
            try:
                num = int(pg.get("page_num") or 0)
            except (TypeError, ValueError):
                num = 0
            (boards if num >= 10000 else slides).append(pg)
        s_chars = sum(len((pg.get("text") or "").strip()) for pg in slides)
        b_chars = sum(len((pg.get("text") or "").strip()) for pg in boards)
        t_chars = len((row.get("transcript") or "").strip())
        m_chars = len((row.get("summary") or "").strip())
        # 有转录、课件文字不足、且完全没有板书 = 纯 ASR 交付，正是 Phase E3 要补的。
        gap = "MISSING-BOARD" if (t_chars and s_chars < 300
                                  and b_chars == 0) else ""
        print(f"  [audit ] transcript={t_chars}c summary={m_chars}c "
              f"slides={len(slides)}p/{s_chars}c "
              f"boards={len(boards)}p/{b_chars}c {gap}", flush=True)
        return dict(t=t_chars, m=m_chars, sc=s_chars, bc=b_chars,
                    gap=bool(gap))

    seen = set()
    rollup = []
    for cid, sid, date in targets:
        if sid in seen:
            continue
        seen.add(sid)
        row = db.get_lecture(sid) or {}
        reporter.info(
            f"=== DIAG {sid} course={cid} date={date} "
            f"processed={bool(row.get('processed_at'))} "
            f"err={row.get('error_stage')} ==="
        )
        try:
            rollup.append((sid, _audit(sid, row)))
        except Exception as e:
            print(f"  [audit ] failed: {type(e).__name__}: {e}", flush=True)
        if config.DIAG_AUDIT_ONLY:
            # Pure DB inventory: no network probes, no downloads.
            continue
        tr = client.probe_official_transcript(sid)
        dens = (tr["chars"] / tr["last_end_s"]
                if tr.get("found") and tr.get("last_end_s") else 0)
        print(f"  [trans ] density={dens:.2f} chars/s "
              + json.dumps(tr, ensure_ascii=False)[:150], flush=True)
        _probe_ppt_feed(client, sid, row=row)
        _probe_video_frames(client, cid, sid, date=date)
        try:
            urls = client.get_video_url_candidates(cid, sid, date=date or None)
            if not urls:
                print("  [env   ] no video url", flush=True)
                continue
            vpn_url, headers = client.get_stream_params(urls[0])
            # The mono envelope probe moved below: it now only runs as a
            # fallback when the stereo pull itself fails — and the stereo
            # window's mix view IS the same measurement anyway.  Order
            # matters: the second full mp4 pull of a lecture gets burst-
            # refused (run 38713503977: mono [env] decoded, both [anti]
            # pulls died instantly x2 retries, both lectures).
            # Stereo anti-phase check (upstream PR #51 hypothesis): a
            # "0 segments" lecture is either physically silent OR a
            # stereo track whose channels cancel in the default (L+R)/2
            # mono mix.  diff_speech_frac > mix_speech_frac with
            # mix/max_channel < 0.25 proves anti-phase (PR #51's
            # (L-R) mix would recover the voice).
            from src.ai.transcriber import probe_stereo_antiphase
            ap = probe_stereo_antiphase(vpn_url, headers,
                                        position=0, window_s=90)
            if ap.get("error"):
                print(f"  [anti  ] {ap['error']}", flush=True)
                res = probe_audio(vpn_url, headers, positions=(0,),
                                  window_s=90)
                print(f"  [env   ] max={res['max']:.5f} "
                      f"median={res['median']:.5f} "
                      f"speech_frac={res['speech_frac']:.0%} "
                      f"am_spread={res.get('am_spread', -1):.2f} "
                      f"ok={res['ok']} err={res['error']}", flush=True)
            else:
                print(f"  [anti  ] VERDICT: {ap['verdict']}", flush=True)
                for tag in ("L", "R", "mix", "diff"):
                    s = ap[tag]
                    print(f"  [anti  ]   {tag:>4}: sf={s['speech_frac']:.0%} "
                          f"am={s['am_spread']:.2f} mx={s['max']:.4f} "
                          f"md={s['median']:.4f}", flush=True)
                print(f"  [anti  ]   mix/ch={ap['mix_over_max_channel']:.2f} "
                      f"pwr-rule(PR51)={ap['antiphase']} "
                      f"lp={ap['l_power']:.5f} rp={ap['r_power']:.5f} "
                      f"mixp={ap['mix_power']:.5f} "
                      f"diffp={ap['diff_power']:.5f}", flush=True)
        except Exception as e:
            print(f"  [env   ] probe failed: "
                  f"{type(e).__name__}: {str(e)[:90]}", flush=True)

    if rollup:
        missing = [sid for sid, a in rollup if a["gap"]]
        have_b = [sid for sid, a in rollup if a["bc"] > 0]
        no_tr = [sid for sid, a in rollup if a["t"] == 0]
        print(f"  [audit ] === course rollup: {len(rollup)} lecture(s), "
              f"{len(have_b)} with board, {len(missing)} missing board "
              f"(transcript but no slide text, no board), "
              f"{len(no_tr)} with no transcript ===", flush=True)
        if missing:
            print(f"  [audit ] board candidates: {missing}", flush=True)


def _run_asr_probes(transcriber, reporter) -> None:
    """One-off forensics for force_audio entries flagged asr_test: chunk-
    decode their audio asset (16 kHz mono f32le raw) with the recognizer
    directly — VAD is bypassed entirely — and log the text per 5 s chunk.
    Lecture rows are untouched.  Used to decide whether near-inaudible
    speech exists in a recording the VAD never fires on (664140, 09-29).
    """
    import os
    import numpy as np
    from urllib.request import urlopen
    from src.ai.transcriber import SAMPLE_RATE
    probes = {k: v for k, v in config.FORCE_AUDIO_MAP.items()
              if v.get("asr_test")}
    for sid, entry in probes.items():
        try:
            # "url" may be a single asset or an ordered list of small parts
            # (long uploads from this uplink get RST mid-stream, so a whole
            # lecture is shipped as ~5 MB slices that each survive).
            import subprocess
            import tempfile
            urls = entry["url"]
            if isinstance(urls, str):
                urls = [urls]
            parts = []
            for u in urls:
                reporter.info(
                    f"[ASR probe] {sid}: downloading part "
                    f"{len(parts) + 1}/{len(urls)} ..."
                )
                with urlopen(u, timeout=1800) as r:
                    raw = r.read()
                if u.split("?")[0].endswith(".f32"):
                    parts.append(np.frombuffer(raw, dtype=np.float32))
                    continue
                # any container ffmpeg understands (opus/mp3/m4a…) decoded to
                # 16 kHz mono f32le — lets probes ship as tiny lossy-audio
                # assets instead of raw PCM hundreds of MB big.
                suffix = os.path.splitext(u.split("?")[0])[1] or ".bin"
                with tempfile.NamedTemporaryFile(suffix=suffix,
                                                 delete=False) as tf:
                    tf.write(raw)
                    tmp = tf.name
                try:
                    p = subprocess.run(
                        ["ffmpeg", "-hide_banner", "-v", "error", "-i", tmp,
                         "-ar", str(SAMPLE_RATE), "-ac", "1",
                         "-f", "f32le", "-"],
                        capture_output=True, timeout=900)
                finally:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
                if p.returncode != 0:
                    raise RuntimeError(
                        "ffmpeg decode failed: "
                        + p.stderr[-200:].decode(errors="replace"))
                parts.append(np.frombuffer(p.stdout, dtype=np.float32))
            buf = np.concatenate(parts) if len(parts) > 1 else parts[0]
            reporter.info(
                f"[ASR probe] {sid} audio ready: "
                f"{len(buf) / SAMPLE_RATE:.1f}s from {len(parts)} part(s)"
            )
            # 15 s windows on a 12 s stride (3 s overlap) — long enough to
            # carry a sentence across chunk seams, overlap means a word
            # straddling a boundary is caught whole by one of them.
            win = 15 * SAMPLE_RATE
            hop = 12 * SAMPLE_RATE
            offset = int(float(entry.get("lead_offset_sec") or 0) * SAMPLE_RATE)
            hits = 0
            n_chunks = max(1, (len(buf) - offset + hop - 1) // hop)
            for i in range(n_chunks):
                chunk = buf[offset + i * hop: offset + (i + 1) * hop + (win - hop)]
                if not len(chunk):
                    break
                t0 = (offset + i * hop) / SAMPLE_RATE
                text = transcriber.decode_chunk(chunk)
                if text.strip():
                    hits += 1
                    rms = float(np.sqrt(
                        (chunk.astype(np.float64) ** 2).mean()))
                    reporter.info(
                        f"[ASR probe] {sid} t={t0:07.1f}s "
                        f"rms={rms:.4f}: {text!r}"
                    )
                if (i + 1) % 100 == 0:
                    reporter.info(
                        f"[ASR probe] {sid} progress {i + 1}/{n_chunks} "
                        f"chunks, {hits} with text so far"
                    )
            reporter.info(
                f"[ASR probe] {sid} DONE: {n_chunks} chunks decoded, "
                f"{hits} produced text"
            )
        except Exception as e:
            reporter.info(
                f"[ASR probe] {sid} failed: {type(e).__name__}: {e}"
            )


def _run_ab_summary(client, db, reporter) -> None:
    """A/B a lecture's summary with and without board OCR — measures what the
    blackboard actually adds, instead of asserting it.

    For each sub_id in AB_SUMMARY_IDS: read the ALREADY DELIVERED transcript
    from the DB, fetch board frames from the video (read-only, same path a
    normal run uses), then summarize TWICE with the SAME model and the SAME
    transcript — once with the platform's page text only (arm A, the status
    quo for a board course: nothing) and once with the board supplement
    (arm B).  Prints both lengths, the board contribution, a term-coverage
    delta, and the head of each summary so the difference is readable.

    NOTHING is written: no summary, no transcript, no flags, no email.  The
    DB only gets board ppt_pages rows, which the normal pipeline also creates
    and which are idempotent by page_num.
    """
    import re as _re
    from src.ai import bucketer
    for sid in config.AB_SUMMARY_IDS:
        row = db.get_lecture(sid) or {}
        transcript = (row.get("transcript") or "").strip()
        print(f"\n=== AB {sid} course={row.get('course_id')} "
              f"title={row.get('sub_title')} "
              f"transcript={len(transcript)} chars ===", flush=True)
        if not transcript:
            print("  [ab   ] no delivered transcript in DB — skip "
                  "(nothing to hold constant across arms)", flush=True)
            continue
        # Board rows are stored in ppt_pages with page_num >= 10000 and are
        # INDISTINGUISHABLE from platform pages via get_done_ppt_pages.  That
        # broke the A/B twice: once a previous run had OCR'd the board into
        # the DB, pages_before already contained it and BOTH arms got the
        # same material (run 39918668873: "platform page text: 63391 chars,
        # 115 page(s)" before extraction, delta +32 chars, +0%).
        #
        # So the arms are defined by CONTENT, not by query time:
        #   platform pages = page_num <  BOARD_PAGE_NUM_BASE  (real slides)
        #   board pages    = page_num >= BOARD_PAGE_NUM_BASE  (video frames)
        _BOARD_BASE = 10000

        def _split_pages(rows):
            slides, boards = [], []
            for p in rows:
                try:
                    num = int(p.get("page_num") or 0)
                except (TypeError, ValueError):
                    num = 0
                (boards if num >= _BOARD_BASE else slides).append(p)
            return slides, boards

        pages_now = db.get_done_ppt_pages(sid)
        slides_now, boards_now = _split_pages(pages_now)
        base_text = "\n".join(p.get("text", "") for p in slides_now).strip()
        print(f"  [ab   ] existing: {len(slides_now)} slide page(s) "
              f"({len(base_text)} chars), {len(boards_now)} board page(s) "
              f"({len(chr(10).join(p.get('text','') for p in boards_now).strip())} chars)",
              flush=True)

        # Arm B needs board frames; reuse the production extraction path.
        # A REAL Scheduler is mandatory: extract_board_pages submits its OCR
        # work through scheduler.submit_ocr, so passing None made every
        # variant die with "'NoneType' object has no attribute 'submit_ocr'"
        # — run 39911424109 recovered 488 frames and 35 deduped pages, then
        # threw them all away, so the A/B compared two audio-only summaries.
        board_scheduler = Scheduler(reporter=reporter)
        try:
            runner = LectureRunner(client, db, board_scheduler, None, None,
                                   reporter)
            runner._last_media_duration = None
            recovered = 0
            if len(base_text) < config.BOARD_TEXT_MIN_CHARS and config.BOARD_FALLBACK:
                try:
                    recovered = runner._board_from_video(
                        str(row.get("course_id") or ""), sid,
                        str(row.get("date") or ""))
                except Exception as e:
                    print(f"  [ab   ] board extraction failed: "
                          f"{type(e).__name__}: {str(e)[:110]}", flush=True)
        finally:
            board_scheduler.shutdown()
        slides_after, boards_after = _split_pages(db.get_done_ppt_pages(sid))
        board_text = "\n".join(
            p.get("text", "") for p in boards_after).strip()
        print(f"  [ab   ] board available: {len(boards_after)} page(s), "
              f"{len(board_text)} chars (recovered {recovered} this run)",
              flush=True)
        if not board_text.strip():
            print("  [ab   ] NO board text — arms would be identical, "
                  "skipping this lecture rather than reporting a fake +0%",
                  flush=True)
            continue

        # Arm A: transcript + platform SLIDES only (board rows removed).
        prompt_a, mode_a = bucketer.assemble(transcript, [], slides_after)
        # Arm B: transcript + platform slides + board rows.
        prompt_b, _ = bucketer.assemble(
            transcript, [], slides_after + boards_after)

        arm_a = arm_b = ""
        _summ = _get_summarizer()
        _pin_model(_summ, config.AB_MODEL)
        try:
            arm_a, model_a = _summ.summarize(
                row.get("sub_title") or sid, prompt_a)
            print(f"  [ab   ] arm A (asr only)      : {len(arm_a)} chars "
                  f"via {model_a}", flush=True)
        except Exception as e:
            print(f"  [ab   ] arm A failed: {type(e).__name__}: "
                  f"{str(e)[:110]}", flush=True)
        try:
            arm_b, model_b = _summ.summarize(
                row.get("sub_title") or sid, prompt_b)
            print(f"  [ab   ] arm B (asr + board)   : {len(arm_b)} chars "
                  f"via {model_b}", flush=True)
        except Exception as e:
            print(f"  [ab   ] arm B failed: {type(e).__name__}: "
                  f"{str(e)[:110]}", flush=True)

        if arm_a and arm_b and model_a != model_b:
            print(f"  [ab   ] WARNING: arms used DIFFERENT models "
                  f"(A={model_a}, B={model_b}) — the delta below mixes board "
                  f"contribution with model difference. Set AB_MODEL to pin "
                  f"both arms before trusting it.", flush=True)
        if arm_a and arm_b:
            # Objective checks.  (1) Term coverage: proper names / theorem
            # names that appear in the BOARD text are exactly what ASR
            # mangles ("Hahn-Bamaeh" in the 686005 transcript vs
            # "Hahn-Banach" on the board) or omits entirely, so counting how
            # many the summary actually captured is the measurable signal.
            # (2) Section-structure coverage: board lectures are organized
            # by §/Def/Thm/Cor numbering that only exists on the board.
            terms = sorted({t for t in _re.findall(
                r"[A-Za-z][A-Za-z\-]{3,}", board_text)
                if t.lower() not in ("thm", "def", "cor", "sup", "inf",
                                     "theorem", "definition")})[:18]

            # OCR writes "Hahn-Banach" (ASCII hyphen) while an LLM summary
            # writes "Hahn–Banach" (en dash) — and may space the hyphen
            # ("Hahn - Banach"). Matching the raw strings undercounts both
            # arms; normalize every dash variant and surrounding spaces on
            # BOTH sides before comparing.
            def _norm(s: str) -> str:
                s = s.lower()
                for dash in ("\u2010", "\u2011", "\u2012", "\u2013",
                             "\u2014", "\u2015", "\u2212", "\uff0d"):
                    s = s.replace(dash, "-")
                s = _re.sub(r"\s*-\s*", "-", s)
                return s

            def _hits(summary, terms):
                low = _norm(summary)
                return [t for t in terms if _norm(t) in low]

            markers = sorted({
                m for m in _re.findall(
                    r"(?:§\s*\d+(?:\.\d+)*|(?:Thm|Def|Cor|Lem|Prop)\.?\s*"
                    r"\d+\.\d+(?:\.\d+)?)", board_text)})
            ha, hb = _hits(arm_a, terms), _hits(arm_b, terms)
            ma, mb = _hits(arm_a, markers), _hits(arm_b, markers)
            print(f"  [ab   ] delta: {len(arm_b) - len(arm_a):+d} chars "
                  f"({(len(arm_b) / max(len(arm_a), 1) - 1):+.0%})", flush=True)
            print(f"  [ab   ] board-term coverage: A={len(ha)}/"
                  f"{len(terms)} B={len(hb)}/{len(terms)}", flush=True)
            print(f"  [ab   ] board-section coverage: A={len(ma)}/"
                  f"{len(markers)} B={len(mb)}/{len(markers)}", flush=True)
            new_terms = [t for t in hb if t not in ha]
            print(f"  [ab   ] terms only B captured: {new_terms[:10]}",
                  flush=True)
            print(f"  [ab   ] --- arm A head ---\n{arm_a[:300]}", flush=True)
            print(f"  [ab   ] --- arm B head ---\n{arm_b[:300]}", flush=True)
        print("  [ab   ] (read-only: nothing persisted, nothing emailed)",
              flush=True)


_summ_singleton = None


def _get_summarizer():
    """Summarizer for read-only modes.

    A/B mode PINS the model when AB_MODEL is set: both arms must run on the
    same model or the comparison measures the model, not the board.  Run
    39927218197 hit exactly that — arm A on ModelScope V4.1-Flash, arm B
    silently fallen through to deepseek-flash after a 429.
    """
    global _summ_singleton
    if _summ_singleton is None:
        _summ_singleton = Summarizer()
    return _summ_singleton


def _pin_model(summarizer, spec: str):
    """Restrict a Summarizer to exactly one "provider/model", or no-op."""
    spec = (spec or "").strip()
    if not spec or "/" not in spec:
        return True
    want_prov, want_model = spec.split("/", 1)
    kept = []
    for prov in summarizer.providers:
        if prov.get("name") != want_prov:
            continue
        if want_model not in (prov.get("models") or []):
            print(f"  [ab   ] AB_MODEL {spec!r}: model not in provider list "
                  f"{prov.get('models')}", flush=True)
            return False
        prov = dict(prov)
        prov["models"] = [want_model]
        kept.append(prov)
    if not kept:
        print(f"  [ab   ] AB_MODEL {spec!r}: provider not configured",
              flush=True)
        return False
    summarizer.providers = kept
    print(f"  [ab   ] pinned BOTH arms to {spec} (clean comparison)",
          flush=True)
    return True


def run():
    """Single execution of the full pipeline."""
    reporter = Reporter()
    reporter.run_header()

    if not config.COURSE_IDS and not config.CRAWL_TERM:
        reporter.info(
            "No COURSE_IDS configured. Set COURSE_IDS to process lectures "
            "or leave empty for crawl-only mode."
        )
        # Fall through — crawl-only mode is valid.

    db = Database()
    corrected = db.sync_dates_from_sub()
    if corrected:
        print(f"  [Date] Synced {corrected} lecture date(s) from sub_title", flush=True)
    if config.FORCE_SUB_IDS:
        for sid in config.FORCE_SUB_IDS:
            if db.force_reset_lecture(sid):
                reporter.info(f"[Force] reset lecture {sid} for full reprocessing")
            else:
                reporter.info(f"[Force] lecture {sid} not found, nothing to reset")
    transcriber = Transcriber()
    _run_asr_probes(transcriber, reporter)
    summarizer = Summarizer() if config.COURSE_IDS else None
    emailer = Emailer() if (
        config.SMTP_EMAIL and config.SMTP_PASSWORD
    ) else None

    vpn = login_with_retry()
    client = ICourseClient(vpn)

    if config.DIAG_SOURCE_IDS:
        # Read-only forensics mode: object identity per video URL across
        # both stacks + official-transcript availability, then stop — no
        # lecture processing this run (DB untouched → commit skipped).
        _run_source_diag(client, db, reporter)
        return

    if config.AB_SUMMARY_IDS:
        # Read-only A/B mode: same transcript, same model, with vs without
        # board OCR — quantifies the board contribution, changes nothing.
        _run_ab_summary(client, db, reporter)
        return

    email_items: list = []

    # Refresh the semester catalog: run on the 5th and 25th of each month,
    # immediately if the database has no catalog data yet, or when forced
    # via FORCE_CRAWL (manual workflow_dispatch).
    has_catalog = db.has_all_courses()
    today = datetime.datetime.now().day
    if not has_catalog or today in (5, 25) or config.FORCE_CRAWL:
        _crawl_semester_catalog(client, db, reporter)
    else:
        reporter.info("Skipping catalog crawl (has data, not the 5th or 25th).")

    if not config.COURSE_IDS:
        # Crawl-only mode: nothing to process, just persist + exit.
        reporter.info("\n[Crawl-only mode] No COURSE_IDS — skipping lectures.")
        reporter.run_footer()
        return

    scheduler = Scheduler(reporter=reporter)

    try:
        all_lectures = _enumerate_lectures(client, db, reporter)
        _drive_lectures(
            client, db, scheduler, transcriber, summarizer, reporter,
            all_lectures, email_items,
        )

    finally:
        scheduler.shutdown()

    _send_email(emailer, db, reporter, email_items)
    reporter.run_footer()


if __name__ == "__main__":
    run()
