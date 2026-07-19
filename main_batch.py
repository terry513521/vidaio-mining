#!/usr/bin/env python3
"""Fleet SLA compression: optional download/upload, one CQ probe, one x265 final.

By default, jobs that already have a usable ``best.json`` under
``published_results/<vmaf_threshold>/<job_id>/`` (or the per-job work dir)
are skipped so a stopped fleet run can resume. Use ``--force`` to re-run all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from batch_search import (
    filter_pending_fleet_jobs,
    fleet_job_final_payload,
    fleet_jobs_from_request,
    load_fleet_jobs,
    run_fleet_sla,
)
from logutil import log
from request import CompressionRequest
from results_db import DEFAULT_DB_PATH, finish_run, start_run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fleet HEVC SLA runner (local paths or HTTP jobs)"
    )
    p.add_argument("-r", "--request", required=True, help="Shared request JSON template")
    p.add_argument(
        "-m",
        "--manifest",
        help="Local manifest: one input path per line (or input<TAB>output)",
    )
    p.add_argument(
        "--local",
        action="store_true",
        help="Skip HTTP download/upload (local files only)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max videos (0 = all jobs in request; default 0)",
    )
    p.add_argument("--output-dir", default="output/fleet", help="Output dir for encodes")
    p.add_argument("--work-root", default="work_fleet", help="Per-job work directories")
    p.add_argument(
        "--results",
        default="work_fleet/batch_results.json",
        help="Summary JSON path",
    )
    p.add_argument(
        "--results-db",
        default=str(DEFAULT_DB_PATH),
        help="SQLite results database path",
    )
    p.add_argument(
        "--published-root",
        default="published_results",
        help="Root of published JSON results used to skip finished jobs",
    )
    p.add_argument(
        "--skip-published",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip jobs with a usable best.json in published_results/ "
            "or work_dir (default: true)"
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run all jobs (same as --no-skip-published)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    template = CompressionRequest.from_json(args.request)
    template.serial_cq_search = True
    template.max_workers = 1
    if args.local:
        template.skip_transfer = True
        template.download_reserve_sec = 0.0
        template.upload_reserve_sec = 0.0

    if template.jobs:
        jobs = fleet_jobs_from_request(template)
        if args.limit and args.limit > 0:
            jobs = jobs[: args.limit]
    elif args.manifest:
        jobs = load_fleet_jobs(
            args.manifest,
            output_dir=args.output_dir,
            work_root=args.work_root,
            limit=args.limit,
        )
    else:
        log("Request must contain jobs, or provide --manifest", file=sys.stderr)
        return 2
    if not jobs:
        log("No jobs found", file=sys.stderr)
        return 1

    skip_complete = bool(args.skip_published) and not bool(args.force)
    pending, skipped = filter_pending_fleet_jobs(
        jobs,
        threshold=template.vmaf_threshold,
        published_root=args.published_root,
        skip_complete=skip_complete,
        seed_work=True,
    )
    if skipped:
        sample = ", ".join(j.job_id for j in skipped[:12])
        more = f" … (+{len(skipped) - 12})" if len(skipped) > 12 else ""
        log(
            f"Skipping {len(skipped)} finished job(s) "
            f"(published_results / work best.json): {sample}{more}"
        )
    jobs = pending
    if not jobs:
        log(
            f"All {len(skipped)} job(s) already complete under "
            f"{args.published_root}/ — nothing to run"
        )
        return 0

    mode = "local" if template.skip_transfer else "http"
    if template.is_abr:
        rate = template.target_compression_rate
        if template.target_bitrate:
            codec_txt = f"ABR/{template.target_bitrate}"
        elif rate is not None:
            codec_txt = f"ABR/rate={float(rate):.4f}"
        else:
            codec_txt = "ABR/per-job"
        if template.vbr_fallback_to_crf:
            codec_txt += "+crf_fallback"
    else:
        codec_txt = "RC/CRF"
    run_id = start_run(
        db_path=args.results_db,
        request_path=args.request,
        work_root=args.work_root,
        strategy=(
            "fleet_sla_x265_fixed_vbr"
            if template.is_abr
            else "fleet_sla_x265_full_crf"
        ),
    )
    log(
        f"Fleet SLA ({mode}, {codec_txt}): {len(jobs)} video(s)"
        f"{f', skipped={len(skipped)}' if skipped else ''}, "
        f"batch_size={template.fleet_batch_size}, run_id={run_id}, "
        f"db={args.results_db}"
    )
    results = run_fleet_sla(
        template,
        jobs,
        run_id=run_id,
        results_db_path=args.results_db,
    )

    summary = [
        fleet_job_final_payload(job, elapsed_sec=result.elapsed_sec)
        for result, job in zip(results, jobs)
    ]
    out_path = Path(args.results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    for job in jobs:
        log(f"  [{job.job_id}] final → {Path(job.work_dir) / 'result.json'}")

    ok = sum(
        1
        for r, job in zip(results, jobs)
        if r.best and r.best.score.s_f > 0 and job.uploaded and not job.error
    )
    finish_run(
        run_id,
        db_path=args.results_db,
        job_count=len(results),
        ok_count=ok,
    )
    log(
        f"Done: {ok}/{len(results)} videos delivered "
        f"(skipped {len(skipped)} already complete; db run_id={run_id})"
    )
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
