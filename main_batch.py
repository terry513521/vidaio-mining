#!/usr/bin/env python3
"""Fleet SLA compression: optional download/upload, one CQ probe, one x265 final."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from batch_search import (
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

    mode = "local" if template.skip_transfer else "http"
    codec_txt = (
        f"ABR/{template.target_bitrate}"
        if template.is_abr
        else "RC/CRF"
    )
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
        f"Fleet SLA ({mode}, {codec_txt}): {len(jobs)} video(s), "
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
    log(f"Done: {ok}/{len(results)} videos delivered (db run_id={run_id})")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
