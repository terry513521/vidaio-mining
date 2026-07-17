#!/usr/bin/env python3
"""Collect JSON encode results (no videos) and publish to GitHub.

Copies ``work_fleet/<threshold>/*/best.json`` (+ light summary) into
``published_results/<threshold>/``. If that folder has new/untracked/changed
files, commits and pushes automatically (videos are never included).

Examples:
  python scripts/publish_results.py --threshold 85
  python scripts/publish_results.py --all
  python scripts/publish_results.py --threshold 85 --no-push
  python scripts/publish_results.py --threshold 85 --no-commit
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PUBLISH_ROOT = ROOT / "published_results"


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=check,
    )


def _git(*args: str, check: bool = True) -> str:
    proc = _run(["git", *args], check=check)
    return (proc.stdout or "").strip()


def discover_thresholds(explicit: list[int] | None, all_thr: bool) -> list[int]:
    if explicit:
        return sorted(set(explicit))
    if all_thr:
        work = ROOT / "work_fleet"
        found: list[int] = []
        if work.is_dir():
            for path in work.iterdir():
                if path.is_dir() and path.name.isdigit():
                    found.append(int(path.name))
        return sorted(found)
    req = ROOT / "request.json"
    if req.is_file():
        data = json.loads(req.read_text(encoding="utf-8"))
        if "vmaf_threshold" in data:
            return [int(data["vmaf_threshold"])]
    return discover_thresholds(None, True)


def collect_threshold(threshold: int) -> dict[str, Any]:
    src_root = ROOT / "work_fleet" / str(threshold)
    dst_root = PUBLISH_ROOT / str(threshold)
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    if not src_root.is_dir():
        return {
            "threshold": threshold,
            "job_count": 0,
            "ok_count": 0,
            "entries": [],
            "dest": str(dst_root),
        }

    for best_path in sorted(src_root.glob("*/best.json")):
        job_id = best_path.parent.name
        try:
            payload = json.loads(best_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  skip {best_path}: {exc}", file=sys.stderr)
            continue

        slim = {
            "job_id": payload.get("job_id") or job_id,
            "input_path": payload.get("input_path"),
            "vmaf_threshold": payload.get("vmaf_threshold", threshold),
            "crf": payload.get("crf"),
            "libx265_params": payload.get("libx265_params"),
            "s_f": payload.get("s_f"),
            "vmaf": payload.get("vmaf"),
            "vmaf_base": payload.get("vmaf_base"),
            "compression_rate": payload.get("compression_rate"),
            "features": payload.get("features") or {},
            "param_tune_trials": payload.get("param_tune_trials"),
            "error": payload.get("error"),
        }
        out_dir = dst_root / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "best.json").write_text(
            json.dumps(slim, indent=2) + "\n", encoding="utf-8"
        )

        result_src = best_path.parent / "result.json"
        if result_src.is_file():
            try:
                result = json.loads(result_src.read_text(encoding="utf-8"))
                light_result = {
                    "job_id": result.get("job_id"),
                    "vmaf_threshold": result.get("vmaf_threshold", threshold),
                    "chosen_crf": result.get("chosen_crf"),
                    "libx265_params": result.get("libx265_params"),
                    "strategy": result.get("strategy"),
                    "elapsed_sec": result.get("elapsed_sec"),
                    "error": result.get("error"),
                    "best": result.get("best"),
                    "features": result.get("features"),
                    "stage_timings": result.get("stage_timings"),
                }
                (out_dir / "result.json").write_text(
                    json.dumps(light_result, indent=2) + "\n", encoding="utf-8"
                )
            except (OSError, json.JSONDecodeError):
                pass

        entries.append(
            {
                "job_id": slim["job_id"],
                "crf": slim["crf"],
                "libx265_params": slim["libx265_params"],
                "s_f": slim["s_f"],
                "vmaf": slim["vmaf"],
                "compression_rate": slim["compression_rate"],
                "error": slim["error"],
            }
        )

    ok_count = sum(
        1 for e in entries if e.get("crf") is not None and not e.get("error")
    )
    summary = {
        "threshold": threshold,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "job_count": len(entries),
        "ok_count": ok_count,
        "entries": entries,
    }
    (dst_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    csv_lines = ["job_id,crf,s_f,vmaf,compression_rate,error,libx265_params"]
    for e in entries:
        params = str(e.get("libx265_params") or "").replace('"', "'")
        err = str(e.get("error") or "").replace(",", ";")
        csv_lines.append(
            f"{e.get('job_id')},{e.get('crf')},{e.get('s_f')},{e.get('vmaf')},"
            f"{e.get('compression_rate')},{err},\"{params}\""
        )
    (dst_root / "summary.csv").write_text(
        "\n".join(csv_lines) + "\n", encoding="utf-8"
    )
    summary["dest"] = str(dst_root)
    return summary


def commit_and_push(*, thresholds: list[int], push: bool, message: str) -> int:
    """Commit ``published_results/`` when it has new/untracked/changed files."""
    # Stage everything under published_results (new + modified).
    _run(["git", "add", "-A", "--", "published_results"])
    staged = _git("diff", "--cached", "--name-only")
    untracked = _git(
        "ls-files", "--others", "--exclude-standard", "--", "published_results"
    )
    if not staged and not untracked:
        # Also treat "already staged nothing" after add as no-op.
        status = _git("status", "--porcelain", "--", "published_results")
        if not status:
            print("No untracked/changed files under published_results/ — skip commit")
            return 0

    # Re-add in case ls-files showed leftover untracked (should be empty after add).
    _run(["git", "add", "-A", "--", "published_results"])
    staged = _git("diff", "--cached", "--name-only")
    if not staged:
        print("No untracked/changed files under published_results/ — skip commit")
        return 0

    thr_txt = ",".join(str(t) for t in thresholds)
    msg = message or f"Publish fleet JSON results for VMAF threshold(s) {thr_txt}"
    proc = _run(["git", "commit", "-m", msg], check=False)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        return proc.returncode
    print(f"Committed published_results/ ({len(staged.splitlines())} files, thr={thr_txt})")

    if not push:
        print("Commit done (push skipped: --no-push)")
        return 0

    branch = _git("branch", "--show-current") or "HEAD"
    proc = _run(["git", "push", "-u", "origin", "HEAD"], check=False)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        print(
            "Push failed — commit is local. Fix auth/remote, then: "
            f"git push -u origin {branch}",
            file=sys.stderr,
        )
        return proc.returncode
    remote = _git("remote", "get-url", "origin", check=False)
    print(f"Pushed to {remote} ({branch})")
    print(
        "Browse: https://github.com/terry513521/vidaio-mining/"
        f"tree/{branch}/published_results"
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--threshold",
        type=int,
        action="append",
        default=[],
        help="VMAF threshold to publish (repeatable)",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Publish all thresholds found under work_fleet/",
    )
    p.add_argument(
        "--push",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="git push after commit when there are changes (default: true)",
    )
    p.add_argument("--message", default="", help="Commit message override")
    p.add_argument(
        "--no-commit",
        action="store_true",
        help="Only write published_results/ (no git commit/push)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    thresholds = discover_thresholds(args.threshold or None, args.all)
    if not thresholds:
        print("No thresholds found to publish", file=sys.stderr)
        return 1

    PUBLISH_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for thr in thresholds:
        print(f"Collecting threshold {thr} …")
        summary = collect_threshold(thr)
        summaries.append(summary)
        print(
            f"  → {summary['dest']}: {summary['ok_count']}/{summary['job_count']} ok"
        )

    index = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": [
            {
                "threshold": s["threshold"],
                "job_count": s["job_count"],
                "ok_count": s["ok_count"],
            }
            for s in summaries
        ],
    }
    (PUBLISH_ROOT / "index.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )
    (PUBLISH_ROOT / "README.md").write_text(
        "# Published fleet results\n\n"
        "JSON-only encode configs (CRF, libx265 params, features, scores).\n"
        "No videos are included.\n\n"
        "Layout: `published_results/<vmaf_threshold>/<job_id>/best.json`\n",
        encoding="utf-8",
    )

    if args.no_commit:
        print(f"Wrote {PUBLISH_ROOT} (no commit)")
        return 0
    return commit_and_push(
        thresholds=thresholds,
        push=bool(args.push),
        message=args.message,
    )


if __name__ == "__main__":
    raise SystemExit(main())
