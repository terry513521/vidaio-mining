#!/usr/bin/env python3
"""Monitor a fleet SLA run and publish JSON results to GitHub when done.

Examples:
  # Watch until done, then commit+push published_results/ if anything new
  python scripts/monitor_fleet.py --watch

  # Watch without publishing
  python scripts/monitor_fleet.py --watch --no-publish

  # One-shot status
  python scripts/monitor_fleet.py --status
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _latest_log() -> Path | None:
    log_dir = ROOT / "logs"
    if not log_dir.is_dir():
        return None
    logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def _running_pids() -> list[int]:
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "main_batch.py --request"],
            text=True,
        )
    except subprocess.CalledProcessError:
        return []
    pids = []
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def _load_request_job_count(request_path: Path) -> tuple[int, int]:
    data = json.loads(request_path.read_text(encoding="utf-8"))
    jobs = data.get("jobs") or []
    thr = int(data.get("vmaf_threshold") or 85)
    return len(jobs), thr


def _count_results(threshold: int) -> tuple[int, int, int]:
    """Return (best_json, result_json, ok_best) under work_fleet/<thr>/."""
    root = ROOT / "work_fleet" / str(threshold)
    if not root.is_dir():
        return 0, 0, 0
    best = list(root.glob("*/best.json"))
    results = list(root.glob("*/result.json"))
    ok = 0
    for path in best:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("error"):
            continue
        if payload.get("crf") is None:
            continue
        ok += 1
    return len(best), len(results), ok


def _tail_lines(path: Path, n: int = 12) -> list[str]:
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-n:]


def print_status(
    *,
    request_path: Path,
    log_path: Path | None,
) -> dict:
    expected, thr = _load_request_job_count(request_path)
    best_n, result_n, ok_n = _count_results(thr)
    pids = _running_pids()
    status = {
        "threshold": thr,
        "expected_jobs": expected,
        "best_json": best_n,
        "result_json": result_n,
        "ok_best": ok_n,
        "running_pids": pids,
        "done": (not pids) and ok_n >= expected and expected > 0,
        "log": str(log_path) if log_path else None,
    }
    print(
        f"[fleet] thr={thr} ok={ok_n}/{expected} "
        f"best={best_n} result={result_n} "
        f"running={pids or '-'} "
        f"done={status['done']}"
    )
    if log_path and log_path.is_file():
        print(f"[fleet] log={log_path}")
        for line in _tail_lines(log_path, 8):
            print(f"  {line}")
    return status


def watch(
    *,
    request_path: Path,
    log_path: Path | None,
    interval_sec: float,
    publish: bool,
) -> int:
    stop = {"flag": False}

    def _handle(_sig, _frame) -> None:
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    print(
        f"[fleet] watching every {interval_sec:.0f}s "
        f"(Ctrl+C to stop monitor; batch keeps running)"
    )
    while not stop["flag"]:
        status = print_status(request_path=request_path, log_path=log_path)
        print("-" * 60)
        if status["done"]:
            print("[fleet] all jobs finished")
            if publish:
                return _publish(threshold=status["threshold"])
            return 0
        # If process died early with incomplete results, keep watching a bit
        # but report; user can decide.
        if not status["running_pids"] and status["ok_best"] < status["expected_jobs"]:
            print(
                "[fleet] main_batch not running and results incomplete — "
                "waiting in case another wave starts"
            )
        time.sleep(max(1.0, interval_sec))
    return 130


def _publish(threshold: int) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "publish_results.py"),
        "--threshold",
        str(threshold),
    ]
    print(f"[fleet] publishing (commit+push if changes): {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--request",
        default=str(ROOT / "request.json"),
        help="Request JSON (for expected job count + threshold)",
    )
    p.add_argument("--log", default="", help="Fleet log path (default: newest in logs/)")
    p.add_argument("--status", action="store_true", help="Print one status line and exit")
    p.add_argument("--watch", action="store_true", help="Poll until done")
    p.add_argument("--interval", type=float, default=30.0, help="Watch poll seconds")
    p.add_argument(
        "--publish",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After watch completes, publish JSON results (commit+push if changes; default: true)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    request_path = Path(args.request)
    if not request_path.is_file():
        print(f"request not found: {request_path}", file=sys.stderr)
        return 2
    log_path = Path(args.log) if args.log else _latest_log()
    if args.watch:
        return watch(
            request_path=request_path,
            log_path=log_path,
            interval_sec=args.interval,
            publish=bool(args.publish),
        )
    print_status(request_path=request_path, log_path=log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
