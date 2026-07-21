#!/usr/bin/env python3
"""Analyze vidaio validator compression score logs (HEVC, AV1, …).

Parses:
  - challenge protocol lines (codec / mode / VMAF threshold)
  - payload URL lists (reference videos for each scoring batch)
  - score_compressions result lines (uid, VMAF, rate, final, status, …)

Prints results grouped by UID, then by video within that UID.

Usage:
  python3 analyze_compression_log.py "/root/workspace/files_output (3).log"
  python3 analyze_compression_log.py log.txt --codec hevc
  python3 analyze_compression_log.py log.txt --codec av1 --csv out.csv
  python3 analyze_compression_log.py log.txt --uid 112
  python3 analyze_compression_log.py log.txt --uid 66,98,112 --codec hevc
  python3 analyze_compression_log.py log.txt --failures
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Iterable, Optional
from urllib.parse import urlparse, unquote


CHALLENGE_RE = re.compile(
    r"Built compression challenge protocol for (?P<n_miners>\d+) miners, "
    r"(?P<n_queries>\d+) queries, VMAF threshold (?P<vmaf_threshold>\d+), "
    r"codec (?P<codec>\w+), mode (?P<mode>\w+)"
    r"(?:, bitrate (?P<target_bitrate>[\d.]+) Mbps)?"
)

SCORE_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?).*"
    r"score_compressions:\d+ - "
    r"(?P<uid>\d+) \*\* VMAF NEG: (?P<vmaf_neg>[\d.]+) \*\* "
    r"VMAF: (?P<vmaf>\S+) \*\* "
    r"VMAF Threshold: (?P<vmaf_threshold>[\d.]+) \*\* "
    r"Compression Rate: (?P<compression_rate>[\d.]+) \*\* "
    r"Final: (?P<final>[\d.]+) "
    r"\|\| (?P<details>.*)$"
)

ENCODING_RE = re.compile(
    r"Valid encoding \((?P<codec>[^,]+),\s*(?P<resolution>[^,]*),\s*[^,]*,\s*"
    r"level\s+(?P<level>[^,]*),\s*fps\s+(?P<fps>[\d.]+),\s*"
    r"bitrate\s+(?P<bitrate>[\d.]+)\s*Mbps\)"
)

# payload: ['https://.../uuid.mp4?...', ...]
PAYLOAD_RE = re.compile(r"payload:\s*\[(?P<body>.*)\]\s*$")
URL_RE = re.compile(r"https?://[^\s'\"\]]+")
BATCH_UIDS_RE = re.compile(r"Uids:\s*\[(?P<body>[^\]]*)\]")


@dataclass
class Challenge:
    line: int
    n_miners: int
    n_queries: int
    vmaf_threshold: float
    codec: str
    mode: str
    target_bitrate: Optional[float] = None


@dataclass
class ScoreRow:
    line: int
    ts: str
    uid: int
    vmaf_neg: float
    vmaf: Optional[float]
    vmaf_threshold: float
    compression_rate: float
    compression_ratio: Optional[float]
    final: float
    success: bool
    status: str
    codec: Optional[str] = None
    mode: Optional[str] = None
    resolution: Optional[str] = None
    fps: Optional[float] = None
    bitrate_mbps: Optional[float] = None
    target_bitrate: Optional[float] = None
    dist_bpp: Optional[float] = None
    ref_bpp: Optional[float] = None
    challenge_codec: Optional[str] = None
    challenge_mode: Optional[str] = None
    video_idx: Optional[int] = None  # 1-based within the UID batch
    video_id: Optional[str] = None  # filename stem / uuid from payload URL
    video_url: Optional[str] = None
    details: str = ""
    extras: dict = field(default_factory=dict)


def _f(x: Optional[str]) -> Optional[float]:
    if x is None or x in ("", "N/A", "None"):
        return None
    try:
        return float(x)
    except ValueError:
        return None


def classify_status(details: str) -> tuple[bool, str]:
    d = details or ""
    if d.startswith("success"):
        return True, "success"
    if "below hard cutoff" in d:
        return False, "vmaf_hard_fail"
    if "soft zone" in d:
        return False, "vmaf_soft"
    if "Invalid or missing" in d:
        return False, "missing_video"
    if "MINER FAILURE" in d:
        return False, "miner_failure"
    if "VMAF" in d and "below" in d:
        return False, "vmaf_fail"
    return False, "other"


def parse_kv_tail(details: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in details.split(";"):
        part = part.strip()
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def video_id_from_url(url: str) -> str:
    """Extract a short video id from a reference payload URL."""
    path = unquote(urlparse(url).path)
    name = path.rsplit("/", 1)[-1]
    if name.endswith(".mp4"):
        name = name[:-4]
    return name or url


def parse_payload_urls(line: str) -> list[str]:
    m = PAYLOAD_RE.search(line)
    if not m:
        # payload may be split across the same log line after "payload: "
        if "payload:" not in line:
            return []
        body = line.split("payload:", 1)[1]
    else:
        body = m.group("body")
    return URL_RE.findall(body)


def parse_log_lines(lines: Iterable[str]) -> tuple[list[Challenge], list[ScoreRow]]:
    challenges: list[Challenge] = []
    scores: list[ScoreRow] = []
    current: Optional[Challenge] = None
    pending_videos: list[str] = []  # URLs from latest payload=
    # Track consecutive score index within current uid streak
    streak_uid: Optional[int] = None
    streak_i = 0

    for lineno, line in enumerate(lines, 1):
        cm = CHALLENGE_RE.search(line)
        if cm:
            current = Challenge(
                line=lineno,
                n_miners=int(cm.group("n_miners")),
                n_queries=int(cm.group("n_queries")),
                vmaf_threshold=float(cm.group("vmaf_threshold")),
                codec=cm.group("codec").lower(),
                mode=cm.group("mode").upper(),
                target_bitrate=_f(cm.group("target_bitrate") or ""),
            )
            challenges.append(current)
            pending_videos = []
            streak_uid = None
            streak_i = 0
            continue

        urls = parse_payload_urls(line)
        if urls and "payload:" in line:
            pending_videos = urls
            streak_uid = None
            streak_i = 0
            continue

        # Reset streak when a new scoring batch uid list appears
        if BATCH_UIDS_RE.search(line):
            streak_uid = None
            streak_i = 0
            continue

        sm = SCORE_RE.search(line)
        if not sm:
            continue

        details = sm.group("details").strip()
        success, status = classify_status(details)
        rate = float(sm.group("compression_rate"))
        ratio = (1.0 / rate) if rate > 0 else None

        enc = ENCODING_RE.search(details)
        kv = parse_kv_tail(details)

        codec = enc.group("codec").strip().lower() if enc else None
        challenge_codec = current.codec if current else None
        challenge_mode = current.mode if current else None
        if codec is None and challenge_codec:
            codec = challenge_codec

        mode = kv.get("mode")
        if mode:
            mode = mode.upper()
        elif challenge_mode:
            mode = challenge_mode

        target_br = _f((kv.get("target_bitrate") or "").replace("Mbps", "").strip())
        if target_br is None and current:
            target_br = current.target_bitrate

        uid = int(sm.group("uid"))
        if streak_uid == uid:
            streak_i += 1
        else:
            streak_uid = uid
            streak_i = 1

        video_url = None
        video_id = None
        if 0 <= streak_i - 1 < len(pending_videos):
            video_url = pending_videos[streak_i - 1]
            video_id = video_id_from_url(video_url)

        scores.append(
            ScoreRow(
                line=lineno,
                ts=sm.group("ts"),
                uid=uid,
                vmaf_neg=float(sm.group("vmaf_neg")),
                vmaf=_f(sm.group("vmaf")),
                vmaf_threshold=float(sm.group("vmaf_threshold")),
                compression_rate=rate,
                compression_ratio=ratio,
                final=float(sm.group("final")),
                success=success,
                status=status,
                codec=codec,
                mode=mode,
                resolution=enc.group("resolution").strip() if enc else None,
                fps=_f(enc.group("fps")) if enc else None,
                bitrate_mbps=_f(enc.group("bitrate")) if enc else None,
                target_bitrate=target_br,
                dist_bpp=_f(kv.get("dist_bpp", "")),
                ref_bpp=_f(kv.get("ref_bpp", "")),
                challenge_codec=challenge_codec,
                challenge_mode=challenge_mode,
                video_idx=streak_i,
                video_id=video_id,
                video_url=video_url,
                details=details[:200],
            )
        )

    # Fallback video labels via ref_bpp when payload URLs were missing
    assign_fallback_video_labels(scores)
    return challenges, scores


def parse_log(path: str) -> tuple[list[Challenge], list[ScoreRow]]:
    with open(path, "r", errors="replace") as fh:
        return parse_log_lines(fh)


def assign_fallback_video_labels(scores: list[ScoreRow]) -> None:
    """If video_id is missing, label videos by challenge + ref_bpp order."""
    # Map (codec, mode, thr, ref_bpp) -> stable video index/name
    catalogs: dict[tuple, dict[float, int]] = defaultdict(dict)
    for r in scores:
        if r.ref_bpp is None:
            continue
        key = (r.codec or "?", r.mode or "?", r.vmaf_threshold)
        cat = catalogs[key]
        if r.ref_bpp not in cat:
            cat[r.ref_bpp] = len(cat) + 1

    for r in scores:
        if r.video_id:
            continue
        if r.ref_bpp is None:
            if r.video_idx is not None:
                r.video_id = f"video{r.video_idx}"
            continue
        key = (r.codec or "?", r.mode or "?", r.vmaf_threshold)
        idx = catalogs[key][r.ref_bpp]
        r.video_idx = r.video_idx or idx
        r.video_id = f"video{idx}_bpp{r.ref_bpp:.4f}"


def filter_scores(
    scores: Iterable[ScoreRow],
    *,
    codec: Optional[str],
    include_failures: bool,
    uids: Optional[Iterable[int]] = None,
    mode: Optional[str] = None,
    vmaf_threshold: Optional[float] = None,
) -> list[ScoreRow]:
    """Filter score rows by codec, mode, VMAF threshold, UID, and failure status."""
    uid_set: Optional[set[int]] = None
    if uids is not None:
        uid_set = {int(u) for u in uids}
        if not uid_set:
            uid_set = None
    mode_norm = mode.upper().strip() if mode else None
    if mode_norm in {"VBR", "ABR", "BITRATE"}:
        mode_norm = "VBR"
    elif mode_norm in {"CRF", "RC", "CQ"}:
        mode_norm = "CRF"
    thr_val: Optional[float] = None
    if vmaf_threshold is not None:
        thr_val = float(vmaf_threshold)

    out: list[ScoreRow] = []
    for s in scores:
        if uid_set is not None and s.uid not in uid_set:
            continue
        if codec and (s.codec or "").lower() != codec.lower():
            continue
        if mode_norm:
            row_mode = (s.mode or "").upper()
            if row_mode in {"ABR", "BITRATE"}:
                row_mode = "VBR"
            elif row_mode in {"RC", "CQ"}:
                row_mode = "CRF"
            if row_mode != mode_norm:
                continue
        if thr_val is not None and float(s.vmaf_threshold) != thr_val:
            continue
        if not include_failures and s.status in ("miner_failure", "missing_video"):
            if s.bitrate_mbps is None and s.vmaf is None:
                continue
        out.append(s)
    return out


def parse_uid_args(values: Optional[list[str]]) -> list[int]:
    """Parse ``--uid`` values: ``12``, ``12,34``, or repeated flags."""
    if not values:
        return []
    out: list[int] = []
    seen: set[int] = set()
    for raw in values:
        for part in str(raw).replace(" ", "").split(","):
            if not part:
                continue
            uid = int(part)
            if uid not in seen:
                seen.add(uid)
                out.append(uid)
    return out


def print_challenges(challenges: list[Challenge]) -> None:
    print("=" * 96)
    print("CHALLENGES")
    print("=" * 96)
    if not challenges:
        print("(none found)")
        return
    for c in challenges:
        br = f", bitrate={c.target_bitrate} Mbps" if c.target_bitrate is not None else ""
        print(
            f"  L{c.line}: codec={c.codec:<5} mode={c.mode:<4} thr={c.vmaf_threshold:.0f} "
            f"miners={c.n_miners} queries={c.n_queries}{br}"
        )


def _fmt_row(r: ScoreRow) -> str:
    vmaf = f"{r.vmaf:.2f}" if r.vmaf is not None else "N/A"
    ratio = f"{r.compression_ratio:.2f}" if r.compression_ratio is not None else "N/A"
    br = f"{r.bitrate_mbps:.2f}" if r.bitrate_mbps is not None else "-"
    bpp = f"{r.dist_bpp:.4f}" if r.dist_bpp is not None else "-"
    ref = f"{r.ref_bpp:.4f}" if r.ref_bpp is not None else "-"
    vid = r.video_id or f"video{r.video_idx or '?'}"
    vidx = r.video_idx if r.video_idx is not None else "-"
    return (
        f"  v{vidx:<2} {vid:<40.40} "
        f"NEG={r.vmaf_neg:7.2f} VMAF={vmaf:>7} "
        f"rate={r.compression_rate:7.4f} ratio={ratio:>7} "
        f"final={r.final:7.4f} ok={'Y' if r.success else 'N'} "
        f"status={r.status:<14} br={br:>6} dist_bpp={bpp} ref_bpp={ref}"
    )


def print_by_uid_per_video(rows: list[ScoreRow]) -> None:
    print("=" * 96)
    print(f"RESULTS BY UID → VIDEO ({len(rows)} rows)")
    print("=" * 96)

    # Group: challenge key -> uid -> rows (preserve video order)
    by_chal: dict[tuple, dict[int, list[ScoreRow]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r.codec or "?", r.mode or "?", r.vmaf_threshold)
        by_chal[key][r.uid].append(r)

    for chal_key in sorted(by_chal.keys(), key=lambda k: (k[0], k[1], k[2])):
        codec, mode, thr = chal_key
        uid_map = by_chal[chal_key]
        print(f"\n### codec={codec}  mode={mode}  thr={thr:.0f}  uids={len(uid_map)}")

        # Rank UIDs by mean final within this challenge
        ranked_uids = sorted(
            uid_map.keys(),
            key=lambda u: (-mean(x.final for x in uid_map[u]), u),
        )

        for uid in ranked_uids:
            rs = sorted(
                uid_map[uid],
                key=lambda x: (x.video_idx or 0, x.line),
            )
            mean_final = mean(x.final for x in rs)
            ok_n = sum(1 for x in rs if x.success)
            print(
                f"\nUID {uid}  videos={len(rs)}  mean_final={mean_final:.4f}  "
                f"success={ok_n}/{len(rs)}"
            )
            for r in rs:
                print(_fmt_row(r))


def summarize(rows: list[ScoreRow]) -> None:
    print("\n" + "=" * 96)
    print("PER-UID SUMMARY (by codec + mode + threshold)")
    print("=" * 96)

    groups: dict[tuple, list[ScoreRow]] = defaultdict(list)
    for r in rows:
        key = (r.codec or "?", r.mode or "?", r.vmaf_threshold)
        groups[key].append(r)

    for key in sorted(groups.keys(), key=lambda k: (k[0], k[1], k[2])):
        codec, mode, thr = key
        subset = groups[key]
        print(f"\n--- codec={codec} mode={mode} thr={thr:.0f}  (n={len(subset)}) ---")
        by_uid: dict[int, list[ScoreRow]] = defaultdict(list)
        for r in subset:
            by_uid[r.uid].append(r)

        print(
            f"{'uid':>4} {'n':>3} {'meanFinal':>10} {'maxFinal':>9} "
            f"{'meanNEG':>8} {'minNEG':>7} {'meanRate':>9} {'meanRatio':>9} "
            f"{'ok':>3} {'soft':>4} {'hard':>4}"
        )
        ranked = []
        for uid, rs in by_uid.items():
            finals = [x.final for x in rs]
            negs = [x.vmaf_neg for x in rs]
            rates = [x.compression_rate for x in rs]
            ratios = [x.compression_ratio for x in rs if x.compression_ratio is not None]
            ranked.append(
                (
                    mean(finals),
                    uid,
                    {
                        "n": len(rs),
                        "mean_final": mean(finals),
                        "max_final": max(finals),
                        "mean_neg": mean(negs),
                        "min_neg": min(negs),
                        "mean_rate": mean(rates),
                        "mean_ratio": mean(ratios) if ratios else None,
                        "ok": sum(1 for x in rs if x.success),
                        "soft": sum(1 for x in rs if x.status == "vmaf_soft"),
                        "hard": sum(1 for x in rs if x.status == "vmaf_hard_fail"),
                    },
                )
            )
        ranked.sort(key=lambda t: (-t[0], t[1]))
        for _, uid, s in ranked:
            mr = f"{s['mean_ratio']:.2f}" if s["mean_ratio"] is not None else "N/A"
            print(
                f"{uid:4d} {s['n']:3d} {s['mean_final']:10.4f} {s['max_final']:9.4f} "
                f"{s['mean_neg']:8.2f} {s['min_neg']:7.2f} {s['mean_rate']:9.4f} "
                f"{mr:>9} {s['ok']:3d} {s['soft']:4d} {s['hard']:4d}"
            )


def write_csv(path: str, rows: list[ScoreRow]) -> None:
    fields = [
        "line",
        "ts",
        "uid",
        "video_idx",
        "video_id",
        "video_url",
        "codec",
        "mode",
        "vmaf_threshold",
        "vmaf_neg",
        "vmaf",
        "compression_rate",
        "compression_ratio",
        "final",
        "success",
        "status",
        "resolution",
        "fps",
        "bitrate_mbps",
        "target_bitrate",
        "dist_bpp",
        "ref_bpp",
        "challenge_codec",
        "challenge_mode",
        "details",
    ]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            d = asdict(r)
            d.pop("extras", None)
            w.writerow({k: d.get(k) for k in fields})
    print(f"\nWrote CSV: {path} ({len(rows)} rows)")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("log", help="Path to validator output.log / files_output*.log")
    p.add_argument(
        "--codec",
        default=None,
        help="Filter by codec (hevc, av1, …). Default: all codecs found in the log.",
    )
    p.add_argument(
        "--uid",
        action="append",
        default=None,
        metavar="UID",
        help=(
            "Filter by miner UID. Repeatable, or comma-separated "
            "(e.g. --uid 112 --uid 66 or --uid 66,98,112)."
        ),
    )
    p.add_argument(
        "--failures",
        action="store_true",
        help="Also print miner_failure / missing_video rows (default: hide those).",
    )
    p.add_argument("--csv", default=None, help="Optional CSV output path")
    p.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip per-uid summary tables",
    )
    args = p.parse_args(argv)

    challenges, scores = parse_log(args.log)
    print_challenges(challenges)

    codecs_seen = sorted(
        {(c.codec) for c in challenges} | {(s.codec or "") for s in scores if s.codec}
    )
    print(f"\nCodecs seen: {', '.join(c for c in codecs_seen if c) or '(none)'}")

    uid_filter = parse_uid_args(args.uid)
    rows = filter_scores(
        scores,
        codec=args.codec,
        include_failures=args.failures,
        uids=uid_filter or None,
    )
    parts: list[str] = []
    if args.codec:
        parts.append(f"codec={args.codec.lower()}")
    if uid_filter:
        parts.append("uid=" + ",".join(str(u) for u in uid_filter))
    filt = "  ".join(parts) if parts else "all codecs/uids"
    print(f"Filter: {filt}  rows={len(rows)} (of {len(scores)} total score lines)")

    print_by_uid_per_video(rows)
    if not args.no_summary:
        summarize(rows)
    if args.csv:
        write_csv(args.csv, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
