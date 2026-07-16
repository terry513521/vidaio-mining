"""
Download Vidaio eval videos from http://194.163.164.157:8090/

Examples:
  python download_eval_videos.py --resolution 4k --count 1
  python download_eval_videos.py --resolution 8k --count 1
  python download_eval_videos.py --resolution both --count 1
  python download_eval_videos.py --resolution 4k --index 0
  python download_eval_videos.py --list 4k
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "http://194.163.164.157:8090"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "eval_samples"


def fetch_json(url: str, timeout: float = 30.0) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": "vidaio-eval-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_manifest(resolution: str) -> list[dict]:
    url = f"{BASE_URL}/{resolution}/manifest.json"
    data = fetch_json(url)
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected manifest format from {url}")
    return data


def download_file(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    req = urllib.request.Request(url, headers={"User-Agent": "vidaio-eval-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = resp.headers.get("Content-Length")
        total_bytes = int(total) if total else None
        downloaded = 0

        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    pct = 100.0 * downloaded / total_bytes
                    mb = downloaded / (1024 * 1024)
                    total_mb = total_bytes / (1024 * 1024)
                    print(f"\r  {pct:5.1f}%  {mb:7.1f}/{total_mb:.1f} MB", end="", flush=True)
                else:
                    print(f"\r  {downloaded / (1024 * 1024):.1f} MB", end="", flush=True)

    tmp.replace(dest)
    print()


def resolve_entries(resolution: str, count: int | None, index: int | None) -> list[dict]:
    items = list_manifest(resolution)
    if not items:
        raise RuntimeError(f"No videos in {resolution} manifest")

    if index is not None:
        if index < 0 or index >= len(items):
            raise IndexError(f"{resolution}: index {index} out of range 0..{len(items) - 1}")
        return [items[index]]

    n = count if count is not None else 1
    if n < 1:
        raise ValueError("--count must be >= 1")
    return items[:n]


def download_resolution(
    resolution: str,
    out_dir: Path,
    count: int | None,
    index: int | None,
) -> list[Path]:
    entries = resolve_entries(resolution, count, index)
    saved: list[Path] = []

    for i, entry in enumerate(entries):
        filename = entry["file"]
        url = f"{BASE_URL}/{resolution}/{filename}"
        dest = out_dir / f"{resolution}_{filename}"

        print(f"[{resolution}] ({i + 1}/{len(entries)}) {filename}")
        print(f"  {entry.get('resolution')}  {entry.get('duration')}s  ~{entry.get('mb')} MB")
        print(f"  {url}")

        if dest.is_file() and dest.stat().st_size > 0:
            print(f"  already exists: {dest}")
            saved.append(dest)
            continue

        download_file(url, dest)
        print(f"  saved: {dest}")
        saved.append(dest)

    return saved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download Vidaio eval videos")
    parser.add_argument(
        "--resolution",
        choices=["4k", "8k", "both"],
        default="both",
        help="Which resolution set to download (default: both)",
    )
    parser.add_argument("--count", type=int, default=None, help="How many videos per resolution")
    parser.add_argument("--index", type=int, default=None, help="Download a specific manifest index")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument(
        "--list",
        choices=["4k", "8k"],
        default=None,
        help="Only list manifest entries, do not download",
    )
    args = parser.parse_args(argv)

    try:
        if args.list:
            items = list_manifest(args.list)
            print(f"{args.list} manifest ({len(items)} videos):\n")
            for i, item in enumerate(items):
                print(
                    f"[{i:02d}] {item['file']}  "
                    f"{item.get('resolution')}  {item.get('duration')}s  {item.get('mb')} MB"
                )
            return 0

        if args.count is not None and args.index is not None:
            print("Use either --count or --index, not both", file=sys.stderr)
            return 2

        resolutions = ["4k", "8k"] if args.resolution == "both" else [args.resolution]
        saved_all: list[Path] = []

        for resolution in resolutions:
            saved_all.extend(
                download_resolution(
                    resolution,
                    args.out_dir,
                    count=args.count if args.count is not None else 1,
                    index=args.index,
                )
            )

        print("\nDone:")
        for path in saved_all:
            print(f"  {path}")
        return 0

    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        print(f"Network error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
