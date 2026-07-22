"""Extract validator reference video URLs from a Vidaio validator log and download them."""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlparse

URL_RE = re.compile(
    r"https://s3\.us-east-005\.backblazeb2\.com/vidaiosubnet/[^\s\"'\\,\]]+"
)


def extract_urls(log_text: str) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for match in URL_RE.finditer(log_text):
        url = match.group(0).rstrip("',]\\")
        key = url.split("?", 1)[0]
        if key in seen:
            continue
        seen.add(key)
        urls.append(url)
    return urls


def object_name(url: str) -> str:
    return Path(unquote(urlparse(url).path)).name


def download_file(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "vidaio-validation-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        total = resp.headers.get("Content-Length")
        total_bytes = int(total) if total else None
        downloaded = 0
        with open(tmp, "wb") as out:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    pct = 100.0 * downloaded / total_bytes
                    print(
                        f"\r  {pct:5.1f}%  {downloaded/1e6:7.1f}/{total_bytes/1e6:.1f} MB",
                        end="",
                        flush=True,
                    )
                else:
                    print(f"\r  {downloaded/1e6:.1f} MB", end="", flush=True)
    tmp.replace(dest)
    print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        type=Path,
        default=Path(r"C:\Users\com\Downloads\files_output.log"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "raw videos",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Download at most N videos (0 = all)",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only write URL list, do not download",
    )
    args = parser.parse_args()

    log_text = args.log.read_text(encoding="utf-8", errors="replace")
    urls = extract_urls(log_text)
    if args.limit > 0:
        urls = urls[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = args.out_dir / "validation_urls.txt"
    json_path = args.out_dir / "validation_urls.json"

    items = [{"url": u, "object": object_name(u)} for u in urls]
    txt_path.write_text("\n".join(urls) + ("\n" if urls else ""), encoding="utf-8")
    json_path.write_text(json.dumps(items, indent=2), encoding="utf-8")

    print(f"Found {len(urls)} unique validator reference URLs")
    print(f"Wrote {txt_path}")
    print(f"Wrote {json_path}")

    if args.extract_only:
        return 0

    ok = 0
    fail = 0
    for i, item in enumerate(items, start=1):
        dest = args.out_dir / item["object"]
        print(f"[{i}/{len(items)}] {item['object']}")
        if dest.is_file() and dest.stat().st_size > 0:
            print(f"  already exists: {dest}")
            ok += 1
            continue
        try:
            download_file(item["url"], dest)
            print(f"  saved: {dest}")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc}")
            fail += 1

    print(f"\nDone: ok={ok} fail={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
