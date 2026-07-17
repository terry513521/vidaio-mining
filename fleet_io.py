"""Deadline-aware HTTP transfers for the fleet SLA runner."""

from __future__ import annotations

import http.client
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlsplit


@dataclass(frozen=True)
class TransferResult:
    ok: bool
    path: str
    bytes_transferred: int
    elapsed_sec: float
    error: str = ""
    url: str = ""


def seconds_left(deadline: float) -> float:
    """Remaining seconds for a monotonic absolute deadline."""
    return max(0.0, float(deadline) - time.monotonic())


def _timeout(deadline: float, *, minimum: float = 0.1) -> float:
    left = seconds_left(deadline)
    if left <= 0:
        raise TimeoutError("end-to-end deadline exhausted")
    return max(minimum, left)


def download_to_path(
    url: str,
    destination: str | Path,
    *,
    deadline: float,
    chunk_size: int = 1024 * 1024,
) -> TransferResult:
    """Stream an HTTP(S) object to a temporary file, then atomically publish it."""
    started = time.monotonic()
    dest = Path(destination)
    part = dest.with_suffix(dest.suffix + ".part")
    transferred = 0
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            url, headers={"User-Agent": "vidaio-fleet/1.0"}
        )
        with urllib.request.urlopen(request, timeout=_timeout(deadline)) as response:
            with part.open("wb") as output:
                while True:
                    if seconds_left(deadline) <= 0:
                        raise TimeoutError("download exceeded end-to-end deadline")
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    output.write(chunk)
                    transferred += len(chunk)
        part.replace(dest)
        return TransferResult(
            True, str(dest), transferred, time.monotonic() - started, url=url
        )
    except Exception as exc:  # network errors vary by Python/platform
        part.unlink(missing_ok=True)
        return TransferResult(
            False,
            str(dest),
            transferred,
            time.monotonic() - started,
            str(exc),
            url=url,
        )


def upload_presigned_put(
    source: str | Path,
    url: str,
    *,
    deadline: float,
    content_type: str = "video/mp4",
    chunk_size: int = 1024 * 1024,
) -> TransferResult:
    """Stream a file to a presigned HTTP(S) PUT URL with Content-Length."""
    started = time.monotonic()
    path = Path(source)
    transferred = 0
    connection: http.client.HTTPConnection | None = None
    try:
        size = path.stat().st_size
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("upload_url must be HTTP(S)")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        timeout = _timeout(deadline)
        conn_type = (
            http.client.HTTPSConnection
            if parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = conn_type(parsed.hostname, port, timeout=timeout)
        target = parsed.path or "/"
        if parsed.query:
            target += "?" + parsed.query
        connection.putrequest("PUT", target)
        connection.putheader("Content-Length", str(size))
        connection.putheader("Content-Type", content_type)
        connection.putheader("User-Agent", "vidaio-fleet/1.0")
        connection.endheaders()

        with path.open("rb") as source_file:
            while True:
                if seconds_left(deadline) <= 0:
                    raise TimeoutError("upload exceeded end-to-end deadline")
                chunk = source_file.read(chunk_size)
                if not chunk:
                    break
                connection.send(chunk)
                transferred += len(chunk)
        response = connection.getresponse()
        response.read()
        if not 200 <= response.status < 300:
            raise RuntimeError(f"upload returned HTTP {response.status} {response.reason}")
        return TransferResult(
            True, str(path), transferred, time.monotonic() - started, url=url
        )
    except Exception as exc:
        return TransferResult(
            False,
            str(path),
            transferred,
            time.monotonic() - started,
            str(exc),
            url=url,
        )
    finally:
        if connection is not None:
            connection.close()


def download_many(
    items: Iterable[tuple[str, str]],
    *,
    deadline: float,
    max_workers: int = 5,
) -> list[TransferResult]:
    """Download multiple (url, destination) pairs concurrently under one deadline."""
    jobs = list(items)
    if not jobs:
        return []
    workers = max(1, min(int(max_workers), len(jobs)))
    results: list[Optional[TransferResult]] = [None] * len(jobs)

    def _one(index: int, url: str, dest: str) -> tuple[int, TransferResult]:
        return index, download_to_path(url, dest, deadline=deadline)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_one, index, url, dest) for index, (url, dest) in enumerate(jobs)
        ]
        for future in as_completed(futures):
            index, result = future.result()
            results[index] = result
    return [
        r if r is not None else TransferResult(False, "", 0, 0.0, "missing")
        for r in results
    ]
