"""Tests for deadline-aware fleet HTTP transfers."""

from __future__ import annotations

import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fleet_io import download_many, download_to_path, upload_presigned_put


class _Response(io.BytesIO):
    status = 200
    reason = "OK"

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class _PutConnection:
    instances: list["_PutConnection"] = []

    def __init__(self, *_args, **_kwargs):
        self.body = bytearray()
        self.headers: dict[str, str] = {}
        self.__class__.instances.append(self)

    def putrequest(self, method: str, target: str) -> None:
        self.method = method
        self.target = target

    def putheader(self, name: str, value: str) -> None:
        self.headers[name] = value

    def endheaders(self) -> None:
        pass

    def send(self, chunk: bytes) -> None:
        self.body.extend(chunk)

    def getresponse(self) -> _Response:
        return _Response(b"")

    def close(self) -> None:
        pass


class FleetIoTests(unittest.TestCase):
    def test_download_streams_and_atomically_publishes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "input.mp4"
            with patch("fleet_io.urllib.request.urlopen", return_value=_Response(b"video")):
                result = download_to_path(
                    "https://download.invalid/video",
                    output,
                    deadline=time.monotonic() + 2,
                    chunk_size=2,
                )
            self.assertTrue(result.ok)
            self.assertEqual(output.read_bytes(), b"video")
            self.assertFalse(output.with_suffix(".mp4.part").exists())

    def test_presigned_put_streams_content_length(self) -> None:
        _PutConnection.instances.clear()
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "output.mp4"
            source.write_bytes(b"encoded-video")
            with patch("fleet_io.http.client.HTTPSConnection", _PutConnection):
                result = upload_presigned_put(
                    source,
                    "https://upload.invalid/object?signature=x",
                    deadline=time.monotonic() + 2,
                    chunk_size=3,
                )
            self.assertTrue(result.ok)
            connection = _PutConnection.instances[-1]
            self.assertEqual(connection.method, "PUT")
            self.assertEqual(connection.target, "/object?signature=x")
            self.assertEqual(connection.headers["Content-Length"], str(source.stat().st_size))
            self.assertEqual(bytes(connection.body), source.read_bytes())

    def test_expired_deadline_fails_without_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "input.mp4"
            result = download_to_path(
                "https://download.invalid/video",
                output,
                deadline=time.monotonic() - 1,
            )
            self.assertFalse(result.ok)
            self.assertFalse(output.exists())

    def test_download_many_keeps_order(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            outs = [Path(td) / f"{i}.mp4" for i in range(3)]

            def fake_urlopen(request, timeout=None):
                name = Path(request.full_url).name
                return _Response(name.encode())

            with patch("fleet_io.urllib.request.urlopen", side_effect=fake_urlopen):
                results = download_many(
                    [
                        ("https://download.invalid/a.mp4", str(outs[0])),
                        ("https://download.invalid/b.mp4", str(outs[1])),
                        ("https://download.invalid/c.mp4", str(outs[2])),
                    ],
                    deadline=time.monotonic() + 2,
                    max_workers=3,
                )
            self.assertTrue(all(r.ok for r in results))
            self.assertEqual(outs[0].read_bytes(), b"a.mp4")
            self.assertEqual(outs[2].read_bytes(), b"c.mp4")


if __name__ == "__main__":
    unittest.main()
