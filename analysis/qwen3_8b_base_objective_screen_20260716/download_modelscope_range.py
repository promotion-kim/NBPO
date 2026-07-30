#!/usr/bin/env python3
"""Resume one ModelScope LFS object with parallel HTTP byte ranges and verify SHA-256."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import re
import socket
from pathlib import Path

import requests


_original_getaddrinfo = socket.getaddrinfo


def ipv4_getaddrinfo(*args, **kwargs):
    values = [value for value in _original_getaddrinfo(*args, **kwargs) if value[0] == socket.AF_INET]
    if not values:
        raise OSError("no IPv4 address")
    return values


socket.getaddrinfo = ipv4_getaddrinfo


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    probe = requests.get(args.url, headers={"Range": "bytes=0-0"}, stream=True, timeout=60)
    probe.raise_for_status()
    match = re.fullmatch(r"bytes 0-0/(\d+)", probe.headers.get("content-range", ""))
    if not match:
        raise RuntimeError(f"missing total size: {probe.headers}")
    total = int(match.group(1))
    expected = probe.headers.get("x-linked-etag", "").strip('"').lower()
    probe.close()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise RuntimeError(f"expected a SHA-256 linked etag, got {expected!r}")
    start = args.partial.stat().st_size if args.partial.exists() else 0
    if start >= total:
        start = 0
    remaining = total - start
    chunk = (remaining + args.workers - 1) // args.workers
    ranges = [(index, begin, min(total - 1, begin + chunk - 1)) for index, begin in enumerate(range(start, total, chunk))]
    parts = [args.output.with_suffix(args.output.suffix + f".range{index:02d}") for index, _, _ in ranges]

    def fetch(item):
        index, begin, end = item
        path = parts[index]
        expected_bytes = end - begin + 1
        if path.exists() and path.stat().st_size == expected_bytes:
            return
        response = requests.get(args.url, headers={"Range": f"bytes={begin}-{end}"}, stream=True, timeout=120)
        response.raise_for_status()
        with path.open("wb") as handle:
            for block in response.iter_content(8 * 1024 * 1024):
                if block:
                    handle.write(block)
        if path.stat().st_size != expected_bytes:
            raise RuntimeError(f"range {begin}-{end}: {path.stat().st_size} != {expected_bytes}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fetch, ranges))
    assembled = args.output.with_suffix(args.output.suffix + ".assembling")
    with assembled.open("wb") as target:
        if start:
            with args.partial.open("rb") as source:
                while block := source.read(16 * 1024 * 1024):
                    target.write(block)
        for path in parts:
            with path.open("rb") as source:
                while block := source.read(16 * 1024 * 1024):
                    target.write(block)
    if assembled.stat().st_size != total:
        raise RuntimeError(f"assembled bytes {assembled.stat().st_size} != {total}")
    actual = sha256(assembled)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch: {actual} != {expected}")
    os.replace(assembled, args.output)
    for path in parts:
        path.unlink()
    if args.partial.exists():
        args.partial.unlink()
    print(f"complete bytes={total} sha256={actual}")


if __name__ == "__main__":
    main()
