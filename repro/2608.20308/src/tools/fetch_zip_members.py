#!/usr/bin/env python
"""Download selected members of a remote ZIP via HTTP Range requests.

Used to pull the FreiHAND annotation JSONs (~30 MB) out of the 3.9 GB official
archive when the mirror is too slow for a full download.

    python fetch_zip_members.py URL OUT_DIR member_substring [...]
"""
from __future__ import annotations

import os
import struct
import sys
import time
import urllib.request
import zlib

UA = {"User-Agent": "Mozilla/5.0"}


def http_range(url: str, start: int, end: int, retries: int = 20) -> bytes:
    """Fetch bytes [start, end] inclusive, resuming on failures."""
    buf = bytearray()
    pos = start
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={pos}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    buf += chunk
                    pos += len(chunk)
            if pos > end:
                return bytes(buf)
        except Exception as e:  # noqa: BLE001
            print(f"  range {pos}-{end} attempt {attempt}: {e}", flush=True)
            time.sleep(5)
    raise RuntimeError("range download failed")


def content_length(url: str) -> int:
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def central_directory(url: str, size: int) -> list[dict]:
    tail = http_range(url, max(0, size - 1 << 16), size - 1)
    i = tail.rfind(b"PK\x05\x06")
    assert i >= 0, "EOCD not found"
    n, cd_size, cd_off = struct.unpack("<HII", tail[i + 10:i + 20])
    if n == 0xFFFF or cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
        j = tail.rfind(b"PK\x06\x06")
        assert j >= 0, "ZIP64 EOCD not found"
        n, _, cd_size, cd_off = struct.unpack("<QQQQ", tail[j + 24:j + 56])
    print(f"central directory: {n} entries, {cd_size / 1e6:.1f} MB at {cd_off}", flush=True)
    cd = http_range(url, cd_off, cd_off + cd_size - 1)
    entries, p = [], 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        (method, csize, usize, nlen, xlen, clen) = struct.unpack("<H", cd[p + 10:p + 12]) + struct.unpack(
            "<II", cd[p + 20:p + 28]) + struct.unpack("<HHH", cd[p + 28:p + 34])
        off = struct.unpack("<I", cd[p + 42:p + 46])[0]
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
        if csize == 0xFFFFFFFF or usize == 0xFFFFFFFF or off == 0xFFFFFFFF:      # ZIP64 extra
            q = 0
            while q + 4 <= len(extra):
                hid, hlen = struct.unpack("<HH", extra[q:q + 4])
                if hid == 1:
                    vals, r = [], q + 4
                    for cur in (usize, csize, off):
                        if cur == 0xFFFFFFFF:
                            vals.append(struct.unpack("<Q", extra[r:r + 8])[0])
                            r += 8
                        else:
                            vals.append(cur)
                    usize, csize, off = vals
                    break
                q += 4 + hlen
        entries.append({"name": name, "method": method, "csize": csize, "usize": usize, "off": off})
        p += 46 + nlen + xlen + clen
    return entries


def fetch_member(url: str, e: dict, out_path: str) -> None:
    head = http_range(url, e["off"], e["off"] + 29)
    nlen, xlen = struct.unpack("<HH", head[26:30])
    start = e["off"] + 30 + nlen + xlen
    print(f"fetching {e['name']} ({e['csize'] / 1e6:.1f} MB compressed)", flush=True)
    data = http_range(url, start, start + e["csize"] - 1)
    if e["method"] == 8:
        data = zlib.decompress(data, -15)
    elif e["method"] != 0:
        raise RuntimeError(f"unsupported compression {e['method']}")
    assert len(data) == e["usize"], (len(data), e["usize"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(data)
    print(f"  -> {out_path} ({len(data) / 1e6:.1f} MB)", flush=True)


def main():
    url, out_dir, wanted = sys.argv[1], sys.argv[2], sys.argv[3:]
    size = content_length(url)
    entries = central_directory(url, size)
    for e in entries:
        if any(w in e["name"] for w in wanted) and not e["name"].endswith("/"):
            dst = os.path.join(out_dir, os.path.basename(e["name"]))
            if os.path.isfile(dst) and os.path.getsize(dst) == e["usize"]:
                print(f"have {dst}")
                continue
            fetch_member(url, e, dst)
    print("done", flush=True)


if __name__ == "__main__":
    main()
