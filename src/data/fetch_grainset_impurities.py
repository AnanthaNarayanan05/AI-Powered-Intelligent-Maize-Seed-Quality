"""Selectively downloads the GrainSet maize IMPURITY images used by Phase 4.

Why this exists rather than a plain download
--------------------------------------------
GrainSet's maize archive is 6.04 GB, and 95% of it is single-kernel maize images
this project already has plenty of. What it uniquely provides is the class this
project has none of anywhere on disk: ``7_IM`` -- Impurities, meaning non-grain
material photographed on the same rig as the kernels.

So instead of pulling 6 GB to keep 150 MB, this reads the archive's central
directory over HTTP Range requests, picks the members it wants, and fetches only
those byte ranges. figshare's S3 presigned URLs reject HEAD but honour Range,
which is why the archive size is a constant here rather than something probed.

Members are downloaded in offset-ordered runs so that a few dozen large requests
replace several thousand small ones. Each run is inflated in memory; nothing but
the selected images is ever written to disk.

What is selected, and why each part is needed
---------------------------------------------
``train/7_IM`` + ``test/7_IM`` (3,000 images)
    The impurities themselves. These are only ever used to *measure* the catch
    rate of the maize-identity gate. Nothing is calibrated on them -- a threshold
    tuned against the very objects it is scored on would report its own training
    accuracy.

``test/0_NOR`` and the six ``test`` defect classes (a capped sample of each)
    Maize controls from the same camera, lighting and background as the
    impurities. This matters more than it looks: if the only maize available for
    comparison came from other datasets, a gate could separate impurities from
    maize on imaging artefacts alone and appear to work while having learned
    nothing about maize. Same-rig maize removes that confound. The defect classes
    are included because a foreign-object flag that fires on damaged maize is
    worse than useless -- most of what a grader inspects is imperfect grain.

Attribution
-----------
GrainSet -- Zhao et al., Scientific Data 10, 748 (2023),
doi:10.1038/s41597-023-02660-8. Data: figshare doi:10.6084/m9.figshare.22987562.v2,
licensed CC BY 4.0. The original archive is not modified or redistributed; this
script records which members were taken and their SHA-256 digests.

Usage
-----
    python -m src.data.fetch_grainset_impurities
    python -m src.data.fetch_grainset_impurities --limit 8   # smoke test
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import struct
import sys
import urllib.request
import zipfile
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# figshare doi:10.6084/m9.figshare.22987562.v2, file "maize.zip".
ARCHIVE_URL = "https://ndownloader.figshare.com/files/40737164"
ARCHIVE_BYTES = 6_042_360_954

OUT_ROOT = "datasets/grainset_maize_impurities"
CACHE_DIR = "datasets/_downloads/grainset_cd"
MANIFEST = os.path.join(OUT_ROOT, "MANIFEST.json")

# Every impurity image, and a capped sample of same-rig maize as the control.
# The caps keep the download proportionate: test/0_NOR alone is 368 MB for 1,000
# images, and 300 is already more control images than impurities.
SELECT = {
    "maize/train/7_IM/": None,
    "maize/test/7_IM/": None,
    "maize/test/0_NOR/": 300,
    "maize/test/1_F&S/": 50,
    "maize/test/2_SD/": 50,
    "maize/test/3_MY/": 50,
    "maize/test/4_AP/": 50,
    "maize/test/5_BN/": 50,
    "maize/test/6_HD/": 50,
}

# Coalescing rules. A run is one GET, so the trade is request count against
# bytes pulled for members we skipped: allowing a gap wastes that gap's bytes.
RUN_SPAN_BYTES = 64 << 20   # hard cap on one request
RUN_GAP_BYTES = 1 << 20     # skip-over allowed between consecutive members


def _get(url: str, start: int, end: int, attempts: int = 4) -> bytes:
    """One inclusive byte range, retried -- a 6 GB source over a home connection
    drops occasionally and re-running the whole script is not a fix."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    last = None
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001 -- retry on anything transient
            last = e
    raise RuntimeError(f"range {start}-{end} failed after {attempts} attempts: {last}")


def _central_directory() -> bytes:
    """The archive's central directory, cached on disk after the first fetch."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "maize_zip_central_directory.bin")
    meta = os.path.join(CACHE_DIR, "maize_zip_cd_offset.txt")
    if os.path.exists(path) and os.path.exists(meta):
        return open(path, "rb").read()

    # 6 GB means ZIP64: the 32-bit EOCD holds sentinel values and the real
    # directory offset lives in the ZIP64 record the locator points at.
    tail = _get(ARCHIVE_URL, ARCHIVE_BYTES - 100_000, ARCHIVE_BYTES - 1)
    i = tail.rfind(b"PK\x06\x07")
    if i < 0:
        raise RuntimeError("no ZIP64 end-of-central-directory locator found")
    _, z64_off, _ = struct.unpack("<IQI", tail[i + 4:i + 20])
    rec = _get(ARCHIVE_URL, z64_off, z64_off + 55)
    sig, _, _, _, _, _, _, _, cd_size, cd_off = struct.unpack("<IQHHIIQQQQ", rec[:56])
    if sig != 0x06064B50:
        raise RuntimeError(f"unexpected ZIP64 EOCD signature {sig:#x}")

    data = _get(ARCHIVE_URL, cd_off, ARCHIVE_BYTES - 1)
    open(path, "wb").write(data)
    open(meta, "w").write(str(cd_off))
    return data


class _DirectoryOnlyFile(io.RawIOBase):
    """Enough of a file for ``zipfile`` to parse the directory and nothing else.

    Reads that land in the cached directory are served from memory; anything
    outside it raises, so a stray attempt to stream a member through this object
    fails loudly instead of quietly issuing thousands of tiny HTTP reads.
    """

    def __init__(self, size: int, start: int, data: bytes):
        self.size, self.start, self.data, self.pos = size, start, data, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if self.pos < self.start:
            raise RuntimeError(f"read at {self.pos} is outside the cached directory")
        chunk = self.data[self.pos - self.start: self.pos - self.start + n]
        self.pos += len(chunk)
        return chunk

    def readinto(self, b):
        chunk = self.read(len(b))
        b[: len(chunk)] = chunk
        return len(chunk)


def _select(infos, limit=None):
    """Members to download, honouring the per-prefix caps.

    Sorted by name before capping so a re-run picks the same sample; the archive's
    directory order is not guaranteed to be stable across mirrors.
    """
    by_prefix = defaultdict(list)
    for info in infos:
        if info.is_dir():
            continue
        for prefix in SELECT:
            if info.filename.startswith(prefix):
                by_prefix[prefix].append(info)
                break

    chosen = []
    for prefix, cap in SELECT.items():
        members = sorted(by_prefix.get(prefix, []), key=lambda i: i.filename)
        if not members:
            raise RuntimeError(f"{prefix} is missing from the archive")
        if cap is not None:
            members = members[:cap]
        if limit is not None:
            members = members[:limit]
        chosen.extend(members)
    return chosen


def _runs(members):
    """Groups offset-ordered members into single-request byte ranges."""
    members = sorted(members, key=lambda i: i.header_offset)
    out, run = [], [members[0]]
    for info in members[1:]:
        prev = run[-1]
        gap = info.header_offset - (prev.header_offset + prev.compress_size)
        span = info.header_offset + info.compress_size + 4096 - run[0].header_offset
        if gap > RUN_GAP_BYTES or span > RUN_SPAN_BYTES:
            out.append(run)
            run = [info]
        else:
            run.append(info)
    out.append(run)
    return out


def _extract(buf: bytes, base: int, info) -> bytes:
    """Inflates one member out of an already-downloaded run."""
    head = buf[info.header_offset - base: info.header_offset - base + 30]
    if head[:4] != b"PK\x03\x04":
        raise RuntimeError(f"{info.filename}: no local file header at its offset")
    name_len, extra_len = struct.unpack("<HH", head[26:30])
    start = info.header_offset - base + 30 + name_len + extra_len
    raw = buf[start: start + info.compress_size]
    if len(raw) != info.compress_size:
        raise RuntimeError(f"{info.filename}: short read ({len(raw)}/{info.compress_size})")
    if info.compress_type == zipfile.ZIP_STORED:
        return raw
    import zlib
    return zlib.decompressobj(-zlib.MAX_WBITS).decompress(raw)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap every class at N images (smoke test)")
    ap.add_argument("--out", default=OUT_ROOT)
    args = ap.parse_args()

    cd = _central_directory()
    cd_off = int(open(os.path.join(CACHE_DIR, "maize_zip_cd_offset.txt")).read())
    zf = zipfile.ZipFile(_DirectoryOnlyFile(ARCHIVE_BYTES, cd_off, cd))
    members = _select(zf.infolist(), args.limit)

    total = sum(i.compress_size for i in members)
    print(f"{len(members)} members selected, {total / 1e6:.1f} MB compressed")

    records, done, fetched = [], 0, 0
    for run in _runs(members):
        base = run[0].header_offset
        end = run[-1].header_offset + run[-1].compress_size + 4096
        buf = _get(ARCHIVE_URL, base, min(end, ARCHIVE_BYTES - 1))
        fetched += len(buf)
        for info in run:
            data = _extract(buf, base, info)
            if len(data) != info.file_size:
                raise RuntimeError(f"{info.filename}: inflated to {len(data)}, "
                                   f"directory says {info.file_size}")
            dest = os.path.join(args.out, info.filename[len("maize/"):])
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(data)
            records.append({
                "member": info.filename,
                "path": dest.replace(os.sep, "/"),
                "bytes": info.file_size,
                "sha256": hashlib.sha256(data).hexdigest(),
            })
            done += 1
        print(f"  {done}/{len(members)} written, {fetched / 1e6:.1f} MB fetched")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "MANIFEST.json"), "w") as f:
        json.dump({
            "source": "GrainSet maize (figshare doi:10.6084/m9.figshare.22987562.v2)",
            "paper": "Zhao et al., Scientific Data 10, 748 (2023), "
                     "doi:10.1038/s41597-023-02660-8",
            "license": "CC BY 4.0",
            "archive_url": ARCHIVE_URL,
            "archive_bytes": ARCHIVE_BYTES,
            "selection": {k: v for k, v in SELECT.items()},
            "limit": args.limit,
            "count": len(records),
            "bytes_fetched": fetched,
            "files": records,
        }, f, indent=2)
    print(f"manifest -> {os.path.join(args.out, 'MANIFEST.json')}")


if __name__ == "__main__":
    main()
