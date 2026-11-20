"""make data (step 1): fetch the Bosch CNC Machining dataset from its official GitHub repository.

Default: list the repository tree with a blob-less git clone (no API rate limits), download every .h5 file in parallel
from raw.githubusercontent.com, and verify each file against its git blob SHA-1. `--git` does a plain full clone instead.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "https://github.com/boschresearch/CNC_Machining.git"
RAW = "https://raw.githubusercontent.com/boschresearch/CNC_Machining/main/"
DEST = Path("data/raw/CNC_Machining")


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def list_tree() -> list[tuple[str, str]]:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1", REPO, tmp], check=True,
                       capture_output=True)
        out = subprocess.run(["git", "-C", tmp, "ls-tree", "-r", "HEAD"], check=True, capture_output=True, text=True).stdout
    items = []
    for line in out.splitlines():
        meta, path = line.split("\t", 1)
        _, kind, sha = meta.split()
        if kind == "blob" and (path.endswith(".h5") or path in ("README.md", "utils/License")):
            items.append((path, sha))
    return items


def fetch(item: tuple[str, str], retries: int = 4) -> str:
    path, sha = item
    dest = DEST / path
    if dest.exists() and git_blob_sha(dest.read_bytes()) == sha:
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(RAW + path, timeout=120) as r:
                data = r.read()
            if git_blob_sha(data) != sha:
                raise OSError("checksum mismatch")
            dest.write_bytes(data)
            return "ok"
        except OSError:
            if attempt == retries - 1:
                raise
    return "failed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--git", action="store_true", help="plain git clone instead of parallel verified download")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.git:
        subprocess.run(["git", "clone", "--depth", "1", REPO, str(DEST)], check=True)
        return
    items = list_tree()
    print(f"{len(items)} files listed from {REPO}", flush=True)
    done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for status in ex.map(fetch, items):
            done += 1
            if done % 200 == 0 or done == len(items):
                print(f"  {done}/{len(items)} ({status})", flush=True)
    n_h5 = sum(1 for p, _ in items if p.endswith(".h5"))
    print(f"verified {n_h5} .h5 files against git blob SHA-1")


if __name__ == "__main__":
    sys.exit(main())
