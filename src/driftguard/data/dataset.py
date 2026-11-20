"""Bosch CNC Machining dataset: manifest, verification and loading.

Source: https://github.com/boschresearch/CNC_Machining (data CC-BY-4.0).
Layout: data/<M01|M02|M03>/<OP00..OP14>/<good|bad>/<M02_Aug_2019_OP03_000.h5>, each file an (n, 3) array at 2 kHz.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SAMPLE_RATE = 2000
EXPECTED = {"files": 1702, "good": 1632, "bad": 70}
DATA_ROOT = Path("data/raw/CNC_Machining/data")
NAME_RE = re.compile(r"^(M0\d)_([A-Z][a-z]{2})_(\d{4})_(OP\d{2})_(\d{3})\.h5$")
MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


@dataclass(frozen=True)
class Record:
    path: str
    machine: str
    process: str
    label: str
    timeframe: str
    year: int
    month: int
    index: int


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_signal(path: str | Path) -> np.ndarray:
    """Return the (n, 3) float32 vibration array (X, Y, Z) stored in one .h5 file."""
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        arr = f[keys[0]][()]
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{path}: expected (n, 3), got {arr.shape}")
    return arr


def build_manifest(root: Path = DATA_ROOT, with_hash: bool = True) -> pd.DataFrame:
    rows = []
    for p in sorted(root.glob("M0*/OP*/*/*.h5")):
        machine, process, label = p.parts[-4], p.parts[-3], p.parts[-2]
        m = NAME_RE.match(p.name)
        if not m:
            raise ValueError(f"unexpected file name: {p.name}")
        mm, mon, year, op, idx = m.groups()
        if mm != machine or op != process:
            raise ValueError(f"name does not match folders: {p}")
        with h5py.File(p, "r") as f:
            n = f[list(f.keys())[0]].shape[0]
        rows.append({
            "path": str(p), "machine": machine, "process": process, "label": label,
            "timeframe": f"{mon}_{year}", "year": int(year), "month": MONTHS[mon], "index": int(idx),
            "n_samples": int(n), "seconds": n / SAMPLE_RATE,
            "sha256": _sha256(p) if with_hash else "",
        })
    df = pd.DataFrame(rows)
    df["file_id"] = df["path"].map(lambda s: Path(s).stem + "_" + Path(s).parts[-2])
    df["t_order"] = df["year"] * 100 + df["month"]
    return df


def verify(df: pd.DataFrame) -> dict:
    counts = {"files": len(df), "good": int((df.label == "good").sum()), "bad": int((df.label == "bad").sum())}
    if counts != EXPECTED:
        raise SystemExit(f"Dataset counts changed: expected {EXPECTED}, found {counts}. Regenerate data card and splits deliberately.")
    if df["file_id"].duplicated().any():
        raise SystemExit("Duplicate file ids in manifest")
    return counts
