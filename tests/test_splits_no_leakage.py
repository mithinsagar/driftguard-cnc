import json
from pathlib import Path

import pytest

from driftguard.data.splits import ROLES, check_no_leakage

SPLITS = Path(__file__).resolve().parents[1] / "splits"


@pytest.mark.skipif(not (SPLITS / "lopo.json").exists(), reason="run `make data` first")
@pytest.mark.parametrize("proto", ["lopo", "lomo", "time"])
def test_no_file_in_two_roles(proto):
    for fold in json.loads((SPLITS / f"{proto}.json").read_text()):
        check_no_leakage(fold)


@pytest.mark.skipif(not (SPLITS / "lopo.json").exists(), reason="run `make data` first")
@pytest.mark.parametrize("proto", ["lopo", "lomo"])
def test_every_file_is_tested_exactly_once(proto):
    folds = json.loads((SPLITS / f"{proto}.json").read_text())
    tested = [f for fold in folds for f in fold["test"]]
    assert len(tested) == len(set(tested)) == 1702


@pytest.mark.skipif(not (SPLITS / "lopo.json").exists(), reason="run `make data` first")
def test_lopo_test_process_absent_from_other_roles():
    for fold in json.loads((SPLITS / "lopo.json").read_text()):
        op = fold["name"].split("_")[1]
        for role in ROLES[:-1]:
            assert not any(f"_{op}_" in fid for fid in fold[role]), (fold["name"], role)


def test_check_no_leakage_catches_overlap():
    bad = {"name": "x", "train_good": ["a"], "val_good": [], "cal_good": ["a"], "val_bad": [], "test": []}
    with pytest.raises(AssertionError):
        check_no_leakage(bad)
