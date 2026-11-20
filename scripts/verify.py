"""make verify: re-run a fast, fixed subset (Isolation Forest, first 3 LOPO folds) and compare to committed results."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

TOL = {"far": 0.02, "pr_auc": 0.02}


def main() -> int:
    ref_path = Path("results/raw/lopo/B1_seed0.json")
    if not ref_path.exists():
        print("no committed reference results; run `make eval` first")
        return 1
    subprocess.run([sys.executable, "scripts/run_eval.py", "--proto", "lopo", "--scorers", "B1", "--seed", "0", "--quick"], check=True)
    new = json.loads(Path("results/raw/lopo/B1_seed0_quick.json").read_text())
    ref = {f["fold"]: f for f in json.loads(ref_path.read_text())["folds"]}
    failures = 0
    for f in new["folds"]:
        r = ref[f["fold"]]
        d_pr = abs(f["threshold_free"]["pr_auc"] - r["threshold_free"]["pr_auc"]) if r["threshold_free"]["pr_auc"] == r["threshold_free"]["pr_auc"] else 0.0
        checks = [("pr_auc", d_pr)]
        for st, v in f["by_alpha"]["0.05"].items():
            checks.append((f"far {st}", abs(v["far"] - r["by_alpha"]["0.05"][st]["far"])))
        for name, d in checks:
            ok = d <= TOL["far" if name.startswith("far") else "pr_auc"]
            failures += not ok
            print(f"{'OK  ' if ok else 'FAIL'} {f['fold']:<12} {name:<18} |diff| = {d:.4f}")
    print("verify passed" if failures == 0 else f"verify failed: {failures} checks outside tolerance")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
