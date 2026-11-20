"""Headline figure: target vs achieved false-alarm rate and recall vs false-alarm rate on real streams."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SHOW = {"P0_frozen": ("frozen threshold (split conformal)", "#8a99a8"), "P1_naive": ("naive self-labelling", "#e08a1e"),
        "P4_w1acas": ("IBM W1-ACAS, label-free (ICLR 2026)", "#c0392b"),
        "P5_delayed_aci_d5": ("delayed-label + ACI (this work)", "#111111")}


def main(proto="lomo", scorer="B2", scen="burst"):
    g = pd.DataFrame(json.loads(Path(f"results/stream_sweep_{proto}_{scorer}_seed0_{scen}.json").read_text()))
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    for pol, (label, color) in SHOW.items():
        s = g[g.policy == pol].sort_values("alpha")
        a.plot(s.alpha, s.far, "o-", color=color, label=label, lw=1.8, ms=4)
        b.plot(s.far, s.recall, "o-", color=color, label=label, lw=1.8, ms=4)
    a.plot([0, 0.155], [0, 0.155], "k--", lw=1, label="target = achieved")
    a.set_xlabel("target false-alarm rate α")
    a.set_ylabel("achieved false-alarm rate")
    a.set_title("Does the false-alarm promise hold on an unseen machine?", fontsize=10)
    b.set_xlabel("achieved false-alarm rate")
    b.set_ylabel("recall of anomalous recordings")
    b.set_title("Faults caught at each false-alarm rate", fontsize=10)
    for ax in (a, b):
        ax.grid(alpha=0.25)
    a.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Bosch CNC data, leave-one-machine-out, real recordings streamed in time order with a burst of real faulty runs "
                 "(supervised XGBoost scorer; operator labels arrive 5 runs late)", fontsize=8.5, y=0.995)
    fig.tight_layout()
    fig.savefig("reports/figures/headline.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
