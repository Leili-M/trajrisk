"""Proposal section 8: reliability diagrams and risk-coverage curves for the report.

Reads only files written by scripts/judge.py (no model is run):
  reports/judge/<dataset>[_giffcf]_reliability/reliability_<run>.csv
  reports/judge/<dataset>[_giffcf]_risk_coverage.csv
  reports/judge/<dataset>.json (AUROC gains with 95% t-intervals)
Writes reports/figures/reliability.pdf, risk_coverage.pdf and quality_cost.pdf.

    PYTHONPATH=src python scripts/make_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
JUDGE = REPO / "reports" / "judge"
OUT = REPO / "reports" / "figures"
PANELS = [("amazon_video_games", "DiffRec, Video Games"),
          ("amazon_books", "DiffRec, Books"),
          ("amazon_books_giffcf", "GiffCF, Books")]
# validated categorical slots (dataviz default palette); line style and marker
# repeat the identity so colour is never the only cue
STYLE = {"controls3": ("#2a78d6", "-", "o", "Controls3"),
         "controls": ("#eb6834", "--", "s", "All controls"),
         "combined": ("#1baf7a", "-.", "^", "All controls + trajectory")}
REF = {"random": (":", "Random order"), "oracle": ((0, (1, 3)), "Oracle (not deployable)")}
INK, GRID = "#2b2b2a", "#d9d8d2"

plt.rcParams.update({"font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 1.6})


def reliability_figure():
    fig, axes = plt.subplots(1, len(PANELS), figsize=(7.0, 2.4), sharey=False)
    for ax, (name, title) in zip(axes, PANELS):
        files = sorted((JUDGE / f"{name}_reliability").glob("reliability_*.csv"))
        if not files:
            ax.set_visible(False)
            continue
        t = pd.concat([pd.read_csv(f) for f in files]).groupby("bin")[["predicted", "observed"]].mean()
        lo = min(t.min()) - 0.02
        ax.plot([lo, 1], [lo, 1], color=GRID, linewidth=1, label="Perfect calibration")
        ax.plot(t["predicted"], t["observed"], color=STYLE["combined"][0], marker="o",
                markersize=4, label="Risk model (all inputs)")
        ax.set_xlim(lo, 1)
        ax.set_ylim(lo, 1)
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_xlabel("Predicted miss probability")
        ax.grid(color=GRID, linewidth=0.5)
    axes[0].set_ylabel("Observed miss rate")
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "reliability.pdf")
    plt.close(fig)


def risk_coverage_figure():
    fig, axes = plt.subplots(1, len(PANELS), figsize=(7.0, 2.4), sharey=False)
    for ax, (name, title) in zip(axes, PANELS):
        f = JUDGE / f"{name}_risk_coverage.csv"
        if not f.exists():
            ax.set_visible(False)
            continue
        c = pd.read_csv(f).groupby(["predictors", "coverage"])["risk_ndcg"].mean().unstack(0)
        for key, (ls, label) in REF.items():
            ax.plot(c.index, c[key], color="#8a8983", linestyle=ls, linewidth=1.2, label=label)
        for key, (color, ls, marker, label) in STYLE.items():
            ax.plot(c.index, c[key], color=color, linestyle=ls, marker=marker, markersize=3.5,
                    markevery=3, label=label)
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_xlabel("Coverage (share of cases kept)")
        ax.grid(color=GRID, linewidth=0.5)
    axes[0].set_ylabel("Mean 1 - NDCG@10 of kept cases")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "risk_coverage.pdf")
    plt.close(fig)


COST_PANELS = [("amazon_video_games", "DiffRec, Video Games"), ("amazon_books", "DiffRec, Books")]
# (gain key in reports/judge/<dataset>.json, extra model runs per request)
COST_POINTS = {"trajectory": [("controls + lists over controls", 0)],
               "sampling": [("controls + sampling2 over controls", 2),
                            ("controls + sampling5 over controls", 5),
                            ("controls + sampling10 over controls", 10)],
               "masking": [("controls + masking over controls", 10)]}
COST_STYLE = {"trajectory": ("#1baf7a", "-", "^", "Trajectory (lists)"),
              "sampling": ("#2a78d6", "-", "o", "Repeated sampling (2, 5, 10)"),
              "masking": ("#eb6834", "-", "s", "Masked history (10)")}


def quality_cost_figure():
    """Proposal section 9: detection quality against extra model runs per request."""
    import json
    fig, axes = plt.subplots(1, len(COST_PANELS), figsize=(3.4, 2.3), sharey=True)
    for ax, (name, title) in zip(axes, COST_PANELS):
        f = JUDGE / f"{name}.json"
        if not f.exists():
            ax.set_visible(False)
            continue
        gains = json.loads(f.read_text())["gains"]
        for key, points in COST_POINTS.items():
            pts = [(runs, gains[g]) for g, runs in points if g in gains]
            if not pts:
                continue
            color, ls, marker, label = COST_STYLE[key]
            x = [p[0] for p in pts]
            m = [p[1][0] for p in pts]
            err = [[p[1][0] - p[1][1] for p in pts], [p[1][2] - p[1][0] for p in pts]]
            ax.errorbar(x, m, yerr=err, color=color, linestyle=ls, marker=marker,
                        markersize=4, capsize=2, linewidth=1.4, label=label)
        ax.axhline(0, color="#8a8983", linewidth=0.8)
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_xlabel("Extra model runs")
        ax.set_xticks([0, 2, 5, 10])
        ax.grid(color=GRID, linewidth=0.5)
    axes[0].set_ylabel("AUROC gain over controls")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(OUT / "quality_cost.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reliability_figure()
    risk_coverage_figure()
    quality_cost_figure()
    print(f"Figures written to {OUT}")


if __name__ == "__main__":
    main()
