"""Create six report-ready results figures from recorded project outputs."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "generated"
OUT.mkdir(parents=True, exist_ok=True)
COLORS = {"T1": "#5B6CFF", "T2": "#FF8A3D", "T3": "#26A269", "T4": "#D64550", "T5": "#8E63CE", "T6": "#8B5E4A", "T7": "#111827"}


def read(name: str):
    with (ROOT / "results" / name).open(newline="") as f:
        return list(csv.DictReader(f))


def tid(value: str) -> str:
    return "T" + value.lower().replace("treatment", "")


def save(fig, name: str):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    size = read("code_size.csv")
    # 1. New-code trajectories.
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    by = defaultdict(list)
    for r in size:
        by[(tid(r["treatment"]), int(r["step"].replace("step", "")))].append(float(r["new_code_percent"]))
    for treatment in sorted({k[0] for k in by}):
        steps = sorted(k[1] for k in by if k[0] == treatment)
        ax.plot(steps, [np.median(by[(treatment, s)]) for s in steps], marker="o", lw=2, label=treatment, color=COLORS.get(treatment))
    ax.set(xlabel="Ladder step", ylabel="Median new code (%)", title="New code usually falls after the first capability")
    ax.grid(axis="y", alpha=.2); ax.legend(ncol=4, frameon=False)
    save(fig, "results_new_code_trajectories.png")

    # 2. Reuse comparison, complete treatment groups only.
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    summary = defaultdict(float)
    for r in size:
        t = tid(r["treatment"])
        if t in {"T1", "T2", "T3", "T4", "T5", "T6"}:
            summary[(t, r["domain"])] += float(r["new_loc"])
    domains = sorted({k[1] for k in summary})
    pairs = [("T1", "T2"), ("T3", "T4"), ("T5", "T6")]
    labels, vals, cols = [], [], []
    for d in domains:
        for a, b in pairs:
            if (a, d) in summary and (b, d) in summary:
                labels.append(f"{d}\n{a} vs {b}")
                vals.append(100 * (summary[(a, d)] - summary[(b, d)]) / max(summary[(a, d)], 1))
                cols.append("#238636" if vals[-1] > 0 else "#D64550")
    ax.bar(np.arange(len(vals)), vals, color=cols)
    ax.axhline(0, color="#222", lw=.8); ax.set_xticks(np.arange(len(vals)), labels, rotation=55, ha="right", fontsize=8)
    ax.set(ylabel="Reduction in total new LOC (%)", title="Reuse reduces implementation cost in matched pairs")
    ax.grid(axis="y", alpha=.2); save(fig, "results_reuse_comparison.png")

    # 3. Reliability heatmap using all available evaluation runs.
    with (ROOT / "evaluation/results/guess_by_run.csv").open(newline="") as f:
        reliability = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    treatments = [f"T{i}" for i in range(1, 8)]; domains = sorted({r["dataset"] for r in reliability})
    matrix = np.full((len(treatments), len(domains)), np.nan)
    for r in reliability:
        matrix[treatments.index(tid(r["treatment"])), domains.index(r["dataset"])] = 100 * float(r["hit"])
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(domains)), domains, rotation=30, ha="right"); ax.set_yticks(range(len(treatments)), treatments)
    for i in range(len(treatments)):
        for j in range(len(domains)):
            if not np.isnan(matrix[i, j]): ax.text(j, i, f"{matrix[i,j]:.0f}%", ha="center", va="center", fontsize=9)
    ax.set(title="Evaluation reliability across all treatments", xlabel="Domain", ylabel="Treatment")
    fig.colorbar(im, ax=ax, label="Prediction hit rate (%)"); save(fig, "results_reliability_heatmap.png")

    # 4. Treatment 7 quality effects.
    quality = read("quality.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    labels = [f'{r["domain"]}\n{r["objective"]}' for r in quality]
    for i, r in enumerate(quality):
        base, treatment = float(r["baseline_mean"]), float(r["treatment_mean"])
        ax.plot([base, treatment], [i, i], color="#9CA3AF", lw=2)
        ax.scatter(base, i, color="#6B7280", s=55, label="Baseline" if i == 0 else "")
        ax.scatter(treatment, i, color="#238636" if r["treatment_wins"] == "True" else "#D64550", s=65, label="Treatment 7" if i == 0 else "")
    ax.set_yticks(range(len(labels)), labels); ax.set_xlabel("Recorded objective value (direction shown by + or -)")
    ax.set_title("Treatment 7 objective values compared with baseline")
    ax.grid(axis="x", alpha=.2); ax.legend(frameon=False); save(fig, "results_quality_effects.png")

    # 5. Transfer outcomes.
    transfer = read("transfer.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    labels = [r["domain"] for r in transfer]
    for i, r in enumerate(transfer):
        base, shared = float(r["baseline_score"]), float(r["shared_score"])
        ax.plot([base, shared], [i, i], color="#9CA3AF", lw=2)
        ax.scatter(base, i, color="#6B7280", s=60, label="Baseline" if i == 0 else "")
        ax.scatter(shared, i, color="#238636" if r["shared_wins"] == "True" else "#D64550", s=70, label="Shared machinery" if i == 0 else "")
    ax.set_yticks(range(len(labels)), labels); ax.set_xlabel("Evaluation score")
    ax.set_title("Shared machinery changes the selected-domain score")
    ax.grid(axis="x", alpha=.2); ax.legend(frameon=False); save(fig, "results_transfer_outcomes.png")

    # 6. Same decisions.
    same = read("same_comparisons.csv")
    keys = sorted({(r["domain"], r["metric"]) for r in same})
    counts = [sum(r["same"] == "True" for r in same if (r["domain"], r["metric"]) == k) for k in keys]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(range(len(keys)), counts, color="#5B6CFF")
    ax.set_xticks(range(len(keys)), [f"{d}\n{m}" for d, m in keys], rotation=30, ha="right")
    ax.set(ylabel="Statistically same comparisons", title="Same-test decisions prevent overclaiming raw differences")
    ax.grid(axis="y", alpha=.2); save(fig, "results_same_decisions.png")
    print("Wrote six report figures to report/generated/")


if __name__ == "__main__":
    main()
