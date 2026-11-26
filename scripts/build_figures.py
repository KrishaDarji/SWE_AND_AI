"""Build report figures from the machine-readable result tables."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/swe_and_ai_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"


def rows(name):
    with (ROOT / "results" / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def save(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=160)
    plt.close()


def step_number(value: str) -> int:
    return int(value.lower().replace("step", ""))


def treatment_label(value: str) -> str:
    return "T" + value.lower().replace("treatment", "")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    size = rows("code_size.csv")
    measured = rows("metrics.csv")
    # Figure 1: available new-code trajectories, grouped by treatment.
    plt.figure(figsize=(8, 5))
    for treatment in sorted({treatment_label(r["treatment"]) for r in size}):
        subset = [r for r in size if treatment_label(r["treatment"]) == treatment]
        by_step = {}
        for r in subset:
            by_step.setdefault(step_number(r["step"]), []).append(float(r["new_code_percent"]))
        steps = sorted(by_step)
        plt.plot(steps, [np.median(by_step[s]) for s in steps], marker="o", label=treatment)
    plt.xlabel("Step"); plt.ylabel("Median new code (%)"); plt.title("New-code trajectory")
    plt.legend(ncol=4); save("new_code_histogram.png")

    # Figure 2: new versus reused code by treatment/domain.
    grouped = {}
    for r in size:
        key = (treatment_label(r["treatment"]), r["domain"])
        grouped.setdefault(key, [0.0, 0.0])
        grouped[key][0] += float(r["new_loc"])
        grouped[key][1] += max(0.0, float(r["total_loc"]) - float(r["new_loc"]))
    labels = [f"{t}/{d}" for t, d in sorted(grouped)]
    new = [grouped[k][0] for k in sorted(grouped)]
    reused = [grouped[k][1] for k in sorted(grouped)]
    plt.figure(figsize=(12, 5)); x = np.arange(len(labels))
    plt.bar(x, reused, label="Reused"); plt.bar(x, new, bottom=reused, label="New")
    plt.xticks(x, labels, rotation=70, ha="right", fontsize=7); plt.ylabel("LOC"); plt.title("Reused versus new code")
    plt.legend(); save("reuse_vs_new_code.png")

    # Figure 3: quality effects from Treatment 7.
    quality = rows("quality.csv")
    plt.figure(figsize=(8, 5)); labels = [f'{r["domain"]}\n{r["objective"]}' for r in quality]
    colors = ["tab:green" if r["treatment_wins"] == "True" else "tab:red" for r in quality]
    plt.bar(np.arange(len(quality)), [float(r["effect"]) for r in quality], color=colors)
    plt.xticks(np.arange(len(quality)), labels, rotation=35, ha="right", fontsize=8); plt.ylabel("Effect"); plt.title("Treatment 7 quality effects")
    save("quality_vs_code_size.png")

    # Figure 4: transfer improvement.
    transfer = rows("transfer.csv")
    plt.figure(figsize=(7, 4)); labels = [r["domain"] for r in transfer]
    colors = ["tab:green" if r["shared_wins"] == "True" else "tab:red" for r in transfer]
    plt.bar(labels, [float(r["improvement"]) for r in transfer], color=colors)
    plt.axhline(0, color="black", linewidth=.8); plt.ylabel("Shared-search improvement"); plt.title("Cross-domain transfer outcome")
    save("domain_transfer.png")

    # Figure 5: same decisions by domain and metric.
    same = rows("same_comparisons.csv")
    keys = sorted({(r["domain"], r["metric"]) for r in same})
    values = [sum(r["same"] == "True" for r in same if (r["domain"], r["metric"]) == k) for k in keys]
    plt.figure(figsize=(8, 4)); plt.bar([f"{d}\n{m}" for d, m in keys], values)
    plt.ylabel("Indistinguishable comparisons"); plt.title("Same-test decisions")
    save("same_matrix.png")
    failures = rows("failure_analysis.csv")
    labels = [f"T{r['treatment']}/{r['domain']}" for r in failures]
    plt.figure(figsize=(12, 5)); x = np.arange(len(labels))
    plt.bar(x, [int(r["steps_with_errors"]) for r in failures], label="Steps with errors")
    plt.bar(x, [int(r["failed_tests"]) for r in failures], bottom=[int(r["steps_with_errors"]) for r in failures], label="Failed tests")
    plt.xticks(x, labels, rotation=70, ha="right", fontsize=7); plt.ylabel("Count"); plt.title("Failure and error analysis")
    plt.legend(); save("failure_analysis.png")
    # True two-objective Pareto fronts for the available comparable measures:
    # minimize new LOC and runtime. Quality/hypervolume is not substituted.
    canonical = {
        "video encoding": "x264", "video_encoding": "x264", "x264": "x264",
        "software project planning": "nasa93dem", "software project estimation": "nasa93dem",
        "nasa93dem": "nasa93dem", "NASA93DEM": "nasa93dem",
        "model configuration and quality": "health_closedissues", "health_closedissues": "health_closedissues",
        "Health-ClosedIssues": "health_closedissues", "Health-ClosedIssues model configuration": "health_closedissues",
    }
    for domain in ("x264", "nasa93dem", "health_closedissues"):
        points = []
        for r in measured:
            if canonical.get(r.get("domain")) != domain:
                continue
            try:
                points.append((float(r["new_loc"]), float(r["runtime_seconds"]), treatment_label(r["treatment"])))
            except (KeyError, TypeError, ValueError):
                continue
        frontier = [p for p in points if not any(q[0] <= p[0] and q[1] <= p[1] and q[:2] != p[:2] for q in points)]
        plt.figure(figsize=(7, 5))
        for treatment in sorted({p[2] for p in points}):
            group = [p for p in points if p[2] == treatment]
            plt.scatter([p[0] for p in group], [p[1] for p in group], s=18, alpha=.55, label=treatment)
        frontier.sort()
        if frontier:
            plt.plot([p[0] for p in frontier], [p[1] for p in frontier], color="black", linewidth=2, marker="D", label="Pareto frontier")
        plt.xlabel("New LOC (minimize)"); plt.ylabel("Runtime (seconds, minimize)")
        plt.title(f"{domain}: Pareto frontier of code size and runtime")
        plt.legend(ncol=4, fontsize=8); save(f"pareto_{'nasa93' if domain == 'nasa93dem' else 'health' if domain == 'health_closedissues' else 'x264'}.png")
    print(f"Wrote {len(list(OUT.glob('*.png')))} figures to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
