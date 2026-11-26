"""Build one consistent four-panel results figure per domain."""
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
COLORS = {f"T{i}": c for i, c in enumerate(("#5267F5", "#F28E2B", "#36A269", "#D64550", "#8E63CE", "#8B5E4A", "#111827"), 1)}

def read(path):
    with path.open(newline="") as f: return list(csv.DictReader(f))
def treatment(v): return "T" + v.lower().replace("treatment", "")
def canonical(v):
    return {"video encoding":"x264", "video_encoding":"x264", "x264":"x264", "software project planning":"nasa93dem", "software project estimation":"nasa93dem", "nasa93dem":"nasa93dem", "NASA93DEM":"nasa93dem", "model configuration and quality":"health_closedissues", "health_closedissues":"health_closedissues", "Health-ClosedIssues":"health_closedissues"}.get(v, v)

def main():
    metrics = read(ROOT / "results/metrics.csv")
    eval_rows = read(ROOT / "evaluation/results/guess_by_run.csv")
    quality = read(ROOT / "results/quality.csv")
    domains = ("x264", "nasa93dem", "health_closedissues")
    for domain in domains:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
        ax1, ax2, ax3, ax4 = axes.flat
        rows = [r for r in metrics if canonical(r.get("domain")) == domain]
        # New LOC trajectory.
        by = defaultdict(list)
        for r in rows:
            try: by[(treatment(r["treatment"]), int(r["step"].replace("step", "")))].append(float(r["new_loc"]))
            except (KeyError, ValueError, TypeError): pass
        for t in sorted({k[0] for k in by}):
            steps = sorted(k[1] for k in by if k[0] == t)
            ax1.plot(steps, [np.median(by[(t, s)]) for s in steps], marker="o", lw=1.8, label=t, color=COLORS.get(t))
        ax1.set_title("New LOC by ladder step"); ax1.set_xlabel("Step"); ax1.set_ylabel("Median new LOC"); ax1.grid(axis="y", alpha=.2)
        # Runtime distribution by treatment.
        run = defaultdict(list)
        for r in rows:
            try: run[treatment(r["treatment"])].append(float(r["runtime_seconds"]))
            except (KeyError, ValueError, TypeError): pass
        ts = sorted(run)
        ax2.boxplot([run[t] for t in ts], labels=ts, patch_artist=True, boxprops={"facecolor":"#DCE4FF"})
        ax2.set_title("Recorded runtime"); ax2.set_xlabel("Treatment"); ax2.set_ylabel("Seconds"); ax2.grid(axis="y", alpha=.2)
        # Reliability from evaluation hit rates.
        er = [r for r in eval_rows if canonical(r.get("dataset")) == domain]
        hit = defaultdict(list)
        for r in er:
            try: hit[treatment(r["treatment"])].append(100 * float(r["hit"]))
            except (KeyError, ValueError, TypeError): pass
        ts = sorted(hit)
        ax3.bar(ts, [np.mean(hit[t]) for t in ts], color=[COLORS.get(t) for t in ts])
        ax3.set_title("Evaluation hit rate"); ax3.set_xlabel("Treatment"); ax3.set_ylabel("Hit rate (%)"); ax3.set_ylim(0, 100); ax3.grid(axis="y", alpha=.2)
        # Objective comparison where recorded; otherwise explicit absence.
        qr = [r for r in quality if canonical(r.get("domain")) == domain]
        if qr:
            labels = [r["objective"] for r in qr]; x = np.arange(len(labels)); base = [float(r["baseline_mean"]) for r in qr]; t7 = [float(r["treatment_mean"]) for r in qr]
            ax4.plot(x, base, "o-", color="#6B7280", label="Baseline"); ax4.plot(x, t7, "o-", color="#238636", label="T7")
            ax4.set_xticks(x, labels, rotation=25); ax4.legend(frameon=False)
            ax4.set_title("Recorded objective comparison"); ax4.set_ylabel("Objective value")
        else:
            ax4.text(.5, .5, "No structured quality\nmeasurements recorded", ha="center", va="center", transform=ax4.transAxes)
            ax4.set_title("Objective quality")
        ax4.grid(axis="y", alpha=.2)
        fig.suptitle(f"{domain}: implementation, reliability, and quality", fontsize=16, fontweight="bold")
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(.5, .01))
        fig.tight_layout(rect=(0, .05, 1, .95))
        name = "nasa93" if domain == "nasa93dem" else "health" if domain == "health_closedissues" else "x264"
        fig.savefig(OUT / f"domain_{name}_results.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
    print("Wrote three domain-wise composite figures to report/generated/")

if __name__ == "__main__": main()
