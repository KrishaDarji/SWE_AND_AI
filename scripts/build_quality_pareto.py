"""Extract recorded objective values and plot quality/LOC Pareto fronts.

No values are estimated: every value retains its source metrics file. Because
some historical notes mention more than one candidate/baseline, the last
explicit objective mention is used and marked for review.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/swe_and_ai_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
PATTERNS = {
    "x264": [("Performance-", re.compile(r"Performance-\s*[:= ]\s*(-?\d+(?:\.\d+)?|\d+(?:\.\d+)?)(?!\d)"))],
    "nasa93dem": [(name, re.compile(rf"{name}\s*[:= ]\s*(-?\d+(?:\.\d+)?)")) for name in ("Effort-", "Defects-")],
    "health_closedissues": [(name, re.compile(rf"{name}\s*[:= ]\s*(-?\d+(?:\.\d+)?)")) for name in ("MRE-", "ACC+", "PRED40+")],
}
ALIASES = {"x264": "x264", "video encoding": "x264", "video_encoding": "x264", "nasa93dem": "nasa93dem", "NASA93DEM": "nasa93dem", "software project planning": "nasa93dem", "software project estimation": "nasa93dem", "health_closedissues": "health_closedissues", "model configuration and quality": "health_closedissues", "Health-ClosedIssues": "health_closedissues"}


def main() -> None:
    rows = []
    for path in sorted((ROOT / "experiments/runs").glob("treatment[1-7]/trials/*/*/step*/metrics.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle), None)
        if not row:
            continue
        domain = ALIASES.get(row.get("domain", ""))
        if domain not in PATTERNS:
            continue
        notes = " ".join(str(value) for value in row.values() if value is not None)
        for objective, pattern in PATTERNS[domain]:
            matches = pattern.findall(notes)
            if matches:
                rows.append({"treatment": row.get("treatment", "").replace("treatment", ""), "trial": row.get("trial", ""), "domain": domain, "step": row.get("step", ""), "objective": objective, "quality": matches[-1], "new_loc": row.get("new_loc", ""), "runtime_seconds": row.get("runtime_seconds", ""), "source": str(path.relative_to(ROOT)), "extraction": "last explicit objective mention in recorded metrics row"})
    fields = list(rows[0]) if rows else ["treatment", "trial", "domain", "step", "objective", "quality", "new_loc", "runtime_seconds", "source", "extraction"]
    with (OUT / "quality_measurements.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    figout = OUT / "figures"; figout.mkdir(exist_ok=True)
    for domain in ("x264", "nasa93dem", "health_closedissues"):
        subset = [r for r in rows if r["domain"] == domain]
        plt.figure(figsize=(8, 5))
        for objective in sorted({r["objective"] for r in subset}):
            points = []
            for r in subset:
                if r["objective"] != objective:
                    continue
                try: points.append((float(r["new_loc"]), float(r["quality"]), "T" + r["treatment"]))
                except (TypeError, ValueError): continue
            for treatment in sorted({p[2] for p in points}):
                group = [p for p in points if p[2] == treatment]
                plt.scatter([p[0] for p in group], [p[1] for p in group], s=20, alpha=.55, label=f"{treatment} {objective}")
            # All listed objectives are minimized; mark the nondominated points.
            frontier = [p for p in points if not any(q[0] <= p[0] and q[1] <= p[1] and q[:2] != p[:2] for q in points)]
            frontier.sort()
            if frontier: plt.plot([p[0] for p in frontier], [p[1] for p in frontier], linewidth=2, marker="D", label=f"{objective} frontier")
        if not subset:
            plt.text(0.5, 0.5, "No recorded quality data\nfor this domain", ha="center", va="center", transform=plt.gca().transAxes, fontsize=14)
        plt.xlabel("New LOC (minimize)"); plt.ylabel("Recorded quality objective (direction per name)")
        plt.title(f"{domain}: quality/code-size Pareto frontier")
        if subset:
            plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(figout / f"pareto_quality_{'nasa93' if domain == 'nasa93dem' else 'health' if domain == 'health_closedissues' else 'x264'}.png", dpi=160); plt.close()
    print(f"Wrote {len(rows)} recorded quality measurements")


if __name__ == "__main__":
    main()
