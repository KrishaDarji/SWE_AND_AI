"""Build honest report-table placeholders from the available measurements.

This script deliberately does not infer quality, transfer, or statistical
claims from LOC/runtime alone.  It writes a machine-readable status file so
the report cannot accidentally present missing evidence as a result.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
sys.path.insert(0, str(ROOT))
from src.core.same import same


def write_csv(name: str, fields: list[str], rows: list[dict[str, str]]) -> None:
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ladder = json.loads((OUT / "treatment7_ladder.json").read_text())
    quality_rows: list[dict[str, str]] = []
    transfer_rows: list[dict[str, str]] = []
    for domain_run in ladder:
        domain = domain_run["domain"]
        for step in domain_run["steps"]:
            payload = step.get("payload", {})
            if step["step"] == 3:
                for objective, result in payload.items():
                    if isinstance(result, dict) and "effect" in result:
                        quality_rows.append({
                            "domain": domain,
                            "objective": objective,
                            "effect": str(result["effect"]),
                            "treatment_mean": str(result.get("treatment_mean", "")),
                            "baseline_mean": str(result.get("baseline_mean", "")),
                            "p_value": str(result.get("p_value", "")),
                            "significant": str(result.get("significant", "")),
                            "treatment_wins": str(result.get("treatment_wins", "")),
                            "source": "results/treatment7_ladder.json step 3",
                        })
            if step["step"] == 10:
                transfer_rows.append({
                    "domain": domain,
                    "baseline_selected": str(payload.get("baseline_selected", "")),
                    "shared_selected": str(payload.get("shared_selected", "")),
                    "baseline_score": str(payload.get("baseline_score", "")),
                    "shared_score": str(payload.get("shared_score", "")),
                    "improvement": str(payload.get("improvement", "")),
                    "shared_wins": str(payload.get("shared_wins", "")),
                    "source": "results/treatment7_ladder.json step 10",
                })
    write_csv(
        "quality.csv",
        ["domain", "objective", "effect", "treatment_mean", "baseline_mean", "p_value", "significant", "treatment_wins", "source"],
        quality_rows,
    )
    write_csv(
        "transfer.csv",
        ["domain", "baseline_selected", "shared_selected", "baseline_score", "shared_score", "improvement", "shared_wins", "source"],
        transfer_rows,
    )
    aliases = {
        "video encoding": "x264", "video_encoding": "x264", "x264": "x264",
        "software project planning": "nasa93dem", "software project estimation": "nasa93dem",
        "nasa93dem": "nasa93dem", "NASA93DEM": "nasa93dem",
        "model configuration and quality": "health_closedissues", "health_closedissues": "health_closedissues",
        "Health-ClosedIssues": "health_closedissues", "Health-ClosedIssues model configuration": "health_closedissues",
    }
    metrics = []
    with (OUT / "metrics.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            treatment = row["treatment"].replace("treatment", "")
            domain = aliases.get(row["domain"])
            if treatment in {"2", "5", "6"} and domain:
                row.update(treatment=treatment, domain=domain, step=str(int(row["step"])), trial=str(int(row["trial"])))
                metrics.append(row)
    summary = defaultdict(list)
    for row in metrics:
        try:
            float(row["new_loc"]); float(row["total_loc"]); float(row["runtime_seconds"]); float(row["error_count"])
        except (KeyError, TypeError, ValueError):
            continue
        summary[(row["treatment"], row["domain"])].append(row)
    summary_rows = []
    for (treatment, domain), group in sorted(summary.items()):
        summary_rows.append({
            "treatment": treatment,
            "domain": domain,
            "steps": str(len(group)),
            "new_loc_total": f"{sum(float(r['new_loc']) for r in group):.3f}",
            "total_loc_median": f"{sorted(float(r['total_loc']) for r in group)[len(group)//2]:.3f}",
            "reuse_percent": f"{100 * sum(max(0.0, float(r['total_loc']) - float(r['new_loc'])) for r in group) / max(1.0, sum(float(r['total_loc']) for r in group)):.3f}",
            "test_pass_percent": f"{100 * sum(str(r['test_pass']).lower() in {'1', 'true'} for r in group) / len(group):.3f}",
            "runtime_seconds_total": f"{sum(float(r['runtime_seconds']) for r in group):.3f}",
            "error_count_total": str(sum(int(float(r['error_count'])) for r in group)),
            "quality_status": "available only in treatment7_ladder.csv",
        })
    write_csv("summary_table.csv", list(summary_rows[0]), summary_rows)
    failure_rows = []
    for (treatment, domain), group in sorted(summary.items()):
        failure_rows.append({
            "treatment": treatment, "domain": domain,
            "steps": str(len(group)),
            "steps_with_errors": str(sum(int(float(r['error_count'])) > 0 for r in group)),
            "error_count_total": str(sum(int(float(r['error_count'])) for r in group)),
            "failed_tests": str(sum(str(r['test_pass']).lower() not in {'1', 'true'} for r in group)),
            "source": "results/metrics.csv",
        })
    write_csv("failure_analysis.csv", list(failure_rows[0]), failure_rows)
    grouped = defaultdict(list)
    for row in metrics:
        grouped[(row["treatment"], row["domain"], row["step"])].append(row)
    complete = {
        key for key, rows in grouped.items()
        if len(rows) == 3 and {r["trial"] for r in rows} == {"1", "2", "3"}
    }
    rows_out = []
    for domain in sorted({key[1] for key in complete}):
        treatments = sorted({key[0] for key in complete if key[1] == domain})
        for left, right in combinations(treatments, 2):
            for step in range(1, 11):
                for metric in ("new_loc", "runtime_seconds"):
                    lk, rk = (left, domain, str(step)), (right, domain, str(step))
                    if lk not in complete or rk not in complete:
                        continue
                    xs = [float(r[metric]) for r in sorted(grouped[lk], key=lambda r: r["trial"])]
                    ys = [float(r[metric]) for r in sorted(grouped[rk], key=lambda r: r["trial"])]
                    result = same(xs, ys)
                    rows_out.append({
                        "left_treatment": left, "right_treatment": right, "domain": domain,
                        "step": str(step), "metric": metric, "n_left": "3", "n_right": "3",
                        "left_values": ";".join(map(str, xs)), "right_values": ";".join(map(str, ys)),
                        "cohen": f"{result['cohen']:.6f}", "cliffs": f"{result['cliffs']:.6f}",
                        "ks": f"{result['ks']:.6f}", "same": str(result["same"]),
                        "thresholds": "cohen<=0.35;cliffs<=0.195;ks<=1.36",
                    })
    write_csv("same_comparisons.csv", list(rows_out[0]) if rows_out else ["status", "reason"], rows_out or [{"status": "pending", "reason": "No complete comparison groups found."}])
    print(f"Wrote {len(quality_rows)} quality rows and {len(transfer_rows)} transfer rows.")
    print(f"Wrote {len(rows_out)} same-comparison rows from complete treatment groups.")


if __name__ == "__main__":
    main()
