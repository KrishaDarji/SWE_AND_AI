"""Freeze one shared 80/20 train/test split per dataset (seed 2026, test capped at 200 rows).

Every treatment's saved code is scored on these same files, so external metrics are comparable.
"""
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "x264": ("moot/optimize/config/X264_AllMeasurements.csv", "Performance-"),
    "nasa93dem": ("moot/optimize/process/nasa93dem.csv", "Effort-"),
    "health_closedissues": ("moot/optimize/hpo/Health-ClosedIssues0000.csv", "MRE-"),
}
SEED, TEST_FRACTION, TEST_CAP = 2026, 0.2, 200


def main():
    for name, (rel, _target) in DATASETS.items():
        with open(ROOT / rel, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = [r for r in reader if r]
        order = list(range(len(rows)))
        random.Random(SEED).shuffle(order)
        n_test = round(len(rows) * TEST_FRACTION)
        test_idx, train_idx = order[:n_test][:TEST_CAP], order[n_test:]
        out = ROOT / "evaluation/splits" / name
        out.mkdir(parents=True, exist_ok=True)
        for part, idx in (("train", train_idx), ("test", test_idx)):
            with open(out / f"{part}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(rows[i] for i in sorted(idx))
        print(name, "train", len(train_idx), "test", len(test_idx))


if __name__ == "__main__":
    main()
