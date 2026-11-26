"""External metrics for the guess skill on the shared split, plus Scott-style ranks with `same`.

MAE (lower better), hit rate = share of predictions close enough to use (see HIT; higher better),
Spearman rho between predicted and actual (higher better: does it order configurations correctly?).
Ranks: per dataset, each treatment's absolute errors over all test rows and trials; sorted by median,
new rank only when `same` says the neighbour is different (rank 0 = best).
"""
import csv, json, statistics as st, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core.certify import same

DS = ("x264", "nasa93dem", "health_closedissues")
TARGET = {"x264": "Performance-", "nasa93dem": "Effort-", "health_closedissues": "MRE-"}
# hit = prediction close enough to be useful: x264 within 5% of the real time, NASA93 within 25% of real
# effort (the standard PRED(25)), Health within 0.05 of the real MRE (5 error points)
HIT = {"x264": lambda p, a: abs(p - a) <= 0.05 * abs(a), "nasa93dem": lambda p, a: abs(p - a) <= 0.25 * abs(a),
       "health_closedissues": lambda p, a: abs(p - a) <= 0.05}


def rankdata(v):
    order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else 0.0


def main():
    out_rows, rank_rows = [], []
    for d in DS:
        with open(ROOT / "evaluation/splits" / d / "test.csv", newline="") as f:
            actual = [float(r[TARGET[d]]) for r in csv.DictReader(f)]
        with open(ROOT / "evaluation/splits" / d / "train.csv", newline="") as f:
            mean_train = st.mean(float(r[TARGET[d]]) for r in csv.DictReader(f))
        groups = {"mean baseline": [abs(a - mean_train) for a in actual]}
        for t in range(1, 8):
            errs = []
            for tr in ((1,) if t == 7 else (1, 2, 3)):
                p = ROOT / "evaluation/out/guess" / f"T{t}_tr{tr}_{d}.json"
                if not p.exists():
                    continue
                pred = json.load(open(p))
                e = [abs(x - y) for x, y in zip(pred, actual)]
                errs += e
                out_rows.append({"dataset": d, "treatment": t, "trial": tr, "MAE": st.mean(e),
                                 "hit": sum(HIT[d](x, y) for x, y in zip(pred, actual)) / len(actual),
                                 "spearman": spearman(pred, actual)})
            if errs:
                groups[f"T{t}"] = errs
        items = sorted(groups.items(), key=lambda kv: (st.median(kv[1]), st.mean(kv[1])))
        rank = 0
        for i, (name, v) in enumerate(items):
            if i and not same(items[i - 1][1], v):
                rank += 1
            rank_rows.append({"dataset": d, "who": name, "rank": rank, "median_abs_err": st.median(v), "mean_abs_err": st.mean(v)})
    res = ROOT / "evaluation/results"; res.mkdir(exist_ok=True)
    for name, rows in (("guess_by_run.csv", out_rows), ("guess_ranks.csv", rank_rows)):
        with open(res / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # summary per treatment x dataset (median over trials)
    print(f"{'':6}" + "".join(f"{d:>36}" for d in DS))
    print(f"{'':6}" + "".join(f"{'MAE':>12}{'hit':>12}{'rho':>12}" for _ in DS))
    for t in range(1, 8):
        line = f"T{t:<5}"
        for d in DS:
            rs = [r for r in out_rows if r["dataset"] == d and r["treatment"] == t]
            line += f"{st.median(r['MAE'] for r in rs):12.4g}{st.median(r['hit'] for r in rs):12.2f}{st.median(r['spearman'] for r in rs):12.2f}"
        print(line)
    for d in DS:
        print(d, " | ".join(f"{r['who']}:{r['rank']}" for r in rank_rows if r["dataset"] == d))


if __name__ == "__main__":
    main()
