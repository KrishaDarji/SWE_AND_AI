"""External metrics for the race skill.

x264: candidates = 200 shared test configs, budget 20, seeds 1..10; score = real encode time of the pick.
Health: history = shared train.csv; score = real MRE- of the recommended configuration.
regret = (pick - best) / (median - best) over the candidate pool: 0 = found the best, 1 = only as good as the median.
win rate = share of seeds where the pick beats plain random search with the same budget and seed (ties = half).
Ranks: picks pooled over trials, sorted by median, new rank only when `same` says different (0 = best).
"""
import csv, json, random, statistics as st, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core.certify import same

import os
BUDGET, SEEDS = int(os.environ.get("RACE_BUDGET", "20")), list(range(1, 11))
OUT = "race" if BUDGET == 20 else f"race_b{BUDGET}"
POOL = {"x264": ("test", "Performance-"), "health_closedissues": ("train", "MRE-")}
# how each Health program reached its pick (read from the code while writing adapters)
HEALTH_ACCESS = {
    (1, 2): "full history", (1, 3): "held-out", (2, 1): "full history", (2, 2): "full history", (2, 3): "full history",
    (3, 2): "held-out", (3, 3): "held-out", (4, 1): "full history", (4, 2): "full history", (4, 3): "full history",
    (5, 1): "full history", (5, 2): "full history", (5, 3): "full history", (6, 1): "full history",
    (6, 2): "full history", (6, 3): "full history", (7, 1): "budget 20",
}


def main():
    rows_out, ranks_out = [], []
    for d, (part, target) in POOL.items():
        if BUDGET != 20 and d != "x264":
            continue
        with open(ROOT / "evaluation/splits" / d / f"{part}.csv", newline="") as f:
            pool = [float(r[target]) for r in csv.DictReader(f)]
        best, med = min(pool), st.median(pool)
        regret = lambda v: (v - best) / (med - best)
        rnd = [min(random.Random(s).sample(pool, BUDGET)) for s in SEEDS]
        groups = {"random search": [regret(v) for v in rnd]}
        for t in range(1, 8):
            picks_all = []
            for tr in ((1,) if t == 7 else (1, 2, 3)):
                p = ROOT / "evaluation/out" / OUT / f"T{t}_tr{tr}_{d}.json"
                if not p.exists():
                    continue
                picks = json.load(open(p))
                if isinstance(picks, dict):
                    rows_out.append({"dataset": d, "treatment": t, "trial": tr, "status": "n/a: " + picks.get("na", "")})
                    continue
                wins = sum((v < r) + 0.5 * (v == r) for v, r in zip(picks, rnd)) / len(picks)
                access = f"budget {BUDGET}" if d == "x264" else HEALTH_ACCESS.get((t, tr), "?")
                rows_out.append({"dataset": d, "treatment": t, "trial": tr, "status": "ok", "access": access,
                                 "median_pick": st.median(picks), "median_regret": st.median(regret(v) for v in picks),
                                 "win_rate_vs_random": wins})
                picks_all += [regret(v) for v in picks]
            if picks_all:
                groups[f"T{t}"] = picks_all
        items = sorted(groups.items(), key=lambda kv: (st.median(kv[1]), st.mean(kv[1])))
        rank = 0
        for i, (name, v) in enumerate(items):
            if i and not same(items[i - 1][1], v):
                rank += 1
            ranks_out.append({"dataset": d, "who": name, "rank": rank, "median_regret": st.median(v)})
    res = ROOT / "evaluation/results"; res.mkdir(exist_ok=True)
    keys = ["dataset", "treatment", "trial", "status", "access", "median_pick", "median_regret", "win_rate_vs_random"]
    with open(res / f"{OUT}_by_run.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows_out)
    with open(res / f"{OUT}_ranks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ranks_out[0])); w.writeheader(); w.writerows(ranks_out)
    for r in rows_out:
        if r["status"] == "ok":
            print(f"{r['dataset'][:6]} T{r['treatment']} tr{r['trial']} {r['access']:12} pick={r['median_pick']:.4g} regret={r['median_regret']:.3f} win={r['win_rate_vs_random']:.2f}")
        else:
            print(f"{r['dataset'][:6]} T{r['treatment']} tr{r['trial']} {r['status']}")
    for d in POOL:
        if BUDGET != 20 and d != "x264":
            continue
        print(d, " | ".join(f"{r['who']}:{r['rank']} ({r['median_regret']:.2f})" for r in ranks_out if r["dataset"] == d))


if __name__ == "__main__":
    main()
