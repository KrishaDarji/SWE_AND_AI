#!/usr/bin/env python3
"""Independent EZR-informed versus uniform search for measured x264 rows."""
import csv, math, random, sys

TIE_TOLERANCE = 1e-9
EXPLORATION = 5

def load_dataset(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "Performance-" not in rows[0]:
        raise ValueError("dataset must contain Performance-")
    names = [k for k in rows[0] if k != "Performance-"]
    out = []
    seen = set()
    for row in rows:
        cfg = tuple(int(row[k]) for k in names)
        value = float(row["Performance-"])
        if cfg in seen:
            continue
        if not math.isfinite(value):
            continue
        seen.add(cfg)
        out.append((cfg, value))
    if not out:
        raise ValueError("no valid candidates")
    return names, out

def _check(candidates, budget, seed):
    if budget < 1 or not isinstance(budget, int):
        raise ValueError("budget must be a positive integer")
    if len(candidates) < budget:
        raise ValueError("budget exceeds candidate count")
    if len({c for c, _ in candidates}) != len(candidates):
        raise ValueError("duplicate configurations are not permitted")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")

def _recommend(evaluated):
    return min(evaluated, key=lambda x: (x[1], x[0]))

def uniform_search(candidates, budget, seed, callback):
    _check(candidates, budget, seed)
    order = list(range(len(candidates))); random.Random(seed).shuffle(order)
    evaluated = []
    for i in order[:budget]:
        cfg = candidates[i][0]
        evaluated.append((cfg, float(callback(cfg))))
    return order[:budget], evaluated, _recommend(evaluated)

def _distance(a, b):
    return sum(x != y for x, y in zip(a, b))

def ezr_informed_search(candidates, budget, seed, callback):
    """Random exploration, then nearest-neighbor prediction and culling.

    This adapts EZR's numeric summaries/normalization, distance, stable sorting,
    and active-learning idea without importing or executing Lua.
    """
    _check(candidates, budget, seed)
    rng = random.Random(seed); remaining = list(range(len(candidates)))
    rng.shuffle(remaining); chosen = []
    evaluated = []
    for _ in range(min(EXPLORATION, budget)):
        i = remaining.pop(0); chosen.append(i)
        evaluated.append((candidates[i][0], float(callback(candidates[i][0]))))
    while remaining:
        if len(evaluated) >= budget:
            break
        # Stable lexicographic tie-breaking makes all choices reproducible.
        ranked = []
        for i in remaining:
            cfg = candidates[i][0]
            neighbors = sorted((( _distance(cfg, x[0]), x[1], x[0]) for x in evaluated),
                               key=lambda z: (z[0], z[1], z[2]))[:min(3, len(evaluated))]
            prediction = sum(x[1] for x in neighbors) / len(neighbors)
            ranked.append((prediction, cfg, i))
        _, _, i = min(ranked, key=lambda z: (z[0], z[1], z[2]))
        remaining.remove(i); chosen.append(i)
        evaluated.append((candidates[i][0], float(callback(candidates[i][0]))))
    return chosen, evaluated, _recommend(evaluated)

def compare(path, budget=20, seeds=range(42, 52)):
    names, candidates = load_dataset(path); records = []
    for seed in seeds:
        calls = {"tuner": 0, "baseline": 0}
        def counted(kind):
            def cb(cfg):
                calls[kind] += 1
                return dict(candidates)[cfg]
            return cb
        ti, te, tr = ezr_informed_search(candidates, budget, seed, counted("tuner"))
        bi, be, br = uniform_search(candidates, budget, seed, counted("baseline"))
        records.append((seed, ti, te, tr, bi, be, br, calls))
    return names, candidates, records

def main(argv):
    if len(argv) != 2:
        raise SystemExit("usage: race_compare.py DATASET")
    names, candidates, records = compare(argv[1])
    print("seed,position,strategy,identifier,configuration,Performance-")
    for seed, ti, te, tr, bi, be, br, calls in records:
        for strategy, inds, vals, rec in (("tuner", ti, te, tr), ("baseline", bi, be, br)):
            for pos, (cfg, val) in enumerate(vals, 1):
                print(seed, pos, strategy, "cfg-" + "-".join(map(str, cfg)), ";".join(map(str, cfg)), "%.3f" % val, sep=",")
        print("summary", seed, "difference", "%.9f" % (tr[1]-br[1]), "tuner_calls", calls["tuner"], "baseline_calls", calls["baseline"], sep=",")

if __name__ == "__main__":
    main(sys.argv)
