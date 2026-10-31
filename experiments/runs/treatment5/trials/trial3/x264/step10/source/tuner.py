"""Independent EZR-informed active tuner and uniform random baseline."""
import csv, math, random

FEATURES = ("no_asm", "no_8x8dct", "no_cabac", "no_deblock", "no_fast_pskip", "no_mbtree", "no_mixed_refs", "no_weightb", "rc_lookahead", "rc_lookahead_20", "rc_lookahead_40", "rc_lookahead_60", "ref", "ref_1", "ref_5", "ref_9")
TOLERANCE = 1e-12

def load_dataset(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "Performance-" not in rows[0]:
        raise ValueError("dataset lacks Performance-")
    out = []
    for i, row in enumerate(rows):
        try:
            y = float(row["Performance-"])
            x = tuple(int(row[k]) for k in FEATURES)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid row %d" % i) from exc
        if not math.isfinite(y):
            raise ValueError("nonfinite outcome at row %d" % i)
        out.append({"id": "r%04d" % i, "x": x, "y": y})
    return out

def _key(r): return tuple(r["x"])

def _best(records):
    if not records: raise ValueError("no evaluated candidates")
    return min(records, key=lambda r: (r["y"], r["id"]))

def _distance(a, b):
    return sum(v != w for v, w in zip(a, b)) / len(a)

def _ezr_score(candidate, labeled):
    """Discretized near-neighbor estimate; lower is preferred.

    EZR-inspired pieces: distance over x columns, labeled rows, and a
    best-first acquisition rule. No outcome is read before callback evaluation.
    """
    if not labeled: return 0.5
    nearest = sorted((_distance(candidate["x"], r["x"]), r["y"]) for r in labeled)
    k = min(5, len(nearest))
    return sum(y + d * 100.0 for d, y in nearest[:k]) / k

def choose_ezr(pool, budget, seed, callback):
    if budget < 1: raise ValueError("budget must be positive")
    if budget > len(pool): raise ValueError("budget exceeds candidates")
    rng = random.Random(seed); pending = list(pool); rng.shuffle(pending)
    labeled = []
    for _ in range(budget):
        if not labeled:
            candidate = pending[0]
        else:
            candidate = min(pending, key=lambda r: (_ezr_score(r, labeled), r["id"]))
        pending.remove(candidate)
        candidate = dict(candidate); candidate["y"] = callback(candidate["id"])
        labeled.append(candidate)
    return labeled, _best(labeled)

def choose_random(pool, budget, seed, callback):
    if budget < 1: raise ValueError("budget must be positive")
    if budget > len(pool): raise ValueError("budget exceeds candidates")
    pending = list(pool); random.Random(seed).shuffle(pending); labeled = []
    for candidate in pending[:budget]:
        candidate = dict(candidate); candidate["y"] = callback(candidate["id"])
        labeled.append(candidate)
    return labeled, _best(labeled)

def run_strategy(pool, budget, seed, strategy, outcomes=None):
    outcomes = {r["id"]: r["y"] for r in pool} if outcomes is None else dict(outcomes)
    calls = []
    def callback(identifier):
        if identifier in calls: raise AssertionError("duplicate evaluation")
        calls.append(identifier); return outcomes[identifier]
    records, recommendation = strategy(pool, budget, seed, callback)
    return {"seed": seed, "order": calls, "budget": len(calls), "recommendation": recommendation, "records": records}

def compare(pool, budget=20, seeds=range(42, 52), tolerance=TOLERANCE):
    result = []
    for seed in seeds:
        a = run_strategy(pool, budget, seed, choose_ezr)
        b = run_strategy(pool, budget, seed, choose_random)
        gap = a["recommendation"]["y"] - b["recommendation"]["y"]
        result.append((a, b, gap, "win" if gap < -tolerance else "loss" if gap > tolerance else "tie"))
    return result
