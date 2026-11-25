#!/usr/bin/env python3
"""Certify a fixed historical configuration comparison using the stdlib only."""
import csv, math, sys
from summarize_runs import finite_number, stats

REFERENCE = "squared_error"
CANDIDATE = "absolute_error"
PRIMARY = "MRE-"
SECONDARY = {"ACC+": "maximize", "PRED40+": "maximize"}

def _clean(v):
    return "" if v is None else v.strip().lower()

def _betacf(a, b, x):
    # Numerical Recipes' continued fraction, with convergence guards.
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    tiny = 1e-300
    d = max(abs(d), tiny) * (1 if d >= 0 else -1); d = 1.0 / d
    h = d
    for m in range(1, 10000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; d = (tiny if d == 0 else d); c = 1.0 + aa / c; c = (tiny if c == 0 else c)
        d = 1.0 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; d = (tiny if d == 0 else d); c = 1.0 + aa / c; c = (tiny if c == 0 else c)
        d = 1.0 / d; delta = d * c; h *= delta
        if abs(delta - 1.0) < 3e-14: return h
    raise ArithmeticError("incomplete-beta continued fraction did not converge")

def regularized_beta(x, a, b):
    if not (0 <= x <= 1 and a > 0 and b > 0): raise ValueError("invalid beta arguments")
    if x in (0, 1): return x
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - math.lgamma(a) - math.lgamma(b) + math.lgamma(a+b))
    return front * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - front * _betacf(b, a, 1-x) / b

def t_cdf(t, df):
    if not (df > 0 and math.isfinite(t)): raise ValueError("invalid t arguments")
    if t == 0: return 0.5
    z = df / (df + t*t); tail = 0.5 * regularized_beta(z, df/2, 0.5)
    return 1-tail if t > 0 else tail

def t_two_sided_p(t, df):
    """Two-sided probability computed from the tail beta expression directly."""
    if not math.isfinite(t): return 0.0
    z = df / (df + t*t)
    return regularized_beta(z, df/2, 0.5)

def t_quantile(p, df):
    if not (0 < p < 1 and df > 0): raise ValueError("invalid quantile arguments")
    lo, hi = -100.0, 100.0
    for _ in range(160):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < p: lo = mid
        else: hi = mid
    return (lo + hi) / 2

def _finite(rows, outcome, criterion):
    return [x for row in rows if _clean(row.get("criterion")) == criterion
            for x in [finite_number(row.get(outcome, ""))] if x is not None]

def _welch(a, b):
    # a is candidate, b reference; all returned differences are candidate-reference.
    if len(a) < 2 or len(b) < 2: return {"defined": False, "reason": "each group needs at least two finite outcomes"}
    ma, mb = sum(a)/len(a), sum(b)/len(b); va, vb = stats(a)["sample_variance"], stats(b)["sample_variance"]
    diff, se2 = ma-mb, va/len(a) + vb/len(b)
    if se2 == 0: return {"defined": False, "reason": "zero standard error; inference is undefined", "candidate_mean": ma, "reference_mean": mb, "difference": diff, "candidate_variance": va, "reference_variance": vb}
    se = math.sqrt(se2); df = se2*se2 / ((va/len(a))**2/(len(a)-1) + (vb/len(b))**2/(len(b)-1))
    t = diff/se; p = t_two_sided_p(t, df); critical = t_quantile(.975, df)
    return {"defined": True, "candidate_mean": ma, "reference_mean": mb, "difference": diff, "candidate_variance": va, "reference_variance": vb, "standard_error": se, "df": df, "t": t, "p_two_sided": min(1.0, max(0.0, p)), "ci95": [diff-critical*se, diff+critical*se]}

def certify(path, threshold=None):
    if threshold is not None and (not math.isfinite(threshold) or threshold < 0): raise ValueError("threshold must be finite and nonnegative")
    with open(path, newline="", encoding="utf-8") as f: rows = [{k:(v or "").strip() for k,v in r.items()} for r in csv.DictReader(f)]
    groups = {"candidate": _finite(rows, PRIMARY, CANDIDATE), "reference": _finite(rows, PRIMARY, REFERENCE)}
    primary = _welch(groups["candidate"], groups["reference"])
    secondary = {name: {"direction": direction, "candidate": stats(_finite(rows, name, CANDIDATE)), "reference": stats(_finite(rows, name, REFERENCE))} for name,direction in SECONDARY.items()}
    result = {"comparison": {"candidate": CANDIDATE, "reference": REFERENCE, "outcome": PRIMARY, "groups": {k: len(v) for k,v in groups.items()}}, "primary": primary, "secondary": secondary, "threshold": threshold}
    if threshold is not None and primary.get("defined"):
        ci = primary["ci95"]; result["practical"] = {"point_meets": primary["difference"] <= -threshold, "ci_supports": ci[1] <= -threshold, "threshold": threshold}
    return result

def render(result):
    lines = ["Certification: fixed historical configuration comparison", "Comparison: absolute_error minus squared_error on MRE- (lower is better)", "Primary groups: " + str(result["comparison"]["groups"]), "Primary inference: " + str(result["primary"])]
    lines.append("Secondary descriptive outcomes (ACC+ and PRED40+, maximize): " + str(result["secondary"]))
    lines.append("Practical threshold interpretation: " + str(result.get("practical", "undetermined without a threshold")))
    lines.append("Inference uses Welch's two-sided t test, Welch-Satterthwaite df, and a 95% t interval; incomplete beta continued fractions and bisection are stdlib numerical approximations.")
    lines.append("These are historical configurations, not controlled repeated evaluations of otherwise identical models; other hyperparameters may differ, so no causation or guaranteed improvement is claimed.")
    return "\n".join(lines)

if __name__ == "__main__":
    if len(sys.argv) not in (2, 3): raise SystemExit("usage: certify.py DATASET.csv [threshold]")
    threshold = None if len(sys.argv) == 2 else float(sys.argv[2])
    print(render(certify(sys.argv[1], threshold)))
