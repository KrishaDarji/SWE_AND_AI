#!/usr/bin/env python3
"""Certify whether observed Effort- differs between tool=n and tool=h."""
import argparse, csv, math

EPS = 1e-14

def clean(value):
    return value.strip()

def finite_number(value):
    try:
        number = float(clean(value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

def load_groups(path):
    groups = {"n": [], "h": []}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = csv.reader(handle)
        header = [clean(x) for x in next(rows)]
        names = {name: i for i, name in enumerate(header)}
        try:
            tool_i, effort_i = names["tool"], names["Effort-"]
        except KeyError as exc:
            raise ValueError("dataset must contain tool and Effort- columns") from exc
        for row in rows:
            if len(row) <= max(tool_i, effort_i):
                continue
            tool = clean(row[tool_i])
            effort = finite_number(row[effort_i])
            if tool in groups and effort is not None:
                groups[tool].append(effort)
    return groups

def summary(values):
    n = mean = m2 = 0.0
    for value in values:
        n += 1.0
        delta = value - mean
        mean += delta / n
        m2 += delta * (value - mean)
    if not n:
        return {"n": 0, "mean": math.nan, "variance": math.nan, "sd": math.nan}
    variance = m2 / (n - 1.0) if n > 1 else 0.0
    return {"n": int(n), "mean": mean, "variance": max(0.0, variance),
            "sd": math.sqrt(max(0.0, variance))}

def log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

def beta_cf(a, b, x):
    # Continued fraction from Numerical Recipes; iteration is a standard-library
    # approximation to the regularized incomplete beta function.
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / max(abs(d), EPS) * (1 if d >= 0 else -1)
    h = d
    for m in range(1, 10001):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = (1.0 / max(abs(d), EPS)) * (1 if d >= 0 else -1)
        c = 1.0 + aa / max(c, EPS)
        c = math.copysign(max(abs(c), EPS), c)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = (1.0 / max(abs(d), EPS)) * (1 if d >= 0 else -1)
        c = 1.0 + aa / max(c, EPS)
        c = math.copysign(max(abs(c), EPS), c)
        change = d * c
        h *= change
        if abs(change - 1.0) < 3e-14:
            break
    return h

def regularized_beta(x, a, b):
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - log_beta(a, b))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * beta_cf(a, b, x) / a
    return 1.0 - front * beta_cf(b, a, 1.0 - x) / b

def t_cdf(t, df):
    if not (df > 0 and math.isfinite(t)): return math.nan
    x = df / (df + t * t)
    tail = 0.5 * regularized_beta(x, df / 2.0, 0.5)
    return 1.0 - tail if t >= 0 else tail

def t_quantile(prob, df):
    if not 0 < prob < 1 or df <= 0: raise ValueError("invalid quantile inputs")
    lo, hi = -1000.0, 1000.0
    for _ in range(160):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < prob: lo = mid
        else: hi = mid
    return (lo + hi) / 2.0

def welch(a, b):
    sa, sb = summary(a), summary(b)
    if sa["n"] < 2 or sb["n"] < 2:
        raise ValueError("each group needs at least two finite observations")
    va, vb, na, nb = sa["variance"], sb["variance"], sa["n"], sb["n"]
    se2 = va / na + vb / nb
    difference = sa["mean"] - sb["mean"]
    if se2 == 0.0:
        if difference == 0.0: return {"difference": 0.0, "df": math.nan, "t": math.nan, "p": 1.0, "se": 0.0, "ci": (0.0, 0.0)}
        return {"difference": difference, "df": math.nan, "t": math.inf, "p": 0.0, "se": 0.0, "ci": (difference, difference)}
    se = math.sqrt(se2)
    df = se2 * se2 / ((va / na) ** 2 / (na - 1.0) + (vb / nb) ** 2 / (nb - 1.0))
    t = difference / se
    p = 2.0 * min(t_cdf(t, df), 1.0 - t_cdf(t, df))
    critical = t_quantile(0.975, df)
    return {"difference": difference, "df": df, "t": t, "p": p, "se": se,
            "ci": (difference - critical * se, difference + critical * se)}

def certify(groups, threshold):
    if not math.isfinite(threshold) or threshold < 0: raise ValueError("threshold must be finite and nonnegative")
    result = welch(groups["n"], groups["h"])
    a, b = summary(groups["n"]), summary(groups["h"])
    cutoff = threshold
    result.update(a=a, b=b, statistically_detected=result["p"] < 0.05,
                  point_exceeds=abs(result["difference"]) > cutoff,
                  interval_above=result["ci"][0] > cutoff,
                  interval_below=result["ci"][1] < -cutoff,
                  direction="favors B (tool=h)" if result["difference"] > 0 else "favors A (tool=n)" if result["difference"] < 0 else "no point-estimate direction")
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()
    groups = load_groups(args.dataset)
    result = certify(groups, args.threshold)
    print("Comparison: A=tool=n versus B=tool=h; outcome=observed Effort-")
    print(f"A count={result['a']['n']} mean={result['a']['mean']:.6f} sd={result['a']['sd']:.6f}")
    print(f"B count={result['b']['n']} mean={result['b']['mean']:.6f} sd={result['b']['sd']:.6f}")
    print(f"A-B difference={result['difference']:.6f} direction={result['direction']}")
    print(f"Welch t={result['t']:.6f} df={result['df']:.6f} two-sided p={result['p']:.6g}")
    print(f"95% CI for A-B=({result['ci'][0]:.6f}, {result['ci'][1]:.6f})")
    print(f"illustrative threshold={args.threshold:.6f} (10% of tool=n mean; not a client requirement)")
    print(f"statistical_difference={result['statistically_detected']} point_exceeds_threshold={result['point_exceeds']} interval_above_threshold={result['interval_above']} interval_below_negative_threshold={result['interval_below']}")
    print("Interpretation: observed effort is not verified monetary cost; this does not establish causality, savings, or equivalence.")
    print("Limitations: effort may be skewed, groups may be imbalanced, and project scope may differ; repeated-run noise is not measured.")

if __name__ == "__main__": main()
