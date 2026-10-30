#!/usr/bin/env python3
"""Compare NASA93 observed effort for tool=n and tool=h."""
import csv, math, sys

def _clean(s):
    return s.strip() if isinstance(s, str) else s

def load_groups(path):
    groups = {"n": [], "h": []}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, skipinitialspace=False):
            tool = _clean(row.get("tool", ""))
            raw = _clean(row.get("Effort-", ""))
            if tool not in groups or raw in (None, "", "?"):
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if math.isfinite(value):
                groups[tool].append(value)
    return groups

def _mean_var(xs):
    if not xs:
        return float("nan"), float("nan")
    mean = 0.0
    m2 = 0.0
    for i, x in enumerate(xs, 1):
        delta = x - mean
        mean += delta / i
        m2 += delta * (x - mean)
    return mean, (m2 / (len(xs) - 1) if len(xs) > 1 else 0.0)

def _betacf(a, b, x):
    # Numerical Recipes continued fraction for the regularized beta.
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1e-30 if abs(d) < 1e-30 else d
    d = 1.0 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; d = 1e-30 if abs(d) < 1e-30 else d
        c = 1.0 + aa / c; c = 1e-30 if abs(c) < 1e-30 else c
        d, c, h = 1.0 / d, 1.0 / c, h * d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; d = 1e-30 if abs(d) < 1e-30 else d
        c = 1.0 + aa / c; c = 1e-30 if abs(c) < 1e-30 else c
        d, c = 1.0 / d, 1.0 / c
        delta = d * c; h *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return h

def _betai(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    front = math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1-x) / b

def t_cdf(t, df):
    if not math.isfinite(t) or not math.isfinite(df) or df <= 0:
        raise ValueError("t and df must be finite, with df > 0")
    if t == 0: return 0.5
    sign, z = (1, t) if t > 0 else (-1, -t)
    if df == 1: return 0.5 + sign * math.atan(z) / math.pi
    norm = math.exp(math.lgamma((df+1)/2)-math.lgamma(df/2)) / math.sqrt(df*math.pi)
    upper = math.atan(z)
    def f(theta):
        x = math.tan(theta); c = math.cos(theta)
        return norm * math.exp(-((df+1)/2)*math.log1p(x*x/df)) / (c*c)
    def simp(a, b, fa, fb, fm): return (b-a) * (fa + 4*fm + fb) / 6
    def adapt(a, b, fa, fb, fm, whole, tol, depth):
        m=(a+b)/2; lm=(a+m)/2; rm=(m+b)/2
        fl, fr=f(lm), f(rm); left=simp(a,m,fa,fm,fl); right=simp(m,b,fm,fb,fr)
        if depth <= 0 or abs(left+right-whole) <= 15*tol: return left+right+(left+right-whole)/15
        return adapt(a,m,fa,fm,fl,left,tol/2,depth-1)+adapt(m,b,fm,fb,fr,right,tol/2,depth-1)
    area = adapt(0.0, upper, f(0.0), f(upper), f(upper/2), simp(0.0, upper, f(0.0), f(upper), f(upper/2)), 1e-13, 30)
    out = 0.5 + sign * area
    if not math.isfinite(out) or out < -1e-12 or out > 1+1e-12: raise ArithmeticError("invalid t CDF")
    return out

def t_quantile(p, df):
    if not math.isfinite(p) or p <= 0 or p >= 1:
        raise ValueError("p must be finite and strictly between 0 and 1")
    if not math.isfinite(df) or df <= 0:
        raise ValueError("df must be finite and > 0")
    lo, hi = -1.0, 1.0
    while t_cdf(lo, df) > p: lo *= 2
    while t_cdf(hi, df) < p: hi *= 2
    for _ in range(100):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < p: lo = mid
        else: hi = mid
    return (lo + hi) / 2

def compare(a, b, threshold):
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    if len(a) < 2 or len(b) < 2:
        raise ValueError("each group needs at least two finite observations")
    ma, va = _mean_var(a); mb, vb = _mean_var(b)
    se2 = va/len(a) + vb/len(b)
    if se2 == 0:
        diff = ma - mb
        return {"n_a":len(a), "n_b":len(b), "mean_a":ma, "mean_b":mb, "difference":diff,
                "status":"undefined", "reason":"Welch standard error is zero because both groups are constant",
                "df":None, "se":0.0, "ci_low":None, "ci_high":None, "t":None, "p_two_sided":None,
                "threshold":threshold, "significant":None, "point_exceeds":abs(diff)>threshold,
                "entirely_practical":None}
    df = se2*se2 / ((va/len(a))**2/(len(a)-1) + (vb/len(b))**2/(len(b)-1))
    diff = ma - mb; se = math.sqrt(se2); t = diff/se
    crit = t_quantile(0.975, df); low, high = diff-crit*se, diff+crit*se
    p = max(0.0, min(1.0, 2 * (1-t_cdf(abs(t), df))))
    return {"n_a":len(a), "n_b":len(b), "mean_a":ma, "mean_b":mb, "difference":diff,
            "df":df, "se":se, "ci_low":low, "ci_high":high, "t":t, "p_two_sided":p,
            "threshold":threshold, "significant":p < .05, "point_exceeds":abs(diff)>threshold,
            "entirely_practical": low > threshold or high < -threshold}

def report(r):
    direction = "B (h)" if r["difference"] > 0 else "A (n)" if r["difference"] < 0 else "neither"
    infer = (f"Welch t={r['t']:.6g}, df={r['df']:.6g}, two-sided p={r['p_two_sided']:.6g}" if r.get("status") != "undefined"
             else f"Welch test undefined: {r['reason']}")
    interval = (f"95% CI [{r['ci_low']:.6g}, {r['ci_high']:.6g}]" if r.get("ci_low") is not None else "95% CI undefined")
    return (f"A=tool=n (n={r['n_a']}, mean={r['mean_a']:.6g}); B=tool=h (n={r['n_b']}, mean={r['mean_b']:.6g})\n"
            f"mean difference A-B={r['difference']:.6g}; {interval}\n{infer}\n"
            f"directional lower-effort point estimate favors {direction}; point exceeds threshold={r['point_exceeds']}; "
            f"entire CI beyond threshold={r['entirely_practical']}\n"
            "Effort- is observed effort, not verified monetary cost. Statistical difference and practical importance are separate; "
            "a CI crossing a practical threshold is uncertain, and non-significance is not equivalence. Skew, imbalance, and project-scope differences limit interpretation; no causal savings claim is made.")

def main(argv):
    if len(argv) != 3: raise SystemExit("usage: certify.py DATASET THRESHOLD")
    threshold = float(argv[2]); groups = load_groups(argv[1])
    result = compare(groups["n"], groups["h"], threshold)
    print(report(result))
    print("Assumptions: independent observations, finite numeric Effort-, two-sided alpha=.05, Welch unequal variances.")
    print("EZR adaptations: stripped CSV cells like ezr-lib thing/csv, one-pass NUM mean/M2 variance, and explicit group summaries.")

if __name__ == "__main__": main(sys.argv)
