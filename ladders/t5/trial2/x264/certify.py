"""Independent certification of Performance- for the fixed X264 comparison."""
import csv
import math
import sys


def _betacf(a, b, x):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1e-300 if abs(d) < 1e-300 else d
    d = 1.0 / d
    h = d
    for m in range(1, 10000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1e-300 if abs(d) < 1e-300 else d
        c = 1.0 + aa / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1e-300 if abs(d) < 1e-300 else d
        c = 1.0 + aa / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return h


def _betai(a, b, x):
    if not 0.0 <= x <= 1.0:
        raise ValueError("beta domain")
    if x == 0.0 or x == 1.0:
        return x
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t, df):
    """CDF using the incomplete-beta identity; normal limit for huge df."""
    if not math.isfinite(t) or not math.isfinite(df) or df <= 0:
        raise ValueError("finite t and positive df required")
    if df > 1e7:
        return 0.5 * math.erfc(-t / math.sqrt(2.0))
    x = df / (df + t * t)
    tail = 0.5 * _betai(df / 2.0, 0.5, x)
    return 1.0 - tail if t >= 0 else tail


def _mean_var(values):
    n = len(values)
    mean = math.fsum(values) / n
    var = math.fsum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, var


def welch(a, b, threshold=5.0, alpha=0.05):
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    if len(a) < 2 or len(b) < 2:
        raise ValueError("each group needs at least two finite values")
    if any(not math.isfinite(x) for x in a + b):
        raise ValueError("values must be finite")
    ma, va = _mean_var(a); mb, vb = _mean_var(b)
    if va == 0.0 and vb == 0.0:
        return {"n_a": len(a), "n_b": len(b), "mean_a": ma, "mean_b": mb,
                "difference": ma - mb, "threshold": threshold,
                "exceeds": abs(ma - mb) > threshold, "beyond": None,
                "status": "undefined", "reason": "Welch inference is undefined when both groups have zero variance"}
    se2 = va / len(a) + vb / len(b)
    se = math.sqrt(se2)
    d = ma - mb
    df = se2 * se2 / ((va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1))
    t = d / se
    p = 2.0 * (1.0 - student_t_cdf(abs(t), df))
    # 95% critical value by monotone bisection, avoiding a package dependency.
    target = 1.0 - alpha / 2.0
    lo, hi = 0.0, 1.0
    while student_t_cdf(hi, df) < target:
        hi *= 2.0
    for _ in range(90):
        mid = (lo + hi) / 2.0
        if student_t_cdf(mid, df) < target: lo = mid
        else: hi = mid
    margin = hi * se
    ci = (d - margin, d + margin)
    return {"n_a": len(a), "n_b": len(b), "mean_a": ma, "mean_b": mb,
            "difference": d, "df": df, "t": t, "p": p, "ci": ci,
            "threshold": threshold, "exceeds": abs(d) > threshold, "status": "valid",
            "beyond": ci[0] > threshold or ci[1] < -threshold}


def load_groups(path):
    a, b, skipped = [], [], 0
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                value = float(row["Performance-"])
                group = int(row["no_8x8dct"])
                if not math.isfinite(value) or group not in (0, 1): raise ValueError
            except (KeyError, TypeError, ValueError):
                skipped += 1
                continue
            (a if group == 0 else b).append(value)
    return a, b, skipped


def main(argv):
    if len(argv) != 2:
        raise SystemExit("usage: python source/certify.py DATASET")
    a, b, skipped = load_groups(argv[1])
    result = welch(a, b)
    print("Fixed comparison: A=no_8x8dct=0, B=no_8x8dct=1")
    print("Lower Performance- is better; rows are configurations, not repeated runs.")
    print("n_A={n_a} n_B={n_b} mean_A={mean_a:.6f} mean_B={mean_b:.6f}".format(**result))
    if result["status"] == "undefined":
        print("Welch inference status=undefined; reason={}".format(result["reason"]))
        print("mean difference A-B={difference:.6f}; no Welch p-value or interval is reported.".format(**result))
        print("illustrative threshold=5.0; point exceeds threshold={}; interval interpretation=undefined".format(result["exceeds"]))
        print("Conclusion: statistical significance and practical-importance direction are undefined.")
        print("Skipped invalid rows={}; configuration variation cannot establish measurement noise or causality.".format(skipped))
        return
    print("mean difference A-B={difference:.6f}, Welch t={t:.6f}, df={df:.6f}, two-sided p={p:.8g}".format(**result))
    print("95% Welch interval for A-B=({:.6f}, {:.6f})".format(*result["ci"]))
    print("illustrative threshold=5.0; point exceeds threshold={}; entire interval beyond threshold={}".format(result["exceeds"], result["beyond"]))
    print("Conclusion: statistically different from zero={} ; practical importance is {}.".format(result["p"] < 0.05, "supported" if result["beyond"] else ("uncertain" if result["exceeds"] else "not shown")))
    print("Skipped invalid rows={}; configuration variation cannot establish measurement noise or causality.".format(skipped))


if __name__ == "__main__":
    main(sys.argv)
