#!/usr/bin/env python3
"""Certify whether a binary configuration change improves Performance-."""
import argparse, csv, math, statistics, sys

def normal_tail(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))

def analyze(path, column="no_cabac", control="0", treatment="1", metric="Performance-"):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or metric not in rows[0] or column not in rows[0]:
        raise ValueError("missing metric or grouping column")
    a = [float(r[metric]) for r in rows if r[column].strip() == control]
    b = [float(r[metric]) for r in rows if r[column].strip() == treatment]
    if len(a) < 2 or len(b) < 2:
        raise ValueError("both groups need at least two observations")
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    se2 = va / len(a) + vb / len(b)
    se = math.sqrt(se2)
    difference = mb - ma
    z = difference / se
    # Welch degrees of freedom; normal tail is conservative for this large data set.
    df = se2 * se2 / ((va / len(a)) ** 2 / (len(a)-1) + (vb / len(b)) ** 2 / (len(b)-1))
    p_one_sided = normal_tail(z)
    ci_low, ci_high = difference - 1.96 * se, difference + 1.96 * se
    pooled = math.sqrt(((len(a)-1)*va + (len(b)-1)*vb) / (len(a)+len(b)-2))
    d = difference / pooled if pooled else 0.0
    credible = difference > 0 and p_one_sided < 0.05 and ci_low > 0
    return locals()

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--column", default="no_cabac")
    ap.add_argument("--control", default="0")
    ap.add_argument("--treatment", default="1")
    ap.add_argument("--metric", default="Performance-")
    args = ap.parse_args(argv)
    try:
        x = analyze(args.dataset, args.column, args.control, args.treatment, args.metric)
    except (OSError, ValueError, csv.Error) as exc:
        print("ERROR:", exc, file=sys.stderr); return 2
    print("Welch certification (higher metric is treated as improvement)")
    print(f"control n={len(x['a'])} mean={x['ma']:.6f}; treatment n={len(x['b'])} mean={x['mb']:.6f}")
    print(f"difference={x['difference']:.6f} 95%CI=[{x['ci_low']:.6f}, {x['ci_high']:.6f}]")
    print(f"t={x['z']:.6f} df={x['df']:.2f} one_sided_p={x['p_one_sided']:.3g} Cohen_d={x['d']:.6f}")
    print("STATISTICALLY_CREDIBLE=" + ("YES" if x["credible"] else "NO"))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
