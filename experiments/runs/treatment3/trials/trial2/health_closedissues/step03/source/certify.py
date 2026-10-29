#!/usr/bin/env python3
"""Independent certify analysis for the Health-ClosedIssues HPO dataset."""
from __future__ import annotations
import argparse, csv, math, random, statistics, sys, time
from pathlib import Path
from sklearn.preprocessing import StandardScaler

TARGETS = {"MRE-": -1.0, "ACC+": 1.0, "PRED40+": 1.0}
FAMILY = "criterion"
CONFIGURATION_FEATURES = ["N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]

def load_validate(path: Path):
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    required = set(TARGETS) | set(CONFIGURATION_FEATURES)
    if not rows or not required <= set(rows[0]):
        raise ValueError("missing required columns or empty dataset")
    for i, row in enumerate(rows, 2):
        if any(row[k] == "" for k in required): raise ValueError(f"missing value at row {i}")
        for k in TARGETS: float(row[k])
    return rows

def values(rows, target):
    return [float(r[target]) for r in rows]

def quality_scores(rows):
    raw = [[TARGETS[k] * float(r[k]) for k in TARGETS] for r in rows]
    scaled = StandardScaler().fit_transform(raw)
    return [sum(row) / len(TARGETS) for row in scaled]

def mean(xs): return statistics.fmean(xs)

def bootstrap_ci(a, b, seed=20261029, n=4000):
    rng = random.Random(seed); diffs=[]
    for _ in range(n):
        diffs.append(mean([a[rng.randrange(len(a))] for _ in a]) - mean([b[rng.randrange(len(b))] for _ in b]))
    diffs.sort()
    return diffs[int(.025*n)], diffs[int(.975*n)-1]

def permutation_p(a, b, seed=20261029, n=10000):
    observed = mean(a) - mean(b); joined = a + b; na=len(a); rng=random.Random(seed); extreme=0
    for _ in range(n):
        rng.shuffle(joined)
        if mean(joined[:na]) - mean(joined[na:]) >= observed: extreme += 1
    return (extreme + 1) / (n + 1)

def analyze(rows):
    families = sorted({r[FAMILY] for r in rows})
    groups = {f:[i for i,r in enumerate(rows) if r[FAMILY] == f] for f in families}
    scores=quality_scores(rows)
    ranking=sorted(families, key=lambda f: mean([scores[i] for i in groups[f]]), reverse=True)
    a_name,b_name=ranking[:2]; a=[scores[i] for i in groups[a_name]]; b=[scores[i] for i in groups[b_name]]
    result={"family_a":a_name,"family_b":b_name,"n_a":len(a),"n_b":len(b),"mean_a":mean(a),"mean_b":mean(b),"effect":mean(a)-mean(b),"ci_low":bootstrap_ci(a,b)[0],"ci_high":bootstrap_ci(a,b)[1],"p_value":permutation_p(a,b),"targets":{}}
    for t in TARGETS:
        av=[float(rows[i][t]) for i in groups[a_name]]; bv=[float(rows[i][t]) for i in groups[b_name]]
        d=mean(av)-mean(bv)
        if TARGETS[t] < 0: d=-d
        result["targets"][t]={"mean_a":mean(av),"mean_b":mean(bv),"directional_difference":d}
    return result

def render(path, result, elapsed):
    credible=result["ci_low"] > 0 and result["p_value"] < .05
    lines=["CERTIFY: Health-ClosedIssues",f"runtime_seconds={elapsed:.6f}","family_feature=criterion",f"family_A={result['family_a']} (n={result['n_a']})",f"family_B={result['family_b']} (n={result['n_b']})", "primary_metric=direction-aware mean of standardized MRE-, ACC+, PRED40+",f"primary_means={result['mean_a']:.6f} vs {result['mean_b']:.6f}",f"effect_A_minus_B={result['effect']:.6f}",f"bootstrap_95pct_CI=[{result['ci_low']:.6f}, {result['ci_high']:.6f}]",f"one_sided_permutation_p={result['p_value']:.6f}",f"statistically_credible={'yes' if credible else 'no'}"]
    lines.append("target_means_directional_difference:")
    for t,x in result["targets"].items(): lines.append(f"  {t}: A={x['mean_a']:.6f}, B={x['mean_b']:.6f}, A_better_minus_B={x['directional_difference']:.6f}")
    lines += ["interpretation=The observed family difference is statistically credible only if the CI excludes zero and p<0.05; this result is an association within sampled configurations, not proof that criterion causes the difference.","practical_significance=The standardized composite effect is the estimated average quality separation in standard-deviation units; assess target-specific differences and deployment tolerance before treating it as practically important.","assumptions=Rows are treated as exchangeable within families for permutation; bootstrap observations approximate uncertainty; the three targets are equally weighted after direction-aware standardization.","limitations=Observational hyperparameter records may be imbalanced, repeated or dependent; family choice is data-derived; permutation and bootstrap Monte Carlo error remains; no causal adjustment or external validation is performed."]
    return "\n".join(lines)+"\n"

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",default="moot/optimize/hpo/Health-ClosedIssues0000.csv"); ap.add_argument("--output",default=None); args=ap.parse_args(argv)
    start=time.perf_counter(); rows=load_validate(Path(args.dataset)); result=analyze(rows); out=render(args.dataset,result,time.perf_counter()-start)
    if args.output: Path(args.output).write_text(out)
    else: sys.stdout.write(out)

if __name__ == "__main__": main()
