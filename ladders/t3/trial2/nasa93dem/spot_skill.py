#!/usr/bin/env python3
"""Independent anomaly spotting for NASA93DEM."""
import argparse, csv, math, os, statistics, time
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, RobustScaler

EXPECTED = ["idX","centerX","YearX","prec","flex","resl","team","pmat","rely","data","cplx","ruse","docu","time","stor","pvol","acap","pcap","pcon","apex","plex","ltex","tool","site","sced","Kloc+","Effort-","Defects-","MonthsX"]
OUTCOMES = {"Kloc+", "Effort-", "Defects-", "MonthsX"}

def load_validate(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or rows[0].keys() != set(EXPECTED):
        raise ValueError("header does not match NASA93DEM schema")
    for i, r in enumerate(rows, 2):
        for col in EXPECTED:
            if not str(r[col]).strip(): raise ValueError(f"blank {col} at CSV line {i}")
        for col in ("idX","centerX","YearX") + tuple(OUTCOMES):
            float(r[col])
    # NASA93DEM contains repeated project identifiers; they are valid source data.
    # Validation therefore checks type/availability, not uniqueness.
    return rows

def _mad_z(x, values):
    med = statistics.median(values); mad = statistics.median(abs(v-med) for v in values)
    return 0.6745 * (x-med) / (mad if mad else 1.0)

def analyze(path):
    rows = load_validate(path)
    cat = [c for c in EXPECTED if c not in ("idX","centerX","YearX") and c not in OUTCOMES]
    num = ["centerX","YearX"] + sorted(OUTCOMES)
    Xcat = [[r[c].strip() for c in cat] for r in rows]
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    A = enc.fit_transform(Xcat)
    B = RobustScaler().fit_transform([[float(r[c]) for c in num] for r in rows])
    X = __import__("numpy").hstack([A, B])
    model = IsolationForest(n_estimators=400, contamination="auto", random_state=20261029, n_jobs=1)
    model.fit(X)
    scores = -model.score_samples(X)
    order = sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)
    top = order[0]
    explanations = []
    for c in num:
        vals = [float(r[c]) for r in rows]
        explanations.append((abs(_mad_z(float(rows[top][c]), vals)), c, float(rows[top][c]), statistics.median(vals)))
    for c in cat:
        vals = [r[c].strip() for r in rows]
        freq = vals.count(rows[top][c].strip()) / len(vals)
        explanations.append((1.0/freq, c, rows[top][c].strip(), "frequency=" + f"{freq:.3f}"))
    explanations.sort(reverse=True)
    return rows, scores, top, explanations[:8]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("csv_path"); args = ap.parse_args()
    start = time.perf_counter()
    rows, scores, top, details = analyze(args.csv_path)
    r = rows[top]
    print(f"validated_rows={len(rows)}")
    print(f"method=IsolationForest(n_estimators=400, contamination=auto, random_state=20261029)")
    print(f"suspicious_rank=1 total={len(rows)} idX={r['idX']} anomaly_score={scores[top]:.6f}")
    print("record=" + ",".join(f"{k}={r[k].strip()}" for k in EXPECTED))
    print("unusual_features=")
    for magnitude, feature, value, reference in details:
        print(f"  {feature}: value={value} reference={reference} deviation_strength={magnitude:.3f}")
    print("interpretation=Flag is a historical inconsistency, not proof of an error or bad project outcome.")
    print("data_quality_checks=inspect transcription, units, and outcome definitions; rare categorical levels may be valid.")
    print("legitimate_exception=an unusual mission, team, technology, or delivery context could explain the profile.")
    print("limitations=small observational dataset; correlated features, hidden project context, and one flagged rank create uncertainty; no causal claim.")
    print(f"runtime_seconds={time.perf_counter()-start:.6f}")

if __name__ == "__main__": main()
