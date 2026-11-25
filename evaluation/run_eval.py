"""Run adapters in subprocesses and save predictions. usage: run_eval.py guess [T ...] [--probe]"""
import json, subprocess, sys, time, csv, statistics as st
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DS = ("x264", "nasa93dem", "health_closedissues")
TARGET = {"x264": "Performance-", "nasa93dem": "Effort-", "health_closedissues": "MRE-"}

def actual(d):
    with open(ROOT / "evaluation/splits" / d / "test.csv", newline="") as f:
        return [float(r[TARGET[d]]) for r in csv.DictReader(f)]

def main():
    skill = sys.argv[1]
    ts = [int(a) for a in sys.argv[2:] if a.isdigit()] or [1, 2, 3, 4, 5, 6, 7]
    out = ROOT / "evaluation/out" / skill; out.mkdir(parents=True, exist_ok=True)
    for t in ts:
        for tr in ((1,) if t == 7 else (1, 2, 3)):
            for d in DS:
                t0 = time.time()
                p = subprocess.run([sys.executable, "-B", str(ROOT / "evaluation/worker.py"), skill, str(t), str(tr), d],
                                   capture_output=True, text=True, timeout=900)
                dt = time.time() - t0
                tag = f"T{t} tr{tr} {d:20}"
                try:
                    res = json.loads(p.stdout.strip().splitlines()[-1])
                except Exception:
                    err = (p.stderr.strip().splitlines() or ["?"])[-1]
                    print(f"{tag} FAIL {dt:5.1f}s {err[:160]}"); continue
                if isinstance(res, dict):
                    print(f"{tag} {res}"); continue
                json.dump(res, open(out / f"T{t}_tr{tr}_{d}.json", "w"))
                a = actual(d)
                mae = st.mean(abs(x - y) for x, y in zip(res, a)) if len(res) == len(a) else None
                print(f"{tag} ok {dt:5.1f}s n={len(res)} MAE={mae:.4g}" if mae is not None else f"{tag} LEN MISMATCH {len(res)}")

if __name__ == "__main__":
    main()
