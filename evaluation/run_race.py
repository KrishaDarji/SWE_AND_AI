"""Run race adapters (step 10 code) under the shared protocol and save each program's picks."""
import json, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def main():
    ds = [a for a in sys.argv[1:] if not a.isdigit()] or ["x264", "health_closedissues"]
    ts = [int(a) for a in sys.argv[1:] if a.isdigit()] or [1, 2, 3, 4, 5, 6, 7]
    import os
    budget = os.environ.get("RACE_BUDGET", "20")
    out = ROOT / "evaluation/out" / ("race" if budget == "20" else f"race_b{budget}"); out.mkdir(parents=True, exist_ok=True)
    for t in ts:
        for tr in ((1,) if t == 7 else (1, 2, 3)):
            for d in ds:
                t0 = time.time()
                p = subprocess.run([sys.executable, "-B", str(ROOT / "evaluation/worker.py"), "race", str(t), str(tr), d],
                                   capture_output=True, text=True, timeout=1800)
                tag = f"T{t} tr{tr} {d:20}"
                try:
                    res = json.loads(p.stdout.strip().splitlines()[-1])
                except Exception:
                    print(f"{tag} FAIL {time.time()-t0:5.1f}s {(p.stderr.strip().splitlines() or ['?'])[-1][:170]}"); continue
                if isinstance(res, dict):
                    print(f"{tag} {res}"); json.dump(res, open(out / f"T{t}_tr{tr}_{d}.json", "w")); continue
                json.dump(res, open(out / f"T{t}_tr{tr}_{d}.json", "w"))
                print(f"{tag} ok {time.time()-t0:5.1f}s picks={[round(x,3) for x in res]}")

if __name__ == "__main__":
    main()
