"""Treatment 7 (human team, team's prior skills, reuse) trial runner.

One trial per domain: a human team cannot be reset between repeats, so
repeats would not be independent. Domains run in project order; a shared
file counts as "new" only the first time any domain needs it, so domains 2
and 3 show the transfer cost (ideally just their adapter).

    python3 experiments/run_treatment7.py
"""

import csv
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.domains import DATASETS  # noqa: E402
from src.ladder import SKILLS  # noqa: E402

TRIAL = 1
OUT = ROOT / f"experiments/runs/treatment7/trials/trial{TRIAL}"
DOMAINS = ["x264", "nasa93dem", "health_closedissues"]
TESTS = ["tests/test_datasets.py", "tests/test_ladder.py"]
COLUMNS = ["step", "treatment", "trial", "domain", "skill", "total_loc", "new_loc", "test_pass",
           "runtime_seconds", "error_count", "notes", "reused_loc", "tests_run", "quality"]

# Source files each step introduces: its core skill module + its ladder demo.
STEP_FILES = {step: [f"src/core/{skill}.py", f"src/ladder/step{step:02d}_{skill}.py"]
              for step, skill in enumerate(SKILLS, start=1)}
# Step 1 also brings the shared plumbing.
STEP_FILES[1] += ["src/__init__.py", "src/core/__init__.py", "src/core/schema.py", "src/core/datasets.py",
                  "src/domains/__init__.py", "src/ladder/__init__.py", "src/cli.py"]


def loc(path: str) -> int:
    """Non-blank, non-comment source lines."""
    lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
    return sum(1 for line in lines if line.strip() and not line.strip().startswith("#"))


def run(command):
    start = time.perf_counter()
    done = subprocess.run([sys.executable, *command], cwd=ROOT, capture_output=True, text=True)
    return done, round(time.perf_counter() - start, 2)


def quality(step: int, r: dict) -> str:
    """One-line headline of the step's outcome, for the metrics table."""
    if step == 2:
        return "; ".join(f"{o} MAE {v['model_mae']:.3g} vs mean-baseline {v['baseline_mae']:.3g}" for o, v in r["errors"].items())
    if step == 3:
        return "; ".join(f"{o}: wins={v['treatment_wins']} same={v['same']} p={v['p_value']:.3f}"
                         for o, v in r.items() if isinstance(v, dict))
    if step == 4:
        return f"stopped after {r['seen']} of {r['available']} rows"
    if step == 5:
        return f"top factor {r['findings'][0]['feature']} for {r['findings'][0]['objective']}"
    if step == 7:
        return f"score {r['score']:.3f}, beats {r['beats_fraction_of_all_rows']:.1%} of rows, {r['evaluated']} evaluations"
    if step == 8:
        return f"{r['changed_feature']}: {r['old_value']} -> {r['new_value']}, predicted gain {r['improvement']:.3f}"
    if step == 9:
        return f"most unusual row index {r['findings'][0]['row_index']} (z {r['findings'][0]['score']:.2f})"
    if step == 10:
        return f"shared {r['shared_median']:.3f} vs random {r['baseline_median']:.3f}, same={r['same']}, wins={r['shared_wins']}"
    return ""


def run_step(domain: str, step: int, seen: set) -> dict:
    spec = DATASETS[domain]
    out = OUT / domain / f"step{step:02d}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    adapter = f"src/domains/{domain}/__init__.py"
    files = [adapter] + [f for s in range(1, step + 1) for f in STEP_FILES[s]]
    new_files = [f for f in files if f not in seen and (f != adapter or step == 1)]
    for path in files + TESTS:
        target = out / ("tests" if path in TESTS else "source") / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, target)
    (out / "requirement.txt").write_text(spec.requirements[step] + "\n", encoding="utf-8")

    demo, seconds = run(["-m", "src.cli", domain, str(step)])
    (out / "runtime_output.txt").write_text(demo.stdout + demo.stderr, encoding="utf-8")
    tests, _ = run(["-m", "unittest", "discover", "-s", "tests", "-k", f"step{step}_"])
    (out / "test_output.txt").write_text(tests.stdout + tests.stderr, encoding="utf-8")

    ran = int(re.search(r"Ran (\d+) test", tests.stderr).group(1))
    failed = sum(int(n) for n in re.findall(r"(?:failures|errors)=(\d+)", tests.stderr))
    errors = failed + (demo.returncode != 0)
    total, new = sum(map(loc, files)), sum(map(loc, new_files))
    headline = quality(step, json.loads(demo.stdout)["result"]) if demo.returncode == 0 else "demo failed"
    seen.update(files)
    row = {
        "step": step, "treatment": 7, "trial": TRIAL, "domain": spec.label, "skill": SKILLS[step - 1],
        "total_loc": total, "new_loc": new, "test_pass": int(tests.returncode == 0 and demo.returncode == 0),
        "runtime_seconds": seconds, "error_count": errors,
        "notes": f"new files: {', '.join(new_files) or 'none (all reused)'}; LOC = non-blank non-comment source lines, tests excluded",
        "reused_loc": total - new, "tests_run": ran, "quality": headline,
    }
    write_csv(out / "metrics.csv", [row])
    return row


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    seen, everything = set(), []
    for domain in DOMAINS:
        rows = [run_step(domain, step, seen) for step in range(1, 11)]
        write_csv(OUT / domain / "metrics.csv", rows)
        everything += rows
        for row in rows:
            print(f"{domain:20} step {row['step']:2} {row['skill']:9} new {row['new_loc']:4} "
                  f"total {row['total_loc']:4} pass {row['test_pass']}  {row['quality']}")
    write_csv(OUT / "metrics.csv", everything)


if __name__ == "__main__":
    main()
