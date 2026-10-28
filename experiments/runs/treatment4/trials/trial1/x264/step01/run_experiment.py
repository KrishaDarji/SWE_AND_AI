"""Run tests and the application, preserving logs and measuring actual runtime."""

import csv
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    step = Path(__file__).resolve().parent
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 run_experiment.py DATASET")
    dataset = Path(sys.argv[1]).resolve()
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", X264_DATASET=str(dataset))
    before = hashlib.sha256(dataset.read_bytes()).hexdigest()
    test_command = [sys.executable, "-B", "-m", "unittest", "discover", "-s",
                    str(step / "tests"), "-v"]
    runtime_command = [sys.executable, "-B", str(step / "source" / "remember.py"),
                       str(dataset), "--output-dir", str(step)]
    results = []
    for label, command in [("test", test_command), ("runtime", runtime_command)]:
        start = time.perf_counter()
        process = subprocess.run(command, capture_output=True, text=True, env=environment)
        elapsed = time.perf_counter() - start
        with (step / f"{label}_output.txt").open("a", encoding="utf-8") as stream:
            stream.write(f"Command argv: {command!r}\n")
            stream.write(f"Python: {sys.version}\nscikit-learn: {version('scikit-learn')}\n")
            stream.write(process.stdout)
            stream.write(process.stderr)
            stream.write(f"Exit code: {process.returncode}\nElapsed seconds: {elapsed:.9f}\n")
        results.append((process.returncode, elapsed))
    unchanged = before == hashlib.sha256(dataset.read_bytes()).hexdigest()
    with (step / "runtime_output.txt").open("a", encoding="utf-8") as stream:
        stream.write(f"Dataset SHA256 before: {before}\nDataset unchanged: {unchanged}\n")
    total_loc = sum(1 for path in (step / "source").rglob("*.py")
                    for line in path.read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith("#"))
    fields = ["step", "treatment", "trial", "domain", "skill", "total_loc", "new_loc",
              "test_pass", "runtime_seconds", "error_count", "notes"]
    errors = 2 + sum(code != 0 for code, _ in results) + int(not unchanged)
    notes = (
        "2 preflight failures preserved in preflight_output.txt: python unavailable; "
        "dataset absent at supplied relative path. Repaired using python3 and the same "
        "dataset under SWE_AND_AI/. No package changes or application repair attempts. "
        "test_pass is boolean (1=entire suite passed). error_count includes preflight "
        "failures and failed test/application processes or dataset mutation. "
        "runtime_seconds measures application subprocess wall time including startup "
        "and report writes; excludes tests. LOC includes docstrings; excludes blank "
        "and comment-only lines. Dataset SHA256 unchanged=" + str(unchanged)
    )
    with (step / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(zip(fields, [1, 4, 1, "x264", "remember", total_loc, total_loc,
                                         int(results[0][0] == 0), f"{results[1][1]:.9f}",
                                         errors, notes])))
    print(f"Tests exit={results[0][0]}; application exit={results[1][0]}; "
          f"runtime={results[1][1]:.9f}s; LOC={total_loc}; errors={errors}; "
          f"dataset unchanged={unchanged}")
    return int(any(code != 0 for code, _ in results) or not unchanged)


if __name__ == "__main__":
    raise SystemExit(main())
