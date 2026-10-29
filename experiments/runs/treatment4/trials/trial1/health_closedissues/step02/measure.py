"""Measurement tooling, excluded from application LOC. Preserve every attempt."""
import argparse
import csv
import difflib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent / 'step01/source'


def application_line(line):
    return bool(line.strip()) and not line.lstrip().startswith('#')


def measure(command):
    start = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    return result, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    args = parser.parse_args()
    commands = {
        'tests': [sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'), '-v'],
        'demo': [sys.executable, str(ROOT / 'source/guess.py'), str(args.dataset),
                 '--configuration', str(ROOT / 'example_configuration.json'),
                 '--evaluation', str(ROOT / 'evaluation.json')],
        'regression_demo': [sys.executable, str(ROOT / 'source/summarize.py'), str(args.dataset),
                            '--output-dir', str(ROOT / 'step01_regression_demo')]}
    history_path = ROOT / 'execution_history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    attempt = len(history) + 1
    records = {}
    for label, command in commands.items():
        result, elapsed = measure(command)
        output = result.stdout + result.stderr
        path = 'test_output.txt' if label == 'tests' else f'{label}_output.txt'
        if label == 'demo':
            path = 'runtime_output.txt'
        log = output + f'\nMeasured {label} subprocess wall time: {elapsed:.9f} seconds\n'
        (ROOT / path).write_text(log)
        (ROOT / f'attempt{attempt:02d}_{path}').write_text(log)
        records[label] = {'command': command, 'returncode': result.returncode, 'seconds': elapsed}
        if label == 'tests':
            match = re.search(r'Ran (\d+) tests?', output)
            total = int(match.group(1)) if match else 0
            passed = len(re.findall(r'^test_.* \.\.\. ok$', output, re.MULTILINE))
    history.append({'attempt': attempt, 'executions': records, 'passed': passed, 'total': total})
    history_path.write_text(json.dumps(history, indent=2) + '\n')
    total_loc = new_loc = 0
    for path in sorted((ROOT / 'source').glob('*.py')):
        new = path.read_text().splitlines()
        total_loc += sum(application_line(line) for line in new)
        old_path = BASELINE / path.name
        old = old_path.read_text().splitlines() if old_path.exists() else []
        for op, _, _, start, stop in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
            if op in ('insert', 'replace'):
                new_loc += sum(application_line(line) for line in new[start:stop])
    errors = sum(r['returncode'] != 0 for h in history for r in h['executions'].values())
    success = int(records['tests']['returncode'] == 0 and passed == total and total > 0)
    record = dict(step=2, treatment=4, trial=1, domain='health_closedissues', skill='guess',
                  total_loc=total_loc, new_loc=new_loc, test_pass=success,
                  runtime_seconds=f"{records['demo']['seconds']:.9f}", error_count=errors,
                  notes=f"passed/total={passed}/{total}; test_wall_seconds={records['tests']['seconds']:.9f}; "
                        f"regression_demo_wall_seconds={records['regression_demo']['seconds']:.9f}; "
                        f"attempt={attempt}; baseline=131 application LOC + 65 runner LOC; "
                        "Step 1 runner excluded; new_loc=nonblank noncomment insert/replace new-side lines via difflib; "
                        "measurement runner excluded; initial checks succeeded; expected negative tests excluded; "
                        + ('no repairs' if attempt == 1 and not errors else 'see preserved attempt logs and response.txt for repair history'))
    with (ROOT / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=record)
        writer.writeheader()
        writer.writerow(record)
    print(json.dumps(record, indent=2))
    return 0 if success and all(r['returncode'] == 0 for r in records.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
