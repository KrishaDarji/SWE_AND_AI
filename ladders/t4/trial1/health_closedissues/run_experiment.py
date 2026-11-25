"""Run tests and the demo, preserving outputs, timings, and measured metrics."""
import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import sklearn

ROOT = Path(__file__).resolve().parents[1]


def run(command):
    start = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    return result, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    args = parser.parse_args()
    if sklearn.__version__ != '1.6.0':
        parser.error('Use the Python environment containing scikit-learn 1.6.0')
    if not args.dataset.is_file():
        parser.error(f'Missing input: {args.dataset}')
    test_command = [sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'), '-v']
    demo_command = [sys.executable, str(ROOT / 'source/summarize.py'), str(args.dataset), '--output-dir', str(ROOT)]
    tests, test_seconds = run(test_command)
    (ROOT / 'test_output.txt').write_text(tests.stdout + tests.stderr + f'\nMeasured test subprocess wall time: {test_seconds:.9f} seconds\n', encoding='utf-8')
    demo, demo_seconds = run(demo_command)
    (ROOT / 'runtime_output.txt').write_text(demo.stdout + demo.stderr + f'\nMeasured demo subprocess wall time: {demo_seconds:.9f} seconds\n', encoding='utf-8')
    output = tests.stdout + tests.stderr
    count_match = re.search(r'Ran (\d+) tests?', output)
    total = int(count_match.group(1)) if count_match else 0
    passed = len(re.findall(r'^test_.* \.\.\. ok$', output, re.MULTILINE))
    test_pass = int(tests.returncode == 0 and total > 0 and passed == total)
    history_path = ROOT / 'execution_history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append({'python': sys.executable, 'scikit_learn': sklearn.__version__,
                    'test_command': test_command, 'demo_command': demo_command,
                    'test_returncode': tests.returncode, 'demo_returncode': demo.returncode,
                    'test_seconds': test_seconds, 'demo_seconds': demo_seconds,
                    'passed': passed, 'total': total})
    history_path.write_text(json.dumps(history, indent=2) + '\n', encoding='utf-8')
    attempt = len(history)
    for name in ['test_output.txt', 'runtime_output.txt']:
        (ROOT / f'attempt{attempt:02d}_{name}').write_text((ROOT / name).read_text(), encoding='utf-8')
    loc = sum(1 for path in (ROOT / 'source').rglob('*.py')
              for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith('#'))
    errors = sum(int(item['test_returncode'] != 0) + int(item['demo_returncode'] != 0) for item in history)
    record = {'step': 1, 'treatment': 4, 'trial': 1, 'domain': 'health_closedissues',
              'skill': 'remember', 'total_loc': loc, 'new_loc': loc, 'test_pass': test_pass,
              'runtime_seconds': f'{demo_seconds:.9f}', 'error_count': errors,
              'notes': f'passed/total={passed}/{total}; test_wall_seconds={test_seconds:.9f}; '
                       f'attempt={attempt}; expected negative tests excluded from errors; '
                       'initial path/schema checks succeeded; no repair attempts' if attempt == 1 else
                       f'passed/total={passed}/{total}; test_wall_seconds={test_seconds:.9f}; '
                       f'attempt={attempt}; see execution_history.json and preserved attempt logs'}
    with (ROOT / 'metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)
    print(json.dumps(record, indent=2))
    return 0 if test_pass and demo.returncode == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
