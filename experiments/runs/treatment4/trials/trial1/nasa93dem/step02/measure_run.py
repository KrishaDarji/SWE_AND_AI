"""Re-run tests/demos and record actual timings, failures, and application LOC."""
import csv
import difflib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'step01' / 'source'
DATASET = Path(sys.argv[1]).resolve()
os.environ['NASA93_DATASET'] = str(DATASET)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['TMPDIR'] = str(ROOT / 'tests')


def run(label, args, output):
    start = time.perf_counter()
    result = subprocess.run([sys.executable] + args, cwd=ROOT, capture_output=True, text=True)
    seconds = time.perf_counter() - start
    with output.open('a', encoding='utf-8') as stream:
        stream.write(f'\n=== {label} ===\ncommand: {json.dumps([sys.executable] + args)}\n')
        stream.write(result.stdout + result.stderr)
        stream.write(f'\nexit_code: {result.returncode}; wall_seconds: {seconds:.9f}\n')
    return result, seconds


def counted(line):
    return bool(line.strip()) and not line.lstrip().startswith('#')


tests, test_seconds = run('Full Step 1 and Step 2 suite', ['-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'), '-v'], ROOT / 'test_output.txt')
regression, regression_seconds = run('Preserved Step 1 regression demo', [str(ROOT / 'source/summarize.py'), str(DATASET), '--output-dir', str(ROOT / 'step01_regression')], ROOT / 'runtime_output.txt')
demo, demo_seconds = run('Step 2 demo: illustrative proposal, not a new measured project', [str(ROOT / 'source/predict_effort.py'), str(DATASET), '--proposal', str(ROOT / 'example_proposal.json'), '--evaluation', str(ROOT / 'evaluation.json')], ROOT / 'runtime_output.txt')
error_count = sum(result.returncode != 0 for result in (tests, regression, demo))
files = sorted((ROOT / 'source').glob('*.py'))
total_loc = sum(sum(counted(line) for line in p.read_text().splitlines()) for p in files)
new_loc = 0
for path in files:
    previous = PREVIOUS / path.name
    old = previous.read_text().splitlines() if previous.exists() else []
    new = path.read_text().splitlines()
    for operation, i, j, a, b in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if operation in ('insert', 'replace'):
            new_loc += sum(counted(line) for line in new[a:b])
match = re.search(r'Ran (\d+) tests?', tests.stderr)
total_tests = int(match.group(1)) if match else 0
counts = {key: int(value) for key, value in re.findall(r'(failures|errors|skipped)=(\d+)', tests.stderr)}
passed = total_tests - sum(counts.values())
execution = {'test_seconds': test_seconds, 'step1_regression_demo_seconds': regression_seconds,
             'step2_demo_subprocess_seconds': demo_seconds, 'test_passed': passed,
             'test_total': total_tests, 'unexpected_failed_commands_this_attempt': error_count,
             'repair_attempts': [], 'loc_method': 'Nonblank, non-comment-only Python lines in source; docstrings included. New lines count insert/replace new sides relative to Step 1, not net growth.'}
with (ROOT / 'execution_measurements.jsonl').open('a') as stream:
    stream.write(json.dumps(execution) + '\n')
# Preserve failed-attempt counts across invocations of this runner.
history = [json.loads(line) for line in (ROOT / 'execution_measurements.jsonl').read_text().splitlines()]
all_errors = sum(item['unexpected_failed_commands_this_attempt'] for item in history)
if demo.returncode == 0:
    evaluation = json.loads((ROOT / 'evaluation.json').read_text())
    evaluation['execution_measurements'] = execution
    (ROOT / 'evaluation.json').write_text(json.dumps(evaluation, indent=2, allow_nan=False) + '\n')
with (ROOT / 'metrics.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames='step treatment trial domain skill total_loc new_loc test_pass runtime_seconds error_count notes'.split())
    writer.writeheader()
    writer.writerow(dict(step=2, treatment=4, trial=1, domain='nasa93dem', skill='guess',
                         total_loc=total_loc, new_loc=new_loc, test_pass=int(tests.returncode == 0),
                         runtime_seconds=f'{demo_seconds:.9f}', error_count=all_errors,
                         notes=f'{passed}/{total_tests} tests passed; test_seconds={test_seconds:.9f}; step1_regression_demo_seconds={regression_seconds:.9f}; attempts={len(history)}; failures retained in logs; example proposal is illustrative; units unspecified'))
print(json.dumps(execution, indent=2))
if demo.returncode == 0:
    print(demo.stdout)
if error_count:
    sys.exit(1)
