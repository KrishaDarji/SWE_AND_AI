"""Run, capture and measure this step only; writes artifacts beside this script."""
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
PREVIOUS = ROOT.parent / 'step03' / 'source'
DATASET = Path(sys.argv[1]).resolve()
if not DATASET.is_file():
    raise SystemExit(f'Missing dataset: {DATASET}')
import sklearn
if sklearn.__version__ != '1.6.0':
    raise SystemExit(f'Expected scikit-learn 1.6.0; found {sklearn.__version__}')


def run(label, args, log):
    start = time.perf_counter()
    result = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True,
                            text=True, env=dict(os.environ, NASA93_DATASET=str(DATASET),
                                               PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(ROOT / 'tests')))
    elapsed = time.perf_counter() - start
    with (ROOT / log).open('a', encoding='utf-8') as stream:
        stream.write(f'\n{label}\nCommand: {json.dumps([sys.executable, *args])}\n')
        stream.write(result.stdout + result.stderr)
        stream.write(f'Exit code: {result.returncode}; wall_seconds: {elapsed:.9f}\n')
    return result, elapsed


# Append rather than overwrite logs so repeated attempts remain visible.
result, test_seconds = run('Entire Steps 1–4 suite', ['-m', 'unittest', 'discover', '-s', 'tests', '-v'], 'test_output.txt')
match = re.search(r'Ran (\d+) tests?', result.stderr)
total = int(match.group(1)) if match else 0
failures = re.search(r'FAILED \((.*?)\)', result.stderr)
failed = sum(int(n) for n in re.findall(r'(?:failures|errors)=(\d+)', failures.group(1))) if failures else 0
passed = total - failed
errors = int(result.returncode != 0)
step4, runtime = run('Step 4 sequential demo', ['source/flow.py', str(DATASET), '--output', str(ROOT / 'flow.json')], 'runtime_output.txt')
errors += int(step4.returncode != 0)
(ROOT / 'proposal.json').write_text('{"prec": "n"}\n', encoding='utf-8')
regression = {}
for label, args in [
    ('Step 1 descriptive demo', ['source/summarize.py', str(DATASET), '--output-dir', str(ROOT / 'regression')]),
    ('Step 2 grouped evaluation demo', ['source/predict_effort.py', str(DATASET), '--proposal', str(ROOT / 'proposal.json'), '--evaluation', str(ROOT / 'regression/evaluation.json')]),
    ('Step 3 observed comparison demo', ['source/certify.py', str(DATASET), '--output', str(ROOT / 'regression/certification.json')]),
]:
    demo, seconds = run(label, args, 'runtime_output.txt')
    regression[label] = {'wall_seconds': seconds, 'exit_code': demo.returncode}
    errors += int(demo.returncode != 0)


def countable(line):
    return bool(line.strip()) and not line.lstrip().startswith('#')


total_loc = new_loc = 0
unchanged = {}
for path in sorted((ROOT / 'source').glob('*.py')):
    lines = path.read_text().splitlines()
    total_loc += sum(countable(line) for line in lines)
    old_path = PREVIOUS / path.name
    old = old_path.read_text().splitlines() if old_path.exists() else []
    if old_path.exists():
        unchanged[path.name] = old == lines
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=old, b=lines, autojunk=False).get_opcodes():
        if tag in ('insert', 'replace'):
            new_loc += sum(countable(line) for line in lines[j1:j2])
notes = {
    'passed': passed, 'total': total, 'test_wall_seconds': test_seconds,
    'regression_demos': regression, 'preserved_source_unchanged': unchanged,
    'repairs': 'None' if errors == 0 else 'See captured logs; unexpected failures preserved.',
    'runtime_definition': 'Step 4 demo subprocess wall time only, including startup and JSON output',
    'loc_definition': 'Nonblank, non-comment-only source Python lines, including docstrings; insert/replace new-side lines versus Step 3',
    'error_count_scope': 'Unexpected failures in this run; expected negative tests excluded; no earlier command failures',
}
with (ROOT / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=['step', 'treatment', 'trial', 'domain', 'skill', 'total_loc', 'new_loc', 'test_pass', 'runtime_seconds', 'error_count', 'notes'])
    writer.writeheader()
    writer.writerow(dict(step=4, treatment=4, trial=1, domain='nasa93dem', skill='flow',
                         total_loc=total_loc, new_loc=new_loc, test_pass=int(result.returncode == 0),
                         runtime_seconds=f'{runtime:.9f}', error_count=errors, notes=json.dumps(notes)))
print(json.dumps({'test_pass': result.returncode == 0, 'passed': passed, 'total': total,
                  'runtime_seconds': runtime, 'total_loc': total_loc, 'new_loc': new_loc,
                  'error_count': errors, 'notes': notes}, indent=2))
raise SystemExit(int(errors > 0))
