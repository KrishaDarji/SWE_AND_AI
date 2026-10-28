"""Run preserved and new tests/demos and record measured metrics inside Step 5."""
import csv
import difflib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import tokenize

STEP = Path(__file__).resolve().parent
PREVIOUS = STEP.parent / 'step04'
DATASET = Path('/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/config/X264_AllMeasurements.csv')
ENV = dict(os.environ, X264_DATASET=str(DATASET), PYTHONDONTWRITEBYTECODE='1')


def app_lines(lines):
    return sum(bool(line.strip()) and not line.lstrip().startswith('#') for line in lines)


def run(command):
    start = time.perf_counter()
    p = subprocess.run(command, cwd=STEP, env=ENV, capture_output=True, text=True)
    seconds = time.perf_counter() - start
    output = ('$ ' + shlex.join(command) + '\n' + p.stdout + p.stderr
              + f'\nexit_code={p.returncode}; wall_seconds={seconds:.9f}\n')
    return p, seconds, output


def main():
    digest = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    errors = 0
    test, test_seconds, test_output = run([sys.executable, '-B', '-m', 'unittest', 'discover',
                                         '-s', 'tests', '-v'])
    (STEP / 'test_output.txt').write_text(test_output)
    errors += int(test.returncode != 0)
    match = re.search(r'Ran (\d+) tests?', test.stderr)
    total_tests = int(match.group(1)) if match else 0
    failures = len(re.findall(r'^FAIL:|^ERROR:', test.stderr, re.M))
    passed = max(0, total_tests - failures)
    timings = {}
    output = []
    # First observed configuration, independently loaded; no prediction targets included.
    with DATASET.open(newline='') as f:
        row = next(csv.DictReader(f))
    (STEP / 'demo_config.json').write_text(json.dumps(
        {key: float(value) for key, value in row.items() if key != 'Performance-'}, indent=2) + '\n')
    for name in ('remember', 'guess', 'flow', 'certify', 'blame'):
        command = [sys.executable, '-B', str(STEP / 'source' / (name + '.py')), str(DATASET),
                   '--output-dir', str(STEP if name == 'blame' else STEP / 'regression_demos' / name)]
        if name == 'guess':
            command += ['--config', str(STEP / 'demo_config.json')]
        p, seconds, log = run(command)
        timings[name] = seconds
        output.append(log)
        errors += int(p.returncode != 0)
    assert digest == hashlib.sha256(DATASET.read_bytes()).hexdigest(), 'Dataset changed'
    preserved = []
    for folder in ('source', 'tests'):
        for p in (PREVIOUS / folder).glob('*.py'):
            assert p.read_bytes() == (STEP / folder / p.name).read_bytes(), str(p)
            preserved.append(str(p.relative_to(PREVIOUS)))
    output.append('Dataset SHA256 unchanged: ' + digest + '\n')
    output.append('Step 4 files preserved byte-for-byte: ' + ', '.join(sorted(preserved)) + '\n')
    (STEP / 'runtime_output.txt').write_text('\n'.join(output))
    loc, new = 0, 0
    for p in (STEP / 'source').glob('*.py'):
        lines = p.read_text().splitlines()
        loc += app_lines(lines)
        old = PREVIOUS / 'source' / p.name
        old_lines = old.read_text().splitlines() if old.exists() else []
        for tag, _, _, j1, j2 in difflib.SequenceMatcher(None, old_lines, lines, autojunk=False).get_opcodes():
            if tag in ('insert', 'replace'):
                new += app_lines(lines[j1:j2])
    notes = {'passed_tests': passed, 'total_tests': total_tests, 'test_seconds': test_seconds,
             'regression_demo_seconds': {k: v for k, v in timings.items() if k != 'blame'},
             'python': sys.executable, 'dataset_sha256': digest,
             'loc_scope': 'source/*.py; nonblank, non-comment-only lines; includes docstrings',
             'new_loc_method': 'difflib insert/replace new-side application lines vs Step 4',
             'repair_attempts': [], 'unexpected_failures_before_validation': 0,
             'step5_randomness': 'None; deterministic descriptive statistics; Step 2 seed=42; Step 4 seed=42031'}
    with (STEP / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['step', 'treatment', 'trial', 'domain', 'skill',
                                            'total_loc', 'new_loc', 'test_pass',
                                            'runtime_seconds', 'error_count', 'notes'])
        writer.writeheader()
        writer.writerow(dict(step=5, treatment=4, trial=1, domain='x264', skill='blame',
                             total_loc=loc, new_loc=new, test_pass=int(test.returncode == 0),
                             runtime_seconds=timings['blame'], error_count=errors,
                             notes=json.dumps(notes, sort_keys=True)))
    print(json.dumps({'passed': passed, 'total': total_tests, 'test_seconds': test_seconds,
                      'demo_seconds': timings, 'total_loc': loc, 'new_loc': new,
                      'errors': errors}, indent=2))
    return int(errors != 0)


if __name__ == '__main__':
    raise SystemExit(main())
