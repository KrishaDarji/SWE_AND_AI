"""Run the complete suite and eight demos; measure application LOC and timings."""
import argparse
import csv
import difflib
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

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'step07'
DEFAULT_DATASET = Path('/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/hpo/Health-ClosedIssues0000.csv')


def application_lines(text):
    comment_rows = set()
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.COMMENT:
            row = token.start[0]
            if not text.splitlines()[row - 1][:token.start[1]].strip():
                comment_rows.add(row)
    return {i for i, line in enumerate(text.splitlines(), 1) if line.strip() and i not in comment_rows}


def loc_counts():
    total = added = 0
    for path in sorted((ROOT / 'source').glob('*.py')):
        text = path.read_text()
        counted = application_lines(text)
        total += len(counted)
        old = PREVIOUS / 'source' / path.name
        prior = old.read_text().splitlines() if old.exists() else []
        current = text.splitlines()
        for kind, _, _, start, stop in difflib.SequenceMatcher(None, prior, current, autojunk=False).get_opcodes():
            if kind in ('insert', 'replace'):
                added += sum(i + 1 in counted for i in range(start, stop))
    return total, added


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', nargs='?', type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    os.chdir(ROOT)
    tmp = ROOT / 'tmp'
    tmp.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(tmp), TMP=str(tmp), TEMP=str(tmp))
    (ROOT / 'regression').mkdir(exist_ok=True)
    history_path = ROOT / 'measurement_attempts.jsonl'
    previous_failures = 0
    if history_path.exists():
        for line in history_path.read_text().splitlines():
            previous_failures += sum(a['returncode'] != 0 for a in json.loads(line)['attempts'])
    errors = 0
    attempts = []
    def run(label, command, destination):
        nonlocal errors
        start = time.perf_counter()
        with destination.open('w') as handle:
            process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        elapsed = time.perf_counter() - start
        attempts.append({'label': label, 'command': command, 'returncode': process.returncode,
                         'wall_seconds': elapsed, 'output': str(destination.relative_to(ROOT))})
        errors += process.returncode != 0
        print(f'{label}: exit={process.returncode}, wall={elapsed:.6f}s', flush=True)
        return process.returncode, elapsed
    # Copies must stay byte-identical, rather than silently testing modified regressions.
    for kind in ('source', 'tests'):
        for path in (PREVIOUS / kind).glob('*.py'):
            if path.read_bytes() != (ROOT / kind / path.name).read_bytes():
                raise ValueError(f'Preserved {kind}/{path.name} differs')
    with args.dataset.open(newline='') as handle:
        first = next(csv.DictReader(handle))
    fields = ['N_estimators', 'criterion', 'Min_sample_leaves', 'Min_impurity_decrease', 'Max_depth']
    configuration = {n: first[n] if n == 'criterion' else float(first[n]) for n in fields}
    (ROOT / 'example_configuration.json').write_text(json.dumps(configuration, indent=2) + '\n')
    rc, test_seconds = run('Steps 1–8 complete test suite',
        [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v'], ROOT / 'test_output.txt')
    output = (ROOT / 'test_output.txt').read_text()
    match = re.search(r'Ran (\d+) tests?', output)
    total_tests = int(match.group(1)) if match else 0
    passed_tests = len(re.findall(r' \.\.\. ok$', output, re.MULTILINE))
    base = [sys.executable, '-B']
    data = str(args.dataset.resolve())
    regression = ROOT / 'regression'
    with args.dataset.open(newline='') as handle:
        first = next(csv.DictReader(handle))
    fields = ['N_estimators', 'criterion', 'Min_sample_leaves', 'Min_impurity_decrease', 'Max_depth']
    configuration = {n: first[n] if n == 'criterion' else float(first[n]) for n in fields}
    config_path = regression / 'configuration.json'
    config_path.write_text(json.dumps(configuration, indent=2) + '\n')
    commands = [
        ('summarize', ['--output-dir', str(regression / 'step01')]),
        ('guess', ['--configuration', str(config_path), '--evaluation', str(regression / 'step02.json')]),
        ('certify', ['--output', str(regression / 'step03.json')]),
        ('flow', ['--output', str(regression / 'step04.json')]),
        ('blame', ['--output', str(regression / 'step05.json')]),
        ('justify', ['--output-dir', str(regression / 'step06')]),
        ('choose', ['--max-estimators', '100', '--output', str(regression / 'step07.json')])]
    regression_times = {}
    for module, extra in commands:
        _, seconds = run(f'Preserved demo {module}', base + [str(ROOT / 'source' / f'{module}.py'), data] + extra,
                         regression / f'{module}_output.txt')
        regression_times[module] = seconds
    demo_rc, demo_seconds = run('Step 8 baseline data row 1',
        base + [str(ROOT / 'source/fix.py'), data, '--baseline-row', '1', '--output', str(ROOT / 'repair.json')],
        ROOT / 'runtime_output.txt')
    total_loc, new_loc = loc_counts()
    notes = {'passed_tests': passed_tests, 'total_tests': total_tests, 'test_wall_seconds': test_seconds,
             'regression_demo_wall_seconds': regression_times, 'step08_demo_returncode': demo_rc,
             'loc_convention': 'Nonblank non-comment-only source Python lines, including docstrings; new-side insert/replace counted against Step 7, not net growth; excludes tests and this runner.',
             'preflight': 'Allowed inputs verified. Absent Step 8 output directory and absent AGENTS search results were expected checks, not execution failures. Python 3.12 with scikit-learn 1.6.0.',
             'initial_step08_tests': '11/11 passed; output retained in step08_initial_tests.txt.',
             'repair_attempt': 'Initial full suite: 79/80 passed; preserved Step 2 test required missing example_configuration.json. Created fixture from supplied CSV row 1, without reading previous fixtures or altering preserved source/tests. Initial outputs retained under initial_*; full timing/exit history in measurement_attempts.jsonl.',
             'previous_unexpected_failed_subprocesses': previous_failures,
             'error_scope': 'Cumulative unexpected failed test/demo subprocess attempts in this Step 8 run; wrapper exit propagating the same failure is not double-counted. Expected negative-test failures and preflight absence probes excluded.'}
    with (ROOT / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['step', 'treatment', 'trial', 'domain', 'skill', 'total_loc', 'new_loc', 'test_pass', 'runtime_seconds', 'error_count', 'notes'])
        writer.writeheader()
        writer.writerow(dict(step=8, treatment=4, trial=1, domain='health_closedissues', skill='fix',
            total_loc=total_loc, new_loc=new_loc, test_pass=int(rc == 0 and passed_tests == total_tests and total_tests > 0),
            runtime_seconds=f'{demo_seconds:.9f}', error_count=previous_failures + errors, notes=json.dumps(notes, sort_keys=True)))
    log = ROOT / 'measurement_attempts.jsonl'
    with log.open('a') as handle:
        handle.write(json.dumps({'attempts': attempts, 'notes': notes, 'error_count': errors}) + '\n')
    instructions = [
        'Use python3 from the installed scikit-learn 1.6.0 environment; no installation or version changes.',
        'Application: source/fix.py; reusable APIs: repair(dataset, baseline_row), repair_rows(rows, baseline_row).',
        'Complete preserved source and tests are included. Run commands from Step 8:',
        'cd ' + shlex.quote(str(ROOT)),
        'mkdir -p tmp',
        'export TMPDIR="$PWD/tmp" PYTHONDONTWRITEBYTECODE=1',
        'python3 -B -m unittest discover -s tests -v',
        'python3 -B source/fix.py ' + shlex.quote(data) + ' --baseline-row 1 --output repair.json',
        'All demos, timings, LOC and metrics (logs each attempt; refreshes current outputs):',
        'python3 -B measure.py ' + shlex.quote(data),
        'Exact executed test and demo commands:']
    instructions.extend(shlex.join(item['command']) for item in attempts)
    (ROOT / 'run_instructions.txt').write_text('\n'.join(instructions) + '\n')
    print(json.dumps({'total_loc': total_loc, 'new_loc': new_loc, 'passed': passed_tests,
                      'total': total_tests, 'error_count': previous_failures + errors, 'demo_wall_seconds': demo_seconds}))
    return int(errors != 0)


if __name__ == '__main__':
    sys.exit(main())
