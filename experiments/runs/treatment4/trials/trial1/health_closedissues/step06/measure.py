"""Run the full suite and six demos; measure actual execution and source LOC."""
import argparse
import csv
import difflib
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'step05'


def counted(line):
    return bool(line.strip()) and not line.lstrip().startswith('#')


def loc_metrics():
    total = added = 0
    unchanged = {}
    for path in sorted((ROOT / 'source').glob('*.py')):
        lines = path.read_text().splitlines()
        total += sum(map(counted, lines))
        old = PREVIOUS / 'source' / path.name
        before = old.read_text().splitlines() if old.exists() else []
        for tag, _, _, j1, j2 in difflib.SequenceMatcher(None, before, lines, autojunk=False).get_opcodes():
            if tag in ('insert', 'replace'):
                added += sum(map(counted, lines[j1:j2]))
        if old.exists():
            unchanged['source/' + path.name] = path.read_bytes() == old.read_bytes()
    for path in sorted((PREVIOUS / 'tests').glob('*.py')):
        unchanged['tests/' + path.name] = path.read_bytes() == (ROOT / 'tests' / path.name).read_bytes()
    return total, added, unchanged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', nargs='?', type=Path)
    parser.add_argument('--suite', action='store_true')
    args = parser.parse_args()
    if args.suite:
        suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        summary = {'total': result.testsRun, 'failures': len(result.failures),
                   'errors': len(result.errors), 'skipped': len(result.skipped),
                   'passed': result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}
        (ROOT / 'test_counts.json').write_text(json.dumps(summary, indent=2) + '\n')
        return 0 if result.wasSuccessful() else 1
    if args.dataset is None:
        parser.error('dataset path is required unless --suite is used')
    dataset = args.dataset.resolve()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    (ROOT / 'tmp').mkdir(exist_ok=True)
    env['TMPDIR'] = str(ROOT / 'tmp')
    attempts = sorted(ROOT.glob('execution_attempt_*.json'))
    attempt = len(attempts) + 1
    log = []

    def run(label, command, output):
        start = time.perf_counter()
        result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
        elapsed = time.perf_counter() - start
        content = ('Command: ' + shlex.join(command) + '\n' + f'Wall seconds: {elapsed:.9f}\n'
                   + f'Exit code: {result.returncode}\n\nSTDOUT:\n' + result.stdout + '\nSTDERR:\n' + result.stderr)
        output.write_text(content)
        (ROOT / f'attempt_{attempt:02}_{label}.txt').write_text(content)
        log.append({'label': label, 'command': command, 'seconds': elapsed, 'returncode': result.returncode})
        print(f'{label}: exit={result.returncode}, wall={elapsed:.6f}s', flush=True)
        return result.returncode, elapsed

    # The retained Step 2 test expects this fixture in the current output root.
    with dataset.open(newline='', encoding='utf-8-sig') as handle:
        fixture_row = next(csv.DictReader(handle))
    fixture_fields = ('N_estimators', 'criterion', 'Min_sample_leaves', 'Min_impurity_decrease', 'Max_depth')
    (ROOT / 'example_configuration.json').write_text(json.dumps({n: fixture_row[n] for n in fixture_fields}, indent=2) + '\n')

    test_rc, test_seconds = run('tests', [sys.executable, str(ROOT / 'measure.py'), '--suite'], ROOT / 'test_output.txt')
    demo_rc, demo_seconds = run('step06', [sys.executable, str(ROOT / 'source/justify.py'), str(dataset),
                               '--output-dir', str(ROOT)], ROOT / 'runtime_output.txt')
    regression = ROOT / 'regression'
    regression.mkdir(exist_ok=True)
    with dataset.open(newline='', encoding='utf-8-sig') as handle:
        first = next(csv.DictReader(handle))
    config_fields = ('N_estimators', 'criterion', 'Min_sample_leaves', 'Min_impurity_decrease', 'Max_depth')
    (regression / 'configuration.json').write_text(json.dumps({n: first[n] for n in config_fields}, indent=2) + '\n')
    demos = [('step01', 'summarize.py', ['--output-dir', str(regression / 'summary')]),
             ('step02', 'guess.py', ['--configuration', str(regression / 'configuration.json'),
                                   '--evaluation', str(regression / 'evaluation.json')]),
             ('step03', 'certify.py', ['--output', str(regression / 'comparison.json')]),
             ('step04', 'flow.py', ['--output', str(regression / 'flow.json')]),
             ('step05', 'blame.py', ['--output', str(regression / 'diagnosis.json')])]
    for label, module, flags in demos:
        run(label, [sys.executable, str(ROOT / 'source' / module), str(dataset), *flags],
            regression / (label + '_output.txt'))
    total, new, unchanged = loc_metrics()
    (ROOT / 'execution_attempt_{:02}.json'.format(attempt)).write_text(json.dumps(log, indent=2) + '\n')
    prior_errors = 0
    for path in attempts:
        previous_log = json.loads(path.read_text())
        prior_errors += sum(item['returncode'] != 0 for item in previous_log)
        prior_errors += int(any(item['returncode'] != 0 for item in previous_log))
    errors = 3 + prior_errors + sum(item['returncode'] != 0 for item in log)
    counts = json.loads((ROOT / 'test_counts.json').read_text())
    evidence = {'application_total_loc': total, 'added_application_loc': new,
                'preserved_files_byte_identical': unchanged,
                'test_counts': counts, 'executions': log,
                'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
                'error_count': errors, 'error_note': 'Includes initial absent-output-directory find failure and two premature output reads; expected negative tests and normal rg no-match excluded.'}
    (ROOT / 'measurement_details.json').write_text(json.dumps(evidence, indent=2) + '\n')
    notes = (f"passed/total={counts['passed']}/{counts['total']}; test_seconds={test_seconds:.9f}; "
             + '; '.join(f"{item['label']}_demo_seconds={item['seconds']:.9f}" for item in log[2:])
             + f'; prior_failed_measurement_attempts={len(attempts)}; see setup_log.txt and attempt logs; preserved_files_identical={all(unchanged.values())}; setup find failure and two premature reads documented; no dependency changes; observed-only justification')
    with (ROOT / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['step', 'treatment', 'trial', 'domain', 'skill',
            'total_loc', 'new_loc', 'test_pass', 'runtime_seconds', 'error_count', 'notes'])
        writer.writeheader()
        writer.writerow(dict(step=6, treatment=4, trial=1, domain='health_closedissues', skill='justify',
            total_loc=total, new_loc=new, test_pass=int(test_rc == 0 and counts['passed'] == counts['total']),
            runtime_seconds=f'{demo_seconds:.9f}', error_count=errors, notes=notes))
    return int(any(item['returncode'] for item in log) or not all(unchanged.values()))


if __name__ == '__main__':
    raise SystemExit(main())
