"""Run tests/demos and record measured metrics; intentionally outside source/."""
import csv
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'step08'
DEFAULT_DATA = Path('/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/hpo/Health-ClosedIssues0000.csv')


def application_lines(text):
    return [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith('#')]


def main():
    dataset = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_DATA
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(ROOT / 'tmp'))
    (ROOT / 'tmp').mkdir(exist_ok=True)
    regression = ROOT / 'regression_demos'
    regression.mkdir(exist_ok=True)
    fingerprint = hashlib.sha256(dataset.read_bytes()).hexdigest()
    unexpected = 3  # Historical failures documented in repair_log.txt; exclude expected negative tests.
    timings = {}
    commands = []

    def run(label, arguments, output):
        nonlocal unexpected
        command = [sys.executable, *map(str, arguments)]
        start = time.perf_counter()
        result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
        elapsed = time.perf_counter() - start
        timings[label] = elapsed
        output.write_text('Command: ' + repr(command) + '\n' + result.stdout + result.stderr +
                          f'\nExit status: {result.returncode}\nWall seconds: {elapsed:.9f}\n')
        commands.append({'label': label, 'command': command, 'returncode': result.returncode,
                         'wall_seconds': elapsed, 'output': str(output.relative_to(ROOT))})
        if result.returncode:
            unexpected += 1
        print(f'{label}: exit={result.returncode}; wall={elapsed:.3f}s', flush=True)
        return result

    with dataset.open(newline='') as handle:
        first_record = next(csv.DictReader(handle))
    fixture = {name: first_record[name] for name in
               ('N_estimators', 'criterion', 'Min_sample_leaves', 'Min_impurity_decrease', 'Max_depth')}
    (ROOT / 'example_configuration.json').write_text(json.dumps(fixture, indent=2) + '\n')
    tests = run('tests', ['-m', 'unittest', 'discover', '-s', 'tests', '-v'], ROOT / 'test_output.txt')
    demo = run('step09_demo', ['source/spot.py', dataset, '--output', ROOT / 'anomalies.json'], ROOT / 'runtime_output.txt')
    configuration = {n: None for n in ('N_estimators', 'criterion', 'Min_sample_leaves', 'Min_impurity_decrease', 'Max_depth')}
    with dataset.open(newline='') as handle:
        first = next(csv.DictReader(handle))
        for name in configuration:
            configuration[name] = first[name]
    (regression / 'configuration.json').write_text(json.dumps(configuration, indent=2) + '\n')
    demos = [
        ('step01_summarize', ['source/summarize.py', dataset, '--output-dir', regression / 'summarize']),
        ('step02_guess', ['source/guess.py', dataset, '--configuration', regression / 'configuration.json', '--evaluation', regression / 'guess.json']),
        ('step03_certify', ['source/certify.py', dataset, '--output', regression / 'certify.json']),
        ('step04_flow', ['source/flow.py', dataset, '--output', regression / 'flow.json']),
        ('step05_choose', ['source/choose.py', dataset, '--output', regression / 'choose.json']),
        ('step06_blame', ['source/blame.py', dataset, '--output', regression / 'blame.json']),
        ('step07_justify', ['source/justify.py', dataset, '--output-dir', regression / 'justify']),
        ('step08_fix', ['source/fix.py', dataset, '--baseline-row', '1', '--output', regression / 'fix.json'])]
    for label, arguments in demos:
        run(label, arguments, regression / (label + '_output.txt'))
    preservation = {}
    for folder in ('source', 'tests'):
        for p in sorted((PREVIOUS / folder).glob('*.py')):
            preservation[str(p.relative_to(PREVIOUS))] = p.read_bytes() == (ROOT / folder / p.name).read_bytes()
    unchanged = hashlib.sha256(dataset.read_bytes()).hexdigest() == fingerprint
    if not all(preservation.values()) or not unchanged:
        unexpected += 1
    (ROOT / 'verification.json').write_text(json.dumps({'preserved_files_byte_identical': preservation,
          'dataset_sha256_before': fingerprint, 'dataset_unchanged_after': unchanged,
          'commands': commands}, indent=2) + '\n')
    total_loc, new_loc = 0, 0
    for p in sorted((ROOT / 'source').glob('*.py')):
        current = p.read_text()
        total_loc += len(application_lines(current))
        prior = PREVIOUS / 'source' / p.name
        before = prior.read_text().splitlines() if prior.exists() else []
        after = current.splitlines()
        for tag, i, j, k, end in difflib.SequenceMatcher(a=before, b=after, autojunk=False).get_opcodes():
            if tag in ('insert', 'replace'):
                new_loc += len(application_lines('\n'.join(after[k:end])))
    match = re.search(r'Ran (\d+) tests?', tests.stderr)
    count = int(match.group(1)) if match else 0
    regression_seconds = sum(v for k, v in timings.items() if k not in ('tests', 'step09_demo'))
    notes = (f'passed/total={count if tests.returncode == 0 else "see test_output"}/{count}; '
             f'test_seconds={timings["tests"]:.9f}; regression_demo_seconds={regression_seconds:.9f}; '
             'runtime_seconds is Step 9 subprocess only; historical unexpected failures=3 (preliminary assertion, diagnostic read, missing fixture); '
             'see step09_preliminary_tests.txt and repair_log.txt; all previous source/tests copied byte-for-byte; '
             'no residual prediction; seed not applicable; display limit 10 does not affect threshold 3.5')
    fields = ['step', 'treatment', 'trial', 'domain', 'skill', 'total_loc', 'new_loc', 'test_pass', 'runtime_seconds', 'error_count', 'notes']
    row = dict(zip(fields, [9, 4, 1, 'health_closedissues', 'spot', total_loc, new_loc,
                           int(tests.returncode == 0), f'{timings["step09_demo"]:.9f}', unexpected, notes]))
    with (ROOT / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, indent=2), flush=True)
    return int(tests.returncode != 0 or demo.returncode != 0 or not all(preservation.values()) or not unchanged or any(c['returncode'] for c in commands))


if __name__ == '__main__':
    raise SystemExit(main())
