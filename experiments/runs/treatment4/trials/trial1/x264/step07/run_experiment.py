"""Reproduce recorded Step 7, regression demos, tests, and measured metrics."""

import argparse
import csv
import difflib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import tokenize

STEP = Path(__file__).resolve().parent


def application_lines(text):
    comments = {token.start[0] for token in tokenize.generate_tokens(io.StringIO(text).readline)
                if token.type == tokenize.COMMENT
                and not text.splitlines()[token.start[0] - 1][:token.start[1]].strip()}
    return {index for index, line in enumerate(text.splitlines(), 1)
            if line.strip() and index not in comments}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', X264_DATASET=str(dataset))
    error_count = 0
    records = []
    before = hashlib.sha256(dataset.read_bytes()).hexdigest()

    def run(label, command):
        nonlocal error_count
        start = time.perf_counter()
        process = subprocess.run(command, cwd=STEP, env=env, capture_output=True, text=True)
        elapsed = time.perf_counter() - start
        if process.returncode:
            error_count += 1
        record = {'label': label, 'command': command, 'returncode': process.returncode,
                  'wall_seconds': elapsed, 'stdout': process.stdout, 'stderr': process.stderr}
        records.append(record)
        print(f'{label}: exit={process.returncode}; wall_seconds={elapsed:.6f}', flush=True)
        return record

    # Selection is executed and frozen before any regression demo or test reads all targets.
    demo = run('Step 7 demo', [sys.executable, '-B', str(STEP / 'source/choose.py'), str(dataset),
                             '--output-dir', str(STEP), '--budget', '20', '--seed', '42'])
    runtime = STEP / 'runtime_output.txt'
    runtime.write_text('Step 7 demo subprocess wall time only: ' + str(demo['wall_seconds']) + '\n'
                       + 'Command: ' + repr(demo['command']) + '\n'
                       + 'Return code: ' + str(demo['returncode']) + '\n'
                       + demo['stdout'] + '\nSTDERR:\n' + demo['stderr'])
    suite = run('Complete tests (Steps 1–7)', [sys.executable, '-B', '-m', 'unittest',
                                            'discover', '-s', 'tests', '-v'])
    (STEP / 'test_output.txt').write_text('Wall seconds: ' + str(suite['wall_seconds']) + '\n'
                                         + suite['stdout'] + suite['stderr'])
    for module in ('remember', 'guess', 'blame', 'certify', 'flow', 'justify'):
        run('Regression demo ' + module,
            [sys.executable, '-B', str(STEP / 'source' / (module + '.py')), str(dataset),
             '--output-dir', str(STEP / 'regression_demos' / module)])
    after = hashlib.sha256(dataset.read_bytes()).hexdigest()
    preserved = all(path.read_bytes() == (STEP / 'source' / path.name).read_bytes()
                    for path in (STEP.parent / 'step06/source').glob('*.py'))
    preserved_tests = all(path.read_bytes() == (STEP / 'tests' / path.name).read_bytes()
                          for path in (STEP.parent / 'step06/tests').glob('*.py'))
    if before != after or not preserved or not preserved_tests:
        error_count += 1
    (STEP / 'execution_log.json').write_text(json.dumps({
        'records': records, 'dataset_sha256_before': before, 'dataset_sha256_after': after,
        'preserved_source_identical': preserved, 'preserved_tests_identical': preserved_tests,
        'unexpected_error_count': error_count}, indent=2) + '\n')
    total_loc, new_loc = 0, 0
    for path in sorted((STEP / 'source').glob('*.py')):
        text = path.read_text()
        counted = application_lines(text)
        total_loc += len(counted)
        previous = STEP.parent / 'step06/source' / path.name
        old_lines = previous.read_text().splitlines() if previous.exists() else []
        new_lines = text.splitlines()
        for tag, _, _, start, end in difflib.SequenceMatcher(
                None, old_lines, new_lines, autojunk=False).get_opcodes():
            if tag in ('insert', 'replace'):
                new_loc += sum(index in counted for index in range(start + 1, end + 1))
    output = suite['stdout'] + suite['stderr']
    match = re.search(r'Ran (\d+) tests?', output)
    total_tests = int(match.group(1)) if match else 0
    failures = re.search(r'FAILED \((.*?)\)', output)
    failed_tests = sum(map(int, re.findall(r'(?:failures|errors)=(\d+)', failures.group(1)))) if failures else 0
    passed_tests = total_tests - failed_tests
    notes = {
        'tests_passed': passed_tests, 'tests_total': total_tests,
        'test_wall_seconds': suite['wall_seconds'],
        'regression_demo_wall_seconds': {r['label']: r['wall_seconds'] for r in records[2:]},
        'regression_demo_exit_codes': {r['label']: r['returncode'] for r in records[2:]},
        'loc_rule': 'Nonblank, non-comment-only source Python lines, including docstrings; new-side insert/replace vs Step 6.',
        'dataset_unchanged': before == after, 'preserved_source_identical': preserved,
        'preserved_tests_identical': preserved_tests,
        'quality': 'No separate quality measurement; constraint cannot be verified.',
        'repairs': 'None' if not error_count else 'See execution_log.json and preserved command outputs.',
    }
    with (STEP / 'metrics.csv').open('w', newline='') as stream:
        fields = ['step', 'treatment', 'trial', 'domain', 'skill', 'total_loc', 'new_loc',
                  'test_pass', 'runtime_seconds', 'error_count', 'notes']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(zip(fields, [7, 4, 1, 'x264', 'choose', total_loc, new_loc,
                                          int(suite['returncode'] == 0), demo['wall_seconds'],
                                          error_count, json.dumps(notes, sort_keys=True)])))
    print(json.dumps({'total_loc': total_loc, 'new_loc': new_loc, 'passed': passed_tests,
                      'total': total_tests, 'error_count': error_count, 'notes': notes}, indent=2))
    return int(error_count != 0)


if __name__ == '__main__':
    raise SystemExit(main())
