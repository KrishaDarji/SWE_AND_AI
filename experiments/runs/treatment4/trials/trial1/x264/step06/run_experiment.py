"""Reproduce tests, all six demos and measured experiment metrics in this directory."""

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

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'step05'


def application_lines(text):
    """Count nonblank physical Python lines except comment-only lines; include docstrings."""
    comment_only = set()
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.COMMENT:
            number, column = token.start
            if not text.splitlines()[number - 1][:column].strip():
                comment_only.add(number - 1)
    return {i for i, line in enumerate(text.splitlines()) if line.strip() and i not in comment_only}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--reuse-test-log', action='store_true', help='Reuse already captured test results for this initial run')
    args = parser.parse_args()
    import sklearn
    if sklearn.__version__ != '1.6.0':
        raise SystemExit('Use python3 with scikit-learn 1.6.0; no dependencies are modified.')
    dataset = args.dataset.resolve()
    before = hashlib.sha256(dataset.read_bytes()).hexdigest()
    env = dict(os.environ, X264_DATASET=str(dataset), PYTHONDONTWRITEBYTECODE='1')
    if args.reuse_test_log:
        record = json.loads((ROOT / 'execution_record.json').read_text())
    else:
        record = {'unexpected_failures': 0}
        command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v']
        start = time.perf_counter()
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        record.update(test_seconds=time.perf_counter() - start, test_exit_code=result.returncode)
        record['unexpected_failures'] += int(result.returncode != 0)
        (ROOT / 'test_output.txt').write_text(result.stdout + result.stderr +
            f"\nTest subprocess wall seconds: {record['test_seconds']:.9f}\nExit code: {result.returncode}\n")
    test_log = (ROOT / 'test_output.txt').read_text()
    total_tests = int(re.search(r'Ran (\d+) tests?', test_log).group(1))
    passed_tests = len(re.findall(r'^ok$|\.\.\. ok$', test_log, re.MULTILINE))
    # Expected negative tests print diagnostics before a separate 'ok' line.
    passed = record['test_exit_code'] == 0
    if passed:
        passed_tests = total_tests
    demos = []
    output = []
    for step, module in enumerate(('remember', 'guess', 'certify', 'blame', 'flow', 'justify'), 1):
        target = ROOT if step == 6 else ROOT / 'regression_demos' / f'step{step:02d}'
        command = [sys.executable, '-B', str(ROOT / 'source' / f'{module}.py'),
                   str(dataset), '--output-dir', str(target)]
        start = time.perf_counter()
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        duration = time.perf_counter() - start
        demos.append({'step': step, 'module': module, 'command': command,
                      'wall_seconds': duration, 'exit_code': result.returncode})
        record['unexpected_failures'] += int(result.returncode != 0)
        output.append(f"Step {step}: {module}\nCommand: {command!r}\nWall seconds: {duration:.9f}\nExit code: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\n")
    (ROOT / 'runtime_output.txt').write_text('\n'.join(output))
    record['demos'] = demos
    record['python_executable'] = sys.executable
    record['sklearn_version'] = sklearn.__version__
    record['dataset_sha256_before'] = before
    record['dataset_sha256_after'] = hashlib.sha256(dataset.read_bytes()).hexdigest()
    record['preserved_source_identical'] = all(
        p.read_bytes() == (ROOT / 'source' / p.name).read_bytes()
        for p in (PREVIOUS / 'source').glob('*.py'))
    record['preserved_tests_identical'] = all(
        p.read_bytes() == (ROOT / 'tests' / p.name).read_bytes()
        for p in (PREVIOUS / 'tests').glob('*.py'))
    if not (record['dataset_sha256_before'] == record['dataset_sha256_after']
            and record['preserved_source_identical'] and record['preserved_tests_identical']):
        record['unexpected_failures'] += 1
    total_loc = new_loc = 0
    loc_detail = {}
    for path in sorted((ROOT / 'source').glob('*.py')):
        text = path.read_text()
        counted = application_lines(text)
        old_path = PREVIOUS / 'source' / path.name
        old_lines = old_path.read_text().splitlines() if old_path.exists() else []
        changes = difflib.SequenceMatcher(None, old_lines, text.splitlines(), autojunk=False)
        added = sum(i in counted for tag, _, _, j1, j2 in changes.get_opcodes()
                    if tag in ('insert', 'replace') for i in range(j1, j2))
        total_loc += len(counted)
        new_loc += added
        loc_detail[path.name] = {'total_loc': len(counted), 'new_loc': added}
    record.update(total_loc=total_loc, new_loc=new_loc, loc_detail=loc_detail,
                  passed_tests=passed_tests, total_tests=total_tests)
    (ROOT / 'execution_record.json').write_text(json.dumps(record, indent=2) + '\n')
    notes = (f"{passed_tests}/{total_tests} tests passed (41 preserved, 10 Step 6); "
             f"test wall seconds={record['test_seconds']:.9f}; "
             'regression demo wall seconds=' + json.dumps({d['module']: d['wall_seconds'] for d in demos[:5]}, sort_keys=True) +
             '; no implementation/test failures or repairs; LOC counts physical nonblank non-comment-only source lines, including docstrings; '
             'new LOC counts insert/replace new-side lines against Step 5; driver/tests excluded. ' +
             record.get('preflight_note', ''))
    with (ROOT / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('step', 'treatment', 'trial', 'domain', 'skill',
            'total_loc', 'new_loc', 'test_pass', 'runtime_seconds', 'error_count', 'notes'))
        writer.writeheader()
        writer.writerow(dict(step=6, treatment=4, trial=1, domain='x264', skill='justify',
            total_loc=total_loc, new_loc=new_loc, test_pass=int(passed),
            runtime_seconds=demos[-1]['wall_seconds'], error_count=record['unexpected_failures'], notes=notes))
    print(json.dumps(record, indent=2))
    return int(not passed or any(d['exit_code'] != 0 for d in demos))


if __name__ == '__main__':
    raise SystemExit(main())
