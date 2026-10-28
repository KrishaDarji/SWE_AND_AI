"""Reproduce recorded demo, preserved regressions, tests and measured metrics locally."""
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
PREVIOUS = ROOT.parent / 'step09'


def app_lines(text):
    comments = {tok.start[0] for tok in tokenize.generate_tokens(io.StringIO(text).readline)
                if tok.type == tokenize.COMMENT}
    return {i for i, line in enumerate(text.splitlines(), 1)
            if line.strip() and not (i in comments and line.lstrip().startswith('#'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    environment = dict(os.environ, X264_DATASET=str(dataset), PYTHONDONTWRITEBYTECODE='1')
    before = hashlib.sha256(dataset.read_bytes()).hexdigest()
    timings, errors = {}, 0

    def execute(label, argv, log):
        nonlocal errors
        start = time.perf_counter()
        result = subprocess.run([sys.executable, '-B', *argv], cwd=ROOT,
                                env=environment, capture_output=True, text=True)
        elapsed = time.perf_counter() - start
        timings[label] = elapsed
        errors += int(result.returncode != 0)
        log.write(f'COMMAND: {sys.executable} -B ' + repr(argv) + '\n')
        log.write(result.stdout)
        log.write(result.stderr)
        log.write(f'EXIT: {result.returncode}; subprocess wall seconds: {elapsed:.9f}\n\n')
        log.flush()
        return result

    with (ROOT / 'runtime_output.txt').open('w') as log:
        log.write('Recorded Step 10 comparison runs before any regression demos or full-data test checks.\n')
        demo = execute('step10_demo', ['source/race.py', str(dataset), '--output-dir', str(ROOT),
                       '--budget', '20', '--seeds', *map(str, range(42, 52))], log)
        for module in ('remember', 'blame', 'guess', 'certify', 'fix', 'justify', 'choose', 'flow', 'spot'):
            argv = [f'source/{module}.py', str(dataset), '--output-dir',
                    str(ROOT / 'regression_demos' / module)]
            if module == 'fix':
                argv += ['--baseline-row', '1']
            execute('regression_' + module, argv, log)
    with (ROOT / 'test_output.txt').open('w') as log:
        tests = execute('tests', ['-m', 'unittest', 'discover', '-s', 'tests', '-v'], log)
    after = hashlib.sha256(dataset.read_bytes()).hexdigest()
    if after != before:
        errors += 1
    total, added = 0, 0
    preserved = True
    for path in sorted((ROOT / 'source').glob('*.py')):
        text = path.read_text()
        counted = app_lines(text)
        total += len(counted)
        old = PREVIOUS / 'source' / path.name
        if not old.exists():
            added += len(counted)
        else:
            old_text = old.read_text()
            preserved = preserved and text == old_text
            matcher = difflib.SequenceMatcher(a=old_text.splitlines(), b=text.splitlines(), autojunk=False)
            for tag, _, _, lo, hi in matcher.get_opcodes():
                if tag in ('insert', 'replace'):
                    added += sum(i in counted for i in range(lo + 1, hi + 1))
    tests_preserved = all((ROOT / 'tests' / p.name).read_bytes() == p.read_bytes()
                          for p in (PREVIOUS / 'tests').glob('*.py'))
    match = re.search(r'Ran (\d+) tests?', tests.stderr)
    count = int(match.group(1)) if match else None
    passed = count if tests.returncode == 0 else None
    notes = {'passed_tests': passed, 'total_tests': count, 'timings_seconds': timings,
             'dataset_sha256_before': before, 'dataset_sha256_after': after,
             'dataset_unchanged': before == after, 'previous_source_copies_unchanged': preserved,
             'previous_test_copies_unchanged': tests_preserved,
             'new_loc_rule': 'Added nonblank non-comment-only new-side insert/replace lines versus Step 9; docstrings included.',
             'runtime_scope': 'runtime_seconds is Step 10 demo subprocess wall time only.',
             'failures_and_repairs': 'No unexpected execution failures or repair attempts.' if errors == 0 else
                 'See captured output files for unexpected failures; all attempts preserved.',
             'statistical_analysis': 'pending; required project comparison procedure absent'}
    with (ROOT / 'metrics.csv').open('w', newline='') as stream:
        row = dict(step=10, treatment=4, trial=1, domain='x264', skill='race', total_loc=total,
                   new_loc=added, test_pass=int(tests.returncode == 0),
                   runtime_seconds=timings['step10_demo'], error_count=errors,
                   notes=json.dumps(notes, sort_keys=True))
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    (ROOT / 'execution_summary.json').write_text(json.dumps(notes, indent=2) + '\n')
    print(json.dumps({'test_pass': tests.returncode == 0, 'total_tests': count,
                      'error_count': errors, 'total_loc': total, 'new_loc': added,
                      'demo_returncode': demo.returncode, **notes}, indent=2))
    return int(errors != 0 or tests.returncode != 0)


if __name__ == '__main__':
    raise SystemExit(main())
