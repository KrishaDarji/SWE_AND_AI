"""Run the local suite and four demos, preserving execution logs and measured metrics."""

import argparse
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

import sklearn

STEP = Path(__file__).resolve().parent
PREVIOUS = STEP.parent / 'step03'


def counted(lines):
    return sum(bool(line.strip()) and not line.lstrip().startswith('#') for line in lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    args = parser.parse_args()
    if sklearn.__version__ != '1.6.0':
        parser.error('Use python3 from the environment containing scikit-learn 1.6.0')
    dataset = args.dataset.resolve()
    env = dict(os.environ, X264_DATASET=str(dataset), PYTHONDONTWRITEBYTECODE='1')
    evidence_path = STEP / 'execution_records.json'
    records = json.loads(evidence_path.read_text()) if evidence_path.exists() else []
    before = hashlib.sha256(dataset.read_bytes()).hexdigest()

    def execute(label, arguments, log_name):
        command = [sys.executable, '-B', *arguments]
        start = time.perf_counter()
        process = subprocess.run(command, cwd=STEP, env=env, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - start
        record = {'label': label, 'command': command, 'returncode': process.returncode,
                  'wall_seconds': elapsed}
        records.append(record)
        with (STEP / log_name).open('a', encoding='utf-8') as log:
            log.write(json.dumps(record) + '\n' + process.stdout + '\n')
        evidence_path.write_text(json.dumps(records, indent=2) + '\n')
        print(f'{label}: exit={process.returncode}, wall_seconds={elapsed:.6f}', flush=True)
        return process, elapsed

    tests, test_seconds = execute('all tests', ['-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                                  'test_output.txt')
    match = re.search(r'Ran (\d+) tests?', tests.stdout)
    total = int(match.group(1)) if match else 0
    failed = sum(int(n) for n in re.findall(r'(?:failures|errors)=(\d+)', tests.stdout))
    passed = total - failed
    timings = {}
    for module in ('remember', 'guess', 'certify', 'flow'):
        destination = STEP if module == 'flow' else STEP / 'regression_demos' / module
        _, elapsed = execute(module + ' demo', [str(STEP / 'source' / (module + '.py')),
                             str(dataset), '--output-dir', str(destination)], 'runtime_output.txt')
        timings[module] = elapsed
    after = hashlib.sha256(dataset.read_bytes()).hexdigest()
    unchanged = all((STEP / folder / path.name).read_bytes() == path.read_bytes()
                    for folder in ('source', 'tests')
                    for path in (PREVIOUS / folder).glob('*.py'))
    total_loc = 0
    new_loc = 0
    for path in sorted((STEP / 'source').glob('*.py')):
        lines = path.read_text().splitlines()
        total_loc += counted(lines)
        prior = PREVIOUS / 'source' / path.name
        old = prior.read_text().splitlines() if prior.exists() else []
        for tag, _, _, start, end in difflib.SequenceMatcher(None, old, lines,
                                                           autojunk=False).get_opcodes():
            if tag in ('insert', 'replace'):
                new_loc += counted(lines[start:end])
    errors = sum(record['returncode'] != 0 for record in records)
    notes = (f'passed/total={passed}/{total}; test_wall_seconds={test_seconds:.9f}; '
             f'regression_demo_wall_seconds={json.dumps({k: v for k, v in timings.items() if k != "flow"}, sort_keys=True)}; '
             f'python={sys.version.split()[0]}; sklearn={sklearn.__version__}; '
             f'dataset_sha256_before={before}; dataset_sha256_after={after}; '
             f'previous_source_and_tests_byte_identical={unchanged}; '
             'LOC counts nonblank, non-comment-only source Python lines including docstrings; '
             'new LOC uses insert/replace new-side lines against allowed Step 3 source; '
             'expected negative-test exits excluded; logs append across repair attempts; '
             'no unexpected command failures before this runner')
    with (STEP / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['step', 'treatment', 'trial', 'domain', 'skill', 'total_loc',
                         'new_loc', 'test_pass', 'runtime_seconds', 'error_count', 'notes'])
        writer.writerow([4, 4, 1, 'x264', 'flow', total_loc, new_loc,
                         int(tests.returncode == 0 and total > 0),
                         f'{timings["flow"]:.9f}', errors, notes])
    print(f'LOC total={total_loc}, added={new_loc}; tests={passed}/{total}; unexpected failures={errors}')
    if before != after or not unchanged:
        raise RuntimeError('Input or preserved implementation changed')
    return int(errors != 0)


if __name__ == '__main__':
    raise SystemExit(main())
