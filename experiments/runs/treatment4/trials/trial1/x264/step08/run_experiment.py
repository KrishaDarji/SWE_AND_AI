"""Reproduce tests, all eight demos, output captures, and measured metrics."""
import csv
import difflib
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import tokenize

STEP = Path(__file__).resolve().parent
PREVIOUS = STEP.parent / 'step07'
DATASET = Path(sys.argv[1]).resolve()
ENV = dict(os.environ, X264_DATASET=str(DATASET), PYTHONDONTWRITEBYTECODE='1')


def application_lines(text):
    """Nonblank physical Python lines after removing comment tokens only."""
    lines = text.splitlines()
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.COMMENT:
            row, col = token.start
            lines[row - 1] = lines[row - 1][:col]
    return {i for i, line in enumerate(lines) if line.strip()}


def run(args):
    started = time.perf_counter()
    process = subprocess.run(args, cwd=STEP, env=ENV, capture_output=True, text=True)
    return process, time.perf_counter() - started


def capture(args, process, elapsed):
    return (f'$ {shlex.join(map(str, args))}\nreturncode={process.returncode}; '
            f'wall_seconds={elapsed:.9f}\nSTDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}\n')


if __name__ == '__main__':
    before = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    import sklearn
    if sklearn.__version__ != '1.6.0':
        raise SystemExit('Requires installed scikit-learn 1.6.0; do not change packages')
    test_args = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v']
    tests, test_seconds = run(test_args)
    (STEP / 'test_output.txt').write_text(capture(test_args, tests, test_seconds))
    errors = int(tests.returncode != 0)
    regression = {}
    log = [f'Python: {sys.version}\nExecutable: {sys.executable}\nscikit-learn: {sklearn.__version__}\n']
    for module in ('remember', 'guess', 'certify', 'blame', 'choose', 'flow', 'justify'):
        args = [sys.executable, '-B', str(STEP / 'source' / f'{module}.py'),
                str(DATASET), '--output-dir', str(STEP / 'regression_demos' / module)]
        p, seconds = run(args)
        regression[module] = seconds
        errors += int(p.returncode != 0)
        log.append(capture(args, p, seconds))
    args = [sys.executable, '-B', str(STEP / 'source/fix.py'), str(DATASET),
            '--baseline-row', '1', '--output-dir', str(STEP)]
    p, demo_seconds = run(args)
    errors += int(p.returncode != 0)
    log.append(capture(args, p, demo_seconds))
    (STEP / 'runtime_output.txt').write_text('\n'.join(log))
    total_loc = new_loc = 0
    preserved = True
    for path in sorted((STEP / 'source').glob('*.py')):
        text = path.read_text()
        counted = application_lines(text)
        total_loc += len(counted)
        previous = PREVIOUS / 'source' / path.name
        old = previous.read_text() if previous.exists() else ''
        if previous.exists():
            preserved &= path.read_bytes() == previous.read_bytes()
        matcher = difflib.SequenceMatcher(None, old.splitlines(), text.splitlines(), autojunk=False)
        for tag, _, _, j1, j2 in matcher.get_opcodes():
            if tag in ('insert', 'replace'):
                new_loc += len(counted.intersection(range(j1, j2)))
    preserved_tests = all((STEP / 'tests' / old.name).read_bytes() == old.read_bytes()
                          for old in (PREVIOUS / 'tests').glob('*.py'))
    unchanged = before == hashlib.sha256(DATASET.read_bytes()).hexdigest()
    assert preserved and preserved_tests and unchanged
    # unittest's verbose per-test result lines give separately recorded counts.
    results = [line for line in tests.stderr.splitlines() if line.startswith('test_')]
    passed = sum(line.endswith(' ... ok') or line == 'ok'
                 for line in tests.stderr.splitlines())
    notes = {
        'passed_tests': passed, 'total_tests': len(results),
        'test_wall_seconds': test_seconds, 'regression_demo_wall_seconds': regression,
        'python_executable': sys.executable, 'scikit_learn': sklearn.__version__,
        'preserved_source_byte_identical': preserved,
        'preserved_tests_byte_identical': preserved_tests, 'dataset_sha256': before,
        'dataset_unchanged': unchanged,
        'loc_rule': 'Nonblank non-comment-only Python lines, docstrings included; new LOC counts insert/replace new-side lines against allowed Step 7 source.',
        'error_count_scope': 'Unexpected failed commands or execution attempts in this run; expected negative tests excluded.',
        'repair_attempts': [],
    }
    with (STEP / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['step','treatment','trial','domain','skill',
                                                    'total_loc','new_loc','test_pass','runtime_seconds','error_count','notes'])
        writer.writeheader()
        writer.writerow(dict(step=8,treatment=4,trial=1,domain='x264',skill='fix',total_loc=total_loc,
                             new_loc=new_loc,test_pass=int(tests.returncode == 0),runtime_seconds=demo_seconds,
                             error_count=errors,notes=json.dumps(notes,sort_keys=True)))
    print(json.dumps(dict(total_loc=total_loc,new_loc=new_loc,passed=passed,total=len(results),
                          demo_seconds=demo_seconds,error_count=errors),indent=2))
    raise SystemExit(int(errors != 0))
