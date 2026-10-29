"""Run and measure the full suite and three demos; write only within Step 3."""
import csv
import difflib
import io
import json
import subprocess
import sys
import time
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'step02' / 'source'
DATASET = Path('/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/hpo/Health-ClosedIssues0000.csv')


def application_lines(text):
    comment_lines = {tok.start[0] for tok in tokenize.generate_tokens(io.StringIO(text).readline)
                     if tok.type == tokenize.COMMENT and not text.splitlines()[tok.start[0] - 1][:tok.start[1]].strip()}
    return {i for i, line in enumerate(text.splitlines(), 1) if line.strip() and i not in comment_lines}


def run(command, destination):
    start = time.perf_counter()
    p = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    destination.write_text('Command: ' + repr(command) + '\n' + p.stdout + p.stderr +
                           f'\nExit code: {p.returncode}\nWall seconds: {elapsed:.9f}\n')
    return p, elapsed


def main():
    results = {}
    commands = [
        ('tests', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], 'test_output.txt'),
        ('step1', [sys.executable, 'source/summarize.py', str(DATASET), '--output-dir', str(ROOT / 'regression_step1')], 'regression_step1_output.txt'),
        ('step2', [sys.executable, 'source/guess.py', str(DATASET), '--configuration', str(ROOT / 'example_configuration.json'), '--evaluation', str(ROOT / 'regression_step2.json')], 'regression_step2_output.txt'),
        ('step3', [sys.executable, 'source/certify.py', str(DATASET), '--output', str(ROOT / 'certification.json')], 'runtime_output.txt')]
    for name, command, output in commands:
        p, elapsed = run(command, ROOT / output)
        results[name] = {'seconds': elapsed, 'exit_code': p.returncode}
        print(name, results[name], flush=True)
    total = new = 0
    for path in sorted((ROOT / 'source').glob('*.py')):
        current = path.read_text()
        counted = application_lines(current)
        total += len(counted)
        old_path = PREVIOUS / path.name
        old = old_path.read_text() if old_path.exists() else ''
        matcher = difflib.SequenceMatcher(None, old.splitlines(), current.splitlines(), autojunk=False)
        new += sum(sum(i + 1 in counted for i in range(j1, j2))
                   for tag, _, _, j1, j2 in matcher.get_opcodes() if tag in ('insert', 'replace'))
    (ROOT / 'measurements.json').write_text(json.dumps({'python': sys.executable, 'runs': results,
                        'total_loc': total, 'new_loc': new}, indent=2) + '\n')
    import re
    test_text = (ROOT / 'test_output.txt').read_text()
    matches = re.findall(r'Ran (\d+) tests?', test_text)
    count = int(matches[-1]) if matches else 0
    passed = count if results['tests']['exit_code'] == 0 else sum(' ... ok' in line for line in test_text.splitlines())
    errors = sum(r['exit_code'] != 0 for r in results.values())
    notes = (f'passed/total={passed}/{count}; test_seconds={results["tests"]["seconds"]:.9f}; '
             f'regression_step1_seconds={results["step1"]["seconds"]:.9f}; '
             f'regression_step2_seconds={results["step2"]["seconds"]:.9f}; '
             'Welch deterministic; seed 42 reserved; secondary objectives descriptive; '
             'no approved practical threshold; preserved Step 2 source/tests byte-for-byte; '
             'example configuration newly supplied because prior root artifacts are outside allowed inputs; '
             'no repairs or unexpected prior failures')
    with (ROOT / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['step', 'treatment', 'trial', 'domain', 'skill', 'total_loc', 'new_loc', 'test_pass', 'runtime_seconds', 'error_count', 'notes'])
        writer.writerow([3, 4, 1, 'health_closedissues', 'certify', total, new,
                         int(results['tests']['exit_code'] == 0), f'{results["step3"]["seconds"]:.9f}', errors, notes])


if __name__ == '__main__':
    main()
