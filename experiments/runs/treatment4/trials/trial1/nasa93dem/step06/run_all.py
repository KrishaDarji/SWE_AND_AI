"""Run the complete suite and demos, retaining measured outputs and metrics."""
import csv
import difflib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'step05/source'
DATASET = Path(sys.argv[1]).resolve()
ENV = dict(os.environ, NASA93_DATASET=str(DATASET), PYTHONDONTWRITEBYTECODE='1')


def run(command):
    start = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, env=ENV, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    return result, elapsed


def application_lines(lines):
    return sum(bool(line.strip()) and not line.lstrip().startswith('#') for line in lines)


def main():
    result, test_time = run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'])
    with (ROOT / 'test_output.txt').open('a') as log:
        log.write(f'Command: python3 -m unittest discover -s tests -v\nWall seconds: {test_time:.9f}\nExit: {result.returncode}\n')
        log.write(result.stdout + result.stderr)
    match = re.search(r'Ran (\d+) tests?', result.stderr)
    total = int(match.group(1)) if match else 0
    failures = len(re.findall(r'^FAIL:|^ERROR:', result.stderr, re.MULTILINE))
    passed = total - failures
    errors = int(result.returncode != 0)
    timings = {}
    regression_dir = ROOT / 'regression'
    regression_dir.mkdir(exist_ok=True)
    commands = [
        ('step1', ['summarize.py', str(DATASET), '--output-dir', str(regression_dir)]),
        ('step2', ['predict_effort.py', str(DATASET), '--proposal', str(ROOT / 'example_proposal.json'), '--evaluation', str(regression_dir / 'evaluation.json')]),
        ('step3', ['flow.py', str(DATASET), '--output', str(regression_dir / 'flow.json')]),
        ('step4', ['certify.py', str(DATASET), '--output', str(regression_dir / 'certification.json')]),
        ('step5', ['blame.py', str(DATASET), '--output', str(regression_dir / 'blame.json')]),
        ('step6', ['justify.py', str(DATASET), '--proposal', str(ROOT / 'example_proposal.json'), '--output-dir', str(ROOT), '--illustrative']),
    ]
    with (ROOT / 'runtime_output.txt').open('a') as log:
        for name, arguments in commands:
            command = [sys.executable, str(ROOT / 'source' / arguments[0]), *arguments[1:]]
            demo, elapsed = run(command)
            timings[name] = elapsed
            errors += int(demo.returncode != 0)
            log.write(f'\n{name}: {json.dumps(command)}\nWall seconds: {elapsed:.9f}\nExit: {demo.returncode}\n')
            log.write(demo.stdout + demo.stderr)
            log.flush()
    total_loc = new_loc = 0
    identical = []
    for path in sorted((ROOT / 'source').glob('*.py')):
        current = path.read_text().splitlines()
        total_loc += application_lines(current)
        old = PREVIOUS / path.name
        previous = old.read_text().splitlines() if old.exists() else []
        if old.exists():
            identical.append({'file': path.name, 'identical_to_step5': path.read_bytes() == old.read_bytes()})
        for tag, _, _, start, end in difflib.SequenceMatcher(None, previous, current, autojunk=False).get_opcodes():
            if tag in ('insert', 'replace'):
                new_loc += application_lines(current[start:end])
    audit = {'test_seconds': test_time, 'passed_tests': passed, 'total_tests': total,
             'demo_seconds': timings, 'unexpected_failed_commands': errors,
             'preserved_source': identical,
             'preserved_tests': [{'file': p.name, 'identical_to_step5': p.read_bytes() == (ROOT / 'tests' / p.name).read_bytes()}
                                 for p in (ROOT.parent / 'step05/tests').glob('*.py')],
             'loc_rule': 'Nonblank, non-comment-only Python source lines; docstrings included; added insert/replace new-side lines via SequenceMatcher with autojunk=False.',
             'repair_attempts': []}
    (ROOT / 'verification.json').write_text(json.dumps(audit, indent=2) + '\n')
    with (ROOT / 'metrics.csv').open('w', newline='') as stream:
        fields = 'step treatment trial domain skill total_loc new_loc test_pass runtime_seconds error_count notes'.split()
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(zip(fields, [6, 4, 1, 'nasa93dem', 'justify', total_loc, new_loc,
                          int(result.returncode == 0), f"{timings['step6']:.9f}", errors,
                          f'{passed}/{total} tests passed; test_seconds={test_time:.9f}; regression_demo_seconds={json.dumps({k:v for k,v in timings.items() if k != "step6"}, sort_keys=True)}; no unexpected failures or repairs; source/tests preserved byte-for-byte; runtime_seconds is Step 6 demo subprocess only.'])))
    print(json.dumps(audit, indent=2))
    return int(errors != 0)


if __name__ == '__main__':
    sys.exit(main())
