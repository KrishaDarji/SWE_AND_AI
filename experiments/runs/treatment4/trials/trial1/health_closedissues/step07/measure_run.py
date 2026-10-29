"""Reproduce tests, all seven demos and measured metrics; outside application source."""
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
PREVIOUS = ROOT.parent / 'step06'
DATASET = Path('/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/hpo/Health-ClosedIssues0000.csv')
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')


def application_lines(lines):
    return sum(bool(line.strip()) and not line.lstrip().startswith('#') for line in lines)


def main():
    errors = 0
    timings = {}
    attempts = []
    archive = ROOT / 'attempt_history'
    archive.mkdir(exist_ok=True)
    for name in ['test_output.txt', 'runtime_output.txt', 'metrics.csv', 'measurements.json']:
        p = ROOT / name
        if p.exists():
            destination = archive / (str(time.time_ns()) + '_' + name)
            destination.write_bytes(p.read_bytes())
    old = ROOT / 'run_history.json'
    history = json.loads(old.read_text()) if old.exists() else []

    def run(label, args):
        nonlocal errors
        command = [sys.executable, *map(str, args)]
        start = time.perf_counter()
        result = subprocess.run(command, cwd=ROOT, env=ENV, text=True, capture_output=True)
        elapsed = time.perf_counter() - start
        timings[label] = elapsed
        errors += int(result.returncode != 0)
        attempts.append({'label': label, 'command': command, 'returncode': result.returncode,
                         'wall_seconds': elapsed})
        print(f'{label}: exit={result.returncode}, seconds={elapsed:.6f}', flush=True)
        log = 'Command: ' + repr(command) + f'\nExit: {result.returncode}\nWall seconds: {elapsed:.9f}\nSTDOUT:\n' + result.stdout + '\nSTDERR:\n' + result.stderr + '\n'
        return result, log

    tests, test_log = run('tests', ['-m', 'unittest', 'discover', '-s', 'tests', '-v'])
    (ROOT / 'test_output.txt').write_text(test_log)
    regression = ROOT / 'regression_demos'
    regression.mkdir(exist_ok=True)
    commands = [
        ('step1', ['source/summarize.py', DATASET, '--output-dir', regression / 'step1']),
        ('step2', ['source/guess.py', DATASET, '--configuration', ROOT / 'example_configuration.json', '--evaluation', regression / 'step2.json']),
        ('step3', ['source/certify.py', DATASET, '--output', regression / 'step3.json']),
        ('step4', ['source/flow.py', DATASET, '--output', regression / 'step4.json']),
        ('step5', ['source/blame.py', DATASET, '--output', regression / 'step5.json']),
        ('step6', ['source/justify.py', DATASET, '--output-dir', regression / 'step6']),
        ('step7', ['source/choose.py', DATASET, '--max-estimators', '100', '--output', ROOT / 'selection.json'])]
    with (ROOT / 'runtime_output.txt').open('w') as handle:
        for label, command in commands:
            result, log = run(label, command)
            handle.write(label + '\n' + log + '\n')
            handle.flush()
    total = new = 0
    unchanged = {}
    for path in sorted((ROOT / 'source').glob('*.py')):
        lines = path.read_text().splitlines()
        total += application_lines(lines)
        prev = PREVIOUS / 'source' / path.name
        before = prev.read_text().splitlines() if prev.exists() else []
        for tag, _, _, j1, j2 in difflib.SequenceMatcher(a=before, b=lines, autojunk=False).get_opcodes():
            if tag in ('insert', 'replace'):
                new += application_lines(lines[j1:j2])
        if prev.exists():
            unchanged['source/' + path.name] = path.read_bytes() == prev.read_bytes()
    for path in (PREVIOUS / 'tests').glob('*.py'):
        unchanged['tests/' + path.name] = path.read_bytes() == (ROOT / 'tests' / path.name).read_bytes()
    if not all(unchanged.values()):
        errors += 1
    match = re.search(r'Ran (\d+) tests?', tests.stderr)
    count = int(match.group(1)) if match else 0
    failures = sum(int(n) for n in re.findall(r'(?:failures|errors)=(\d+)', tests.stderr))
    skipped = sum(int(n) for n in re.findall(r'skipped=(\d+)', tests.stderr))
    passed = count - failures - skipped
    history.append({'attempts': attempts, 'unexpected_error_count': errors})
    old.write_text(json.dumps(history, indent=2) + '\n')
    cumulative = sum(attempt['unexpected_error_count'] for attempt in history)
    notes = (f'passed/total={passed}/{count}; skipped={skipped}; test_seconds={timings["tests"]:.9f}; '
             f'regression_demo_seconds={json.dumps({k:v for k,v in timings.items() if k not in ("tests","step7")}, sort_keys=True)}; '
             f'all_previous_source_and_tests_byte_identical={all(unchanged.values())}; '
             'new_loc counts nonblank/non-comment insert/replace new-side lines versus allowed Step 6 source. '
             'Step 2 fixture regenerated from CSV data row 1 because allowed prior tests reference a root fixture outside allowed input paths. '
             'Expected negative-test subprocess exits excluded from error_count. No packages installed or versions changed.')
    with (ROOT / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['step','treatment','trial','domain','skill','total_loc','new_loc','test_pass','runtime_seconds','error_count','notes'])
        writer.writeheader()
        writer.writerow(dict(step=7,treatment=4,trial=1,domain='health_closedissues',skill='choose',
                             total_loc=total,new_loc=new,test_pass=int(tests.returncode == 0 and count > 0),
                             runtime_seconds=f'{timings["step7"]:.9f}',error_count=cumulative,notes=notes))
    (ROOT / 'measurements.json').write_text(json.dumps({'timings_seconds':timings,'tests':{'passed':passed,'total':count,'skipped':skipped},'total_loc':total,'new_loc':new,'unchanged_previous_files':unchanged,'unexpected_errors_this_attempt':errors,'cumulative_unexpected_errors':cumulative},indent=2)+'\n')
    print(f'passed/total={passed}/{count}, total_loc={total}, new_loc={new}, errors={cumulative}')
    return int(errors > 0)


if __name__ == '__main__':
    sys.exit(main())
