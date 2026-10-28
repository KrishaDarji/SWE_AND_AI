"""Run the Step 8 demo, preserved demos and complete tests; record actual metrics."""
import argparse
import csv
import difflib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'step07'


def application_lines(lines):
    return sum(bool(line.strip()) and not line.lstrip().startswith('#') for line in lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    args = parser.parse_args()
    dataset = str(args.dataset.resolve())
    timings = {}
    errors = 0
    commands = []
    env = dict(os.environ, NASA93_DATASET=dataset, PYTHONDONTWRITEBYTECODE='1')
    log = (ROOT / 'runtime_output.txt').open('a', encoding='utf-8')
    log.write('\n=== Experiment invocation ===\n')
    def run(label, arguments):
        nonlocal errors
        command = [sys.executable] + arguments
        commands.append(shlex.join(command))
        start = time.perf_counter()
        process = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
        elapsed = time.perf_counter() - start
        timings[label] = elapsed
        errors += process.returncode != 0
        log.write(f'\n{label}\n$ {shlex.join(command)}\nwall_seconds={elapsed:.9f}; exit={process.returncode}\n')
        log.write(process.stdout + process.stderr)
        log.flush()
        print(f'{label}: exit={process.returncode}, wall_seconds={elapsed:.6f}', flush=True)
        return process
    demo = run('step08_demo', ['source/fix.py', dataset, '--demo', '--example-proposal', 'example_proposal.json', '--output', 'repair.json'])
    regression = ROOT / 'regression'
    regression.mkdir(exist_ok=True)
    demos = [
        ('step01_summarize', ['source/summarize.py', dataset, '--output-dir', str(regression / 'summary')]),
        ('step02_predict', ['source/predict_effort.py', dataset, '--proposal', 'example_proposal.json', '--evaluation', str(regression / 'evaluation.json')]),
        ('step03_justify', ['source/justify.py', dataset, '--proposal', 'example_proposal.json', '--output-dir', str(regression / 'justification'), '--illustrative']),
        ('step04_blame', ['source/blame.py', dataset, '--output', str(regression / 'blame.json')]),
        ('step05_certify', ['source/certify.py', dataset, '--output', str(regression / 'certification.json')]),
        ('step06_choose', ['source/choose.py', dataset, '--budget', '100', '--output', str(regression / 'selection.json')]),
        ('step07_flow', ['source/flow.py', dataset, '--output', str(regression / 'flow.json')]),
    ]
    for label, arguments in demos:
        run(label, arguments)
    suite = run('test_suite', ['-m', 'unittest', 'discover', '-s', 'tests', '-v'])
    test_text = suite.stdout + suite.stderr
    with (ROOT / 'test_output.txt').open('a', encoding='utf-8') as stream:
        stream.write('\n=== Full suite invocation ===\n' + test_text)
    matched = re.search(r'Ran (\d+) tests?', test_text)
    total = int(matched.group(1)) if matched else 0
    failed = re.search(r'FAILED \((.*?)\)', test_text)
    failures = sum(int(n) for n in re.findall(r'(?:failures|errors)=(\d+)', failed.group(1))) if failed else 0
    total_loc = new_loc = 0
    for path in sorted((ROOT / 'source').glob('*.py')):
        after = path.read_text().splitlines()
        total_loc += application_lines(after)
        old = PREVIOUS / 'source' / path.name
        before = old.read_text().splitlines() if old.exists() else []
        for operation, _, _, start, end in difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
            if operation in ('insert', 'replace'):
                new_loc += application_lines(after[start:end])
    history_path = ROOT / 'execution_history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append({'timings_seconds': timings, 'unexpected_failed_commands': errors,
                    'tests_total': total, 'tests_failed_or_errors': failures,
                    'test_returncode': suite.returncode, 'demo_returncode': demo.returncode})
    history_path.write_text(json.dumps(history, indent=2) + '\n')
    error_count = sum(item['unexpected_failed_commands'] for item in history) + int((ROOT / 'inspection_failure.txt').exists())
    regression_seconds = sum(timings[label] for label, _ in demos)
    notes = (f'passed/total={total - failures}/{total}; test_seconds={timings["test_suite"]:.9f}; '
             f'regression_demo_seconds={regression_seconds:.9f}; '
             f'individual timings and commands in runtime_output.txt; attempts={len(history)}; '
             'Step 8 runtime is demo subprocess wall time only; source LOC excludes blank/comment-only lines, includes docstrings; '
             'new LOC uses insert/replace new-side lines against allowed Step 7 source; '
             'expected negative-test subprocess failures excluded; no packages installed or versions changed')
    with (ROOT / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames='step treatment trial domain skill total_loc new_loc test_pass runtime_seconds error_count notes'.split())
        writer.writeheader()
        writer.writerow(dict(step=8, treatment=4, trial=1, domain='nasa93dem', skill='fix',
                             total_loc=total_loc, new_loc=new_loc,
                             test_pass=int(suite.returncode == 0 and total > 0),
                             runtime_seconds=f'{timings["step08_demo"]:.9f}', error_count=error_count, notes=notes))
    log.close()
    print(f'Tests: {total - failures}/{total}; application LOC={total_loc}; added LOC={new_loc}; unexpected errors={error_count}')
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
