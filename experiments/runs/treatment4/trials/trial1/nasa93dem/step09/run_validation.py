"""Reproduce tests, all nine demos and measured metrics within Step 9 only."""
import argparse
import csv
import difflib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'step08'


def counted_lines(text):
    # Exclude only blank and comment-only lines; retain docstrings.
    return [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith('#')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    args = parser.parse_args()
    dataset = str(args.dataset.resolve())
    sys.path.insert(0, str(ROOT / 'source'))
    from fix import illustrative_proposal
    from summarize import read_projects
    proposal_fixture, _ = illustrative_proposal(read_projects(dataset))
    (ROOT / 'example_proposal.json').write_text(json.dumps(proposal_fixture, indent=2) + '\n')
    errors = 0
    timings = {}
    start = time.perf_counter()
    p = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'), '-v'],
                       cwd=ROOT, capture_output=True, text=True)
    timings['tests_seconds'] = time.perf_counter() - start
    test_text = p.stdout + p.stderr
    (ROOT / 'test_output.txt').write_text(test_text)
    total_match = re.search(r'Ran (\d+) tests?', test_text)
    total = int(total_match.group(1)) if total_match else 0
    failed = sum(int(n) for n in re.findall(r'(?:failures|errors)=(\d+)', test_text))
    passed = total - failed
    errors += int(p.returncode != 0)
    print(test_text[-500:], flush=True)
    demos = ROOT / 'regression_demos'
    demos.mkdir(exist_ok=True)
    proposal = demos / 'proposal.json'
    proposal.write_text('{"prec": "h", "tool": "n"}\n')
    commands = [
        ('step01', 'summarize.py', ['--output-dir', str(demos / 'summary')]),
        ('step02', 'predict_effort.py', ['--proposal', str(proposal), '--evaluation', str(demos / 'evaluation.json')]),
        ('step03', 'choose.py', ['--budget', '1000', '--output', str(demos / 'choice.json')]),
        ('step04', 'flow.py', ['--output', str(demos / 'flow.json')]),
        ('step05', 'blame.py', ['--output', str(demos / 'blame.json')]),
        ('step06', 'justify.py', ['--proposal', str(proposal), '--illustrative', '--output-dir', str(demos / 'justify')]),
        ('step07', 'fix.py', ['--demo', '--example-proposal', str(demos / 'fix_proposal.json'), '--output', str(demos / 'fix.json')]),
        ('step08', 'certify.py', ['--output', str(demos / 'certification.json')]),
        ('step09', 'spot.py', ['--output', str(ROOT / 'anomalies.json'), '--limit', '10']),
    ]
    for label, module, extra in commands:
        command = [sys.executable, str(ROOT / 'source' / module), dataset] + extra
        start = time.perf_counter()
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        elapsed = time.perf_counter() - start
        timings[label + '_seconds'] = elapsed
        errors += int(result.returncode != 0)
        log = f'Command: {command!r}\nReturn code: {result.returncode}\nWall seconds: {elapsed:.9f}\n' + result.stdout + result.stderr
        (demos / (label + '_output.txt')).write_text(log)
        if label == 'step09':
            (ROOT / 'runtime_output.txt').write_text(log)
        print(f'{label}: returncode={result.returncode}, seconds={elapsed:.6f}', flush=True)
    total_loc = 0
    new_loc = 0
    for path in sorted((ROOT / 'source').glob('*.py')):
        content = path.read_text()
        total_loc += len(counted_lines(content))
        old_path = PREVIOUS / 'source' / path.name
        old_lines = old_path.read_text().splitlines() if old_path.exists() else []
        new_lines = content.splitlines()
        for tag, a, b, c, d in difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False).get_opcodes():
            if tag in ('insert', 'replace'):
                new_loc += len(counted_lines('\n'.join(new_lines[c:d])))
    timings['regression_demo_seconds'] = sum(timings[f'step{i:02d}_seconds'] for i in range(1, 9))
    (ROOT / 'timings.json').write_text(json.dumps(timings, indent=2) + '\n')
    notes = (f'{passed}/{total} tests passed; test_seconds={timings["tests_seconds"]:.9f}; '
             f'regression_demo_seconds={timings["regression_demo_seconds"]:.9f}; '
             'initial output-directory ls failed because directory did not yet exist; created directory; premature tail of test_output.txt failed while tests still running; no code/test repair needed; '
             'generated fixture via allowed existing demo function and reran; failures preserved in failed_attempt_01; error_count includes two inspections, failed suite subprocess and failed enclosing validation command; threshold heuristic; no verified anomaly labels; source LOC excludes validation runner')
    with (ROOT / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['step', 'treatment', 'trial', 'domain', 'skill', 'total_loc', 'new_loc', 'test_pass', 'runtime_seconds', 'error_count', 'notes'])
        writer.writeheader()
        writer.writerow(dict(step=9, treatment=4, trial=1, domain='nasa93dem', skill='spot', total_loc=total_loc,
                             new_loc=new_loc, test_pass=int(p.returncode == 0), runtime_seconds=f'{timings["step09_seconds"]:.9f}',
                             error_count=errors + 4, notes=notes))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
