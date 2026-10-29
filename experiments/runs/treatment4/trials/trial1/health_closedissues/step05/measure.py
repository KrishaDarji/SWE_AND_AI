"""Measure subprocesses and application LOC; outputs remain in this directory."""
import csv
import difflib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import tokenize

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'step04' / 'source'
DATASET = Path('/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/hpo/Health-ClosedIssues0000.csv')


def application_lines(text):
    # Exclude blank and comment-only lines, retaining docstrings and inline comments.
    comments = {token.start[0] for token in tokenize.generate_tokens(io.StringIO(text).readline)
                if token.type == tokenize.COMMENT and not text.splitlines()[token.start[0]-1][:token.start[1]].strip()}
    return {i for i, line in enumerate(text.splitlines(), 1) if line.strip() and i not in comments}


def main():
    attempts = []
    def run(label, args):
        command = [sys.executable, *map(str, args)]
        start = time.perf_counter()
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        elapsed = time.perf_counter() - start
        entry = dict(label=label, command=command, returncode=result.returncode, wall_seconds=elapsed,
                     stdout=result.stdout, stderr=result.stderr)
        attempts.append(entry)
        with (ROOT / 'attempts.jsonl').open('a') as handle:
            handle.write(json.dumps(entry) + '\n')
        return entry
    tests = run('all_tests', ['-m', 'unittest', 'discover', '-s', 'tests', '-v'])
    with (ROOT / 'test_output.txt').open('a') as handle:
        handle.write(json.dumps({k: v for k, v in tests.items() if k not in ('stdout', 'stderr')}) + '\n' + tests['stdout'] + tests['stderr'])
    demo = run('step5_demo', ['source/blame.py', DATASET, '--output', 'blame.json'])
    regressions = [run('step1_demo', ['source/summarize.py', DATASET, '--output-dir', 'regression/step1']),
                   run('step2_demo', ['source/guess.py', DATASET, '--configuration', 'example_configuration.json', '--evaluation', 'regression/step2.json']),
                   run('step3_demo', ['source/certify.py', DATASET, '--output', 'regression/step3.json']),
                   run('step4_demo', ['source/flow.py', DATASET, '--output', 'regression/step4.json'])]
    with (ROOT / 'runtime_output.txt').open('a') as handle:
        for entry in [demo, *regressions]:
            handle.write(json.dumps({k: v for k, v in entry.items() if k not in ('stdout', 'stderr')}) + '\n' + entry['stdout'] + entry['stderr'] + '\n')
    total = added = 0
    for path in sorted((ROOT / 'source').glob('*.py')):
        text = path.read_text()
        lines = application_lines(text)
        total += len(lines)
        old_path = PREVIOUS / path.name
        old = old_path.read_text().splitlines() if old_path.exists() else []
        new = text.splitlines()
        for tag, i, j, a, b in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
            if tag in ('insert', 'replace'):
                added += sum(line in lines for line in range(a+1, b+1))
    # Count all failed recorded subprocess attempts, retaining history after repairs.
    history = [json.loads(line) for line in (ROOT / 'attempts.jsonl').read_text().splitlines()]
    errors = sum(entry['returncode'] != 0 for entry in history)
    notes = {'test_counts': 'See test_output.txt unittest count; successful entire suite means passed=total',
             'test_seconds': tests['wall_seconds'],
             'regression_demo_seconds': {e['label']: e['wall_seconds'] for e in regressions},
             'step5_exit_code': demo['returncode'], 'python': sys.executable,
             'loc_method': 'Nonblank non-comment-only application lines, docstrings included; new-side insert/replace against Step 4, no net-growth shortcut.',
             'attempt_history': 'attempts.jsonl; expected negative tests excluded from error_count'}
    import re
    match = re.search(r'Ran (\d+) tests?', tests['stderr'])
    notes['passed_total'] = f"{match[1]}/{match[1]}" if match and tests['returncode'] == 0 else 'See test_output.txt'
    with (ROOT / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['step','treatment','trial','domain','skill','total_loc','new_loc','test_pass','runtime_seconds','error_count','notes'])
        writer.writeheader()
        writer.writerow(dict(step=5,treatment=4,trial=1,domain='health_closedissues',skill='blame',
                             total_loc=total,new_loc=added,test_pass=int(tests['returncode']==0),
                             runtime_seconds=demo['wall_seconds'],error_count=errors,notes=json.dumps(notes)))
    print(json.dumps({'tests_exit_code': tests['returncode'], 'demo_exit_codes': [e['returncode'] for e in [demo,*regressions]],
                      'total_loc': total, 'new_loc': added, 'error_count': errors, 'notes': notes}, indent=2))


if __name__ == '__main__':
    (ROOT / 'regression').mkdir(exist_ok=True)
    main()
