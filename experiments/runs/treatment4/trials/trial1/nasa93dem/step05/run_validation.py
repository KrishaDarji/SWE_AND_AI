"""Capture actual test/demo subprocesses and application LOC metrics locally."""
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
PREVIOUS = ROOT.parent / 'step04'
DATASET = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/process/nasa93dem.csv')
os.chdir(ROOT)
env = dict(os.environ, NASA93_DATASET=str(DATASET), PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(ROOT / 'tests'))


def run(command):
    start = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - start
    return result, elapsed


attempts = []
test, test_seconds = run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'])
attempts.append(('tests', test.returncode))
with (ROOT / 'test_output.txt').open('a') as stream:
    stream.write(f'COMMAND: python3 -m unittest discover -s tests -v\nWall time: {test_seconds:.9f} seconds\nExit code: {test.returncode}\n')
    stream.write(test.stdout + test.stderr + '\n')
commands = [
    ('Step 1 summary', [sys.executable, 'source/summarize.py', str(DATASET), '--output-dir', str(ROOT / 'regression_summary')]),
    ('Step 2 prediction', [sys.executable, 'source/predict_effort.py', str(DATASET), '--proposal', 'proposal.json', '--evaluation', 'regression_prediction.json']),
    ('Step 3 flow', [sys.executable, 'source/flow.py', str(DATASET), '--output', 'regression_flow.json']),
    ('Step 4 comparison', [sys.executable, 'source/certify.py', str(DATASET), '--output', 'regression_certify.json']),
    ('Step 5 blame', [sys.executable, 'source/blame.py', str(DATASET), '--output', 'blame.json']),
]
timings = {}
with (ROOT / 'runtime_output.txt').open('a') as stream:
    for label, command in commands:
        result, elapsed = run(command)
        attempts.append((label, result.returncode))
        timings[label] = elapsed
        stream.write(f'{label}\nCOMMAND: {command!r}\nWall time: {elapsed:.9f} seconds\nExit code: {result.returncode}\n')
        stream.write(result.stdout + result.stderr + '\n')
        stream.flush()


def application_line(line):
    return bool(line.strip()) and not line.lstrip().startswith('#')


total = new = 0
preserved = {}
for path in sorted((ROOT / 'source').glob('*.py')):
    lines = path.read_text().splitlines()
    total += sum(map(application_line, lines))
    earlier = PREVIOUS / 'source' / path.name
    old = earlier.read_text().splitlines() if earlier.exists() else []
    for tag, a, b, c, d in difflib.SequenceMatcher(a=old, b=lines, autojunk=False).get_opcodes():
        if tag in ('insert', 'replace'):
            new += sum(map(application_line, lines[c:d]))
    if earlier.exists():
        preserved[str(path.relative_to(ROOT))] = path.read_bytes() == earlier.read_bytes()
for earlier in sorted((PREVIOUS / 'tests').glob('*.py')):
    preserved['tests/' + earlier.name] = earlier.read_bytes() == (ROOT / 'tests' / earlier.name).read_bytes()
match = re.search(r'Ran (\d+) tests?', test.stderr)
count = int(match.group(1)) if match else 0
command_history = json.loads((ROOT / 'command_attempts.json').read_text()) if (ROOT / 'command_attempts.json').exists() else []
errors = sum(code != 0 for _, code in attempts) + sum(item['exit_code'] != 0 for item in command_history)
notes = {'passed_total': f'{count}/{count}' if test.returncode == 0 else f'see test_output.txt; {count} total',
         'test_seconds': test_seconds, 'regression_demo_seconds': {k: v for k, v in timings.items() if k != 'Step 5 blame'},
         'subprocess_exit_codes': dict(attempts), 'preserved_step04_files_byte_identical': preserved,
         'loc_convention': 'nonblank non-comment-only source Python lines, includes docstrings; new=insert/replace new-side application lines against allowed Step 4',
         'repairs': 'No application repairs. One premature log inspection failed before logs existed; waited for validation to finish. See command_attempts.json.',
         'other_command_attempts': command_history,
         'runtime_scope': 'Step 5 demo subprocess wall time only, includes Python/dependency startup'}
with (ROOT / 'metrics.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=['step','treatment','trial','domain','skill','total_loc','new_loc','test_pass','runtime_seconds','error_count','notes'])
    writer.writeheader()
    writer.writerow(dict(step=5,treatment=4,trial=1,domain='nasa93dem',skill='blame',total_loc=total,new_loc=new,
                         test_pass=int(test.returncode == 0),runtime_seconds=timings['Step 5 blame'],error_count=errors,
                         notes=json.dumps(notes, sort_keys=True)))
(ROOT / 'validation_details.json').write_text(json.dumps(notes, indent=2) + '\n')
print(json.dumps({'test_exit_code': test.returncode, 'test_count': count, 'test_seconds': test_seconds,
                  'timings': timings, 'error_count': errors, 'total_loc': total, 'new_loc': new,
                  'preserved_files': preserved}, indent=2))
