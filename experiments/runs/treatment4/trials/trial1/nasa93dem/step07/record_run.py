"""Reproduce this step's suite, demos, independent check, and measured metrics."""
import csv
from decimal import Decimal
import difflib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

import sklearn

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'step06'
DEFAULT_DATASET = Path('/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/process/nasa93dem.csv')
DATASET = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATASET
if sklearn.__version__ != '1.6.0':
    raise SystemExit('Requires the existing scikit-learn 1.6.0 environment; do not install or change versions.')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['NASA93_DATASET'] = str(DATASET)
run_id = time.time_ns()
attempts = []


def execute(label, args, log):
    command = [sys.executable, *map(str, args)]
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    seconds = time.perf_counter() - started
    record = {'label': label, 'command': command, 'returncode': result.returncode, 'seconds': seconds}
    attempts.append(record)
    with (ROOT / log).open('a') as stream:
        stream.write(f'\nRUN {run_id}: {label}\nCommand: {shlex.join(command)}\n')
        stream.write(result.stdout + result.stderr)
        stream.write(f'\nReturn code: {result.returncode}; subprocess wall seconds: {seconds:.9f}\n')
    print(f'{label}: returncode={result.returncode}, seconds={seconds:.6f}', flush=True)
    return result, seconds


suite, test_seconds = execute('entire suite: Steps 1–7', ['-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v'], 'test_output.txt')
reg = ROOT / 'regression_demos'
reg.mkdir(exist_ok=True)
# Newly constructed illustrative empty proposal; no previous proposal file read.
(ROOT / 'example_proposal.json').write_text('{}\n')
demos = [
    ('Step 1 summarize', ['source/summarize.py', DATASET, '--output-dir', reg]),
    ('Step 2 predict', ['source/predict_effort.py', DATASET, '--proposal', ROOT / 'example_proposal.json', '--evaluation', reg / 'evaluation.json']),
    ('Step 3 flow', ['source/flow.py', DATASET, '--output', reg / 'flow.json']),
    ('Step 4 certify', ['source/certify.py', DATASET, '--output', reg / 'certification.json']),
    ('Step 5 blame', ['source/blame.py', DATASET, '--output', reg / 'associations.json']),
    ('Step 6 justify', ['source/justify.py', DATASET, '--proposal', ROOT / 'example_proposal.json', '--output-dir', reg, '--illustrative']),
]
for label, command in demos:
    execute(label, command, 'regression_demo_output.txt')
demo, runtime = execute('Step 7 choose budget 100', ['source/choose.py', DATASET, '--budget', '100', '--output', ROOT / 'selection.json'], 'runtime_output.txt')

# Independent exhaustive oracle: raw CSV + Decimal + pairwise dominance.
verification = {'method': 'Raw CSV Decimal parsing and exhaustive pairwise dominance; no production selection imports.'}
try:
    with DATASET.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    eligible, feasible = [], []
    for number, row in enumerate(rows, 1):
        if any(row[f].strip() in {'', '?', 'NA', 'N/A', 'null'} for f in ['Effort-', 'Kloc+']):
            continue
        e, v = Decimal(row['Effort-']), Decimal(row['Kloc+'])
        eligible.append(number)
        if e <= Decimal('100'):
            feasible.append((number, e, v))
    winners = []
    for n, e, v in feasible:
        if not any(v2 > v or (v2 == v and e2 < e) or (v2 == v and e2 == e and n2 < n)
                   for n2, e2, v2 in feasible):
            winners.append((n, e, v))
    result = json.loads((ROOT / 'selection.json').read_text())
    winner = winners[0] if winners else None
    selected = result['selected_project']
    assert result['counts']['eligible_rows'] == len(eligible)
    assert result['counts']['feasible_candidates'] == len(feasible)
    assert (selected['csv_data_row'] if selected else None) == (winner[0] if winner else None)
    if winner:
        assert selected['observed_outcomes']['Effort-'] == float(winner[1])
        assert selected['observed_outcomes']['Kloc+'] == float(winner[2])
        assert winner[1] <= Decimal('100')
    verification.update({'passed': True, 'total_rows': len(rows), 'eligible_rows': len(eligible),
                         'feasible_candidates': len(feasible), 'excluded_rows': len(rows) - len(eligible),
                         'winning_csv_data_row': winner[0] if winner else None,
                         'winning_effort_decimal': str(winner[1]) if winner else None,
                         'winning_kloc_decimal': str(winner[2]) if winner else None})
except (AssertionError, OSError, ValueError, KeyError) as exc:
    verification.update({'passed': False, 'error': repr(exc)})
(ROOT / 'independent_verification.json').write_text(json.dumps(verification, indent=2) + '\n')


def counted(line):
    return bool(line.strip()) and not line.lstrip().startswith('#')


total_loc = new_loc = 0
unchanged = {}
for path in sorted((ROOT / 'source').rglob('*.py')):
    lines = path.read_text().splitlines()
    total_loc += sum(map(counted, lines))
    old = PREVIOUS / 'source' / path.relative_to(ROOT / 'source')
    before = old.read_text().splitlines() if old.exists() else []
    for operation, a, b, c, d in difflib.SequenceMatcher(a=before, b=lines, autojunk=False).get_opcodes():
        if operation in ('insert', 'replace'):
            new_loc += sum(map(counted, lines[c:d]))
    if old.exists():
        unchanged[str(path.relative_to(ROOT))] = path.read_bytes() == old.read_bytes()
for path in (PREVIOUS / 'tests').glob('*.py'):
    unchanged['tests/' + path.name] = (ROOT / 'tests' / path.name).read_bytes() == path.read_bytes()
assert all(unchanged.values()), 'Preserved modules/tests unexpectedly changed'
(ROOT / 'preservation_check.json').write_text(json.dumps(unchanged, indent=2) + '\n')
match = re.search(r'Ran (\d+) tests?', suite.stderr)
total_tests = int(match.group(1)) if match else 0
fail = re.search(r'FAILED \((.*?)\)', suite.stderr)
failed_tests = sum(int(n) for n in re.findall(r'(?:failures|errors)=(\d+)', fail.group(1))) if fail else 0
passed = total_tests - failed_tests
ledger = ROOT / 'execution_history.jsonl'
with ledger.open('a') as stream:
    for attempt in attempts:
        stream.write(json.dumps({'run_id': run_id, **attempt}) + '\n')
    stream.write(json.dumps({'run_id': run_id, 'label': 'independent verification', 'returncode': 0 if verification['passed'] else 1}) + '\n')
error_count = sum(json.loads(line)['returncode'] != 0 for line in ledger.read_text().splitlines())
notes = {
    'passed_tests': passed, 'total_tests': total_tests,
    'test_wall_seconds': test_seconds,
    'regression_demo_wall_seconds': {a['label']: a['seconds'] for a in attempts if a['label'].startswith('Step ') and not a['label'].startswith('Step 7')},
    'independent_verification_passed': verification['passed'],
    'input_path_correction': 'Authorized tests path typo corrected by user; initial blocked response preserved in run_history.txt; not an application failure.',
    'expected_negative_tests': 'Handled in passing tests; excluded from error_count.',
    'reuse': 'Six source files and six test files byte-identical to authorized Step 6. Newly created empty illustrative proposal {} supports preserved tests.',
    'loc_convention': 'Nonblank, non-comment-only source Python lines, including docstrings; new LOC counts insert/replace new-side lines versus Step 6, not net growth.',
    'runtime_scope': 'Step 7 demo subprocess only, including interpreter/import/CLI/output overhead.',
    'budget': '100 illustrative unspecified effort units; no monetary/sponsor-approved threshold or objective-evaluation budget.',
}
fields = ['step', 'treatment', 'trial', 'domain', 'skill', 'total_loc', 'new_loc', 'test_pass', 'runtime_seconds', 'error_count', 'notes']
with (ROOT / 'metrics.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerow(dict(zip(fields, [7, 4, 1, 'nasa93dem', 'choose', total_loc, new_loc, int(suite.returncode == 0), f'{runtime:.9f}', error_count, json.dumps(notes, sort_keys=True)])))
print(json.dumps({'tests': f'{passed}/{total_tests}', 'total_loc': total_loc, 'new_loc': new_loc,
                  'error_count': error_count, 'verification': verification}, indent=2), flush=True)
