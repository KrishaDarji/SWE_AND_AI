"""Run the preserved suite and demos, recording actual subprocess timings."""
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

STEP = Path(__file__).resolve().parent
PREVIOUS = STEP.parent / 'step02'
DATASET = Path('/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/config/X264_AllMeasurements.csv')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counted(line):
    return bool(line.strip()) and not line.lstrip().startswith('#')


def main():
    inputs = [DATASET, *sorted((PREVIOUS / 'source').glob('*.py')),
              *sorted((PREVIOUS / 'tests').glob('*.py'))]
    before = {str(p): digest(p) for p in inputs}
    env = dict(os.environ, X264_DATASET=str(DATASET), PYTHONDONTWRITEBYTECODE='1')
    records = []
    for name, arguments, logfile in (
        ('tests', ['-m', 'unittest', 'discover', '-s', 'tests', '-v'], 'test_output.txt'),
        ('step1_demo', ['source/remember.py', str(DATASET), '--output-dir', str(STEP / 'regression_step01')], 'runtime_output.txt'),
        ('step2_demo', ['source/guess.py', str(DATASET), '--output-dir', str(STEP / 'regression_step02')], 'runtime_output.txt'),
        ('step3_demo', ['source/certify.py', str(DATASET), '--output-dir', str(STEP)], 'runtime_output.txt'),
    ):
        start = time.perf_counter()
        result = subprocess.run([sys.executable, '-B', *arguments], cwd=STEP,
                                env=env, text=True, capture_output=True)
        elapsed = time.perf_counter() - start
        record = dict(name=name, command=[sys.executable, '-B', *arguments],
                      returncode=result.returncode, wall_seconds=elapsed)
        records.append(record)
        # Append rather than discard earlier attempts or failures.
        with (STEP / logfile).open('a', encoding='utf-8') as output:
            output.write(json.dumps(record) + '\nSTDOUT:\n' + result.stdout
                         + '\nSTDERR:\n' + result.stderr + '\n')
        print(json.dumps(record), flush=True)
        if name == 'tests':
            total_match = re.search(r'Ran (\d+) tests?', result.stderr)
            total = int(total_match.group(1)) if total_match else 0
            passed = total if result.returncode == 0 else len(re.findall(r'\.\.\. ok$', result.stderr, re.M))
            suite_pass = result.returncode == 0 and total > 0 and passed == total
    after = {str(p): digest(p) for p in inputs}
    unchanged = before == after
    copies_equal = all((STEP / p.parent.name / p.name).read_bytes() == p.read_bytes()
                       for p in inputs if p != DATASET)
    inventory = {}
    total_loc = new_loc = 0
    for path in sorted((STEP / 'source').rglob('*.py')):
        lines = path.read_text().splitlines()
        old_path = PREVIOUS / 'source' / path.relative_to(STEP / 'source')
        old = old_path.read_text().splitlines() if old_path.exists() else []
        added = sum(counted(line)
                    for tag, _, _, start, end in difflib.SequenceMatcher(None, old, lines, autojunk=False).get_opcodes()
                    if tag in ('insert', 'replace') for line in lines[start:end])
        count = sum(map(counted, lines))
        inventory[path.name] = dict(total_loc=count, new_loc=added)
        total_loc += count
        new_loc += added
    unexpected = 2 + sum(r['returncode'] != 0 for r in records)
    notes = dict(passed=passed, total=total,
                 test_seconds=records[0]['wall_seconds'],
                 step1_demo_seconds=records[1]['wall_seconds'],
                 step2_demo_seconds=records[2]['wall_seconds'],
                 all_inputs_unchanged=unchanged, preserved_copies_identical=copies_equal,
                 initial_failure='python: command not found (exit 127); repaired by using python3; no dependency changes',
                 expected_negative_test_exits_excluded=True,
                 runner_repair='First runner exited 1 because expected-error stderr split two unittest status lines; all 25 tests had passed. Count successful suite from return code and Ran summary. Initial metrics and execution record preserved.',
                 loc_method='Nonblank, non-comment-only lines; docstrings included. difflib insert/replace new-side application lines against allowed Step 2 source.',
                 loc_by_file=inventory)
    with (STEP / 'metrics.csv').open('w', newline='') as f:
        fields = ['step','treatment','trial','domain','skill','total_loc','new_loc',
                  'test_pass','runtime_seconds','error_count','notes']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(step=3, treatment=4, trial=1, domain='x264', skill='certify',
                             total_loc=total_loc, new_loc=new_loc, test_pass=int(suite_pass),
                             runtime_seconds=records[3]['wall_seconds'], error_count=unexpected,
                             notes=json.dumps(notes, sort_keys=True)))
    (STEP / 'execution_record.json').write_text(json.dumps(
        dict(runs=records, input_sha256_before=before, input_sha256_after=after,
             notes=notes), indent=2) + '\n')
    if not unchanged or not copies_equal:
        raise RuntimeError('Input preservation or unchanged reuse check failed')
    return 0 if suite_pass and all(r['returncode'] == 0 for r in records) else 1


if __name__ == '__main__':
    raise SystemExit(main())
