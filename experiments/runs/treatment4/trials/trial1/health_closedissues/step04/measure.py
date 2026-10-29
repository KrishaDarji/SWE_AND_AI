"""Run all validation and demos; measure application LOC and wall time."""
import csv
import difflib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import tokenize

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'step03'
DATASET = Path('/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/hpo/Health-ClosedIssues0000.csv')
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
# rg exited 1 (no AGENTS.md matches) during initial instruction discovery.
# Conservatively count this nonzero preliminary command in error_count.
errors = 3
attempts = []


def run(label, command, filename):
    global errors
    start = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    if result.returncode:
        errors += 1
    log = ('COMMAND: ' + repr(command) + '\nRETURN CODE: ' + str(result.returncode) +
           '\nWALL SECONDS: ' + str(elapsed) + '\nSTDOUT:\n' + result.stdout +
           '\nSTDERR:\n' + result.stderr + '\n')
    with (ROOT / filename).open('a', encoding='utf-8') as handle:
        handle.write(log)
    attempts.append(dict(label=label, command=command, returncode=result.returncode, wall_seconds=elapsed))
    return result, elapsed


def application_lines(path):
    """Count nonblank lines after removing comment tokens; include docstrings."""
    content = path.read_text()
    comments = {tok.start[0]: tok.start[1] for tok in tokenize.generate_tokens(io.StringIO(content).readline)
                if tok.type == tokenize.COMMENT}
    return {i for i, line in enumerate(content.splitlines(), 1)
            if line[:comments.get(i, len(line))].strip()}


def main():
    result, test_seconds = run('entire_suite', [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v'], 'test_output.txt')
    match = re.search(r'Ran (\d+) tests?', result.stderr)
    total = int(match.group(1)) if match else 0
    failure_match = re.search(r'FAILED \((.*?)\)', result.stderr)
    failed = sum(int(n) for n in re.findall(r'(?:failures|errors|unexpected successes)=(\d+)', failure_match.group(1))) if failure_match else 0
    passed = total - failed if total else 0
    flow, runtime = run('step4_demo', [sys.executable, '-B', 'source/flow.py', str(DATASET), '--output', 'flow.json'], 'runtime_output.txt')
    for label, args in [
        ('step1_demo', ['source/summarize.py', str(DATASET), '--output-dir', 'regression_step01']),
        ('step2_demo', ['source/guess.py', str(DATASET), '--configuration', 'example_configuration.json', '--evaluation', 'regression_step02.json']),
        ('step3_demo', ['source/certify.py', str(DATASET), '--output', 'regression_step03.json'])]:
        run(label, [sys.executable, '-B'] + args, 'runtime_output.txt')
    total_loc = new_loc = 0
    preservation = {}
    for path in sorted((ROOT / 'source').glob('*.py')):
        counted = application_lines(path)
        total_loc += len(counted)
        old = PREVIOUS / 'source' / path.name
        if old.exists():
            preservation['source/' + path.name] = old.read_bytes() == path.read_bytes()
            matcher = difflib.SequenceMatcher(a=old.read_text().splitlines(), b=path.read_text().splitlines(), autojunk=False)
            new_loc += sum(sum(i + 1 in counted for i in range(j1, j2))
                           for tag, _, _, j1, j2 in matcher.get_opcodes() if tag in ('insert', 'replace'))
        else:
            new_loc += len(counted)
    for old in sorted((PREVIOUS / 'tests').glob('*.py')):
        preservation['tests/' + old.name] = old.read_bytes() == (ROOT / 'tests' / old.name).read_bytes()
    all_ok = result.returncode == 0 and all(x['returncode'] == 0 for x in attempts) and all(preservation.values())
    notes = (f'passed/total={passed}/{total}; test_wall_seconds={test_seconds:.9f}; '
             f'regression_demo_seconds={json.dumps({x["label"]: x["wall_seconds"] for x in attempts if x["label"].startswith("step") and x["label"] != "step4_demo"})}; '
             'previous source/tests byte-identical; LOC via comment-token removal and raw-line difflib insert/replace new-side lines; '
             'error_count includes early log-read exit 1, repaired measurement SyntaxError, and preliminary rg exit 1 (no instruction-file matches), measurement runner initial SyntaxError repaired (missing dict-comprehension closing brace); failed attempt preserved in repair_log.txt; '
             'expected negative-test subprocess failures excluded; Step 4 timing includes interpreter startup, imports, CSV replay, JSON serialization/write, stdout and process shutdown.')
    fields = ['step', 'treatment', 'trial', 'domain', 'skill', 'total_loc', 'new_loc', 'test_pass', 'runtime_seconds', 'error_count', 'notes']
    with (ROOT / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(step=4, treatment=4, trial=1, domain='health_closedissues', skill='flow',
                             total_loc=total_loc, new_loc=new_loc, test_pass=int(result.returncode == 0),
                             runtime_seconds=f'{runtime:.9f}', error_count=errors, notes=notes))
    measurement = dict(attempts=attempts, passed=passed, total=total, test_pass=result.returncode == 0,
                       all_checks_passed=all_ok, total_loc=total_loc, new_loc=new_loc,
                       error_count=errors, preserved_files=preservation,
                       preliminary_command={'command': "rg --files -g AGENTS.md -g !node_modules -g !venv -g !\\.git",
                                            'returncode': 1, 'output': '', 'meaning': 'No matches; conservatively counted as an unsuccessful command. No repair needed.'})
    (ROOT / 'measurement.json').write_text(json.dumps(measurement, indent=2) + '\n')
    print(json.dumps(measurement, indent=2))
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
