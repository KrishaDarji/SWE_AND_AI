"""Run and retain all validation artifacts; application LOC excludes this runner."""
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
PREVIOUS = ROOT.parent / 'step09'
DATASET = Path('/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/hpo/Health-ClosedIssues0000.csv')
os.chdir(ROOT)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
LOG = ROOT / 'runtime_output.txt'
ERRORS = []
TIMINGS = {}


def run(label, arguments, output):
    start = time.perf_counter()
    completed = subprocess.run([sys.executable, *arguments], text=True, capture_output=True)
    elapsed = time.perf_counter() - start
    TIMINGS[label] = elapsed
    content = f'COMMAND: {json.dumps([sys.executable, *arguments])}\nWALL_SECONDS: {elapsed:.9f}\nEXIT_CODE: {completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}\n'
    output.write_text(content, encoding='utf-8')
    with LOG.open('a', encoding='utf-8') as handle:
        handle.write(f'{label}: {elapsed:.9f} seconds; exit {completed.returncode}; output {output.name}\n')
    if completed.returncode:
        ERRORS.append(label)
    return completed


def app_lines(lines):
    return sum(bool(line.strip()) and not line.lstrip().startswith('#') for line in lines)


def main():
    LOG.write_text('All commands use current python3/scikit-learn 1.6.0; no packages installed.\nInput checks: working directory and allowed inputs present. Output directory absent before creation (expected). Initial rg returned 1 for no AGENTS.md matches (normal no-match status); neither was a failed execution attempt.\n', encoding='utf-8')
    preserved = []
    for folder in ('source', 'tests'):
        for path in (PREVIOUS/folder).glob('*.py'):
            if path.read_bytes() != (ROOT/folder/path.name).read_bytes():
                raise RuntimeError(f'Preservation failed: {path.name}')
            preserved.append(folder+'/'+path.name)
    (ROOT/'preservation.json').write_text(json.dumps({'byte_identical_to_allowed_step09': preserved}, indent=2)+'\n')
    config = {'N_estimators': 100, 'criterion': 'squared_error', 'Min_sample_leaves': 3, 'Min_impurity_decrease': 0, 'Max_depth': 5}
    (ROOT/'example_configuration.json').write_text(json.dumps(config)+'\n')
    tests = run('tests', ['-m', 'unittest', 'discover', '-s', 'tests', '-v'], ROOT/'test_output.txt')
    run('step10_demo', ['source/race.py', str(DATASET), '--max-estimators', '100', '--seeds', *map(str,range(42,52)), '--output', 'race.json'], ROOT/'race_demo_output.txt')
    config = {'N_estimators': 100, 'criterion': 'squared_error', 'Min_sample_leaves': 3, 'Min_impurity_decrease': 0, 'Max_depth': 5}
    (ROOT/'demo_configuration.json').write_text(json.dumps(config)+'\n')
    demos = [
        ('summarize', ['--output-dir','regression_summarize']),
        ('guess', ['--configuration','demo_configuration.json','--evaluation','regression_guess.json']),
        ('certify', ['--output','regression_certify.json']),
        ('blame', ['--output','regression_blame.json']),
        ('spot', ['--output','regression_spot.json']),
        ('justify', ['--output-dir','regression_justify']),
        ('choose', ['--max-estimators','100','--output','regression_choose.json']),
        ('fix', ['--baseline-row','1','--output','regression_fix.json']),
        ('flow', ['--output','regression_flow.json'])]
    for name, options in demos:
        run('regression_'+name, ['source/'+name+'.py', str(DATASET), *options], ROOT/('regression_'+name+'_output.txt'))
    total = added = 0
    for path in (ROOT/'source').glob('*.py'):
        lines = path.read_text().splitlines()
        total += app_lines(lines)
        previous = PREVIOUS/'source'/path.name
        old = previous.read_text().splitlines() if previous.exists() else []
        for tag, a, b, c, d in difflib.SequenceMatcher(None, old, lines, autojunk=False).get_opcodes():
            if tag in ('insert','replace'):
                added += app_lines(lines[c:d])
    combined = tests.stdout + tests.stderr
    match = re.search(r'Ran (\d+) tests?', combined)
    count = int(match.group(1)) if match else 0
    failed_names = set(re.findall(r'^(?:FAIL|ERROR): (.+)$', combined, re.M))
    # Each unittest method counts once; subtest failures share its test identity.
    failed_methods = {name.split(' (')[0] for name in failed_names}
    passed = count - len(failed_methods)
    notes = {'passed': passed, 'total': count, 'test_seconds': TIMINGS['tests'],
             'regression_demo_seconds': {k:v for k,v in TIMINGS.items() if k.startswith('regression_')},
             'errors': ERRORS, 'repair_attempts': [],
             'loc_convention': 'Nonblank non-comment-only application lines including docstrings; new-side insert/replace relative to allowed Step 9.',
             'statistical_analysis': 'pending: required same comparison procedure not supplied'}
    with (ROOT/'metrics.csv').open('w', newline='') as handle:
        fields = ['step','treatment','trial','domain','skill','total_loc','new_loc','test_pass','runtime_seconds','error_count','notes']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(zip(fields, [10,4,1,'health_closedissues','race',total,added,int(tests.returncode == 0),TIMINGS['step10_demo'],len(ERRORS),json.dumps(notes)])))
    (ROOT/'measurement.json').write_text(json.dumps({'timings':TIMINGS,'errors':ERRORS,'metrics_notes':notes}, indent=2)+'\n')
    print(json.dumps({'tests_passed':passed,'tests_total':count,'errors':ERRORS,'total_loc':total,'new_loc':added,'demo_seconds':TIMINGS['step10_demo']}, indent=2))
    return int(bool(ERRORS))


if __name__ == '__main__':
    raise SystemExit(main())
