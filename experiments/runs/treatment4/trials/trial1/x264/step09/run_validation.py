"""Run the complete suite and nine demos, preserving outputs and measured timings."""
import csv
import difflib
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent/'step08'
DATASET = Path('/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/config/X264_AllMeasurements.csv')


def app_lines(text):
    return sum(bool(line.strip()) and not line.lstrip().startswith('#') for line in text.splitlines())


def main():
    global DATASET
    if len(sys.argv) > 2:
        raise SystemExit("usage: python3 -B run_validation.py [dataset]")
    if len(sys.argv) == 2:
        DATASET = Path(sys.argv[1]).resolve()
    env = dict(os.environ, X264_DATASET=str(DATASET), PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(ROOT/'tmp'))
    original_hash = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    failures = []
    timings = {}
    tests = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v']
    started = time.perf_counter()
    run = subprocess.run(tests, cwd=ROOT, env=env, capture_output=True, text=True)
    test_returncode = run.returncode
    timings['tests'] = time.perf_counter()-started
    (ROOT/'test_output.txt').write_text('$ '+shlex.join(tests)+'\n'+run.stdout+run.stderr+
                                        f'\nwall_seconds={timings["tests"]:.9f}; exit={run.returncode}\n')
    if run.returncode:
        failures.append('test suite')
    output = []
    modules = ['remember', 'guess', 'choose', 'blame', 'fix', 'justify', 'flow', 'certify', 'spot']
    for module in modules:
        destination = ROOT if module == 'spot' else ROOT/'regression_demos'/module
        command = [sys.executable, '-B', str(ROOT/'source'/f'{module}.py'), str(DATASET),
                   '--output-dir', str(destination)]
        if module == 'fix':
            command += ['--baseline-row', '1']
        started = time.perf_counter()
        run = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        timings[module] = time.perf_counter()-started
        output.append('$ '+shlex.join(command)+'\n'+run.stdout+run.stderr+
                      f'\nwall_seconds={timings[module]:.9f}; exit={run.returncode}\n')
        if run.returncode:
            failures.append(module+' demo')
    unchanged = original_hash == hashlib.sha256(DATASET.read_bytes()).hexdigest()
    if not unchanged:
        failures.append('dataset hash changed')
    for folder in ('source', 'tests'):
        for old in (PREVIOUS/folder).glob('*.py'):
            if old.read_bytes() != (ROOT/folder/old.name).read_bytes():
                failures.append('preserved file changed: '+old.name)
    output.append(f'Dataset SHA256 before/after: {original_hash}; unchanged={unchanged}\n')
    (ROOT/'runtime_output.txt').write_text('\n'.join(output))
    total = new = 0
    for path in (ROOT/'source').glob('*.py'):
        content = path.read_text()
        total += app_lines(content)
        old = PREVIOUS/'source'/path.name
        if not old.exists():
            new += app_lines(content)
        else:
            before, after = old.read_text().splitlines(), content.splitlines()
            for tag, a, b, c, d in difflib.SequenceMatcher(a=before,b=after,autojunk=False).get_opcodes():
                if tag in ('insert','replace'):
                    new += app_lines('\n'.join(after[c:d]))
    test_output = (ROOT/'test_output.txt').read_text()
    import re
    count = re.search(r'Ran (\d+) tests?',test_output)
    test_total = int(count.group(1)) if count else None
    excluded_passes = sum(int(v) for v in re.findall(
        r'(?:failures|errors|skipped|expected failures|unexpected successes)=(\d+)', test_output))
    status = {'timings_seconds': timings, 'unexpected_failures': failures,
              'error_count': len(failures), 'test_total': test_total,
              'test_passed': test_total - excluded_passes if test_total is not None else None,
              'test_pass': int(test_returncode == 0 and '\nOK\n' in test_output),
              'total_loc': total, 'new_loc': new, 'dataset_sha256': original_hash,
              'preserved_files_identical': not any('preserved file' in f for f in failures)}
    (ROOT/'validation.json').write_text(json.dumps(status,indent=2)+'\n')
    with (ROOT/'metrics.csv').open('w',newline='') as stream:
        names = ['step','treatment','trial','domain','skill','total_loc','new_loc','test_pass',
                 'runtime_seconds','error_count','notes']
        writer = csv.DictWriter(stream,fieldnames=names)
        writer.writeheader()
        writer.writerow(dict(step=9,treatment=4,trial=1,domain='x264',skill='spot',
                             total_loc=total,new_loc=new,test_pass=status['test_pass'],
                             runtime_seconds=f'{timings["spot"]:.9f}',error_count=len(failures),
                             notes=json.dumps(status,sort_keys=True)))
    print(json.dumps(status,indent=2))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
