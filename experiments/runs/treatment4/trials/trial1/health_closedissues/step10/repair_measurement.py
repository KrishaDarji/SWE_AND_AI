"""Retain first-run failure, rerun only affected tests and finalize actual metrics."""
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for name in ('metrics.csv', 'measurement.json'):
    path = ROOT/name
    destination = path.with_name(path.stem+'_attempt1'+path.suffix)
    if not destination.exists():
        destination.write_bytes(path.read_bytes())
command = [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v']
start = time.perf_counter()
result = subprocess.run(command, text=True, capture_output=True)
elapsed = time.perf_counter() - start
content = f'COMMAND: {json.dumps(command)}\nWALL_SECONDS: {elapsed:.9f}\nEXIT_CODE: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\n'
(ROOT/'test_output.txt').write_text(content)
with (ROOT/'runtime_output.txt').open('a') as handle:
    handle.write('Preserved failure: initial unittest run had one missing example_configuration.json fixture (98/99 passed). Measurement wrapper exit 1 propagated that same failure, counted once. See test_output_attempt1.txt, metrics_attempt1.csv and measurement_attempt1.json.\n')
    handle.write('Additional operational failure: premature tail of test_output.txt before capture completed: tail: test_output.txt: No such file or directory. Switched to existence-checked progress reads; no application impact. Counted once.\n')
    handle.write('Repair: generated example_configuration.json from an explicit valid illustrative configuration inside Step 10; no prior artifact was read. Preserved tests/source unchanged.\n')
    handle.write(f'tests_after_fixture_repair: {elapsed:.9f} seconds; exit {result.returncode}; test_output.txt. No demo reruns: all original demos passed.\n')
count = int(re.search(r'Ran (\d+) tests?', result.stderr).group(1))
with (ROOT/'metrics.csv').open(newline='') as handle:
    reader = csv.DictReader(handle)
    fields = reader.fieldnames
    row = next(reader)
notes = json.loads(row['notes'])
notes['initial_test_seconds'] = notes['test_seconds']
notes['initial_passed_total'] = '98/99'
notes['test_seconds'] = elapsed
notes['passed'] = count if result.returncode == 0 else None
notes['total'] = count
notes['errors'] = ['initial test execution: missing fixture', 'premature tail progress-read command']
notes['repair_attempts'] = ['Created explicit valid example_configuration.json locally; reran entire suite. Retained original test failure and original demo measurements.']
notes['error_count_convention'] = 'Two distinct unexpected root failures; propagation of initial test failure by measurement wrapper is not double-counted. Expected negative tests, absent output directory check and rg no-match status excluded.'
row['test_pass'] = str(int(result.returncode == 0))
row['error_count'] = str(2 + int(result.returncode != 0))
row['notes'] = json.dumps(notes)
with (ROOT/'metrics.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerow(row)
measurement = json.loads((ROOT/'measurement.json').read_text())
measurement['timings']['tests_after_fixture_repair'] = elapsed
measurement['errors'] = notes['errors']
measurement['metrics_notes'] = notes
(ROOT/'measurement.json').write_text(json.dumps(measurement,indent=2)+'\n')
print(json.dumps({'test_returncode':result.returncode,'total':count,'test_seconds':elapsed,'cumulative_error_count':row['error_count']},indent=2))
raise SystemExit(result.returncode)
