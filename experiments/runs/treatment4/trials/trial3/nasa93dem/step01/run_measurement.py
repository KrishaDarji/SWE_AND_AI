import subprocess, sys, time
from pathlib import Path
base = Path(__file__).resolve().parent
dataset = base.parents[6] / 'moot' / 'optimize' / 'process' / 'nasa93dem.csv'
test_log = base / 'test_attempt_002.txt'
with test_log.open('w') as out:
    test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-t', '.'], cwd=base, stdout=out, stderr=subprocess.STDOUT, text=True)
start = time.perf_counter()
runtime_log = base / 'runtime_attempt_002.txt'
with runtime_log.open('w') as out:
    demo = subprocess.run([sys.executable, 'source/nasa93_summary.py', str(dataset)], cwd=base, stdout=out, stderr=subprocess.STDOUT, text=True)
elapsed = time.perf_counter() - start
(base / 'test_output.txt').write_text('Latest attempt: test_attempt_001.txt\n' + test_log.read_text())
(base / 'runtime_output.txt').write_text(f'Latest attempt: runtime_attempt_001.txt\nreal_seconds={elapsed:.6f}\n' + runtime_log.read_text())
(base / 'measured_real_seconds.txt').write_text(f'{elapsed:.6f}\n')
sys.exit(test.returncode or demo.returncode)
