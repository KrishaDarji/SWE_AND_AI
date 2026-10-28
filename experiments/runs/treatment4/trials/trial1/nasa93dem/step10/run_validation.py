"""Run and capture the complete suite and all ten step demos within this directory."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
DATASET = Path('/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/process/nasa93dem.csv')


def main():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', NASA93_DATASET=str(DATASET))
    sys.path.insert(0, str(ROOT / 'source'))
    from summarize import PREDICTORS, read_projects
    from predict_effort import split_projects
    rows = read_projects(DATASET)
    train, _, _ = split_projects(rows)
    (ROOT / 'example_proposal.json').write_text(json.dumps({p: rows[train[0]][p] for p in PREDICTORS}, indent=2) + '\n')
    records = []
    def run(label, args, log):
        start = time.perf_counter()
        p = subprocess.run([sys.executable] + args, cwd=ROOT, env=env, text=True, capture_output=True)
        elapsed = time.perf_counter() - start
        with log.open('a', encoding='utf-8') as stream:
            stream.write(f'\n{label}\nCommand: {[sys.executable] + args!r}\n')
            stream.write(p.stdout + p.stderr)
            stream.write(f'\nExit: {p.returncode}; wall seconds: {elapsed:.9f}\n')
        records.append(dict(label=label, command=[sys.executable]+args, returncode=p.returncode, seconds=elapsed))
        print(label, p.returncode, elapsed, flush=True)
        return p
    tests = ROOT/'test_output.txt'
    runtime = ROOT/'runtime_output.txt'
    tests.write_text('Complete Steps 1–10 unittest suite\n')
    runtime.write_text('Demo subprocess wall times include interpreter startup, imports, IO and serialization.\n')
    run('tests', ['-m','unittest','discover','-s','tests','-v'], tests)
    demo = ROOT/'regression_demos'
    demo.mkdir(exist_ok=True)
    proposal = demo/'proposal.json'
    proposal.write_text('{"prec": "h", "flex": "h"}\n')
    commands = [
        ('step1', ['source/summarize.py',str(DATASET),'--output-dir',str(demo/'summary')]),
        ('step2', ['source/predict_effort.py',str(DATASET),'--proposal',str(proposal),'--evaluation',str(demo/'evaluation.json')]),
        ('step3', ['source/spot.py',str(DATASET),'--output',str(demo/'spot.json')]),
        ('step4', ['source/blame.py',str(DATASET),'--output',str(demo/'blame.json')]),
        ('step5', ['source/fix.py',str(DATASET),'--demo','--example-proposal',str(demo/'complete_proposal.json'),'--output',str(demo/'fix.json')]),
        ('step6', ['source/choose.py',str(DATASET),'--budget','1000','--output',str(demo/'choose.json')]),
        ('step7', ['source/justify.py',str(DATASET),'--proposal',str(proposal),'--illustrative','--output-dir',str(demo/'justify')]),
        ('step8', ['source/certify.py',str(DATASET),'--output',str(demo/'certify.json')]),
        ('step9', ['source/flow.py',str(DATASET),'--output',str(demo/'flow.json')]),
        ('step10', ['source/race.py',str(DATASET),'--seeds',*[str(s) for s in range(42,52)],'--output',str(ROOT/'race.json')]),
    ]
    for label, command in commands:
        run(label, command, runtime)
    (ROOT/'validation_runs.json').write_text(json.dumps(records, indent=2)+'\n')


if __name__ == '__main__':
    main()
