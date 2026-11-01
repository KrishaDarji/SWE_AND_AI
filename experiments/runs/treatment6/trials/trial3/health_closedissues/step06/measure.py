import subprocess, sys, time
from pathlib import Path
root=Path(__file__).parent
dataset=root.parents[6]/'moot/optimize/hpo/Health-ClosedIssues0000.csv'
def run(name,args):
    measured=args
    if name=='justify': measured=['/usr/bin/time','-p']+args
    t=time.perf_counter(); p=subprocess.run(measured,cwd=root,capture_output=True,text=True); elapsed=time.perf_counter()-t
    (root/(name+'.stdout')).write_text(p.stdout); (root/(name+'.stderr')).write_text(p.stderr); (root/(name+'.status')).write_text(str(p.returncode)); (root/(name+'.time')).write_text('real %.6f\n'%elapsed)
    if name=='justify':
        for line in p.stderr.splitlines():
            if line.startswith('real '): elapsed=float(line.split()[1]); break
    return p.returncode,elapsed
run('tests',['python3','-m','unittest','discover','-s','tests','-v'])
for name,script in [('summary','run_summary.py'),('prediction','run_prediction.py'),('certify','run_certify.py'),('flow','run_flow.py'),('blame','run_blame.py')]:
    run(name,['python3','source/'+script,str(dataset)])
code,elapsed=run('justify',['python3','source/run_justify.py',str(dataset)])
(root/'justify_runtime_seconds.txt').write_text('%.6f\n'%elapsed)
sys.exit(code)
