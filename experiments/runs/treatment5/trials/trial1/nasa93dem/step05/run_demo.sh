#!/bin/sh
set -eu
OUT='/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/experiments/runs/treatment5/trials/trial1/nasa93dem/step05'
DATA='/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/process/nasa93dem.csv'
cd "$OUT"
exec /usr/bin/python3 source/blame.py "$DATA"
