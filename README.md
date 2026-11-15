# Shrinking-Code Experiment at Scale

Graduate CSC491/591 project using three MOOT datasets and one shared codebase.

## Selected datasets

- `moot/optimize/config/X264_AllMeasurements.csv`
- `moot/optimize/process/nasa93dem.csv`
- `moot/optimize/hpo/Health-ClosedIssues0000.csv`

## Run the foundation checks

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
python3 scripts/inspect_datasets.py
python3 scripts/run_ladder.py
python3 scripts/prepare_treatments.py
```

## Treatment 7 (human team, shared core)

The shared core lives in `src/core/`, one thin adapter per domain in `src/domains/`,
and one small demo per ladder step in `src/ladder/`.

```bash
python3 -m src.cli x264 3                 # one ladder step for one domain
python3 -m src.cli nasa93dem all          # all ten steps
python3 experiments/run_treatment7.py     # full trial: demos, tests, LOC metrics
```

The runner writes `experiments/runs/treatment7/trials/trial1/<domain>/stepNN/`
(requirement, source snapshot, tests, outputs, `metrics.csv`) plus per-domain and
trial-level `metrics.csv`. Treatment 7 is run once per domain because a human team
cannot be reset between repeats.

The seven experimental treatments and milestone plan are documented in [project.md](project.md). Results and visualization requirements are in [result.md](result.md).
