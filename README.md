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

The seven experimental treatments and milestone plan are documented in [project.md](project.md). Results and visualization requirements are in [result.md](result.md).
