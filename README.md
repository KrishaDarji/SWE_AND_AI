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

## One-command demos

Each ladder step runs in one command for each domain (`<domain>` is `x264`, `nasa93dem` or `health_closedissues`):

```bash
python3 -m src.cli <domain> <step>        # e.g. python3 -m src.cli x264 3; use "all" for all ten steps
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Step tags and shrinking-code histograms

Every step of every ladder is tagged on the `ladders` branch: `t<T>/trial<R>/<domain>/stepNN` for the LLM treatments 1–6, and `t7/<domain>/stepNN` for the human team. The human team's tags form one continuous ladder across the three domains, with the shared codebase accumulating.

The steps were not committed one at a time while the experiment ran. `scripts/build_ladder_tags.py` reconstructs the tagged history from the per-step snapshots the experiment saved (`experiments/runs/.../stepNN/source`).

```bash
git fetch origin 'refs/tags/*:refs/tags/*'                  # get the step tags
git diff --stat t2/trial1/x264/step02 t2/trial1/x264/step03 # new code in one step
python3 scripts/tag_histogram.py                            # % new code per step, from the tags -> results/tag_histogram.csv
python3 scripts/build_ladder_tags.py                        # rebuild the branch and tags from the snapshots
```

## External metrics: are the treatments any good?

`evaluation/` runs every treatment's saved guess (step 2) and race (step 10) code on one shared 80/20 split, then ranks the seven treatments with `same`. It makes no new LLM calls and edits no generated code. See [evaluation/README.md](evaluation/README.md).

## Documents

- Requirements: [docs/requirements.md](docs/requirements.md) (three domains, twenty asks each)
- AI disclosure notes: [docs/ai_disclosure.md](docs/ai_disclosure.md)
- Final presentation source: `presentation/source/`
- Plan and milestones: [project.md](project.md); results and figures plan: [result.md](result.md)
