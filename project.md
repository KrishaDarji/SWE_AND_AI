# Graduate Project: Shrinking-Code Experiment at Scale

## Project summary

This project tests whether a reusable neuro-symbolic software architecture allows an AI system to add new software-engineering skills with less new code and fewer errors than a collection of independent, monolithic solutions.

The project will use three different MOOT datasets, three invented clients, and one shared codebase. The three domains represent different stages of software work:

1. **Run:** video-encoding configuration.
2. **Build:** software-project planning and estimation.
3. **Maintain:** open-source project health.

The project must honestly support or refute the hypothesis. A domain that does not transfer is a valid result if it is documented and diagnosed.

## Research claim

### Timm hypothesis

Neuro-symbolic systems improve when their symbolic half is factored into small reusable structures. The ten skills are synonyms: one machine—cluster, tree, or leaf—wearing ten hats. Therefore, an LLM or human should extend a well-factored library with less new code and fewer errors than a monolithic system.

### Rival hypothesis

The ten skills are ten separate problems. Factoring provides little or no benefit: skill `i+1` costs approximately the same regardless of the library underneath it.

### Falsifiable prediction

If the Timm hypothesis is correct:

- new-code-per-step should decrease across the ladder;
- reuse should transfer from the first domain to the second and third domains;
- reuse treatments should outperform no-reuse treatments in `same` comparisons;
- execution and test quality should not significantly deteriorate as reuse increases.

If these patterns do not occur, the rival hypothesis gains support.

## Selected domains and datasets

### Domain 1: Video encoding

- Client: video-platform engineer
- Dataset: `X264_AllMeasurements.csv`
- Client goal: choose encoder settings that balance speed, resource use, and output quality.
- Project stage: run

### Domain 2: Software project planning

- Client: engineering manager
- Dataset: `nasa93dem.csv`
- Client goal: estimate and improve effort, defects, development time, and delivered code.
- Project stage: build

### Domain 3: Open-source project health

- Client: open-source project maintainer
- Dataset: `Health-ClosedIssues0000.csv`
- Client goal: identify project-health risks, explain likely causes, and prioritize interventions.
- Project stage: maintain

These datasets are intentionally different. The shared codebase must be tested through domain adapters rather than by copying three separate implementations.

Source repository: [timm/moot](https://github.com/timm/moot)

## Customer requirements

Before implementation, create one requirements document per domain:

```text
docs/requirements/x264.md
docs/requirements/nasa93dem.md
docs/requirements/health-closedissues.md
```

Each document must contain twenty distinct customer asks. Every ask must:

- be written in client/domain language;
- be useful to the client;
- be tagged with one of the ten skills;
- map to one ladder step;
- avoid simply restating another ask.

The ten skill tags are:

```text
remember  guess  certify  flow  blame
justify   choose fix    spot  race
```

The requirements documents must be written and reviewed before the corresponding implementation begins. They should be submitted to the lecturer for approval before the initial deliverable.

## Shared architecture

The codebase should have a domain-independent core and thin domain adapters:

```text
src/
├── core/
│   ├── schema.py
│   ├── data.py
│   ├── goals.py
│   ├── statistics.py
│   ├── ranking.py
│   ├── prediction.py
│   ├── explanation.py
│   ├── optimization.py
│   └── evaluation.py
├── domains/
│   ├── x264/
│   ├── nasa93dem/
│   └── health_closedissues/
├── ladder/
├── treatments/
└── cli.py
```

Supporting directories:

```text
data/                 # downloaded or documented MOOT inputs
tests/                # unit, integration, and regression tests
experiments/          # treatment and repeat runners
results/              # generated tables, plots, and raw outputs
docs/requirements/    # three customer requirements documents
report/               # LaTeX report and figures
scripts/              # reproducible command-line utilities
```

The core should handle common operations such as loading data, identifying independent and dependent attributes, normalizing goals, ranking candidates, learning simple models, measuring uncertainty, explaining decisions, and comparing treatments.

Each domain adapter should define only domain-specific metadata:

- dataset path;
- feature and objective names;
- optimization direction;
- customer vocabulary;
- requirement-to-skill mapping;
- any justified preprocessing.

## Ten-step ladder

The exact wording may differ by domain, but the shared ladder should contain these capabilities:

1. **Remember:** load, represent, summarize, and rank the dataset.
2. **Guess:** predict objective values before a full evaluation.
3. **Certify:** determine whether an observed improvement is meaningful rather than noise.
4. **Flow:** evaluate records or configurations incrementally as they arrive.
5. **Blame:** diagnose which inputs or factors explain poor outcomes.
6. **Justify:** generate an understandable explanation for a recommendation.
7. **Choose:** select a strong candidate while evaluating as few candidates as possible.
8. **Fix:** propose the smallest or cheapest change that improves a poor candidate.
9. **Spot:** detect anomalies, suspicious records, or unusual outcomes.
10. **Race:** compare the shared system with an established baseline or competing method.

Every step must:

- have a runnable demo;
- have tests;
- reuse earlier machinery where appropriate;
- be tagged in Git as `step1`, `step2`, ..., `step10`;
- report the amount of new code introduced.

## Seven treatments

The ladder must be run using seven implementation treatments:

| Treatment | Writer | Background | Reuse instructed? |
|---:|---|---|---|
| 1 | LLM | Plain Python | No |
| 2 | LLM | Plain Python | Yes |
| 3 | LLM | scikit-learn | No |
| 4 | LLM | scikit-learn | Yes |
| 5 | LLM | ezr | No |
| 6 | LLM | ezr | Yes |
| 7 | Human team | Team's prior skills | Yes, as much as possible |

For every treatment, provide the background library, the prior skills, and the next one-line requirement. Measure:

- new lines of code;
- total lines of code;
- test pass rate;
- successful execution rate;
- quality or objective performance;
- errors and failure modes;
- time or token cost where available.

## Repeats and reset rule

Run the full experiment for multiple independent trials. The minimum is three trials; ten is recommended; twenty is ideal.

Each LLM trial must start with no memory of earlier trials:

- use a fresh directory;
- use a fresh process or headless call;
- disable memory features;
- do not provide previous trial context;
- preserve prompts, outputs, code, and logs.

A new chat window alone is not sufficient if persistent project context or memory remains enabled.

## Statistical comparison

Use the repository's `same` procedure to compare treatments and domains. Do not claim a meaningful difference merely because two treatments produce different raw numbers.

Every comparison table should identify:

- treatment pair;
- domain;
- skill or step;
- metric;
- sample size;
- `same` result;
- practical interpretation.

The third domain must include transfer numbers showing how much existing machinery was reused and how much new code was required.

## Repository structure

The project repository should eventually contain:

```text
SWE_AND_AI/
├── project.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── data/
├── docs/
│   ├── requirements/
│   ├── preregistration.md
│   └── experiment-protocol.md
├── src/
├── tests/
├── experiments/
├── results/
├── scripts/
├── report/
└── logs/
```

The README must explain how to:

1. install dependencies;
2. obtain or locate MOOT data;
3. run each domain;
4. run each ladder step;
5. run the treatments;
6. repeat the experiment;
7. regenerate the tables and figures;
8. build the report.

## Milestones

### Milestone 1 — Confirm scope and datasets

**Target: October 8–10, 2026**

- Confirm the three selected MOOT files and exact paths.
- Download or reference the datasets reproducibly.
- Inspect schemas, rows, objectives, missing values, and data types.
- Confirm that the three domains are sufficiently different.
- Create the initial repository structure.

Exit criterion: all three datasets load successfully through a small inspection script.

### Milestone 2 — Write requirements

**Target: October 11–16**

- Invent the three clients.
- Write twenty customer asks per domain.
- Tag every ask with one of the ten skills.
- Map every ask to a planned ladder step.
- Submit the requirements to the lecturer for approval.

Exit criterion: three approved requirements documents exist in `docs/requirements/`.

### Milestone 3 — Preregister the experiment

**Target: October 17–20**

Write `docs/preregistration.md` containing:

- the Timm hypothesis;
- the rival hypothesis;
- new-code-per-step metric;
- error and execution-quality metrics;
- `same` comparison procedure;
- treatment definitions;
- number of repeats;
- win conditions;
- rules for handling failures and missing results.

Exit criterion: hypotheses and win conditions are committed before the main implementation and experiments.

### Milestone 4 — Build the shared foundation

**Target: October 21–26**

Implement:

- data loading;
- schema detection;
- objective definitions;
- normalization;
- summary statistics;
- ranking;
- configuration management;
- command-line execution;
- initial tests.

Exit criterion: the first domain can load and summarize its data with one command.

### Milestone 5 — Complete one domain ladder

**Target: October 27–November 4**

- Implement all ten skills for the first domain.
- Add tests and one-command demos.
- Tag commits `step1` through `step10`.
- Measure new code at every step.
- Generate the first histogram.

Exit criterion: one complete domain ladder runs from a clean checkout.

### Milestone 6 — Initial deliverable

**Target: November 9, 2026**

Submit one page plus repository containing:

- three requirements documents;
- twenty asks per domain;
- one domain's running ladder;
- tagged steps;
- tests;
- one-command demos;
- preregistered hypotheses and win conditions;
- one paragraph describing domains and datasets.

### Milestone 7 — Transfer to the second domain

**Target: November 10–13**

- Add the second domain adapter.
- Reuse the shared core.
- Avoid copying the first domain implementation.
- Run all ten skills.
- Record reused and new code.
- Record failures and domain-specific adjustments.

Exit criterion: the second domain runs with substantially less domain-specific code than the first, or the failure is documented.

### Milestone 8 — Transfer to the third domain

**Target: November 14–17**

- Add the third domain adapter.
- Run all ten skills.
- Record transfer numbers.
- Investigate cases where the shared machinery does not transfer.
- Freeze the main implementation before final comparisons.

Exit criterion: all three domains have reproducible results and none is silently omitted.

### Milestone 9 — Run treatments and repeats

**Target: November 18–21**

- Run all seven treatments.
- Run at least three independent repeats.
- Preserve prompts, generated code, logs, metrics, and failures.
- Apply the reset rule to every LLM trial.
- Run `same` comparisons.

Exit criterion: raw experimental outputs are stored and every claimed comparison is reproducible.

### Milestone 10 — Task presentation

**Target: November 16 or 23, 2026**

Prepare a twenty-minute presentation, aiming to finish in fifteen minutes:

- claim and rival first;
- datasets and domains;
- treatments and repeats;
- evidence and `same` tables;
- shrinking-code histograms;
- transfer results;
- honest verdict: persevere, re-plan, or descope.

Slides must be public and editable by `timm@ieee.org`, and the project must be discussed with the lecturer the preceding week.

### Milestone 11 — Report and final verification

**Target: November 22–29**

- Generate final figures and tables from scripts.
- Write the six-to-nine-page ACM Transactions report.
- Include all three domains and all major failures.
- Verify that the repository works from a clean environment.
- Build the PDF.
- Put the repository URL on page one.

### Final submission — November 30, 2026

Submit:

- ACM Transactions-format report, six to nine pages;
- public repository URL on page one;
- reproducible source code and data instructions;
- treatment logs and results;
- requirements documents;
- tests;
- histograms;
- `same` comparison tables;
- task presentation materials.

## Final success criteria

The project is complete when:

- all three domains have twenty approved requirements;
- one shared codebase supports all three domains;
- all ten skills run for all three domains;
- all steps are Git-tagged;
- all seven treatments are defined and evaluated;
- repeats are independent and documented;
- new-code histograms are reproducible;
- `same` comparisons support the statistical claims;
- transfer to the third domain is quantified;
- failures are reported rather than hidden;
- a clean checkout can reproduce the headline results.
