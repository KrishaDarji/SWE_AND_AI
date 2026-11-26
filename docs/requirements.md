# Graduate Project Requirements

This repository evaluates the shrinking-code hypothesis across three clients,
three MOOT datasets, and one shared ten-skill ladder. Each domain has twenty
customer-facing requirements. The complete requirements are maintained in the
domain documents below.

## Domains

| Domain | Client | Dataset | Requirements |
|---|---|---|---|
| X264 video encoding | Video-platform performance engineer | `moot/optimize/config/X264_AllMeasurements.csv` | [x264 requirements](requirements/x264.md) |
| NASA93 software planning | Engineering planning manager | `moot/optimize/process/nasa93dem.csv` | [NASA93DEM requirements](requirements/nasa93dem.md) |
| Health model configuration | Machine-learning platform maintainer | `moot/optimize/hpo/Health-ClosedIssues0000.csv` | [Health-ClosedIssues requirements](requirements/health-closedissues.md) |

Each document contains twenty distinct customer asks written in domain
language, with every ask tagged by the skill it demonstrates.

## Shared ten-skill vocabulary

1. `remember` — summarize historical data and identify strong examples.
2. `guess` — predict an outcome before full evaluation.
3. `certify` — distinguish a meaningful improvement from noise.
4. `flow` — evaluate observations arriving sequentially.
5. `blame` — identify factors associated with poor outcomes.
6. `justify` — explain a recommendation to a stakeholder.
7. `choose` — select an option under explicit constraints.
8. `fix` — find a small or inexpensive improvement.
9. `spot` — detect anomalies or suspicious records.
10. `race` — compare the system with a baseline.

## Traceability

The implementation maps each step to one of these skills. The experiment
results must connect each skill to its source code, tests, runtime output, and
metrics. The requirements documents are the semantic acceptance criteria; a
passing syntax test alone does not demonstrate that a customer ask was met.
