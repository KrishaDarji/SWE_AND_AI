# Results and Metrics Plan

This document defines the metrics, tables, and visualizations for the shrinking-code experiment.

The results should answer five questions:

1. Does the amount of new code shrink across ladder steps?
2. Does code reuse transfer across the three domains?
3. Does reuse preserve or improve solution quality?
4. Does reuse affect reliability and correctness?
5. Which treatments are statistically better according to `same`?

## 1. Code-reuse metrics

These metrics directly test the Timm hypothesis.

- New lines of code per step.
- Percentage of new code per step.
- Cumulative lines of code.
- Reused-code percentage.
- Number of files changed per step.
- Prompt and token cost for LLM treatments, when available.

### New-code percentage

```text
new-code percentage =
new lines introduced at step i / total lines after step i × 100
```

### Transfer reuse percentage

```text
transfer reuse percentage =
shared code reused / total code for the new domain × 100
```

## 2. Solution-quality metrics

Report these metrics separately for each MOOT domain:

- Objective value.
- Hypervolume.
- Number of non-dominated solutions.
- Best solution found.
- Average solution quality.
- Distance from a reference Pareto front, when available.
- Optimization regret, when a reliable reference optimum exists.

For multi-objective datasets, normalize objectives before comparing domains. Clearly document whether an objective is minimized or maximized.

## 3. Reliability and correctness metrics

A smaller implementation is not useful if it fails more often. Measure:

- Test pass rate.
- Demo execution success rate.
- Runtime error count.
- Failed-step count.
- Incorrect-output count.
- Reproducibility across repeats.
- Runtime.
- Memory usage, if practical.

## 4. Transfer metrics

For each new domain, record:

- New domain-specific lines of code.
- Percentage of existing code reused.
- Number of new files.
- Number of shared components used.
- Number of domain-specific exceptions.
- Time required to add the domain.
- Performance change after transfer.

The third domain must include explicit transfer numbers. It must not be silently omitted from the analysis.

## 5. Statistical comparison

Use the repository's `same` procedure to compare treatments and domains. Raw numerical differences alone are not sufficient evidence of a meaningful improvement.

Every comparison should record:

- Treatment pair.
- Domain.
- Step or skill.
- Metric.
- Number of repeats.
- `same` result.
- Practical interpretation.

Possible summary categories are:

- Statistically certified winner.
- Statistically indistinguishable.
- Statistically worse.

## 6. Primary figures

### Figure 1 — New-code histogram

- X-axis: ladder step.
- Y-axis: percentage of new code.
- Series or panels: seven treatments.

Purpose: test whether new code decreases as skills are added.

### Figure 2 — Reused code versus new code

Use stacked bars for every treatment and domain.

```text
100% = reused code + new code
```

Purpose: show whether reuse dominates as the ladder grows.

### Figure 3 — Domain-transfer plot

- X-axis: domain.
- Y-axis: new domain-specific LOC.

Purpose: compare the cost of adding the first, second, and third domains.

### Figure 4 — Quality versus code size

- X-axis: total code or new code.
- Y-axis: solution quality or hypervolume.
- Color: treatment.
- Shape: domain.

Purpose: show whether smaller implementations remain effective.

### Figure 5 — Reliability heatmap

- Rows: treatments.
- Columns: domains.
- Cell value: test pass rate, execution success rate, or failure percentage.

Purpose: identify treatments that produce compact but unreliable systems.

### Figure 6 — Pareto-front comparison

Create one panel for each domain. Plot the reference or best-known solutions against the human, LLM-reuse, and LLM-no-reuse treatments.

Purpose: compare optimization quality and trade-offs.

### Figure 7 — `same` comparison matrix

- Rows and columns: treatments.
- Cell color: certified win, indistinguishable, or certified loss.
- Use one matrix per domain or one combined matrix.

Purpose: show which treatments actually differ statistically.

### Figure 8 — Error and failure analysis

Use stacked bars for:

- Syntax errors.
- Runtime errors.
- Test failures.
- Incorrect objective handling.
- Data-processing errors.
- Domain-transfer failures.

Purpose: support an honest explanation of failures.

## 7. Summary table

Generate a table similar to the following for the final report:

| Domain | Treatment | New LOC | Reuse % | Test pass % | Hypervolume | Runtime | Failures | `same` wins |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| X264 | LLM + reuse |  |  |  |  |  |  |  |
| NASA93 | LLM + reuse |  |  |  |  |  |  |  |
| Health | LLM + reuse |  |  |  |  |  |  |  |

The complete experiment data should be stored in machine-readable form, for example:

```text
results/raw/
results/metrics.csv
results/same_comparisons.csv
results/figures/
```

## 8. Suggested result files

```text
results/
├── raw/
├── metrics.csv
├── code_size.csv
├── quality.csv
├── reliability.csv
├── transfer.csv
├── same_comparisons.csv
├── summary_table.csv
└── figures/
    ├── new_code_histogram.png
    ├── reuse_vs_new_code.png
    ├── domain_transfer.png
    ├── quality_vs_code_size.png
    ├── reliability_heatmap.png
    ├── pareto_x264.png
    ├── pareto_nasa93.png
    ├── pareto_health.png
    ├── same_matrix.png
    └── failure_analysis.png
```

## 9. Interpretation rules

Every figure must answer a specific research question. Do not add graphs merely to increase the number of graphs.

- A falling new-code curve supports the shrinking-code portion of the Timm hypothesis.
- Increasing reuse across domains supports transferability.
- Stable or improved quality supports the usefulness of reuse.
- Higher failure rates weaken the claim, even if code size decreases.
- A failed domain is a valid result if it is reported and diagnosed.
- A raw difference without a `same` result should not be described as a statistically meaningful win.

Eight carefully selected figures and two or three summary tables should be sufficient for the final report.
