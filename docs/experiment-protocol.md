# Experiment Protocol

## Treatments and repeats

The repository records Treatments 1–7 across the three domains. Each treatment
has three trial directories (`trial1`, `trial2`, and `trial3`) where the
recorded outputs exist. Treatment 7 is the human reuse control.

## Reset protocol for LLM trials

Each trial is treated as an independent run: prompts and outputs are preserved
under its own treatment/trial/domain directory, and later trials are not used
as prompt context. Results are read from the saved run artifacts rather than
from a shared conversational transcript.

## Evidence preserved per step

Each recorded step may contain `prompt.txt`, generated source, tests, runtime
output, response text, and `metrics.csv`. The aggregate builders read these
step-level measurements and retain the source path in the generated result
tables.

## Git-tag coverage

Historical step tags are present in the checkout using the form
`treatment/trial/domain/stepNN`:

- Treatments 1–6: 90 tags each (3 trials × 3 domains × 10 steps).
- Treatment 7: 30 tags (1 human-control ladder × 3 domains × 10 steps).
- Total: 570 step tags.

For example: `t1/trial1/x264/step01` and `t7/x264/step10`.
