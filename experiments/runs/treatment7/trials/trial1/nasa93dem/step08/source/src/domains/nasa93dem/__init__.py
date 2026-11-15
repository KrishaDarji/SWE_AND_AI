"""Adapter: software-project planning and estimation (project stage: build).

Preprocessing note: idX, centerX, YearX and MonthsX end in "X" and are ignored
by the core. Kloc+ is kept as an input, because project size is known when a
proposal is estimated.
"""

from src.core.schema import ROOT, DatasetSpec

SPEC = DatasetSpec(
    name="nasa93dem",
    label="software project estimation",
    client="engineering manager",
    path=ROOT / "moot/optimize/process/nasa93dem.csv",
    objectives=["Effort-", "Defects-"],
    # High versus low process maturity: is one process genuinely less costly?
    contrast=("pmat", "h", "l"),
    requirements={
        1: "Summarize the historical projects and their delivery outcomes.",
        2: "Estimate effort before approving a new project proposal.",
        3: "Tell us whether one process is genuinely less costly than another.",
        4: "Assess project proposals as they arrive during portfolio planning.",
        5: "Explain which project factors are associated with excessive effort.",
        6: "Justify an estimate to a skeptical project sponsor.",
        7: "Choose project settings that maximize delivered value under a fixed budget.",
        8: "Suggest a repair for a proposal likely to exceed its effort budget.",
        9: "Spot a project record that looks inconsistent with the historical data.",
        10: "Compare our estimator against a simple historical-baseline estimator.",
    },
)
