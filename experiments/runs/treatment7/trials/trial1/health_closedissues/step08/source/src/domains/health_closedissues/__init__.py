"""Adapter: open-source project health, closed-issue models (project stage: maintain)."""

from src.core.schema import ROOT, DatasetSpec

SPEC = DatasetSpec(
    name="health_closedissues",
    label="open-source project health",
    client="open-source project maintainer",
    path=ROOT / "moot/optimize/hpo/Health-ClosedIssues0000.csv",
    objectives=["MRE-", "ACC+", "PRED40+"],
    # Two split criteria: is the apparent quality difference real or noise?
    contrast=("criterion", "squared_error", "absolute_error"),
    requirements={
        1: "Summarize model-quality metrics and configuration settings.",
        2: "Predict model error and accuracy before a full evaluation.",
        3: "Tell us whether an apparent quality improvement is meaningful or noise.",
        4: "Evaluate configurations as validation results arrive one at a time.",
        5: "Explain which hyperparameters are associated with high model error.",
        6: "Justify a recommended configuration to the platform team.",
        7: "Choose a configuration that minimizes error under a compute budget.",
        8: "Suggest the smallest hyperparameter change that improves error.",
        9: "Spot an unusual model-quality result or possible measurement error.",
        10: "Compare our configuration selector against a historical baseline.",
    },
)
