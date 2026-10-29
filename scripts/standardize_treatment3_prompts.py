"""Write complete canonical prompts for every Treatment 3 trial/dataset/step."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / "experiments/runs/treatment3/trials"

requirements = {
    "x264": [
        ("remember", "Summarize encoder settings and quality measures in historical X264 runs."),
        ("guess", "Predict encoding time before running a new x264 configuration."),
        ("certify", "Determine whether a quality improvement is statistically credible."),
        ("flow", "Evaluate configurations as benchmark results arrive one at a time."),
        ("blame", "Explain which settings are associated with slow encodes."),
        ("justify", "Recommend and justify an x264 configuration to a platform administrator."),
        ("choose", "Select the best x264 configuration under an explicit quality constraint."),
        ("fix", "Suggest the smallest setting change that improves a slow configuration."),
        ("spot", "Detect an anomalous benchmark result or possible measurement error."),
        ("race", "Compare a configuration selector against a simple baseline search strategy."),
    ],
    "nasa93dem": [
        ("remember", "Summarize historical software projects and delivery outcomes."),
        ("guess", "Estimate effort before approving a new software-project proposal."),
        ("certify", "Determine whether one process group genuinely has lower effort."),
        ("flow", "Assess project proposals as they arrive during portfolio planning."),
        ("blame", "Explain which project factors are associated with excessive effort."),
        ("justify", "Justify an effort estimate to a skeptical project sponsor."),
        ("choose", "Choose project settings that maximize delivered value under a fixed budget."),
        ("fix", "Recommend the cheapest process change that reduces estimated risk."),
        ("spot", "Spot a project record that looks inconsistent with historical data."),
        ("race", "Compare an effort estimator against a simple historical baseline."),
    ],
    "health_closedissues": [
        ("remember", "Summarize model-quality metrics and configuration settings."),
        ("guess", "Predict model error and accuracy before a full evaluation."),
        ("certify", "Determine whether one configuration family is reliably better."),
        ("flow", "Evaluate configurations as validation results arrive one at a time."),
        ("blame", "Explain which hyperparameters are associated with high model error."),
        ("justify", "Recommend and justify a model configuration to the platform team."),
        ("choose", "Choose a configuration that balances accuracy and prediction quality."),
        ("fix", "Suggest the smallest hyperparameter change that improves model error."),
        ("spot", "Detect an unusual model-quality result or possible measurement error."),
        ("race", "Compare a configuration selector against a simple historical baseline."),
    ],
}

info = {
    "x264": ("video encoding", "X264_AllMeasurements.csv", "moot/optimize/config/X264_AllMeasurements.csv", 1),
    "nasa93dem": ("software project planning", "nasa93dem.csv", "moot/optimize/process/nasa93dem.csv", 2),
    "health_closedissues": ("machine-learning model configuration", "Health-ClosedIssues0000.csv", "moot/optimize/hpo/Health-ClosedIssues0000.csv", 3),
}

method_notes = {
    "remember": "Produce descriptive summaries and relationships; do not claim causality.",
    "guess": "Use a reproducible train/test protocol, appropriate metrics, and leakage checks.",
    "certify": "Report group sizes, effect size, bootstrap confidence interval, and one-sided permutation p-value.",
    "flow": "Process records sequentially, maintain state, and handle ties and invalid rows deterministically.",
    "blame": "Rank influential factors while explicitly separating association from causation.",
    "justify": "Include evidence, uncertainty, trade-offs, risks, assumptions, and a concise stakeholder report.",
    "choose": "Define the objective and constraint explicitly; handle infeasible candidates.",
    "fix": "Compare a baseline with minimal valid changes and report gains or no-improvement results.",
    "spot": "Use a reproducible anomaly method and distinguish data errors from legitimate extremes.",
    "race": "Use the same reproducible evaluation protocol for the Treatment 3 method and baseline.",
}

count = 0
for trial_number in range(1, 4):
    for domain, steps in requirements.items():
        domain_name, dataset, dataset_path, dataset_number = info[domain]
        for step, (skill, objective) in enumerate(steps, 1):
            step_name = f"step{step:02d}"
            directory = base / f"trial{trial_number}" / domain / step_name
            directory.mkdir(parents=True, exist_ok=True)
            text = f"""You are running Treatment 3, Trial {trial_number}, Dataset {dataset_number}, Step {step} of a controlled experiment.

Treatment 3 requirements:
- Writer: LLM
- Background: Python with scikit-learn
- Reuse instructed: No
- Domain: {domain_name}
- Dataset: {dataset}
- Dataset path: {dataset_path}
- Current skill: {skill}

Implement the following independently:
{objective}

Use Python, scikit-learn, and the Python standard library only. Do not reuse
code, architecture, tests, generated output, or reasoning from Treatment 1,
Treatment 2, another treatment, another trial, another dataset, another step,
or a previous conversation. Build a self-contained implementation for this
step. {method_notes[skill]}

The implementation must load and validate the dataset, handle malformed or
missing values explicitly, use a reproducible random seed where randomness is
needed, and report assumptions, limitations, uncertainty, and possible bias.

Return:
1. Complete source code.
2. Tests covering normal behavior and important edge cases.
3. One command to run the implementation.
4. A concise explanation of the approach.
5. Actual measured outputs and limitations.

Save every artifact only inside this exact directory:
experiments/runs/treatment3/trials/trial{trial_number}/{domain}/{step_name}/

Do not save files directly in the parent {domain}/ directory or in any other
treatment, trial, dataset, or step. The step directory must contain exactly
these required artifacts:
- prompt.txt
- response.txt
- source/
- tests/
- test_output.txt
- runtime_output.txt
- metrics.csv

Run the tests and implementation, capture their actual output, and record
actual measured results. Do not fabricate metrics. Record these CSV columns:
step,treatment,trial,domain,skill,total_loc,new_loc,test_pass,
runtime_seconds,error_count,notes
"""
            (directory / "prompt.txt").write_text(text, encoding="utf-8")
            count += 1

print(f"Standardized {count} Treatment 3 prompt files.")
