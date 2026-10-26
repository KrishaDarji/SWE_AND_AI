"""Preserve saved prompts and write complete canonical Treatment 1 prompts."""

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / "experiments/runs/treatment1/trials"

requirements = {
    "x264": [
        ("remember", "Summarize the encoder settings and quality measures in our historical runs."),
        ("guess", "Predict encoding time before running a new configuration."),
        ("certify", "Tell us whether a quality improvement is statistically credible."),
        ("flow", "Evaluate configurations as benchmark results arrive one at a time."),
        ("blame", "Explain which settings are responsible for slow encodes."),
        ("justify", "Justify a recommended configuration to a platform administrator."),
        ("choose", "Select the best configuration under a quality constraint."),
        ("fix", "Suggest the smallest setting change that improves a slow configuration."),
        ("spot", "Detect an anomalous benchmark result or possible measurement error."),
        ("race", "Compare the system against a baseline search strategy."),
    ],
    "nasa93dem": [
        ("remember", "Summarize the historical projects and their delivery outcomes."),
        ("guess", "Estimate effort before approving a new project proposal."),
        ("certify", "Tell us whether one process is genuinely less costly than another."),
        ("flow", "Assess project proposals as they arrive during portfolio planning."),
        ("blame", "Explain which project factors are associated with excessive effort."),
        ("justify", "Justify an estimate to a skeptical project sponsor."),
        ("choose", "Choose project settings that maximize delivered value under a fixed budget."),
        ("fix", "Recommend the cheapest process change that reduces estimated risk."),
        ("spot", "Spot a project record that looks inconsistent with the historical data."),
        ("race", "Compare our estimator against a simple historical-baseline estimator."),
    ],
    "health_closedissues": [
        ("remember", "Summarize model-quality metrics and configuration settings."),
        ("guess", "Predict model error and accuracy before a full evaluation."),
        ("certify", "Tell us whether one configuration family is reliably better than another."),
        ("flow", "Evaluate configurations as validation results arrive one at a time."),
        ("blame", "Explain which hyperparameters are associated with high model error."),
        ("justify", "Justify a recommended model configuration to the platform team."),
        ("choose", "Choose a configuration that balances accuracy and prediction quality."),
        ("fix", "Suggest the smallest hyperparameter change that improves model error."),
        ("spot", "Spot an unusual model-quality result or possible measurement error."),
        ("race", "Compare our configuration selector against a historical baseline."),
    ],
}

dataset_info = {
    "x264": ("video encoding", "X264_AllMeasurements.csv", "moot/optimize/config/X264_AllMeasurements.csv"),
    "nasa93dem": ("software project planning", "nasa93dem.csv", "moot/optimize/process/nasa93dem.csv"),
    "health_closedissues": ("machine-learning model configuration", "Health-ClosedIssues0000.csv", "moot/optimize/hpo/Health-ClosedIssues0000.csv"),
}

for prompt_path in sorted(base.glob("trial*/**/prompt.txt")):
    parts = prompt_path.relative_to(base).parts
    trial, domain, step_name = parts[0], parts[1], parts[2]
    step = int(step_name.replace("step", ""))
    skill, requirement = requirements[domain][step - 1]
    trial_number = int(trial.replace("trial", ""))
    domain_name, dataset, dataset_path = dataset_info[domain]
    canonical = f"""You are running Treatment 1, Trial {trial_number}, Dataset {3 if domain == 'health_closedissues' else 2 if domain == 'nasa93dem' else 1} of a controlled experiment.

Treatment:
- Writer: LLM
- Background: plain Python only
- Reuse instructed: No
- Domain: {domain_name}
- Dataset: {dataset}
- Dataset path: {dataset_path}
- Current skill: {skill}
- Current requirement: {requirement}

Use only the Python standard library.

Implement this skill independently. Do not reuse or assume any previous skill,
architecture, generated code, earlier trial output, or previous conversation.

Return:
1. Complete source code.
2. Tests.
3. One command to run the implementation.
4. A short explanation of the approach.
5. Known assumptions and limitations.

Save all outputs in exactly:
experiments/runs/treatment1/trials/{trial}/{domain}/{step_name}/

The directory must contain:
- prompt.txt
- response.txt
- source/
- tests/
- test_output.txt
- runtime_output.txt
- metrics.csv

Record these metrics in metrics.csv:
step
treatment
trial
domain
skill
total_loc
new_loc
test_pass
runtime_seconds
error_count
notes

For Treatment 1, new_loc should normally be close to total_loc because reuse is
not instructed.

Run the tests and implementation, capture outputs and runtime, and record
actual measured metrics. Do not fabricate results.

Do not save files in the parent {domain} directory.
Do not copy code or context from other trials, other steps, or previous chats.
"""
    original = prompt_path.with_name("prompt_original.txt")
    if not original.exists():
        shutil.copy2(prompt_path, original)
    prompt_path.write_text(canonical, encoding="utf-8")

print("Standardized all existing Treatment 1 prompt files; originals preserved as prompt_original.txt")
