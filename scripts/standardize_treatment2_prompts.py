"""Write complete canonical prompts for every existing Treatment 2 step."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / "experiments/runs/treatment2/trials"
requirements = {
    "x264": [("remember", "Summarize the encoder settings and quality measures in our historical runs."), ("guess", "Predict encoding time before running a new configuration."), ("certify", "Tell us whether a quality improvement is statistically credible."), ("flow", "Evaluate configurations as benchmark results arrive one at a time."), ("blame", "Explain which settings are responsible for slow encodes."), ("justify", "Justify a recommended configuration to a platform administrator."), ("choose", "Select the best configuration under a quality constraint."), ("fix", "Suggest the smallest setting change that improves a slow configuration."), ("spot", "Detect an anomalous benchmark result or possible measurement error."), ("race", "Compare the system against a baseline search strategy.")],
    "nasa93dem": [("remember", "Summarize the historical projects and their delivery outcomes."), ("guess", "Estimate effort before approving a new project proposal."), ("certify", "Tell us whether one process is genuinely less costly than another."), ("flow", "Assess project proposals as they arrive during portfolio planning."), ("blame", "Explain which project factors are associated with excessive effort."), ("justify", "Justify an estimate to a skeptical project sponsor."), ("choose", "Choose project settings that maximize delivered value under a fixed budget."), ("fix", "Recommend the cheapest process change that reduces estimated risk."), ("spot", "Spot a project record that looks inconsistent with the historical data."), ("race", "Compare our estimator against a simple historical-baseline estimator.")],
    "health_closedissues": [("remember", "Summarize model-quality metrics and configuration settings."), ("guess", "Predict model error and accuracy before a full evaluation."), ("certify", "Tell us whether one configuration family is reliably better than another."), ("flow", "Evaluate configurations as validation results arrive one at a time."), ("blame", "Explain which hyperparameters are associated with high model error."), ("justify", "Justify a recommended model configuration to the platform team."), ("choose", "Choose a configuration that balances accuracy and prediction quality."), ("fix", "Suggest the smallest hyperparameter change that improves model error."), ("spot", "Spot an unusual model-quality result or possible measurement error."), ("race", "Compare our configuration selector against a historical baseline.")],
}
info = {"x264": ("video encoding", "X264_AllMeasurements.csv", "moot/optimize/config/X264_AllMeasurements.csv", 1), "nasa93dem": ("software project planning", "nasa93dem.csv", "moot/optimize/process/nasa93dem.csv", 2), "health_closedissues": ("machine-learning model configuration", "Health-ClosedIssues0000.csv", "moot/optimize/hpo/Health-ClosedIssues0000.csv", 3)}
count = 0
for prompt_path in sorted(base.glob("trial*/**/prompt.txt")):
    trial, domain, step_name, _filename = prompt_path.relative_to(base).parts
    step = int(step_name[4:])
    trial_number = int(trial[5:])
    trial_label = f"Trial {trial_number}"
    skill, requirement = requirements[domain][step - 1]
    domain_name, dataset, dataset_path, dataset_number = info[domain]
    project = f"/Users/RadhaKrishna/Desktop/SWE_AND_AI/experiments/runs/treatment2/trials/{trial}/{domain}"
    source = "\n".join(f"{project}/step{i:02d}/source/" for i in range(1, step))
    tests = "\n".join(f"{project}/step{i:02d}/tests/" for i in range(1, step))
    if step == 1:
        reuse = "This is Step 1 for this dataset/trial. Build a reusable foundation for later skills."
        prior = ""
    else:
        reuse = f"Reuse and extend all previous Treatment 2 {trial_label}, Dataset {dataset_number} implementations."
        prior = f"\nRead the previous source code from:\n\n{source}\n\nRead the previous tests from:\n\n{tests}\n"
    text = f"""You are running Treatment 2, {trial_label}, Dataset {dataset_number} of a controlled experiment.

Treatment:
- Writer: LLM
- Background: plain Python only
- Reuse instructed: Yes
- Domain: {domain_name}
- Dataset: {dataset}
- Dataset path: /Users/RadhaKrishna/Desktop/SWE_AND_AI/{dataset_path}
- Current skill: {skill}
- Current requirement: {requirement}

Project directory:
{project}

Use only the Python standard library.

{reuse}
Do not create an unrelated standalone implementation.
Keep all previous behavior working while adding Step {step}.
{prior}
Return:
1. Complete updated source code.
2. Tests for Step {step} and existing Step 1–{step - 1} behavior.
3. One command to run the implementation.
4. A short explanation of what was reused.
5. Known assumptions and limitations.

Save all outputs in exactly:

{project}/{step_name}/

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

For Treatment 2, new_loc must measure only the code added beyond the reused previous implementation.

Run the tests and implementation, capture outputs and runtime, and record actual measured results. Do not fabricate results.

Do not save files in the parent {domain} directory.
Do not copy code from other treatments, other trials, other datasets, or previous chats.
Reuse must come only from the Treatment 2 files in the paths above.
"""
    prompt_path.write_text(text, encoding="utf-8")
    count += 1
print(f"Standardized {count} Treatment 2 prompt files.")
