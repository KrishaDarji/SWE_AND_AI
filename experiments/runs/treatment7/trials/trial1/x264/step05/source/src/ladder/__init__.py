"""The ten-step ladder. Each step is one small module with run(spec, rows) -> dict."""

from importlib import import_module

SKILLS = ["remember", "guess", "certify", "flow", "blame", "justify", "choose", "fix", "spot", "race"]


def run_step(spec, rows, step: int) -> dict:
    module = import_module(f"src.ladder.step{step:02d}_{SKILLS[step - 1]}")
    return {"dataset": spec.name, "step": step, "skill": SKILLS[step - 1],
            "requirement": spec.requirements.get(step, ""), "result": module.run(spec, rows)}
