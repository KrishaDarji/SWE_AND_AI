"""Step 6: explain a recommendation using evidence from earlier steps."""

from __future__ import annotations

from typing import Dict, Sequence

from .blame import blame
from .datasets import DatasetSpec


def justify(
    spec: DatasetSpec,
    rows: Sequence[Dict[str, str]],
    recommendation: Dict[str, str],
    top_k: int = 3,
) -> Dict[str, object]:
    """Create an auditable explanation for a selected row."""

    findings = blame(spec, rows, top_k=top_k)
    reasons = []
    for finding in findings:
        feature = str(finding["feature"])
        objective = str(finding["objective"])
        value = recommendation.get(feature, "unknown")
        direction = "higher is better" if objective.endswith("+") else "lower is better"
        reasons.append({
            "feature": feature,
            "value": value,
            "objective": objective,
            "direction": direction,
            "effect": finding["effect"],
            "explanation": (
                f"{feature}={value} is one of the strongest observed separators "
                f"for {objective}; {direction}, with an oriented group difference "
                f"of {float(finding['effect']):.3f}."
            ),
        })
    text = (
        f"Recommendation for the {spec.client}: this row was selected using "
        f"the shared ranking machinery. The main evidence is "
        + " ".join(reason["explanation"] for reason in reasons)
    )
    return {"dataset": spec.name, "reasons": reasons, "text": text}
