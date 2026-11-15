"""Step 3 (certify): is the difference between two settings real or noise?"""

from src.core.certify import certify
from src.core.remember import numeric_values


def run(spec, rows):
    feature, a, b = spec.contrast
    group_a = [row for row in rows if row.get(feature) == a]
    group_b = [row for row in rows if row.get(feature) == b]
    result = {"contrast": f"{feature}={a} (n={len(group_a)}) vs {feature}={b} (n={len(group_b)})"}
    for objective in spec.objectives:
        result[objective] = certify(objective, numeric_values(group_a, objective), numeric_values(group_b, objective))
    return result
