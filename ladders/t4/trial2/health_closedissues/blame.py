"""Rank five configuration settings by a common adjusted association score."""
from __future__ import annotations
import csv, math
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.linear_model import Ridge

NUMERIC = ["N_estimators", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
SETTINGS = NUMERIC + ["criterion"]
EXCLUDED = ["MRE-", "ACC+", "PRED40+"]

def _float(value: Any) -> float:
    try:
        value = float(value); return value if math.isfinite(value) else float("nan")
    except (TypeError, ValueError): return float("nan")

def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort"); result = np.empty(len(values), float); i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]: j += 1
        result[order[i:j]] = (i + j - 1) / 2 + 1; i = j
    return result

def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    if len(x) < 2 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2: return None, None
    signed = float(np.corrcoef(_rank(x), _rank(y))[0, 1])
    return (signed, abs(signed)) if math.isfinite(signed) else (None, None)

def _eta2(groups: list[str], y: np.ndarray) -> float | None:
    if len(y) < 2 or len(set(groups)) < 2: return None
    grand = float(np.mean(y)); total = float(np.sum((y - grand) ** 2))
    if total == 0: return 0.0
    return float(sum(np.sum(np.array(groups) == g) * (float(np.mean(y[np.array(groups) == g])) - grand) ** 2 for g in sorted(set(groups))) / total)

def _design(rows: list[dict[str, str]]) -> tuple[np.ndarray, dict[str, list[int]], dict[str, Any]]:
    """Build blocks explicitly so every imputation/indicator column has an owner."""
    blocks: dict[str, list[np.ndarray]] = {}; evidence: dict[str, Any] = {}
    for setting in NUMERIC:
        raw = np.array([_float(r.get(setting, "")) for r in rows], float); missing = ~np.isfinite(raw); finite = raw[~missing]
        fill = float(np.median(finite)) if len(finite) else 0.0
        blocks[setting] = [np.where(missing, fill, raw), missing.astype(float)]
        evidence[setting] = {"missing_or_nonfinite": int(missing.sum()), "imputation": "median of finite values" if len(finite) else "0.0 because all values are missing/non-finite", "block_columns": 2}
    levels = sorted({r.get("criterion", "") or "__MISSING__" for r in rows}); groups = [r.get("criterion", "") or "__MISSING__" for r in rows]
    blocks["criterion"] = [np.array([float(g == level) for g in groups]) for level in levels]
    evidence["criterion"] = {"levels": levels, "block_columns": len(levels)}
    columns: list[np.ndarray] = []; mapping: dict[str, list[int]] = {}
    for setting in SETTINGS:
        mapping[setting] = list(range(len(columns), len(columns) + len(blocks[setting]))); columns.extend(blocks[setting])
    return np.column_stack(columns) if columns else np.empty((len(rows), 0)), mapping, evidence

def _partial_scores(X: np.ndarray, mapping: dict[str, list[int]], y: np.ndarray) -> dict[str, float]:
    all_columns = list(range(X.shape[1]))
    def sse(columns: list[int]) -> float:
        if not columns: return float(np.sum((y - np.mean(y)) ** 2))
        model = Ridge(alpha=1e-8).fit(X[:, columns], y); return float(np.sum((y - model.predict(X[:, columns])) ** 2))
    full = sse(all_columns); scores = {}
    for setting, block in mapping.items():
        reduced_sse = sse([c for c in all_columns if c not in block]); scores[setting] = max(0.0, (reduced_sse - full) / reduced_sse) if reduced_sse else 0.0
    return scores

def analyze(path: str | Path) -> dict[str, Any]:
    with Path(path).open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    valid = [r for r in rows if math.isfinite(_float(r.get("MRE-", "")))]; y = np.array([_float(r["MRE-"]) for r in valid], float); marginal: dict[str, Any] = {}
    for setting in SETTINGS:
        if setting in NUMERIC:
            pairs = [(i, _float(r.get(setting, ""))) for i, r in enumerate(valid)]; pairs = [(i, x) for i, x in pairs if math.isfinite(x)]
            signed, magnitude = _spearman(np.array([x for _, x in pairs]), y[[i for i, _ in pairs]]) if pairs else (None, None)
            marginal[setting] = {"signed_spearman": signed, "magnitude": magnitude, "evidence": {"count": len(pairs), "missing_or_nonfinite": len(valid) - len(pairs), "min": min((x for _, x in pairs), default=None), "max": max((x for _, x in pairs), default=None)}}
        else:
            groups = [r.get(setting, "") or "__MISSING__" for r in valid]; summary = {}
            for level in sorted(set(groups)):
                values = y[np.array(groups) == level]; summary[level] = {"count": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)), "min": float(np.min(values)), "max": float(np.max(values))}
            marginal[setting] = {"eta_squared": _eta2(groups, y), "levels": summary}
    X, mapping, block_evidence = _design(valid); adjusted = _partial_scores(X, mapping, y); ranking = sorted(SETTINGS, key=lambda s: (-adjusted[s], s))
    return {"outcome": "MRE-", "direction": "lower is better", "row_count": len(rows), "finite_mre_count": len(valid), "outcome_summary": {"min": float(np.min(y)) if len(y) else None, "mean": float(np.mean(y)) if len(y) else None, "median": float(np.median(y)) if len(y) else None, "max": float(np.max(y)) if len(y) else None}, "excluded_outcomes": EXCLUDED, "numeric_grouping": "none; numeric settings remain continuous", "marginal_descriptive_evidence": marginal, "transformed_blocks": mapping, "block_evidence": block_evidence, "adjusted_partial_r2": adjusted, "primary_score": "adjusted_partial_r2; complete transformed block removed", "ranking": ranking, "interpretation": "Primary ranking uses one common adjusted partial-R2 score. Spearman and eta-squared are descriptive evidence only. Correlated settings can share explanatory information. Missing/non-finite numeric values are omitted from Spearman but retained through imputation and an owned missing indicator in adjusted scoring; all-missing numeric settings use 0.0 and an all-one indicator. Ties use setting-name order. Associations are not causal responsibility or guaranteed gains."}

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(); parser.add_argument("dataset"); args = parser.parse_args(); print(json.dumps(analyze(args.dataset), indent=2, sort_keys=True))
