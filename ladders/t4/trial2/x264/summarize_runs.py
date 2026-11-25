#!/usr/bin/env python3
"""Summarize x264 encoder settings and measured performance."""
from __future__ import annotations

import argparse
import csv
import json
import random
from time import perf_counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy import stats


class IncrementalFlow:
    """Online statistics for an ordered stream of encoder measurements."""

    def __init__(self, settings: list[str]):
        self.settings = list(settings)
        self.count = 0
        self.mean = 0.0
        self._m2 = 0.0
        self.best: dict | None = None

    def update(self, row: dict[str, float | int]) -> dict:
        if "Performance-" not in row:
            raise ValueError("missing required measure column: Performance-")
        value = float(row["Performance-"])
        configuration = {name: float(row[name]) for name in self.settings}
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self._m2 += delta * (value - self.mean)
        if self.best is None or value < self.best["value"]:
            self.best = {"value": value, "settings": configuration}
        return self.snapshot()

    def snapshot(self) -> dict:
        return {"count": self.count, "mean": self.mean if self.count else None,
                "sample_variance": (self._m2 / (self.count - 1)
                                     if self.count > 1 else None),
                "best": self.best}


def flow(path: str | Path) -> list[dict]:
    """Process CSV rows in file order and return the prefix snapshots."""
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        settings = [name for name in (reader.fieldnames or []) if name != "Performance-"]
        if "Performance-" not in (reader.fieldnames or []):
            raise ValueError("missing required measure column: Performance-")
        state = IncrementalFlow(settings)
        return [state.update(row) for row in reader]


def load_dataset(path: str | Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset is empty")
    measure = "Performance-"
    if measure not in rows[0]:
        raise ValueError(f"missing required measure column: {measure}")
    settings = [name for name in rows[0] if name != measure]
    x = np.asarray([[float(row[name].strip()) for name in settings] for row in rows])
    y = np.asarray([float(row[measure].strip()) for row in rows])
    return settings, x, y


def summarize(path: str | Path) -> dict:
    settings, x, y = load_dataset(path)
    scaler = StandardScaler()
    model = LinearRegression().fit(scaler.fit_transform(x), y)
    effects = sorted(
        ({"setting": name, "mean_when_on": float(y[x[:, i] == 1].mean()) if np.any(x[:, i] == 1) else None,
          "mean_when_off": float(y[x[:, i] == 0].mean()) if np.any(x[:, i] == 0) else None,
          "difference_on_minus_off": float(y[x[:, i] == 1].mean() - y[x[:, i] == 0].mean()) if np.any(x[:, i] == 0) and np.any(x[:, i] == 1) else None,
          "standardized_model_coefficient": float(model.coef_[i])}
         for i, name in enumerate(settings)),
        key=lambda item: abs(item["difference_on_minus_off"] or 0), reverse=True)
    best = int(np.argmin(y))
    worst = int(np.argmax(y))
    return {
        "dataset": str(path), "runs": int(len(y)), "setting_count": len(settings),
        "measure": "Performance-", "measure_summary": {
            "mean": float(y.mean()), "std": float(y.std()), "min": float(y.min()),
            "max": float(y.max()), "median": float(np.median(y))},
        "setting_on_counts": {name: int(x[:, i].sum()) for i, name in enumerate(settings)},
        "best_run": {"row": best + 2, "value": float(y[best]),
                     "settings_on": [name for i, name in enumerate(settings) if x[best, i] == 1]},
        "worst_run": {"row": worst + 2, "value": float(y[worst]),
                      "settings_on": [name for i, name in enumerate(settings) if x[worst, i] == 1]},
        "setting_effects": effects,
        "model_r2": float(model.score(scaler.transform(x), y)),
    }


def blame(path: str | Path) -> dict:
    """Rank settings by marginal and adjusted association with Performance-.

    The marginal score is eta-squared (between-level sum of squares / total
    sum of squares).  The adjusted score is the increase in training R-squared
    from adding one setting's one-hot columns after all other settings.  Both
    are descriptive associations; lower Performance- is better.
    """
    settings, x, y = load_dataset(path)
    overall = float(y.mean())
    total_ss = float(np.sum((y - overall) ** 2))
    marginal = []
    encoded = []
    levels_by_setting = {}
    for i, name in enumerate(settings):
        levels = np.unique(x[:, i])
        levels_by_setting[name] = [float(v) for v in levels]
        groups = []
        between = 0.0
        for level in levels:
            values = y[x[:, i] == level]
            groups.append({"value": float(level), "n": int(len(values)),
                           "mean": float(values.mean()), "min": float(values.min()),
                           "max": float(values.max())})
            between += len(values) * float((values.mean() - overall) ** 2)
        eta2 = between / total_ss if total_ss else 0.0
        marginal.append({"setting": name, "levels": groups, "level_count": len(levels),
                         "constant": len(levels) == 1, "eta_squared": eta2,
                         "mean_min": min(g["mean"] for g in groups),
                         "mean_max": max(g["mean"] for g in groups),
                         "mean_range": max(g["mean"] for g in groups) - min(g["mean"] for g in groups)})
        if len(levels) > 1:
            encoded.append(np.column_stack([(x[:, i] == level).astype(float) for level in levels[1:]]))
        else:
            encoded.append(np.empty((len(y), 0)))
    full_parts = [part for part in encoded if part.shape[1]]
    full_x = np.column_stack(full_parts) if full_parts else np.empty((len(y), 0))
    full_r2 = LinearRegression().fit(full_x, y).score(full_x, y) if full_x.shape[1] else 0.0
    for item, part in zip(marginal, encoded):
        other = [p for p, candidate in zip(encoded, marginal) if candidate["setting"] != item["setting"] and p.shape[1]]
        other_x = np.column_stack(other) if other else np.empty((len(y), 0))
        reduced_r2 = LinearRegression().fit(other_x, y).score(other_x, y) if other_x.shape[1] else 0.0
        item["adjusted_delta_r2"] = max(0.0, float(full_r2 - reduced_r2))
        item["interpretation"] = "constant: no within-dataset association" if item["constant"] else "marginal and adjusted association"
    complements = []
    for i, left in enumerate(settings):
        for right in settings[i + 1:]:
            if len(np.unique(x[:, i])) == 2 and np.array_equal(x[:, i] + x[:, settings.index(right)], np.ones(len(y))):
                complements.append([left, right])
    marginal.sort(key=lambda item: (-item["adjusted_delta_r2"], -item["eta_squared"], item["setting"]))
    return {"measure": "Performance-", "lower_is_better": True, "runs": len(y),
            "outcome_summary": {"mean": overall, "std": float(y.std()), "min": float(y.min()), "max": float(y.max()), "median": float(np.median(y))},
            "ranking_method": "descending adjusted delta R-squared, then eta-squared, then setting name; eta-squared is marginal group association and adjusted delta R-squared is the full one-hot linear model increase after all other settings",
            "settings": marginal, "mutually_exclusive_pairs": complements,
            "full_model_r2": float(full_r2), "assumptions": ["configuration rows are observational", "associations are not causal responsibility", "one-hot encoding uses the first sorted level as reference"]}


def fit_predictor(x: np.ndarray, y: np.ndarray) -> Pipeline:
    """Fit the reusable preprocessing/model pipeline on supplied training data."""
    return Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]).fit(x, y)


def evaluate(path: str | Path, test_size: float = 0.2, random_state: int = 42) -> dict:
    """Evaluate with a reproducible grouped split; identical configurations stay together."""
    settings, x, y = load_dataset(path)
    groups = np.asarray([",".join(map(str, row)) for row in x])
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(x, y, groups))
    model = fit_predictor(x[train_idx], y[train_idx])
    predictions = model.predict(x[test_idx])
    baseline = np.full(len(test_idx), y[train_idx].mean())
    return {"settings": settings, "train_rows": len(train_idx), "test_rows": len(test_idx),
            "train_groups": len(set(groups[train_idx])), "test_groups": len(set(groups[test_idx])),
            "mae": float(mean_absolute_error(y[test_idx], predictions)),
            "baseline_mae": float(mean_absolute_error(y[test_idx], baseline)),
            "random_state": random_state}


def predict(path: str | Path, configuration: dict[str, float | int]) -> float:
    """Train on all available measurements and predict Performance- for one configuration."""
    settings, x, y = load_dataset(path)
    missing = [name for name in settings if name not in configuration]
    extra = [name for name in configuration if name not in settings]
    if missing or extra:
        raise ValueError(f"configuration keys mismatch; missing={missing}, extra={extra}")
    model = fit_predictor(x, y)
    return float(model.predict(np.asarray([[float(configuration[name]) for name in settings]]))[0])


def certify(path: str | Path, practical_threshold: float = 5.0,
            permutations: int = 10000, random_state: int = 42) -> dict:
    """Compare predeclared no_asm groups; lower Performance- is better.

    The groups are defined before outcomes are inspected: group A has
    no_8x8dct=0 and group B has no_8x8dct=1. Rows are configurations, not
    repeated runs.
    """
    settings, x, y = load_dataset(path)
    if "no_8x8dct" not in settings:
        raise ValueError("required comparison setting missing: no_8x8dct")
    column = settings.index("no_8x8dct")
    group_a = y[x[:, column] == 0]
    group_b = y[x[:, column] == 1]
    if len(group_a) < 2 or len(group_b) < 2:
        raise ValueError("each comparison group needs at least two rows")
    observed = float(group_a.mean() - group_b.mean())
    test = stats.ttest_ind(group_a, group_b, equal_var=False)
    variance_a = group_a.var(ddof=1) / len(group_a)
    variance_b = group_b.var(ddof=1) / len(group_b)
    standard_error = np.sqrt(variance_a + variance_b)
    welch_df = (variance_a + variance_b) ** 2 / (
        variance_a ** 2 / (len(group_a) - 1) + variance_b ** 2 / (len(group_b) - 1))
    interval = stats.t.interval(0.95, welch_df, loc=observed, scale=standard_error)
    rng = np.random.default_rng(random_state)
    pooled = np.concatenate([group_a, group_b])
    exceed = 0
    for _ in range(permutations):
        shuffled = rng.permutation(pooled)
        if abs(shuffled[:len(group_a)].mean() - shuffled[len(group_a):].mean()) >= abs(observed):
            exceed += 1
    permutation_p = float((exceed + 1) / (permutations + 1))
    statistically_detected = bool(test.pvalue < 0.05)
    point_exceeds_threshold = bool(abs(observed) >= practical_threshold)
    interval_beyond_threshold = bool(interval[0] > practical_threshold or
                                     interval[1] < -practical_threshold)
    return {
        "measure": "Performance-", "lower_is_better": True,
        "group_definition": {"A": "no_8x8dct=0", "B": "no_8x8dct=1"},
        "group_a_n": int(len(group_a)), "group_b_n": int(len(group_b)),
        "group_a_mean": float(group_a.mean()), "group_b_mean": float(group_b.mean()),
        "observed_difference_a_minus_b": observed,
        "difference_ci_95": [float(interval[0]), float(interval[1])],
        "welch_satterthwaite_df": float(welch_df),
        "welch_t": float(test.statistic), "welch_p_value": float(test.pvalue),
        "permutation_p_value": permutation_p,
        "permutations": permutations, "random_state": random_state,
        "practical_threshold": float(practical_threshold),
        "practical_threshold_label": "illustrative default" if practical_threshold == 5.0 else "user-supplied",
        "statistically_detected": statistically_detected,
        "difference_from_zero_statistically_detected": statistically_detected,
        "point_estimate_exceeds_threshold": point_exceeds_threshold,
        "confidence_interval_entirely_beyond_threshold": interval_beyond_threshold,
        "practical_importance": ("supported by interval" if interval_beyond_threshold else
                                  "uncertain: interval crosses practical threshold" if point_exceeds_threshold else
                                  "not shown by point estimate"),
        "conclusion": ("statistical difference detected; practical importance supported" if interval_beyond_threshold else
                       "statistical difference detected; point estimate exceeds threshold but practical importance is uncertain" if statistically_detected and point_exceeds_threshold else
                       "statistical difference detected, but point estimate does not exceed threshold" if statistically_detected else
                       "no statistical difference detected; this does not establish equivalence"),
        "assumptions": ["independent configuration rows", "Welch unequal-variance t test",
                        "two-sided alpha=0.05", "observational association, not causation",
                        "non-detection is not equivalence"]}


def justify(path: str | Path) -> dict:
    """Produce a retrospective, evidence-labeled recommendation for an administrator.

    The recommendation is the lowest observed Performance-; ``min`` over the
    input order makes ties resolve to the earliest CSV record.  The first CSV
    record is retained as a clearly named historical reference.
    """
    settings, x, y = load_dataset(path)
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    best_index = int(np.argmin(y))
    reference_index = 0
    model = fit_predictor(x, y)

    def record(index: int) -> dict:
        return {"record_id": f"csv_record_{index + 1}", "csv_row": index + 2,
                "configuration": {name: float(x[index, i]) for i, name in enumerate(settings)},
                "measured_performance": float(y[index]),
                "model_estimate": float(model.predict(x[index:index + 1])[0]),
                "raw_fields": dict(rows[index])}

    recommendation = record(best_index)
    reference = record(reference_index)
    return {
        "measure": "Performance-", "lower_is_better": True,
        "recommendation_basis": "lowest observed Performance-, ties resolved by earliest CSV record",
        "recommendation": recommendation,
        "historical_reference": {"label": "first CSV record", **reference},
        "measured_comparison": {"recommendation_minus_reference": recommendation["measured_performance"] - reference["measured_performance"],
                                "reference_minus_recommendation": reference["measured_performance"] - recommendation["measured_performance"]},
        "association_evidence": blame(path),
        "evidence_notes": [
            "Measured results are the observed CSV values.",
            "Model estimates are in-sample linear-model estimates and are not measured results.",
            "Associations do not establish the effect of changing a setting.",
            "Correlated settings can share explanatory information.",
            "This is retrospective evidence from this dataset, not a guarantee of future performance or statistical superiority."
        ]
    }


class CountedEvaluator:
    """Expose outcomes only through a finite, duplicate-free evaluation API."""
    def __init__(self, outcomes: dict[tuple[float, ...], float], settings: list[str], budget: int):
        if not isinstance(budget, int) or budget < 1:
            raise ValueError("budget must be a positive integer")
        self._outcomes, self.settings, self.budget = outcomes, list(settings), budget
        self.evaluations: list[dict] = []
        self._seen: set[tuple[float, ...]] = set()

    def evaluate(self, configuration: dict[str, float | int]) -> float:
        key = tuple(float(configuration[name]) for name in self.settings)
        if key in self._seen:
            raise ValueError("duplicate evaluation")
        if len(self.evaluations) >= self.budget:
            raise RuntimeError("evaluation budget exhausted")
        if key not in self._outcomes:
            raise KeyError("unknown configuration")
        value = float(self._outcomes[key])
        self._seen.add(key)
        self.evaluations.append({"configuration": dict(configuration), "Performance-": value})
        return value


def select_budget(candidates: list[dict[str, float]], evaluator: CountedEvaluator,
                  budget: int = 20, random_state: int = 42) -> dict:
    """Randomly sample unique candidates, then recommend the best measured one."""
    if not isinstance(budget, int) or budget < 1:
        raise ValueError("budget must be a positive integer")
    if not candidates:
        raise ValueError("no candidates available")
    count = min(budget, len(candidates))
    order = random.Random(random_state).sample(list(range(len(candidates))), count)
    observations = []
    for index in order:
        observations.append({"configuration": dict(candidates[index]),
                             "Performance-": float(evaluator.evaluate(candidates[index]))})
    best = min(enumerate(observations), key=lambda item: (item[1]["Performance-"], item[0]))
    return {"strategy": "seeded random unique sample; lowest measured Performance-, earliest evaluation breaks ties",
            "random_state": random_state, "requested_budget": budget,
            "budget_used": len(observations), "candidate_count": len(candidates),
            "evaluation_order": order, "observations": observations,
            "recommendation": observations[best[0]],
            "recommendation_frozen": True,
            "limitations": ["Only evaluated candidates can be recommended.", "Random sampling may miss the global minimum."]}


def load_candidates(path: str | Path) -> tuple[list[str], list[dict[str, float]]]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        settings = [name for name in (reader.fieldnames or []) if name != "Performance-"]
        return settings, [{name: float(row[name]) for name in settings} for row in reader]


def budget_select_dataset(path: str | Path, budget: int = 20, random_state: int = 42) -> dict:
    settings, candidates = load_candidates(path)
    _, x, y = load_dataset(path)
    outcomes = {tuple(map(float, row)): float(value) for row, value in zip(x, y)}
    evaluator = CountedEvaluator(outcomes, settings, budget)
    result = select_budget(candidates, evaluator, budget, random_state)
    global_index = int(np.argmin(y))
    result["retrospective_audit"] = {"observed_global_minimum": float(y[global_index]),
        "recommendation_gap": float(result["recommendation"]["Performance-"] - y[global_index]),
        "audit_influenced_selection": False}
    return result


def repair(path: str | Path, baseline: dict[str, float | int] | None = None) -> dict:
    """Select the smallest observed improvement over an observed baseline.

    This is retrospective observed-data repair, not a causal or deployment
    guarantee. Raw setting-column differences are a proxy for conceptual
    change count because correlated or one-hot fields can encode one change.
    """
    settings, x, y = load_dataset(path)
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if baseline is None:
        baseline_index = 0
        baseline_configuration = {name: float(x[0, i]) for i, name in enumerate(settings)}
    else:
        missing = [name for name in settings if name not in baseline]
        extra = [name for name in baseline if name not in settings]
        if missing or extra:
            raise ValueError(f"baseline keys mismatch; missing={missing}, extra={extra}")
        try:
            baseline_configuration = {name: float(baseline[name]) for name in settings}
        except (TypeError, ValueError) as exc:
            raise ValueError("baseline values must be numeric") from exc
        matches = np.all(x == np.asarray([baseline_configuration[name] for name in settings]), axis=1)
        if not np.any(matches):
            raise ValueError("baseline configuration was not observed in the dataset")
        baseline_index = int(np.flatnonzero(matches)[0])
    improving = np.flatnonzero(y < y[baseline_index])
    if not len(improving):
        return {"status": "no_repair", "measure": "Performance-", "lower_is_better": True,
                "baseline": {"record_id": f"csv_record_{baseline_index + 1}", "csv_row": baseline_index + 2,
                             "configuration": baseline_configuration, "measured_performance": float(y[baseline_index])},
                "reason": "no observed configuration has strictly lower Performance-",
                "change_count_definition": "number of differing raw encoder-setting columns",
                "change_count_proxy_note": "Raw-column change count is a proxy: correlated or one-hot fields may represent fewer conceptual changes."}
    def cost(index: int) -> int:
        return int(np.count_nonzero(x[index] != x[baseline_index]))
    chosen = int(min(improving, key=lambda index: (cost(index), y[index], index)))
    changed = [{"setting": name, "from": float(x[baseline_index, i]), "to": float(x[chosen, i])}
               for i, name in enumerate(settings) if x[baseline_index, i] != x[chosen, i]]
    return {"status": "repaired", "measure": "Performance-", "lower_is_better": True,
            "baseline": {"record_id": f"csv_record_{baseline_index + 1}", "csv_row": baseline_index + 2,
                         "configuration": baseline_configuration, "measured_performance": float(y[baseline_index])},
            "repaired": {"record_id": f"csv_record_{chosen + 1}", "csv_row": chosen + 2,
                         "configuration": {name: float(x[chosen, i]) for i, name in enumerate(settings)},
                         "measured_performance": float(y[chosen])},
            "changed_settings": changed, "change_cost": int(cost(chosen)),
            "observed_improvement": float(y[baseline_index] - y[chosen]),
            "selection_rule": "strict improvement; minimum raw-column change cost, then Performance-, then CSV record order",
            "change_count_proxy_note": "Raw-column change count is a proxy: correlated or one-hot fields may represent fewer conceptual changes.",
            "raw_fields": dict(rows[chosen])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--certify", action="store_true")
    parser.add_argument("--practical-threshold", type=float, default=5.0)
    parser.add_argument("--configuration", type=str, help="JSON object of encoder settings")
    parser.add_argument("--flow", action="store_true")
    parser.add_argument("--blame", action="store_true")
    parser.add_argument("--justify", action="store_true")
    parser.add_argument("--select-budget", type=int)
    parser.add_argument("--repair", action="store_true")
    args = parser.parse_args()
    started = perf_counter()
    if args.repair:
        result = repair(args.dataset)
    elif args.select_budget is not None:
        result = budget_select_dataset(args.dataset, args.select_budget)
    elif args.justify:
        result = justify(args.dataset)
    elif args.blame:
        result = blame(args.dataset)
    elif args.flow:
        result = {"csv_order_is_demonstration_sequence": True, "snapshots": flow(args.dataset)}
    elif args.certify:
        result = certify(args.dataset, args.practical_threshold)
    elif args.evaluate:
        result = evaluate(args.dataset)
    elif args.configuration:
        result = {"prediction": predict(args.dataset, json.loads(args.configuration))}
    else:
        result = summarize(args.dataset)
    result["runtime_seconds"] = perf_counter() - started
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
