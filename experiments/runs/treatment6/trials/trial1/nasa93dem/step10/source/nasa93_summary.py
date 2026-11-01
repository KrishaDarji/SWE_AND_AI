#!/usr/bin/env python3
"""NASA93 descriptive summary, using only the Python standard library."""
import csv, math, statistics, sys, random, json
from collections import Counter

MISSING = {"", "?", "na", "n/a", "null", "none"}
OUTCOMES = {"Kloc+": "maximize", "Effort-": "minimize", "Defects-": "minimize"}

def _number(value):
    text = value.strip()
    if text.lower() in MISSING: return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None

def _kind(values):
    present = [v for v in values if v.strip().lower() not in MISSING]
    return "numeric" if present and all(_number(v) is not None for v in present) else "categorical"

def _stats(values):
    xs = [v for v in values if v is not None]
    if not xs: return {"count": 0, "mean": None, "sd": None, "min": None, "median": None, "max": None}
    return {"count": len(xs), "mean": statistics.fmean(xs),
            "sd": statistics.stdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs), "median": statistics.median(xs), "max": max(xs)}

def read_dataset(path):
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream, skipinitialspace=False))
    if not rows: return [], [], []
    header = [cell.strip() for cell in rows[0]]
    if not any(header): return [], [], header
    records = []
    for raw in rows[1:]:
        if not raw or all(not cell.strip() for cell in raw): continue
        cells = list(raw[:len(header)]) + [""] * max(0, len(header) - len(raw))
        records.append({name: cells[i].strip() for i, name in enumerate(header)})
    return header, records, rows

def summarize(path):
    header, records, _ = read_dataset(path)
    kinds = {name: ("numeric" if name in OUTCOMES else _kind([r.get(name, "") for r in records])) for name in header}
    roles = {}
    for name in header:
        roles[name] = ("metadata" if name.endswith("X") else
                       "outcome" if name in OUTCOMES else "descriptor")
    numeric, categorical, missing, invalid = {}, {}, {}, {}
    for name in header:
        vals = [r.get(name, "") for r in records]
        missing[name] = sum(v.strip().lower() in MISSING for v in vals)
        if kinds[name] == "numeric":
            parsed = [_number(v) for v in vals]
            numeric[name] = _stats(parsed)
            invalid[name] = sum(v.strip().lower() not in MISSING and _number(v) is None for v in vals)
        else:
            categorical[name] = dict(sorted(Counter(v for v in vals if v.strip().lower() not in MISSING).items()))
            invalid[name] = 0
    return {"record_count": len(records), "columns": header, "types": kinds,
            "roles": roles, "missing": missing, "invalid_numeric": invalid,
            "numeric": numeric, "categorical": categorical,
            "outcomes": {k: {"direction": v, "role": "outcome"} for k, v in OUTCOMES.items() if k in header},
            "model_predictors": [n for n in header if roles[n] == "descriptor"]}

def _group_key(row, predictors):
    return tuple(row.get(n, "").strip() if row.get(n, "").strip().lower() not in MISSING else "<MISSING>" for n in predictors)

class EffortGuesser:
    def __init__(self, predictors, target="Effort-", seed=42):
        self.predictors, self.target, self.seed = list(predictors), target, seed
        self.groups, self.mean, self.fitted = {}, None, False
    def fit(self, rows):
        usable = [(r, _number(r.get(self.target, ""))) for r in rows]
        usable = [(r, y) for r, y in usable if y is not None]
        if not usable: raise ValueError("no finite training targets")
        self.mean = statistics.fmean(y for _, y in usable); totals = {}
        for row, y in usable: totals.setdefault(_group_key(row, self.predictors), []).append(y)
        self.groups = {k: statistics.fmean(v) for k, v in totals.items()}; self.fitted = True
        return self
    def predict(self, rows):
        if not self.fitted: raise ValueError("model is not fitted")
        return [self.groups.get(_group_key(r, self.predictors), self.mean) for r in rows]

def grouped_split(rows, predictors, seed=42, test_fraction=.25):
    groups = {}
    for row in rows: groups.setdefault(_group_key(row, predictors), []).append(row)
    if len(groups) < 2: raise ValueError("at least two distinct predictor groups are required")
    keys = list(groups); random.Random(seed).shuffle(keys); target = max(1, int(round(len(rows)*test_fraction)))
    test, train = [], []
    for key in keys:
        bucket = groups[key]
        if test and len(test) >= target: train.extend(bucket)
        elif test and len(test)+len(bucket) > target: train.extend(bucket)
        else: test.extend(bucket)
    if not train or not test: raise ValueError("split must have nonempty train and test sets")
    return train, test

def guess(path, proposal=None, seed=42):
    header, rows, _ = read_dataset(path)
    predictors = [n for n in header if not n.endswith("X") and n not in OUTCOMES]
    train, test = grouped_split(rows, predictors, seed); model = EffortGuesser(predictors).fit(train)
    actual = [_number(r.get("Effort-", "")) for r in test]; pred = model.predict(test)
    scored = [(a,p) for a,p in zip(actual,pred) if a is not None]
    mae = statistics.fmean(abs(a-p) for a,p in scored) if scored else None
    baseline = statistics.fmean(abs(a-model.mean) for a,p in scored) if scored else None
    result = {"predictors": predictors, "train_count": len(train), "test_count": len(test),
              "groups": len(set(_group_key(r,predictors) for r in rows)), "seed": seed,
              "model": "categorical exact-group mean; unseen/missing groups use training mean",
              "mae": mae, "baseline_mae": baseline, "evaluation": "held-out grouped 75/25 split"}
    if proposal is not None:
        extra, missing = set(proposal)-set(predictors), set(predictors)-set(proposal)
        if extra or missing: raise ValueError("proposal fields must exactly match predictors")
        result["proposal_prediction"] = model.predict([proposal])[0]
    return result

def _beta_cf(a, b, x):
    """Continued fraction for the regularized incomplete beta function."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1, a - 1
    c, d = 1.0, 1.0 - qab * x / qap; d = max(tiny, d); d = 1.0 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = max(tiny, 1 + aa * d); c = max(tiny, 1 + aa / c); d = 1 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = max(tiny, 1 + aa * d); c = max(tiny, 1 + aa / c); d = 1 / d
        delta = d * c; h *= delta
        if abs(delta - 1) < 3e-14: break
    return h

def _ibeta(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    front = math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log1p(-x))
    if x < (a+1)/(a+b+2): return front * _beta_cf(a,b,x) / a
    return 1 - front * _beta_cf(b,a,1-x) / b

def _t_cdf(t, df):
    x = df / (df + t*t); tail = 0.5 * _ibeta(df/2, 0.5, x)
    return 1-tail if t >= 0 else tail

def _t_quantile(p, df):
    lo, hi = -100.0, 100.0
    for _ in range(100):
        mid = (lo+hi)/2
        if _t_cdf(mid, df) < p: lo = mid
        else: hi = mid
    return (lo+hi)/2

def certify(path, threshold):
    """Compare observed Effort- for tool n (A) and h (B), not monetary cost."""
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("practical-effect threshold must be finite and nonnegative")
    _, rows, _ = read_dataset(path)
    groups = {"A": [], "B": []}
    for row in rows:
        tool = row.get("tool", "").strip().lower(); value = _number(row.get("Effort-", ""))
        if tool == "n" and value is not None: groups["A"].append(value)
        elif tool == "h" and value is not None: groups["B"].append(value)
    if len(groups["A"]) < 2 or len(groups["B"]) < 2: raise ValueError("both groups need at least two finite outcomes")
    a, b = groups["A"], groups["B"]; ma, mb = statistics.fmean(a), statistics.fmean(b)
    va, vb = statistics.variance(a), statistics.variance(b); diff = ma-mb
    se2 = va/len(a)+vb/len(b)
    if se2 == 0: raise ValueError("Welch comparison has zero standard error")
    df = se2*se2/((va/len(a))**2/(len(a)-1)+(vb/len(b))**2/(len(b)-1))
    se = math.sqrt(se2); critical = _t_quantile(.975, df); lo, hi = diff-critical*se, diff+critical*se
    p = 2*(1-_t_cdf(abs(diff/se), df)); direction = "B" if diff > 0 else "A" if diff < 0 else "neither"
    return {"groups": {"A": {"tool": "n", "count": len(a), "mean_effort": ma}, "B": {"tool": "h", "count": len(b), "mean_effort": mb}},
            "mean_difference_A_minus_B": diff, "confidence_interval_95": [lo, hi], "welch_df": df, "welch_two_sided_p": p,
            "difference_direction_favors": direction, "threshold": threshold,
            "point_estimate_exceeds_threshold": abs(diff) > threshold,
            "interval_entirely_practically_below_A": hi < -threshold,
            "interval_entirely_practically_below_B": lo > threshold,
            "interpretation": "observed Effort- is not verified monetary cost; practical importance is uncertain if the interval crosses a threshold"}

def flow(path, warmup=2):
    """Evaluate arrivals in CSV order, freezing each estimate before revealing its outcome."""
    header, rows, _ = read_dataset(path)
    predictors = [n for n in header if not n.endswith("X") and n not in OUTCOMES]
    if warmup < 0: raise ValueError("warmup must be nonnegative")
    results, history, scored = [], [], []
    cumulative_model = cumulative_baseline = 0.0
    for index, row in enumerate(rows, 1):
        identifier = row.get("idX", "").strip() or row.get("id", "").strip() or str(index)
        actual = _number(row.get("Effort-", ""))
        if len(history) < warmup:
            prediction = baseline = None
            phase = "warm-up"
        elif not history:
            prediction = baseline = None
            phase = "no-history"
        else:
            model = EffortGuesser(predictors).fit(history)
            prediction = model.predict([row])[0]
            baseline = model.mean
            phase = "scored"
        row_result = {"arrival": index, "identifier": identifier,
                      "prior_training_count": len(history), "phase": phase,
                      "prediction": prediction, "baseline": baseline,
                      "revealed_effort": actual, "absolute_error": None,
                      "baseline_absolute_error": None, "cumulative_mae": None,
                      "baseline_cumulative_mae": None}
        if phase == "scored" and actual is not None:
            error, base_error = abs(prediction - actual), abs(baseline - actual)
            scored.append((error, base_error)); cumulative_model += error; cumulative_baseline += base_error
            row_result.update(absolute_error=error, baseline_absolute_error=base_error,
                              cumulative_mae=cumulative_model / len(scored),
                              baseline_cumulative_mae=cumulative_baseline / len(scored))
        results.append(row_result)
        if actual is not None:
            history.append(row.copy())
    return {"predictors": predictors, "warmup": warmup, "sequence": "CSV order; demonstration sequence, not a verified timeline",
            "records": results, "scored_count": len(scored),
            "mae": cumulative_model / len(scored) if scored else None,
            "baseline_mae": cumulative_baseline / len(scored) if scored else None}

def _effort_groups(rows, descriptor):
    groups = {}
    for row in rows:
        y = _number(row.get("Effort-", ""))
        value = row.get(descriptor, "").strip()
        key = value if value.lower() not in MISSING else "<MISSING>"
        if y is not None: groups.setdefault(key, []).append(y)
    return groups

def blame(path):
    """Rank categorical descriptors by marginal association with observed effort.

    The primary score is eta-squared: between-group sum of squares divided by
    total sum of squares.  This is adapted from EZR's grouped summaries,
    diversity, and discretize/divide ideas; it is descriptive, not causal.
    """
    header, rows, _ = read_dataset(path)
    predictors = [n for n in header if not n.endswith("X") and n not in OUTCOMES]
    ys = [_number(r.get("Effort-", "")) for r in rows]
    finite = [y for y in ys if y is not None]
    overall = statistics.fmean(finite) if finite else None
    total_ss = sum((y - overall) ** 2 for y in finite) if overall is not None else 0.0
    ranked = []
    for order, name in enumerate(predictors):
        groups = _effort_groups(rows, name)
        n = sum(map(len, groups.values()))
        means = {k: statistics.fmean(v) for k, v in groups.items() if v}
        between = sum(len(v) * (means[k] - overall) ** 2 for k, v in groups.items()) if overall is not None else 0.0
        score = between / total_ss if total_ss > 0 else 0.0
        largest = max(means, key=lambda k: (means[k], k)) if means else None
        smallest = min(means, key=lambda k: (means[k], k)) if means else None
        ranked.append({"descriptor": name, "score": score, "direction":
                      "higher in " + largest + "; lower in " + smallest if means and largest != smallest else "no directional contrast",
                      "group_count": len(groups), "usable_count": n,
                      "groups": {k: {"count": len(v), "effort": _stats(v), "mean": means[k]} for k, v in sorted(groups.items())},
                      "note": "marginal categorical association; groups with fewer than 2 observations are retained"})
    ranked.sort(key=lambda x: (-x["score"], predictors.index(x["descriptor"])))
    return {"target": "Effort-", "predictors": predictors, "usable_effort_count": len(finite),
            "overall_effort": _stats(finite), "score": "eta_squared_between_over_total",
            "association_type": "marginal; correlated descriptors can share explanatory information",
            "limitations": ["higher effort alone does not establish inefficiency because scope differs",
                            "delivered size is excluded; this is not size-adjusted productivity",
                            "no causal, defect-risk, process-improvement, or monetary claim"], "ranking": ranked}

def justify(path, seed=42):
    """Freeze a sponsor-facing estimate and evidence before an outcome audit.

    The selected demonstration is the earliest CSV-order member of the grouped
    held-out partition.  All model fitting and association summaries use train
    rows only; the retrospective value is attached only after the frozen report.
    """
    header, rows, _ = read_dataset(path)
    predictors = [n for n in header if not n.endswith("X") and n not in OUTCOMES]
    usable = [(i, r) for i, r in enumerate(rows) if _number(r.get("Effort-", "")) is not None]
    if len(usable) < 2: raise ValueError("at least two rows with finite Effort- targets are required")
    clean = [r for _, r in usable]
    train, test = grouped_split(clean, predictors, seed)
    model = EffortGuesser(predictors, seed=seed).fit(train)
    selected = min(test, key=lambda r: usable[[x[1] for x in usable].index(r)][0])
    prediction = model.predict([selected])[0]
    key = _group_key(selected, predictors)
    matched = [r for r in train if _group_key(r, predictors) == key]
    source = "exact training descriptor group" if matched else "training mean fallback for unseen group"
    evidence = [{"label": "training matched row", "identifier": r.get("idX", "").strip() or str(train.index(r)+1),
                 "effort": _number(r.get("Effort-", ""))} for r in matched]
    associations = blame_rows(train, predictors)
    frozen = {"record_identifier": selected.get("idX", "").strip() or selected.get("id", "").strip() or str(rows.index(selected)+1),
              "csv_position": rows.index(selected)+1, "predictors": predictors,
              "proposal": {p: selected.get(p, "").strip() for p in predictors},
              "train_count": len(train), "test_count": len(test), "seed": seed,
              "estimate": prediction, "training_reference": {"label": "training mean", "value": model.mean},
              "explanation": {"label": source, "group_key": key, "matched_training_evidence": evidence,
                              "historical_associations_training_only": associations},
              "limitations": ["associations are marginal and not additive local contributions or causal effects",
                              "missing or unseen descriptor groups use the training mean",
                              "correlated descriptors can share explanatory information",
                              "scope, size, and deployment behavior are not inferred; Effort- units are not invented"]}
    audit = {"label": "retrospective audit; measured after freeze", "observed_effort": _number(selected.get("Effort-", "")),
             "absolute_error": abs(prediction - _number(selected.get("Effort-", ""))) if _number(selected.get("Effort-", "")) is not None else None}
    return {"pre_audit": frozen, "retrospective_audit": audit}

def choose(path, budget):
    """Select the best observed record whose observed effort fits a budget.

    This is retrospective selection, not prediction or causal optimization:
    Kloc+ is only an explicitly labeled delivered-size proxy for value.
    """
    if not isinstance(budget, (int, float)) or isinstance(budget, bool) or not math.isfinite(budget) or budget < 0:
        raise ValueError("budget must be a finite nonnegative number")
    header, rows, _ = read_dataset(path)
    outcomes = {"Kloc+", "Effort-", "Defects-"}
    descriptors = [n for n in header if not n.endswith("X") and n not in outcomes]
    candidates = []
    for position, row in enumerate(rows, 1):
        effort, size = _number(row.get("Effort-", "")), _number(row.get("Kloc+", ""))
        if effort is not None and size is not None and effort <= budget:
            candidates.append((size, effort, position, row))
    if not candidates:
        raise ValueError("no feasible records with finite Effort- and Kloc+")
    _, effort, position, selected = max(candidates, key=lambda x: (x[0], -x[1], -x[2]))
    identifier = selected.get("idX", "").strip() or selected.get("id", "").strip() or str(position)
    return {"budget": budget, "feasible_count": len(candidates), "total_count": len(rows),
            "record_identifier": identifier, "csv_position": position,
            "descriptor_settings": {name: selected.get(name, "").strip() for name in descriptors},
            "observed_effort": effort, "delivered_size_proxy": _number(selected.get("Kloc+", "")),
            "selection_rule": "among finite observed Effort- <= budget, maximize observed Kloc+; tie-break lower observed Effort-, then earliest CSV record",
            "outcome_basis": "observed outcomes in the supplied records, not predictions for unseen proposals",
            "limitations": ["Kloc+ is a delivered-size proxy, not business value or productivity",
                            "metadata and outcomes are not descriptor settings",
                            "selection does not establish that settings caused outcomes or guarantee future feasibility"]}

def fix(path, proposal, budget, seed=42):
    """Suggest a minimum-change repair using training rows only.

    This is a hypothetical descriptor change, not a causal or operational
    recommendation.  Candidate configurations are deduplicated in CSV order.
    """
    if not isinstance(budget, (int, float)) or isinstance(budget, bool) or not math.isfinite(budget) or budget < 0:
        raise ValueError("budget must be a finite nonnegative number")
    if not isinstance(proposal, dict): raise ValueError("proposal must be an object")
    header, rows, _ = read_dataset(path)
    predictors = [n for n in header if not n.endswith("X") and n not in OUTCOMES]
    if set(proposal) != set(predictors):
        raise ValueError("proposal must contain exactly all descriptor fields")
    normalized = {name: str(proposal[name]).strip() for name in predictors}
    usable = [(position, row) for position, row in enumerate(rows, 1)
              if _number(row.get("Effort-", "")) is not None]
    if len(usable) < 2: raise ValueError("at least two finite training targets are required")
    clean = [row for _, row in usable]
    train, _ = grouped_split(clean, predictors, seed)
    model = EffortGuesser(predictors, seed=seed).fit(train)
    proposal_prediction = model.predict([normalized])[0]
    original = dict(normalized)
    base = {"budget": budget, "seed": seed, "predictors": predictors,
            "original_descriptors": original, "original_estimated_effort": proposal_prediction,
            "model": "training-only categorical exact-group mean; unseen groups use training mean",
            "candidate_source": "distinct descriptor configurations in training rows only",
            "limitations": ["change count is a proxy, not monetary cost or operational risk",
                            "candidates are hypothetical; descriptors may not be actionable",
                            "observed combinations may not preserve project scope",
                            "model differences are not causal improvements or guaranteed savings"]}
    if proposal_prediction <= budget:
        base.update(status="no-repair-needed", predicted_feasible=True,
                    candidate_descriptors=original, changed_fields=[], change_count=0,
                    candidate_estimated_effort=proposal_prediction, candidate_record_identifier=None)
        return base
    seen, candidates = set(), []
    for position, row in usable:
        config = {name: row.get(name, "").strip() for name in predictors}
        key = tuple(config[name] if config[name].lower() not in MISSING else "<MISSING>" for name in predictors)
        if key in seen: continue
        seen.add(key)
        predicted = model.predict([config])[0]
        if predicted <= budget:
            changed = [{"field": name, "old": original[name], "new": config[name]}
                       for name in predictors if original[name] != config[name]]
            identifier = row.get("idX", "").strip() or row.get("id", "").strip() or str(position)
            candidates.append((len(changed), predicted, position, config, changed, identifier))
    if not candidates:
        base.update(status="no-feasible-repair", predicted_feasible=False,
                    candidate_descriptors=None, changed_fields=[], change_count=None,
                    candidate_estimated_effort=None, candidate_record_identifier=None)
        return base
    count, predicted, position, config, changed, identifier = min(candidates, key=lambda x: (x[0], x[1], x[2]))
    base.update(status="repair-suggested", predicted_feasible=True,
                candidate_descriptors=config, changed_fields=changed, change_count=count,
                candidate_estimated_effort=predicted, candidate_record_identifier=identifier,
                candidate_csv_position=position)
    return base

def _spot_reference(values):
    """Return robust center/scale on log1p(nonnegative) values."""
    transformed = [math.log1p(v) for v in values]
    center = statistics.median(transformed)
    deviations = [abs(v - center) for v in transformed]
    mad = statistics.median(deviations)
    return center, 1.4826 * mad

def spot(path, threshold=3.5):
    """Screen Kloc+, Effort-, and Defects- for marginal unusual records.

    Each record is scored against leave-one-out historical references. Values
    are log1p transformed to reduce skew; the robust scale is 1.4826*MAD.
    A zero scale makes a non-median value a candidate but its score undefined.
    This is an investigation screen, not an error detector or size adjustment.
    """
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be a finite nonnegative number")
    header, rows, _ = read_dataset(path)
    fields = [x for x in ("Kloc+", "Effort-", "Defects-") if x in header]
    flags, per_field = [], {field: 0 for field in fields}
    for position, row in enumerate(rows, 1):
        record_flags = []
        for field in fields:
            observed = _number(row.get(field, ""))
            if observed is None or observed < 0:
                if row.get(field, "").strip().lower() not in MISSING:
                    per_field[field] += 1
                    record_flags.append({"field": field, "observed": None,
                                         "reference": None, "score": None,
                                         "threshold": threshold,
                                         "reason": "missing, nonfinite, or negative value is not scored"})
                continue
            reference_values = [_number(other.get(field, "")) for i, other in enumerate(rows) if i != position - 1]
            reference_values = [v for v in reference_values if v is not None and v >= 0]
            if not reference_values:
                continue
            center, scale = _spot_reference(reference_values)
            transformed = math.log1p(observed)
            reference = {"transformed_median": center,
                         "original_median": math.expm1(center),
                         "robust_scale_transformed": scale,
                         "reference_count": len(reference_values),
                         "excludes_assessed_record": True}
            if scale == 0:
                score = None
                unusual = transformed != center
                reason = "zero robust scale; non-median value is a candidate" if unusual else "zero robust scale; equals reference median"
            else:
                score = abs(transformed - center) / scale
                unusual = score >= threshold
                reason = "absolute robust z-score meets threshold" if unusual else "below threshold"
            if unusual:
                per_field[field] += 1
                record_flags.append({"field": field, "observed": observed, "reference": reference,
                                     "score": score, "threshold": threshold, "reason": reason})
        if record_flags:
            identifier = row.get("idX", "").strip() or row.get("id", "").strip() or None
            flags.append({"identifier": identifier, "csv_position": position,
                          "flagged_fields": [x["field"] for x in record_flags],
                          "flags": record_flags,
                          "reason": "marginal unusualness candidate; investigate rather than assume error"})
    return {"method": "leave-one-out log1p robust univariate screen",
            "threshold": threshold, "fields": fields, "flags": flags,
            "unique_flagged_record_count": len(flags), "per_outcome_flag_counts": per_field,
            "limitations": ["large projects may legitimately have large effort or defect counts",
                            "marginal screen is not scope-adjusted and has no labels for accuracy",
                            "scores are not prediction uncertainty or causal explanations",
                            "duplicate identifiers remain distinguishable by csv_position"]}

RACE_TIE_TOLERANCE = 1e-12

def paired_difference(model_mae, baseline_mae, tolerance=RACE_TIE_TOLERANCE):
    """Return model-minus-baseline MAE and its fixed-tolerance label."""
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (model_mae, baseline_mae, tolerance)) or tolerance < 0:
        raise ValueError("MAEs and tolerance must be finite and nonnegative")
    difference = model_mae - baseline_mae
    label = "tie" if abs(difference) <= tolerance else "win" if difference < 0 else "loss"
    return {"difference": difference, "classification": label, "tolerance": tolerance}

def race(path, seeds=range(42, 52), test_fraction=.25):
    """Compare the exact-group estimator with a training-mean baseline.

    This translates EZR's CSV normalization, SYM-style categorical handling,
    seeded shuffle, and table/group summaries into a reproducible comparison.
    Every split fits a fresh model on finite-target training rows only.
    """
    header, rows, _ = read_dataset(path)
    predictors = [n for n in header if not n.endswith("X") and n not in OUTCOMES]
    numbered = []
    for position, row in enumerate(rows, 1):
        if _number(row.get("Effort-", "")) is not None:
            copy = dict(row); copy["__csv_position"] = position; numbered.append(copy)
    if len({ _group_key(r, predictors) for r in numbered }) < 2:
        raise ValueError("at least two distinct finite-target descriptor groups are required")
    split_results = []
    for seed in seeds:
        train, test = grouped_split(numbered, predictors, seed, test_fraction)
        model = EffortGuesser(predictors, seed=seed).fit(train)
        baseline = model.mean
        predictions = model.predict(test)
        observed = [_number(r.get("Effort-", "")) for r in test]
        model_mae = statistics.fmean(abs(y-p) for y, p in zip(observed, predictions))
        baseline_mae = statistics.fmean(abs(y-baseline) for y in observed)
        paired = paired_difference(model_mae, baseline_mae)
        projects = []
        for row, prediction, actual in zip(test, predictions, observed):
            projects.append({"csv_position": row["__csv_position"], "idX": row.get("idX", ""),
                             "prediction": prediction, "baseline_prediction": baseline,
                             "observed_effort": actual})
        split_results.append({"seed": seed, "train_count": len(train), "test_count": len(test),
                              "train_groups": len({_group_key(r, predictors) for r in train}),
                              "test_groups": len({_group_key(r, predictors) for r in test}),
                              "group_separation": not ({_group_key(r, predictors) for r in train} &
                                                         {_group_key(r, predictors) for r in test}),
                              "train_csv_positions": [r["__csv_position"] for r in train],
                              "test_csv_positions": [r["__csv_position"] for r in test],
                              "training_mean": baseline, "projects": projects,
                              "model_mae": model_mae, "baseline_mae": baseline_mae,
                              "paired_difference": paired["difference"],
                              "classification": paired["classification"],
                              "negative_prediction_count": sum(p < 0 for p in predictions)})
    differences = [x["paired_difference"] for x in split_results]
    counts = Counter(x["classification"] for x in split_results)
    return {"method": "race: exact normalized 22-descriptor group means vs training mean",
            "predictors": predictors, "predictor_count": len(predictors),
            "seeds": list(seeds), "test_fraction": test_fraction,
            "tie_tolerance": RACE_TIE_TOLERANCE, "splits": split_results,
            "paired_difference_summary": {"mean": statistics.fmean(differences),
                                           "median": statistics.median(differences),
                                           "wins": counts["win"], "ties": counts["tie"], "losses": counts["loss"]},
            "limitations": ["repeated splits overlap and are not independent datasets or independent LLM trials",
                            "identical model and baseline results are valid when held-out groups are unseen",
                            "negative predictions are reported and are not silently clipped",
                            "no statistical superiority or deployment reliability claim; final same procedure analysis pending"]}

def blame_rows(rows, predictors):
    finite = [_number(r.get("Effort-", "")) for r in rows]
    finite = [x for x in finite if x is not None]
    overall = statistics.fmean(finite) if finite else None
    result = []
    for name in predictors:
        groups = _effort_groups(rows, name)
        means = {k: statistics.fmean(v) for k, v in groups.items() if v}
        ss = sum((y-overall)**2 for y in finite) if overall is not None else 0
        between = sum(len(v)*(means[k]-overall)**2 for k,v in groups.items()) if overall is not None else 0
        result.append({"descriptor": name, "label": "historical association; training rows only",
                       "eta_squared": between/ss if ss else 0.0, "group_means": dict(sorted(means.items()))})
    return sorted(result, key=lambda x: (-x["eta_squared"], predictors.index(x["descriptor"])))

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) not in (1, 2, 3, 4): raise SystemExit("usage: nasa93_summary.py DATASET.csv [proposal.json] or --fix DATASET.csv PROPOSAL.json BUDGET")
    if len(argv) == 3 and argv[0] == "--spot": print(json.dumps(spot(argv[1], float(argv[2])), indent=2, sort_keys=True, allow_nan=False))
    elif len(argv) == 2 and argv[0] == "--spot": print(json.dumps(spot(argv[1]), indent=2, sort_keys=True, allow_nan=False))
    elif len(argv) == 3 and argv[0] == "--choose": print(json.dumps(choose(argv[1], float(argv[2])), indent=2, sort_keys=True))
    elif len(argv) == 2 and argv[0] == "--justify": print(json.dumps(justify(argv[1]), indent=2, sort_keys=True))
    elif len(argv) == 2 and argv[0] == "--flow": print(json.dumps(flow(argv[1]), indent=2, sort_keys=True))
    elif len(argv) == 2 and argv[0] == "--blame": print(json.dumps(blame(argv[1]), indent=2, sort_keys=True))
    elif len(argv) == 2 and argv[0] == "--race": print(json.dumps(race(argv[1]), indent=2, sort_keys=True, allow_nan=False))
    elif len(argv) == 3 and argv[0] == "--certify": print(json.dumps(certify(argv[1], float(argv[2])), indent=2, sort_keys=True))
    elif len(argv) == 4 and argv[0] == "--fix":
        with open(argv[2], encoding="utf-8") as f: proposal = json.load(f)
        print(json.dumps(fix(argv[1], proposal, float(argv[3])), indent=2, sort_keys=True))
    elif len(argv) == 1: print(json.dumps(summarize(argv[0]), indent=2, sort_keys=True))
    else:
        with open(argv[1], encoding="utf-8") as f: proposal = json.load(f)
        print(json.dumps(guess(argv[0], proposal), indent=2, sort_keys=True))

if __name__ == "__main__": main()
