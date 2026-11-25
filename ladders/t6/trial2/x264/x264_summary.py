"""Summarize historical X264 encoder settings and measured outcomes."""
import csv, math, statistics, sys, random

MISSING = {"", "?", "nil", "NA", "N/A", "null"}

def _number(value):
    text = value.strip()
    if text in MISSING:
        return None
    try:
        number = float(text)
        return number if math.isfinite(number) else None
    except ValueError:
        return None

def iter_measurements(path):
    """Yield (data_position, header, record) without reading future rows."""
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        try:
            header = [cell.strip() for cell in next(reader)]
        except StopIteration:
            return
        position = 0
        for row in reader:
            if not any(cell.strip() for cell in row):
                continue
            position += 1
            values = (row + [""] * max(0, len(header) - len(row)))[:len(header)]
            yield position, header, dict(zip(header, values))

class FlowAccumulator:
    """Incremental finite-outcome statistics and earliest-best tracking."""
    def __init__(self, header=None, outcome="Performance-"):
        self.header = list(header or [])
        self.outcome = outcome
        self.seen = self.valid = self.missing = self.invalid = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.best = None

    def update(self, position, record):
        self.seen += 1
        if not self.header:
            self.header = list(record)
        raw = record.get(self.outcome, "")
        value = _number(raw)
        if raw.strip() in MISSING:
            self.missing += 1
        elif value is None:
            self.invalid += 1
        else:
            self.valid += 1
            delta = value - self.mean
            self.mean += delta / self.valid
            self.m2 += delta * (value - self.mean)
            if self.best is None or value < self.best["value"]:
                self.best = {"value": value, "position": position,
                             "configuration": {name: record.get(name, "")
                                                for name in self.header
                                                if name != self.outcome}}
        return self.snapshot()

    def snapshot(self):
        best = None if self.best is None else {
            "value": self.best["value"], "position": self.best["position"],
            "configuration": dict(self.best["configuration"])}
        return {"rows_seen": self.seen, "valid_count": self.valid,
                "missing_outcomes": self.missing, "invalid_outcomes": self.invalid,
                "mean": self.mean if self.valid else None,
                "sample_variance": self.m2 / (self.valid - 1) if self.valid > 1 else None,
                "best": best}

def flow(path):
    accumulator = None
    for position, header, record in iter_measurements(path):
        if accumulator is None:
            accumulator = FlowAccumulator(header)
        snapshot = accumulator.update(position, record)
    return accumulator.snapshot() if accumulator else FlowAccumulator().snapshot()

def render_flow(result):
    lines = ["X264 flow (one measurement at a time)",
             "rows_seen: %d; valid_count: %d; missing_outcomes: %d; invalid_outcomes: %d" %
             (result["rows_seen"], result["valid_count"], result["missing_outcomes"], result["invalid_outcomes"]),
             "mean: %s; sample_variance: %s" % (result["mean"], result["sample_variance"])]
    if result["best"] is None:
        lines.append("best: unavailable")
    else:
        lines.append("best: value=%s; position=%d; configuration=%s" %
                     (result["best"]["value"], result["best"]["position"], result["best"]["configuration"]))
    lines.append("position counts nonblank data rows starting at 1; blank rows are ignored; missing settings are retained as empty strings")
    return "\n".join(lines)
def load(path):
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        return [], [], []
    header = [cell.strip() for cell in rows[0]]
    records = []
    invalid = []
    for line, row in enumerate(rows[1:], 2):
        if not any(cell.strip() for cell in row):
            continue
        values = row + [""] * max(0, len(header) - len(row))
        values = values[:len(header)]
        records.append(dict(zip(header, values)))
    return header, records, invalid

def summarize(path):
    header, records, _ = load(path)
    outcome_names = [name for name in header if name.endswith("-")]
    outcome = outcome_names[0] if outcome_names else None
    settings = [name for name in header if name != outcome]
    frequencies = {}
    for name in settings:
        counts = {}
        for record in records:
            value = record.get(name, "").strip()
            key = value if value not in MISSING else "<missing>"
            counts[key] = counts.get(key, 0) + 1
        frequencies[name] = counts
    outcomes = []
    missing = 0
    invalid = 0
    if outcome:
        for record in records:
            raw = record.get(outcome, "")
            value = _number(raw)
            if raw.strip() in MISSING:
                missing += 1
            elif value is None:
                invalid += 1
            else:
                outcomes.append(value)
    stats = None
    if outcomes:
        stats = {"count": len(outcomes), "min": min(outcomes),
                 "max": max(outcomes), "mean": statistics.fmean(outcomes),
                 "median": statistics.median(outcomes),
                 "stdev": statistics.stdev(outcomes) if len(outcomes) > 1 else 0.0}
    return {"record_count": len(records), "headers": header, "settings": settings,
            "outcome": outcome, "direction": "lower is better" if outcome else None,
            "frequencies": frequencies, "outcome_stats": stats,
            "missing_outcomes": missing, "invalid_outcomes": invalid,
            "missing_input": not bool(header)}

def render(summary):
    lines = ["Historical X264 run summary", "record_count: %d" % summary["record_count"],
             "outcome: %s (%s)" % (summary["outcome"] or "none", summary["direction"] or "unavailable")]
    if summary["outcome_stats"]:
        lines.append("outcome_stats: " + ", ".join("%s=%.6g" % item for item in summary["outcome_stats"].items()))
    else:
        lines.append("outcome_stats: unavailable")
    lines.append("missing_outcomes: %d; invalid_outcomes: %d" % (summary["missing_outcomes"], summary["invalid_outcomes"]))
    lines.append("setting_frequencies:")
    for name, counts in summary["frequencies"].items():
        lines.append("  %s: %s" % (name, ", ".join("%s=%d" % pair for pair in sorted(counts.items()))))
    return "\n".join(lines)

def _groups(records, settings):
    groups = {}
    for record in records:
        key = tuple(record.get(name, "").strip() for name in settings)
        groups.setdefault(key, []).append(record)
    return [(key, groups[key]) for key in sorted(groups)]

def _target(record, outcome):
    return _number(record.get(outcome, ""))

def split_groups(records, settings, seed=42):
    groups = _groups(records, settings)
    if len(groups) < 2: raise ValueError("at least two configuration groups are required")
    order = list(range(len(groups))); random.Random(seed).shuffle(order)
    cut = min(max(1, int(round(len(groups) * .8))), len(groups) - 1)
    train_keys = {groups[i][0] for i in order[:cut]}
    train = [r for key, rows in groups if key in train_keys for r in rows]
    test = [r for key, rows in groups if key not in train_keys for r in rows]
    return train, test

class NearestNeighborModel:
    """1-NN on training-fitted numeric/categorical normalized settings."""
    def __init__(self, settings, outcome): self.settings, self.outcome = settings, outcome
    def fit(self, records):
        valid = [(r, _target(r, self.outcome)) for r in records if _target(r, self.outcome) is not None]
        if not valid: raise ValueError("no finite training targets")
        self.rows = [r for r, _ in valid]; self.values = [y for _, y in valid]
        self.encoders = {}
        for name in self.settings:
            vals = sorted({r.get(name, "").strip() for r in self.rows})
            self.encoders[name] = {v: i / max(1, len(vals)-1) for i, v in enumerate(vals)}
        self.mean = statistics.fmean(self.values); return self
    def _vector(self, record):
        missing = [n for n in self.settings if n not in record]
        extra = [n for n in record if n not in self.settings and n != self.outcome]
        if missing or extra: raise ValueError("configuration fields must exactly match settings")
        out = []
        for name in self.settings:
            value = record.get(name, "").strip()
            if value in MISSING: raise ValueError("missing setting: %s" % name)
            if value not in self.encoders[name]: raise ValueError("unseen setting value for %s" % name)
            out.append(self.encoders[name][value])
        return out
    def predict_one(self, record):
        try: vector = self._vector(record)
        except ValueError as exc:
            if str(exc).startswith("unseen setting value"): return self.mean
            raise
        best = min(range(len(self.rows)), key=lambda i: sum((a-b)**2 for a,b in zip(vector, self._vector(self.rows[i]))))
        return self.values[best]
    def predict(self, records): return [self.predict_one(r) for r in records]

def evaluate(path, seed=42):
    header, records, _ = load(path); outcome = "Performance-"; settings = [n for n in header if n != outcome]
    train, test = split_groups(records, settings, seed); model = NearestNeighborModel(settings, outcome).fit(train)
    valid_test = [(r, _target(r, outcome)) for r in test if _target(r, outcome) is not None]
    predictions = [model.predict_one(r) for r, _ in valid_test]
    mae = statistics.fmean(abs(p-y) for p, (_, y) in zip(predictions, valid_test)) if valid_test else None
    baseline = statistics.fmean(model.values)
    base_mae = statistics.fmean(abs(b-y) for _, (_, y) in zip(valid_test, valid_test)) if False else (statistics.fmean(abs(baseline-y) for _, y in valid_test) if valid_test else None)
    return {"model": model, "settings": settings, "train_rows": len(train), "test_rows": len(test), "train_groups": len(_groups(train, settings)), "test_groups": len(_groups(test, settings)), "evaluated_test_rows": len(valid_test), "mae": mae, "baseline_mae": base_mae}

def render_prediction(result, configuration):
    return "prediction: %.6f\ntrain_rows: %d; test_rows: %d; train_groups: %d; test_groups: %d\nheld_out_mae: %s; baseline_mae: %s" % (result["model"].predict_one(configuration), result["train_rows"], result["test_rows"], result["train_groups"], result["test_groups"], result["mae"], result["baseline_mae"])

def _betacf(a, b, x):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab*x/qap; d = max(1e-300, d); d = 1.0/d; h = d
    for m in range(1, 201):
        m2 = 2*m; aa = m*(b-m)*x/((qam+m2)*(a+m2))
        d = 1.0 + aa*d; d = max(1e-300, d); c = 1.0 + aa/c; c = max(1e-300, c); d = 1.0/d; h *= d*c
        aa = -(a+m)*(qab+m)*x/((a+m2)*(qap+m2)); d = 1.0 + aa*d; d = max(1e-300, d); c = 1.0 + aa/c; c = max(1e-300, c); d = 1.0/d
        delta = d*c; h *= delta
        if abs(delta-1.0) < 3e-14: break
    return h

def _ibeta(a, b, x):
    if not 0 <= x <= 1: raise ValueError("beta argument outside [0,1]")
    if x in (0, 1): return x
    front = math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log1p(-x))
    return front*_betacf(a,b,x)/a if x < (a+1)/(a+b+2) else 1-front*_betacf(b,a,1-x)/b

def _t_cdf(t, df):
    if df <= 0: raise ValueError("degrees of freedom must be positive")
    if t == 0: return 0.5
    x = df/(df+t*t); tail = 0.5*_ibeta(df/2, 0.5, x)
    return 1-tail if t > 0 else tail

def _t_quantile(p, df):
    if not 0 < p < 1: raise ValueError("probability must be inside (0,1)")
    lo, hi = -1.0, 1.0
    while _t_cdf(lo, df) > p: lo *= 2
    while _t_cdf(hi, df) < p: hi *= 2
    for _ in range(100):
        mid = (lo+hi)/2
        if _t_cdf(mid, df) < p: lo = mid
        else: hi = mid
    return (lo+hi)/2

def certify(path, threshold=None):
    header, records, _ = load(path); setting = "no_8x8dct"; outcome = "Performance-"
    if setting not in header or outcome not in header: raise ValueError("required columns are absent")
    groups = {0: [], 1: []}; excluded = 0
    for r in records:
        raw = r.get(setting, '').strip(); y = _number(r.get(outcome, ''))
        try: key = float(raw)
        except ValueError: key = None
        if key in (0.0, 1.0) and y is not None: groups[int(key)].append(y)
        else: excluded += 1
    result = {"reference": _sample_stats(groups[0]), "candidate": _sample_stats(groups[1]), "excluded": excluded, "threshold": threshold}
    a,b = groups[0], groups[1]
    if len(a) < 2 or len(b) < 2: result["inference"] = None; return result
    diff = statistics.fmean(b)-statistics.fmean(a); va, vb = statistics.variance(a), statistics.variance(b); se = math.sqrt(va/len(a)+vb/len(b))
    if se == 0: result["inference"] = None; return result
    df = (va/len(a)+vb/len(b))**2/((va/len(a))**2/(len(a)-1)+(vb/len(b))**2/(len(b)-1)); t = diff/se; p = 2*min(_t_cdf(t,df), 1-_t_cdf(t,df)); crit = _t_quantile(.975,df)
    result["inference"] = {"difference": diff, "standard_error": se, "df": df, "t": t, "p_two_sided": p, "ci95": (diff-crit*se, diff+crit*se), "statistically_significant": p < .05}
    if threshold is not None:
        result["practical"] = {"point_estimate_meets": diff <= -threshold, "ci_entirely_supports": result["inference"]["ci95"][1] <= -threshold}
    return result

def _sample_stats(values):
    return {"n": len(values), "mean": statistics.fmean(values) if values else None, "variance": statistics.variance(values) if len(values)>1 else None}

def render_certify(result):
    lines = ["Certification: no_8x8dct=1 candidate minus no_8x8dct=0 reference", "lower Performance- is better"]
    for name in ("reference", "candidate"): lines.append("%s: n=%s mean=%s sample_variance=%s" % (name, result[name]["n"], result[name]["mean"], result[name]["variance"]))
    inf = result.get("inference"); lines.append("excluded=%d" % result["excluded"])
    if inf is None: lines.append("inference: undefined (insufficient samples or zero standard error)")
    else: lines.append("difference=%g se=%g df=%g t=%g p_two_sided=%g ci95=(%g,%g)" % (inf["difference"],inf["standard_error"],inf["df"],inf["t"],inf["p_two_sided"],*inf["ci95"]))
    if result.get("threshold") is not None and inf is not None: lines.append("threshold=%g point_estimate_meets=%s ci_entirely_supports=%s" % (result["threshold"],result["practical"]["point_estimate_meets"],result["practical"]["ci_entirely_supports"]))
    lines.append("Observational historical configurations differ in other settings; this does not prove causation or guarantee future improvement.")
    return "\n".join(lines)

def _setting_key(raw):
    """Normalize numeric-looking binary settings while retaining symbols."""
    text = raw.strip()
    if text in MISSING: return None
    try:
        value = float(text)
        if not math.isfinite(value): return None
        return str(int(value)) if value in (0.0, 1.0) else str(value)
    except ValueError:
        return text

def blame(path, outcome="Performance-"):
    """Rank each setting by marginal eta-squared against a finite outcome."""
    header, records, _ = load(path)
    settings = [name for name in header if name != outcome]
    result = []
    for setting in settings:
        groups = {}; excluded = 0
        for record in records:
            key = _setting_key(record.get(setting, "")); y = _number(record.get(outcome, ""))
            if key is None or y is None:
                excluded += 1
            else:
                groups.setdefault(key, []).append(y)
        eligible = sum(len(values) for values in groups.values())
        reference = statistics.fmean([y for values in groups.values() for y in values]) if eligible else None
        between = sum(len(values) * (statistics.fmean(values) - reference) ** 2 for values in groups.values()) if eligible else None
        total = sum((y - reference) ** 2 for values in groups.values() for y in values) if eligible else None
        score = between / total if total and len(groups) >= 2 else None
        summaries = [{"value": key, "count": len(values), "mean": statistics.fmean(values)}
                     for key, values in sorted(groups.items())]
        means = [item["mean"] for item in summaries]
        result.append({"setting": setting, "eligible": eligible, "excluded": excluded,
                       "groups": summaries, "reference_mean": reference,
                       "between_ss": between, "total_ss": total, "eta_squared": score,
                       "higher_mean_groups": [x["value"] for x in summaries if reference is not None and x["mean"] > reference],
                       "lower_mean_groups": [x["value"] for x in summaries if reference is not None and x["mean"] < reference]})
    result.sort(key=lambda x: (x["eta_squared"] is None, -(x["eta_squared"] or 0.0), x["setting"]))
    for rank, item in enumerate(result, 1): item["rank"] = rank
    return {"outcome": outcome, "direction": "lower is better", "settings": result,
            "conventions": "Missing/invalid settings or outcomes are excluded per setting. Score is undefined for fewer than two groups, fewer than one eligible row, or constant outcome; constant settings are reported with undefined score."}

def render_blame(result):
    lines = ["Marginal encoder-setting blame", "outcome: %s (lower is better)" % result["outcome"], result["conventions"]]
    for item in result["settings"]:
        score = "undefined" if item["eta_squared"] is None else "%.9g" % item["eta_squared"]
        lines.append("rank=%d setting=%s eligible=%d excluded=%d reference_mean=%s between_ss=%s total_ss=%s eta_squared=%s higher=%s lower=%s" %
                     (item["rank"], item["setting"], item["eligible"], item["excluded"], item["reference_mean"], item["between_ss"], item["total_ss"], score, item["higher_mean_groups"], item["lower_mean_groups"]))
        lines.extend("  group=%s count=%d mean=%.9g" % (g["value"], g["count"], g["mean"]) for g in item["groups"])
    lines.append("These are marginal historical associations, not independent causal contributions or significance tests; correlated/one-hot settings can reflect the same underlying choice, and changing a setting is not guaranteed to improve performance.")
    return "\n".join(lines)

def justify(path, outcome="Performance-"):
    """Select the lowest finite outcome among complete binary historical rows."""
    header, records, _ = load(path)
    if not header:
        return {"status": "empty input", "position_convention": "nonblank data rows start at 1; blank rows are ignored"}
    if outcome not in header:
        return {"status": "unavailable reference", "reason": "outcome column is absent"}
    settings = [name for name in header if name != outcome]
    eligible = []; excluded = {"missing_outcome": 0, "invalid_outcome": 0,
                               "invalid_settings": 0, "incomplete_settings": 0}
    position = 0
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream); next(reader, None)
        for row in reader:
            if not any(cell.strip() for cell in row): continue
            position += 1
            values = (row + [""] * max(0, len(header) - len(row)))[:len(header)]
            record = dict(zip(header, values)); raw = record.get(outcome, "")
            value = _number(raw)
            if raw.strip() in MISSING: excluded["missing_outcome"] += 1; continue
            if value is None: excluded["invalid_outcome"] += 1; continue
            parsed = {}; bad = False
            for name in settings:
                text = record.get(name, "").strip()
                try: number = float(text)
                except ValueError: number = None
                if number not in (0.0, 1.0) or not math.isfinite(number): bad = True; break
                parsed[name] = int(number)
            if bad:
                excluded["invalid_settings"] += 1
                if any(record.get(name, "").strip() in MISSING for name in settings):
                    excluded["incomplete_settings"] += 1
                continue
            eligible.append({"position": position, "value": value, "configuration": parsed})
    finite = []
    for _, recs, _ in []: pass
    # The reference is all finite observed outcomes, including excluded configurations.
    for rec in records:
        val = _number(rec.get(outcome, ""))
        if val is not None: finite.append(val)
    if not finite:
        return {"status": "no eligible recommendation", "eligible_count": 0, "excluded": excluded,
                "reference": None, "position_convention": "nonblank data rows start at 1; blank rows are ignored"}
    reference = statistics.median(finite)
    if not eligible:
        return {"status": "no eligible recommendation", "eligible_count": 0, "excluded": excluded,
                "reference": {"count": len(finite), "median": reference},
                "position_convention": "nonblank data rows start at 1; blank rows are ignored"}
    winner = min(eligible, key=lambda x: (x["value"], x["position"]))
    reduction = reference - winner["value"]
    return {"status": "ok", "selection_label": "retrospective selection from measured runs",
            "winner": winner, "eligible_count": len(eligible), "excluded": excluded,
            "reference": {"count": len(finite), "median": reference},
            "absolute_reduction": reduction,
            "percentage_reduction": reduction / reference * 100 if reference != 0 else None,
            "association": blame(path, outcome),
            "position_convention": "nonblank data rows start at 1; blank rows are ignored"}

def render_justify(result):
    lines = ["Justify recommended encoder configuration"]
    if result["status"] != "ok":
        lines.append("status: %s" % result["status"])
        if result.get("reason"): lines.append("reason: " + result["reason"])
        if result.get("reference"): lines.append("reference: count=%d median=%s" % (result["reference"]["count"], result["reference"]["median"]))
        lines.append("excluded: %s" % result.get("excluded", {})); lines.append(result.get("position_convention", "")); return "\n".join(lines)
    w = result["winner"]; ref = result["reference"]
    lines += [result["selection_label"], "position: %d; Performance-: %s" % (w["position"], w["value"]),
              "configuration: %s" % w["configuration"], "eligible_count: %d; excluded: %s" % (result["eligible_count"], result["excluded"]),
              "historical reference: finite observed Performance- median; count=%d; median=%s" % (ref["count"], ref["median"]),
              "absolute_reduction: %s" % result["absolute_reduction"]]
    lines.append("percentage_reduction: %s" % (result["percentage_reduction"] if result["percentage_reduction"] is not None else "undefined because reference is zero"))
    lines += ["Context uses global marginal associations from all eligible rows; they are not causal effects or local contributions and do not prove changing a setting will improve performance.",
              "Rule: choose the lowest finite Performance- among complete valid binary settings, breaking ties by earliest original nonblank row position.",
              "This is best only among eligible measured historical configurations and does not guarantee future performance.", result["position_convention"]]
    return "\n".join(lines)

def _binary_configuration(record, settings):
    parsed = {}
    for name in settings:
        value = _setting_key(record.get(name, ""))
        if value not in ("0", "1"):
            return None
        parsed[name] = int(value)
    return parsed

def _choose_pool(path, outcome="Performance-"):
    header, records, _ = load(path)
    if not header:
        return [], [], None
    if outcome not in header:
        raise ValueError("outcome column is absent")
    settings = [name for name in header if name != outcome]
    representatives = {}
    excluded = {"incomplete_or_invalid_settings": 0, "duplicate_rows": 0}
    for position, _header, record in iter_measurements(path):
        configuration = _binary_configuration(record, settings)
        if configuration is None:
            excluded["incomplete_or_invalid_settings"] += 1
            continue
        key = tuple(configuration[name] for name in settings)
        if key in representatives:
            excluded["duplicate_rows"] += 1
            continue
        representatives[key] = {"position": position, "configuration": configuration,
                                "raw_outcome": record.get(outcome, "")}
    return settings, list(representatives.values()), excluded

def historical_audit(path, outcome="Performance-"):
    """Retrospective replay over all rows; never used by choose."""
    settings, pool, excluded = _choose_pool(path, outcome)
    eligible = [item for item in pool if _number(item["raw_outcome"]) is not None]
    best = min(eligible, key=lambda item: (_number(item["raw_outcome"]), item["position"])) if eligible else None
    return {"eligible_count": len(eligible), "best": None if best is None else {
        "position": best["position"], "value": _number(best["raw_outcome"]),
        "configuration": dict(best["configuration"])}, "excluded": excluded,
        "label": "retrospective historical replay, not newly measured encoding or a guarantee about future performance"}

def choose(path, budget=20, seed=42, outcome="Performance-", audit=False):
    """Randomly evaluate distinct complete binary configurations through an oracle."""
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    settings, pool, excluded = _choose_pool(path, outcome)
    order = list(range(len(pool)))
    random.Random(seed).shuffle(order)
    selected = order[:min(budget, len(order))]
    history = []; successful = []
    for attempt, index in enumerate(selected, 1):
        item = pool[index]
        value = _number(item["raw_outcome"])
        entry = {"attempt": attempt, "position": item["position"],
                 "configuration": dict(item["configuration"]), "observed": value,
                 "outcome_status": "success" if value is not None else "invalid_or_missing"}
        history.append(entry)
        if value is not None:
            successful.append(entry)
    winner = min(successful, key=lambda item: (item["observed"], item["position"])) if successful else None
    result = {"status": "ok" if winner else "no successful evaluations", "strategy": "seeded random permutation without replacement",
              "seed": seed, "budget": budget, "candidate_count": len(pool),
              "attempted_evaluations": len(selected), "successful_evaluations": len(successful),
              "settings": settings, "winner": None if winner is None else dict(winner),
              "evaluation_history": history, "duplicate_policy": "earliest complete row represents a normalized configuration; later duplicates are not evaluated",
              "excluded": excluded, "position_convention": "nonblank data rows start at 1; blank rows are ignored"}
    if audit:
        result["audit"] = historical_audit(path, outcome)
        if winner and result["audit"]["best"]:
            result["audit"]["winner_gap"] = winner["observed"] - result["audit"]["best"]["value"]
    return result

def render_choose(result):
    lines = ["Budgeted X264 configuration choice", "status: %s" % result["status"],
             "strategy: %s; seed=%s; budget=%d" % (result["strategy"], result["seed"], result["budget"]),
             "candidate_count=%d; attempted_evaluations=%d; successful_evaluations=%d" % (result["candidate_count"], result["attempted_evaluations"], result["successful_evaluations"])]
    if result["winner"] is None: lines.append("winner: unavailable")
    else:
        w = result["winner"]; lines.append("winner: position=%d; Performance-=%s; configuration=%s" % (w["position"], w["observed"], w["configuration"]))
    lines.append("evaluation_history: %s" % result["evaluation_history"])
    lines.append("duplicate_policy: " + result["duplicate_policy"])
    if "audit" in result: lines.append("audit: " + result["audit"]["label"] + "; result=%s" % result["audit"])
    return "\n".join(lines)

def main(argv):
    if len(argv) not in (2, 3):
        print("usage: x264_summary.py DATASET [predict|certify|flow|blame|justify]", file=sys.stderr); return 2
    try:
        if len(argv) == 2: print(render(summarize(argv[1])))
        elif argv[2] == "predict":
            result = evaluate(argv[1]); config = {n: "0" for n in result["settings"]}; print(render_prediction(result, config))
        elif argv[2] == "certify": print(render_certify(certify(argv[1], 5)))
        elif argv[2] == "flow": print(render_flow(flow(argv[1])))
        elif argv[2] == "blame": print(render_blame(blame(argv[1])))
        elif argv[2] == "justify": print(render_justify(justify(argv[1])))
        elif argv[2] == "choose": print(render_choose(choose(argv[1], 20, 42, audit=True)))
        else: print("usage: x264_summary.py DATASET [predict|certify|flow|blame]", file=sys.stderr); return 2
        return 0
    except (OSError, csv.Error) as exc:
        print("error: %s" % exc, file=sys.stderr); return 1

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
