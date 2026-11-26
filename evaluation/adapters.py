"""One small adapter per saved program: train on the shared train.csv, predict every test row.

Each adapter only calls the program's own public functions; the generated code is not edited.
Returns a list of floats (guess) aligned with ctx.test_rows.
"""
guess, race = {}, {}


def each(ctx, f):
    return [f(row) for row in ctx.test_rows]


def reg(table, t, tr, d):
    def deco(fn):
        table[(t, tr, d)] = fn
        return fn
    return deco


# ---------------- T7 (human team shared core) ----------------
for _d in ("x264", "nasa93dem", "health_closedissues"):
    @reg(guess, 7, 1, _d)
    def _t7(ctx):
        from src.core.datasets import get_dataset
        from src.core.guess import predict_row
        spec = get_dataset(ctx.d)
        return each(ctx, lambda r: predict_row(spec, ctx.train_rows, r)[ctx.target])

# ---------------- T1 (plain Python, no reuse) ----------------
@reg(guess, 1, 1, "x264")
def _(ctx):
    m = ctx.mod("guess")
    return each(ctx, lambda r: ctx.pick(m.predict(ctx.train_path, ctx.config(r)), "prediction", "predicted_time", "Performance-"))

@reg(guess, 1, 1, "nasa93dem")
def _(ctx):
    m = ctx.mod("predict_effort")
    model = m.fit(m.read_projects(ctx.train_path))
    test = m.read_projects(ctx.test_path)
    return [ctx.pick(m.predict(model, r)) for r in test]

@reg(guess, 1, 1, "health_closedissues")
def _(ctx):
    m = ctx.mod("health_closedissues")
    rows = m.read_rows(ctx.train_path)
    return each(ctx, lambda r: ctx.pick(m.predict(rows, ctx.config(r)), "MRE-", "mre"))

@reg(guess, 1, 2, "x264")
def _(ctx):
    m = ctx.mod("x264_time_predictor")
    feats, recs = m.load_dataset(ctx.train_path)
    return each(ctx, lambda r: ctx.pick(m.predict(feats, recs, ctx.config(r)), "prediction", "predicted_time", "Performance-"))

@reg(guess, 1, 2, "nasa93dem")
def _(ctx):
    m = ctx.mod("estimate_effort")
    rows = m.load_rows(ctx.train_path)
    return each(ctx, lambda r: ctx.pick(m.estimate(rows, ctx.config(r)), "estimate", "effort", "Effort-"))

@reg(guess, 1, 2, "health_closedissues")
def _(ctx):
    m = ctx.mod("predictor")
    p = m.ConfigurationPredictor(m.load(ctx.train_path))
    return each(ctx, lambda r: ctx.pick(p.predict(ctx.config(r)), "MRE-", "mre"))

@reg(guess, 1, 3, "x264")
def _(ctx):
    m = ctx.mod("predictor")
    feats, data = m.load_measurements(ctx.train_path)
    p = m.EncodingTimePredictor(feats, data)
    return each(ctx, lambda r: ctx.pick(p.predict(ctx.config(r))))

@reg(guess, 1, 3, "nasa93dem")
def _(ctx):
    m = ctx.mod("effort_estimator")
    obs = m.load_observations(ctx.train_path)
    return each(ctx, lambda r: ctx.pick(m.estimate_effort(float(r["Kloc+"]), obs), "effort", "estimate", "Effort-"))

@reg(guess, 1, 3, "health_closedissues")
def _(ctx):
    m = ctx.mod("guess")
    rows = m.load_rows(ctx.train_path)
    return each(ctx, lambda r: ctx.pick(m.predict_from_rows(rows, ctx.config(r)), "MRE-", "mre"))

# ---------------- T2 (plain Python, reuse) ----------------
@reg(guess, 2, 1, "x264")
def _(ctx):
    from pathlib import Path
    m = ctx.mod("x264_summary")
    return each(ctx, lambda r: ctx.pick(m.predict_encoding_time(Path(ctx.train_path), ctx.config(r))))

@reg(guess, 2, 1, "nasa93dem")
def _(ctx):
    m = ctx.mod("nasa93_summary")
    return each(ctx, lambda r: ctx.pick(m.estimate_effort(ctx.train_path, float(r["Kloc+"]))))

@reg(guess, 2, 1, "health_closedissues")
def _(ctx):
    m = ctx.mod("model_quality_summary")
    return each(ctx, lambda r: ctx.pick(m.predict(ctx.train_path, ctx.config(r)), "MRE-"))

@reg(guess, 2, 2, "x264")
def _(ctx):
    m = ctx.mod("x264_summary")
    ds = m.load_dataset(ctx.train_path)
    return each(ctx, lambda r: ctx.pick(m.predict(ds, ctx.config(r))))

@reg(guess, 2, 2, "nasa93dem")
def _(ctx):
    est = ctx.imp("project_summary.estimate"); ld = ctx.imp("project_summary.loader")
    projects = ld.load_projects(ctx.train_path)
    return each(ctx, lambda r: ctx.pick(est.estimate_effort(projects, float(r["Kloc+"]))))

@reg(guess, 2, 2, "health_closedissues")
def _(ctx):
    m = ctx.mod("model_summary")
    cols = list(ctx.config(ctx.train_rows[0]))
    return each(ctx, lambda r: ctx.pick(m.predict_quality(ctx.train_rows, ctx.config(r), configuration_columns=cols), "MRE-"))

@reg(guess, 2, 3, "x264")
def _(ctx):
    m = ctx.imp("x264_summary.predict")
    return each(ctx, lambda r: ctx.pick(m.predict_encoding_time(ctx.train_path, ctx.config(r), measure="Performance-"), "value"))

@reg(guess, 2, 3, "nasa93dem")
def _(ctx):
    est = ctx.imp("project_history.estimator"); rep = ctx.imp("project_history.repository")
    projects = list(rep.CsvProjectRepository(ctx.train_path).projects())
    return each(ctx, lambda r: ctx.pick(est.estimate_effort(projects, float(r["Kloc+"]))))

@reg(guess, 2, 3, "health_closedissues")
def _(ctx):
    m = ctx.mod("model_summary")
    return each(ctx, lambda r: ctx.pick(m.predict_quality(ctx.train_path, ctx.config(r)), "MRE-"))

# ---------------- T3 (scikit-learn, no reuse) ----------------
@reg(guess, 3, 1, "x264")
def _(ctx):
    m = ctx.mod("predict_encoding_time")
    model, feats = m.fit_predictor(ctx.train_path)
    return each(ctx, lambda r: m.predict_configuration(model, feats, ctx.config(r, numeric=True)))

@reg(guess, 3, 1, "nasa93dem")
def _(ctx):
    m = ctx.mod("effort_estimator")
    return each(ctx, lambda r: ctx.pick(m.estimate(ctx.train_path, ctx.config(r))))

@reg(guess, 3, 1, "health_closedissues")
def _(ctx):
    m = ctx.mod("guess_model")
    x, y = m.read_data(ctx.train_path)
    model = m.build_model(); model.fit(x, y)
    tx, _ = m.read_data(ctx.test_path)
    i = list(m.TARGETS).index("MRE-")
    return [float(p[i]) for p in model.predict(tx)]

@reg(guess, 3, 2, "x264")
def _(ctx):
    m = ctx.mod("x264_guess")
    feats, X, y = m.load_dataset(ctx.train_path)
    g = m.X264GuessModel().fit(feats, X, y)
    return each(ctx, lambda r: ctx.pick(g.predict(ctx.config(r, numeric=True)), "prediction"))

@reg(guess, 3, 2, "nasa93dem")
def _(ctx):
    m = ctx.mod("nasa93dem_guess")
    rows, fields = m.load_dataset(ctx.train_path)
    model, _names = m.train_model(rows, fields)
    test, _ = m.load_dataset(ctx.test_path)
    return [m.predict_effort(model, r) for r in test]

@reg(guess, 3, 2, "health_closedissues")
def _(ctx):
    m = ctx.mod("health_closedissues_guess")
    feats, targs = m.load_dataset(ctx.train_path)
    g = m.build_model()
    g.pipeline.fit([[f[n] for n in g.feature_names] for f in feats], [[t[n] for n in g.target_names] for t in targs])
    tfeats, _ = m.load_dataset(ctx.test_path)
    return [p["MRE-"] for p in g.predict(tfeats)]

@reg(guess, 3, 3, "x264")
def _(ctx):
    m = ctx.mod("guess_skill")
    loaded = m.load_dataset(ctx.train_path)
    feats, X, y = loaded[:3] if len(loaded) >= 3 else loaded
    g = m.X264TimeGuesser(); g.fit(feats, X, y)
    return each(ctx, lambda r: ctx.pick(g.predict(ctx.config(r, numeric=True)), "prediction"))

@reg(guess, 3, 3, "nasa93dem")
def _(ctx):
    m = ctx.mod("guess_skill")
    loaded = m.load_dataset(ctx.train_path)
    ctx_probe = loaded
    feats, X, y = loaded[:3]
    g = m.EffortGuesser(); g.fit(feats, X, y)
    return each(ctx, lambda r: ctx.pick(g.predict({k: v for k, v in ctx.config(r, numeric=True, drop_x=False).items() if k in feats}), "prediction", "effort", "Effort-"))

@reg(guess, 3, 3, "health_closedissues")
def _(ctx):
    m = ctx.mod("guess_skill")
    model, _info = m.train_guess_model(ctx.train_path, test_size=1)
    i = list(m.TARGETS).index("MRE-")
    rows = [ctx.config(r) for r in ctx.test_rows]
    feats = getattr(model, "feature_names_")
    conv = []
    for r in rows:
        conv.append({k: (float(r[k]) if r[k].replace(".", "", 1).replace("-", "", 1).isdigit() else r[k]) for k in feats})
    return [float(p[i]) for p in m.predict(model, conv)]

# ---------------- T4 (scikit-learn, reuse) ----------------
@reg(guess, 4, 1, "x264")
def _(ctx):
    g = ctx.mod("guess"); rem = ctx.mod("remember")
    data = rem.load_measurements(ctx.train_path)
    model, report = g.evaluate(data)
    return each(ctx, lambda r: ctx.pick(g.predict_configuration(model, data, report, ctx.config(r, numeric=True)), "prediction", "Performance-", "predicted_performance"))

@reg(guess, 4, 1, "nasa93dem")
def _(ctx):
    m = ctx.mod("predict_effort")
    model = m.fit_training(m.read_projects(ctx.train_path))
    model = model[0] if isinstance(model, tuple) else model
    keys = list(m.PREDICTORS)
    return each(ctx, lambda r: m.estimate(model, {k: r.get(k) for k in keys}))

@reg(guess, 4, 1, "health_closedissues")
def _(ctx):
    m = ctx.mod("guess")
    _, rows = m.load_rows(ctx.train_path)
    models, _rep = m.fit_evaluate(rows)
    keys = list(m.CONFIGURATION)
    return each(ctx, lambda r: m.predict_configuration(models, {k: r.get(k) for k in keys})["MRE-"])

@reg(guess, 4, 2, "x264")
def _(ctx):
    import numpy as np
    m = ctx.mod("summarize_runs")
    settings, x, y = m.load_dataset(ctx.train_path)
    model = m.fit_predictor(x, y)
    X = np.asarray([[float(r[s]) for s in settings] for r in ctx.test_rows])
    return [float(v) for v in model.predict(X)]

@reg(guess, 4, 2, "nasa93dem")
def _(ctx):
    m = ctx.mod("nasa93_summary")
    model, predictors = m.fit_final_model(ctx.train_path)
    return each(ctx, lambda r: m.predict_proposal(model, predictors, {k: m.parse_value(r.get(k)) for k in predictors}))

@reg(guess, 4, 2, "health_closedissues")
def _(ctx):
    m = ctx.mod("predictor")
    models = m.fit_final(ctx.train_path)
    models = models[0] if isinstance(models, tuple) else models
    return each(ctx, lambda r: ctx.pick(m.predict_configuration(models, ctx.config(r, numeric=True)), "MRE-"))

@reg(guess, 4, 3, "x264")
def _(ctx):
    m = ctx.mod("summarize_x264")
    preds = m.predict(ctx.train_path, [ctx.config(r, numeric=True) for r in ctx.test_rows])
    return [ctx.pick(p, "prediction", "Performance-", "predicted_performance") for p in preds]

@reg(guess, 4, 3, "nasa93dem")
def _(ctx):
    m = ctx.mod("nasa93_effort")
    res = m.evaluate(ctx.train_path)
    model, names = res["model"], res["predictor_names"]
    return each(ctx, lambda r: m.predict_proposal(model, {k: ctx_num(r.get(k)) for k in names}, names))

@reg(guess, 4, 3, "health_closedissues")
def _(ctx):
    import numpy as np
    m = ctx.mod("quality_predict")
    rows = m._rows(ctx.train_path); X = m._frame(rows)
    y = m._target(rows, "MRE-"); ok = np.isfinite(y)
    model = m._pipeline().fit(X[ok], y[ok])
    return [float(v) for v in model.predict(m._frame([ctx.config(r, numeric=True) for r in ctx.test_rows]))]


def ctx_num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v

# ---------------- T5 (ezr, no reuse) ----------------
@reg(guess, 5, 1, "x264")
def _(ctx):
    m = ctx.mod("predictor")
    names, rows = m.read_csv(ctx.train_path)
    p = m.Predictor(names, rows)
    return each(ctx, lambda r: float(p.predict({n: float(r[n]) for n in names})))

@reg(guess, 5, 1, "nasa93dem")
def _(ctx):
    m = ctx.mod("estimator")
    header, raw = m.read_csv(ctx.train_path)
    pn = m.predictor_names(header); pi = [header.index(x) for x in pn]; yi = header.index("Effort-")
    est = m.CategoricalEstimator(pn); est.fit([[row[i] for i in pi] for row in raw], [row[yi] for row in raw])
    th, traw = m.read_csv(ctx.test_path)
    return [float(est.predict_one({x: row[th.index(x)] for x in pn})) for row in traw]

@reg(guess, 5, 1, "health_closedissues")
def _(ctx):
    m = ctx.mod("guess_model")
    g = m.GuessModel(m.load_csv(ctx.train_path), test_fraction=0.001)
    return each(ctx, lambda r: ctx.pick(g.predict({k: r[k] for k in m.PREDICTORS}), "MRE-"))

@reg(guess, 5, 2, "x264")
def _(ctx):
    m = ctx.mod("predictor")
    names, data = m.read_csv(ctx.train_path)
    model = m.Model(names).fit(data)
    return each(ctx, lambda r: float(model.predict({n: r[n] for n in names})))

@reg(guess, 5, 2, "nasa93dem")
def _(ctx):
    m = ctx.mod("estimator")
    rows = m.read_csv(ctx.train_path); preds = m.schema(rows)
    model = m.Estimator(preds).fit(m.valid_rows(rows))
    test = m.read_csv(ctx.test_path)
    return [float(v) for v in model.predict([{n: r.get(n) for n in preds} for r in test])]

@reg(guess, 5, 2, "health_closedissues")
def _(ctx):
    m = ctx.mod("predictor")
    g = m.ConfigPredictor(m.read_csv(ctx.train_path), test_fraction=0.001)
    return each(ctx, lambda r: ctx.pick(g.predict({k: r[k] for k in m.PREDICTORS}), "MRE-"))

@reg(guess, 5, 3, "x264")
def _(ctx):
    m = ctx.mod("predictor")
    names, rows = m.read_csv(ctx.train_path)
    model = m.RidgePredictor(names).fit(rows)
    return each(ctx, lambda r: float(model.predict_one({n: float(r[n]) for n in names})))

@reg(guess, 5, 3, "nasa93dem")
def _(ctx):
    m = ctx.mod("estimator")
    rows = m.load_csv(ctx.train_path); names = m.predictor_names(rows)
    model = m.CategoricalEstimator(names).fit(rows)
    test = m.load_csv(ctx.test_path)
    return [float(v) for v in model.predict([{k: r.get(k) for k in names} for r in test])]

@reg(guess, 5, 3, "health_closedissues")
def _(ctx):
    m = ctx.mod("guess_model")
    g = m.GuessModel(m.load_rows(ctx.train_path))  # fixed internal 80% fit; cannot be changed without editing code
    return each(ctx, lambda r: ctx.pick(g.predict({k: r[k] for k in m.PREDICTORS}), "MRE-"))

# ---------------- T6 (ezr, reuse) ----------------
@reg(guess, 6, 1, "x264")
def _(ctx):
    m = ctx.mod("x264_predict")
    names, rows = m._read(ctx.train_path)
    p = m.Predictor(names, rows)
    sample = rows[0][0]
    cast = (lambda v: v) if isinstance(sample[0], str) else (lambda v: type(sample[0])(float(v)))
    return each(ctx, lambda r: float(p.predict({nm: cast(r[nm]) for nm in names})))

@reg(guess, 6, 1, "nasa93dem")
def _(ctx):
    m = ctx.mod("nasa93_summary")
    header, rows, _ = m.read_dataset(ctx.train_path)
    preds = [n for n in header if not n.endswith("X") and n not in m.OUTCOMES]
    g = m.EffortGuesser(preds).fit(rows)
    _, test, _ = m.read_dataset(ctx.test_path)
    return [float(v) for v in g.predict(test)]

@reg(guess, 6, 1, "health_closedissues")
def _(ctx):
    m = ctx.mod("analyzer")
    _, rows = m.read_rows(ctx.train_path)
    cp = m.ConfigurationPredictor(test_fraction=0.001); cp.fit(rows, "MRE-")
    return each(ctx, lambda r: float(cp.predict({k: m.clean(r[k]) for k in m.CONFIG}, "MRE-")))

@reg(guess, 6, 2, "x264")
def _(ctx):
    m = ctx.mod("x264_summary")
    header, recs, _ = m.load(ctx.train_path)
    settings = [n for n in header if n != "Performance-"]
    model = m.NearestNeighborModel(settings, "Performance-").fit(recs)
    _, test, _ = m.load(ctx.test_path)
    return [float(model.predict_one(r)) for r in test]

@reg(guess, 6, 2, "nasa93dem")
def _(ctx):
    m = ctx.mod("nasa93_summary")
    projects = m.load(ctx.train_path)[1]
    model = m.CategoricalNeighborModel().fit(projects)
    test = m.load(ctx.test_path)[1]
    return [float(model.predict(p)) for p in test]

@reg(guess, 6, 2, "health_closedissues")
def _(ctx):
    m = ctx.mod("summarize_runs")
    model = m.evaluate(ctx.train_path)
    return each(ctx, lambda r: float(m.predict_configuration(model, {k: r[k].strip() for k in m.CONFIG})["MRE-"]))

@reg(guess, 6, 3, "x264")
def _(ctx):
    m = ctx.mod("summary")
    names, rows, _ = m.read_runs(ctx.train_path)
    settings = [n for n in names if n != "Performance-"]
    model = m.NearestMeanModel(settings)
    model.fit([{k: r[k] for k in settings} for r in rows], [float(r["Performance-"]) for r in rows])
    return each(ctx, lambda r: float(ctx.pick(model.predict_one({k: r[k] for k in settings}), "prediction", "Performance-")))

@reg(guess, 6, 3, "nasa93dem")
def _(ctx):
    m = ctx.mod("nasa_summary")
    model = m.CategoricalKNN(m.load(ctx.train_path)["projects"])
    test = m.load(ctx.test_path)["projects"]
    return [float(model.predict({c: p["values"].get(c, "") for c in m.PREDICTORS})) for p in test]

@reg(guess, 6, 3, "health_closedissues")
def _(ctx):
    m = ctx.mod("analyzer")
    # predict_dataset keeps no fitted model, so fit the program's own _Tree on all training rows
    _h, rows, _bad = m._read(ctx.train_path)
    usable = [r for _p, r in rows if m._number(r.get("MRE-")) is not None]
    model = m._Tree(usable, "MRE-")
    return each(ctx, lambda r: ctx.pick(m.predict_configuration({k: r[k] for k in m.CONFIGS}, model), "MRE-"))


# ======================= RACE (step 10) =======================
# x264 protocol: candidates = the 200 shared test configurations (outcome revealed only when evaluated),
# budget 20 evaluations, seeds 1..10. History = train.csv, used only by programs built to learn from history.
# Each adapter returns the TRUE Performance- of the configuration the program picks, one per seed.
import os as _os
BUDGET, SEEDS = int(_os.environ.get("RACE_BUDGET", "20")), list(range(1, 11))


def perf_of(ctx, obj):
    """True Performance- of a picked row/record, looked up from the pool when needed."""
    return ctx.pick(obj, "score", "Performance-", "best_score", "objective", "value", "y", "performance")


for _d in ("x264", "health_closedissues"):
    @reg(race, 7, 1, _d)
    def _t7r(ctx):
        from src.core.datasets import get_dataset
        from src.core.choose import choose
        spec = get_dataset(ctx.d)
        return [float(choose(spec, ctx.test_rows, budget=BUDGET, seed=s)["selected"][ctx.target]) for s in SEEDS]

@reg(race, 1, 1, "x264")
def _(ctx):
    m = ctx.mod("race")
    recs = m.load_records(ctx.test_path)
    return [m.race_search(recs, BUDGET, seed=s)[0].score for s in SEEDS]

@reg(race, 1, 2, "x264")
def _(ctx):
    m = ctx.mod("race_search")
    recs = m.load_records(ctx.test_path)
    return [m.race_search(recs, BUDGET, seed=s)[0].score for s in SEEDS]

@reg(race, 1, 3, "x264")
def _(ctx):
    m = ctx.mod("race_search")
    rows = m.load_rows(ctx.test_path)
    # program only accepts power-of-two budgets; use the largest one not above BUDGET (16 for 20, 8 for 10)
    b = 1 << (BUDGET.bit_length() - 1)
    return [m.race(rows, b, seed=s).best_score for s in SEEDS]

@reg(race, 2, 1, "x264")
def _(ctx):
    from pathlib import Path
    m = ctx.mod("x264_summary")
    return [m.compare_search_strategies(Path(ctx.test_path), budget=BUDGET, seed=s)["race"]["best_objective"] for s in SEEDS]

@reg(race, 2, 2, "x264")
def _(ctx):
    m = ctx.mod("x264_summary")
    # deterministic staged race with no seed option: one pick, repeated for every seed
    v = m.compare_search(m.load_dataset(ctx.test_path), BUDGET)["system"]["objective"]
    return [v] * len(SEEDS)

@reg(race, 2, 3, "x264")
def _(ctx):
    m = ctx.imp("x264_summary.race")
    return [m.compare_searches(ctx.test_path, "Performance-", BUDGET, seed=s).system.objective for s in SEEDS]

@reg(race, 3, 1, "x264")
def _(ctx):
    m = ctx.mod("race_search")
    xtr, ytr, _ = m.load_measurements(ctx.train_path); xte, yte, _ = m.load_measurements(ctx.test_path)
    return [m.model_guided_race(xtr, ytr, xte, yte, BUDGET, s)["best"] for s in SEEDS]

@reg(race, 3, 2, "x264")
def _(ctx):
    m = ctx.mod("race_x264")
    data = m.load_dataset(ctx.test_path)
    return [m.race(data, s, budget=BUDGET)[0] for s in SEEDS]

@reg(race, 3, 3, "x264")
def _(ctx):
    m = ctx.mod("x264_race")
    _f, Xtr, ytr = m.load_dataset(ctx.train_path); _f, Xte, yte = m.load_dataset(ctx.test_path)
    return [min(yte[i] for i in m.race_select(Xtr, ytr, Xte, budget=BUDGET, seed=s)) for s in SEEDS]

def _shape(o, depth=0):
    """Compact structure dump for probing nested results."""
    if depth > 3: return "..."
    if isinstance(o, dict): return {k: _shape(v, depth + 1) for k, v in list(o.items())[:12]}
    if isinstance(o, (list, tuple)): return [_shape(o[0], depth + 1), f"len={len(o)}"] if o else []
    return o if isinstance(o, (int, float, str, bool)) or o is None else type(o).__name__

@reg(race, 4, 1, "x264")
def _(ctx):
    import tempfile
    m = ctx.mod("race")
    res = m.run_race(ctx.test_path, tempfile.mkdtemp(), budget=BUDGET, seeds=tuple(SEEDS))
    runs = sorted((r for r in res["runs"] if r["method"] == "step7_selector"), key=lambda r: r["seed"])
    return [float(r["selected_candidate"]["Performance-"]) for r in runs]

@reg(race, 4, 2, "x264")
def _(ctx):
    m = ctx.mod("summarize_runs")
    res = m.compare_searches(ctx.test_path, budget=BUDGET, seeds=SEEDS)
    return [res["observed_global_minimum"] + r["retained_gap"] for r in res["runs"]]

@reg(race, 4, 3, "x264")
def _(ctx):
    m = ctx.mod("summarize_x264")
    res = m.compare_search_strategies(ctx.test_path, budget=BUDGET, seeds=SEEDS)
    return [float(r["selector"]["recommendation"]["performance"]) for r in res["repetitions"]]

@reg(race, 5, 1, "x264")
def _(ctx):
    m = ctx.mod("tuner")
    results, _summary = m.compare(ctx.test_path, budget=BUDGET, seeds=SEEDS)
    return [float(r["measured"]) for r in sorted((r for r in results if r["strategy"] == "race"), key=lambda r: r["seed"])]

@reg(race, 5, 2, "x264")
def _(ctx):
    m = ctx.mod("race_compare")
    names, cands, records = m.compare(ctx.test_path, budget=BUDGET, seeds=SEEDS)
    return [float(rec[3][1]) for rec in records]  # (seed, ti, te, tuner_recommendation=(cfg, value), ...)

@reg(race, 5, 3, "x264")
def _(ctx):
    m = ctx.mod("tuner")
    res = m.compare(m.load_dataset(ctx.test_path), budget=BUDGET, seeds=SEEDS)
    return [float(a["recommendation"]["y"]) for a, _b, _gap, _cls in res]

@reg(race, 6, 2, "x264")
def _(ctx):
    m = ctx.mod("x264_summary")
    res = m.race(ctx.test_path, budget=BUDGET, seeds=SEEDS)
    if ctx.probe:
        return {"probe": [k for k in res["rows"][0]["selector"]["winner"]]}
    w = [r["selector"]["winner"] for r in res["rows"]]
    key = next(k for k in w[0] if k not in ("position", "configuration") and isinstance(w[0][k], (int, float)))
    return [float(x[key]) for x in w]

@reg(race, 6, 3, "x264")
def _(ctx):
    m = ctx.mod("summary")
    res = m.race(ctx.test_path, budget=BUDGET, seeds=SEEDS)
    return [float(p["selector"]["winner"]["value"]) for p in res["pairs"]]

@reg(race, 6, 1, "x264")
def _(ctx):
    m = ctx.mod("x264_race")
    cands, outs = m.load(ctx.test_path)
    lookup = {m._key(c): m._finite(y) for c, y in zip(cands, outs)}
    f = lambda c: lookup[m._key(c)]
    return [float(m.race(cands, f, f, BUDGET, s)["tuner"]["recommendation"]["outcome"]) for s in SEEDS]

# ---------------- Health race: history = train.csv; score = true MRE- of the recommended configuration ----------------
HCFG = ("N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")


def _norm(v):
    try:
        f = float(v)
        return repr(round(f, 9))
    except (TypeError, ValueError):
        return str(v).strip()


def mre_of(ctx, cfg):
    """Look up the true MRE- of a configuration in the shared pool (train.csv)."""
    if not hasattr(ctx, "_hidx"):
        ctx._hidx = {tuple(_norm(r[k]) for k in HCFG): float(r["MRE-"]) for r in ctx.train_rows}
    if hasattr(cfg, "values") and not isinstance(cfg, dict):
        cfg = cfg.values
    if isinstance(cfg, dict) and "MRE-" in cfg and cfg["MRE-"] not in (None, ""):
        return float(cfg["MRE-"])
    return ctx._hidx[tuple(_norm(cfg[k]) for k in HCFG)]


def deterministic(v):
    return [float(v)] * len(SEEDS)


@reg(race, 1, 1, "health_closedissues")
def _(ctx):
    # this step compares a streaming estimator against a baseline estimator; it never recommends a configuration
    return {"na": "no configuration recommended (estimator-vs-baseline comparison)"}

@reg(race, 1, 2, "health_closedissues")
def _(ctx):
    m = ctx.mod("selector")
    return deterministic(mre_of(ctx, m.race_select(m.read_records(ctx.train_path))))

@reg(race, 1, 3, "health_closedissues")
def _(ctx):
    m = ctx.mod("config_selector")
    out = m.select(m.read_rows(ctx.train_path))
    return deterministic(mre_of(ctx, out[2]))

@reg(race, 2, 1, "health_closedissues")
def _(ctx):
    m = ctx.mod("model_quality_summary")
    return deterministic(mre_of(ctx, m.choose(ctx.train_path)["configuration"]))

@reg(race, 2, 2, "health_closedissues")
def _(ctx):
    m = ctx.mod("model_summary")
    return deterministic(mre_of(ctx, m.choose_configuration(ctx.train_rows)["chosen"]["configuration"]))

@reg(race, 2, 3, "health_closedissues")
def _(ctx):
    m = ctx.mod("model_summary")
    return deterministic(mre_of(ctx, m.choose_configuration(ctx.train_path)["recommended_configuration"]))

@reg(race, 3, 1, "health_closedissues")
def _(ctx):
    # online block selector: reports mean ACC+ of its picks over 70 blocks, never one recommended configuration
    return {"na": "no single configuration recommended (online block-by-block selection scored on ACC+)"}

@reg(race, 3, 2, "health_closedissues")
def _(ctx):
    m = ctx.mod("race_skill")
    rows = m.load_dataset(ctx.train_path)
    res = m.evaluate(rows, train_end=6000, eval_end=len(rows))
    # selector trained on the first 6000 pool rows picks among the rest; mean_mre is its pick's measured MRE-
    return deterministic(res[0][0].mean_mre)

@reg(race, 3, 3, "health_closedissues")
def _(ctx):
    m = ctx.mod("race")
    rows = m.load_validate(ctx.train_path)
    records, summary = m.evaluate(rows)
    # 5-fold: selector picks on 4 folds of the pool, its pick is measured on the held-out fold
    return [float(r["MRE-"]) for r in records if r["method"] == "treatment3_selector"]

@reg(race, 4, 1, "health_closedissues")
def _(ctx):
    m = ctx.mod("race")
    rep = m.race(ctx.train_path, seeds=tuple(SEEDS))
    return deterministic(rep["candidate_record"]["observed_objectives"]["MRE-"])

@reg(race, 4, 2, "health_closedissues")
def _(ctx):
    m = ctx.mod("race")
    return deterministic(m.compare(ctx.train_path, seeds=tuple(SEEDS))["selector"]["outcomes"]["MRE-"])

@reg(race, 4, 3, "health_closedissues")
def _(ctx):
    m = ctx.mod("race")
    return [float(r["selector"]["measured_outcomes"]["MRE-"]) for r in m.compare(ctx.train_path, seeds=tuple(SEEDS))["runs"]]

@reg(race, 5, 1, "health_closedissues")
def _(ctx):
    m = ctx.mod("selector")
    records, _info = m.load_records(ctx.train_path)
    return deterministic(m.select(records)["outcomes"]["MRE-"])

@reg(race, 5, 2, "health_closedissues")
def _(ctx):
    m = ctx.mod("selector")
    return deterministic(mre_of(ctx, m.select(m.load(ctx.train_path))))

@reg(race, 5, 3, "health_closedissues")
def _(ctx):
    m = ctx.mod("selector")
    _rows, _valid, feasible = m.load(ctx.train_path)
    return deterministic(float(m.selector(feasible)[1]["MRE-"]))

@reg(race, 6, 1, "health_closedissues")
def _(ctx):
    m = ctx.mod("analyzer")
    _h, rows = m.read_rows(ctx.train_path)
    res = m.choose(rows)
    if ctx.probe: return {"probe": list(res)}
    rec = next(v for k, v in res.items() if isinstance(v, dict) and ("outcomes" in v or "MRE-" in v or "configuration" in v))
    return deterministic(rec["outcomes"]["MRE-"] if "outcomes" in rec else rec.get("MRE-", mre_of(ctx, rec.get("configuration", rec))))

@reg(race, 6, 2, "health_closedissues")
def _(ctx):
    m = ctx.mod("race")
    return [float(r["selector"]["outcomes"]["MRE-"]) for r in m.race(ctx.train_path, seeds=SEEDS)["results"]]

@reg(race, 6, 3, "health_closedissues")
def _(ctx):
    m = ctx.mod("analyzer")
    res = m.race(ctx.train_path, seeds=SEEDS)
    if ctx.probe: return {"probe": [k for k in res]}
    rows = next(v for v in res.values() if isinstance(v, list) and v and isinstance(v[0], dict) and "selector" in v[0])
    return [float(r["selector"]["outcomes"]["MRE-"]) if "outcomes" in r["selector"] else mre_of(ctx, r["selector"]) for r in rows]
