"""Build every figure, table and in-text number in the report from data in the repo.

Inputs (all in the repo):
  results/tag_histogram.csv      new/total lines per step, derived from the step tags
  evaluation/results/*.csv       external metrics on the shared split
  results/cost.csv               tokens, time and API-price cost per treatment
  docs/requirements/*.md         the twenty customer asks per domain
Outputs: report/generated/*.tex (tables, macros) and report/generated/*.pdf (figures).

usage (from the repo root):
  python3 scripts/tag_histogram.py   # refresh the tag-derived histogram first
  python3 report/build_assets.py
"""
import csv
import re
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report/generated"
sys.path.insert(0, str(ROOT))
from src.core.certify import cliffs, same  # noqa: E402

DOMAINS = ("x264", "nasa93dem", "health_closedissues")
DNAME = {"x264": "x264", "nasa93dem": "NASA93", "health_closedissues": "Health"}
TNAME = {1: "T1 Python", 2: "T2 Python+reuse", 3: "T3 sklearn", 4: "T4 sklearn+reuse",
         5: "T5 ezr", 6: "T6 ezr+reuse", 7: "T7 human team"}
PAIRS = {"Python": (1, 2), "scikit-learn": (3, 4), "ezr": (5, 6)}
COLOR = {1: "#2F6FB0", 2: "#2F6FB0", 3: "#C2621A", 4: "#C2621A", 5: "#1B8A5A", 6: "#1B8A5A", 7: "#172033"}
macros = {}


def tex_escape(s):
    return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("$", r"\$").replace("#", r"\#").replace("_", r"\_"))


def write(name, text):
    (OUT / name).write_text(text)


def macro(name, value):
    macros[name] = value


# ---------------------------------------------------------------- tag data
def load_tags():
    rows = list(csv.DictReader(open(ROOT / "results/tag_histogram.csv")))
    for r in rows:
        r["new"], r["total"], r["step"] = int(r["new_lines"]), int(r["total_lines"]), int(r["step"])
        r["pct"] = float(r["pct_new"])
        parts = r["tag"].split("/")
        r["t"] = int(parts[0][1:])
        r["domain"] = parts[2] if r["t"] < 7 else parts[1]
        r["step"] = int(parts[-1][4:])  # step within its dataset (T7's tags run 1-30 as one ladder)
    return rows


def series(rows, t, key, domain=None, step=None):
    return [r[key] for r in rows if r["t"] == t and (domain is None or r["domain"] == domain)
            and (step is None or r["step"] == step)]


# ---------------------------------------------------------------- figures
def fig_histograms(rows):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    steps = range(1, 11)
    for t in range(1, 8):
        style = dict(color=COLOR[t], lw=1.8 if t == 7 else 1.4,
                     ls="-" if t in (2, 4, 6, 7) else (0, (4, 2)), marker="o" if t == 7 else None, ms=3)
        dom = "x264" if t == 7 else None
        pct = [st.median(series(rows, t, "pct", dom, s)) for s in steps]
        new = [st.median(series(rows, t, "new", dom, s)) for s in steps]
        axes[0].plot(steps, pct, label=TNAME[t], **style)
        axes[1].plot(steps, new, **style)
    axes[0].set_ylabel("% of code that is new")
    axes[1].set_ylabel("new lines in the step")
    for ax in axes:
        ax.set_xlabel("ladder step")
        ax.set_xticks(list(steps))
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(0, 105)
    axes[1].set_ylim(bottom=0)
    fig.legend(loc="lower center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "fig_histograms.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_transfer(rows):
    t7 = [r for r in rows if r["t"] == 7]
    fig, ax = plt.subplots(figsize=(7.2, 2.0))
    x = list(range(1, 31))
    ax.bar(x, [r["new"] for r in t7], color=COLOR[7], width=0.7)
    for i, d in enumerate(DOMAINS):
        ax.axvline(10 * i + 0.5, color="#999", lw=0.8, ls=":")
        ax.text(10 * i + 5.5, max(r["new"] for r in t7) * 0.92, DNAME[d], ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([str((i % 10) + 1) for i in range(30)], fontsize=6)
    ax.set_xlabel("ladder step, by dataset (human team, one shared codebase)")
    ax.set_ylabel("new lines")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_transfer.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- same tables
def table_reuse(rows):
    lines = []
    certified = 0
    for bg, (a, b) in PAIRS.items():
        for d in DOMAINS:
            x = [r["new"] for r in rows if r["t"] == a and r["domain"] == d and r["step"] >= 2]
            y = [r["new"] for r in rows if r["t"] == b and r["domain"] == d and r["step"] >= 2]
            diff = not same(x, y) and st.median(y) < st.median(x)
            certified += diff
            verdict = r"\textbf{reuse smaller}" if diff else ("tie" if same(x, y) else "reuse larger")
            lines.append(f"{bg} & {DNAME[d]} & {st.median(x):.0f} & {st.median(y):.0f} & {cliffs(x, y):.2f} & {verdict} \\\\")
    totals = {}
    for t in range(1, 7):
        per = {}
        for r in rows:
            if r["t"] == t:
                per.setdefault(r["tag"].rsplit("/", 1)[0], 0)
                per[r["tag"].rsplit("/", 1)[0]] += r["new"]
        totals[t] = st.median(per.values())
    for bg, (a, b) in PAIRS.items():
        key = re.sub(r"[^A-Za-z]", "", bg)
        macro(f"save{key}", f"{100 * (1 - totals[b] / totals[a]):.0f}")
        macro(f"total{key}NoReuse", f"{totals[a]:.0f}")
        macro(f"total{key}Reuse", f"{totals[b]:.0f}")
    macro("reuseCertified", str(certified))
    write("tab_reuse.tex", "\n".join([
        r"\begin{tabular}{llrrrl}", r"\toprule",
        r"Background & Dataset & No reuse & Reuse & Cliff's $\delta$ & \texttt{same} verdict \\", r"\midrule",
        *lines, r"\bottomrule", r"\end{tabular}"]))


def table_shrink(rows):
    lines = []
    flat = 0
    for t in range(1, 8):
        cells = []
        for d in DOMAINS:
            if t == 7:
                ladder = [r for r in rows if r["t"] == 7 and r["domain"] == d]
                e = [r["new"] for r in ladder if 2 <= r["step"] <= 5]
                l = [r["new"] for r in ladder if 7 <= r["step"] <= 10]
            else:
                e = [r["new"] for r in rows if r["t"] == t and r["domain"] == d and 2 <= r["step"] <= 5]
                l = [r["new"] for r in rows if r["t"] == t and r["domain"] == d and 7 <= r["step"] <= 10]
            me, ml = st.median(e), st.median(l)
            shrinks = (not same(e, l)) and ml < me
            if not shrinks:
                flat += 1
            ratio = f"{ml / me:.2f}" if me else "--"
            cells.append(f"{me:.0f}$\\to${ml:.0f} ({ratio}) {'$\\downarrow$' if shrinks else '$=$' if same(e, l) else '$\\uparrow$'}")
        lines.append(f"{TNAME[t]} & " + " & ".join(cells) + r" \\")
    macro("shrinkFlatCells", str(flat))
    write("tab_shrink.tex", "\n".join([
        r"\begin{tabular}{llll}", r"\toprule",
        r"Treatment & x264 & NASA93 & Health \\", r"\midrule",
        *lines, r"\bottomrule", r"\end{tabular}"]))


def transfer_macros(rows):
    t7 = [r for r in rows if r["t"] == 7]
    per = [sum(r["new"] for r in t7[i * 10:(i + 1) * 10]) for i in range(3)]
    macro("tSevenXtwo", str(per[0]))
    macro("tSevenNasa", str(per[1]))
    macro("tSevenHealth", str(per[2]))
    macro("tSevenFinal", str(t7[-1]["total"]))
    macro("tSevenNasaShare", f"{100 * per[1] / t7[19]['total']:.1f}")
    macro("tSevenHealthShare", f"{100 * per[2] / t7[29]['total']:.1f}")


# ---------------------------------------------------------------- external metrics
def table_guess():
    by = list(csv.DictReader(open(ROOT / "evaluation/results/guess_by_run.csv")))
    ranks = {(r["dataset"], r["who"]): r for r in csv.DictReader(open(ROOT / "evaluation/results/guess_ranks.csv"))}
    lines = []
    for t in range(1, 8):
        cells = []
        for d in DOMAINS:
            rs = [r for r in by if r["dataset"] == d and int(r["treatment"]) == t]
            mae = st.median(float(r["MAE"]) for r in rs)
            hit = st.median(float(r["hit"]) for r in rs)
            rk = ranks[(d, f"T{t}")]["rank"]
            fmt = f"{mae:.3f}" if d == "health_closedissues" else (f"{mae:.1f}" if mae < 100 else f"{mae:.0f}")
            cells.append(f"{fmt} & {hit:.2f} & {'\\textbf{0}' if rk == '0' else rk}")
        lines.append(f"{TNAME[t]} & " + " & ".join(cells) + r" \\")
    base = []
    for d in DOMAINS:
        b = ranks[(d, "mean baseline")]
        mae = float(b["mean_abs_err"])
        fmt = f"{mae:.3f}" if d == "health_closedissues" else (f"{mae:.1f}" if mae < 100 else f"{mae:.0f}")
        base.append(f"{fmt} & -- & {b['rank']}")
    lines.append(r"\midrule Guess the average & " + " & ".join(base) + r" \\")
    write("tab_guess.tex", "\n".join([
        r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
        r" & \multicolumn{3}{c}{x264 (seconds)} & \multicolumn{3}{c}{NASA93 (effort)} & \multicolumn{3}{c}{Health (MRE)} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r"Treatment & MAE & hit & rank & MAE & hit & rank & MAE & hit & rank \\", r"\midrule",
        *lines, r"\bottomrule", r"\end{tabular}"]))


def table_race():
    def load(name):
        by = list(csv.DictReader(open(ROOT / f"evaluation/results/{name}_by_run.csv")))
        rk = {(r["dataset"], r["who"]): r for r in csv.DictReader(open(ROOT / f"evaluation/results/{name}_ranks.csv"))}
        return by, rk
    b10, r10 = load("race_b10")
    b20, r20 = load("race")
    lines = []
    for t in range(1, 8):
        x = [r for r in b10 if r["dataset"] == "x264" and int(r["treatment"]) == t and r["status"] == "ok"]
        h = [r for r in b20 if r["dataset"] == "health_closedissues" and int(r["treatment"]) == t and r["status"] == "ok"]
        xr = st.median(float(r["median_regret"]) for r in x)
        xw = st.mean(float(r["win_rate_vs_random"]) for r in x)
        rk10 = r10[("x264", f"T{t}")]["rank"]
        rk20 = r20[("x264", f"T{t}")]["rank"]
        access = sorted({r["access"] for r in h})
        acc = "full history$^*$" if access == ["full history"] else ", ".join(access).replace("full history", "full history$^*$")
        hr = st.median(float(r["median_regret"]) for r in h) if h else float("nan")
        lines.append(f"{TNAME[t]} & {xr:.2f} & {100 * xw:.0f}\\% & {rk10} & {rk20} & {acc} & {hr:.2f} \\\\")
    rx = r10[("x264", "random search")]
    rh = r20[("health_closedissues", "random search")]
    lines.append(rf"\midrule Random search & {float(rx['median_regret']):.2f} & -- & {rx['rank']} & {r20[('x264', 'random search')]['rank']} & budget 20 & {float(rh['median_regret']):.2f} \\")
    write("tab_race.tex", "\n".join([
        r"\begin{tabular}{lrrrrlr}", r"\toprule",
        r" & \multicolumn{4}{c}{x264: tuner, 200 unseen configs} & \multicolumn{2}{c}{Health: selector} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}",
        r"Treatment & regret & beats random & rank@10 & rank@20 & how it picked & regret \\", r"\midrule",
        *lines, r"\bottomrule", r"\end{tabular}"]))


def table_cost():
    rows = list(csv.DictReader(open(ROOT / "results/cost.csv")))
    lines = []
    tot_tok = tot_h = tot_usd = 0.0
    for r in rows:
        if r["treatment"] == "setup":
            continue
        tok, h, usd = float(r["tokens_total"]), float(r["llm_hours"]), float(r["usd_api_price"])
        tot_tok += tok; tot_h += h; tot_usd += usd
        cached = 100 * float(r["cached_input_tokens"]) / float(r["input_tokens"])
        lines.append(f"{TNAME[int(r['treatment'])]} & {tok / 1e6:.0f}M & {cached:.0f}\\% & {h:.1f} & \\${usd:.2f} \\\\")
    setup = next(r for r in rows if r["treatment"] == "setup")
    lines.append(rf"\midrule All six LLM arms & {tot_tok / 1e6:.0f}M & & {tot_h:.1f} & \${tot_usd:.2f} \\")
    lines.append(rf"Setup and planning chat & {float(setup['tokens_total']) / 1e6:.0f}M & & {float(setup['llm_hours']):.1f} & \${float(setup['usd_api_price']):.2f} \\")
    macro("costRuns", f"{tot_usd:.0f}")
    macro("costSetup", f"{float(setup['usd_api_price']):.0f}")
    macro("costAll", f"{tot_usd + float(setup['usd_api_price']):.0f}")
    macro("tokensRuns", f"{tot_tok / 1e6:.0f}")
    macro("hoursRuns", f"{tot_h:.1f}")
    write("tab_cost.tex", "\n".join([
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Treatment & tokens & cached & LLM hours & \$ at API prices \\", r"\midrule",
        *lines, r"\bottomrule", r"\end{tabular}"]))


# ---------------------------------------------------------------- requirements
def table_requirements():
    files = {"x264": "x264.md", "nasa93dem": "nasa93dem.md", "health_closedissues": "health-closedissues.md"}
    for d, f in files.items():
        text = (ROOT / "docs/requirements" / f).read_text()
        client = re.search(r"\*\*Client:\*\*\s*(.+)", text).group(1).strip()
        asks = re.findall(r"^\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*(\w+)\s*\|\s*$", text, re.M)
        lines = [f"{n} & {tex_escape(a)} & \\texttt{{{s}}} \\\\" for n, a, s in asks]
        write(f"tab_req_{d}.tex", "\n".join([
            r"\begin{tabular}{@{}rp{0.78\linewidth}l@{}}", r"\toprule",
            rf"\multicolumn{{3}}{{@{{}}l}}{{\textbf{{{DNAME[d]}}} --- client: {tex_escape(client)}}} \\", r"\midrule",
            *lines, r"\bottomrule", r"\end{tabular}"]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif"})
    rows = load_tags()
    fig_histograms(rows)
    fig_transfer(rows)
    table_reuse(rows)
    table_shrink(rows)
    transfer_macros(rows)
    table_guess()
    table_race()
    table_cost()
    table_requirements()
    write("numbers.tex", "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
    print("wrote", ", ".join(sorted(p.name for p in OUT.iterdir())))
    print("\n".join(f"  {k} = {v}" for k, v in sorted(macros.items())))


if __name__ == "__main__":
    main()
