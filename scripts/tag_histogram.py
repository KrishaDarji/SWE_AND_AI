"""Shrinking-code histograms derived from the step tags (see scripts/build_ladder_tags.py).

For each ladder and step: new lines = non-blank lines added since the previous step's tag
(`git diff -U0`), total = non-blank lines in the step's tree, % new = new / total.
Python source only (*.py). Writes results/tag_histogram.csv and prints the median
% new code per step for each treatment.
usage: python3 scripts/tag_histogram.py
"""
import csv, statistics as st, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_ladder_tags import ladders  # noqa: E402


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def total_lines(tag, folder):
    n = 0
    for path in git("ls-tree", "-r", "--name-only", tag, "--", folder).split():
        if path.endswith(".py"):
            n += sum(1 for line in git("show", f"{tag}:{path}").splitlines() if line.strip())
    return n


def added_lines(prev, tag, folder):
    """Non-blank lines added since the previous step (same counting rule as total_lines)."""
    if not prev:
        return total_lines(tag, folder)
    out = git("diff", "-U0", "--no-color", prev, tag, "--", f"{folder}/*.py", f"{folder}/**/*.py")
    return sum(1 for l in out.splitlines() if l.startswith("+") and not l.startswith("+++") and l[1:].strip())


def main():
    rows = []
    for folder, steps in ladders():
        prev = None
        for i, (tag, _src, _label) in enumerate(steps):
            tot = total_lines(tag, folder)
            new = added_lines(prev, tag, folder)
            t = tag.split("/")[0]
            rows.append({"tag": tag, "treatment": t, "step": i + 1, "new_lines": new, "total_lines": tot,
                         "pct_new": round(100 * new / tot, 1) if tot else 0.0})
            prev = tag
    out = ROOT / "results/tag_histogram.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("median % new code per step (t7 = one 30-step ladder across x264, nasa93dem, health)")
    for t in sorted({r["treatment"] for r in rows}):
        n = max(r["step"] for r in rows if r["treatment"] == t)
        med = [st.median(r["pct_new"] for r in rows if r["treatment"] == t and r["step"] == s) for s in range(1, n + 1)]
        print(t, " ".join(f"{m:5.1f}" for m in med))
    print(f"wrote {out.relative_to(ROOT)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
