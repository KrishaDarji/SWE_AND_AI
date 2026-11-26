"""Rebuild the `ladders` branch and its step tags from the saved per-step snapshots.

The experiment saved every ladder step's source as a snapshot folder
(experiments/runs/treatment<T>/trials/trial<R>/<domain>/stepNN/source) but did not commit
each step separately. This script RECONSTRUCTS a step-by-step history from those
snapshots: one commit per step, tagged

    t<T>/trial<R>/<domain>/stepNN     for the LLM treatments 1-6 (each ladder separately)
    t7/<domain>/stepNN                for the human team (one continuous 30-step ladder:
                                      x264, then nasa93dem, then health_closedissues)

so that `git diff --stat <tag_a> <tag_b>` and `scripts/tag_histogram.py` derive the
new-code histograms from tags. Commits are dated when this script runs; they are a
reconstruction, not the original development history.

usage: python3 scripts/build_ladder_tags.py   (from the repo root; rewrites branch `ladders`)
"""
import shutil, subprocess, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments/runs"
DOMAINS = ("x264", "nasa93dem", "health_closedissues")
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def ladders():
    """Yield (directory inside the branch, [(tag, snapshot source dir, label)])."""
    for t in range(1, 7):
        for tr in (1, 2, 3):
            for d in DOMAINS:
                base = RUNS / f"treatment{t}/trials/trial{tr}/{d}"
                yield f"ladders/t{t}/trial{tr}/{d}", [
                    (f"t{t}/trial{tr}/{d}/step{s:02d}", base / f"step{s:02d}/source", f"treatment {t} trial {tr} {d} step {s}")
                    for s in range(1, 11)]
    steps = []
    for d in DOMAINS:
        base = RUNS / f"treatment7/trials/trial1/{d}"
        steps += [(f"t7/{d}/step{s:02d}", base / f"step{s:02d}/source", f"treatment 7 (human team) {d} step {s}")
                  for s in range(1, 11)]
    yield "ladders/t7", steps


README = """# Reconstructed ladder history

This branch is rebuilt by `scripts/build_ladder_tags.py` on the main branch. Each commit is
one ladder step, copied from the snapshot the experiment saved at that step
(`experiments/runs/.../stepNN/source`). The steps were not committed one at a time while the
experiment ran, so this is a reconstruction from saved artifacts, dated when it was built.

Tags: `t<T>/trial<R>/<domain>/stepNN` (LLM treatments 1-6) and `t7/<domain>/stepNN` (human
team, one continuous ladder across the three domains, with the shared codebase accumulating). Compare two steps with
`git diff --stat <tag> <tag>`; `scripts/tag_histogram.py` turns this into the histograms.
"""


def main():
    tmp = Path(tempfile.mkdtemp(prefix="ladders_"))
    git("worktree", "add", "--detach", str(tmp), "HEAD", cwd=ROOT)
    try:
        git("checkout", "--orphan", "ladders-new", cwd=tmp)
        git("rm", "-rfq", "--ignore-unmatch", ".", cwd=tmp)
        for p in tmp.iterdir():
            if p.name != ".git":
                shutil.rmtree(p) if p.is_dir() else p.unlink()
        (tmp / "README.md").write_text(README)
        git("add", "-A", cwd=tmp)
        git("commit", "-qm", "Reconstructed ladder history: README", cwd=tmp)
        tags = []
        for folder, steps in ladders():
            dest = tmp / folder
            for tag, src, label in steps:
                if not src.is_dir():
                    raise SystemExit(f"missing snapshot: {src}")
                # LLM ladders: the step's snapshot replaces the folder. Human team: one shared
                # codebase that only grows, so the snapshot is copied over the previous tree.
                if dest.exists() and not tag.startswith("t7/"):
                    shutil.rmtree(dest)
                shutil.copytree(src, dest, ignore=IGNORE, dirs_exist_ok=True)
                git("add", "-A", cwd=tmp)
                git("commit", "-q", "--allow-empty", "-m",
                    f"{label} (reconstructed from {src.relative_to(ROOT)})", cwd=tmp)
                tags.append((tag, git("rev-parse", "HEAD", cwd=tmp).strip()))
        for tag, sha in tags:
            git("tag", "-f", tag, sha, cwd=ROOT)
        git("branch", "-f", "ladders", "ladders-new", cwd=ROOT)
        print(f"branch ladders rebuilt: {len(tags)} step tags")
    finally:
        git("worktree", "remove", "--force", str(tmp), cwd=ROOT)
        subprocess.run(["git", "branch", "-D", "ladders-new"], cwd=ROOT, capture_output=True)


if __name__ == "__main__":
    main()
