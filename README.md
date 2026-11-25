# Reconstructed ladder history

This branch is rebuilt by `scripts/build_ladder_tags.py` on the main branch. Each commit is
one ladder step, copied from the snapshot the experiment saved at that step
(`experiments/runs/.../stepNN/source`). The steps were not committed one at a time while the
experiment ran, so this is a reconstruction from saved artifacts, dated when it was built.

Tags: `t<T>/trial<R>/<domain>/stepNN` (LLM treatments 1-6) and `t7/<domain>/stepNN` (human
team, one continuous ladder across the three domains, with the shared codebase accumulating). Compare two steps with
`git diff --stat <tag> <tag>`; `scripts/tag_histogram.py` turns this into the histograms.
