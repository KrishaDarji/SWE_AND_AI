# Pinned course EZR background

This directory supplies the course's Lua source as reference material for
generating Python implementations in Treatments 5 and 6. This interpretation was
confirmed by the user in the master chat; it is not a claim of lecturer approval.
The initial Lua application runs (Trial 1 X264 Steps 1–5) were removed at the
user's request before restarting Treatment 5 in Python.

- Repository: https://github.com/txt/seai26f
- Commit: `a73453bcfcbe9a34539b6357a4cb282abc2b3b17`
- Upstream path: `src/ezr-lua/`
- Local snapshot: `upstream/` (unmodified, including bundled examples and data)
- License: copied from the repository root to `LICENSE.md`
- Provenance and SHA-256 file checksums: `provenance.json`
- Local validation commands and actual outputs: `validation.json`

Use this same snapshot throughout both treatments and all trials. Do not run the
upstream installer or update these files during the experiment: those commands
fetch a moving branch. The commit, rather than the rockspec's version label,
identifies this background precisely.

## Reference validation

The existing `/opt/homebrew/bin/lua` reports Lua 5.5.1. No interpreter or package
installation was needed. The module import and numeric-summary smoke check
passed. The supplied week 1 suite completed with `failures: 0`:

```sh
# Run from the SWE_AND_AI repository root.
lua experiments/backgrounds/ezr/upstream/ezr-eg1.lua --all
```

This checks basic library compatibility, not every upstream algorithm or an
experiment implementation. No experiment steps were executed during setup.

For reference-library validation only, Lua's search path can be set from the
repository root:

```sh
LUA_PATH="$PWD/experiments/backgrounds/ezr/upstream/?.lua;;" lua path/to/application.lua
```

`ezr.lua` requires `ezr-lib.lua`; copying the former alone is insufficient for
executing the reference library. The restarted experiment applications and tests
must be Python, translating or adapting relevant reference machinery as needed.
Use Python's standard library unless a subsequent agreed protocol explicitly
permits additional dependencies; do not use scikit-learn as the EZR background.

## Experiment access

Keeping the complete folder is for provenance and setup validation. It does not
automatically authorize an execution LLM to read all its contents. Each prompt
must explicitly name its permitted background files. The core pair is
`upstream/ezr.lua` and `upstream/ezr-lib.lua`. Any additional library modules or
examples must be explicitly declared and made consistently available to both
treatments; do not silently broaden access between steps.

Treatment 5 implementations are independent: access to the supplied background
library does not authorize reading or copying prior generated steps, trials,
datasets, or other treatments. Treatment 6 uses the same background, with reuse
of generated work limited to the scope explicitly stated in its prompts.

The supplied Lua reference is excluded from generated application LOC metrics.
Any translated or adapted Python implementation must be saved with the step's
source and counted in application LOC. Treatment 5 must translate independently
for each step; it may not reuse previously generated Python implementations.
Keep this setup documentation and validation outside the treatment run directories.
