# Treatment 4 cross-domain proposal — SUPERSEDED

## Current execution rule (2026-10-28)

Do not use the cross-domain transfer proposal below for Treatment 4 runs.
Follow the user's selected rule, consistent with Treatment 2's saved prompts:
each dataset starts with a fresh Step 1; later steps reuse only earlier steps
from the same dataset, treatment, and trial. No generated code or regression
tests are carried from X264 into NASA93 or Health.

Verification found this prohibition in all 90 saved Treatment 2 prompts,
fresh-foundation instructions in all nine Step 1 prompts, and no explicit
other-domain source-path/import references in the saved source snapshots.
Saved prompts were standardized by a repository script, so this evidence
establishes the current artifact design, not the original session context.
The broader project-plan transfer requirement remains a separate unresolved
scope question; this note does not claim lecturer approval.

## Historical proposal (superseded; retained for audit)

Recorded on 2026-10-28, after Trial 1 X264 steps 1–10 and before NASA93 implementation.

## Decision and basis

For the remaining Treatment 4 work, use the domain-transfer design in
`project.md`: a shared implementation extended through domain adapters.
Its shared-architecture and second-domain milestones explicitly require reuse
across domains. This takes precedence for this workflow over existing prompt
templates that prohibit all cross-domain access. This is a documented protocol
interpretation, not evidence of lecturer approval or a preregistered decision.

## Allowed transfer

- NASA93 step 1 may read the completed X264 step 10 source and tests from the
  same treatment and trial. Later NASA93 steps receive their preceding NASA93
  step's accumulated source and tests.
- Health step 1 may receive the completed NASA93 step 10 library from the same
  treatment and trial. Later steps receive their preceding Health step.
- Carry library code and regression tests forward, preserving earlier behavior.
  Store a self-contained snapshot per step, with shared operations factored
  rather than maintaining separate copies of algorithms for each domain.
- Earlier domains' exact datasets may be supplied solely for regression tests
  and demos. Their saved outputs and metrics are not implementation inputs.
- Never transfer generated code between treatments or independent trials.

## Measurement and interpretation

At a domain's first step, count added/changed new-side application lines against
the preceding domain's final source snapshot, not against an empty directory.
At subsequent steps, compare against the preceding step. Record retained,
added, removed, and domain-adapter lines with the counting convention.
Report total carried library LOC separately from the new domain-specific LOC.

Preserve original prompts, results, failures, and measurement conventions.
Normalize differing metric encodings only in separately documented analysis.
If other treatments followed a different cross-domain rule, disclose the
protocol difference; do not attribute all resulting differences to library
choice or reuse instruction alone.
