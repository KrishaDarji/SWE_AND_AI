"""Retrospective constrained selection of one recorded NASA93 project."""
import argparse
import csv
import json
import math
from pathlib import Path

from predict_effort import normalize_proposal
from summarize import METADATA, OBJECTIVES, PREDICTORS, numeric_summary, read_projects

RULE = ('Among recorded rows with both observed Effort- and Kloc+, require Effort- <= budget; '
        'maximize Kloc+, then minimize Effort-, then choose the earliest one-based CSV data-row number.')
LIMITATIONS = [
    'The supplied budget is illustrative in unspecified effort units, not a monetary budget or sponsor-approved threshold.',
    'Kloc+ is delivered code size, an illustrative value proxy, not demonstrated business value. More code is not necessarily better for a real client.',
    'Historical outcomes illustrate constrained selection; they do not prove a proposed project will satisfy the budget.',
    'Select one recorded project, not a portfolio or an unseen combination of categorical settings.',
    'No proposal-time prediction is performed. Delivered Kloc+, Effort-, Defects- and X metadata are never proposal-time predictor inputs.',
    'Rows missing either required selection outcome are excluded, never imputed. Missing defects do not exclude a candidate.',
    'This demonstration has no objective-evaluation budget and does not perform budgeted benchmark search.',
    'Feasibility uses an inclusive comparison of parsed finite numeric effort with the budget, with no tolerance or unit conversion.',
]


def nonnegative_number(value, name):
    """API values must be finite nonnegative Python numbers, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite nonnegative number')
    try:
        valid = math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f'{name} must be a finite nonnegative number')
    return float(value)


def choose(rows, budget):
    """Select among parsed rows in original CSV order; an empty iterable is valid."""
    budget = nonnegative_number(budget, 'budget')
    rows = list(rows)
    candidates = []
    excluded = []
    missing_counts = dict.fromkeys(['Effort-', 'Kloc+'], 0)
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f'Data row {number}: expected a project mapping')
        outcomes = {}
        missing = []
        for field in missing_counts:
            value = row.get(field)
            if value is None:
                missing.append(field)
                missing_counts[field] += 1
            else:
                outcomes[field] = nonnegative_number(value, f'Data row {number} {field}')
        if missing:
            excluded.append({'csv_data_row': number, 'missing_required_outcomes': missing})
            continue
        candidates.append({'csv_data_row': number, **outcomes,
                           'feasible': outcomes['Effort-'] <= budget})
    feasible = [c for c in candidates if c['feasible']]
    winner = min(feasible, key=lambda c: (-c['Kloc+'], c['Effort-'], c['csv_data_row'])) if feasible else None
    selected = None
    if winner is not None:
        row = rows[winner['csv_data_row'] - 1]
        # Only known categorical predictors reach proposal normalization.
        characteristics = dict(zip(PREDICTORS, normalize_proposal({p: row.get(p) for p in PREDICTORS})))
        selected = {
            'csv_data_row': winner['csv_data_row'],
            'characteristics': characteristics,
            'observed_outcomes': {field: row.get(field) for field in OBJECTIVES},
            'metadata': {field: row.get(field) for field in METADATA},
            'feasibility': {'observed_effort': winner['Effort-'], 'budget': budget,
                            'comparison': 'Effort- <= budget', 'satisfied': True,
                            'remaining_effort_budget': budget - winner['Effort-']},
        }
    report = {
        'status': 'selected' if winner is not None else 'no_feasible_candidate',
        'budget': budget, 'budget_units': 'unspecified effort units',
        'selection_rule': RULE,
        'identity_convention': 'One-based CSV data-row number, excluding header; idX is metadata and need not be unique.',
        'counts': {'total_rows': len(rows), 'eligible_rows': len(candidates),
                   'excluded_missing_required_outcomes': len(excluded),
                   'missing_by_required_outcome': missing_counts,
                   'feasible_candidates': len(feasible),
                   'over_budget_candidates': len(candidates) - len(feasible)},
        'selected_project': selected,
        'candidate_evidence': candidates, 'excluded_rows': excluded,
        'feasible_outcome_summaries': {
            field: numeric_summary([c[field] for c in feasible]) for field in ('Effort-', 'Kloc+')},
        'limitations': LIMITATIONS,
    }
    report['readable_explanation'] = render_text(report)
    return report


def render_text(report):
    """Render the structured decision and its caveats without fitting a model."""
    lines = [f"Illustrative effort budget: {report['budget']:g} (unspecified effort units).",
             report['selection_rule'], report['identity_convention'],
             'Candidate counts: ' + json.dumps(report['counts'], sort_keys=True)]
    selected = report['selected_project']
    if selected is None:
        lines.append('No feasible candidate: no recorded row has both required outcomes and Effort- <= budget.')
    else:
        lines.extend([f"Selected CSV data row: {selected['csv_data_row']}.",
                      'Recorded characteristics: ' + json.dumps(selected['characteristics'], sort_keys=True),
                      'Observed outcomes: ' + json.dumps(selected['observed_outcomes'], sort_keys=True),
                      'Metadata (not identity or predictor inputs): ' + json.dumps(selected['metadata'], sort_keys=True),
                      'Feasibility evidence: ' + json.dumps(selected['feasibility'], sort_keys=True)])
    lines.extend(report['limitations'])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--budget', required=True, type=float)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        budget = nonnegative_number(args.budget, 'budget')
        report = choose(read_projects(args.dataset), budget)
        report['dataset_path'] = str(args.dataset.resolve())
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(report['readable_explanation'], end='')
    except (OSError, ValueError, OverflowError, csv.Error) as exc:
        parser.exit(2, f'Input error: {exc}\n')


if __name__ == '__main__':
    main()
