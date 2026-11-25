"""Recommend minimum categorical changes for strictly lower predicted effort."""
import argparse
import csv
import json
from pathlib import Path

from predict_effort import estimate, evaluate, load_proposal, normalize_proposal, split_projects
from summarize import PREDICTORS, read_projects

LIMITATIONS = [
    'The dataset supplies neither intervention costs nor a calibrated risk measure.',
    'Lower predicted Effort- is an illustrative planning proxy, not a probability of failure or measured risk.',
    'Change cost counts differing normalized categorical predictor fields, not monetary or operational cost; all fields have equal weight.',
    'Physical effort units are unspecified. Predicted reduction is not causal savings, calibrated risk reduction, or guaranteed budget compliance.',
    'Observed historical combinations do not establish operational feasibility for this proposal.',
    'Not all project characteristics are controllable process settings; a human must review controllability, feasibility, dependencies, and actual costs.',
    'Small observational training sample, confounding, prediction error, unseen categories, and model-selection optimism limit this suggestion.',
    'Search is exhaustive only within complete training predictor configurations, not all possible interventions; no global real-world cheapest change is established.',
    'Predictor availability at proposal time is assumed. No uncertainty bounds or prospective intervention validation are supplied.',
    'Training eligibility and grouped partition reuse the existing observed-target availability rule; held-out outcome values never rank candidates.',
]
TIE_RULE = 'Strictly lower predicted effort; minimize differing normalized fields, then predicted effort, then earliest one-based CSV data row. Exact comparisons, no tolerance.'


def complete_proposal(proposal):
    """Require every predictor to be present and nonmissing after normalization."""
    values = normalize_proposal(proposal)
    missing = [field for field, value in zip(PREDICTORS, values) if value is None]
    if missing:
        raise ValueError('Repair requires complete nonmissing predictors: ' + ', '.join(missing))
    return dict(zip(PREDICTORS, values))


def illustrative_proposal(rows):
    """Earliest training row; target values do not choose the demonstration proposal."""
    eligible = [i for i, row in enumerate(rows) if row.get('Effort-') is not None]
    train, _, _ = split_projects([rows[i] for i in eligible])
    index = min(eligible[i] for i in train)
    proposal = complete_proposal({field: rows[index].get(field) for field in PREDICTORS})
    return proposal, index + 1


def recommend(rows, proposal):
    """Fit the unchanged training model and search only its training configurations."""
    normalized = complete_proposal(proposal)
    model, evaluation = evaluate(rows)
    train = evaluation['split']['training_indices']
    original_prediction = estimate(model, normalized)
    unique = {}
    incomplete = 0
    for index in sorted(train):
        values = normalize_proposal({field: rows[index].get(field) for field in PREDICTORS})
        if any(value is None for value in values):
            incomplete += 1
            continue
        unique.setdefault(tuple(values), index + 1)
    candidates = []
    for values, number in unique.items():
        configuration = dict(zip(PREDICTORS, values))
        prediction = estimate(model, configuration)
        changes = [{'field': field, 'old': normalized[field], 'new': configuration[field]}
                   for field in PREDICTORS if normalized[field] != configuration[field]]
        candidates.append({'csv_data_row': number, 'predictors': configuration,
                           'predicted_effort': prediction, 'change_cost': len(changes),
                           'changed_fields': changes})
    improving = [candidate for candidate in candidates
                 if candidate['predicted_effort'] < original_prediction]
    winner = min(improving, key=lambda c: (c['change_cost'], c['predicted_effort'], c['csv_data_row'])) if improving else None
    return {
        'status': 'recommendation' if winner else 'no_improving_candidate',
        'target': 'Effort-', 'units': 'unspecified',
        'original_proposal': dict(proposal), 'normalized_proposal': normalized,
        'original_predicted_effort': original_prediction,
        'recommended_candidate': winner,
        'changed_fields': winner['changed_fields'] if winner else [],
        'change_cost': winner['change_cost'] if winner else None,
        'predicted_reduction': original_prediction - winner['predicted_effort'] if winner else None,
        'no_improvement_reason': None if winner else 'No complete training configuration has strictly lower predicted effort.',
        'search': {
            'scope': 'Complete predictor configurations observed in training rows only; duplicates collapsed to earliest CSV data row.',
            'training_rows': len(train), 'excluded_incomplete_training_rows': incomplete,
            'complete_training_rows': len(train) - incomplete,
            'duplicate_complete_rows': len(train) - incomplete - len(unique),
            'unique_candidates_evaluated': len(candidates), 'improving_candidates': len(improving),
            'training_csv_data_rows': sorted(i + 1 for i in train),
            'held_out_rows_excluded': evaluation['split']['evaluation_samples'],
            'tie_rule': TIE_RULE, 'change_cost_definition': 'Number of differing normalized categorical predictor fields',
        },
        'model': {'implementation': 'Unchanged predict_effort.evaluate, fit_training, normalization and estimate',
                  'seed': 42, 'fit_scope': 'training only; no full-data refit',
                  'partition': 'Existing seed-42 grouped partition; identical normalized predictors stay together',
                  'dependency_versions': evaluation['dependency_versions']},
        'human_review_required': True, 'limitations': LIMITATIONS,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--proposal', type=Path, help='Complete JSON object with allowed predictor fields only')
    mode.add_argument('--demo', action='store_true', help='Use predictors of earliest training CSV data row')
    parser.add_argument('--example-proposal', type=Path, help='Save demo proposal JSON (requires --demo)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.example_proposal and not args.demo:
        parser.error('--example-proposal requires --demo')
    try:
        rows = read_projects(args.dataset)
        if args.demo:
            proposal, number = illustrative_proposal(rows)
        else:
            proposal = load_proposal(args.proposal)
        report = recommend(rows, proposal)
        report['dataset_path'] = str(args.dataset.resolve())
        if args.demo:
            report['demo'] = {'illustrative': True, 'proposal_csv_data_row': number,
                              'selection': 'Predictors of earliest training data row; observed effort did not select proposal or rank repairs.'}
            if args.example_proposal:
                args.example_proposal.write_text(json.dumps(proposal, indent=2) + '\n', encoding='utf-8')
        text = json.dumps(report, indent=2, allow_nan=False) + '\n'
        args.output.write_text(text, encoding='utf-8')
        print(text, end='')
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f'Input error: {exc}\n')


if __name__ == '__main__':
    main()
