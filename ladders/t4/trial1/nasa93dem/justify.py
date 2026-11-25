"""Explain an effort estimate using training evidence and local model sensitivity."""
import argparse
import csv
import json
from pathlib import Path

from blame import diagnose
from predict_effort import estimate, evaluate, load_proposal, normalize_proposal
from summarize import PREDICTORS, numeric_summary, read_projects

METHOD = {
    'name': 'Training historical context plus one-field model sensitivity',
    'rationale': 'Directly query the unchanged fitted estimator; no surrogate, attribution fit or additional randomness.',
    'fit_scope': 'Reuse evaluate: grouped seed-42 partition; preprocessing and forest fit only on training rows.',
    'local': 'For each field, replace only that field by each observed nonmissing training category; keep all other normalized proposal values fixed.',
    'delta': 'hypothetical model estimate minus original proposal model estimate; not additive contributions, causal effects, or feasible changes',
    'historical': 'Training-only raw effort summaries and nominal Hamming nearest examples; equal weight per predictor, missing equals missing; ties by original row index.',
    'neighbors': 5,
    'global': 'Training-only Step 5 categorical eta squared; univariate marginal association, never a local prediction contribution.',
    'seed': 42,
    'randomness': 'No additional randomized procedure.',
}
LIMITATIONS = [
    'Physical effort units are unspecified. MAE is aggregate held-out error, not an interval for this proposal; no confidence or prediction interval is supplied.',
    'Historical observational associations and local model sensitivity establish neither causal savings nor operational feasibility. Sensitivities are not additive contributions.',
    'Small, potentially related historical projects and one grouped split limit generalization; identical inputs are grouped, but broader project dependencies may remain.',
    'Missing inputs use the existing explicit missing level; unseen categories encode as zeros. Either can place the proposal outside well-supported training combinations.',
    'Equal-weight nominal similarity is descriptive and does not establish project comparability; marginal category support does not establish joint support.',
    'Delivered size, defects, identifiers and other metadata are excluded. Proposal-time availability of supplied predictors must be verified.',
    'Training-based global associations can be confounded by project size and correlated characteristics; small categories and extreme efforts can dominate.',
    'Random forest averages historical training outcomes and may miss novel or extreme effort. Missing-target rows are excluded without imputation.',
    'An effort estimate alone does not justify approving a proposal.',
]
APPROVAL = [
    'Confirm scope, deliverables, project size, acceptance criteria and proposal-time predictor ratings with accountable owners.',
    'Specify effort units and conversion to cost, staffing capacity, schedule dependencies and a sponsor-approved budget.',
    'Review comparable projects, novelty, risks, alternatives, expected benefits and contingency requirements; validate on relevant recent projects.',
    'Obtain explicit sponsor acceptance of uncertainty and decision criteria before approval.',
]


def justify(rows, proposal, illustrative=False):
    """Return detached JSON-compatible evidence; held-out outcomes affect evaluation only."""
    values = normalize_proposal(proposal)
    normalized = dict(zip(PREDICTORS, values))
    model, evaluation = evaluate(rows)
    indices = evaluation['split']['training_indices']
    training = [rows[i] for i in indices]
    training_values = [normalize_proposal({p: row.get(p) for p in PREDICTORS}) for row in training]
    prediction = estimate(model, normalized)
    characteristics = []
    sensitivities = []
    for position, field in enumerate(PREDICTORS):
        category = values[position]
        levels = sorted({v[position] for v in training_values if v[position] is not None})
        matches = [row['Effort-'] for row, v in zip(training, training_values) if v[position] == category]
        characteristics.append({
            'field': field, 'supplied_key': field in proposal,
            'raw_value': proposal.get(field), 'normalized_value': category,
            'status': 'missing' if category is None else 'seen' if category in levels else 'unseen',
            'training_levels': levels,
            'training_missing_count': sum(v[position] is None for v in training_values),
            'matching_category_effort_summary': numeric_summary(matches),
        })
        alternatives = []
        for level in levels:
            if level == category:
                continue
            hypothetical = dict(normalized, **{field: level})
            predicted = estimate(model, hypothetical)
            alternatives.append({'replacement': level,
                                 'training_support': sum(v[position] == level for v in training_values),
                                 'estimated_effort': predicted, 'delta_from_proposal': predicted - prediction})
        sensitivities.append({'field': field, 'original': category, 'alternatives': alternatives})
    neighbors = []
    for index, row, v in zip(indices, training, training_values):
        mismatch = [p for p, a, b in zip(PREDICTORS, values, v) if a != b]
        neighbors.append({'row_index': index, 'mismatch_count': len(mismatch),
                          'mismatching_fields': mismatch, 'predictors': dict(zip(PREDICTORS, v)),
                          'observed_effort': row['Effort-']})
    neighbors.sort(key=lambda r: (r['mismatch_count'], r['row_index']))
    associations = diagnose(training)
    report = {
        'target': 'Effort-', 'units': 'unspecified',
        'proposal': {'label': 'Illustrative demo proposal; not an actual approval request' if illustrative else 'User-supplied proposal',
                     'normalized': normalized, 'characteristics': characteristics},
        'estimate': {'value': prediction, 'model': 'RandomForestRegressor',
                     'settings': evaluation['model_settings'],
                     'preprocessing': evaluation['preprocessing'],
                     'dependency_versions': evaluation['dependency_versions']},
        'method': METHOD,
        'training_context': {'row_indices': indices,
                             'effort_summary': numeric_summary([r['Effort-'] for r in training]),
                             'exact_configuration_count': sum(v == values for v in training_values),
                             'nearest_examples': neighbors[:METHOD['neighbors']]},
        'global_historical_associations': {
            'scope': 'Training rows only; global marginal associations, not causal effects or local additive contributions.',
            'method': associations['method'],
            'ranked_findings': associations['ranked_findings'],
            'unranked_factors': associations['unranked_factors']},
        'local_model_sensitivity': sensitivities,
        'evaluation': evaluation,
        'limitations': LIMITATIONS,
        'information_needed_before_approval': APPROVAL,
        'decision': 'Evidence supports review of an estimate, not automatic approval.',
    }
    report['readable_explanation'] = render_text(report)
    return report


def render_text(r):
    """All numerical statements are formatted from the structured evidence above."""
    e, context, evaluation = r['estimate'], r['training_context'], r['evaluation']
    lines = [r['proposal']['label'],
             f"Proposal model estimate: Effort- = {e['value']:.6f}; physical units unspecified.",
             f"Model: {e['model']}; settings: {json.dumps(e['settings'], sort_keys=True)}.",
             'Normalization, explicit missing level and one-hot encoding reuse the preserved estimator; all learned preprocessing uses training rows only.',
             'Proposal characteristics (missing includes omitted keys and normalized missing tokens):']
    for c in r['proposal']['characteristics']:
        s = c['matching_category_effort_summary']
        lines.append(f"  {c['field']}: raw={c['raw_value']!r}, normalized={c['normalized_value']!r}, status={c['status']}, supplied key={c['supplied_key']}; training category count={s['count']}, mean effort={s['mean']}.")
    s = context['effort_summary']
    lines.extend([
        f"Historical training context: n={s['count']}, mean={s['mean']}, median={s['median']}, range=[{s['min']}, {s['max']}]; exact normalized configuration count={context['exact_configuration_count']}.",
        'Nearest historical training examples (equal-weight nominal mismatch; comparability is not established):'])
    for n in context['nearest_examples']:
        lines.append(f"  Data row index={n['row_index']}, mismatched predictors={n['mismatch_count']}, observed effort={n['observed_effort']}.")
    lines.extend([
        f"Separate aggregate evaluation: held-out n={evaluation['split']['evaluation_samples']}; model MAE={evaluation['model_mae']:.6f}; training-mean baseline MAE={evaluation['baseline_mae']:.6f}; constant baseline prediction={evaluation['baseline']['prediction']:.6f}.",
        'These MAEs describe one held-out partition, not this proposal error or an uncertainty interval.',
        'Global historical associations (training only; no causal effects or additive contributions to the proposal prediction):'])
    for f in r['global_historical_associations']['ranked_findings']:
        lines.append(f"  {f['factor']}: eta squared={f['eta_squared']:.6f}; eligible n={f['eligible_effort_summary']['count']}.")
    lines.append('Local model sensitivity: each query changes one field, holding other proposal inputs fixed. These are model responses, not feasible plans or causal effort savings; deltas must not be summed.')
    for f in r['local_model_sensitivity']:
        for a in f['alternatives']:
            lines.append(f"  {f['field']}: {f['original']!r} -> {a['replacement']!r}: estimate={a['estimated_effort']:.6f}, delta={a['delta_from_proposal']:.6f}, training level support={a['training_support']}.")
    lines.append('Limitations:')
    lines.extend(r['limitations'])
    lines.append('Information needed before approval:')
    lines.extend(r['information_needed_before_approval'])
    lines.append(r['decision'])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--proposal', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--illustrative', action='store_true', help='Label the proposal as illustrative')
    args = parser.parse_args()
    try:
        report = justify(read_projects(args.dataset), load_proposal(args.proposal), args.illustrative)
        report['dataset_path'] = str(args.dataset.resolve())
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / 'justification.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        (args.output_dir / 'justification.txt').write_text(report['readable_explanation'], encoding='utf-8')
        print(report['readable_explanation'], end='')
    except (OSError, ValueError, OverflowError, csv.Error) as exc:
        parser.exit(2, f'Input error: {exc}\n')


if __name__ == '__main__':
    main()
