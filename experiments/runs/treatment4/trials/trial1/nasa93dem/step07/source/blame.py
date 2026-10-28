"""Exploratory categorical associations with observed effort; no causal attribution."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from predict_effort import normalize_proposal
from summarize import PREDICTORS, numeric_summary, read_projects

LIMITATIONS = [
    'Exploratory descriptive ranking, not significance testing, certified wins, or causal attribution. No p-values or confidence guarantees are supplied.',
    'Higher raw effort is not automatically excessive for a large project. No budget, expected-effort threshold, or size-adjusted excess is defined.',
    'Kloc+ is excluded from ranking and no size adjustment is performed. Project-size differences can explain category associations.',
    'Correlated factors can share the same association; univariate scores do not isolate independent contributions or interactions.',
    'Observational confounding, related projects, selection bias and historical context preclude claims that changing a factor causes effort savings.',
    'Small categories (fewer than five observed efforts) are flagged; their summaries and scores may be unstable. Five is a descriptive warning threshold, not a reliability guarantee.',
    'Eta squared is in-sample, sensitive to extreme efforts and biased upward with more categories, especially singletons; it is not predictive feature importance.',
    'Missing effort is excluded, never imputed. Missing predictor categories are summarized separately and excluded from that factor score; feature-specific samples can differ and missingness may be informative.',
    'Constant factors cannot support between-category comparisons. Zero effort variance or fewer than two observed categories makes association strength undefined.',
    'Physical effort units are unspecified. Category labels remain nominal; no ordinal mapping or favorable direction is inferred from their names.',
    'All data describe historical associations. No fitted prediction model, learned preprocessing, held-out attribution, or future-performance guarantee is used.',
]
METHOD = {
    'name': 'Univariate categorical eta squared on raw observed Effort-',
    'formula': 'eta2 = sum_g n_g * (mean_g - mean_all)^2 / sum_i (effort_i - mean_all)^2',
    'interpretation': 'Fraction of within-sample raw effort variation between observed categories, ranging from 0 to 1; strength has no sign.',
    'eligible_rows': 'For each factor: observed nonnegative finite effort and nonmissing normalized category only.',
    'undefined': 'null score and no rank for fewer than two observed categories or zero total effort variance.',
    'ordering': 'Descending eta2 rounded to 12 decimal places; equal rounded scores share competition rank, then factor name alphabetical. Undefined factors are separate.',
    'direction': 'Category mean minus the feature-specific eligible mean; higher/lower/equal uses relative tolerance 1e-12 and absolute tolerance 1e-12. Medians and quartiles are also reported.',
    'small_category_threshold': 5,
    'inference': 'Exploratory only; no significance tests, multiplicity claims or certified wins.',
    'fitting': 'No fitted model or learned preprocessing; no train/evaluation split is required for descriptive summaries.',
    'seed': 42,
    'seed_usage': 'No randomness in this method.',
}


def diagnose(rows):
    """Accept parsed project mappings; ignore nonpredictors except the target."""
    rows = list(rows)
    if not rows:
        raise ValueError('Need at least one project')
    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f'Row {index}: expected a project mapping')
        values = normalize_proposal({p: row.get(p) for p in PREDICTORS})
        effort = row.get('Effort-')
        if effort is not None and (isinstance(effort, bool) or not isinstance(effort, (int, float))
                                   or not math.isfinite(effort) or effort < 0):
            raise ValueError(f'Row {index}: effort must be finite nonnegative numeric or None')
        normalized.append((values, float(effort) if effort is not None else None))
    observed = [e for _, e in normalized if e is not None]
    factors = []
    for position, field in enumerate(PREDICTORS):
        groups = defaultdict(list)
        counts = defaultdict(int)
        for values, effort in normalized:
            category = values[position]
            counts[category] += 1
            if effort is not None:
                groups[category].append(effort)
        labels = sorted(c for c in counts if c is not None)
        eligible_labels = [c for c in labels if groups[c]]
        eligible = [e for c in eligible_labels for e in groups[c]]
        summary = numeric_summary(eligible)
        center = summary['mean']
        within = sum(sum((e - numeric_summary(groups[c])['mean']) ** 2 for e in groups[c])
                     for c in eligible_labels)
        total = sum((e - center) ** 2 for e in eligible) if eligible else 0.0
        between = sum(len(groups[c]) * (numeric_summary(groups[c])['mean'] - center) ** 2
                      for c in eligible_labels)
        if not all(math.isfinite(x) for x in [within, total, between]):
            raise ValueError('Effort magnitudes exceed supported summary arithmetic')
        status = ('no_observed_categories' if not eligible_labels else
                  'constant_on_observed_rows' if len(eligible_labels) == 1 else
                  'zero_effort_variance' if total == 0 else 'rankable')
        categories = []
        for label in labels + ([None] if None in counts else []):
            stats = numeric_summary(groups[label])
            difference = stats['mean'] - center if label is not None and stats['count'] and center is not None else None
            direction = ('unavailable' if difference is None else 'equal' if math.isclose(stats['mean'], center, rel_tol=1e-12, abs_tol=1e-12)
                         else 'higher' if difference > 0 else 'lower')
            categories.append({
                'category': label, 'missing_predictor': label is None,
                'project_count': counts[label], 'observed_effort_count': stats['count'],
                'missing_effort_count': counts[label] - stats['count'], 'effort_summary': stats,
                'included_in_score': label is not None and stats['count'] > 0,
                'mean_minus_eligible_mean': difference, 'observed_mean_direction': direction,
                'small_category': stats['count'] < METHOD['small_category_threshold'],
            })
        score = min(1.0, max(0.0, between / total)) if status == 'rankable' else None
        factors.append({
            'factor': field, 'status': status, 'eta_squared': score, 'rank': None,
            'nonmissing_category_count_all_rows': len(labels),
            'constant_all_rows': len(labels) == 1,
            'observed_category_count': len(eligible_labels), 'eligible_effort_summary': summary,
            'excluded_missing_predictor_count': counts.get(None, 0),
            'missing_effort_count': len(rows) - len(observed),
            'sums_of_squares': {'between': between, 'within': within, 'total': total},
            'categories': categories,
            'explanation': (f'{field}: association strength eta2={score:.6f}; direction is shown by each category mean relative to {center:.6g}.'
                            if score is not None else f'{field}: {status}; no supported between-category strength ranking.'),
        })
    ranked = sorted((f for f in factors if f['eta_squared'] is not None),
                    key=lambda f: (-round(f['eta_squared'], 12), f['factor']))
    prior = None
    rank = None
    for index, item in enumerate(ranked, 1):
        key = round(item['eta_squared'], 12)
        if key != prior:
            rank = index
        item['rank'] = rank
        prior = key
    report = {
        'target': 'Effort-', 'objective': 'minimize', 'units': 'unspecified',
        'project_count': len(rows), 'observed_effort_summary': numeric_summary(observed),
        'missing_effort_count': len(rows) - len(observed),
        'predictor_fields': PREDICTORS, 'excluded_from_ranking': ['Effort-', 'Defects-', 'Kloc+', 'all X-suffixed metadata', 'unknown fields'],
        'method': METHOD, 'ranked_findings': ranked,
        'unranked_factors': [f for f in factors if f['eta_squared'] is None],
        'constant_factors': [f['factor'] for f in factors if f['constant_all_rows']],
        'limitations': LIMITATIONS,
    }
    report['manager_explanation'] = render_text(report)
    return report


def render_text(report):
    lines = ['Project factors accompanying higher observed effort (physical units unspecified).',
             'This is an exploratory description of raw effort, not a diagnosis of budget overrun or excessive effort.',
             'Ranking uses categorical eta squared: between-category variation divided by total variation.',
             'The score measures association strength; category means below supply direction. No category ordering is assumed.']
    for factor in report['ranked_findings']:
        lines.append(f"Rank {factor['rank']}: {factor['factor']}, eta2={factor['eta_squared']:.6f}, eligible n={factor['eligible_effort_summary']['count']}.")
        for category in factor['categories']:
            stats = category['effort_summary']
            lines.append(f"  {category['category']!r}: projects={category['project_count']}, observed n={stats['count']}, mean={stats['mean']}, median={stats['median']}; {category['observed_mean_direction']} than eligible mean; small-category warning={category['small_category']}.")
    lines.append('Constant factors across nonmissing inputs: ' + ', '.join(report['constant_factors']))
    for factor in report['unranked_factors']:
        lines.append(factor['explanation'])
    lines.append('Planning use: review these historical category groups alongside project scope, size and context; the ranking does not show savings from changing a factor.')
    lines.extend(report['limitations'])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        report = diagnose(read_projects(args.dataset))
        report['dataset_path'] = str(args.dataset.resolve())
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(report['manager_explanation'], end='')
    except (OSError, ValueError, OverflowError, csv.Error) as exc:
        parser.exit(2, f'Input error: {exc}\n')


if __name__ == '__main__':
    main()
