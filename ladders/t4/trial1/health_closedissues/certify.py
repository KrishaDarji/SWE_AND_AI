"""Compare two predeclared observed configuration families, without model predictions."""
import argparse
import json
import math
import platform
import statistics
from collections import Counter
from pathlib import Path

import scipy
import sklearn
from scipy.stats import t

from summarize import CONFIGURATION, OBJECTIVES, duplicates, load_rows, quantile


def observed_summary(values, total):
    """Summarize finite observed values; missingness is counted per objective."""
    clean = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Outcomes must be finite numbers or None')
        clean.append(float(value))
    ordered = sorted(clean)
    return {'n': len(clean), 'missing': total - len(clean),
            'mean': statistics.mean(clean) if clean else None,
            'std_sample': statistics.stdev(clean) if len(clean) > 1 else None,
            'median': statistics.median(clean) if clean else None,
            'q25': quantile(ordered, .25) if clean else None,
            'q75': quantile(ordered, .75) if clean else None,
            'min': min(clean) if clean else None, 'max': max(clean) if clean else None}


def welch_comparison(baseline, candidate, direction='minimize', practical_threshold=None):
    """Two-sided Welch test of equal population means; CI is candidate minus baseline.

    A threshold, if supplied, is a positive minimum meaningful improvement in the
    recorded objective scale. It must be justified externally before interpreting
    results. This function does not treat a nonsignificant test as equivalence.
    """
    if direction not in ('minimize', 'maximize'):
        raise ValueError('direction must be minimize or maximize')
    if practical_threshold is not None and (isinstance(practical_threshold, bool) or
            not isinstance(practical_threshold, (int, float)) or
            not math.isfinite(practical_threshold) or practical_threshold <= 0):
        raise ValueError('practical_threshold must be positive and finite')
    a = observed_summary(baseline, len(baseline))
    b = observed_summary(candidate, len(candidate))
    delta = b['mean'] - a['mean'] if a['n'] and b['n'] else None
    improvement = None if delta is None else (-delta if direction == 'minimize' else delta)
    result = {'candidate_minus_baseline_mean': delta, 'objective_direction': direction,
              'observed_direction': ('unavailable' if delta is None else 'equal' if delta == 0
                                     else 'candidate_better' if improvement > 0 else 'candidate_worse'),
              'status': 'insufficient_data', 'standard_error': None, 'degrees_of_freedom': None,
              't_statistic': None, 'p_two_sided': None, 'confidence_level': .95,
              'confidence_interval': None, 'alpha': .05, 'statistically_detected': None,
              'practical_threshold': practical_threshold,
              'practical_assessment': 'Undetermined: no externally justified minimum meaningful improvement supplied.'
              if practical_threshold is None else 'Unassessed: inferential interval unavailable.'}
    if min(a['n'], b['n']) < 2:
        return result
    va = a['std_sample'] ** 2 / a['n']
    vb = b['std_sample'] ** 2 / b['n']
    variance = va + vb
    if variance == 0:
        result['status'] = 'zero_variance_inference_unavailable'
        return result
    se = math.sqrt(variance)
    df = variance ** 2 / (va ** 2 / (a['n'] - 1) + vb ** 2 / (b['n'] - 1))
    statistic = delta / se
    p = float(2 * t.sf(abs(statistic), df))
    margin = float(t.ppf(.975, df)) * se
    interval = [delta - margin, delta + margin]
    result.update(status='ok', standard_error=se, degrees_of_freedom=df,
                  t_statistic=statistic, p_two_sided=p, confidence_interval=interval,
                  statistically_detected=p < .05)
    if practical_threshold is not None:
        benefit_interval = [-interval[1], -interval[0]] if direction == 'minimize' else interval
        result['practical_assessment'] = (
            'Interval supports improvement exceeding the supplied threshold.'
            if benefit_interval[0] > practical_threshold else
            'Interval does not support improvement exceeding the supplied threshold; this is not an equivalence test.')
    return result


def select_families(rows, field, baseline, candidate):
    """Membership uses only the declared configuration field, never objectives."""
    if field not in CONFIGURATION or baseline is None or candidate is None or baseline == candidate:
        raise ValueError('Use a configuration field and two distinct nonmissing family values')
    groups = {'baseline': [], 'candidate': []}
    excluded = []
    for i, row in enumerate(rows):
        value = row[field]
        label = 'baseline' if value == baseline else 'candidate' if value == candidate else None
        if label is None:
            excluded.append(i)
        else:
            groups[label].append(i)
    return groups, excluded


def compare_families(rows, field='criterion', baseline='squared_error', candidate='absolute_error',
                     primary='MRE-', practical_threshold=None):
    """Reusable comparison on validated observed rows; no surrogate is consulted."""
    if primary not in OBJECTIVES:
        raise ValueError('Unknown primary objective')
    groups, excluded = select_families(rows, field, baseline, candidate)
    summaries = {}
    for label, indices in groups.items():
        summaries[label] = {'total_rows': len(indices), 'row_ids': [i + 1 for i in indices],
                            'objectives': {name: observed_summary([rows[i][name] for i in indices], len(indices))
                                           for name in OBJECTIVES},
                            'configuration_repetitions': duplicates(tuple(rows[i][name] for name in CONFIGURATION)
                                                                    for i in indices)}
    comparison = welch_comparison([rows[i][primary] for i in groups['baseline']],
                                 [rows[i][primary] for i in groups['candidate']],
                                 OBJECTIVES[primary], practical_threshold)
    differences = {name: (summaries['candidate']['objectives'][name]['mean'] -
                         summaries['baseline']['objectives'][name]['mean'])
                   if all(summaries[label]['objectives'][name]['n'] for label in groups) else None
                   for name in OBJECTIVES}
    if comparison['status'] != 'ok':
        evidence = 'Inference unavailable because of insufficient observations or zero estimated variance.'
    elif comparison['statistically_detected']:
        evidence = 'A mean difference is detected under the stated sampling assumptions.'
    else:
        evidence = 'No mean difference is detected at alpha 0.05; this does not prove equivalence.'
    conclusion = (f"Observed primary direction: {comparison['observed_direction']}. {evidence} "
                  f"{comparison['practical_assessment']} This comparison does not certify a causal "
                  'improvement or repeatability for a particular configuration.')
    return {'group_definitions': {'field': field, 'baseline': baseline, 'candidate': candidate,
                                 'selection': 'Fixed before examining outcomes; membership uses configuration only'},
            'record_count': len(rows), 'included_count': sum(map(len, groups.values())),
            'excluded_count': len(excluded),
            'excluded_family_counts': [{'value': value, 'count': count} for value, count in
                                       Counter(rows[i][field] for i in excluded).items()],
            'excluded_missing_family_count': sum(rows[i][field] is None for i in excluded),
            'groups': summaries, 'primary_objective': primary,
            'candidate_minus_baseline_means': differences, 'primary_comparison': comparison,
            'secondary_objectives': {'names': [name for name in OBJECTIVES if name != primary],
                                     'scope': 'Descriptive only; no secondary tests, intervals, or significance claims'},
            'method': {'name': 'Two-sided Welch unequal-variance t test and 95% Welch t interval for mean difference',
                       'null': 'Equal population means', 'randomized': False, 'seed': 42,
                       'seed_note': 'Deterministic procedure; no random draws',
                       'multiplicity': 'One predeclared primary test; secondary summaries only',
                       'small_samples': 'At least two observed outcomes per group; small-sample inference requires approximately normal populations',
                       'missing': 'Available outcomes separately by objective; no imputation',
                       'practical_criterion': 'A positive externally justified minimum meaningful mean improvement is required; none is assumed in the demo'},
            'limitations': [
                'Welch inference assumes independent observations within and between groups, finite variances, and approximate normality of sample means; large samples help but do not verify these assumptions.',
                'Rows are observational configurations, not documented random samples. P values and intervals are conditional approximations, not guarantees for this fixed dataset or a wider population.',
                'Other hyperparameters may differ between families; confounding and configuration selection prevent causal attribution to criterion.',
                'Repeated configurations are counted but not assumed to be independent repeated measurements. Unreported dependence can invalidate row-level standard errors.',
                'Missing outcomes may be selective; available-outcome comparisons need not represent missing configurations.',
                'No client-approved practical threshold is available; statistical significance alone cannot establish practical relevance. No equivalence test is performed.',
                'Recorded objective scales only: formulas, physical units, and compute costs are unspecified.',
                'No surrogate predictions are used. This does not establish measurement repeatability for a particular configuration.',
                'The separate required same analysis is not claimed complete; its exact procedure was not supplied.'],
            'maintainer_conclusion': conclusion,
            'versions': {'python': platform.python_version(), 'scikit_learn': sklearn.__version__, 'scipy': scipy.__version__}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        if sklearn.__version__ != '1.6.0':
            raise ValueError('Use scikit-learn 1.6.0')
        _, rows = load_rows(args.dataset)
        report = compare_families(rows)
        report['dataset_path'] = str(args.dataset.resolve())
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    except (OSError, ValueError) as exc:
        parser.exit(2, f'Error: {exc}\n')
    print(json.dumps({key: report[key] for key in ('group_definitions', 'record_count', 'included_count',
                     'excluded_count', 'excluded_family_counts', 'candidate_minus_baseline_means',
                     'primary_comparison', 'maintainer_conclusion')}, indent=2, allow_nan=False))
    for label, group in report['groups'].items():
        print(label + ': ' + json.dumps({k: v for k, v in group.items() if k != 'row_ids'}))


if __name__ == '__main__':
    main()
