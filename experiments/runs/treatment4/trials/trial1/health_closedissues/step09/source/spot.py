"""Exploratory two-sided screening of recorded Health objective distributions."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import sklearn

from justify import normalize_records
from summarize import CONFIGURATION, OBJECTIVES, load_rows, quantile

THRESHOLD = 3.5
MIN_OBSERVED = 5
LIMITATIONS = [
    'No verified anomaly labels: flags are investigation priorities, not confirmed errors or estimates of precision, recall or error prevalence.',
    'Marginal rarity does not establish a measurement error. Legitimately good or poor configurations may be flagged; conditional anomalies may be missed.',
    'Distributions include the evaluated records and duplicate configurations, which can mask extremes or overweight common settings. These are descriptive distances, not unseen prediction errors.',
    'The maximum across available objectives is an OR screening rule; 3.5 is a heuristic distance threshold with no family-wise error or probability interpretation.',
    'MAD/IQR degeneracy uses a conservative range fallback that can suppress flags in discrete or concentrated distributions. Constant columns score zero; fewer than five observations are unscored.',
    'Missing outcomes are not imputed; subsets differ and selective missingness may bias screening. Missing configurations compare equal for context only.',
    'No metric formulas, theoretical ranges, costs, causal effects, independence, or guaranteed relationships between objectives are assumed.',
    'Exact normalized float equality defines ties and duplicate configurations. CSV row identities are one-based data-record positions, not timestamps or persistent IDs.',
    'Scores measure unusualness, not optimization quality. There is no training, prediction, calibrated hypothesis test, or automatic record repair.'
]
CHECKS = [
    'Trace flagged CSV row IDs to original evaluation logs and verify parsing, metric names, directions, aggregation and transcription.',
    'Confirm evaluation data, splits, sample counts, seeds and preprocessing; check for leakage or inconsistent evaluation protocols.',
    'Re-evaluate unusual configurations on a common independent validation set, recording repeated outcomes and uncertainty before deciding whether an error exists.',
    'For repeated configurations with differing outcomes, compare data versions, evaluation seeds, software versions and logging provenance; duplicates need not be independent repeats.',
    'Review both unusually good and unusually poor outcomes with the maintainer; retain original records and document any verified corrections separately.'
]


def finite(value):
    """Reject arithmetic overflow rather than emit invalid JSON or bogus scores."""
    if not math.isfinite(value):
        raise ValueError('Objective magnitudes exceed finite screening arithmetic')
    return value


def reference(values):
    """Learn descriptive robust location and scale from observed batch values."""
    ordered = sorted(values)
    result = {'observed_count': len(values), 'median': None, 'mad': None,
              'iqr': None, 'scale': None, 'scale_method': 'insufficient_observations'}
    if not values:
        return result
    center = finite(statistics.median(ordered))
    result['median'] = center
    if len(values) < MIN_OBSERVED:
        return result
    mad = finite(statistics.median([abs(finite(x - center)) for x in ordered]))
    iqr = finite(quantile(ordered, .75) - quantile(ordered, .25))
    result.update(mad=mad, iqr=iqr)
    if mad > 0:
        scale, method = finite(1.4826 * mad), '1.4826 * MAD'
    elif iqr > 0:
        scale, method = finite(iqr / 1.349), 'IQR / 1.349 fallback'
    elif ordered[-1] > ordered[0]:
        scale, method = finite(ordered[-1] - ordered[0]), 'observed range fallback'
    else:
        scale, method = 0.0, 'constant'
    result.update(scale=scale, scale_method=method)
    return result


def score_value(value, ref):
    """Higher absolute standardized deviation means more unusual, either tail."""
    if value is None or ref['scale'] is None:
        return None
    if ref['scale'] == 0:
        return 0.0
    return finite(abs(finite(value - ref['median'])) / ref['scale'])


def crosses_threshold(score, threshold):
    """The boundary is strict; null means unscored and never flagged."""
    return score is not None and score > threshold


def screen_rows(rows, threshold=THRESHOLD):
    """Return all records ranked by max marginal score, then ascending row ID."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError('threshold must be a positive finite number')
    rows = normalize_records(rows)
    refs = {name: reference([r[name] for r in rows if r[name] is not None])
            for name in OBJECTIVES}
    groups = {}
    for row in rows:
        key = tuple(row[n] for n in CONFIGURATION)
        groups.setdefault(key, []).append(row)
    contexts = {}
    for key, members in groups.items():
        ranges = {}
        for name in OBJECTIVES:
            values = [r[name] for r in members if r[name] is not None]
            ranges[name] = {'observed_count': len(values),
                            'min': min(values) if values else None,
                            'max': max(values) if values else None}
        contexts[key] = {'record_count': len(members), 'observed_objective_ranges': ranges}
    findings = []
    for row_id, row in enumerate(rows, 1):
        evidence = {}
        for name, direction in OBJECTIVES.items():
            ref, value = refs[name], row[name]
            score = score_value(value, ref)
            delta = finite(value - ref['median']) if value is not None and ref['median'] is not None else None
            relation = ('missing' if delta is None else 'equal' if delta == 0 else
                        'better' if (delta < 0) == (direction == 'minimize') else 'worse')
            evidence[name] = {'observed': value, 'score': score,
                              'crosses_threshold': crosses_threshold(score, threshold),
                              'observed_minus_median': delta,
                              'relation_to_median_in_objective_direction': relation,
                              'status': 'missing' if value is None else
                              'insufficient_observations' if score is None else 'scored'}
        available = [e['score'] for e in evidence.values() if e['score'] is not None]
        score = max(available) if available else None
        findings.append({'row_id': row_id, 'configuration': {n: row[n] for n in CONFIGURATION},
                         'observed_objectives': {n: row[n] for n in OBJECTIVES},
                         'anomaly_score': score, 'crosses_threshold': crosses_threshold(score, threshold),
                         'supporting_evidence': evidence,
                         'same_configuration_context': contexts[tuple(row[n] for n in CONFIGURATION)]})
    findings.sort(key=lambda r: (r['anomaly_score'] is None,
                                -(r['anomaly_score'] or 0), r['row_id']))
    for rank, record in enumerate(findings, 1):
        record['rank'] = rank
    return {
        'method_settings': {'method': 'Two-sided marginal median/MAD screening',
                            'minimum_observed_per_objective': MIN_OBSERVED,
                            'randomness': 'None; deterministic',
                            'preprocessing': 'Reuse finite-float/trimmed-category normalization; no transforms, imputation, estimator fitting or learned feature preprocessing.',
                            'scale_fallback_order': ['1.4826 * MAD', 'IQR / 1.349', 'observed max - min', 'constant: zero score'],
                            'quantiles': 'Linear interpolation at (n - 1) * p'},
        'field_roles': {'configuration_context_only': list(CONFIGURATION),
                        'objectives_screened': dict(OBJECTIVES), 'prediction_inputs': []},
        'score_definition': 'Per objective abs(value - median) / scale; constant columns score 0; missing or insufficient data score null. Record score is maximum available objective score, or null if none.',
        'score_direction': 'Larger means more unusual; both better and worse than median can be unusual.',
        'threshold_definition': {'value': threshold, 'comparison': 'strictly greater than',
                                 'combination': 'OR over scored objectives, equivalent to maximum > threshold',
                                 'interpretation': 'Heuristic exploratory distance cutoff; no estimated error rate and no fixed quota.'},
        'ranking_definition': 'Descending score; ties ascending one-based CSV data-row ID; null scores last. Rank is ordinal.',
        'references': refs, 'record_count': len(rows),
        'flagged_count': sum(r['crosses_threshold'] for r in findings),
        'unscored_count': sum(r['anomaly_score'] is None for r in findings),
        'missing_data_handling': {'objectives': 'Omit missing values from reference; no imputation; score remaining objectives independently.',
                                  'configuration': 'Retain missing fields; exact normalized tuples define context groups, including missing-equals-missing.',
                                  'counts': {n: sum(r[n] is None for r in rows) for n in (*CONFIGURATION, *OBJECTIVES)}},
        'ranked_findings': findings, 'suggested_maintainer_checks': CHECKS.copy(),
        'limitations': LIMITATIONS.copy(), 'environment': {'scikit_learn_version': sklearn.__version__}}


def screen(dataset, threshold=THRESHOLD):
    _, rows = load_rows(dataset)
    result = screen_rows(rows, threshold)
    result['dataset_path'] = str(Path(dataset).resolve())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--threshold', type=float, default=THRESHOLD)
    parser.add_argument('--display-limit', type=int, default=10)
    args = parser.parse_args()
    try:
        if sklearn.__version__ != '1.6.0':
            raise ValueError('Use scikit-learn 1.6.0')
        if args.display_limit < 0:
            raise ValueError('display-limit must be nonnegative')
        report = screen(args.dataset, args.threshold)
        report['display'] = {'limit': args.display_limit, 'scope': 'Console only; all ranked records retained in JSON; no effect on flags.'}
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f'Error: {exc}\n')
    print(json.dumps({'record_count': report['record_count'], 'flagged_count': report['flagged_count'],
                      'threshold': report['threshold_definition'],
                      'displayed_findings': report['ranked_findings'][:args.display_limit]}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
