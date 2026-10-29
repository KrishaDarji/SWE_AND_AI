"""Exploratory marginal associations with recorded MRE- (smaller is better)."""
import argparse
from bisect import bisect_left
from itertools import combinations
import json
import math
from pathlib import Path
import statistics

import sklearn

from certify import observed_summary
from guess import validate_configuration
from summarize import CONFIGURATION, OBJECTIVES, duplicates, load_rows, quantile

METHOD = {
    'name': 'Marginal grouped eta-squared of observed MRE-',
    'formula': 'SS_between = sum_g n_g*(mean_g-grand_mean)^2; SS_total = sum_i (y_i-grand_mean)^2; eta_squared = SS_between/SS_total.',
    'ranking': 'Descending eta_squared, competition ranks (1,1,3); exact floating-score ties share rank, display ties in CONFIGURATION order. Unavailable scores are unranked.',
    'direction': 'Eta-squared is unsigned association strength, not direction or conditional importance. Group means, medians and differences from the field-specific overall mean describe direction.',
    'grouping': 'criterion uses exact string categories, never ordinal codes. Numeric fields with <=8 distinct nonmissing values use exact values; otherwise use quartile cuts at p=.25,.5,.75, linear interpolation (n-1)*p over ALL nonmissing predictor rows, independently of outcomes. Duplicate cuts removed; cuts equal to observed maximum removed. Intervals: first [minimum,cut], subsequent (previous_cut,cut], last (last_cut,maximum]. No extrapolation.',
    'missing': 'No imputation. Ranking excludes missing predictor or MRE per field; report missing-predictor MRE separately. Boundaries include predictor rows with missing MRE. Report every nonempty predictor group, even if all outcomes missing.',
    'sparse': 'Flag groups with fewer than 5 observed MRE values; retain them in exploratory scores, without treating the flag as a significance threshold.',
    'degenerate': 'Constant/all-missing predictors, fewer than two labeled groups, and zero outcome variance have null score and no rank.',
    'inference': 'Exploratory descriptive analysis of this finite dataset. No significance tests, certified wins, uncertainty intervals, fitting, tuning, or predictive validation.',
    'seed': 42,
    'randomness': 'No random operations; seed is documented for consistency, not used.',
    'pairwise': 'Joint nonmissing predictor group counts and observed MRE summaries for all ten pairs expose overlap, imbalance and possible interactions. These are descriptive cells, not adjusted effects or formal interaction tests.'
}
LIMITATIONS = [
    'Observational configuration selection and unrecorded factors confound associations; changing a setting is not shown to cause error reduction.',
    'Marginal scores do not isolate a field from correlated settings. Joint tables expose co-occurrence but do not control confounding. Correlated fields can share or mask associations.',
    'Interactions may reverse or cancel marginal directions. Joint tables are exploratory and can be sparse; no interaction model or conditional importance is estimated.',
    'Eta-squared favors fine partitions and is sample-specific; category counts and numeric coarsening affect ranking. It is not out-of-sample importance, and differently missing subsets need not be comparable.',
    'Sparse or singleton groups can exaggerate apparent strength. No p values, multiple-test claims, uncertainty guarantees, practical thresholds, or optimized recommendations are provided.',
    'Repeated configurations count as rows and may overweight settings; independence, random sampling and measurement repeatability are not established.',
    'Missing outcomes or predictors may be selective. Constants cannot support observed setting comparisons; constant outcomes cannot support strength estimates.',
    'Quartile bins hide within-range trends; observed extreme mean groups do not establish monotonic numeric effects. Boundary choices use predictor values only.',
    'MRE- is the recorded scale with unspecified formula and units. Neither ACC+ nor PRED40+ is a diagnostic predictor.',
    'Finite float arithmetic is used; unrepresentable summary arithmetic is rejected. Rankings near ties may be sensitive to rounding.'
]


def normalize_rows(rows):
    """Validate reusable record input without mutating it; ignore no unknown keys."""
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError('rows must be a nonempty list or tuple of record dictionaries')
    clean = []
    for row in rows:
        if not isinstance(row, dict) or set(row) - (set(CONFIGURATION) | set(OBJECTIVES)):
            raise ValueError('Record must contain only Health schema fields')
        if 'MRE-' not in row:
            raise ValueError('Each record requires MRE-; use None for missing')
        config = validate_configuration({name: row.get(name) for name in CONFIGURATION})
        target = validate_configuration({'N_estimators': row['MRE-']})['N_estimators']
        clean.append({**config, 'MRE-': target})
    return clean


def partition(values, categorical=False):
    """Return JSON definitions and group indices; missing predictor index is None."""
    ordered = sorted(v for v in values if v is not None)
    distinct = sorted(set(ordered))
    if categorical or len(distinct) <= 8:
        lookup = {v: i for i, v in enumerate(distinct)}
        return {'kind': 'categories' if categorical else 'exact_values', 'cuts': [],
                'groups': [{'value': v} for v in distinct]}, [lookup.get(v) for v in values]
    cuts = sorted({quantile(ordered, p) for p in (.25, .5, .75)} - {ordered[-1]})
    bounds = [ordered[0], *cuts, ordered[-1]]
    definitions = [{'lower': a, 'upper': b, 'lower_inclusive': i == 0,
                    'upper_inclusive': True} for i, (a, b) in enumerate(zip(bounds, bounds[1:]))]
    return {'kind': 'quartile_ranges', 'cuts': cuts, 'groups': definitions}, [
        None if v is None else bisect_left(cuts, v) for v in values]


def summary(values):
    return observed_summary(values, len(values))


def diagnose(rows):
    """Rank only five configuration fields; report observations, not interventions."""
    rows = normalize_rows(rows)
    fields, memberships = {}, {}
    for name in CONFIGURATION:
        values = [r[name] for r in rows]
        grouping, ids = partition(values, name == 'criterion')
        memberships[name] = ids
        groups = []
        for group_id, definition in enumerate(grouping['groups']):
            outcomes = [r['MRE-'] for r, g in zip(rows, ids) if g == group_id]
            stats = summary(outcomes)
            groups.append({'group_id': group_id, 'definition': definition,
                           'row_count': len(outcomes), 'mre': stats, 'sparse': stats['n'] < 5})
        included = [r['MRE-'] for r, g in zip(rows, ids) if g is not None and r['MRE-'] is not None]
        overall = summary(included)
        labeled = [g for g in groups if g['mre']['n']]
        status = ('all_missing_predictor' if not grouping['groups'] else
                  'constant_predictor' if len(set(v for v in values if v is not None)) == 1 else
                  'insufficient_labeled_groups' if len(labeled) < 2 else 'ok')
        mean = overall['mean']
        ss_total = math.fsum((y - mean) ** 2 for y in included) if included else 0.0
        ss_between = math.fsum(g['mre']['n'] * (g['mre']['mean'] - mean) ** 2 for g in labeled) if labeled else 0.0
        if status == 'ok' and ss_total == 0:
            status = 'constant_outcome'
        score = ss_between / ss_total if status == 'ok' else None
        for g in groups:
            delta = g['mre']['mean'] - mean if g['mre']['n'] else None
            g['mean_minus_included_mean'] = delta
            g['observed_direction'] = ('unavailable' if delta is None else 'equal' if delta == 0
                                       else 'higher_MRE' if delta > 0 else 'lower_MRE')
        low = min((g['mre']['mean'] for g in labeled), default=None)
        high = max((g['mre']['mean'] for g in labeled), default=None)
        fields[name] = {'field': name, 'rank': None, 'status': status, 'eta_squared': score,
                        'ss_between': ss_between, 'ss_total': ss_total,
                        'nonmissing_predictor_count': len(values) - values.count(None),
                        'distinct_nonmissing_values': len(set(v for v in values if v is not None)),
                        'grouping': grouping, 'groups': groups, 'included_mre': overall,
                        'missing_predictor_mre': summary([r['MRE-'] for r in rows if r[name] is None]),
                        'highest_mean_group_ids': [g['group_id'] for g in labeled if g['mre']['mean'] == high],
                        'lowest_mean_group_ids': [g['group_id'] for g in labeled if g['mre']['mean'] == low]}
    ranked = sorted((f for f in fields.values() if f['eta_squared'] is not None),
                    key=lambda f: -f['eta_squared'])
    for f in ranked:
        f['rank'] = 1 + sum(other['eta_squared'] > f['eta_squared'] for other in ranked)
    joint = []
    for a, b in combinations(CONFIGURATION, 2):
        cells = {}
        for r, ga, gb in zip(rows, memberships[a], memberships[b]):
            if ga is not None and gb is not None:
                cells.setdefault((ga, gb), []).append(r['MRE-'])
        joint.append({'fields': [a, b], 'excluded_missing_predictor_rows': len(rows) - sum(map(len, cells.values())),
                      'cells': [{'group_ids': list(key), 'row_count': len(ys),
                                 'mre': summary(ys), 'sparse': sum(y is not None for y in ys) < 5}
                                for key, ys in sorted(cells.items())]})
    report = {'method': METHOD, 'target': {'field': 'MRE-', 'direction': 'minimize'},
              'predictor_fields': list(CONFIGURATION), 'excluded_predictors': list(OBJECTIVES),
              'record_count': len(rows), 'overall_mre': summary([r['MRE-'] for r in rows]),
              'repeated_configurations': duplicates(tuple(r[n] for n in CONFIGURATION) for r in rows),
              'ranked_findings': ranked,
              'unranked_fields': [f for f in fields.values() if f['eta_squared'] is None],
              'pairwise_group_summaries': joint, 'limitations': LIMITATIONS,
              'environment': {'scikit_learn_version': sklearn.__version__},
              'reuse': 'Unchanged Step 1 loader, quantiles and duplicate counts; Step 2 configuration normalization; Step 3 observed summaries. Steps 1–4 source and tests retained unchanged. Added grouped marginal ranking, directions, missing/sparse/constant handling and joint summaries.'}
    report['maintainer_explanation'] = render_text(report)
    json.dumps(report, allow_nan=False)
    return report


def render_text(report):
    lines = ['Observed MRE- is minimized. Exploratory marginal associations only.',
             'Rank by grouped eta-squared (between-group / total sum of squares); strength is unsigned.']
    for f in report['ranked_findings']:
        lines.append(f"Rank {f['rank']}: {f['field']}, eta-squared={f['eta_squared']:.8g}; n={f['included_mre']['n']}; mean MRE={f['included_mre']['mean']:.8g}.")
        for label, key in [('Highest', 'highest_mean_group_ids'), ('Lowest', 'lowest_mean_group_ids')]:
            for i in f[key]:
                g = f['groups'][i]
                lines.append(f"  {label} observed mean: {g['definition']}; n={g['mre']['n']}; mean={g['mre']['mean']:.8g}; median={g['mre']['median']:.8g}; sparse={g['sparse']}.")
    for f in report['unranked_fields']:
        lines.append(f"Unranked {f['field']}: {f['status']}; observed values/groups={f['grouping']['groups']}.")
    lines.append('Use JSON group tables for all counts, missingness, quartile boundaries, MRE dispersion and pairwise overlap. Numeric directions need not be monotonic. Correlation, interactions, sparse groups and observational confounding limit attribution. Changing settings is not shown to reduce error.')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        if sklearn.__version__ != '1.6.0':
            raise ValueError('Use scikit-learn 1.6.0')
        _, rows = load_rows(args.dataset)
        report = diagnose(rows)
        report['dataset_path'] = str(args.dataset.resolve())
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    except (OSError, ValueError, OverflowError) as exc:
        parser.exit(2, f'Error: {exc}\n')
    print(report['maintainer_explanation'])


if __name__ == '__main__':
    main()
