"""Describe which observed setting values accompany worse Performance- outcomes."""

import argparse
import json
import math
from numbers import Real
from pathlib import Path
import statistics

from remember import Measurements, OBJECTIVE, load_measurements, summarize

LIMITATIONS = [
    'Exploratory observational associations, not proven causal effects or significance tests; no p-values or multiplicity-adjusted claims.',
    'Each setting is assessed marginally. Other settings, interactions, sampling imbalance and unrecorded workload or hardware can confound the ranking.',
    'Correlated or redundant encodings can share the same association; their scores must not be added or interpreted as independent contributions.',
    'Different indicators for one encoded choice can be mutually exclusive; changing one alone may be invalid. Names do not establish encoder semantics.',
    'Eta squared measures separation of observed value-group means, not direction, fitted-model feature importance, or a guaranteed improvement.',
    'All exact observed numeric values are treated as categories. Many levels and singleton groups can inflate eta squared; no out-of-sample guarantee is made.',
    'Every row has equal weight, including duplicates. Counts are not independent replicate counts or uncertainty estimates.',
    'Constants cannot support comparisons; zero score does not exclude interactions or distribution differences beyond the mean.',
    'Performance- is minimized; physical units and output quality are unspecified. Larger means indicate worse measured outcomes, not a known number of seconds.',
    'No model is fitted by this diagnosis, so model evaluation and random seed are not applicable. Step 2 prediction remains a separate holdout-evaluated tool.',
    'Scores use floating-point arithmetic; exact score ties share competition rank and are displayed alphabetically. Near ties have no stability guarantee.',
]


def _validate(data):
    """Validate direct API callers as strictly as CSV loading, without coercions."""
    if not isinstance(data, Measurements):
        raise ValueError('Expected Measurements')
    names = data.settings
    if (not isinstance(names, (tuple, list)) or not names
            or any(not isinstance(n, str) or not n.strip() or n == OBJECTIVE for n in names)
            or len(set(names)) != len(names)):
        raise ValueError('Distinct nonempty settings must exclude Performance-')
    if not data.rows:
        raise ValueError('Need at least one measurement')
    for row in data.rows:
        if not isinstance(row, (tuple, list)) or len(row) != len(names) + 1:
            raise ValueError('Each row must contain settings followed by Performance-')
        for value in row:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError('Measurements must be finite real numbers excluding booleans')
            try:
                finite = math.isfinite(float(value))
            except (OverflowError, ValueError) as error:
                raise ValueError('Measurements exceed finite numeric range') from error
            if not finite:
                raise ValueError('Measurements must be finite real numbers')


def diagnose(data):
    """Rank variable settings by categorical eta squared on all observed rows.

    eta² = sum_v n_v (mean_y_v - mean_y)^2 / sum_i (y_i - mean_y)^2.
    Target scaling improves numeric range and leaves the ratio unchanged.
    No targets are used as inputs; no prediction model is fitted.
    """
    _validate(data)
    try:
        return _diagnose(data)
    except (OverflowError, statistics.StatisticsError) as error:
        raise ValueError('Statistics exceed supported finite numeric range') from error


def _diagnose(data):
    summary = summarize(data)
    targets = [float(row[-1]) for row in data.rows]
    scale = max(1., max(map(abs, targets)))
    normalized = [v / scale for v in targets]
    mean = statistics.mean(normalized)
    total_ss = math.fsum((v - mean) ** 2 for v in normalized)
    overall = summary['objective']['statistics']['mean']
    findings, constants = [], []
    for index, name in enumerate(data.settings):
        detail = summary['settings'][name]
        values = detail['values']
        for group in values:
            group['mean_minus_overall'] = group['objective']['mean'] - overall
            group['association'] = ('higher' if group['mean_minus_overall'] > 0 else
                                    'lower' if group['mean_minus_overall'] < 0 else 'equal')
        if detail['constant']:
            constants.append({'setting': name, 'values': values, 'rank': None,
                              'score': None, 'explanation': 'Constant in observed rows; uninformative for comparisons.'})
            continue
        between_ss = math.fsum(
            g['count'] * (statistics.mean(normalized[i] for i, row in enumerate(data.rows)
                                          if row[index] == g['value']) - mean) ** 2
            for g in values)
        score = min(1., max(0., between_ss / total_ss)) if total_ss else 0.
        high = max(g['objective']['mean'] for g in values)
        low = min(g['objective']['mean'] for g in values)
        high_values = [g['value'] for g in values if g['objective']['mean'] == high]
        low_values = [g['value'] for g in values if g['objective']['mean'] == low]
        explanation = (
            f"{name}: eta squared {score:.6f}. Highest observed mean Performance- "
            f"{high:.6g} at {high_values}; lowest {low:.6g} at {low_values}; "
            f"mean gap {high - low:.6g} (units unspecified). "
            'See value-level counts and summaries; these are associations, not causal effects.')
        if high == low:
            explanation += ' All observed value groups have equal means; no directional mean association.'
        findings.append({'setting': name, 'score': score, 'score_name': 'eta_squared',
                         'between_ss_scaled': between_ss, 'values': values,
                         'highest_mean_values': high_values, 'lowest_mean_values': low_values,
                         'mean_gap': high - low, 'explanation': explanation})
    findings.sort(key=lambda f: (-f['score'], f['setting']))
    for i, finding in enumerate(findings):
        finding['rank'] = (findings[i - 1]['rank'] if i and finding['score'] == findings[i - 1]['score']
                           else i + 1)
    variable = [data.settings.index(f['setting']) for f in findings]
    relationships = []
    for pos, i in enumerate(variable):
        for j in variable[pos + 1:]:
            x, y = ([float(row[k]) for row in data.rows] for k in (i, j))
            pairs = set(zip(x, y))
            equivalent = len(pairs) == len(set(x)) == len(set(y))
            sx, sy = max(1., max(map(abs, x))), max(1., max(map(abs, y)))
            a = [v / sx for v in x]
            b = [v / sy for v in y]
            am, bm = statistics.mean(a), statistics.mean(b)
            da, db = [v - am for v in a], [v - bm for v in b]
            xx, yy = math.fsum(v * v for v in da), math.fsum(v * v for v in db)
            if not xx or not yy:
                raise ValueError('Setting variation below supported numeric range')
            r = math.fsum(u * v for u, v in zip(da, db)) / math.sqrt(xx) / math.sqrt(yy)
            relationships.append({'settings': [data.settings[i], data.settings[j]],
                                  'pearson_r': min(1., max(-1., r)),
                                  'equivalent_observed_partitions': equivalent})
    return {
        'objective': OBJECTIVE, 'direction': 'minimize', 'units': None,
        'row_count': summary['row_count'], 'features': list(data.settings),
        'overall_objective': summary['objective']['statistics'],
        'unique_configurations': summary['unique_configurations'],
        'duplicate_measurement_rows': summary['duplicate_measurement_rows'],
        'repeated_configuration_rows': summary['repeated_configuration_rows'],
        'method': {
            'name': 'Marginal categorical eta squared',
            'formula': 'sum_v n_v*(mean_y_v-mean_y)^2 / sum_i (y_i-mean_y)^2',
            'ranking': 'Variable settings by descending eta squared; exact ties share competition rank; alphabetical display tie break.',
            'zero_target_variance': 'All variable settings receive score 0.',
            'weighting': 'Every observed row equally weighted; exact numeric values form categories.',
            'direction': 'Value-level mean minus overall mean and highest/lowest group means; independent of score.',
            'seed': None, 'fitted_model': None, 'evaluation': 'Not applicable: descriptive statistics only.',
            'inference': 'Exploratory; no significance tests, p-values or confidence intervals.',
            'target_scale': scale, 'total_ss_scaled': total_ss,
            'relationships': 'All variable pairs: Pearson correlation and exact equivalence of observed value partitions; neither implies causality.',
        },
        'ranked_findings': findings, 'constant_settings': constants,
        'setting_relationships': relationships,
        'client_explanation': (
            'Larger Performance- means worse measured encoding performance. The ranking prioritizes '
            'settings whose observed value groups separate overall mean outcomes most strongly. '
            'Use group means and counts to identify the values accompanying slower encodes. '
            'Correlated encoded choices can share credit; validate changes with controlled measurements.'),
        'limitations': list(LIMITATIONS),
    }


def render_text(report):
    lines = [report['client_explanation'],
             f"Rows: {report['row_count']}; Performance- units unspecified."]
    for finding in report['ranked_findings']:
        lines.append(f"Rank {finding['rank']}: {finding['explanation']}")
        for g in finding['values']:
            lines.append(f"  value={g['value']:g}; n={g['count']}; mean={g['objective']['mean']:.6g}; "
                         f"vs overall={g['mean_minus_overall']:+.6g} ({g['association']})")
    lines.append('Constant settings: ' + ', '.join(f['setting'] for f in report['constant_settings']))
    lines.extend(report['limitations'])
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        paths = [args.output_dir / name for name in ('blame.json', 'blame.txt')]
        if any(p.resolve() == args.dataset.resolve() for p in paths):
            raise ValueError('Output must not overwrite the dataset')
        result = diagnose(load_measurements(args.dataset))
        result['dataset'] = str(args.dataset.resolve())
        rendered = render_text(result)
        serialized = json.dumps(result, indent=2, allow_nan=False) + '\n'
        args.output_dir.mkdir(parents=True, exist_ok=True)
        paths[0].write_text(serialized, encoding='utf-8')
        paths[1].write_text(rendered, encoding='utf-8')
    except (OSError, ValueError) as error:
        parser.exit(2, f'error: {error}\n')
    print(rendered, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
