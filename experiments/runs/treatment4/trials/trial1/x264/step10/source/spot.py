"""Exploratory, leave-configuration-out local benchmark anomaly screening."""

import argparse
import json
import math
from pathlib import Path
import platform
import statistics

import numpy as np
import sklearn
from sklearn.metrics import pairwise_distances

from blame import _validate
from remember import OBJECTIVE, load_measurements

NEIGHBORS = 10
MULTIPLIER = 3.0
MAD_FACTOR = 1.4826
RANGE_FLOOR = 0.01
DISTANCE_TOLERANCE = 1e-12
CHECKS = [
    'Locate the original log using the 1-based CSV data-row identity; verify parsing and transcription.',
    'Confirm encoder version, effective flags, input video, workload, hardware and measurement units.',
    'Repeat this configuration and nearby controls with randomized order and multiple independent runs.',
    'Check system load, thermal throttling, caching, warm-up, timeouts and incomplete encodes.',
    'Verify output validity and quality before interpreting unusually low Performance- as an improvement.',
]
LIMITATIONS = [
    'Exploratory screening only: flagged records are not confirmed measurement errors.',
    'No verified anomaly labels; no precision, recall or detection accuracy is estimated.',
    'The threshold is a descriptive heuristic, not a p-value, error-rate estimate or fixed screening quota.',
    'Genuine nonlinear setting effects can be flagged; broad or clustered errors can be missed.',
    'All settings have equal range-normalized distance weight; redundant encodings can count a choice repeatedly.',
    'Numeric encodings are treated as numeric distances; names do not establish flag semantics or valid combinations.',
    'Neighbors exclude the complete evaluated configuration, but the threshold uses all observed residuals.',
    'This is a transductive historical screen, not independent validation of predictive accuracy.',
    'Each neighboring configuration has equal weight via its median; repeated rows still each receive a score.',
    'No measured repeatability or causal conclusion follows from differences across configurations.',
    'Performance- is minimized and physical units are unspecified; both unusually high and low results may flag.',
    'Distance ties within 1e-12 are all included; sparse data can produce broad, weakly comparable neighborhoods.',
    'At least three distinct configurations are required; quadratic configuration-distance storage limits large inputs.',
    'Floating-point arithmetic and extreme dynamic ranges can limit numerical resolution.',
]


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def screen(data, *, neighbors=NEIGHBORS, multiplier=MULTIPLIER):
    """Use local medians with every evaluated configuration fully excluded.

    Score = absolute observed minus local-median outcome. Flag iff score is
    strictly greater than median(scores) + multiplier * robust_scale, where
    robust_scale = max(1.4826 * MAD(scores), .01 * observed outcome range).
    Common target scaling is internal numerical protection, not a model input.
    """
    _validate(data)
    _positive_integer(neighbors, 'neighbors')
    if (isinstance(multiplier, bool) or not isinstance(multiplier, (int, float))
            or not math.isfinite(multiplier) or multiplier <= 0):
        raise ValueError('multiplier must be finite and positive')
    groups = {}
    for i, row in enumerate(data.rows):
        groups.setdefault(tuple(row[:-1]), []).append(i)
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError('Screening needs at least three distinct configurations')
    features = np.asarray(keys, dtype=float)
    magnitude = np.maximum(1., np.max(np.abs(features), axis=0))
    scaled = features / magnitude
    low, high = scaled.min(axis=0), scaled.max(axis=0)
    width = high - low
    active = width > 0
    if any(len({key[j] for key in keys}) > 1 and not active[j]
           for j in range(len(data.settings))):
        raise ValueError('Setting variation below supported numeric resolution')
    normalized = (scaled[:, active] - low[active]) / width[active]
    # Manhattan sum divided by active-field count: mean range-normalized difference.
    distances = pairwise_distances(normalized, metric='manhattan') / int(active.sum())
    np.fill_diagonal(distances, np.inf)
    scale = max(1., max(abs(float(row[-1])) for row in data.rows))
    targets = [float(row[-1]) / scale for row in data.rows]
    medians = [statistics.median(targets[i] for i in groups[key]) for key in keys]
    effective_k = min(neighbors, len(keys) - 1)
    findings = []
    for g, key in enumerate(keys):
        cutoff = float(np.sort(distances[g])[effective_k - 1])
        selected = [j for j in range(len(keys))
                    if j != g and distances[g, j] <= cutoff + DISTANCE_TOLERANCE]
        local = statistics.median(medians[j] for j in selected)
        peer_rows = sorted(i + 1 for j in selected for i in groups[keys[j]])
        for i in groups[key]:
            signed = targets[i] - local
            findings.append({
                'data_row': i + 1, 'settings': dict(zip(data.settings, key)),
                OBJECTIVE: float(data.rows[i][-1]), 'score_scaled': abs(signed),
                'evidence': {
                    'local_median_Performance-': local * scale,
                    'signed_difference': signed * scale,
                    'direction_vs_neighbors': 'higher' if signed > 0 else 'lower' if signed < 0 else 'equal',
                    'excluded_configuration_data_rows': [r + 1 for r in groups[key]],
                    'neighbor_data_rows': peer_rows,
                    'neighbor_configuration_count': len(selected),
                    'nearest_distance': float(min(distances[g])),
                    'kth_distance': cutoff,
                    'neighbor_configuration_median_min': min(medians[j] for j in selected) * scale,
                    'neighbor_configuration_median_max': max(medians[j] for j in selected) * scale,
                    'within_neighbor_configuration_median_range': bool(
                        min(medians[j] for j in selected) <= targets[i]
                        <= max(medians[j] for j in selected)),
                    'neighbor_configuration_median_MAD': statistics.median(
                        abs(medians[j] - local) for j in selected) * scale,
                },
            })
    scores = [f['score_scaled'] for f in findings]
    center = statistics.median(scores)
    mad = statistics.median(abs(v - center) for v in scores)
    floor = RANGE_FLOOR * (max(targets) - min(targets))
    robust_scale = max(MAD_FACTOR * mad, floor)
    threshold = center + multiplier * robust_scale
    for finding in findings:
        value = finding.pop('score_scaled')
        finding['score'] = value * scale
        finding['flagged'] = value > threshold
        finding['threshold'] = threshold * scale
        finding['evidence']['explanation'] = (
            f"Absolute difference from the median of other nearby configurations is "
            f"{finding['score']:.6g}; screening threshold is {threshold * scale:.6g}; "
            f"{'strictly exceeds' if finding['flagged'] else 'does not exceed'} the threshold.")
        finding['suggested_checks'] = list(CHECKS) if finding['flagged'] else []
    findings.sort(key=lambda f: (-f['score'], f['data_row']))
    for i, finding in enumerate(findings):
        finding['rank'] = (findings[i - 1]['rank'] if i and finding['score'] == findings[i - 1]['score']
                           else i + 1)
    report = {
        'method': 'Leave-configuration-out nearest-configuration median discrepancy',
        'parameters': {'requested_neighbors': neighbors, 'effective_neighbors_before_ties': effective_k,
                       'multiplier': multiplier, 'mad_factor': MAD_FACTOR,
                       'outcome_range_floor_fraction': RANGE_FLOOR,
                       'distance_tolerance': DISTANCE_TOLERANCE, 'seed': None,
                       'randomness': 'None; deterministic method'},
        'versions': {'python': platform.python_version(), 'scikit_learn': sklearn.__version__},
        'input_fields': {'settings': list(data.settings), 'outcome': OBJECTIVE,
                         'distance_fields': [n for n, a in zip(data.settings, active) if a],
                         'constant_fields_excluded_from_distance': [n for n, a in zip(data.settings, active) if not a]},
        'preprocessing': {
            'settings': 'Each variable setting mapped to [0,1] using observed minimum and range; constants excluded.',
            'setting_minima': dict(zip(data.settings, map(float, features.min(axis=0)))),
            'setting_maxima': dict(zip(data.settings, map(float, features.max(axis=0)))),
            'outcome_internal_divisor': scale,
            'outcome': 'Divide by max(1, max(abs(outcome))) internally; report scores in original unspecified units.',
        },
        'neighborhood_rule': 'Mean Manhattan distance over variable settings. Exclude own exact setting tuple and all its rows. Select k closest distinct configurations including all ties within tolerance. Predict median of their configuration-level outcome medians; no fitted model.',
        'score_definition': 'abs(observed Performance- - leave-configuration-out local median); higher means more unusual in either direction.',
        'threshold_definition': 'Strict score > median(scores) + multiplier * max(1.4826*MAD(scores), 0.01*(max(outcomes)-min(outcomes))). Computed in common scaled units; equality is not flagged. Constant outcomes give zero scores and no flags.',
        'threshold': threshold * scale,
        'threshold_components': {'median_score': center * scale, 'mad_score': mad * scale,
                                 'range_floor': floor * scale, 'robust_scale': robust_scale * scale},
        'ranking_rule': 'Descending score; exact ties share competition rank, then ascending CSV data-row identity.',
        'row_identity': '1-based CSV data rows excluding the header; original order retained through loading.',
        'row_count': len(data.rows), 'unique_configurations': len(keys),
        'flagged_count': sum(f['flagged'] for f in findings),
        'objective_direction': 'minimize', 'units': None,
        'ranked_findings': findings, 'suggested_checks': list(CHECKS), 'limitations': list(LIMITATIONS),
    }
    try:
        json.dumps(report, allow_nan=False)
    except (ValueError, OverflowError) as error:
        raise ValueError('Screening exceeds supported finite numeric range') from error
    return report


def render_text(report, top=10):
    """Display limit never changes the threshold, flags or complete JSON ranking."""
    _positive_integer(top, 'top')
    lines = [f"Exploratory screen: {report['row_count']} rows; {report['flagged_count']} flagged; "
             f"threshold={report['threshold']:.9g}; showing up to {top} records."]
    for finding in report['ranked_findings'][:top]:
        lines.append(json.dumps(finding, sort_keys=True, allow_nan=False))
    lines.append('Flags are investigation priorities, not confirmed measurement errors. Units unspecified.')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--top', type=int, default=10)
    parser.add_argument('--neighbors', type=int, default=NEIGHBORS)
    parser.add_argument('--multiplier', type=float, default=MULTIPLIER)
    args = parser.parse_args(argv)
    try:
        paths = [args.output_dir / name for name in ('anomalies.json', 'anomalies.txt')]
        if any(p.resolve() == args.dataset.resolve() for p in paths):
            raise ValueError('Output must not overwrite the dataset')
        report = screen(load_measurements(args.dataset), neighbors=args.neighbors,
                        multiplier=args.multiplier)
        report['dataset'] = str(args.dataset.resolve())
        report['display_limit'] = args.top
        rendered = render_text(report, args.top)
        serialized = json.dumps(report, indent=2, allow_nan=False) + '\n'
        args.output_dir.mkdir(parents=True, exist_ok=True)
        paths[0].write_text(serialized, encoding='utf-8')
        paths[1].write_text(rendered, encoding='utf-8')
    except (OSError, ValueError) as error:
        parser.exit(2, f'error: {error}\n')
    print(rendered, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
