"""Find the smallest encoded-setting change with a lower recorded Performance-."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from blame import _validate
from remember import OBJECTIVE, load_measurements

TIE_RULE = ('Among strictly lower observed Performance- rows, minimize differing '
            'encoded setting columns, then Performance-, then 1-based CSV data row. '
            'Exact numeric comparisons; no tolerance.')
LIMITATIONS = [
    'Offline exhaustive search over all recorded measurements with full-data objective access; not budgeted search or evaluation on unseen configurations.',
    'Encoded-column distance is a proxy for change size, not a measured operational cost. Redundant or one-hot columns may represent one underlying choice.',
    'Only recorded configurations are recommended; setting semantics and validity of unseen combinations are not inferred.',
    'A lower historical outcome does not prove causal, repeatable, or statistically certified improvement. Selection on the same history can be optimistic.',
    'Duplicate configurations remain separate observations; a lower observation for identical settings has change cost zero and does not imply that any setting repair occurred.',
    'Performance- is minimized; physical units are unspecified. No quality measurements are available and quality preservation is not established.',
    'CSV numeric fields are parsed as finite floats by the reused loader; exact float comparisons define changes and ties. Near ties have no stability guarantee.',
]


def recommend_repair(data, baseline_row):
    """Search O(rows * settings) time without fitting a prediction model.

    Reuse Step 4's direct-input validation and Step 1's Measurements schema.
    Repeated configurations are deliberately neither deduplicated nor averaged.
    """
    _validate(data)
    if (isinstance(baseline_row, bool) or not isinstance(baseline_row, int)
            or not 1 <= baseline_row <= len(data.rows)):
        raise ValueError(f'baseline_row must be an integer from 1 to {len(data.rows)}')
    baseline = data.rows[baseline_row - 1]

    def record(number):
        row = data.rows[number - 1]
        return {'data_row': number, 'settings': dict(zip(data.settings, row[:-1])),
                OBJECTIVE: row[-1]}

    best = None
    improving_count = 0
    for number, row in enumerate(data.rows, 1):
        if row[-1] < baseline[-1]:
            improving_count += 1
            cost = sum(old != new for old, new in zip(baseline[:-1], row[:-1]))
            key = (cost, row[-1], number)
            if best is None or key < best:
                best = key
    changes = []
    reduction = None
    if best is not None:
        chosen = data.rows[best[2] - 1]
        changes = [{'column': name, 'old': old, 'new': new}
                   for name, old, new in zip(data.settings, baseline[:-1], chosen[:-1])
                   if old != new]
        reduction = baseline[-1] - chosen[-1]
        if not math.isfinite(reduction):
            raise ValueError('Observed objective reduction exceeds finite numeric range')
    return {
        'status': 'improving_candidate_found' if best is not None else 'no_improving_candidate',
        'objective': OBJECTIVE, 'direction': 'minimize', 'units': None,
        'baseline': record(baseline_row),
        'recommendation': record(best[2]) if best is not None else None,
        'changed_columns': changes, 'change_cost': best[0] if best is not None else None,
        'observed_objective_reduction': reduction,
        'search': {
            'scope': 'Every recorded CSV data row, including duplicates; full-data objective access.',
            'recorded_row_count': len(data.rows), 'improving_row_count': improving_count,
            'setting_columns': list(data.settings), 'rule': TIE_RULE,
            'distance': 'Count of differing encoded setting columns; objective excluded.',
        },
        'limitations': list(LIMITATIONS),
    }


def render_text(report):
    """Render settings and every change directly from the structured evidence."""
    lines = ['Recorded-configuration repair (Performance- minimized; units unspecified)',
             'Baseline: ' + json.dumps(report['baseline'], sort_keys=True),
             report['search']['scope'], report['search']['rule'],
             f"Searched {report['search']['recorded_row_count']} rows; strictly improving rows: {report['search']['improving_row_count']}."]
    if report['recommendation'] is None:
        lines.append('No improving candidate: no recorded row has strictly lower Performance-.')
    else:
        lines.append('Recommended recorded row: ' + json.dumps(report['recommendation'], sort_keys=True))
        lines.extend(f"Changed {c['column']}: {c['old']} -> {c['new']}"
                     for c in report['changed_columns'])
        lines.append(f"Change cost: {report['change_cost']}; observed objective reduction: {report['observed_objective_reduction']}.")
        if not report['changed_columns']:
            lines.append('No setting changes: identical recorded settings have a lower observation.')
    lines.extend(report['limitations'])
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--baseline-row', type=int, required=True,
                        help='1-based CSV data row, excluding the header')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        paths = [args.output_dir / name for name in ('repair.json', 'repair.txt')]
        if any(path.resolve() == args.dataset.resolve() for path in paths):
            raise ValueError('Output must not overwrite the dataset')
        result = recommend_repair(load_measurements(args.dataset), args.baseline_row)
        result['provenance'] = {
            'dataset': str(args.dataset.resolve()),
            'sha256': hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            'data_row_convention': '1-based CSV data rows, excluding header',
            'reuse': ['remember.load_measurements', 'remember.Measurements', 'blame._validate'],
        }
        serialized = json.dumps(result, indent=2, allow_nan=False) + '\n'
        rendered = render_text(result)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        paths[0].write_text(serialized, encoding='utf-8')
        paths[1].write_text(rendered, encoding='utf-8')
    except (OSError, ValueError, OverflowError) as error:
        parser.exit(2, f'error: {error}\n')
    print(rendered, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
