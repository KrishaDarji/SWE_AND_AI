"""Describe Health CSV configurations and objectives without fitting a model."""
import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import sklearn

CONFIGURATION = ('N_estimators', 'criterion', 'Min_sample_leaves',
                 'Min_impurity_decrease', 'Max_depth')
OBJECTIVES = {'MRE-': 'minimize', 'ACC+': 'maximize', 'PRED40+': 'maximize'}
CATEGORICAL = {'criterion'}
EXPECTED = set(CONFIGURATION) | set(OBJECTIVES)


def quantile(values, fraction):
    """Compute a linearly interpolated quantile on sorted, nonempty values."""
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def duplicates(keys):
    """Count groups, participating records, and extra records separately."""
    counts = Counter(keys)
    return {'unique_count': len(counts),
            'repeated_group_count': sum(n > 1 for n in counts.values()),
            'records_in_repeated_groups': sum(n for n in counts.values() if n > 1),
            'extra_record_count': sum(n - 1 for n in counts.values())}


def summarize(dataset):
    """Validate the schema and return a JSON-serializable descriptive report."""
    rows = []
    with Path(dataset).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle)
        headers = next(reader, None)
        if not headers or len(headers) != len(set(headers)) or set(headers) != EXPECTED:
            raise ValueError('Expected exactly the eight unique Health column names')
        for line, raw in enumerate(reader, 2):
            if len(raw) != len(headers):
                raise ValueError(f'Row {line}: expected {len(headers)} cells')
            row = {}
            for name, cell in zip(headers, raw):
                cell = cell.strip()
                if not cell:
                    row[name] = None
                elif name in CATEGORICAL:
                    row[name] = cell
                else:
                    try:
                        number = float(cell)
                    except ValueError as exc:
                        raise ValueError(f'Row {line}, {name}: invalid numeric value') from exc
                    if not math.isfinite(number):
                        raise ValueError(f'Row {line}, {name}: non-finite numeric value')
                    row[name] = number
            rows.append(row)
    if not rows:
        raise ValueError('Dataset must contain at least one record')
    columns = {}
    for name in headers:
        values = [row[name] for row in rows if row[name] is not None]
        distinct = sorted(set(values))
        info = {'role': 'objective' if name in OBJECTIVES else 'configuration',
                'type': 'categorical' if name in CATEGORICAL else 'numeric',
                'nonmissing_count': len(values), 'missing_count': len(rows) - len(values),
                'distinct_count': len(distinct), 'distinct_values': distinct}
        if name in CATEGORICAL:
            info['value_counts'] = dict(sorted(Counter(values).items()))
        else:
            ordered = sorted(values)
            info['statistics'] = None if not values else {
                'count': len(values), 'mean': statistics.mean(values),
                'std_sample': statistics.stdev(values) if len(values) > 1 else None,
                'min': ordered[0], 'q25': quantile(ordered, .25),
                'median': statistics.median(ordered), 'q75': quantile(ordered, .75),
                'max': ordered[-1]}
        if name in OBJECTIVES:
            info['direction'] = OBJECTIVES[name]
            best = (min(values) if OBJECTIVES[name] == 'minimize' else max(values)) if values else None
            info['best_observed'] = best
            info['best_observed_count'] = values.count(best) if values else 0
        columns[name] = info
    return {'dataset_path': str(Path(dataset).resolve()),
            'record_count': len(rows), 'column_count': len(headers), 'column_names': headers,
            'configuration_fields': list(CONFIGURATION), 'objective_directions': OBJECTIVES,
            'columns': columns,
            'duplicate_records': duplicates(tuple(row[n] for n in headers) for row in rows),
            'repeated_configurations': duplicates(tuple(row[n] for n in CONFIGURATION) for row in rows),
            'environment': {'scikit_learn_version': sklearn.__version__},
            'conventions': {
                'missing': 'Empty or whitespace-only cells; other numeric tokens must be finite numbers.',
                'distinct': 'Nonmissing normalized values only; numeric values parsed as float.',
                'quantiles': 'Linear interpolation at (n - 1) * p.',
                'std': 'Sample standard deviation (n - 1); null for fewer than two values.',
                'duplicates': 'After trimming whitespace and parsing numbers; missing cells compare equal.',
                'scope': 'Descriptive observed metrics only; no training, formulas, units, or timeline inferred.'}}


def render_text(report):
    lines = [f"Records: {report['record_count']}; columns: {report['column_count']}"]
    for role in ('configuration', 'objective'):
        lines.append(f'\n{role.title()} fields:')
        for name, info in report['columns'].items():
            if info['role'] != role:
                continue
            lines.append(f"{name}: {info['type']}; missing={info['missing_count']}; distinct={info['distinct_count']}")
            if info['type'] == 'categorical':
                lines.append('  Value counts: ' + json.dumps(info['value_counts'], sort_keys=True))
            else:
                lines.append('  Statistics: ' + json.dumps(info['statistics'], sort_keys=True))
                if role == 'configuration':
                    lines.append('  Distinct values: ' + json.dumps(info['distinct_values']))
            if role == 'objective':
                lines.append(f"  Direction: {info['direction']}; best observed={info['best_observed']}; count={info['best_observed_count']}")
    for key in ('duplicate_records', 'repeated_configurations'):
        lines.append(f'\n{key}: ' + json.dumps(report[key], sort_keys=True))
    lines.append('\nFull distinct values for every column are in summary.json.')
    lines.append('\nConventions: ' + json.dumps(report['conventions'], sort_keys=True))
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    try:
        report = summarize(args.dataset)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f'Error: {exc}\n')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    rendered = render_text(report)
    (args.output_dir / 'summary.txt').write_text(rendered, encoding='utf-8')
    print(rendered, end='')


if __name__ == '__main__':
    main()
