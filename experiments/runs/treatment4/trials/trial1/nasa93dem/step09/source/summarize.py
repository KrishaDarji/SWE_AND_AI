"""Reusable descriptive NASA93 summaries; no fitting or ordinal assumptions."""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median, stdev

PREDICTORS = 'prec flex resl team pmat rely data cplx ruse docu time stor pvol acap pcap pcon apex plex ltex tool site sced'.split()
METADATA = ['idX', 'centerX', 'YearX', 'MonthsX']
OBJECTIVES = {'Kloc+': 'maximize', 'Effort-': 'minimize', 'Defects-': 'minimize'}
COLUMNS = ['idX', 'centerX', 'YearX'] + PREDICTORS + list(OBJECTIVES) + ['MonthsX']
NUMERIC = {'idX', 'YearX', 'Kloc+', 'Effort-', 'Defects-', 'MonthsX'}
MISSING = {'', '?', 'NA', 'N/A', 'null'}


def read_projects(dataset):
    """Validate the NASA93 schema and parse cells without altering the input."""
    with Path(dataset).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, None)
        if header is None or len(header) != len(set(header)) or set(header) != set(COLUMNS):
            raise ValueError('Expected exactly the 29 unique NASA93 column names')
        rows = []
        for line, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f'Row {line}: expected {len(header)} cells, found {len(row)}')
            parsed = {}
            for name, raw in zip(header, row):
                value = raw.strip()
                if value in MISSING:
                    parsed[name] = None
                elif name in NUMERIC:
                    try:
                        number = float(value)
                    except ValueError as exc:
                        raise ValueError(f'Row {line}, {name}: invalid numeric value {value!r}') from exc
                    if not math.isfinite(number):
                        raise ValueError(f'Row {line}, {name}: numeric value must be finite')
                    parsed[name] = number
                else:
                    parsed[name] = value
            rows.append(parsed)
    if not rows:
        raise ValueError('Dataset must contain at least one project')
    return rows


def quantile(values, probability):
    """Linear interpolation at index (n - 1) * probability, for sorted values."""
    index = (len(values) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def numeric_summary(values):
    values = sorted(values)
    if not values:
        return dict.fromkeys(['count', 'sum', 'mean', 'std_sample', 'min', 'q25', 'median', 'q75', 'max'], None) | {'count': 0}
    return {
        'count': len(values), 'sum': sum(values), 'mean': mean(values),
        'std_sample': stdev(values) if len(values) > 1 else None,
        'min': values[0], 'q25': quantile(values, 0.25), 'median': median(values),
        'q75': quantile(values, 0.75), 'max': values[-1],
    }


def summarize(dataset):
    rows = read_projects(dataset)
    columns = {}
    for name in COLUMNS:
        values = [row[name] for row in rows if row[name] is not None]
        role = 'metadata' if name.endswith('X') else 'objective' if name in OBJECTIVES else 'candidate_predictor'
        item = {'type': 'numeric' if name in NUMERIC else 'categorical', 'role': role,
                'missing_count': len(rows) - len(values), 'nonmissing_count': len(values),
                'unique_count': len(set(values))}
        if name in OBJECTIVES:
            item['direction'] = OBJECTIVES[name]
        if name in NUMERIC:
            item['statistics'] = numeric_summary(values)
        else:
            item['value_counts'] = dict(sorted(Counter(values).items()))
        columns[name] = item
    return {
        'dataset_path': str(Path(dataset).resolve()), 'project_count': len(rows),
        'column_count': len(COLUMNS), 'numeric_column_count': len(NUMERIC),
        'categorical_column_count': len(COLUMNS) - len(NUMERIC),
        'candidate_predictors': PREDICTORS, 'objectives': OBJECTIVES, 'metadata': METADATA,
        'total_missing_values': sum(c['missing_count'] for c in columns.values()),
        'delivery_outcomes': ['Kloc+', 'Effort-', 'Defects-', 'MonthsX'],
        'columns': columns,
        'conventions': {
            'missing_tokens_after_stripping': sorted(MISSING),
            'standard_deviation': 'sample, denominator n-1; null when fewer than two observations',
            'quartiles': 'linear interpolation at sorted index (n-1)*p',
            'centerX': 'categorical center label; no ordering inferred',
            'idX_and_YearX': 'numeric metadata; identifier statistics have no predictive interpretation',
            'excluded_inputs': 'All X metadata and objective fields are excluded from candidate predictors',
            'units': 'No units supplied or inferred; original field names retained',
            'limitations': 'Descriptive observations only; no baseline, deadline, success label, causal claim, or prediction',
        },
    }


def render_text(report):
    lines = [
        'NASA93 historical projects and delivery outcomes',
        f"Projects: {report['project_count']}; columns: {report['column_count']}",
        f"Numeric columns: {report['numeric_column_count']}; categorical columns: {report['categorical_column_count']}",
        f"Missing cells: {report['total_missing_values']}",
        'Candidate predictors: ' + ', '.join(report['candidate_predictors']),
        'Metadata excluded from optimization and predictors: ' + ', '.join(report['metadata']),
        'Objectives: Kloc+ maximize; Effort- minimize; Defects- minimize.',
        'Delivery outcomes (MonthsX is descriptive metadata):',
    ]
    for name in report['delivery_outcomes']:
        item = report['columns'][name]
        lines.append(f"  {name} [{item['role']}]: " + json.dumps(item['statistics'], sort_keys=True))
    lines.append('All project characteristics, missing counts, numeric summaries, and categorical counts:')
    for name, item in report['columns'].items():
        lines.append(f"  {name}: " + json.dumps(item, sort_keys=True))
    lines.append('Assumptions and limitations:')
    lines.extend(f'  {key}: {value}' for key, value in report['conventions'].items())
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    try:
        report = summarize(args.dataset)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f'Input error: {exc}\n')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    readable = render_text(report)
    (args.output_dir / 'summary.txt').write_text(readable, encoding='utf-8')
    print(readable, end='')


if __name__ == '__main__':
    main()
