"""Evaluate one observed encoder configuration and Performance- result at a time."""

import argparse
from collections.abc import Mapping
import csv
import json
import math
from numbers import Real
from pathlib import Path

from remember import OBJECTIVE

ARRIVAL_ORDER = ('Replay every CSV data row exactly once in original file order, '
                 'starting at 1; CSV order is not evidence of real chronological order.')
LIMITATIONS = [
    'Performance- is minimized; physical units are unspecified.',
    'Statistics describe all received measurements, not repeated trials of one configuration.',
    'Duplicate records count as separate arrivals; no independence or causal claim is made.',
    'Population variance uses divisor n; a single observation has zero variability.',
    'Exact objective ties retain the first observed configuration; no tolerance is applied.',
    'Finite numeric settings are accepted without inferring encoder flag or combination validity.',
    'Floating-point rounding applies; updates exceeding finite numeric range are rejected atomically.',
    'The accumulator stores O(d) state and takes O(d) work per arrival for d settings; '
    'the report retains O(n*d) history for audit.',
    'No prediction, stopping rule, uncertainty guarantee, or sequential significance test is supplied.',
]


def _number(value):
    """Reject coercions and booleans; return a representable finite real float."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError('Values must be finite real numbers, excluding booleans')
    try:
        value = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError('Value exceeds finite numeric range') from error
    if not math.isfinite(value):
        raise ValueError('Values must be finite real numbers')
    return value


class Flow:
    """Incremental Welford statistics; update(configuration, result) reveals one target.

    No records, CSV, fitted model, or future targets are consulted by update.
    Snapshots are detached copies. Rejected updates leave all state unchanged.
    """

    def __init__(self, settings):
        if isinstance(settings, (str, bytes)):
            raise ValueError('settings must be a sequence of distinct names')
        try:
            settings = tuple(settings)
        except TypeError as error:
            raise ValueError('settings must be a sequence of distinct names') from error
        if (not settings or any(not isinstance(s, str) or not s.strip()
                                or s == OBJECTIVE for s in settings)
                or len(set(settings)) != len(settings)):
            raise ValueError('settings must be nonempty, distinct names excluding Performance-')
        self.settings = settings
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._best = None
        self._ties = 0

    def snapshot(self):
        """Population statistics of the received prefix; empty mean/variability are null."""
        variance = max(0.0, self._m2 / self._count) if self._count else None
        best = None if self._best is None else {
            **self._best, 'settings': dict(self._best['settings'])}
        return {
            'count': self._count,
            'mean': self._mean if self._count else None,
            'population_variance': variance,
            'population_stddev': math.sqrt(variance) if variance is not None else None,
            'best': best,
            'best_tie_count': self._ties,
        }

    def update(self, configuration, result):
        """Consume exactly one configuration and its observed scalar Performance-."""
        if not isinstance(configuration, Mapping) or set(configuration) != set(self.settings):
            raise ValueError('Configuration must contain exactly the settings; exclude Performance-')
        config = {name: _number(configuration[name]) for name in self.settings}
        value = _number(result)
        count = self._count + 1
        delta = value - self._mean
        mean = self._mean + delta / count
        m2 = self._m2 + delta * (value - mean)
        if not all(math.isfinite(v) for v in (mean, m2)):
            raise ValueError('Update exceeds finite numeric range')
        best, ties = self._best, self._ties
        if best is None or value < best[OBJECTIVE]:
            best = {'arrival_index': count, 'settings': config, OBJECTIVE: value}
            ties = 1
        elif value == best[OBJECTIVE]:
            ties += 1
        self._count, self._mean, self._m2 = count, mean, m2
        self._best, self._ties = best, ties
        return self.snapshot()


def replay_csv(path):
    """Yield updates lazily; validate each row only when it arrives.

    Header rules match the preserved Step 1 loader. Numeric rows are parsed here
    instead of using its eager loader so future records are never read for updates.
    """
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header or any(not name.strip() for name in header):
            raise ValueError('CSV must have nonempty column names')
        if len(set(header)) != len(header):
            raise ValueError('Duplicate column names')
        if OBJECTIVE not in header:
            raise ValueError(f'Missing registered objective: {OBJECTIVE}')
        settings = tuple(name for name in header if name != OBJECTIVE)
        tracker = Flow(settings)
        indices = [header.index(name) for name in settings]
        target = header.index(OBJECTIVE)
        for index, cells in enumerate(reader, 1):
            if len(cells) != len(header):
                raise ValueError(f'CSV data row {index}: expected {len(header)} fields')
            try:
                values = [float(cell) for cell in cells]
            except ValueError as error:
                raise ValueError(f'CSV data row {index}: missing or nonnumeric value') from error
            config = dict(zip(settings, (values[i] for i in indices)))
            summary = tracker.update(config, values[target])
            yield {'arrival_index': index,
                   'observed': {'settings': config, OBJECTIVE: values[target]},
                   'summary': summary}
        if tracker.snapshot()['count'] == 0:
            raise ValueError('CSV has no measurement rows')


def report(path):
    """Retain the audit history; updates themselves never rescan it."""
    history = list(replay_csv(path))
    return {
        'dataset': str(Path(path).resolve()),
        'objective': OBJECTIVE, 'direction': 'minimize', 'units': None,
        'arrival_order_rule': ARRIVAL_ORDER,
        'algorithm': 'Welford online mean and M2; strict running minimum; no model fitting',
        'history': history,
        'final_summary': history[-1]['summary'],
        'limitations': LIMITATIONS,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        target = args.output_dir / 'flow.json'
        if target.resolve() == args.dataset.resolve():
            raise ValueError('Output must not overwrite the dataset')
        result = report(args.dataset)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    except (OSError, ValueError) as error:
        parser.exit(2, f'error: {error}\n')
    print(ARRIVAL_ORDER)
    print(json.dumps(result['final_summary'], sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
