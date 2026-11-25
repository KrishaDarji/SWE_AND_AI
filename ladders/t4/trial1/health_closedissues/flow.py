"""Incremental descriptive evaluation of separately supplied observed objectives."""
import argparse
import copy
import csv
import json
import math
from pathlib import Path

import sklearn

from guess import validate_configuration
from summarize import CONFIGURATION, EXPECTED, OBJECTIVES

ARRIVAL_RULES = {
    'order': 'CSV original data-record order simulates arrivals, not verified chronological order.',
    'identity': 'One-based arrival index; duplicate configurations count as separate arrivals.',
    'input': 'update(configuration, outcomes): separate dictionaries; unknown or misplaced keys rejected.',
    'missing': 'Omitted keys, None, and blank strings mean missing; no objective imputation. Numeric strings normalize to finite floats; booleans and non-finite values rejected.',
    'statistics': 'Welford incremental mean and M2 separately per objective, using observed values only. Sample variance M2/(n-1) and sample standard deviation are null for n<2; mean null for n=0.',
    'ties': 'Exact equality after float normalization; earliest arrival retained as best. best_tie_count counts all arrivals attaining the current best, including repeated configurations.',
    'causality': 'Each update reads only its arguments and current accumulators; no batch summaries, rescans, fitting, predictions, repeated tests, or stopping rules.',
    'invalid': 'Invalid arrivals raise ValueError without changing state. Finite inputs causing non-finite moment arithmetic are also rejected.',
    'snapshots': 'Returned snapshots are detached copies. Evaluator stores constant-size accumulators and best configurations; replay retains O(arrivals) history.'
}
LIMITATIONS = [
    'Row order is a simulation, not verified time order; arrivals are not asserted independent or randomized.',
    'Observed best values describe received outcomes only; no optimality, future-performance, uncertainty, significance, or stopping guarantee.',
    'Missing outcomes may be selective; each objective uses its own observed subset.',
    'Recorded metric scales and directions only; formulas, physical units, compute costs, and theoretical ranges are unspecified.',
    'Float arithmetic can round near ties and running moments; extreme magnitudes overflowing moment arithmetic are rejected.',
    'Configuration validation reuses Step 2 normalization; absent configuration fields are permitted, with no invented hyperparameter domains.',
    'No overall objective score, prediction, or Step 3 inferential procedure is applied during streaming.'
]


def normalize_outcomes(outcomes):
    """Reuse finite-number normalization, with objective-specific keys."""
    if not isinstance(outcomes, dict) or set(outcomes) - set(OBJECTIVES):
        raise ValueError('Outcomes must be a dictionary containing only objective keys')
    return {name: validate_configuration({'N_estimators': outcomes.get(name)})['N_estimators']
            for name in OBJECTIVES}


class FlowEvaluator:
    """Consume one arrival per call, using O(number of objectives) state."""

    def __init__(self):
        self._received = 0
        self._state = {name: {'observed_count': 0, 'missing_count': 0,
                             'mean': None, 'm2': 0.0, 'best': None, 'best_tie_count': 0}
                       for name in OBJECTIVES}

    def snapshot(self):
        """Return detached, JSON-ready current summaries, without mutation."""
        objectives = copy.deepcopy(self._state)
        for name, state in objectives.items():
            count = state['observed_count']
            m2 = state.pop('m2')
            variance = max(0.0, m2) / (count - 1) if count > 1 else None
            state.update(direction=OBJECTIVES[name], variance_sample=variance,
                         std_sample=math.sqrt(variance) if variance is not None else None)
        return {'configurations_received': self._received, 'objectives': objectives}

    def update(self, configuration, outcomes):
        """Validate atomically, update Welford moments, and return a snapshot."""
        config = validate_configuration(configuration)
        values = normalize_outcomes(outcomes)
        states = copy.deepcopy(self._state)
        arrival = self._received + 1
        for name, value in values.items():
            state = states[name]
            if value is None:
                state['missing_count'] += 1
                continue
            state['observed_count'] += 1
            count = state['observed_count']
            if count == 1:
                state['mean'] = value
            else:
                delta = value - state['mean']
                state['mean'] += delta / count
                state['m2'] += delta * (value - state['mean'])
                if not all(math.isfinite(state[key]) for key in ('mean', 'm2')):
                    raise ValueError('Arrival exceeds finite running-moment arithmetic')
            best = state['best']
            better = (best is None or
                      (value < best['value'] if OBJECTIVES[name] == 'minimize'
                       else value > best['value']))
            if better:
                state['best'] = {'arrival_index': arrival, 'value': value,
                                 'configuration': config.copy()}
                state['best_tie_count'] = 1
            elif value == best['value']:
                state['best_tie_count'] += 1
        self._state = states
        self._received = arrival
        return self.snapshot()


def iter_arrivals(dataset):
    """Validate the reused schema and reveal one CSV record at a time."""
    with Path(dataset).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle)
        headers = next(reader, None)
        if not headers or len(headers) != len(set(headers)) or set(headers) != EXPECTED:
            raise ValueError('Expected exactly the eight unique Health column names')
        received = False
        for line, cells in enumerate(reader, 2):
            if len(cells) != len(headers):
                raise ValueError(f'Row {line}: expected {len(headers)} cells')
            row = dict(zip(headers, cells))
            try:
                config = validate_configuration({name: row[name] for name in CONFIGURATION})
                outcomes = normalize_outcomes({name: row[name] for name in OBJECTIVES})
            except ValueError as exc:
                raise ValueError(f'Row {line}: {exc}') from exc
            received = True
            yield config, outcomes
        if not received:
            raise ValueError('Dataset must contain at least one record')


def replay(dataset):
    """Retain per-arrival snapshots; never consult later records in an update."""
    evaluator = FlowEvaluator()
    history = []
    for config, outcomes in iter_arrivals(dataset):
        snapshot = evaluator.update(config, outcomes)
        history.append({'arrival_index': snapshot['configurations_received'],
                        'configuration': config, 'observed_objectives': outcomes,
                        'summary': snapshot})
    return {'dataset_path': str(Path(dataset).resolve()),
            'configuration_fields': list(CONFIGURATION),
            'objective_directions': dict(OBJECTIVES), 'arrival_rules': dict(ARRIVAL_RULES),
            'history': history, 'final_summary': evaluator.snapshot(),
            'limitations': list(LIMITATIONS),
            'environment': {'scikit_learn_version': sklearn.__version__}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        if sklearn.__version__ != '1.6.0':
            raise ValueError('Use scikit-learn 1.6.0')
        report = replay(args.dataset)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f'Error: {exc}\n')
    print(ARRIVAL_RULES['order'])
    print(json.dumps(report['final_summary'], indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
