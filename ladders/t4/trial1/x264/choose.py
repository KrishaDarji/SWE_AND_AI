"""Budgeted sequential selection with settings-only inputs and a counted oracle."""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import random

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor

from flow import Flow, _number
from remember import OBJECTIVE

QUALITY_LIMITATION = ('No separate quality measurement exists in this dataset. The quality '
                      'constraint cannot be verified; no quality values or satisfaction claim are supplied.')
STRATEGY = {
    'name': 'Sequential random-forest lower-confidence heuristic with exploration',
    'initial_points': 5,
    'initial_rule': 'Seeded uniform sampling without replacement; all points count toward budget.',
    'surrogate': 'RandomForestRegressor; refit only on evaluated settings and outcomes.',
    'trees': 64,
    'acquisition': 'Minimize tree-prediction mean minus one population standard deviation.',
    'exploration': 'After initialization, every fifth evaluation is uniform among remaining rows.',
    'ties': 'Acquisition ties: earliest CSV row. Observed objective ties: first evaluated.',
    'budget_rule': 'Positive integer at most candidate count; evaluate exactly budget distinct rows.',
    'uncertainty': 'Tree spread is an exploration heuristic, not a calibrated confidence bound.',
}


@dataclass(frozen=True)
class Candidate:
    """Data-row identity and settings only; no objective field."""

    data_row: int
    values: tuple[float, ...]


def positive_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{label} must be a positive integer')
    return value


def load_candidates(path):
    """Read every setting; skip objective cells without parsing or validating them.

    CSV framing necessarily reads raw cells, but no target cell is returned,
    converted, summarized, ranked, or passed to the selector.
    """
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if (not header or any(not name.strip() for name in header)
                or len(set(header)) != len(header) or OBJECTIVE not in header):
            raise ValueError('Need distinct nonempty column names including Performance-')
        settings = tuple(name for name in header if name != OBJECTIVE)
        Flow(settings)
        indices = [header.index(name) for name in settings]
        candidates = []
        for row_id, cells in enumerate(reader, 1):
            if len(cells) != len(header):
                raise ValueError(f'Data row {row_id}: wrong field count')
            values = tuple(_number(float(cells[i])) for i in indices)
            candidates.append(Candidate(row_id, values))
    if not candidates:
        raise ValueError('Need at least one candidate')
    return settings, tuple(candidates)


class CSVOracle:
    """Trusted evaluation adapter, isolated from selection inputs and audit.

    Only __call__ reads a requested objective. Each accepted call is charged
    before reading, including a failed parse. No bulk-target accessor exists.
    This is an API information boundary, not a hostile-code Python sandbox.
    """

    def __init__(self, path, budget):
        self.__path = Path(path)
        self.__budget = positive_integer(budget, 'budget')
        self.__count = 0
        self.__seen = set()

    @property
    def count(self):
        return self.__count

    def __call__(self, data_row):
        positive_integer(data_row, 'data_row')
        if self.__count >= self.__budget:
            raise ValueError('Evaluation budget exhausted')
        if data_row in self.__seen:
            raise ValueError('Candidate already evaluated')
        self.__count += 1
        self.__seen.add(data_row)
        with self.__path.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.reader(stream)
            header = next(reader)
            target_index = header.index(OBJECTIVE)
            for index, cells in enumerate(reader, 1):
                if index == data_row:
                    return _number(float(cells[target_index]))
        raise ValueError('Requested data row does not exist')


def select(settings, candidates, evaluate, *, budget, seed):
    """Return best observed row; never accepts full targets, models, or audit data.

    evaluate(data_row) is the sole objective capability. Every call is included
    in history on successful runs; failed evaluations abort without a selection.
    """
    positive_integer(budget, 'budget')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    settings, candidates = tuple(settings), tuple(candidates)
    tracker = Flow(settings)
    if not candidates or budget > len(candidates):
        raise ValueError('budget must not exceed the nonempty candidate count')
    ids = []
    for candidate in candidates:
        if not isinstance(candidate, Candidate):
            raise ValueError('Expected settings-only Candidate records')
        positive_integer(candidate.data_row, 'data_row')
        if len(candidate.values) != len(settings):
            raise ValueError('Candidate settings width does not match schema')
        for value in candidate.values:
            _number(value)
        ids.append(candidate.data_row)
    if len(set(ids)) != len(ids):
        raise ValueError('Candidate data rows must be distinct')
    candidates = tuple(sorted(candidates, key=lambda c: c.data_row))
    rng = random.Random(seed)
    remaining = list(range(len(candidates)))
    initial = rng.sample(remaining, min(STRATEGY['initial_points'], budget))
    observed_x, observed_y, history = [], [], []
    best = None
    for step in range(budget):
        acquisition = None
        if step < len(initial):
            index, reason = initial[step], 'initial uniform'
        elif (step + 1) % 5 == 0:
            index, reason = rng.choice(remaining), 'uniform exploration'
        else:
            model = RandomForestRegressor(n_estimators=STRATEGY['trees'],
                                          random_state=seed, n_jobs=1)
            model.fit(observed_x, observed_y)
            predictions = np.asarray([tree.predict([candidates[i].values for i in remaining])
                                      for tree in model.estimators_])
            scores = predictions.mean(axis=0) - predictions.std(axis=0)
            position = int(np.argmin(scores))
            index, reason = remaining[position], 'surrogate mean minus tree stddev'
            acquisition = float(scores[position])
        candidate = candidates[index]
        # This is the only objective access in selection, initial points included.
        value = _number(evaluate(candidate.data_row))
        remaining.remove(index)
        observed_x.append(candidate.values)
        observed_y.append(value)
        config = dict(zip(settings, candidate.values))
        summary = tracker.update(config, value)
        if best is None or value < best[OBJECTIVE]:
            best = {'data_row': candidate.data_row, 'settings': config, OBJECTIVE: value}
        history.append({'evaluation': step + 1, 'data_row': candidate.data_row,
                        'settings': config, OBJECTIVE: value, 'reason': reason,
                        'acquisition_score': acquisition,
                        'best_so_far': {**best, 'settings': dict(best['settings'])},
                        'observed_mean': summary['mean']})
    return {'strategy': dict(STRATEGY), 'budget': budget, 'seed': seed,
            'actual_evaluation_count': len(history), 'candidate_count': len(candidates),
            'features': list(settings), 'objective': OBJECTIVE, 'direction': 'minimize',
            'units': None, 'evaluation_history': history, 'selected_candidate': best,
            'quality_constraint_limitation': QUALITY_LIMITATION,
            'quality_constraint_verified': False,
            'claim': 'Best evaluated candidate within budget; global optimality is not claimed.'}


def post_selection_audit(path, frozen_selection_json):
    """Separate full-target access after selection serialization; never fit or select.

    Receive an immutable JSON string and return audit facts without modifying
    selection. The preserved eager loader is safe only in this post-selection role.
    """
    from remember import load_measurements

    selection = json.loads(frozen_selection_json)
    data = load_measurements(path)
    selected = selection['selected_candidate']
    row = data.rows[selected['data_row'] - 1]
    if (dict(zip(data.settings, row[:-1])) != selected['settings']
            or row[-1] != selected[OBJECTIVE]):
        raise ValueError('Selected observation does not match audit dataset')
    minimum = min(row[-1] for row in data.rows)
    gap = selected[OBJECTIVE] - minimum
    return {'phase': 'After selection frozen; full-target access used only for this audit.',
            'selection_sha256': hashlib.sha256(frozen_selection_json.encode()).hexdigest(),
            'dataset_minimum': minimum, 'selected_value': selected[OBJECTIVE],
            'absolute_gap': gap,
            'gap_percent_of_minimum': 100 * gap / minimum if minimum > 0 else None,
            'rows_inspected': len(data.rows), 'fed_back_into_selection': False,
            'quality_constraint_limitation': QUALITY_LIMITATION}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--budget', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    try:
        output = args.output_dir / 'selection.json'
        if output.resolve() == args.dataset.resolve():
            raise ValueError('Output must not overwrite the dataset')
        settings, candidates = load_candidates(args.dataset)
        oracle = CSVOracle(args.dataset, args.budget)
        result = select(settings, candidates, oracle, budget=args.budget, seed=args.seed)
        if oracle.count != result['actual_evaluation_count']:
            raise RuntimeError('Oracle and selector evaluation counts disagree')
        result['dataset'] = str(args.dataset.resolve())
        result['versions'] = {'python': platform.python_version(), 'scikit_learn': sklearn.__version__}
        frozen = json.dumps(result, sort_keys=True, allow_nan=False)
        # Freeze on disk before the first full-target audit access.
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        result['post_selection_audit'] = post_selection_audit(args.dataset, frozen)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    except (OSError, ValueError) as error:
        parser.exit(2, f'error: {error}\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
