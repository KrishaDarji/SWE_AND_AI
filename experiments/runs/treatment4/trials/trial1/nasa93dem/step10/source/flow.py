"""Sequential effort assessment: predict, then reveal, then update history."""
import argparse
import copy
import csv
import json
import math
from pathlib import Path

import sklearn

from predict_effort import estimate, fit_training, normalize_proposal
from summarize import PREDICTORS, numeric_summary, read_projects

MIN_HISTORY = 2
LIMITATIONS = [
    'CSV order is a simulated arrival sequence, not verified chronological order or actual proposal-time data.',
    'Predictor availability at proposal time is assumed, not demonstrated by this historical dataset.',
    'Effort units are unspecified. No automatic project approval or uncertainty guarantees are provided.',
    'Earlier identical normalized predictor configurations may inform later arrivals once their efforts are revealed.',
    'Sequential cumulative MAE is distinct from Step 2 independent grouped holdout evaluation; duplicates are not held out here.',
    'Every present outcome is assumed revealed immediately after its prediction; real reporting delays are not modeled.',
    'Missing outcomes are counted but neither scored nor added to training history; delayed backfill is not supported.',
    'Two observed efforts are a deterministic availability threshold, not evidence of model reliability.',
    'Small observational sample, missingness, distribution changes, and novel categories can impair accuracy.',
    'The seeded 200-tree forest and learned preprocessing are refitted on prior observed records for every available prediction.',
    'Unseen categories encode as all zeros; missing predictors use the existing explicit empty-string level.',
    'Kloc+, Defects-, Effort-, all X-suffixed metadata and unknown fields are excluded from proposal predictors.',
]


class ProposalFlow:
    """One pending arrival at a time; only reveal() accepts an observed effort."""

    def __init__(self):
        self._history = []
        self._assessments = []
        self._pending = None
        self._errors = []
        self._observed = []

    def receive(self, proposal):
        """Record prediction using strictly prior revealed history, before any outcome."""
        if self._pending is not None:
            raise ValueError('Reveal the pending arrival before receiving another')
        values = normalize_proposal(proposal)
        normalized = dict(zip(PREDICTORS, values))
        history_size = len(self._history)
        prediction = None
        if history_size >= MIN_HISTORY:
            model, _ = fit_training(self._history)
            prediction = estimate(model, normalized)
        self._pending = {
            'row_index': len(self._assessments),
            'predictors': normalized,
            'history_size_before': history_size,
            'warm_up': history_size < MIN_HISTORY,
            'prediction_status': 'unavailable_insufficient_history' if prediction is None else 'available',
            'predicted_effort': prediction,
        }
        return copy.deepcopy(self._pending)

    def reveal(self, observed_effort):
        """Attach current outcome after recorded prediction; missing means no update."""
        if self._pending is None:
            raise ValueError('Receive an arrival before revealing its outcome')
        if observed_effort is not None and (
            isinstance(observed_effort, bool) or not isinstance(observed_effort, (int, float))
            or not math.isfinite(observed_effort) or observed_effort < 0
        ):
            raise ValueError('Observed effort must be finite, nonnegative, or None')
        item = self._pending
        prediction = item['predicted_effort']
        error = None
        if observed_effort is not None:
            observed_effort = float(observed_effort)
            if prediction is not None:
                error = abs(prediction - observed_effort)
                self._errors.append(error)
            self._observed.append(observed_effort)
            self._history.append(dict(item['predictors'], **{'Effort-': observed_effort}))
        item.update({
            'observed_effort': observed_effort,
            'outcome_status': 'missing' if observed_effort is None else 'observed',
            'absolute_error': error,
            'history_size_after': len(self._history),
            'scored_arrivals': len(self._errors),
            'cumulative_absolute_error': sum(self._errors),
            'cumulative_mae': sum(self._errors) / len(self._errors) if self._errors else None,
            'running_error_summary': numeric_summary(self._errors),
            'running_observed_effort_summary': numeric_summary(self._observed),
        })
        self._assessments.append(item)
        self._pending = None
        return copy.deepcopy(item)

    def report(self):
        """Return a detached report of completed arrivals; reject unfinished arrivals."""
        if self._pending is not None:
            raise ValueError('Pending prediction has no revealed outcome yet')
        count = len(self._assessments)
        return copy.deepcopy({
            'target': 'Effort-', 'units': 'unspecified', 'predictor_fields': PREDICTORS,
            'arrival_rule': 'Original CSV data row order; zero-based row_index excludes header. Simulated arrival sequence, not verified chronological order or actual proposal-time data.',
            'warm_up_policy': {'minimum_prior_observed_efforts': MIN_HISTORY,
                               'rule': 'No prediction until two prior observed outcomes are available; missing outcomes do not advance warm-up. No external history or future fitting.'},
            'event_order': ['receive allowed predictors', 'fit prior observed history and record prediction',
                            'reveal current effort', 'update history and running summaries'],
            'identical_configurations': 'Allowed from prior revealed history; unlike the separate Step 2 grouped holdout.',
            'model': {'implementation': 'predict_effort.fit_training and estimate, unchanged from Step 3',
                      'random_state': 42, 'n_estimators': 200, 'min_samples_leaf': 2,
                      'n_jobs': 1, 'scikit_learn_version': sklearn.__version__},
            'assessments': self._assessments,
            'final_summary': {
                'arrivals': count, 'observed_outcomes': len(self._observed),
                'missing_outcomes': count - len(self._observed),
                'available_predictions': sum(r['predicted_effort'] is not None for r in self._assessments),
                'warm_up_arrivals': sum(r['warm_up'] for r in self._assessments),
                'scored_arrivals': len(self._errors), 'history_size': len(self._history),
                'cumulative_absolute_error': sum(self._errors),
                'cumulative_mae': sum(self._errors) / len(self._errors) if self._errors else None,
                'prediction_error_summary': numeric_summary(self._errors),
                'observed_effort_summary': numeric_summary(self._observed),
            },
            'limitations': LIMITATIONS,
        })


def replay(rows):
    """Consume rows sequentially without accessing each target until receive returns."""
    flow = ProposalFlow()
    for row in rows:
        flow.receive({field: row.get(field) for field in PREDICTORS})
        flow.reveal(row.get('Effort-'))
    return flow.report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        report = replay(read_projects(args.dataset))
        report['dataset_path'] = str(args.dataset.resolve())
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(json.dumps({'arrival_rule': report['arrival_rule'], 'units': report['units'],
                          'warm_up_policy': report['warm_up_policy'],
                          'final_summary': report['final_summary'],
                          'limitations': report['limitations']}, indent=2, allow_nan=False))
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f'Input error: {exc}\n')


if __name__ == '__main__':
    main()
