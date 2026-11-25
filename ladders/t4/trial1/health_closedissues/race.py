"""Retrospective Step 7 selector versus uniform recorded feasible rows."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import random
import statistics
from time import perf_counter

import sklearn

from choose import RULE, choose_rows, record, validate_cap
from justify import normalize_records, selection_key
from summarize import CONFIGURATION, OBJECTIVES, load_rows

DEFAULT_SEEDS = tuple(range(42, 52))
LIMITATIONS = [
    'Recorded outcomes are compared retrospectively; CSV row order identifies records, not historical chronology.',
    'The exhaustive selector examines all feasible objectives; uniform selection does not optimize outcomes. This is an information and search-effort advantage.',
    "A selector win is expected by construction under its own MRE-first preference rule. It does not establish learning effectiveness, sample efficiency, or prospective improvement.",
    'N_estimators cap is a configuration constraint, not measured compute cost. Hardware and other settings also affect compute.',
    'Seed repeats remain within Treatment 4 Trial 1; they are not independent LLM trials. Repeated row selections remain separate seed draws.',
    'Raw differences, descriptive means and win counts do not establish statistical superiority. No significance claim or invented substitute test is supplied.',
    'No measurement repeatability or deployment guarantee is established; independent representative validation is needed.',
    'Missing fields exclude records without imputation; duplicate configurations remain separate rows. Metric formulas, scales and practical thresholds are unspecified.',
    'Exact normalized float equality defines outcome ties. Row identity breaks selector ties but is excluded from outcome equality.',
    'Runtime scopes differ and are stated explicitly; these timings are not controlled compute-efficiency comparisons.',
]


def validate_seeds(seeds):
    """Preserve order and duplicates as explicit repeated seed requests."""
    if not isinstance(seeds, (list, tuple)) or not seeds:
        raise ValueError('seeds must be a nonempty list or tuple of integers')
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError('seeds must contain integers, excluding booleans')
    return list(seeds)


def uniform_identity(row_ids, seed):
    """Only identities enter the fresh generator; no objective values enter."""
    return random.Random(seed).choice(row_ids)


def compare_records(candidate, baseline):
    """Compare recorded objectives independently and in the unchanged priority order."""
    c, b = candidate['observed_objectives'], baseline['observed_objectives']
    objectives = {}
    for name, direction in OBJECTIVES.items():
        delta = c[name] - b[name]
        if not math.isfinite(delta):
            raise ValueError('Nonfinite candidate-minus-baseline arithmetic')
        benefit = -delta if direction == 'minimize' else delta
        objectives[name] = {'direction': direction, 'candidate': c[name], 'baseline': b[name],
                            'candidate_minus_baseline': delta,
                            'improvement_in_objective_direction': benefit,
                            'observed_relation': 'tie' if benefit == 0 else 'win' if benefit > 0 else 'loss'}
    # selection_key is Step 7's unchanged priority; omit the final row identity.
    ck = selection_key((candidate['row_id'], c))[:-1]
    bk = selection_key((baseline['row_id'], b))[:-1]
    relation = 'tie' if ck == bk else 'win' if ck < bk else 'loss'
    return {'per_objective': objectives, 'lexicographic_relation': relation,
            'secondary_objective_losses': [n for n in ('ACC+', 'PRED40+')
                                           if objectives[n]['observed_relation'] == 'loss']}


def race_rows(rows, max_estimators=100, seeds=DEFAULT_SEEDS):
    """Reuse the Step 7 selection and its eligibility audit without tuning."""
    cap, seeds = validate_cap(max_estimators), validate_seeds(seeds)
    if not isinstance(rows, (list, tuple)):
        raise ValueError('rows must be a list or tuple')
    start = perf_counter()
    normalized = normalize_records(rows) if rows else []
    normalization_seconds = perf_counter() - start
    start = perf_counter()
    selected = choose_rows(normalized, cap)
    selector_seconds = perf_counter() - start
    start = perf_counter()
    # Reuse Step 7's exact exclusions. Only identity membership reaches the RNG.
    excluded = {item['row_id'] for item in selected['excluded_records']}
    ids = [i for i in range(1, len(normalized) + 1) if i not in excluded]
    pool_seconds = perf_counter() - start
    trials = []
    baseline_seconds = 0.0
    for seed in seeds:
        start = perf_counter()
        identity = uniform_identity(ids, seed) if ids else None
        elapsed = perf_counter() - start
        baseline_seconds += elapsed
        baseline = record(identity, normalized[identity - 1]) if identity is not None else None
        comparison = compare_records(selected['selected_record'], baseline) if baseline else None
        trials.append({'seed': seed, 'baseline_record': baseline, 'comparison': comparison,
                       'baseline_runtime_seconds': elapsed})
    comparisons = [t['comparison'] for t in trials if t['comparison'] is not None]
    def counts(relations):
        counter = Counter(relations)
        return {name: counter[name] for name in ('win', 'tie', 'loss')}
    summaries = {'compared_seed_count': len(comparisons),
                 'lexicographic': counts(c['lexicographic_relation'] for c in comparisons),
                 'per_objective': {}}
    for name in OBJECTIVES:
        deltas = [c['per_objective'][name]['candidate_minus_baseline'] for c in comparisons]
        mean = statistics.mean(deltas) if deltas else None
        if mean is not None and not math.isfinite(mean):
            raise ValueError('Nonfinite descriptive mean')
        summaries['per_objective'][name] = {
            **counts(c['per_objective'][name]['observed_relation'] for c in comparisons),
            'mean_candidate_minus_baseline': mean,
            'minimum_candidate_minus_baseline': min(deltas) if deltas else None,
            'maximum_candidate_minus_baseline': max(deltas) if deltas else None}
    report = {
        'status': selected['status'],
        'methods': {
            'candidate': {'name': 'Unmodified Step 7 exhaustive constrained selector',
                          'rule': RULE, 'outcome_access': 'All feasible recorded objectives'},
            'baseline': {'name': 'Uniform selection of one recorded feasible row per seed',
                         'rule': 'Fresh random.Random(seed).choice(feasible row IDs); duplicates are distinct records.',
                         'outcome_access': 'Outcome presence for shared eligibility only; values read after identity selection'}},
        'eligibility': {'rule': 'All five configuration fields and all three outcomes present after Step 7 normalization; finite numeric values; N_estimators <= cap.',
                        'source': 'Exact Step 7 excluded-record audit, reused by baseline without objective ranking.',
                        'counts': selected['counts'], 'baseline_feasible_count': len(ids),
                        'feasible_row_ids': ids, 'excluded_records': selected['excluded_records']},
        'cap': selected['constraint'], 'seeds': seeds, 'objective_directions': dict(OBJECTIVES),
        'candidate_record': selected['selected_record'], 'trials': trials,
        'descriptive_summaries': summaries,
        'timing': {'unit': 'seconds; time.perf_counter wall time',
                   'shared_normalization_seconds': normalization_seconds,
                   'shared_pool_materialization_seconds': pool_seconds,
                   'candidate_seconds': selector_seconds, 'baseline_total_seconds': baseline_seconds,
                   'candidate_scope': 'One choose_rows call, including repeated normalization, eligibility, exhaustive selection, tie/champion audit and report validation; excludes CSV I/O, imports, race comparisons and output.',
                   'baseline_scope': 'Per seed: fresh RNG construction and one identity choice (or empty-pool check); excludes shared pool setup, record materialization, comparisons, I/O and imports.',
                   'reuse_scope': 'Baseline pool derived from Step 7 audit; selector audit time charged only to candidate. Shared normalization and pool materialization reported separately; scopes are unequal.'},
        'statistical_analysis': {'status': 'pending', 'reason': "The project's required same comparison procedure is not supplied among allowed inputs.",
                                 'performed': False, 'claim_of_statistical_superiority': False},
        'limitations': list(LIMITATIONS),
        'environment': {'scikit_learn_version': sklearn.__version__},
        'reuse': 'Steps 1–9 application and tests retained byte-for-byte. Step 10 imports Step 7 choose_rows/record/cap validation and Step 6 normalization/priority key; reads no saved selection.'}
    json.dumps(report, allow_nan=False)
    return report


def race(dataset, max_estimators=100, seeds=DEFAULT_SEEDS):
    start = perf_counter()
    _, rows = load_rows(dataset)
    load_seconds = perf_counter() - start
    report = race_rows(rows, max_estimators, seeds)
    report['dataset_path'] = str(Path(dataset).resolve())
    report['timing']['shared_csv_load_seconds'] = load_seconds
    return report


def render_text(report):
    lines = ['Retrospective configuration race', json.dumps(report['methods']),
             'Cap (not measured compute cost): ' + json.dumps(report['cap']),
             'Seeds: ' + json.dumps(report['seeds']),
             'Eligibility: ' + report['eligibility']['rule'],
             'Counts: ' + json.dumps(report['eligibility']['counts']),
             'Status: ' + report['status'],
             'Selected candidate: ' + json.dumps(report['candidate_record'])]
    for trial in report['trials']:
        lines.append('Seed comparison: ' + json.dumps(trial))
    lines.extend(['Descriptive summaries: ' + json.dumps(report['descriptive_summaries']),
                  'Timing and scopes: ' + json.dumps(report['timing']),
                  'Statistical analysis: ' + json.dumps(report['statistical_analysis'])])
    lines.extend(report['limitations'])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--max-estimators', type=float, default=100)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if sklearn.__version__ != '1.6.0':
            raise ValueError('Use scikit-learn 1.6.0')
        report = race(args.dataset, args.max_estimators, args.seeds)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    except (OSError, ValueError, OverflowError, csv.Error) as exc:
        parser.exit(2, f'Error: {exc}\n')
    print(render_text(report), end='')


if __name__ == '__main__':
    main()
