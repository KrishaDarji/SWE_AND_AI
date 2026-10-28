"""Compare two observed samples; demonstrate Performance- association, not quality."""

import argparse
from itertools import combinations
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np

from remember import OBJECTIVE, load_measurements

SEED = 42031
ALPHA = 0.05
MIN_EFFECT = 1.0
RESAMPLES = 19999
EXACT_LIMIT = 50000


def _sample(values):
    """Accept finite real observations, with at least two in each sample."""
    values = list(values)
    if len(values) < 2:
        raise ValueError('Each sample needs at least two observed measurements')
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
           for v in values):
        raise ValueError('Samples must contain finite real numbers, excluding booleans')
    result = np.asarray(values, dtype=float)
    if not math.isfinite(float(result.mean())):
        raise ValueError('Sample mean exceeds finite numeric range')
    return result


def compare_samples(baseline, candidate, *, direction='minimize', alpha=ALPHA,
                    min_effect=MIN_EFFECT, seed=SEED, resamples=RESAMPLES):
    """Two-sided unstudentized permutation test of difference in observed means.

    Under the null, labels must be exchangeable: independent observations from
    identical distributions. This is stronger than equal means when variances
    differ. Exact enumeration counts assignments, including tied values. Larger
    problems use seeded random permutations and the (extreme + 1)/(B + 1) rule.
    Practical relevance is an observed-effect gate, not a test or confidence
    bound that the population improvement exceeds min_effect.
    """
    a, b = _sample(baseline), _sample(candidate)
    if direction not in ('minimize', 'maximize'):
        raise ValueError('direction must be minimize or maximize')
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not 0 < alpha < 1:
        raise ValueError('alpha must be between zero and one')
    if (isinstance(min_effect, bool) or not isinstance(min_effect, Real)
            or not math.isfinite(min_effect) or min_effect < 0):
        raise ValueError('min_effect must be finite and nonnegative')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError('resamples must be a positive integer')
    mean_a, mean_b = float(a.mean()), float(b.mean())
    delta = mean_b - mean_a
    if not math.isfinite(delta):
        raise ValueError('Observed difference exceeds finite numeric range')
    # Common scaling avoids overflow without changing permutation ordering.
    pooled = np.concatenate((a, b))
    scale = max(1.0, float(np.max(np.abs(pooled))))
    pooled = pooled / scale
    n, m = len(a), len(b)
    total = float(pooled.sum())
    observed = abs(float(pooled[n:].mean() - pooled[:n].mean()))
    tolerance = 1e-12 * max(1.0, observed)
    assignments = math.comb(n + m, n)

    def extreme(indices):
        subtotal = float(pooled[list(indices)].sum())
        statistic = abs((total - subtotal) / m - subtotal / n)
        return statistic >= observed - tolerance

    if assignments <= EXACT_LIMIT:
        count = sum(extreme(indices) for indices in combinations(range(n + m), n))
        draws = assignments
        p_value = count / draws
        method = 'exact permutation'
        mc_se = None
    else:
        rng = np.random.default_rng(seed)
        count = sum(extreme(rng.permutation(n + m)[:n]) for _ in range(resamples))
        draws = resamples
        p_value = (count + 1) / (draws + 1)
        method = 'Monte Carlo permutation with plus-one correction'
        mc_se = math.sqrt(p_value * (1 - p_value) / (draws + 1))
    improvement = -delta if direction == 'minimize' else delta
    significant = p_value <= alpha
    practical = improvement > 0 and improvement >= min_effect
    credible = significant and practical
    if credible:
        conclusion = ('Statistically detected improvement with an observed effect meeting '
                      'the practical threshold, conditional on exchangeability assumptions.')
    elif not significant:
        conclusion = 'No detected difference; this is not proof of equivalence.'
    elif improvement <= 0:
        conclusion = 'Statistically detected difference is in the unfavorable direction.'
    else:
        conclusion = 'Statistically detected improvement is below the practical threshold.'
    return {
        'baseline_n': n, 'candidate_n': m, 'baseline_mean': mean_a,
        'candidate_mean': mean_b, 'difference_candidate_minus_baseline': delta,
        'direction': direction, 'observed_improvement': improvement,
        'alpha': alpha, 'minimum_practical_improvement': min_effect,
        'method': method, 'statistic': 'absolute difference of sample means',
        'alternative': 'two-sided', 'null': 'identical exchangeable distributions',
        'p_value': p_value, 'extreme_assignments': count, 'draws': draws,
        'possible_assignments': assignments, 'seed': seed,
        'monte_carlo_standard_error_approx': mc_se,
        'statistically_detected_difference': significant,
        'observed_practical_improvement': practical,
        'credible_improvement_conditional_on_assumptions': credible,
        'conclusion': conclusion,
    }


def dataset_demo(data, *, min_effect=MIN_EFFECT, resamples=RESAMPLES):
    """Fixed encoder-setting rule; never rank or filter by objective values."""
    setting = 'no_8x8dct'
    if setting not in data.settings:
        raise ValueError('Dataset demo requires the no_8x8dct encoder setting')
    index = data.settings.index(setting)
    if {row[index] for row in data.rows} != {0.0, 1.0}:
        raise ValueError('Dataset demo requires exactly setting values 0 and 1')
    groups = [[row[-1] for row in data.rows if row[index] == value] for value in (0, 1)]
    return {
        'group_selection_rule': 'Before interpretation: baseline no_8x8dct=0; candidate no_8x8dct=1; include every observed row; no objective-based selection.',
        'objective': OBJECTIVE, 'units': None, 'quality_measures': [],
        'quality_limitation': 'No separate output-quality measure is available; this certifies Performance- association only, with smaller values preferred.',
        'comparison': compare_samples(*groups, min_effect=min_effect, resamples=resamples),
        'unique_configurations': len({row[:-1] for row in data.rows}),
        'repeated_configuration_rows': len(data.rows) - len({row[:-1] for row in data.rows}),
        'assumptions_and_limitations': [
            'Independent observations and exchangeable group labels under the null of identical distributions; no normality assumption.',
            'Historical configurations are not randomized samples; dependence, heteroskedasticity, confounding and selection can invalidate the p-value interpretation.',
            'Other encoder settings may differ between groups; association does not establish causal improvement.',
            'Comparisons across different configurations do not establish repeatability for an individual configuration.',
            'Observed CSV measurements only; model predictions are not independent observed measurements and are never used here.',
            'One prespecified comparison; no multiple-comparison search or adjustment.',
            'The 1.0-unit default practical threshold is illustrative, not validated domain relevance; physical units are unspecified.',
            'Practical criterion uses the observed effect; no confidence bound or population-effect guarantee is supplied.',
            'A nonsignificant test is not evidence of equivalence; small samples have low power and coarse exact p-values.',
            'Monte Carlo p-values have sampling error; seed and draw count are recorded.',
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--min-effect', type=float, default=MIN_EFFECT)
    parser.add_argument('--resamples', type=int, default=RESAMPLES)
    args = parser.parse_args(argv)
    try:
        report = dataset_demo(load_measurements(args.dataset), min_effect=args.min_effect,
                              resamples=args.resamples)
        report['dataset'] = str(args.dataset.resolve())
        target = args.output_dir / 'certification.json'
        if target.resolve() == args.dataset.resolve():
            raise ValueError('Output must not overwrite the dataset')
        args.output_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    except (OSError, ValueError) as error:
        parser.exit(2, f'error: {error}\n')
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
