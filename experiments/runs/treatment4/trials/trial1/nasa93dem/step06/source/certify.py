"""Compare observed independent historical effort groups; no causal certification."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import t

from summarize import numeric_summary, read_projects

SEED = 42
ALPHA = 0.05
BOOTSTRAPS = 20000
LIMITATIONS = [
    'Observational groups, not randomized process assignments: project size (Kloc+) and other characteristics may confound raw effort differences. No causal cost savings are established.',
    'Effort- is minimized; physical units are unspecified. Only observed outcomes are used, never model predictions.',
    'Welch inference assumes independent projects, representative sampling, finite variances, and approximately normal sampling distributions of means. Related projects, skew, extreme values, and small samples can undermine it.',
    'Unequal group variances are allowed. Historical convenience samples do not establish generalizability to future proposals.',
    'Missing outcomes are excluded without imputation; selection bias remains possible.',
    'The practical threshold is an illustrative planning rule: 10% of the observed baseline mean, not a manager-approved economic requirement. It is data-dependent; uncertainty in this threshold is not propagated.',
    'No detected difference is not proof of equivalence. No equivalence test or causal adjustment is performed.',
    'This does not complete the separate required same analysis: its exact procedure was not supplied.',
    'Bootstrap percentile intervals are a sensitivity description, not a second decision test; they can be unreliable with small or skewed samples.',
]


def clean_sample(values):
    """None means missing; reject invalid observed efforts rather than silently drop."""
    observed = []
    missing = 0
    for value in values:
        if value is None:
            missing += 1
        elif isinstance(value, (bool, str)) or not isinstance(value, (int, float, np.number)):
            raise ValueError('Observed effort must be a finite nonnegative number')
        elif not math.isfinite(value) or value < 0:
            raise ValueError('Observed effort must be a finite nonnegative number')
        else:
            observed.append(float(value))
    if len(observed) < 2:
        raise ValueError('Each group needs at least two observed efforts')
    return np.array(observed), missing


def compare_effort(baseline, candidate, practical_fraction=0.10):
    """Welch two-sided mean comparison; negative candidate-minus-baseline is better."""
    if isinstance(practical_fraction, bool) or not isinstance(practical_fraction, (int, float)) or not math.isfinite(practical_fraction) or not 0 < practical_fraction < 1:
        raise ValueError('Practical fraction must be finite and strictly between zero and one')
    a, missing_a = clean_sample(baseline)
    b, missing_b = clean_sample(candidate)
    avg_a, avg_b = float(a.mean()), float(b.mean())
    difference = avg_b - avg_a
    va, vb = float(a.var(ddof=1)) / len(a), float(b.var(ddof=1)) / len(b)
    se = math.sqrt(va + vb)
    threshold = practical_fraction * avg_a
    if se > 0:
        df = (va + vb) ** 2 / (va * va / (len(a) - 1) + vb * vb / (len(b) - 1))
        statistic = difference / se
        p_value = float(2 * t.sf(abs(statistic), df))
        margin = float(t.ppf(1 - ALPHA / 2, df)) * se
        interval = [difference - margin, difference + margin]
        detected = p_value < ALPHA
        evidence = ('candidate_lower' if difference < 0 else 'candidate_higher') if detected else 'no_detected_difference'
        practical_supported = threshold > 0 and interval[1] < -threshold
    else:
        df = statistic = p_value = interval = None
        evidence = 'inference_unavailable_zero_variance'
        practical_supported = False
    rng = np.random.default_rng(SEED)
    # Independent within-group resampling, preserving each group's sample size.
    boot_a = rng.choice(a, size=(BOOTSTRAPS, len(a)), replace=True).mean(axis=1)
    boot_b = rng.choice(b, size=(BOOTSTRAPS, len(b)), replace=True).mean(axis=1)
    bootstrap_ci = np.quantile(boot_b - boot_a, [0.025, 0.975]).tolist()
    conclusion = {
        'candidate_lower': 'Candidate projects have statistically lower observed mean effort under Welch assumptions.',
        'candidate_higher': 'Candidate projects have statistically higher observed mean effort under Welch assumptions.',
        'no_detected_difference': 'A difference in population mean effort was not detected; this is not proof of equivalence.',
        'inference_unavailable_zero_variance': 'Both samples have zero variance; Welch inference is unavailable.',
    }[evidence]
    conclusion += (' The interval also supports the illustrative practical reduction.' if practical_supported else ' The interval does not establish the illustrative practical reduction.')
    conclusion += ' These observations do not establish that changing the process causes cost savings.'
    return {
        'target': 'Effort-', 'units': 'unspecified',
        'baseline': numeric_summary(a.tolist()) | {'excluded_missing_outcomes': missing_a},
        'candidate': numeric_summary(b.tolist()) | {'excluded_missing_outcomes': missing_b},
        'difference_candidate_minus_baseline': difference,
        'relative_mean_reduction': (avg_a - avg_b) / avg_a if avg_a > 0 else None,
        'observed_direction': 'lower' if difference < 0 else 'higher' if difference > 0 else 'equal',
        'statistical': {'method': 'Welch independent two-sample two-sided t test and 95% t confidence interval',
                        'null': 'population mean candidate effort equals population mean baseline effort',
                        'alpha': ALPHA, 'standard_error': se, 'degrees_of_freedom': df,
                        't_statistic': statistic, 'p_value_two_sided': p_value,
                        'mean_difference_ci95': interval, 'evidence': evidence},
        'practical': {'fraction_of_baseline_mean': practical_fraction, 'absolute_reduction_threshold': threshold,
                      'criterion': 'candidate mean is at least the threshold below baseline mean; strict CI upper bound below negative threshold supports it',
                      'observed_threshold_met': threshold > 0 and difference <= -threshold,
                      'supported_by_ci95': practical_supported,
                      'zero_baseline_note': 'A zero baseline cannot support a positive relative reduction.'},
        'bootstrap_sensitivity': {'method': 'independent within-group percentile bootstrap of mean difference',
                                  'seed': SEED, 'resamples': BOOTSTRAPS, 'ci95': bootstrap_ci},
        'small_sample_warning': len(a) < 10 or len(b) < 10,
        'versions': {'numpy': np.__version__, 'scipy': scipy.__version__},
        'manager_conclusion': conclusion, 'limitations': LIMITATIONS,
    }


def certify(dataset):
    rows = read_projects(dataset)
    report = compare_effort([r['Effort-'] for r in rows if r['tool'] == 'n'],
                            [r['Effort-'] for r in rows if r['tool'] == 'h'])
    report['dataset_path'] = str(Path(dataset).resolve())
    report['group_definitions'] = {'baseline': 'tool=n', 'candidate': 'tool=h',
                                   'selection': 'Fixed before interpreting outcomes; all observed efforts in these groups',
                                   'excluded_other_or_missing_tool_rows': sum(r['tool'] not in ('n', 'h') for r in rows)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        report = certify(args.dataset)
        text = json.dumps(report, indent=2, allow_nan=False) + '\n'
        args.output.write_text(text, encoding='utf-8')
        print(text, end='')
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f'Input error: {exc}\n')


if __name__ == '__main__':
    main()
