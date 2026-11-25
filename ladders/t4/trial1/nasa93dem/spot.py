"""Exploratory retrospective screening of nonnegative delivered outcomes."""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median

from predict_effort import normalize_proposal
from summarize import METADATA, OBJECTIVES, PREDICTORS, numeric_summary, read_projects

FIELDS = list(OBJECTIVES)
THRESHOLD = 3.5
SCALE_FLOOR = math.log(2)
LIMITATIONS = [
    'Flags are exploratory investigation priorities, not confirmed errors or poor performance.',
    'No verified anomaly labels: precision, recall and detection accuracy cannot be estimated.',
    'Marginal outcome screening misses unusual combinations and configuration-conditional inconsistencies.',
    'Large effort or defects may reflect scope and size; this method does not adjust for them.',
    'Full historical reference includes each evaluated record; these are not unseen prediction errors.',
    'Median/MAD can mask clusters; small references and missing outcomes weaken comparability.',
    'The 3.5 threshold is a heuristic, not a probability or calibrated false-positive rate; maximum over three fields adds multiplicity.',
    'The log(2) scale floor is an explicit conservative practical convention on 1+value, sensitive to unspecified measurement units.',
    'Physical units are unspecified. No causal, success, failure, or proposal-time prediction claims.',
    'No categorical ordinal mappings, supervised predictions, record deletion or repair are performed.',
]
CHECKS = [
    'Verify original source, transcription, missing entries, field definitions and measurement units.',
    'Compare scope, delivered size, complexity, reuse and reporting coverage with comparable projects.',
    'Check whether effort and defects cover the same lifecycle and deliverables as size.',
    'Ask project owners about legitimate exceptional circumstances before interpreting a flag.',
]


def screen(rows, threshold=THRESHOLD):
    """Return every row; rank by descending score, ties by one-based CSV data row."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError('Threshold must be finite and positive')
    if not isinstance(rows, (list, tuple)) or len(rows) < 3 or any(not isinstance(r, dict) for r in rows):
        raise ValueError('Need at least three project dictionaries')
    for row in rows:
        for field in FIELDS:
            v = row.get(field)
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0):
                raise ValueError(f'{field}: expected finite nonnegative number or None')
    normalized = [dict(zip(PREDICTORS, normalize_proposal({p: r.get(p) for p in PREDICTORS}))) for r in rows]
    counts = {p: Counter(r[p] for r in normalized) for p in PREDICTORS}
    reference = {}
    for field in FIELDS:
        raw = [r[field] for r in rows if r.get(field) is not None]
        log = [math.log1p(v) for v in raw]
        center = median(log) if log else None
        mad = median([abs(v - center) for v in log]) if log else None
        reference[field] = {'observed_summary': numeric_summary(raw), 'nonmissing_count': len(raw),
                            'missing_count': len(rows) - len(raw), 'usable': len(raw) >= 3,
                            'log_median': center, 'log_mad': mad,
                            'scale': max(1.4826 * mad, SCALE_FLOOR) if mad is not None else None}
    if not any(r['usable'] for r in reference.values()):
        raise ValueError('Need at least three observed values in an outcome field')
    findings = []
    for number, row in enumerate(rows, 1):
        evidence = []
        for field, ref in reference.items():
            v = row.get(field)
            if v is None or not ref['usable']:
                continue
            signed = (math.log1p(v) - ref['log_median']) / ref['scale']
            evidence.append({'field': field, 'observed': v, 'signed_score': signed,
                             'absolute_score': abs(signed), 'crosses_threshold': abs(signed) > threshold,
                             'direction': 'above' if signed > 0 else 'below' if signed < 0 else 'at log median',
                             'reference_raw_median': ref['observed_summary']['median']})
        score = max((e['absolute_score'] for e in evidence), default=None)
        findings.append({'csv_data_row': number, 'score': score,
                         'flagged': score is not None and score > threshold,
                         'status': 'scored' if score is not None else 'unscored: no usable observed outcomes',
                         'supporting_evidence': sorted(evidence, key=lambda e: (-e['absolute_score'], e['field'])),
                         'observed_characteristics': normalized[number - 1],
                         'category_support_counts': {p: counts[p][normalized[number - 1][p]] for p in PREDICTORS},
                         'observed_outcomes': {f: row.get(f) for f in FIELDS},
                         'descriptive_metadata': {f: row.get(f) for f in METADATA}})
    findings.sort(key=lambda r: (r['score'] is None, -(r['score'] or 0), r['csv_data_row']))
    for rank, finding in enumerate(findings, 1):
        finding['rank'] = rank
    return {
        'method_settings': {'name': 'Maximum absolute robust log outcome score',
                            'rationale': 'Transparent robust marginal tails on skewed delivered outcomes; no ordinal assumptions or prediction leakage.',
                            'inputs': FIELDS, 'preprocessing': 'log1p of observed nonnegative outcomes; full-history median and MAD per field',
                            'mad_multiplier': 1.4826, 'scale_floor': SCALE_FLOOR,
                            'fit_scope': 'All observed historical rows, including the row being screened',
                            'randomness': 'None; deterministic', 'minimum_observations_per_field': 3},
        'field_roles': {'retrospective_scoring_outcomes': FIELDS,
                        'candidate_proposal_time_fields_context_only': PREDICTORS,
                        'proposal_availability': 'Assumed; must be verified operationally',
                        'excluded_from_scores': 'All identifiers, all X-suffixed metadata, categorical predictors and unknown fields',
                        'metadata_context_only': METADATA},
        'score_definition': 'max over usable observed fields of abs(log1p(value)-median(log1p(reference)))/max(1.4826*MAD,log(2)); higher is more unusual',
        'threshold_definition': {'value': threshold, 'comparison': 'strictly greater than; equality is not flagged',
                                 'selection': 'Fixed heuristic before viewing rankings; not fitted to select a count'},
        'index_convention': 'One-based CSV data row excluding header; unrelated to idX',
        'tie_rule': 'Equal score sorted by ascending CSV data row; unscored rows last',
        'missing_data_handling': 'No outcome imputation; per-field references omit missing; fields with fewer than three values omitted from scores; no scored fields gives null score and no flag. Predictor missing tokens normalized for context only.',
        'reference': reference, 'ranked_findings': findings,
        'record_count': len(rows), 'flagged_count': sum(r['flagged'] for r in findings),
        'unscored_count': sum(r['score'] is None for r in findings),
        'suggested_manager_checks': CHECKS, 'limitations': LIMITATIONS,
    }


def render_text(report, limit=10):
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError('Display limit must be a nonnegative integer')
    lines = [f"Exploratory historical screening: {report['flagged_count']} flags / {report['record_count']} rows.",
             report['score_definition'], f"Threshold: score > {report['threshold_definition']['value']}; units unspecified.",
             f'Displaying {min(limit, report["record_count"])} rows; display limit never changes flags or JSON rankings.']
    for r in report['ranked_findings'][:limit]:
        lines.append(f"Rank {r['rank']}, CSV data row {r['csv_data_row']}: score={r['score']}, flagged={r['flagged']}")
        lines.append('  Outcomes: ' + json.dumps(r['observed_outcomes'], sort_keys=True))
        lines.append('  Characteristics: ' + json.dumps(r['observed_characteristics'], sort_keys=True))
        for e in r['supporting_evidence']:
            lines.append(f"  {e['field']}: observed={e['observed']}, raw historical median={e['reference_raw_median']}, signed log score={e['signed_score']:.6f}, {e['direction']}.")
    lines.extend(CHECKS + LIMITATIONS)
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--limit', default=10, type=int)
    parser.add_argument('--threshold', default=THRESHOLD, type=float)
    args = parser.parse_args()
    try:
        report = screen(read_projects(args.dataset), args.threshold)
        report['dataset_path'] = str(args.dataset.resolve())
        text = render_text(report, args.limit)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(text, end='')
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f'Input error: {exc}\n')


if __name__ == '__main__':
    main()
