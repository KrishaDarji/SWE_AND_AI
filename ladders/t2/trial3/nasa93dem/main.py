"""Command-line entry point for the reusable project-history summarizer."""

import argparse
import json

from project_history.repository import CsvProjectRepository
from project_history.service import summarize_projects
from project_history.estimator import estimate_effort
from project_history.certifier import certify_less_costly
from project_history.portfolio import assess_proposals
from project_history.blame import explain_excessive_effort
from project_history.justification import justify_estimate, justification_lines
from project_history.optimizer import maximize_delivered_value
from project_history.risk import recommend_cheapest_risk_change


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize NASA93 historical delivery outcomes")
    parser.add_argument("csv_path")
    parser.add_argument("--kloc", type=float, help="proposed project size in KLOC")
    parser.add_argument("--max-effort", type=float, help="approve only when estimated effort is at most this value")
    parser.add_argument("--compare-attribute", help="CSV categorical field used as the process")
    parser.add_argument("--process-a", help="first process value")
    parser.add_argument("--process-b", help="second process value")
    parser.add_argument("--proposals", help="JSON array or newline-delimited JSON proposal file")
    parser.add_argument("--portfolio-budget", type=float, help="maximum approved effort")
    parser.add_argument("--blame", action="store_true", help="explain factors associated with excessive effort")
    parser.add_argument("--justify", action="store_true", help="justify the proposed estimate to a skeptical sponsor")
    parser.add_argument("--optimize-proposals", help="JSON proposals with id, kloc, and value")
    parser.add_argument("--value-budget", type=float, help="fixed effort budget for value optimization")
    parser.add_argument("--recommend-risk", nargs=2, metavar=("ATTRIBUTE", "CURRENT"),
                        help="recommend the cheapest lower-risk process value")
    args = parser.parse_args()
    projects = list(CsvProjectRepository(args.csv_path).projects())
    summary = summarize_projects(projects)
    print("\n".join(summary.as_lines()))
    if args.kloc is not None:
        estimate = estimate_effort(projects, args.kloc)
        if estimate is None:
            print("Estimated effort: unavailable")
            print("Approval: cannot assess")
        else:
            print(f"Proposed size (KLOC): {args.kloc:.2f}")
            print(f"Estimated effort: {estimate:.2f} person-months")
            approved = args.max_effort is None or estimate <= args.max_effort
            print(f"Approval: {'approved' if approved else 'not approved'}")
        if args.justify:
            print("Sponsor justification:")
            print("\n".join(justification_lines(justify_estimate(projects, args.kloc))))
    elif args.justify:
        parser.error("--justify requires --kloc")
    if any(value is not None for value in (args.compare_attribute, args.process_a, args.process_b)):
        if not all(value is not None for value in (args.compare_attribute, args.process_a, args.process_b)):
            parser.error("comparison requires --compare-attribute, --process-a, and --process-b")
        result = certify_less_costly(projects, args.compare_attribute, args.process_a, args.process_b)
        print(f"Cost metric: effort per KLOC")
        median_a = "unavailable" if result["median_a"] is None else f"{result['median_a']:.2f}"
        median_b = "unavailable" if result["median_b"] is None else f"{result['median_b']:.2f}"
        print(f"{args.process_a} median: {median_a}")
        print(f"{args.process_b} median: {median_b}")
        print(f"Certification: {result['verdict']}")
    if args.proposals:
        with open(args.proposals, encoding="utf-8") as handle:
            raw = handle.read()
        decoded = json.loads(raw) if raw.lstrip().startswith("[") else [json.loads(line) for line in raw.splitlines() if line.strip()]
        assessments = assess_proposals(projects, decoded, portfolio_budget=args.portfolio_budget)
        print("Portfolio assessments:")
        for item in assessments:
            effort = "unavailable" if item.estimated_effort is None else f"{item.estimated_effort:.2f}"
            print(f"{item.proposal_id}: {item.status}; estimated effort {effort}; cumulative {item.cumulative_effort if item.cumulative_effort is not None else 'unavailable'}")
    if args.optimize_proposals:
        if args.value_budget is None:
            parser.error("--optimize-proposals requires --value-budget")
        with open(args.optimize_proposals, encoding="utf-8") as handle:
            raw = handle.read()
        decoded = json.loads(raw) if raw.lstrip().startswith("[") else [json.loads(line) for line in raw.splitlines() if line.strip()]
        plan = maximize_delivered_value(projects, decoded, args.value_budget)
        print("Value-maximizing plan:")
        print(f"Selected proposals: {', '.join(plan.selected_ids) if plan.selected_ids else 'none'}")
        print(f"Delivered value: {plan.total_value:.2f}")
        print(f"Reserved effort: {plan.total_effort:.2f} / {args.value_budget:.2f}")
        print(f"Not selected: {', '.join(plan.rejected_ids) if plan.rejected_ids else 'none'}")
    if args.blame:
        explanation = explain_excessive_effort(projects)
        print(f"Excessive-effort threshold: {explanation['threshold']:.2f}")
        print(f"Excessive-effort projects: {explanation['excessive_count']}")
        print("Categorical factors:")
        for item in explanation["categorical"][:5]:
            print(f"{item['factor']}: {item['high_value']} {item['high_rate']:.2f} vs {item['low_value']} {item['low_rate']:.2f}")
        print("Numeric factors:")
        for item in explanation["numeric"]:
            print(f"{item['factor']}: Spearman {item['spearman']:.2f}")
    if args.recommend_risk:
        attribute, current = args.recommend_risk
        result = recommend_cheapest_risk_change(projects, attribute, current)
        print(f"Risk metric: median defects per KLOC; cost metric: median effort per KLOC")
        print(f"Current process: {current}; estimated risk: {result['current_risk'] if result['current_risk'] is not None else 'unavailable'}")
        recommendation = result["recommendation"]
        if recommendation is None:
            print(f"Risk-change recommendation: none ({result['reason']})")
        else:
            print(f"Risk-change recommendation: {recommendation['value']} at cost {recommendation['cost']:.2f}; risk {recommendation['risk']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
