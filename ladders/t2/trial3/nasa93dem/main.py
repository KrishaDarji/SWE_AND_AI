"""Command-line entry point for the reusable project-history summarizer."""

import argparse
import json

from project_history.repository import CsvProjectRepository
from project_history.service import summarize_projects
from project_history.estimator import estimate_effort
from project_history.certifier import certify_less_costly
from project_history.portfolio import assess_proposals


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
