"""Command-line entry point for the reusable project-history summarizer."""

import argparse

from project_history.repository import CsvProjectRepository
from project_history.service import summarize_projects
from project_history.estimator import estimate_effort


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize NASA93 historical delivery outcomes")
    parser.add_argument("csv_path")
    parser.add_argument("--kloc", type=float, help="proposed project size in KLOC")
    parser.add_argument("--max-effort", type=float, help="approve only when estimated effort is at most this value")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
