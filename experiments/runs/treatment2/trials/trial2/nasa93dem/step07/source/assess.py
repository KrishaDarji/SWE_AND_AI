#!/usr/bin/env python3
"""Assess an incoming proposal against portfolio limits."""

import argparse
import json
from project_summary import assess_proposal, load_projects


def main():
    parser = argparse.ArgumentParser(description="Assess an incoming project proposal")
    parser.add_argument("csv_path")
    parser.add_argument("proposal_id")
    parser.add_argument("proposed_kloc", type=float)
    parser.add_argument("remaining_budget", type=float)
    parser.add_argument("remaining_capacity", type=float)
    parser.add_argument("--neighbors", type=int, default=7)
    args = parser.parse_args()
    result = assess_proposal(load_projects(args.csv_path), args.proposal_id,
                             args.proposed_kloc, args.remaining_budget,
                             args.remaining_capacity, args.neighbors)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
