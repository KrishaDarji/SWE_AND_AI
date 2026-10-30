import json, os
from repair import read_rows, grouped_split, Estimator, repair, validate_proposal

DATA = "/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/process/nasa93dem.csv"
def main():
    descriptors, rows, invalid = read_rows(DATA)
    train, test = grouped_split(rows)
    if not test: raise ValueError("held-out set is empty")
    model = Estimator(descriptors, train)
    proposal_row = min(test, key=lambda row: row["csv_line"])
    proposal = dict(proposal_row["descriptors"])
    validate_proposal(proposal, descriptors)
    result = repair(proposal, 100, model)
    result.update({"train_rows": len(train), "test_rows": len(test), "invalid_targets": invalid,
                   "proposal_csv_line": proposal_row["csv_line"], "descriptors": descriptors})
    print(json.dumps(result, sort_keys=True))
if __name__ == "__main__": main()
