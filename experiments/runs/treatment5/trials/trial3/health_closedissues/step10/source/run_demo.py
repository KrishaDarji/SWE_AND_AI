import csv, json, os, sys
from selector import comparison, CONFIG, OUTCOMES

path = sys.argv[1]
result = comparison(path)
print(json.dumps({k: v for k, v in result.items() if k not in ("records", "selector")}, sort_keys=True))
print("selector_original_record", result["selector"][0])
for seed, chosen, base, diff in result["records"]:
    print("seed", seed, "selector_record", chosen[0], "baseline_record", base[0],
          "selector_config", [chosen[1][k] for k in CONFIG], "baseline_config", [base[1][k] for k in CONFIG],
          "selector_outcomes", [chosen[1][k] for k in OUTCOMES], "baseline_outcomes", [base[1][k] for k in OUTCOMES],
          "paired_MRE_difference", diff)
