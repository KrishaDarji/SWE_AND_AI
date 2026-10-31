import csv, json
from pathlib import Path
from source.selector import read_dataset, select_configurations

ROOT = Path(__file__).resolve().parent
DATA = Path("/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/config/X264_AllMeasurements.csv")
names, configs, outcomes = read_dataset(DATA)
by_config = dict(zip(configs, outcomes))
evaluated = []
result = select_configurations(configs, 20, lambda config: evaluated.append(config) or by_config[config])
audit_min = min(outcomes)
recommended_value = by_config[result.recommended_config]
gap = recommended_value - audit_min
payload = {"evaluation_order": list(result.evaluations), "observations": [{"id": i, "configuration": list(c), "Performance-": v} for i, c, v in result.observations], "budget_used": result.budget_used, "recommended_identifier": result.recommended_id, "recommended_configuration": list(result.recommended_config), "audit_observed_minimum": audit_min, "audit_gap": gap, "candidate_count": len(configs), "feature_names": names}
print(json.dumps(payload, sort_keys=True))
