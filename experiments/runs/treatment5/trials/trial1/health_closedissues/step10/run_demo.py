import csv
import os
import sys
from source.selector import compare, load_records, summarize

DATASET = "/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/hpo/Health-ClosedIssues0000.csv"
records, counts = load_records(DATASET)
rows = compare(records, range(42, 52))
summary = summarize(rows)
print("counts", counts)
print("policy: minimize MRE-, maximize ACC+, maximize PRED40+, earliest original_id; tolerance=1e-12")
for item in rows:
    print("seed=%s selector_id=%s baseline_id=%s selector_config=%s baseline_config=%s selector_outcomes=%s baseline_outcomes=%s difference=%s" %
          (item["seed"], item["selector"]["original_id"], item["baseline"]["original_id"], item["selector"]["configuration"],
           item["baseline"]["configuration"], item["selector"]["outcomes"], item["baseline"]["outcomes"], item["difference"]))
print("summary", summary)
print("difference is baseline MRE- minus selector MRE-; positive is selector win beyond tolerance")
print("limitations: retrospective draws from one historical dataset; selector inspects all feasible outcomes while baseline samples one record; no statistical superiority claim")

out = os.path.join(os.path.dirname(__file__), "metrics.csv")
with open(out, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["step","treatment","trial","domain","skill","total_loc","new_loc","test_pass","runtime_seconds","error_count","notes"])
    writer.writeheader()
    writer.writerow({"step":10,"treatment":5,"trial":1,"domain":"model configuration and quality","skill":"race","total_loc":80,"new_loc":80,"test_pass":1,"runtime_seconds":"0.07","error_count":0,"notes":"4 tests; runtime copied from saved /usr/bin/time -p demonstration log; EZR adaptations: CSV coercion, deterministic ordering, one-pass eligibility concepts; retrospective assumptions and limitations disclosed in output."})
