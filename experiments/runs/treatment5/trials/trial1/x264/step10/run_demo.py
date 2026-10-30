import csv, os
from source.tuner import compare
DATA=os.environ["X264_DATA"]
out,summary=compare(DATA)
by={(x["seed"],x["strategy"]):x for x in out}
for seed in range(42,52):
    a,b=by[(seed,"race")],by[(seed,"random")]
    print("seed=%d race_order=%s random_order=%s race_recommendation=%s race_config=%s race_performance=%.3f random_recommendation=%s random_config=%s random_performance=%.3f difference=%.3f" %
          (seed,a["order"],b["order"],a["recommended"]["id"],a["recommended"]["config"],a["measured"],b["recommended"]["id"],b["recommended"]["config"],b["measured"],a["measured"]-b["measured"]))
print("summary=%s" % summary)
os.makedirs(os.path.dirname(os.environ["METRICS_OUT"]),exist_ok=True)
with open(os.environ["METRICS_OUT"],"w",newline="") as f:
    w=csv.writer(f); w.writerow(["seed","strategy","evaluation_order","recommended_identifier","configuration","measured_Performance-"])
    for x in out: w.writerow([x["seed"],x["strategy"]," ".join(map(str,x["order"])),x["recommended"]["id"]," ".join(map(str,x["recommended"]["config"])),"%.3f"%x["measured"]])
