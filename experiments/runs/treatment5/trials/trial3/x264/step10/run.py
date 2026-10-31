import csv, os, sys, time
from source.tuner import load_dataset, compare

ROOT=os.path.dirname(os.path.abspath(__file__))
DATA="/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/config/X264_AllMeasurements.csv"
def main():
    rows=load_dataset(DATA); results=compare(rows)
    with open(os.path.join(ROOT,"metrics_detail.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["seed","strategy","order","budget","recommendation","configuration","performance","gap","outcome"])
        for ezr, base, gap, outcome in results:
            for name, item in (("ezr",ezr),("baseline",base)):
                rec=item["recommendation"]
                w.writerow([item["seed"],name," ".join(item["order"]),item["budget"],rec["id"]," ".join(map(str,rec["x"])),"%.6f"%rec["y"],"%.6f"%gap if name=="ezr" else "",outcome if name=="ezr" else ""])
    print("dataset=%d candidates budget=20 seeds=42..51 tolerance=%.1e" % (len(rows), 1e-12))
    for ezr, base, gap, outcome in results:
        print("seed=%d ezr=%s %.6f baseline=%s %.6f difference=%.6f %s" % (ezr["seed"],ezr["recommendation"]["id"],ezr["recommendation"]["y"],base["recommendation"]["id"],base["recommendation"]["y"],gap,outcome))
    wins=sum(x[3]=="win" for x in results); ties=sum(x[3]=="tie" for x in results); losses=sum(x[3]=="loss" for x in results)
    print("paired arithmetic: differences are EZR minus baseline; wins=%d ties=%d losses=%d mean=%.6f" % (wins,ties,losses,sum(x[2] for x in results)/len(results)))
    print("retrospective gaps: recommendations frozen before audit; full-dataset audit pending")
if __name__ == "__main__": main()
