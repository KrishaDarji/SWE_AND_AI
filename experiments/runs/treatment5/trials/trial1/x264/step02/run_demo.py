import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source"))
from predictor import fit_evaluate
r=fit_evaluate("/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/config/X264_AllMeasurements.csv")
print("predictors=" + ",".join(r["predictors"])); print("rows=%d train=%d test=%d groups=%d"%(r["rows"],r["train"],r["test"],r["groups"])); print("ridge=1e-6 seed=42 grouped_test_fraction=0.2"); print("held_out_mae=%.9f training_mean_baseline_mae=%.9f"%(r["mae"],r["baseline_mae"]))
