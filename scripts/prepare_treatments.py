"""Prepare isolated workspaces and metadata for the seven treatments."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/treatments.json"
with MANIFEST.open(encoding="utf-8") as stream:
    manifest = json.load(stream)

base = ROOT / "experiments/runs"
base.mkdir(parents=True, exist_ok=True)
for treatment in manifest["treatments"]:
    treatment_dir = base / f"treatment{treatment['id']}"
    treatment_dir.mkdir(exist_ok=True)
    metadata = dict(treatment)
    metadata["status"] = "prepared"
    metadata["domains"] = manifest["domains"]
    metadata["ladder_steps"] = manifest["ladder_steps"]
    (treatment_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (treatment_dir / "trials").mkdir(exist_ok=True)
print(f"Prepared {len(manifest['treatments'])} isolated treatment workspaces under {base}")
