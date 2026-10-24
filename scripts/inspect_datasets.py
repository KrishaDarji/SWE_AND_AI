"""Print schemas for all selected MOOT datasets."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.datasets import DATASETS, schema


for name, spec in DATASETS.items():
    print(json.dumps(schema(spec), indent=2))
