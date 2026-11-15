"""What a domain adapter must tell the shared core about its dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    client: str
    path: Path
    objectives: List[str]  # suffix "-" means minimize, "+" means maximize
    label: str = ""
    contrast: Tuple[str, str, str] = ("", "", "")  # (feature, value A, value B) for certify
    requirements: Dict[int, str] = field(default_factory=dict)  # ladder step -> customer ask
    max_rows: int = 500  # rows sampled for ladder demos
