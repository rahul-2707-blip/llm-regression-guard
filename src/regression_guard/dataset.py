"""Load the golden test dataset."""
import json
from pathlib import Path
from .contracts import TestCase


DATASET_DIR = Path(__file__).resolve().parents[2] / "golden_dataset"


def load_dataset(name: str = "cases_v1") -> list[TestCase]:
    path = DATASET_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return [TestCase(**c) for c in data["cases"]]
