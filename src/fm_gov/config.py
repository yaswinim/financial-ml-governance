from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class Paths:
    artifacts: Path = ROOT / "artifacts"
    registry: Path = ROOT / "artifacts" / "registry"
    monitoring: Path = ROOT / "artifacts" / "monitoring"
    reports: Path = ROOT / "artifacts" / "reports"

PATHS = Paths()
