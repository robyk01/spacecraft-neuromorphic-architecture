from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODELS_DIR = PROJECT_ROOT / "models"


@dataclass
class ModelPaths:
    run_name: str

    @property
    def folder(self) -> Path:
        return MODELS_DIR / self.run_name

    @property
    def keras(self) -> Path:
        return self.folder / "model.keras"

    @property
    def fbz(self) -> Path:
        return self.folder / "model.fbz"

    @property
    def config(self) -> Path:
        return self.folder / "config.json"

    @property
    def metrics(self) -> Path:
        return self.folder / "metrics.json"

    def create(self):
        self.folder.mkdir(parents=True, exist_ok=True)