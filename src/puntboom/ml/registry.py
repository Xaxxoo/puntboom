"""Model version registry for tracking trained models."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from puntboom.config import settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Track model versions and metadata."""

    def __init__(self):
        self.registry_path = settings.model_dir / "registry.json"
        self._registry: dict = self._load()

    def _load(self) -> dict:
        if self.registry_path.exists():
            return json.loads(self.registry_path.read_text())
        return {"models": {}}

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(self._registry, indent=2))

    def register(
        self,
        sport: str,
        version: str,
        model_type: str,
        metrics: dict,
        sample_count: int,
    ) -> None:
        """Register a new model version."""
        if sport not in self._registry["models"]:
            self._registry["models"][sport] = []

        entry = {
            "version": version,
            "model_type": model_type,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "sample_count": sample_count,
        }
        self._registry["models"][sport].append(entry)
        self._save()
        logger.info(f"Registered model {sport}/{version}")

    def get_latest(self, sport: str) -> dict | None:
        """Get the latest model entry for a sport."""
        entries = self._registry["models"].get(sport, [])
        return entries[-1] if entries else None

    def get_all(self, sport: str) -> list[dict]:
        """Get all model entries for a sport."""
        return self._registry["models"].get(sport, [])
