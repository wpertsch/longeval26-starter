"""Loads config.yaml into a plain dataclass-like object.

Kept deliberately small so students can inspect it in one sitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass
class Config:
    """In-memory view of ``config.yaml``."""

    raw: dict[str, Any] = field(default_factory=dict)

    # ----- convenience accessors ------------------------------------------------
    @property
    def snapshots(self) -> list[str]:
        return list(self.raw["dataset"]["snapshots"])

    @property
    def dataset_prefix(self) -> str:
        return str(self.raw["dataset"]["prefix"])

    @property
    def qrels_variant(self) -> str:
        return str(self.raw["dataset"].get("qrels_variant", "dctr"))

    @property
    def indexes_dir(self) -> Path:
        return Path(self.raw["paths"]["indexes"])

    @property
    def runs_dir(self) -> Path:
        return Path(self.raw["paths"]["runs"])

    @property
    def run_name(self) -> str:
        return str(self.raw["run"]["name"])

    # ----- dataset id helpers ---------------------------------------------------
    def dataset_id(self, snapshot: str, *, with_qrels: bool = False) -> str:
        """Return the ir-datasets-longeval identifier for a snapshot.

        >>> cfg.dataset_id("snapshot-1")
        'longeval-sci-2026/snapshot-1'
        >>> cfg.dataset_id("snapshot-1", with_qrels=True)
        'longeval-sci-2026/snapshot-1/dctr'
        """
        base = f"{self.dataset_prefix}/{snapshot}"
        return f"{base}/{self.qrels_variant}" if with_qrels else base

    def index_path(self, snapshot: str) -> Path:
        return self.indexes_dir / snapshot

    def run_path(self, snapshot: str) -> Path:
        return self.runs_dir / snapshot / "run.txt.gz"


def load_config(path: Path | str | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(raw=data)
