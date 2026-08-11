"""Project paths and execution context for the notebook-derived pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ProjectConfig:
    """Filesystem locations used by the generated pipeline."""

    # Repo root is two levels above this package (src/credit_copula/).
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"

    @property
    def figures_dir(self) -> Path:
        return self.outputs_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.outputs_dir / "tables"

    @property
    def steps_dir(self) -> Path:
        return Path(__file__).resolve().parent / "steps"


def build_execution_context(
    config: ProjectConfig,
    context_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the shared global namespace used when executing step scripts.

    Step scripts are plain notebook cells: they read/write names in this dict.
    Keys such as ``FIGURES_DIR`` and ``OVERRIDES`` are the contract between
    ``run_pipeline`` and the generated step files.
    """
    # Ensure output folders exist before any step writes figures or tables.
    config.processed_data_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    config.tables_dir.mkdir(parents=True, exist_ok=True)

    overrides = dict(context_overrides or {})

    context: dict[str, Any] = {
        "__name__": "__main__",
        "PROJECT_ROOT": config.project_root,
        "DATA_DIR": config.data_dir,
        "PROCESSED_DATA_DIR": config.processed_data_dir,
        "OUTPUTS_DIR": config.outputs_dir,
        "FIGURES_DIR": config.figures_dir,
        "TABLES_DIR": config.tables_dir,
        "OVERRIDES": overrides,
        "SMOKE_TEST_MODE": bool(overrides.get("SMOKE_TEST_MODE", False)),
    }
    # CLI flags (e.g. ``n_simulations``) land in the same namespace as step code.
    context.update(overrides)
    return context
