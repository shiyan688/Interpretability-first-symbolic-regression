from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorkflowConfig:
    """Thin config wrapper.

    The workflow intentionally keeps config as dictionaries so paths and
    per-project source definitions can evolve without changing the schema.
    """

    path: Path
    data: dict[str, Any]

    @classmethod
    def from_json(cls, path: str | Path) -> "WorkflowConfig":
        cfg_path = Path(path).expanduser().resolve()
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        return cls(path=cfg_path, data=data)

    @property
    def input_csv(self) -> Path:
        return Path(self.data["input_csv"]).expanduser()

    @property
    def output_root(self) -> Path:
        return Path(self.data["output_root"]).expanduser()

    @property
    def targets(self) -> list[str]:
        return [str(x) for x in self.data.get("targets", [])]

    def pysr_options(self) -> dict[str, Any]:
        return dict(self.data.get("pysr", {}))

    def format_path(self, value: str | None, target: str) -> Path | None:
        if not value:
            return None
        return Path(str(value).format(target=target)).expanduser()

