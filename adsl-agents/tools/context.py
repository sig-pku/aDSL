from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ToolEvent:
    tool: str
    path: str


@dataclass
class AgentToolContext:
    workspace: Path
    source_path: Path
    executor_timeout: float = 300.0
    events: list[ToolEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.workspace = self.workspace.expanduser().resolve()
        self.source_path = self.source_path.expanduser().resolve()
        if self.source_path != self.workspace and self.workspace not in self.source_path.parents:
            raise ValueError("source_path must be inside workspace")

    def resolve(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        resolved = candidate.resolve()
        if resolved != self.workspace and self.workspace not in resolved.parents:
            raise ValueError(f"path escapes workspace: {path}")
        return resolved

    def record(self, tool: str, path: Path) -> None:
        self.events.append(ToolEvent(tool=tool, path=path.relative_to(self.workspace).as_posix()))


__all__ = ["AgentToolContext", "ToolEvent"]
