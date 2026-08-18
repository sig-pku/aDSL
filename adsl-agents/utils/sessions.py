from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agents import SQLiteSession


@dataclass(frozen=True)
class SessionManager:
    workspace: Path
    task_id: str

    def __post_init__(self) -> None:
        workspace = self.workspace.expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        object.__setattr__(self, "workspace", workspace)
        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")

    @property
    def database_path(self) -> Path:
        return self.workspace / "sessions.sqlite3"

    def for_role(self, role: str) -> SQLiteSession:
        normalized = role.strip()
        if not normalized:
            raise ValueError("role must not be empty")
        return SQLiteSession(
            session_id=f"{self.task_id}:{normalized}",
            db_path=self.database_path,
        )


__all__ = ["SessionManager"]
