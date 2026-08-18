from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass
class ChatSessionState:
    version: int = 1
    conversation_workspace: str = ""
    task_id: str = ""
    active_workspace: str | None = None
    pending_workspace: str | None = None
    known_workspaces: list[str] = field(default_factory=list)
    model_config: str | None = None

    @classmethod
    def create(
        cls,
        *,
        conversation_workspace: str | Path,
        task_id: str | None = None,
        model_config: str | Path | None = None,
    ) -> "ChatSessionState":
        return cls(
            conversation_workspace=str(Path(conversation_workspace).expanduser().resolve()),
            task_id=(task_id or uuid4().hex).strip(),
            model_config=(
                None
                if model_config is None
                else str(Path(model_config).expanduser().resolve())
            ),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        conversation_workspace: str | Path | None = None,
        task_id: str | None = None,
        model_config: str | Path | None = None,
    ) -> "ChatSessionState":
        state_path = Path(path).expanduser().resolve()
        if not state_path.is_file():
            if conversation_workspace is None:
                conversation_workspace = state_path.parent
            return cls.create(
                conversation_workspace=conversation_workspace,
                task_id=task_id,
                model_config=model_config,
            )

        payload = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Chat session state must contain an object: {state_path}")

        workspace_value = (
            payload.get("conversation_workspace")
            or conversation_workspace
            or state_path.parent
        )
        task_value = str(payload.get("task_id") or task_id or uuid4().hex).strip()
        config_value = payload.get("model_config") or payload.get("active_llm_config")
        if config_value is None:
            config_value = model_config

        known: list[str] = []
        for value in payload.get("known_workspaces", []):
            normalized = str(Path(str(value)).expanduser().resolve())
            if normalized not in known:
                known.append(normalized)
        active = payload.get("active_workspace")
        if active:
            active = str(Path(str(active)).expanduser().resolve())
            if active not in known:
                known.append(active)

        pending = payload.get("pending_workspace")
        return cls(
            version=max(1, int(payload.get("version", 1))),
            conversation_workspace=str(Path(workspace_value).expanduser().resolve()),
            task_id=task_value,
            active_workspace=active,
            pending_workspace=(
                None
                if not pending
                else str(Path(str(pending)).expanduser().resolve())
            ),
            known_workspaces=known,
            model_config=(
                None
                if config_value is None
                else str(Path(str(config_value)).expanduser().resolve())
            ),
        )

    def save(self, path: str | Path) -> Path:
        state_path = Path(path).expanduser().resolve()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "version": self.version,
            "conversation_workspace": self.conversation_workspace,
            "task_id": self.task_id,
            "active_workspace": self.active_workspace,
            "pending_workspace": self.pending_workspace,
            "known_workspaces": self.known_workspaces,
            "model_config": self.model_config,
        }
        temporary = state_path.with_suffix(state_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(state_path)
        return state_path

    def remember_workspace(self, workspace: str | Path) -> None:
        resolved = str(Path(workspace).expanduser().resolve())
        if resolved not in self.known_workspaces:
            self.known_workspaces.append(resolved)
        self.active_workspace = resolved

    def set_pending_workspace(self, workspace: str | Path | None) -> None:
        self.pending_workspace = (
            None
            if workspace is None
            else str(Path(workspace).expanduser().resolve())
        )


__all__ = ["ChatSessionState"]
