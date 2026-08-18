from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

from adsl.agents.models import ObjectRunResult
from adsl.agents.utils.usage import UsageTotals


ChatAction = Literal["chat", "create", "continue", "extend", "variant"]


class ChatDecision(BaseModel):
    action: ChatAction
    assistant_message: str
    requirement: str | None = None
    articulation: bool = False

    @model_validator(mode="after")
    def validate_requirement(self) -> "ChatDecision":
        if self.action == "chat" and self.requirement is not None:
            raise ValueError("chat decisions must not contain a generation requirement")
        if self.action != "chat" and not (self.requirement or "").strip():
            raise ValueError("asset decisions require a generation requirement")
        return self


@dataclass(frozen=True)
class ChatRequest:
    message: str
    conversation_workspace: Path
    task_id: str
    output_workspace: Path | None = None
    source_path: Path | None = None
    asset_context: str | None = None
    image_paths: tuple[Path, ...] = ()
    max_rounds: int = 2


@dataclass(frozen=True)
class ChatResult:
    decision: ChatDecision
    asset: ObjectRunResult | None
    conversation_usage: UsageTotals


__all__ = ["ChatAction", "ChatDecision", "ChatRequest", "ChatResult"]
