from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .utils.usage import UsageTotals


class ObjectComponent(BaseModel):
    name: str
    description: str


class ObjectPlan(BaseModel):
    object_name: str
    components: list[ObjectComponent]
    relations: list[str]
    critic_checklist: list[str]


class EditPlan(BaseModel):
    summary: str
    preserved_features: list[str]
    changes: list[str]
    patch_scope: list[str]


class DebuggerDecision(BaseModel):
    bug_description: str
    suggested_fix: str


class ImageCriticDecision(BaseModel):
    approved: bool
    observations: list[str]
    required_changes: list[str] = Field(default_factory=list)


class CodeCriticDecision(BaseModel):
    approved: bool
    observations: list[str]
    required_changes: list[str] = Field(default_factory=list)
    image_critic_corrections: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class ObjectRequest:
    requirement: str
    workspace: Path
    task_id: str
    image_paths: tuple[Path, ...] = ()
    articulation: bool = False
    max_rounds: int = 2


@dataclass(frozen=True)
class ObjectRunResult:
    workspace: Path
    source_path: Path
    glb_path: Path
    urdf_path: Path | None
    render_paths: tuple[Path, ...]
    selected_round: int
    approved: bool
    usage: UsageTotals


EditKind = Literal["continue", "extend", "variant"]


__all__ = [
    "CodeCriticDecision",
    "DebuggerDecision",
    "EditKind",
    "EditPlan",
    "ImageCriticDecision",
    "ObjectPlan",
    "ObjectComponent",
    "ObjectRequest",
    "ObjectRunResult",
]
