from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Iterable

from agents import function_tool


def _read_text(path: Path, *, limit: int | None = None) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if limit is not None and len(text) > limit:
        return text[:limit] + f"\n... [{len(text) - limit} characters omitted]"
    return text


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _compact(value: Any, limit: int) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _symbol_names(source: str) -> list[str]:
    names = re.findall(
        r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        source,
        flags=re.MULTILINE,
    )
    seen: list[str] = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return seen


@dataclass(frozen=True)
class WorkspaceSnapshot:
    path: Path
    exists: bool = False
    status: str = "missing"
    task_id: str = ""
    requirement: str = ""
    articulation: bool | None = None
    max_rounds: int | None = None
    selected_round: int | None = None
    approved: bool | None = None
    finalization_reason: str = ""
    plan: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None
    source: str = ""
    symbols: tuple[str, ...] = ()
    image_count: int = 0
    latest_image_critique: dict[str, Any] = field(default_factory=dict)
    latest_code_critique: dict[str, Any] = field(default_factory=dict)

    def summary_text(self, *, include_source: bool = False) -> str:
        lines = [
            f"Workspace: {self.path}",
            f"Status: {self.status}",
            f"Task ID: {self.task_id or 'n/a'}",
            f"Requirement: {_compact(self.requirement, 1200) or 'n/a'}",
            f"Articulation: {self.articulation if self.articulation is not None else 'n/a'}",
            f"Selected round: {self.selected_round if self.selected_round is not None else 'n/a'}",
            f"Approved: {self.approved if self.approved is not None else 'n/a'}",
            f"Finalization: {self.finalization_reason or 'n/a'}",
            f"Reference images: {self.image_count}",
            f"Source file: {self.source_path or 'n/a'}",
            f"Defined classes: {', '.join(self.symbols) if self.symbols else 'n/a'}",
        ]
        if self.plan:
            lines.append(f"Plan: {_compact(self.plan, 3500)}")
        if self.latest_image_critique:
            lines.append(
                "Latest image critique: "
                + _compact(self.latest_image_critique, 1800)
            )
        if self.latest_code_critique:
            lines.append(
                "Latest code critique: "
                + _compact(self.latest_code_critique, 1800)
            )
        if include_source and self.source:
            lines.extend(
                [
                    "",
                    "Current source.py:",
                    "```python",
                    self.source,
                    "```",
                ]
            )
        return "\n".join(lines)


def scan_workspace(workspace: str | Path) -> WorkspaceSnapshot:
    root = Path(workspace).expanduser().resolve()
    if not root.is_dir():
        return WorkspaceSnapshot(path=root)

    run = _read_json(root / "run.json")
    runtime = _read_json(root / "runtime_config.json")
    user_input = _read_json(root / "user_input.json")
    plan = _read_json(root / "plan.json")
    request = runtime.get("request") if isinstance(runtime.get("request"), dict) else {}

    requirement = str(
        run.get("requirement")
        or user_input.get("requirement")
        or request.get("requirement")
        or _read_text(root / "user_input.txt", limit=4000)
        or ""
    )
    source_path = root / "source.py"
    if not source_path.is_file():
        source_path = None
    source = _read_text(source_path, limit=16000) if source_path is not None else ""

    image_root = root / "user_input_images"
    image_count = (
        len([path for path in image_root.iterdir() if path.is_file()])
        if image_root.is_dir()
        else 0
    )
    image_history = run.get("image_critic_history")
    code_history = run.get("code_critic_history")
    latest_image = (
        image_history[-1]
        if isinstance(image_history, list) and image_history and isinstance(image_history[-1], dict)
        else {}
    )
    latest_code = (
        code_history[-1]
        if isinstance(code_history, list) and code_history and isinstance(code_history[-1], dict)
        else {}
    )
    status = str(run.get("status") or ("ready" if source_path is not None else "empty"))
    articulation_value = user_input.get("articulation", request.get("articulation"))
    selected_round = run.get("selected_round")
    max_rounds = user_input.get("max_rounds", request.get("max_rounds"))

    return WorkspaceSnapshot(
        path=root,
        exists=True,
        status=status,
        task_id=str(run.get("task_id") or request.get("task_id") or ""),
        requirement=requirement,
        articulation=(None if articulation_value is None else bool(articulation_value)),
        max_rounds=(None if max_rounds is None else int(max_rounds)),
        selected_round=(None if selected_round is None else int(selected_round)),
        approved=(None if run.get("approved") is None else bool(run.get("approved"))),
        finalization_reason=str(run.get("finalization_reason") or ""),
        plan=plan,
        source_path=source_path,
        source=source,
        symbols=tuple(_symbol_names(source)),
        image_count=image_count,
        latest_image_critique=latest_image,
        latest_code_critique=latest_code,
    )


def format_workspace_table(paths: Iterable[str | Path]) -> str:
    rows: list[str] = []
    for index, value in enumerate(paths, 1):
        snapshot = scan_workspace(value)
        rows.append(
            f"{index}. {snapshot.path} | status={snapshot.status} | "
            f"round={snapshot.selected_round if snapshot.selected_round is not None else 'n/a'} | "
            f"request={_compact(snapshot.requirement, 100) or 'n/a'}"
        )
    return "\n".join(rows) if rows else "No known workspaces."


def build_workspace_context(
    active: WorkspaceSnapshot | None,
    known: Iterable[WorkspaceSnapshot],
) -> str:
    blocks: list[str] = []
    if active is None:
        blocks.append("There is no active asset workspace.")
    else:
        blocks.extend(
            [
                "Active asset workspace:",
                active.summary_text(include_source=True),
            ]
        )

    remembered = [item for item in known if active is None or item.path != active.path]
    if remembered:
        blocks.append("Remembered asset workspaces:")
        blocks.extend(item.summary_text(include_source=False) for item in remembered[-8:])
    return "\n\n".join(blocks)


def read_workspace_text(
    workspace: str | Path,
    relative_path: str | Path,
    *,
    limit: int = 24000,
) -> str:
    root = Path(workspace).expanduser().resolve()
    if not root.is_dir():
        return f"Workspace does not exist: {root}"
    relative = Path(relative_path)
    if relative.is_absolute():
        return "File path must be relative to the workspace."
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        return "File path escapes the workspace."
    if not target.is_file():
        return f"Workspace file does not exist: {relative_path}"
    return _read_text(target, limit=max(1, min(int(limit), 50000)))


@function_tool(failure_error_function=None)
def inspect_workspace(path: str) -> str:
    """Inspect an existing aDSL object workspace, including its current source."""
    snapshot = scan_workspace(path)
    if not snapshot.exists:
        return f"Workspace does not exist: {snapshot.path}"
    return snapshot.summary_text(include_source=True)


@function_tool(failure_error_function=None)
def read_workspace_file(workspace: str, relative_path: str) -> str:
    """Read one UTF-8 text file relative to an existing aDSL workspace."""
    return read_workspace_text(workspace, relative_path)


WORKSPACE_QUERY_TOOLS = [inspect_workspace, read_workspace_file]


__all__ = [
    "WORKSPACE_QUERY_TOOLS",
    "WorkspaceSnapshot",
    "build_workspace_context",
    "format_workspace_table",
    "read_workspace_text",
    "scan_workspace",
]
