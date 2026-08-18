from __future__ import annotations

from agents import RunContextWrapper, function_tool

from .context import AgentToolContext


@function_tool(failure_error_function=None)
def read_file(context: RunContextWrapper[AgentToolContext], path: str) -> str:
    """Read one UTF-8 text file inside the assigned workspace.

    Args:
        path: Workspace-relative file path.
    """
    target = context.context.resolve(path)
    content = target.read_text(encoding="utf-8")
    context.context.record("read_file", target)
    return content


@function_tool(failure_error_function=None)
def write_file(context: RunContextWrapper[AgentToolContext], path: str, content: str) -> str:
    """Write the complete initial program to the assigned empty source file.

    Args:
        path: Workspace-relative path of the assigned source file.
        content: Complete UTF-8 source text.
    """
    target = context.context.resolve(path)
    if target != context.context.source_path:
        raise ValueError("write_file may write only the assigned source file")
    if not target.is_file():
        raise FileNotFoundError(target)
    if not content.strip():
        raise ValueError("content must not be empty")
    existing = target.read_text(encoding="utf-8")
    if existing:
        if existing == content:
            return "no change: assigned source already has identical content"
        return "write rejected: assigned source is no longer empty; use apply_patch"
    target.write_text(content, encoding="utf-8")
    context.context.record("write_file", target)
    return f"wrote {target.relative_to(context.context.workspace).as_posix()}"


@function_tool(failure_error_function=None)
def apply_patch(
    context: RunContextWrapper[AgentToolContext],
    path: str,
    old_text: str,
    new_text: str,
) -> str:
    """Replace one exact, unique text block in an existing workspace file.

    Args:
        path: Workspace-relative file path.
        old_text: Exact current text. It must occur exactly once.
        new_text: Replacement text.
    """
    target = context.context.resolve(path)
    if target != context.context.source_path:
        raise ValueError("apply_patch may edit only the assigned source file")
    content = target.read_text(encoding="utf-8")
    if not old_text:
        raise ValueError("old_text must not be empty")
    occurrences = content.count(old_text)
    if occurrences != 1:
        raise ValueError(f"old_text must occur exactly once; found {occurrences}")
    if old_text == new_text:
        return "no change: new_text is identical to old_text"
    target.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
    context.context.record("apply_patch", target)
    return f"patched {target.relative_to(context.context.workspace).as_posix()}"


READ_TOOLS = [read_file]
WRITE_TOOLS = [write_file]
BUILD_TOOLS = [read_file, write_file, apply_patch]
PATCH_TOOLS = [read_file, apply_patch]

__all__ = [
    "BUILD_TOOLS",
    "PATCH_TOOLS",
    "READ_TOOLS",
    "WRITE_TOOLS",
    "apply_patch",
    "read_file",
    "write_file",
]
