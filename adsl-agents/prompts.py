from __future__ import annotations

from pathlib import Path

from .utils.prompts import render_prompt_resource


PROMPT_ROOT = Path(__file__).resolve().parent / "prompt"
ROLE_PROMPTS = {
    "planner": "planner.md",
    "edit_planner": "edit_planner.md",
    "coder": "coder.md",
    "debugger": "debugger.md",
    "image_critic": "critic_image.md",
    "code_critic": "critic_code_image.md",
}


def object_prompt(role: str, *, articulation: bool) -> str:
    try:
        filename = ROLE_PROMPTS[role]
    except KeyError as exc:
        raise ValueError(f"Unknown object prompt role: {role}") from exc
    return render_prompt_resource(
        PROMPT_ROOT / filename,
        articulation=articulation,
    )


__all__ = ["object_prompt"]
