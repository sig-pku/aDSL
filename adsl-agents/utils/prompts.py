from __future__ import annotations

from importlib.resources import files
from pathlib import Path


ARTICULATION_CODER_GUIDANCE_ENABLED = (
    "Maintain hierarchy with `attach_part(name, child)` for rigid geometry, and access parts via "
    "`self.<part_name>`. For moving parts, use `self.revolute(...)`, `self.prismatic(...)`, or "
    "`self.fixed(...)` so the articulation is represented explicitly. The common pattern is "
    "`self.attach_part(\"door\", door)` then `self.revolute(\"door\", ...)`, or "
    "`self.attach_part(\"drawer\", drawer)` then `self.prismatic(\"drawer\", ...)`. Prefer that "
    "stable sequence instead of creating a differently named joint child and then referencing the "
    "old part name. Strict frame contract: before calling `revolute`, `prismatic`, or `fixed`, "
    "build and align the child in the parent link's zero-pose coordinates. The joint call detaches "
    "the child and rebases its geometry by `inverse(origin)`, so `origin` is the parent-frame "
    "hinge/pivot/slide frame and the moving child's relevant anchor should already coincide with "
    "`origin` at zero pose. Do not leave a moving child built around local `(0, 0, 0)` with a "
    "nonzero `origin`; first translate or align it into the parent zero-pose location. The `axis` "
    "argument is expressed in the joint/child frame at zero pose. Positive revolute motion follows "
    "the right-hand rule, and positive prismatic motion follows the axis; negating the axis reverses "
    "the meaning of positive joint values. Choose the axis sign, signed limits, and initial value "
    "together, and verify the intended motion at both zero and a representative nonzero value. For "
    "example, do not approve a laptop hinge until the configured initial or limit pose moves the lid "
    "upward rather than through the base. Note that low-level `attach_joint(...)` is different: its "
    "axis is expressed in the parent-link frame."
)
ARTICULATION_CODER_GUIDANCE_DISABLED = (
    "Maintain hierarchy with `attach_part(name, child)` for rigid geometry, and access parts via "
    "`self.<part_name>`."
)
ARTICULATION_PLANNER_GUIDANCE_ENABLED = (
    "Focus on big structure, spatial relations, repeated layouts, and functionality. If the "
    "object has moving parts, include the intended articulation type, axis, pivot/slide origin, "
    "zero-pose child placement, positive motion direction, signed limits, initial state, and a "
    "representative nonzero pose in the relations or checklist. Axis sign must be intentional under "
    "the right-hand rule, rather than inferred from the pivot alone."
)
ARTICULATION_PLANNER_GUIDANCE_DISABLED = (
    "Focus on big structure, spatial relations, repeated layouts, and functionality."
)


def _without_markdown_section(content: str, heading: str) -> str:
    marker = f"## {heading}"
    start = content.find(marker)
    if start < 0:
        return content
    end = content.find("\n## ", start + len(marker))
    if end < 0:
        return content[:start].rstrip() + "\n"
    return content[:start].rstrip() + "\n\n" + content[end + 1 :].lstrip()


def dsl_reference(*, articulation: bool = True) -> str:
    resource = files("adsl.core").joinpath("docs/dsl_doc.md")
    content = resource.read_text(encoding="utf-8")
    if not articulation:
        content = _without_markdown_section(content, "Articulation")
    return content


def dsl_example() -> str:
    resource = files("adsl.core").joinpath("docs/dsl_example.md")
    return resource.read_text(encoding="utf-8")


def render_prompt_resource(
    path: str | Path,
    *,
    articulation: bool = True,
    replacements: dict[str, object] | None = None,
) -> str:
    content = Path(path).read_text(encoding="utf-8")
    values: dict[str, object] = {
        "[DSL_DOC]": dsl_reference(articulation=articulation),
        "[DSL_EXAMPLE]": dsl_example(),
        "[ARTICULATION_CODER_GUIDANCE]": (
            ARTICULATION_CODER_GUIDANCE_ENABLED
            if articulation
            else ARTICULATION_CODER_GUIDANCE_DISABLED
        ),
        "[ARTICULATION_PLANNER_GUIDANCE]": (
            ARTICULATION_PLANNER_GUIDANCE_ENABLED
            if articulation
            else ARTICULATION_PLANNER_GUIDANCE_DISABLED
        ),
    }
    values.update(replacements or {})
    for marker, value in values.items():
        content = content.replace(marker, str(value))
    return content.strip()


__all__ = ["dsl_example", "dsl_reference", "render_prompt_resource"]
