from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Optional, Tuple, Dict

from .math_utils import P, T, _I, _as_mat4, _as_vec3


def _normalize_joint_limit(
    joint_type: str,
    limit: Optional[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
    if limit is None:
        return None

    lower, upper = float(limit[0]), float(limit[1])
    if math.isnan(lower) or math.isnan(upper):
        raise ValueError("joint limits cannot contain NaN")
    if lower > upper:
        raise ValueError("joint lower limit cannot exceed upper limit")
    if math.isfinite(lower) and math.isfinite(upper):
        return lower, upper
    if joint_type == "revolute" and lower == -math.inf and upper == math.inf:
        return None
    if joint_type == "revolute":
        raise ValueError(
            "revolute joint limits must be finite, or use (-inf, inf) or None "
            "for continuous rotation"
        )
    raise ValueError(f"{joint_type} joint limits must be finite")


@dataclass
class Joint:
    """URDF-style joint connecting parent and child Assets."""

    name: str
    joint_type: str = "revolute"
    axis: P = field(default_factory=lambda: _as_vec3((0.0, 0.0, 1.0)))
    origin: T = field(default_factory=_I)
    limit: Optional[Tuple[float, float]] = None
    initial: float = 0.0
    effort: Optional[float] = None
    velocity: Optional[float] = None

    def __post_init__(self) -> None:
        self.limit = _normalize_joint_limit(self.joint_type, self.limit)

    def to_dict(self) -> Dict[str, Any]:
        limit = _normalize_joint_limit(self.joint_type, self.limit)
        return {
            "name": self.name,
            "joint_type": self.joint_type,
            "axis": tuple(float(x) for x in _as_vec3(self.axis)),
            "origin": _as_mat4(self.origin).tolist(),
            "limit": limit,
            "initial": float(self.initial),
            "effort": None if self.effort is None else float(self.effort),
            "velocity": None if self.velocity is None else float(self.velocity),
        }


__all__ = ["Joint"]
