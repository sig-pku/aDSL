from __future__ import annotations

from typing import Any, Iterable, Tuple

import numpy as np


P = np.ndarray
T = np.ndarray


def _as_vec3(v: Iterable[float]) -> P:
    a = np.asarray(list(v), dtype=float).reshape(-1)
    if a.shape != (3,):
        raise ValueError("Expected a 3D vector")
    return a


_AXIS_STR = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
    "+x": (1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-x": (-1.0, 0.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "-z": (0.0, 0.0, -1.0),
}


def parse_axis(axis: Any) -> P:
    """Parse axis strings or 3-vectors into a numeric 3-vector."""
    if isinstance(axis, str):
        key = axis.strip().lower()
        if key in _AXIS_STR:
            return _as_vec3(_AXIS_STR[key])
        raise ValueError(
            f"Unknown axis string '{axis}'. Use x/y/z/-x/-y/-z or pass a 3-vector."
        )
    return _as_vec3(axis)


_parse_axis = parse_axis


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _as_color(c: Iterable[float]) -> Tuple[float, float, float]:
    cc = list(c)
    r = _clamp01(cc[0] if len(cc) > 0 else 1.0)
    g = _clamp01(cc[1] if len(cc) > 1 else 1.0)
    b = _clamp01(cc[2] if len(cc) > 2 else 1.0)
    return (r, g, b)


def _I() -> T:
    return np.eye(4, dtype=float)


def _as_mat4(m: Iterable[Iterable[float]] | None) -> T:
    if m is None:
        return _I()
    a = np.asarray(m, dtype=float).reshape(-1)
    if a.shape != (16,):
        raise ValueError("Expected a 4x4 matrix")
    if np.any(np.isnan(a)) or np.any(np.isinf(a)):
        raise ValueError("Matrix contains NaN or Inf")
    return a.reshape(4, 4)


def origin_to_mat4(origin: Any) -> T:
    """Coerce a joint origin into a 4x4 homogeneous transform."""
    if origin is None:
        return _I()
    arr = np.asarray(origin, dtype=float).reshape(-1)
    if arr.shape == (3,):
        matrix = np.eye(4, dtype=float)
        matrix[:3, 3] = arr
        return matrix
    if arr.shape == (16,):
        return arr.reshape(4, 4)
    raise ValueError("origin must be None, a 3-vector, or a 4x4 matrix")


__all__ = [
    "P",
    "T",
    "_as_vec3",
    "parse_axis",
    "_parse_axis",
    "_as_color",
    "_I",
    "_as_mat4",
    "origin_to_mat4",
]
