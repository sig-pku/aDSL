from __future__ import annotations

from typing import Any
import numpy as np


def as_vec3(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (3,):
        raise ValueError("expected a 3-vector")
    return array


def parse_axis(axis: Any) -> np.ndarray:
    if isinstance(axis, str):
        key = axis.strip().lower()
        table = {
            "x": (1, 0, 0), "+x": (1, 0, 0), "-x": (-1, 0, 0),
            "y": (0, 1, 0), "+y": (0, 1, 0), "-y": (0, -1, 0),
            "z": (0, 0, 1), "+z": (0, 0, 1), "-z": (0, 0, -1),
        }
        if key not in table:
            raise ValueError(f"unknown axis: {axis!r}")
        return np.asarray(table[key], dtype=float)
    return as_vec3(axis)


def unsigned_axis_name(axis: str) -> str:
    value = str(axis).strip().lower().lstrip("+").lstrip("-")
    if value not in {"x", "y", "z"}:
        raise ValueError(f"unknown axis: {axis!r}")
    return value


def axis_index(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[unsigned_axis_name(axis)]


__all__ = ["as_vec3", "parse_axis", "unsigned_axis_name", "axis_index"]
