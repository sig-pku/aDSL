from __future__ import annotations

import math
import re
from typing import Iterable, Sequence

import numpy as np

from .asset import Asset, concat_shapes, ensure_shape
from .axis import axis_index, parse_axis, unsigned_axis_name
from .bounds import shape_aabb, shape_center, shape_size
from .math_utils import P, _as_vec3
from .transforms import rotate_shape, translate_shape


_ANCHOR_TOKEN_TO_AXIS_FRACTION = {
    "left": ("x", 0.0),
    "right": ("x", 1.0),
    "front": ("y", 0.0),
    "back": ("y", 1.0),
    "bottom": ("z", 0.0),
    "top": ("z", 1.0),
}


def _anchor_fractions(anchor: str) -> tuple[float, float, float]:
    if not isinstance(anchor, str):
        raise TypeError("anchor must be a string such as 'top' or 'left_front_top'.")
    key = anchor.strip().lower()
    if not key:
        raise ValueError("anchor must be a non-empty string.")
    if key in {"center", "centre", "middle"}:
        return (0.5, 0.5, 0.5)

    fractions = {"x": 0.5, "y": 0.5, "z": 0.5}
    specified_tokens: dict[str, str] = {}
    for token in re.split(r"[\s_]+", key):
        if not token or token in {"center", "centre", "middle"}:
            continue
        try:
            axis, fraction = _ANCHOR_TOKEN_TO_AXIS_FRACTION[token]
        except KeyError as exc:
            raise ValueError(
                f"Unknown anchor token '{token}' in '{anchor}'. Use left/right, "
                "front/back, bottom/top, or center."
            ) from exc
        previous = specified_tokens.get(axis)
        if previous is not None and fractions[axis] != fraction:
            raise ValueError(
                f"Conflicting anchor tokens '{previous}' and '{token}' for axis '{axis}'."
            )
        fractions[axis] = fraction
        specified_tokens[axis] = token
    return (fractions["x"], fractions["y"], fractions["z"])


def _plane_axis_ids(plane: str) -> tuple[int, int]:
    if not isinstance(plane, str):
        raise TypeError("plane must be a string such as 'xy' or 'yz'.")
    axes = [token for token in plane.strip().lower() if token in {"x", "y", "z"}]
    if len(axes) != 2 or axes[0] == axes[1]:
        raise ValueError("plane must specify two distinct axes such as 'xy', 'xz', or 'yz'.")
    return axis_index(axes[0]), axis_index(axes[1])


def _coerce_layout_spacing(spacing: float | Sequence[float]) -> tuple[float, float]:
    if isinstance(spacing, (int, float)):
        value = float(spacing)
        return value, value
    values = list(spacing)
    if len(values) != 2:
        raise ValueError("spacing must be a scalar or a 2-component sequence.")
    return float(values[0]), float(values[1])


def _axis_sign(axis: str) -> float:
    parse_axis(axis)
    return -1.0 if axis.strip().startswith("-") else 1.0


def shape_anchor(shape: Asset, anchor: str = "center") -> P:
    """Return a world-space point on the shape's axis-aligned bounding box.

    Anchor names combine one token per axis. Unspecified axes use their center;
    for example ``top``, ``front_left``, and ``right_back_bottom`` are valid.
    """
    vmin, vmax = shape_aabb(ensure_shape(shape))
    fractions = _as_vec3(_anchor_fractions(anchor))
    return _as_vec3(vmin + (vmax - vmin) * fractions)


def align_anchors(
    shape: Asset,
    target: Asset | Sequence[float],
    anchor: str = "center",
    target_anchor: str | None = None,
    offset: Sequence[float] = (0.0, 0.0, 0.0),
) -> Asset:
    """Return a translated copy whose selected anchor lands on a target anchor.

    ``target`` may be an Asset or a world-space point. For an Asset target,
    ``target_anchor`` defaults to the source ``anchor``. A point target does not
    accept ``target_anchor`` because the point is already exact.
    """
    source_point = shape_anchor(ensure_shape(shape), anchor)
    if isinstance(target, Asset):
        target_point = shape_anchor(target, target_anchor or anchor)
    else:
        if target_anchor is not None:
            raise TypeError("target_anchor is only valid when target is an Asset.")
        target_point = _as_vec3(target)
    return translate_shape(shape, target_point + _as_vec3(offset) - source_point)


def place_on_axis(
    shape: Asset,
    target: Asset | float,
    axis: str = "+z",
    gap: float = 0.0,
) -> Asset:
    """Place ``shape`` immediately after ``target`` along a signed axis.

    Along a positive axis the source minimum is placed after the target maximum;
    along a negative axis the source maximum is placed before the target minimum.
    A numeric target is interpreted as the target boundary coordinate.
    """
    gap_value = float(gap)
    if gap_value < 0.0:
        raise ValueError("gap must be non-negative; use a signed axis to choose direction.")
    axis_id = axis_index(unsigned_axis_name(axis))
    sign = _axis_sign(axis)
    source_min, source_max = shape_aabb(ensure_shape(shape))
    if isinstance(target, (int, float)):
        target_boundary = float(target)
    else:
        target_min, target_max = shape_aabb(ensure_shape(target))
        target_boundary = float(target_max[axis_id] if sign > 0 else target_min[axis_id])
    source_boundary = float(source_min[axis_id] if sign > 0 else source_max[axis_id])
    delta = np.zeros(3, dtype=float)
    delta[axis_id] = target_boundary + sign * gap_value - source_boundary
    return translate_shape(shape, delta)


def _distribute_along_axis(
    shapes: Iterable[Asset],
    *,
    mode: str,
    axis: str,
    spacing: float,
) -> Asset:
    normalized = [ensure_shape(shape) for shape in shapes]
    if not normalized:
        return concat_shapes([], label="Axis Layout")
    if mode not in {"centers", "bounds"}:
        raise ValueError("mode must be 'centers' or 'bounds'.")

    distance = float(spacing)
    if distance < 0.0:
        raise ValueError("spacing must be non-negative; use a signed axis to choose direction.")
    axis_id = axis_index(unsigned_axis_name(axis))
    sign = _axis_sign(axis)
    centers = [shape_center(shape) for shape in normalized]
    sizes = [shape_size(shape) for shape in normalized]
    targets = [float(centers[0][axis_id])]
    for index in range(1, len(normalized)):
        step = distance
        if mode == "bounds":
            step += float(sizes[index - 1][axis_id] + sizes[index][axis_id]) * 0.5
        targets.append(targets[-1] + sign * step)

    placed: list[Asset] = []
    for shape, center, target in zip(normalized, centers, targets):
        delta = np.zeros(3, dtype=float)
        delta[axis_id] = target - float(center[axis_id])
        placed.append(translate_shape(shape, delta))
    return concat_shapes(placed, label="Axis Layout")


def distribute_along_axis(
    shapes: Sequence[Asset],
    axis: str = "+x",
    spacing: float = 1.0,
) -> Asset:
    """Place shape centers at fixed intervals along a signed axis.

    The first input shape is the fixed base: its center remains unchanged. Each
    later center is ``spacing`` farther along ``axis`` than the previous center.
    """
    return _distribute_along_axis(shapes, mode="centers", axis=axis, spacing=spacing)


def stack_shapes(
    shapes: Iterable[Asset],
    axis: str = "+z",
    gap: float = 0.0,
) -> Asset:
    """Stack shape bounds along a signed axis without moving the first shape.

    The first input shape is the fixed base. Every later shape is placed after
    the preceding shape with ``gap`` between their axis-aligned boundaries.
    """
    return _distribute_along_axis(shapes, mode="bounds", axis=axis, spacing=gap)


def grid_shapes(
    shapes: Sequence[Asset],
    rows: int | None = None,
    cols: int | None = None,
    spacing: float | Sequence[float] = (1.0, 1.0),
    plane: str = "xy",
    center: Sequence[float] = (0.0, 0.0, 0.0),
    order: str = "row-major",
) -> Asset:
    """Lay out shape centers on a regular grid centered at ``center``."""
    normalized = [ensure_shape(shape) for shape in shapes]
    if not normalized:
        return concat_shapes([], label="Grid Shapes")

    count = len(normalized)
    if rows is None and cols is None:
        rows, cols = 1, count
    elif rows is None:
        cols = int(cols)
        if cols <= 0:
            raise ValueError("cols must be positive.")
        rows = int(math.ceil(count / cols))
    elif cols is None:
        rows = int(rows)
        if rows <= 0:
            raise ValueError("rows must be positive.")
        cols = int(math.ceil(count / rows))
    else:
        rows, cols = int(rows), int(cols)
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive.")
    if rows * cols < count:
        raise ValueError("rows * cols must be large enough to place every shape.")

    axis_u, axis_v = _plane_axis_ids(plane)
    spacing_u, spacing_v = _coerce_layout_spacing(spacing)
    center_point = _as_vec3(center)
    order_key = order.strip().lower()
    if order_key not in {"row-major", "column-major"}:
        raise ValueError("order must be 'row-major' or 'column-major'.")

    placed: list[Asset] = []
    for index, shape in enumerate(normalized):
        if order_key == "row-major":
            row, col = divmod(index, cols)
        else:
            col, row = divmod(index, rows)
        target = center_point.copy()
        target[axis_u] += (col - (cols - 1) / 2.0) * spacing_u
        target[axis_v] += ((rows - 1) / 2.0 - row) * spacing_v
        placed.append(translate_shape(shape, target - shape_center(shape)))
    return concat_shapes(placed, label="Grid Shapes")


def _radial_basis(axis: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = np.asarray(parse_axis(axis), dtype=float)
    axis_name = unsigned_axis_name(axis)
    if axis_name == "x":
        first = np.array((0.0, 1.0, 0.0))
    elif axis_name == "y":
        first = np.array((0.0, 0.0, 1.0))
    else:
        first = np.array((1.0, 0.0, 0.0))
    second = np.cross(normal, first)
    return first, second, normal


def radial_shapes(
    shapes: Sequence[Asset],
    radius: float,
    axis: str = "+z",
    center: Sequence[float] = (0.0, 0.0, 0.0),
    start_angle: float = 0.0,
    sweep: float = 360.0,
    *,
    rotate_with_layout: bool = False,
    rotation_offset: float = 0.0,
) -> Asset:
    """Lay out shapes on a circle or arc around a signed cardinal axis.

    A full circle uses evenly spaced slots without duplicating the first slot;
    a partial arc includes both endpoints. When ``rotate_with_layout`` is true,
    each shape is rotated around its own center by its slot angle plus
    ``rotation_offset`` before it is translated into place. This preserves the
    input shape's orientation as the zero-angle reference and directly supports
    spoke, fan-blade, and radial-pattern construction.
    """
    normalized = [ensure_shape(shape) for shape in shapes]
    if not normalized:
        return concat_shapes([], label="Radial Shapes")
    radius_value = float(radius)
    if radius_value < 0.0:
        raise ValueError("radius must be non-negative.")

    first, second, normal = _radial_basis(axis)
    center_point = _as_vec3(center)
    count = len(normalized)
    sweep_value = float(sweep)
    if count == 1:
        angles = [float(start_angle)]
    else:
        divisor = count if math.isclose(abs(sweep_value), 360.0, abs_tol=1e-9) else count - 1
        step = sweep_value / divisor
        angles = [float(start_angle) + step * index for index in range(count)]

    placed: list[Asset] = []
    for shape, angle in zip(normalized, angles):
        oriented = shape
        if rotate_with_layout:
            oriented = rotate_shape(
                oriented,
                normal,
                angle + float(rotation_offset),
                center=shape_center(oriented),
            )
        theta = math.radians(angle)
        target = center_point + radius_value * (math.cos(theta) * first + math.sin(theta) * second)
        placed.append(translate_shape(oriented, target - shape_center(oriented)))
    return concat_shapes(placed, label="Radial Shapes")


def align_centers(
    shape: Asset,
    target: Asset,
    axes: Iterable[str] = ("x", "y", "z"),
) -> Asset:
    """Align selected center coordinates of ``shape`` with ``target``."""
    source_center = shape_center(ensure_shape(shape))
    target_center = shape_center(ensure_shape(target))
    offset = np.zeros(3, dtype=float)
    for axis in axes:
        index = axis_index(unsigned_axis_name(axis))
        offset[index] = target_center[index] - source_center[index]
    return translate_shape(shape, offset)


def offset_from(
    shape: Asset,
    reference: Asset | Sequence[float | None] | None,
    offset: Sequence[float | None],
) -> Asset:
    """Place selected center coordinates relative to a shape, point, or origin.

    ``None`` in either coordinate sequence leaves that coordinate of ``shape``
    unchanged. A ``None`` reference uses the world origin.
    """
    offsets = list(offset)
    if len(offsets) != 3:
        raise ValueError("offset must be a 3-component sequence.")
    current = shape_center(ensure_shape(shape))
    if reference is None:
        reference_center: Sequence[float | None] = (0.0, 0.0, 0.0)
    elif isinstance(reference, Asset):
        reference_center = shape_center(reference)
    else:
        reference_center = list(reference)
        if len(reference_center) != 3:
            raise ValueError("reference must be a 3-component sequence.")

    delta = np.zeros(3, dtype=float)
    for index, value in enumerate(offsets):
        reference_value = reference_center[index]
        if value is not None and reference_value is not None:
            delta[index] = float(reference_value) + float(value) - current[index]
    return translate_shape(shape, delta)


__all__ = [
    "align_anchors",
    "align_centers",
    "distribute_along_axis",
    "grid_shapes",
    "offset_from",
    "place_on_axis",
    "radial_shapes",
    "shape_anchor",
    "stack_shapes",
]
