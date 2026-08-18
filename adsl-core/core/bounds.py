from __future__ import annotations
from typing import Iterable, Tuple
import numpy as np
import math 
from .asset import Asset, ensure_shape
from .math_utils import P, _as_mat4, _as_vec3, _parse_axis


# NOTE: Current implementation can over-estimate AABB in some transformed cases.

def _aabb_from_minmax_and_matrix(
    vmin, vmax, M
):
    M = _as_mat4(M)
    corners = np.array([
        [vmin[0], vmin[1], vmin[2], 1.0],
        [vmin[0], vmin[1], vmax[2], 1.0],
        [vmin[0], vmax[1], vmin[2], 1.0],
        [vmin[0], vmax[1], vmax[2], 1.0],
        [vmax[0], vmin[1], vmin[2], 1.0],
        [vmax[0], vmin[1], vmax[2], 1.0],
        [vmax[0], vmax[1], vmin[2], 1.0],
        [vmax[0], vmax[1], vmax[2], 1.0],
    ])
    corners = (M @ corners.T).T
    new_vmin = corners.min(axis=0)[:3]
    new_vmax = corners.max(axis=0)[:3]
    return _as_vec3(new_vmin), _as_vec3(new_vmax)

def _aabb_of_primitive(prim: dict) -> Tuple[P, P]:
    t = prim["type"]
    p = prim["params"]
    
    # Local AABBs
    if t == "sphere":
        cx, cy, cz = _as_vec3(p.get("center", (0, 0, 0)))
        r = float(p["radius"])
        
        vmin = (cx - r, cy - r, cz - r)
        vmax = (cx + r, cy + r, cz + r)
    elif t == "cube":
        cx, cy, cz = _as_vec3(p.get("center", (0, 0, 0)))
        sx, sy, sz = _as_vec3(p.get("scale", (1, 1, 1)))
        vmin = (cx - sx / 2, cy - sy / 2, cz - sz / 2)
        vmax = (cx + sx / 2, cy + sy / 2, cz + sz / 2)
    elif t == "cylinder":
        x0, y0, z0 = _as_vec3(p["p0"])
        x1, y1, z1 = _as_vec3(p["p1"])
        r = float(p["radius"])
        
        dx, dy, dz = float(x1 - x0), float(y1 - y0), float(z1 - z0)
        L = math.sqrt(dx * dx + dy * dy + dz * dz)
        
        if L < 1e-9:
            vmin = (min(float(x0), float(x1)) - r,
                    min(float(y0), float(y1)) - r,
                    min(float(z0), float(z1)) - r)
            vmax = (max(float(x0), float(x1)) + r,
                    max(float(y0), float(y1)) + r,
                    max(float(z0), float(z1)) + r)
        else:
            ux, uy, uz = dx / L, dy / L, dz / L
            ox = r * math.sqrt(max(0.0, 1.0 - ux * ux))
            oy = r * math.sqrt(max(0.0, 1.0 - uy * uy))
            oz = r * math.sqrt(max(0.0, 1.0 - uz * uz))
            
            vmin = (min(float(x0), float(x1)) - ox,
                    min(float(y0), float(y1)) - oy,
                    min(float(z0), float(z1)) - oz)
            vmax = (max(float(x0), float(x1)) + ox,
                    max(float(y0), float(y1)) + oy,
                    max(float(z0), float(z1)) + oz)
    else:
        return (float("inf"),) * 3, (float("-inf"),) * 3
    
    M = prim.get("xform", None)
    if M is None:
        return _as_vec3(vmin), _as_vec3(vmax)
    return _aabb_from_minmax_and_matrix(vmin, vmax, M)


def _normalize_direction(direction: Iterable[float] | str) -> P:
    vec = _parse_axis(direction) if isinstance(direction, str) else _as_vec3(direction)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        raise ValueError("direction must have non-zero length.")
    return _as_vec3(vec / norm)


def _aabb_support_point(vmin: P, vmax: P, direction: P) -> P:
    center = (vmin + vmax) / 2.0
    point = [0.0, 0.0, 0.0]
    for i in range(3):
        if direction[i] > 1e-12:
            point[i] = vmax[i]
        elif direction[i] < -1e-12:
            point[i] = vmin[i]
        else:
            point[i] = center[i]
    return _as_vec3(point)


def _support_of_primitive(prim: dict, direction: P) -> Tuple[float, P] | None:
    if prim["type"] == "boolean":
        return None

    M = _as_mat4(prim.get("xform", None))
    linear = np.asarray(M[:3, :3], dtype=float)
    translation = np.asarray(M[:3, 3], dtype=float).reshape(3)
    local_direction = linear.T @ direction

    params = prim["params"]
    local_point: P
    if prim["type"] == "sphere":
        center = _as_vec3(params.get("center", (0, 0, 0)))
        radius = float(params["radius"])
        local_norm = float(np.linalg.norm(local_direction))
        if local_norm < 1e-12:
            local_point = center
        else:
            local_point = center + radius * (local_direction / local_norm)
    elif prim["type"] == "cube":
        center = _as_vec3(params.get("center", (0, 0, 0)))
        half_size = _as_vec3(params.get("scale", (1, 1, 1))) / 2.0
        local_point = center.copy()
        for i in range(3):
            if local_direction[i] > 1e-12:
                local_point[i] += half_size[i]
            elif local_direction[i] < -1e-12:
                local_point[i] -= half_size[i]
    elif prim["type"] == "cylinder":
        p0 = _as_vec3(params["p0"])
        p1 = _as_vec3(params["p1"])
        radius = float(params["radius"])
        axis = p1 - p0
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-12:
            local_norm = float(np.linalg.norm(local_direction))
            if local_norm < 1e-12:
                local_point = p0
            else:
                local_point = p0 + radius * (local_direction / local_norm)
        else:
            axis_unit = axis / axis_norm
            axial_score = float(np.dot(local_direction, axis_unit))
            local_point = p1.copy() if axial_score >= 0.0 else p0.copy()
            radial_direction = local_direction - axial_score * axis_unit
            radial_norm = float(np.linalg.norm(radial_direction))
            if radial_norm >= 1e-12:
                local_point = local_point + radius * (radial_direction / radial_norm)
    else:
        return None

    world_point = _as_vec3(linear @ local_point + translation)
    return float(np.dot(world_point, direction)), world_point


def _support_result_from_aabb(shape: Asset, direction: P) -> Tuple[float, P] | None:
    vmin, vmax = _shape_aabb(shape)
    if not np.all(np.isfinite(vmin)) or not np.all(np.isfinite(vmax)):
        return None
    point = _aabb_support_point(vmin, vmax, direction)
    return float(np.dot(point, direction)), point


def _select_support_result(
    results: Iterable[Tuple[float, P] | None],
) -> Tuple[float, P] | None:
    best: Tuple[float, P] | None = None
    for result in results:
        if result is None:
            continue
        if best is None or result[0] > best[0]:
            best = result
    return best


def _shape_support_result(shape: Asset, direction: P) -> Tuple[float, P] | None:
    bool_mode = None
    for prim in shape.iter_local_primitives():
        if prim["type"] == "boolean":
            bool_mode = prim["params"]["mode"]
            break

    if bool_mode is not None:
        child_results = {
            name: _shape_support_result(child, direction)
            for name, child in shape._parts.items()
        }
        if bool_mode == "UNION":
            return _select_support_result(child_results.values())
        if bool_mode == "DIFFERENCE":
            base = shape._parts.get("base")
            if base is not None:
                return child_results.get("base")
            return _select_support_result(child_results.values())
        if bool_mode == "INTERSECT":
            return _support_result_from_aabb(shape, direction)

    local_results = [
        _support_of_primitive(prim, direction)
        for prim in shape.iter_local_primitives()
        if prim["type"] != "boolean"
    ]
    child_results = [
        _shape_support_result(child, direction)
        for child in shape._parts.values()
    ]
    result = _select_support_result(local_results + child_results)
    if result is not None:
        return result
    return _support_result_from_aabb(shape, direction)

def _shape_aabb(shape: Asset) -> Tuple[P, P]:
    bool_mode = None
    for prim in shape.iter_local_primitives():
        if prim["type"] == "boolean":
            bool_mode = prim["params"]["mode"]
            break
    if bool_mode is not None:
        child_aabbs = {
            name: _shape_aabb(ch) for name, ch in shape._parts.items()
        }
        if bool_mode == "DIFFERENCE":
            # Use base AABB as final AABB.
            # This can over-estimate when other parts extend beyond base.
            base = shape._parts.get("base")
            if base is None:
                if not shape._parts:
                    return (
                        _as_vec3(float("inf"), * 3), _as_vec3(float("-inf"), * 3)
                    )
                any_name = next(iter(shape._parts))
                return child_aabbs[any_name]
            return child_aabbs["base"]
        elif bool_mode == "UNION":
            # Combine all child AABBs using max extents.
            vmin = [float("inf")] * 3
            vmax = [float("-inf")] * 3
            for aabb in child_aabbs.values():
                vmin = np.minimum(vmin, aabb[0])
                vmax = np.maximum(vmax, aabb[1])
            return _as_vec3(vmin), _as_vec3(vmax)
        elif bool_mode == "INTERSECT":
            # Overlap all child AABBs using min extents.
            vmin = [float("-inf")] * 3
            vmax = [float("inf")] * 3
            for aabb in child_aabbs.values():
                vmin = np.maximum(vmin, aabb[0])
                vmax = np.minimum(vmax, aabb[1])
            if any(vmin[i] > vmax[i] for i in range(3)):
                return (
                    _as_vec3((float("inf"), ) * 3), 
                    _as_vec3((float("-inf"), ) * 3)
                )
            return _as_vec3(vmin), _as_vec3(vmax)
        
    has = False
    vmin = [float("inf")] * 3
    vmax = [float("-inf")] * 3
    for prim in shape.iter_local_primitives():
        if prim["type"] == "boolean":
            continue
        a, b = _aabb_of_primitive(prim)
        if a[0] == float("inf"):
            continue
        has = True
        vmin = np.minimum(vmin, a)
        vmax = np.maximum(vmax, b)
    for ch in shape._parts.values():
        a, b = _shape_aabb(ch)
        if a[0] == float("inf"):
            continue
        has = True
        vmin = np.minimum(vmin, a)
        vmax = np.maximum(vmax, b)
    if not has:
        return (
            _as_vec3((float("inf"),)* 3), 
            _as_vec3((float("-inf"),) * 3)
        )
    return _as_vec3(vmin), _as_vec3(vmax)

def shape_max(shape: Asset) -> P:
    """Get the maximum corner of the shape's AABB
    """
    _, vmax = _shape_aabb(shape)
    return vmax

def shape_min(shape: Asset) -> P:
    """Get the minimum corner of the shape's AABB
    """
    vmin, _ = _shape_aabb(shape)
    return vmin

def shape_size(shape: Asset) -> P:
    """Get the size (extent) of the shape's AABB
    """
    vmin, vmax = _shape_aabb(shape)
    return _as_vec3(vmax - vmin)

def shape_center(shape: Asset) -> P:
    """Get the center of the shape's AABB
    """
    vmin, vmax = _shape_aabb(shape)
    return (vmin + vmax) / 2.0

def shape_aabb(shape: Asset) -> Tuple[P, P]:
    """Get the AABB (min, max) of the shape
    """
    return _shape_aabb(shape)


def shape_support(shape: Asset, direction: Iterable[float] | str) -> P:
    """Return the point on ``shape`` furthest along ``direction``.

    For transformed primitives and non-boolean shape trees this is geometry-aware.
    Boolean difference/intersection may fall back to approximations.
    """
    normalized_direction = _normalize_direction(direction)
    result = _shape_support_result(ensure_shape(shape), normalized_direction)
    if result is None:
        raise ValueError("Cannot compute directional support for an empty shape.")
    return result[1]


def shape_bounds_along(shape: Asset, direction: Iterable[float] | str) -> Tuple[float, float]:
    """Return the min/max projection of ``shape`` along ``direction``.

    The returned distances are measured along the normalized query direction.
    """
    normalized_direction = _normalize_direction(direction)
    max_result = _shape_support_result(ensure_shape(shape), normalized_direction)
    min_result = _shape_support_result(ensure_shape(shape), -normalized_direction)
    if max_result is None or min_result is None:
        raise ValueError("Cannot compute directional bounds for an empty shape.")
    return float(-min_result[0]), float(max_result[0])


def shape_extent_along(shape: Asset, direction: Iterable[float] | str) -> float:
    """Return the span of ``shape`` along ``direction``."""
    vmin, vmax = shape_bounds_along(shape, direction)
    return float(vmax - vmin)

__all__ = [
    "shape_aabb",
    "shape_min",
    "shape_max",
    "shape_size",
    "shape_center",
    "shape_support",
    "shape_bounds_along",
    "shape_extent_along",
]
