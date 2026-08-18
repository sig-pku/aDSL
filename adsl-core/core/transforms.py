from __future__ import annotations
from typing import Iterable, Optional
import numpy as np 

from .asset import Asset
from .axis import parse_axis
from .bounds import shape_center
from .math_utils import T, _as_mat4, _as_vec3


def _matmul(a, b):
    return _as_mat4(a) @ _as_mat4(b)

##########################
# General transformation #
##########################

def _transform_in_place(shape: Asset, M: T) -> Asset:
    """Apply a 4x4 transform to an Asset tree without replacing its identity.

    Semantics (important once joints exist):
    - Geometry parts (attached via `attach_part`) are *baked* by multiplying their
      primitive xforms.
    - Kinematic children (attached via `attach_joint`) are **not** recursively baked.
      Instead, the transform is pushed into the outgoing joint origin/axis so the
      entire articulated subtree moves rigidly without "double translating" link
      geometry.
    """

    M = _as_mat4(M)
    MR = np.asarray(M[:3, :3], dtype=float).reshape(3, 3)

    def _xform_joint(j):
        # Joint = [R t; 0 1]
        # Joint origin is stored in the parent's frame.
        H = _as_mat4(getattr(j, "origin", None))
        j.origin = _as_mat4(M @ H)

        # Joint axis is stored in the parent's frame.
        a = np.asarray(getattr(j, "axis", (0.0, 0.0, 1.0)), dtype=float).reshape(3)
        a2 = MR @ a
        n = float(np.linalg.norm(a2))
        if n < 1e-12:
            a2 = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            a2 = a2 / n
        j.axis = (float(a2[0]), float(a2[1]), float(a2[2]))

    def _apply(s: Asset):
        # Bake into this link's local primitives.
        for prim in s._primitives:
            xm = prim.get("xform", None)
            prim["xform"] = M if xm is None else _matmul(M, xm)

        # Bake into geometric (non-link) children.
        for ch in s._children.values():
            _apply(ch)

        # Push into outgoing joints (but do NOT recurse into joint children).
        for j in s._joints.values():
            _xform_joint(j)

    _apply(shape)
    return shape


def transform(shape: Asset, M: T) -> Asset:
    """Return a transformed independent copy of an Asset tree."""

    return _transform_in_place(shape.copy(), M)

transform_shape = transform

###############
# Translation #
###############

def translation_matrix(
    offset: Iterable[float]
) -> T:
    offset = _as_vec3(offset)
    m = np.eye(4, dtype=float)
    m[:3, 3] = offset
    return m

def translate_shape(
    shape: Asset,
    offset: Iterable[float]
) -> Asset:
    M = translation_matrix(offset)
    return transform(shape, M)

###########
# Scaling #
###########

def scaling_matrix(
    s: Iterable[float] | float,
    center: Optional[Iterable[float]] = None
) -> T:
    if isinstance(s, (int, float)):
        sx = sy = sz = float(s)
    else:
        sx, sy, sz = _as_vec3(s)
    S = np.eye(4, dtype=float)
    S[0,0] = sx
    S[1,1] = sy
    S[2,2] = sz

    if center is None:
        center = (0.0, 0.0, 0.0)
    c = _as_vec3(center)
    t = translation_matrix(c)
    ti = translation_matrix(-c)
    return t @ S @ ti

def scale_shape(
    shape: Asset,
    s: Iterable[float] | float,
    center: Optional[Iterable[float]] = None
) -> Asset:
    if center is None:
        center = shape_center(shape)
    M = scaling_matrix(s, center=center)
    return transform(shape, M)

############
# Rotation #
############

def rotation_matrix(
    axis: Iterable[float] | str, 
    angle: float,   # In DEGREES!
    center: Optional[Iterable[float]] = None
) -> T:
    axis = parse_axis(axis)
    angle_rad = float(angle) * np.pi / 180.0
    ax = _as_vec3(axis)
    norm = float(np.linalg.norm(ax))
    if norm < 1e-12:
        raise ValueError("rotation axis must be non-zero")
    ax = ax / norm
    x, y, z = ax.tolist()
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    C = 1.0 - c
    R = np.array([[x*x*C+c,   x*y*C - z*s, x*z*C + y*s, 0.0],
                  [y*x*C+z*s, y*y*C + c,   y*z*C - x*s, 0.0],
                  [z*x*C-y*s, z*y*C + x*s, z*z*C + c,   0.0],
                  [0.0,       0.0,         0.0,         1.0]], dtype=float)
    
    if center is None:
        center = (0.0, 0.0, 0.0)
    p = _as_vec3(center)
    t = translation_matrix(p)
    ti = translation_matrix(-p)
    return t @ R @ ti

def rotate_shape(
    shape: Asset,
    axis: Iterable[float] | str | None = None,
    angle: float | None = None,
    center: Optional[Iterable[float]] = None,
    euler: Optional[Iterable[float]] = None,
) -> Asset:
    if euler is not None:
        if axis is not None or angle is not None:
            raise TypeError("rotate_shape(euler=...) cannot be combined with axis or angle.")
        angles = _as_vec3(euler)
        rotation_center = shape_center(shape) if center is None else _as_vec3(center)
        result = shape
        for basis, degrees in zip(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            angles,
        ):
            if abs(float(degrees)) > 1e-12:
                result = transform(
                    result,
                    rotation_matrix(basis, float(degrees), center=rotation_center),
                )
        return result
    if axis is None:
        raise TypeError("rotate_shape requires an axis/angle pair or an euler tuple.")
    if angle is None:
        raise TypeError("rotate_shape requires angle when axis is provided.")
    if center is None:
        center = shape_center(shape)
    M = rotation_matrix(axis, angle, center=center)
    return transform(shape, M)
