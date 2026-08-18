from __future__ import annotations
from typing import Sequence
import numpy as np
from .asset import Asset, concat_shapes
from .math_utils import _I, _as_color, _as_vec3


def primitive_record(
    ptype: str,
    params: dict,
    color: Sequence[float] = (1.0, 1.0, 1.0),
    alpha: float | None = None,
) -> dict:
    return {
        "type": ptype,
        "params": params,
        "xform": _I(),
        "color": _as_color(color),
        "alpha": None if alpha is None else float(alpha),
    }


class PrimitiveAsset(Asset):
    def __init__(self, label: str) -> None:
        super().__init__(label=label)

    def _build(
        self,
        name: str,
        params: dict,
        color: Sequence[float]=(1.0, 1.0, 1.0),
        alpha: float | None = None
    ) -> None:
        self.add_primitive(
            primitive_record(name, params, color=color, alpha=alpha)
        )


class Sphere(PrimitiveAsset):
    def __init__(
        self,
        radius: float,
        center: Sequence[float]=(0, 0, 0),
        color: Sequence[float]=(1, 1, 1),
        alpha: float | None = None
    ):
        super().__init__(label="Sphere")
        
        r = float(radius)
        if r <= 0:
            raise ValueError("sphere radius must be > 0")
        
        cx, cy, cz = _as_vec3(center)
        
        self._build(
            name="sphere",
            params={
                "center": (cx, cy, cz), 
                "radius": r
            },
            color=color,
            alpha=alpha
        )

class Cube(PrimitiveAsset):
    def __init__(
        self,
        scale: Sequence[float] | float = 1.0,
        center: Sequence[float]=(0, 0, 0),
        color: Sequence[float]=(1, 1, 1),
        alpha: float | None = None
    ):
        super().__init__(label="Cube")
        
        if isinstance(scale, (int, float)):
            sx = sy = sz = float(scale)
        else:
            sx, sy, sz = _as_vec3(scale)
        if min(sx, sy, sz) <= 0:
            raise ValueError("cube scales must be > 0")

        cx, cy, cz = _as_vec3(center)
        
        self._build(
            name="cube",
            params={
                "center": (cx, cy, cz), 
                "scale": (sx, sy, sz)
            },
            color=color,
            alpha=alpha
        )

class Cylinder(PrimitiveAsset):
    def __init__(
        self,
        radius: float,
        *,
        p0: Sequence[float] | None = None,
        p1: Sequence[float] | None = None,
        height: float | None = None,
        center: Sequence[float] = (0.0, 0.0, 0.0),
        axis: str | Sequence[float] = "+z",
        color: Sequence[float]=(1, 1, 1),
        alpha: float | None = None,
    ):
        super().__init__(label="Cylinder")

        endpoint_mode = p0 is not None or p1 is not None
        if endpoint_mode:
            if p0 is None or p1 is None:
                raise ValueError("p0 and p1 must be provided together.")
            if height is not None:
                raise TypeError("height cannot be combined with p0 and p1.")
        else:
            if height is None:
                raise TypeError("Cylinder requires p0/p1 or height.")
            span = float(height)
            if span <= 0.0:
                raise ValueError("cylinder height must be > 0")
            from .axis import parse_axis

            direction = parse_axis(axis)
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-12:
                raise ValueError("cylinder axis must be non-zero")
            direction = direction / norm
            center_point = _as_vec3(center)
            p0 = center_point - direction * (span / 2.0)
            p1 = center_point + direction * (span / 2.0)

        a = _as_vec3(p0)
        b = _as_vec3(p1)
        r = float(radius)
        if r <= 0:
            raise ValueError("cylinder radius must be > 0")
        
        self._build(
            name="cylinder",
            params={
                "p0": (a[0], a[1], a[2]), 
                "p1": (b[0], b[1], b[2]), 
                "radius": r
            },
            color=color,
            alpha=alpha
    )


class RoundedCube(Asset):
    """Rounded box assembled from primitive faces, edges and corners."""

    def __init__(
        self,
        scale: Sequence[float] | float,
        radius: float,
        center: Sequence[float] = (0, 0, 0),
        color: Sequence[float] = (1, 1, 1),
        alpha: float | None = None,
    ):
        super().__init__(label="RoundedCube")
        size = np.asarray(
            [float(scale)] * 3 if isinstance(scale, (int, float)) else scale,
            dtype=float,
        )
        if size.shape != (3,) or np.min(size) <= 0:
            raise ValueError("rounded cube scales must be three positive values")
        radius = float(radius)
        if radius <= 0 or radius > float(np.min(size)) / 2:
            raise ValueError("radius must be > 0 and at most half the smallest scale")
        center_vec = np.asarray(center, dtype=float)
        inner = size - 2 * radius
        signs = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        corners = [center_vec + inner * np.asarray(sign) / 2 for sign in signs]
        pieces = [Sphere(radius, point, color, alpha) for point in corners]
        for i, left in enumerate(signs):
            for j in range(i + 1, len(signs)):
                right = signs[j]
                if sum(a != b for a, b in zip(left, right)) == 1:
                    pieces.append(
                        Cylinder(
                            radius,
                            p0=corners[i],
                            p1=corners[j],
                            color=color,
                            alpha=alpha,
                        )
                    )
        pieces.extend(
            [
                Cube((size[0], inner[1], inner[2]), center_vec, color, alpha),
                Cube((inner[0], size[1], inner[2]), center_vec, color, alpha),
                Cube((inner[0], inner[1], size[2]), center_vec, color, alpha),
            ]
        )
        self.attach_part("rounded_box", concat_shapes(pieces))


cube = Cube
sphere = Sphere
cylinder = Cylinder
rounded_cube = RoundedCube

__all__ = [
    "Sphere", "Cube", "Cylinder", "RoundedCube",
    "sphere", "cube", "cylinder", "rounded_cube",
]
