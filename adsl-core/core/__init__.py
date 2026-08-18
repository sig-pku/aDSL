from __future__ import annotations

from .bounds import (
    shape_aabb,
    shape_bounds_along,
    shape_center,
    shape_extent_along,
    shape_max,
    shape_min,
    shape_size,
    shape_support,
)
from .boolean import (
    boolean_difference,
    boolean_intersection,
    boolean_union,
    boolean_xor,
)
from .layout import (
    align_anchors,
    align_centers,
    distribute_along_axis,
    grid_shapes,
    offset_from,
    place_on_axis,
    radial_shapes,
    shape_anchor,
    stack_shapes,
)
from .joints import Joint
from .math_utils import P, T
from .primitives import (
    Cube,
    Cylinder,
    RoundedCube,
    Sphere,
    cube,
    cylinder,
    rounded_cube,
    sphere,
)
from .transforms import (
    rotate_shape,
    rotation_matrix,
    scale_shape,
    scaling_matrix,
    transform_shape,
    translate_shape,
    translation_matrix,
)
from .asset import (
    Asset,
    concat_shapes,
)
from .appearance import get_link_rgba, set_link_color
from .axis import as_vec3, axis_index, parse_axis, unsigned_axis_name


def export_glb(*args, **kwargs):  # pragma: no cover
    """Export a scene to GLB, importing Blender support lazily."""
    from .export import export_glb as _export_glb

    return _export_glb(*args, **kwargs)


def export_manifest(*args, **kwargs):  # pragma: no cover
    """Serialize an Asset scene manifest lazily."""
    from .export import export_manifest as _export_manifest

    return _export_manifest(*args, **kwargs)


def to_urdf(*args, **kwargs):  # pragma: no cover
    """Serialize a scene to URDF, importing mesh support lazily."""
    from .export import to_urdf as _to_urdf

    return _to_urdf(*args, **kwargs)


def export_urdf(*args, **kwargs):  # pragma: no cover
    """Export a scene to URDF, importing mesh support lazily."""
    from .export import export_urdf as _export_urdf

    return _export_urdf(*args, **kwargs)

__all__ = [
    "Asset",
    "Joint",
    "concat_shapes",
    "P",
    "T",
    "Cube",
    "Sphere",
    "Cylinder",
    "RoundedCube",
    "sphere",
    "cube",
    "cylinder",
    "rounded_cube",
    "set_link_color",
    "get_link_rgba",
    "as_vec3",
    "parse_axis",
    "unsigned_axis_name",
    "axis_index",
    "boolean_union",
    "boolean_difference",
    "boolean_intersection",
    "boolean_xor",
    "export_manifest",
    "transform_shape",
    "translation_matrix",
    "translate_shape",
    "scaling_matrix",
    "scale_shape",
    "rotation_matrix",
    "rotate_shape",
    "shape_aabb",
    "shape_center",
    "shape_min",
    "shape_max",
    "shape_size",
    "shape_support",
    "shape_bounds_along",
    "shape_extent_along",
    "align_centers",
    "align_anchors",
    "place_on_axis",
    "distribute_along_axis",
    "grid_shapes",
    "radial_shapes",
    "stack_shapes",
    "offset_from",
    "shape_anchor",
    "export_glb",
    "to_urdf",
    "export_urdf",
]
