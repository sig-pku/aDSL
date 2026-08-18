from __future__ import annotations
from typing import List, Dict, Any

from pathlib import Path
import numpy as np

from ..asset import Asset
from ..joints import _normalize_joint_limit
from ..math_utils import _as_mat4

try:  # pragma: no cover - exercised only in Blender's Python
    import bpy
    import mathutils
except Exception as _blender_import_error:  # pragma: no cover
    bpy = None
    mathutils = None
    _BLENDER_IMPORT_ERROR = _blender_import_error
else:  # pragma: no cover
    _BLENDER_IMPORT_ERROR = None


def _require_blender() -> None:
    if bpy is None or mathutils is None:
        raise RuntimeError(
            "GLB export requires Blender's Python environment with bpy available."
        ) from _BLENDER_IMPORT_ERROR


def _joint_state_matrix(joint) -> np.ndarray:
    value = float(getattr(joint, "initial", 0.0))
    matrix = np.eye(4, dtype=float)
    axis = np.asarray(joint.axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    axis = np.array([0.0, 0.0, 1.0]) if norm < 1e-12 else axis / norm
    if joint.joint_type == "prismatic":
        matrix[:3, 3] = axis * value
    elif joint.joint_type == "revolute" and abs(value) > 1e-15:
        x, y, z = axis
        c, s, q = np.cos(value), np.sin(value), 1.0 - np.cos(value)
        matrix[:3, :3] = (
            (c + x*x*q, x*y*q - z*s, x*z*q + y*s),
            (y*x*q + z*s, c + y*y*q, y*z*q - x*s),
            (z*x*q - y*s, z*y*q + x*s, c + z*z*q),
        )
    return matrix


def _material_key(base: str, color, alpha):
    r, g, b = color
    a = 1.0 if alpha is None else float(alpha)
    ri = int(max(0, min(1, r)) * 255)
    gi = int(max(0, min(1, g)) * 255)
    bi = int(max(0, min(1, b)) * 255)
    ai = int(max(0, min(1, a)) * 255)
    return f"{base}_{ri:02x}{gi:02x}{bi:02x}_{ai:02x}"

def _make_material(name: str, color, alpha):
    key = _material_key(name, color, alpha)
    mat = bpy.data.materials.get(key)
    if mat is None:
        mat = bpy.data.materials.new(name=key)
        mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if bsdf is not None:
        r, g, b = color
        bsdf.inputs['Base Color'].default_value = (float(r), float(g), float(b), 1.0)
        if alpha is not None:
            bsdf.inputs['Alpha'].default_value = float(alpha)
            mat.blend_method = 'BLEND'
        else:
            mat.blend_method = 'OPAQUE'
    return mat

def _apply_xform_matrix(obj, M):
    if M is None:
        return
    mm = mathutils.Matrix(_as_mat4(M).tolist())
    obj.matrix_world = mm @ obj.matrix_world
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    obj.select_set(False)

def _make_sphere(prim: Dict[str, Any]) -> "bpy.types.Object":
    p = prim["params"]
    c = prim["color"]
    a = prim["alpha"]
    cx, cy, cz = p["center"]
    r = float(p["radius"])
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=(cx, cy, cz))
    obj = bpy.context.active_object
    _apply_xform_matrix(obj, prim.get("xform"))
    mat = _make_material("mat_sphere", c, a)
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    return obj

def _make_cube(prim: Dict[str, Any]) -> "bpy.types.Object":
    p = prim["params"]
    c = prim["color"]
    a = prim["alpha"]
    cx, cy, cz = p["center"]
    sx, sy, sz = p["scale"]
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(cx, cy, cz))
    obj = bpy.context.active_object
    obj.scale = (sx, sy, sz)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.select_set(False)
    _apply_xform_matrix(obj, prim.get("xform"))
    mat = _make_material("mat_cube", c, a)
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    return obj

def _make_cylinder(prim):
    p = prim["params"]
    c = prim["color"]
    a = prim["alpha"]

    p0 = mathutils.Vector(p["p0"])
    p1 = mathutils.Vector(p["p1"])
    r = float(p["radius"])

    v = p1 - p0
    length = v.length if v.length > 1e-8 else 1e-8
    mid = (p0 + p1) * 0.5

    z = mathutils.Vector((0, 0, 1))
    rot_quat = z.rotation_difference(v.normalized()) if v.length > 1e-8 else z.rotation_difference(z)

    bpy.ops.mesh.primitive_cylinder_add(
        radius=r,
        depth=length,
        location=mid,
        rotation=rot_quat.to_euler()
    )
    obj = bpy.context.active_object


    _apply_xform_matrix(obj, prim.get("xform"))
    mat = _make_material("mat_cylinder", c, a)
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    return obj

def _duplicate_object(
    obj: "bpy.types.Object"
) -> "bpy.types.Object":
    dup = obj.copy()
    dup.data = obj.data.copy()
    bpy.context.collection.objects.link(dup)
    return dup

def _apply_boolean(
    base_obj: "bpy.types.Object",
    other_obj: "bpy.types.Object",
    operation: str
):
    bpy.context.view_layer.objects.active = base_obj
    last_exc: Exception | None = None

    for solver in ("FAST", "EXACT"):
        modifier = base_obj.modifiers.new(
            name=f"Boolean_{operation}",
            type='BOOLEAN'
        )
        modifier.operation = operation
        modifier.solver = solver
        modifier.object = other_obj

        try:
            bpy.ops.object.modifier_apply(modifier=modifier.name)
            bpy.data.objects.remove(other_obj, do_unlink=True)
            return
        except Exception as exc:
            last_exc = exc
            try:
                base_obj.modifiers.remove(modifier)
            except Exception:
                pass

    bpy.data.objects.remove(other_obj, do_unlink=True)
    if last_exc is not None:
        raise last_exc

def _new_asset_node(
    name: str,
    *,
    parent=None,
    path: str,
    attach_mode: str,
    joint=None,
):
    # Blender object names are globally unique, so leaf-only names such as
    # ``front_panel`` are silently rewritten to ``front_panel.001`` when the
    # same part occurs in multiple subassemblies. Use the semantic path as the
    # stable GLB node name and retain the clean leaf name as explicit metadata.
    node = bpy.data.objects.new(str(path), None)
    bpy.context.collection.objects.link(node)
    node.empty_display_type = "PLAIN_AXES"
    node["adsl_kind"] = "asset"
    node["adsl_name"] = str(name)
    node["adsl_path"] = path
    node["adsl_attach_mode"] = attach_mode
    if joint is not None:
        node["adsl_joint_name"] = str(joint.name)
        node["adsl_joint_type"] = str(joint.joint_type)
        node["adsl_joint_axis"] = [float(value) for value in joint.axis]
        node["adsl_joint_initial"] = float(joint.initial)
        joint_limit = _normalize_joint_limit(joint.joint_type, joint.limit)
        if joint_limit is not None:
            node["adsl_joint_limit"] = [joint_limit[0], joint_limit[1]]
    if parent is not None:
        node.parent = parent
    return node


def _parent_geometry(obj, parent, *, name: str) -> None:
    world_matrix = obj.matrix_world.copy()
    obj.parent = parent
    obj.matrix_world = world_matrix
    obj.name = name
    obj["adsl_kind"] = "geometry"


def _build_shape(
    shape: Asset,
    world_xform=None,
    *,
    include_joint_children: bool = True,
    parent_node=None,
    node_name: str | None = None,
    path: str | None = None,
    attach_mode: str = "root",
    joint=None,
) -> List["bpy.types.Object"]:
    if world_xform is None:
        world_xform = np.eye(4, dtype=float)
    else:
        world_xform = _as_mat4(world_xform)

    resolved_name = str(node_name or shape.label or "asset")
    resolved_path = path or resolved_name
    hierarchy_node = _new_asset_node(
        resolved_name,
        parent=parent_node,
        path=resolved_path,
        attach_mode=attach_mode,
        joint=joint,
    )

    objs: List["bpy.types.Object"] = []
    consumed_children = set()

    def _build_local_geometry() -> None:
        for primitive_index, prim in enumerate(shape.iter_local_primitives()):
            prim = dict(prim)
            # Compose with incoming transform (e.g., from kinematic joints)
            px = prim.get("xform")
            if px is None:
                px = np.eye(4, dtype=float)
            else:
                px = _as_mat4(px)
            prim["xform"] = world_xform @ px

            t = prim["type"]
            if t == "sphere":
                cur_obj = _make_sphere(prim)
                _parent_geometry(cur_obj, hierarchy_node, name=f"geometry_{primitive_index}_sphere")
                cur_obj.data.name = f"{resolved_name}_sphere_mesh"
                objs.append(cur_obj)
            elif t == "cube":
                cur_obj = _make_cube(prim)
                _parent_geometry(cur_obj, hierarchy_node, name=f"geometry_{primitive_index}_cube")
                cur_obj.data.name = f"{resolved_name}_cube_mesh"
                objs.append(cur_obj)
            elif t == "cylinder":
                cur_obj = _make_cylinder(prim)
                _parent_geometry(cur_obj, hierarchy_node, name=f"geometry_{primitive_index}_cylinder")
                cur_obj.data.name = f"{resolved_name}_cylinder_mesh"
                objs.append(cur_obj)
            elif t == "boolean":
                mode = prim["params"]["mode"]
                child_names = sorted(shape._parts.keys())
                consumed_children.update(child_names)

                if mode in ["UNION", "INTERSECT"]:
                    built_children = []
                    for name in child_names:
                        built_children.extend(
                            _build_shape(
                                shape._parts[name],
                                world_xform,
                                include_joint_children=include_joint_children,
                                parent_node=hierarchy_node,
                                node_name=name,
                                path=f"{resolved_path}/{name}",
                                attach_mode="part",
                            )
                        )
                    if not built_children:
                        continue
                    base = built_children[0]
                    for other in built_children[1:]:
                        _apply_boolean(
                            base, other,
                            "UNION" if mode == "UNION" else "INTERSECT"
                        )
                    objs.append(base)
                elif mode == "DIFFERENCE":
                    base_objs = []
                    other_objs = []

                    if "base" in shape._parts:
                        base_objs = _build_shape(
                            shape._parts["base"],
                            world_xform,
                            include_joint_children=include_joint_children,
                            parent_node=hierarchy_node,
                            node_name="base",
                            path=f"{resolved_path}/base",
                            attach_mode="part",
                        )
                        for name in child_names:
                            if name.startswith("op_"):
                                other_objs.extend(
                                    _build_shape(
                                        shape._parts[name],
                                        world_xform,
                                        include_joint_children=include_joint_children,
                                        parent_node=hierarchy_node,
                                        node_name=name,
                                        path=f"{resolved_path}/{name}",
                                        attach_mode="part",
                                    )
                                )
                    else:
                        built_children = []
                        for name in child_names:
                            built_children.extend(
                                _build_shape(
                                    shape._parts[name],
                                    world_xform,
                                    include_joint_children=include_joint_children,
                                    parent_node=hierarchy_node,
                                    node_name=name,
                                    path=f"{resolved_path}/{name}",
                                    attach_mode="part",
                                )
                            )
                        if not built_children:
                            continue
                        base_objs, other_objs = [built_children[0]], built_children[1:]
                    if not base_objs:
                        continue
                    results = []
                    for base_obj in base_objs:
                        for other in other_objs:
                            dup = _duplicate_object(other)
                            _apply_boolean(base_obj, dup, "DIFFERENCE")
                        results.append(base_obj)
                    for other in other_objs:
                        try:
                            bpy.data.objects.remove(other, do_unlink=True)
                        except Exception:
                            pass
                    objs.extend(results)
                else:
                    raise ValueError(f"Unknown boolean mode: {mode}")
            else:
                raise ValueError(f"Unknown primitive type: {t}")

    _build_local_geometry()

    for name, child in shape._parts.items():
        if name in consumed_children:
            continue
        objs.extend(
            _build_shape(
                child,
                world_xform,
                include_joint_children=include_joint_children,
                parent_node=hierarchy_node,
                node_name=name,
                path=f"{resolved_path}/{name}",
                attach_mode="part",
            )
        )

    if include_joint_children:
        # Traverse kinematic joint children so articulated scenes export fully.
        for jname, child in getattr(shape, "_joint_children", {}).items():
            j = shape._joints.get(jname)
            if j is None:
                continue
            child_world = world_xform @ _as_mat4(j.origin) @ _joint_state_matrix(j)
            objs.extend(
                _build_shape(
                    child,
                    child_world,
                    include_joint_children=include_joint_children,
                    parent_node=hierarchy_node,
                    node_name=jname,
                    path=f"{resolved_path}/{jname}",
                    attach_mode="joint",
                    joint=j,
                )
            )

    return objs

def export_glb(
    shape: Asset,
    filepath: str | Path,
    *,
    clear_scene: bool = True,
    apply_modifiers: bool = True,
    draco: bool = False,
    include_joint_children: bool = True,
):
    _require_blender()
    export_path = Path(filepath)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    if clear_scene:
        bpy.ops.wm.read_factory_settings(use_empty=True)
    _build_shape(shape, include_joint_children=include_joint_children)
    if not any(obj.type == "MESH" for obj in bpy.data.objects):
        raise RuntimeError("Nothing to export: scene has no objects.")
    bpy.ops.export_scene.gltf(
        filepath=str(export_path),
        export_format='GLB',
        use_selection=False,
        export_apply=apply_modifiers,
        export_draco_mesh_compression_enable=bool(draco),
        export_extras=True,
    )
    return export_path

__all__ = ["export_glb"]
