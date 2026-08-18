from __future__ import annotations

from typing import Dict, Tuple, Optional, Iterable, List, Mapping
import math
import os
import json
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom

from ..asset import Asset
from ..bounds import shape_aabb
from ..joints import Joint, _normalize_joint_limit
from ..math_utils import T, _as_mat4, _as_vec3

# Optional deps for CSG meshing
try:  # pragma: no cover
    import trimesh
except Exception:  # pragma: no cover
    trimesh = None

try:  # pragma: no cover
    from skimage import measure as _sk_measure
except Exception:  # pragma: no cover
    _sk_measure = None

def _sanitize_name(name: str) -> str:
    s = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in (name or "link"))
    s = s.strip("_")
    return s or "link"


def _pretty_xml(elem: ET.Element) -> str:
    rough = ET.tostring(elem, encoding="utf-8")
    reparsed = minidom.parseString(rough)
    return reparsed.toprettyxml(indent="  ")


def _rpy_from_R(R: np.ndarray) -> Tuple[float, float, float]:
    """Convert rotation matrix to roll-pitch-yaw (URDF convention)."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    sy = math.sqrt(float(R[0, 0]) ** 2 + float(R[1, 0]) ** 2)
    singular = sy < 1e-9
    if not singular:
        roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
        pitch = math.atan2(-float(R[2, 0]), sy)
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        roll = math.atan2(-float(R[1, 2]), float(R[1, 1]))
        pitch = math.atan2(-float(R[2, 0]), sy)
        yaw = 0.0
    return (roll, pitch, yaw)


def _decompose_mat4(M: T) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    """Return (xyz, rpy, scale_xyz) from a 4x4 transform.

    Assumes the matrix is composed of rotation * diagonal scale (no shear).
    """
    M = _as_mat4(M)
    t = (float(M[0, 3]), float(M[1, 3]), float(M[2, 3]))
    A = M[:3, :3].copy()
    sx = float(np.linalg.norm(A[:, 0]))
    sy = float(np.linalg.norm(A[:, 1]))
    sz = float(np.linalg.norm(A[:, 2]))
    scale = (sx, sy, sz)
    if sx > 1e-12:
        A[:, 0] /= sx
    if sy > 1e-12:
        A[:, 1] /= sy
    if sz > 1e-12:
        A[:, 2] /= sz
    rpy = _rpy_from_R(A)
    return t, rpy, scale


def _translation_matrix(xyz: Iterable[float]) -> np.ndarray:
    x, y, z = _as_vec3(xyz)
    M = np.eye(4, dtype=float)
    M[:3, 3] = [x, y, z]
    return M


def _rotation_align_z_to(v: np.ndarray) -> np.ndarray:
    """Return 3x3 R so that R*[0,0,1] = v (v must be unit)."""
    z = np.array([0.0, 0.0, 1.0], dtype=float)
    v = np.asarray(v, dtype=float).reshape(3)
    v = v / (np.linalg.norm(v) + 1e-15)

    c = float(np.dot(z, v))
    if c > 1.0 - 1e-9:
        return np.eye(3, dtype=float)
    if c < -1.0 + 1e-9:
        # 180deg: rotate about x
        return np.array([[1, 0, 0],
                         [0, -1, 0],
                         [0, 0, -1]], dtype=float)

    axis = np.cross(z, v)
    s = float(np.linalg.norm(axis))
    axis = axis / (s + 1e-15)
    x, y, zc = axis.tolist()

    K = np.array([[0, -zc, y],
                  [zc, 0, -x],
                  [-y, x, 0]], dtype=float)
    R = np.eye(3, dtype=float) + K * s + (K @ K) * (1.0 - c)
    return R



# -----------------------------
# Primitive URDF element export
# -----------------------------

def _prim_param_pose(prim: Dict) -> np.ndarray:
    """Pose implied by primitive params (center, p0/p1, etc)."""
    t = prim.get("type")
    p = prim.get("params", {})
    if t == "cube":
        return _translation_matrix(p.get("center", (0.0, 0.0, 0.0)))
    if t == "sphere":
        return _translation_matrix(p.get("center", (0.0, 0.0, 0.0)))
    if t == "cylinder":
        p0 = np.asarray(p["p0"], dtype=float).reshape(3)
        p1 = np.asarray(p["p1"], dtype=float).reshape(3)
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 1e-12:
            raise ValueError("Cylinder has zero length (p0 == p1)")
        u = d / L
        R = _rotation_align_z_to(u)
        M = np.eye(4, dtype=float)
        M[:3, :3] = R
        M[:3, 3] = (p0 + p1) / 2.0
        return M
    if t == "boolean":
        raise ValueError("Boolean primitive reached primitive-pose computation; traversal bug.")
    raise ValueError(f"Unknown primitive type: {t}")


def _semantic_entry_name(path: str, tag: str, index: int = 0) -> str:
    """Encode a GLB-compatible asset path in a globally unique URDF entry name."""
    return f"{path}#{tag}_{index}"


def _iter_primitives_for_urdf(shape: Asset, semantic_path: str) -> Iterable[Dict]:
    """Yield non-boolean primitives to export for URDF.

    CSG is approximated here (primitives mode only):
      - UNION: export all operands
      - DIFFERENCE: export only the base operand
      - INTERSECT: export only the first operand
    """
    local = list(shape.iter_local_primitives())
    bool_prims = [p for p in local if p.get("type") == "boolean"]
    if bool_prims:
        mode = str(bool_prims[0].get("params", {}).get("mode", "UNION")).upper()
        children = list(shape._children.items())
        base = shape._children.get("base", None)
        ops = [(k, v) for k, v in children if k.startswith("op_")]
        ops.sort(key=lambda kv: kv[0])

        if mode == "UNION":
            if base is not None:
                yield from _iter_primitives_for_urdf(base, f"{semantic_path}/base")
            for name, ch in ops:
                yield from _iter_primitives_for_urdf(ch, f"{semantic_path}/{name}")
        elif mode == "DIFFERENCE":
            if base is not None:
                yield from _iter_primitives_for_urdf(base, f"{semantic_path}/base")
            elif ops:
                yield from _iter_primitives_for_urdf(ops[0][1], f"{semantic_path}/{ops[0][0]}")
        elif mode == "INTERSECT":
            if base is not None:
                yield from _iter_primitives_for_urdf(base, f"{semantic_path}/base")
            elif ops:
                yield from _iter_primitives_for_urdf(ops[0][1], f"{semantic_path}/{ops[0][0]}")
        else:
            raise ValueError(f"Unknown boolean mode: {mode}")
        return

    # Normal: local primitives + recurse into geometry children
    for primitive_index, prim in enumerate(local):
        if prim.get("type") == "boolean":
            continue
        named_prim = dict(prim)
        named_prim["_adsl_path"] = semantic_path
        named_prim["_adsl_primitive_index"] = primitive_index
        yield named_prim
    for child_name, child in shape._children.items():
        # If a child subtree contains joints, it is exported as its own URDF link
        # connected via a fixed joint. Do not flatten its geometry into this link.
        if _subtree_has_joint(child):
            continue
        yield from _iter_primitives_for_urdf(child, f"{semantic_path}/{child_name}")


def _add_origin(origin_elem: ET.Element, xyz: Tuple[float, float, float], rpy: Tuple[float, float, float]) -> None:
    origin_elem.set("xyz", f"{xyz[0]:.9g} {xyz[1]:.9g} {xyz[2]:.9g}")
    origin_elem.set("rpy", f"{rpy[0]:.9g} {rpy[1]:.9g} {rpy[2]:.9g}")


def _add_visual_or_collision(robot_elem: ET.Element, material_defined: set, link_elem: ET.Element, prim: Dict, tag: str, link_adjust: Optional[np.ndarray] = None) -> None:
    # Combined pose: xform is applied after params pose
    xform = prim.get("xform", None)
    M = (_as_mat4(xform) @ _prim_param_pose(prim)) if xform is not None else _prim_param_pose(prim)

    if link_adjust is not None:
        M = np.asarray(link_adjust, dtype=float).reshape(4, 4) @ M

    xyz, rpy, sxyz = _decompose_mat4(M)

    t = prim["type"]
    p = prim["params"]

    semantic_path = str(prim.get("_adsl_path", link_elem.get("name", "link")))
    primitive_index = int(prim.get("_adsl_primitive_index", 0))
    entry = ET.SubElement(
        link_elem,
        tag,
        {"name": _semantic_entry_name(semantic_path, tag, primitive_index)},
    )
    origin = ET.SubElement(entry, "origin")
    _add_origin(origin, xyz, rpy)

    geom = ET.SubElement(entry, "geometry")

    if t == "cube":
        size = np.asarray(p.get("scale", (1.0, 1.0, 1.0)), dtype=float).reshape(3)
        size = size * np.asarray(sxyz, dtype=float)
        box = ET.SubElement(geom, "box")
        box.set("size", f"{size[0]:.9g} {size[1]:.9g} {size[2]:.9g}")
    elif t == "sphere":
        r = float(p.get("radius", 0.5))
        # assume uniform scale
        r *= float(sxyz[0])
        sph = ET.SubElement(geom, "sphere")
        sph.set("radius", f"{r:.9g}")
    elif t == "cylinder":
        p0 = np.asarray(p["p0"], dtype=float).reshape(3)
        p1 = np.asarray(p["p1"], dtype=float).reshape(3)
        L = float(np.linalg.norm(p1 - p0))
        r = float(p.get("radius", 0.5))
        r *= float(sxyz[0])
        L *= float(sxyz[2])
        cyl = ET.SubElement(geom, "cylinder")
        cyl.set("radius", f"{r:.9g}")
        cyl.set("length", f"{L:.9g}")
    else:
        raise ValueError(f"Unknown primitive type: {t}")

    if tag == "visual":
        c = prim.get("color", (1.0, 1.0, 1.0))
        a = prim.get("alpha", None)
        rgba = (float(c[0]), float(c[1]), float(c[2]), 1.0 if a is None else float(a))
        mat_name = _material_name(f"mat_{t}", rgba)
        _ensure_robot_material(robot_elem, material_defined, mat_name, rgba)

        # Keep the named material and inline color consistent across URDF loaders.
        mat = ET.SubElement(entry, "material", {"name": mat_name})
        ET.SubElement(mat, "color", {"rgba": _rgba_str(rgba)})


# ----------------------------
# Kinematic tree traversal
# ----------------------------


def _subtree_has_joint(shape: Asset) -> bool:
    """Return True if `shape` contains any kinematic joints anywhere in its subtree.

    This is used by the URDF exporter to *promote* jointed subassemblies that are
    attached as rigid geometry parts into proper URDF links connected by fixed joints.
    """
    seen: set[int] = set()

    def rec(s: Asset) -> bool:
        sid = id(s)
        if sid in seen:
            return False
        seen.add(sid)
        if getattr(s, "_joint_children", None):
            if len(s._joint_children) > 0:
                return True
        for ch in getattr(s, "_children", {}).values():
            if rec(ch):
                return True
        return False

    return rec(shape)


def _collect_links_and_joints(root: Asset) -> Tuple[List[Asset], List[Tuple[Asset, str, Joint, Asset]]]:
    """Collect URDF links and joints starting from `root`.

    The DSL allows a jointed subassembly to be attached under `_children` (geometry parts).
    URDF, however, requires the entire kinematic structure to be connected via joints.
    To bridge this, we *promote* any geometry child subtree that contains joints into a
    proper URDF link, connected to its parent by a synthetic fixed joint (identity origin).
    """
    links: List[Asset] = []
    joints: List[Tuple[Asset, str, Joint, Asset]] = []
    seen_links: set[int] = set()

    def _unique_joint_name(parent: Asset, base: str) -> str:
        # Ensure uniqueness among already-added joints on this parent in this export pass.
        cand = base
        k = 1
        existing = {jn for (p, jn, _j, _c) in joints if p is parent}
        while cand in existing:
            cand = f"{base}_{k}"
            k += 1
        return cand

    def dfs(link: Asset):
        lid = id(link)
        if lid in seen_links:
            return
        seen_links.add(lid)
        links.append(link)

        # 1) Real kinematic children
        for jname, child in link._joint_children.items():
            j = link._joints[jname]
            joints.append((link, jname, j, child))
            dfs(child)

        # 2) Promote rigid geometry children that contain joints into fixed-joint links
        for cname, ch in getattr(link, "_children", {}).items():
            if not _subtree_has_joint(ch):
                continue
            jname = _unique_joint_name(link, f"{cname}_fixed")
            j = Joint(
                name=jname,
                joint_type="fixed",
                axis=_as_vec3((0.0, 0.0, 1.0)),
                origin=np.eye(4, dtype=float),
                limit=None,
            )
            joints.append((link, jname, j, ch))
            dfs(ch)

    dfs(root)
    return links, joints


def _assign_unique_link_names(root: Asset) -> Dict[int, str]:
    links, _ = _collect_links_and_joints(root)
    used: set[str] = set()
    out: Dict[int, str] = {}
    counts: Dict[str, int] = {}
    for link in links:
        base = _sanitize_name(getattr(link, "label", None) or "link")
        n = counts.get(base, 0)
        candidate = base if n == 0 else f"{base}_{n}"
        while candidate in used:
            n += 1
            candidate = f"{base}_{n}"
        counts[base] = n + 1
        used.add(candidate)
        out[id(link)] = candidate
    return out


def _assign_link_semantic_paths(
    root: Asset,
) -> Dict[int, str]:
    """Return the same semantic asset paths used by the GLB hierarchy exporter."""
    root_name = str(getattr(root, "label", None) or "root")
    paths: Dict[int, str] = {id(root): root_name}

    def visit(parent: Asset) -> None:
        parent_path = paths[id(parent)]
        link_children = list(parent._joint_children.items())
        link_children.extend(
            (child_name, child)
            for child_name, child in parent._children.items()
            if _subtree_has_joint(child)
        )
        for edge_name, child in link_children:
            child_id = id(child)
            if child_id in paths:
                continue
            paths[child_id] = f"{parent_path}/{edge_name}"
            visit(child)

    visit(root)
    return paths


def _assign_unique_joint_names(
    joints: Iterable[Tuple[Asset, str, Joint, Asset]]
) -> Dict[Tuple[int, str, int], str]:
    used: set[str] = set()
    out: Dict[Tuple[int, str, int], str] = {}
    counts: Dict[str, int] = {}
    for parent, jname, _j, child in joints:
        base = _sanitize_name(jname)
        n = counts.get(base, 0)
        candidate = base if n == 0 else f"{base}_{n}"
        while candidate in used:
            n += 1
            candidate = f"{base}_{n}"
        counts[base] = n + 1
        used.add(candidate)
        out[(id(parent), jname, id(child))] = candidate
    return out


# ----------------------------
# CSG meshing (voxel + MC)
# ----------------------------

def _require_meshing_deps():
    if trimesh is None:
        raise RuntimeError("trimesh is required for mesh URDF export (pip install trimesh).")
    if _sk_measure is None:
        raise RuntimeError("scikit-image is required for CSG meshing (pip install scikit-image).")
    return True


def _shape_has_boolean(shape: Asset) -> bool:
    for p in shape._primitives:
        if p.get("type") == "boolean":
            return True
    for ch in shape._children.values():
        if _shape_has_boolean(ch):
            return True
    return False


def _primitive_inside(prim: Mapping, pts: np.ndarray) -> np.ndarray:
    t = prim.get("type")
    if t == "boolean":
        return np.zeros(len(pts), dtype=bool)

    M = prim.get("xform")
    if M is None:
        Minv = np.eye(4, dtype=float)
    else:
        Minv = np.linalg.inv(_as_mat4(M))

    pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=float)], axis=1)
    local = (pts_h @ Minv.T)[:, :3]
    params = prim.get("params", {})

    if t == "cube":
        s = np.asarray(params.get("scale", (1, 1, 1)), dtype=float).reshape(3)
        c = np.asarray(params.get("center", (0, 0, 0)), dtype=float).reshape(3)
        half = s * 0.5
        d = np.abs(local - c) <= (half + 1e-9)
        return d[:, 0] & d[:, 1] & d[:, 2]

    if t == "sphere":
        r = float(params.get("radius", 0.5))
        c = np.asarray(params.get("center", (0, 0, 0)), dtype=float).reshape(3)
        d = local - c
        return (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2]) <= (r * r + 1e-9)

    if t == "cylinder":
        p0 = np.asarray(params.get("p0", (0, 0, 0)), dtype=float).reshape(3)
        p1 = np.asarray(params.get("p1", (0, 0, 1)), dtype=float).reshape(3)
        r = float(params.get("radius", 0.5))
        v = p1 - p0
        L2 = float(np.dot(v, v))
        if L2 < 1e-12:
            d = local - p0
            return (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2]) <= (r * r + 1e-9)
        w = local - p0
        tproj = (w @ v) / L2
        inside_cap = (tproj >= -1e-6) & (tproj <= 1.0 + 1e-6)
        closest = p0 + np.clip(tproj, 0.0, 1.0)[:, None] * v[None, :]
        d = local - closest
        inside_rad = (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2]) <= (r * r + 1e-9)
        return inside_cap & inside_rad

    raise ValueError(f"Unsupported primitive type for meshing: {t}")


def _shape_occupancy(shape: Asset, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    N = len(pts)
    occ = np.zeros(N, dtype=bool)

    bool_prims = [p for p in shape._primitives if p.get("type") == "boolean"]
    if bool_prims:
        mode = str(bool_prims[0].get("params", {}).get("mode", "UNION")).upper()
        children = dict(shape._children)
        operands: List[Asset] = []
        if "base" in children:
            operands.append(children["base"])
        ops = [(k, v) for k, v in children.items() if k.startswith("op_")]
        ops.sort(key=lambda kv: kv[0])
        operands.extend([v for _, v in ops])

        if operands:
            occs = [_shape_occupancy(op, pts) for op in operands]
            if mode == "UNION":
                occ |= np.logical_or.reduce(occs)
            elif mode == "INTERSECT":
                occ |= np.logical_and.reduce(occs)
            elif mode == "DIFFERENCE":
                base_occ = occs[0]
                sub_occ = np.logical_or.reduce(occs[1:]) if len(occs) > 1 else np.zeros(N, dtype=bool)
                occ |= base_occ & (~sub_occ)
            else:
                raise ValueError(f"Unknown boolean mode: {mode}")

        operand_ids = {id(op) for op in operands}
        for ch in shape._children.values():
            if id(ch) not in operand_ids:
                occ |= _shape_occupancy(ch, pts)
    else:
        for ch in shape._children.values():
            occ |= _shape_occupancy(ch, pts)

    for prim in shape._primitives:
        if prim.get("type") == "boolean":
            continue
        occ |= _primitive_inside(prim, pts)

    return occ


def _choose_pitch(vmin: np.ndarray, vmax: np.ndarray, pitch: float, max_voxels: int) -> float:
    ext = np.maximum(vmax - vmin, 1e-6)
    p = float(pitch)
    dims = np.ceil(ext / p).astype(int) + 5
    vox = int(dims[0] * dims[1] * dims[2])
    if vox <= max_voxels:
        return p
    scale = (vox / float(max_voxels)) ** (1.0 / 3.0)
    return p * scale


def _mesh_from_shape_csg(shape: Asset, *, pitch: float = 0.01, max_voxels: int = 2_000_000) -> "trimesh.Trimesh":
    _require_meshing_deps()

    vmin, vmax = shape_aabb(shape)
    vmin = np.asarray(vmin, dtype=float).reshape(3)
    vmax = np.asarray(vmax, dtype=float).reshape(3)

    pitch = _choose_pitch(vmin, vmax, pitch, max_voxels)
    margin = 2.0 * pitch
    bmin = vmin - margin
    bmax = vmax + margin
    ext = np.maximum(bmax - bmin, 1e-6)

    nx, ny, nz = (np.ceil(ext / pitch).astype(int) + 1).tolist()
    xs = bmin[0] + np.arange(nx) * pitch
    ys = bmin[1] + np.arange(ny) * pitch
    zs = bmin[2] + np.arange(nz) * pitch

    X, Y = np.meshgrid(xs, ys, indexing="ij")
    base_xy = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
    volume = np.zeros((nx, ny, nz), dtype=np.uint8)

    for k, z in enumerate(zs):
        pts = np.concatenate([base_xy, np.full((base_xy.shape[0], 1), float(z))], axis=1)
        occ = _shape_occupancy(shape, pts).astype(np.uint8)
        volume[:, :, k] = occ.reshape(nx, ny)

    if volume.max() == 0:
        return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int), process=False)

    verts, faces, _normals, _values = _sk_measure.marching_cubes(volume.astype(np.float32), level=0.5, spacing=(pitch, pitch, pitch))
    verts = verts + bmin[None, :]

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    if mesh.faces.size > 0:
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_infinite_values()
        mesh.remove_unreferenced_vertices()
    return mesh




def _rgba_str(rgba: Tuple[float, float, float, float]) -> str:
    r, g, b, a = rgba
    return f"{r:.6g} {g:.6g} {b:.6g} {a:.6g}"


def _material_name(prefix: str, rgba: Tuple[float, float, float, float]) -> str:
    r, g, b, a = rgba
    rr = max(0, min(255, int(round(r * 255))))
    gg = max(0, min(255, int(round(g * 255))))
    bb = max(0, min(255, int(round(b * 255))))
    aa = max(0, min(255, int(round(a * 255))))
    return f"{prefix}_{rr:02x}{gg:02x}{bb:02x}{aa:02x}"



def _ensure_robot_material(robot_elem: ET.Element, defined: set, name: str, rgba: Tuple[float, float, float, float]) -> None:
    """Define a named URDF material at the robot-level if not already defined.

    Materials are defined globally and referenced by name in visuals. They are
    inserted before link and joint entries for deterministic parser behavior.
    """
    if name in defined:
        return

    mat = ET.Element("material", {"name": name})
    ET.SubElement(mat, "color", {"rgba": _rgba_str(rgba)})

    # Insert before the first link/joint element if possible.
    insert_at = len(robot_elem)
    for i, child in enumerate(list(robot_elem)):
        if child.tag in ("link", "joint"):
            insert_at = i
            break
    robot_elem.insert(insert_at, mat)
    defined.add(name)

def _export_mesh_collada(mesh: "trimesh.Trimesh", dae_path: str, rgba: Tuple[float, float, float, float]) -> None:
    """Export a minimal COLLADA (.dae) file with a single colored material.

    Why this custom writer?
    - Some URDF viewers ignore OBJ+MTL colors.
    - Some URDF viewers do not parse interleaved VERTEX/NORMAL indices.
      This writer only emits
      POSITION data and triangle indices; normals are left for the viewer
      to compute.
    """
    if mesh.faces is None or len(mesh.faces) == 0 or mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError("Cannot export an empty mesh to COLLADA")

    # Ensure clean arrays
    v = np.asarray(mesh.vertices, dtype=float).reshape(-1, 3)
    faces = np.asarray(mesh.faces, dtype=int).reshape(-1, 3)

    # Flatten arrays
    pos_list = " ".join(f"{x:.9g} {y:.9g} {z:.9g}" for x, y, z in v.tolist())
    p_str = " ".join(f"{int(a)} {int(b)} {int(c)}" for a, b, c in faces.tolist())

    r, g, b, a = rgba
    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>

  <library_effects>
    <effect id="mat0-effect">
      <profile_COMMON>
        <technique sid="common">
          <lambert><diffuse><color>{r:.6g} {g:.6g} {b:.6g} {a:.6g}</color></diffuse></lambert>
        </technique>
      </profile_COMMON>
    </effect>
  </library_effects>

  <library_materials>
    <material id="mat0" name="mat0">
      <instance_effect url="#mat0-effect"/>
    </material>
  </library_materials>

  <library_geometries>
    <geometry id="geom0" name="geom0">
      <mesh>
        <source id="geom0-positions">
          <float_array id="geom0-positions-array" count="{len(v)*3}">{pos_list}</float_array>
          <technique_common>
            <accessor source="#geom0-positions-array" count="{len(v)}" stride="3">
              <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>

        <vertices id="geom0-vertices">
          <input semantic="POSITION" source="#geom0-positions"/>
        </vertices>

        <triangles material="mat0" count="{len(faces)}">
          <input semantic="VERTEX" source="#geom0-vertices" offset="0"/>
          <p>{p_str}</p>
        </triangles>
      </mesh>
    </geometry>
  </library_geometries>

  <library_visual_scenes>
    <visual_scene id="Scene" name="Scene">
      <node id="node0" name="node0">
        <instance_geometry url="#geom0">
          <bind_material>
            <technique_common>
              <instance_material symbol="mat0" target="#mat0"/>
            </technique_common>
          </bind_material>
        </instance_geometry>
      </node>
    </visual_scene>
  </library_visual_scenes>

  <scene><instance_visual_scene url="#Scene"/></scene>
</COLLADA>
"""
    os.makedirs(os.path.dirname(dae_path) or ".", exist_ok=True)
    with open(dae_path, "w", encoding="utf-8") as f:
        f.write(xml)
def _export_mesh_with_optional_mtl(mesh: "trimesh.Trimesh", mesh_path: str, rgba: Tuple[float, float, float, float]) -> None:
    """Export mesh to file. If OBJ, also write a minimal MTL for color."""
    ext = os.path.splitext(mesh_path)[1].lower()
    if ext == ".dae":
        _export_mesh_collada(mesh, mesh_path, rgba)
        return
    if ext != ".obj":
        mesh.export(mesh_path)
        return

    # Ensure normals
    try:
        _ = mesh.vertex_normals
    except Exception:
        pass

    mat_name = "mat0"
    obj_name = os.path.splitext(os.path.basename(mesh_path))[0]
    mtl_name = f"{obj_name}.mtl"
    mtl_path = os.path.join(os.path.dirname(mesh_path), mtl_name)

    r, g, b, a = rgba
    with open(mtl_path, "w", encoding="utf-8") as f:
        f.write(f"newmtl {mat_name}\n")
        f.write(f"Kd {r:.6g} {g:.6g} {b:.6g}\n")
        f.write("Ka 0 0 0\n")
        f.write("Ks 0 0 0\n")
        f.write("Ns 1\n")
        f.write(f"d {a:.6g}\n")

    # Write OBJ (v, vn, f)
    # If normals are missing, trimesh will compute them lazily.
    v = mesh.vertices
    n = mesh.vertex_normals if hasattr(mesh, "vertex_normals") else None
    faces = mesh.faces

    with open(mesh_path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_name}\n")
        f.write(f"o {obj_name}\n")
        for vv in v:
            f.write(f"v {vv[0]:.6g} {vv[1]:.6g} {vv[2]:.6g}\n")
        if n is None or len(n) != len(v):
            # compute normals
            try:
                mesh.rezero()
                mesh.compute_vertex_normals()
                n = mesh.vertex_normals
            except Exception:
                n = None
        if n is not None and len(n) == len(v):
            for nn in n:
                f.write(f"vn {nn[0]:.6g} {nn[1]:.6g} {nn[2]:.6g}\n")
            f.write(f"usemtl {mat_name}\n")
            for tri in faces:
                a_i, b_i, c_i = int(tri[0])+1, int(tri[1])+1, int(tri[2])+1
                f.write(f"f {a_i}//{a_i} {b_i}//{b_i} {c_i}//{c_i}\n")
        else:
            f.write(f"usemtl {mat_name}\n")
            for tri in faces:
                a_i, b_i, c_i = int(tri[0])+1, int(tri[1])+1, int(tri[2])+1
                f.write(f"f {a_i} {b_i} {c_i}\n")

def _add_mesh_visual(
    robot_elem: ET.Element,
    material_defined: set,
    link_elem: ET.Element,
    mesh_filename: str,
    rgba: Tuple[float, float, float, float],
    material_name: str,
    *,
    origin_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    origin_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    mesh_scale: Tuple[float, float, float] | None = None,
    semantic_path: str | None = None,
) -> None:
    path = semantic_path or link_elem.get("name", "link")
    vis = ET.SubElement(link_elem, "visual", {"name": _semantic_entry_name(path, "visual")})
    origin = ET.SubElement(vis, "origin")
    _add_origin(origin, origin_xyz, origin_rpy)
    geom = ET.SubElement(vis, "geometry")
    mesh_attrib = {"filename": mesh_filename}
    if mesh_scale is not None and not np.allclose(mesh_scale, (1.0, 1.0, 1.0), atol=1e-12):
        mesh_attrib["scale"] = f"{mesh_scale[0]:.9g} {mesh_scale[1]:.9g} {mesh_scale[2]:.9g}"
    ET.SubElement(geom, "mesh", mesh_attrib)
    _ensure_robot_material(robot_elem, material_defined, material_name, rgba)
    mat = ET.SubElement(vis, "material", {"name": material_name})
    ET.SubElement(mat, "color", {"rgba": _rgba_str(rgba)})

def _add_mesh_collision(
    link_elem: ET.Element,
    mesh_filename: str,
    *,
    origin_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    origin_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    mesh_scale: Tuple[float, float, float] | None = None,
    semantic_path: str | None = None,
) -> None:
    path = semantic_path or link_elem.get("name", "link")
    col = ET.SubElement(link_elem, "collision", {"name": _semantic_entry_name(path, "collision")})
    origin = ET.SubElement(col, "origin")
    _add_origin(origin, origin_xyz, origin_rpy)
    geom = ET.SubElement(col, "geometry")
    mesh_attrib = {"filename": mesh_filename}
    if mesh_scale is not None and not np.allclose(mesh_scale, (1.0, 1.0, 1.0), atol=1e-12):
        mesh_attrib["scale"] = f"{mesh_scale[0]:.9g} {mesh_scale[1]:.9g} {mesh_scale[2]:.9g}"
    ET.SubElement(geom, "mesh", mesh_attrib)


def _mesh_origin_correction_rpy(
    mesh_filename: str,
    *,
    gltf_axis_fix: bool,
    gltf_axis_fix_rpy: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Return a fixed rotation correction for mesh files whose coordinate convention
    differs from URDF's (Z-up).

    Blender's GLB/GLTF exporter writes meshes in glTF's Y-up convention. Many URDF
    consumers load GLB/GLTF as a raw mesh without applying that axis conversion,
    which makes Z-up shapes appear rotated (e.g., a bottle lying on its side).

    This correction rotates the mesh so that glTF +Y maps to URDF +Z.
    """
    ext = os.path.splitext(mesh_filename.lower())[1]
    if gltf_axis_fix and ext in (".glb", ".gltf"):
        return gltf_axis_fix_rpy
    return (0.0, 0.0, 0.0)


# ----------------------------
# Public URDF API
# ----------------------------

def to_urdf(
    root: Asset,
    *,
    robot_name: Optional[str] = None,
    include_collision: bool = True,
    mesh_mode: str = "csg",
    mesh_visual_filenames: Optional[Dict[int, str]] = None,
    mesh_collision_filenames: Optional[Dict[int, str]] = None,
    default_mesh_dir: str = "meshes",
    visual_mesh_format: str = "obj",
    collision_mesh_format: str = "stl",
    flatten_prismatic_origin_rotation: bool = True,
    gltf_axis_fix: bool = True,
    gltf_axis_fix_rpy: Tuple[float, float, float] = (math.pi / 2.0, 0.0, 0.0),
) -> str:
    """Convert a kinematic Asset tree (via attach_joint) to a URDF XML string.

    mesh_mode:
      - "none": export URDF primitives only (CSG will be approximated)
      - "csg": export meshes for links containing CSG; primitives otherwise
      - "all": export meshes for all links
    ``mesh_visual_filenames`` and ``mesh_collision_filenames`` map
    ``id(link_shape)`` to paths referenced by the URDF.
    """
    if robot_name is None:
        robot_name = getattr(root, "label", None) or "robot"
    mesh_mode = mesh_mode.lower()
    visual_mesh_format = visual_mesh_format.lower()
    collision_mesh_format = collision_mesh_format.lower()

    link_names = _assign_unique_link_names(root)
    links, joints = _collect_links_and_joints(root)
    link_semantic_paths = _assign_link_semantic_paths(root)
    joint_names = _assign_unique_joint_names(joints)

    robot = ET.Element("robot", {"name": _sanitize_name(robot_name)})

    material_defined: set = set()

    # --- Frame adjustment to make prismatic joint dragging consistent ---
    # We optionally "push" the joint-origin rotation of prismatic joints into the child link geometry,
    # so the prismatic joint frame stays aligned with the parent frame (common in interactive viewers).
    link_adjust_map: Dict[int, np.ndarray] = {id(root): np.eye(4, dtype=float)}
    joint_origin_out: Dict[Tuple[int, str], np.ndarray] = {}

    # Build adjacency for a deterministic traversal
    children_map: Dict[int, List[Tuple[str, Joint, Asset]]] = {}
    for parent, jname, j, child in joints:
        children_map.setdefault(id(parent), []).append((jname, j, child))

    def _rot_only(R: np.ndarray) -> np.ndarray:
        M = np.eye(4, dtype=float)
        M[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
        return M

    def _tr_only(t: np.ndarray) -> np.ndarray:
        M = np.eye(4, dtype=float)
        M[:3, 3] = np.asarray(t, dtype=float).reshape(3)
        return M

    def dfs_adjust(parent: Asset):
        Ap = link_adjust_map[id(parent)]
        ApR = Ap[:3, :3]
        for jname, j, child in children_map.get(id(parent), []):
            H = _as_mat4(j.origin)
            R = H[:3, :3]
            t = H[:3, 3]

            if flatten_prismatic_origin_rotation and j.joint_type == "prismatic":
                # Output joint origin with translation only (in URDF parent frame)
                t_u = ApR @ t
                H_out = _tr_only(t_u)

                # Push rotation into child link geometry coordinates
                Ac = _rot_only(ApR @ R)
            else:
                # Keep full origin rotation as a joint transform; child link stays in its usual joint frame
                # expressed in the URDF parent frame.
                Ac = Ap
                H_out = Ap @ H @ np.linalg.inv(Ac)

            link_adjust_map[id(child)] = Ac
            joint_origin_out[(id(parent), jname)] = H_out
            dfs_adjust(child)

    dfs_adjust(root)


    for link in links:
        lname = link_names[id(link)]
        semantic_path = link_semantic_paths[id(link)]
        link_elem = ET.SubElement(robot, "link", {"name": lname})

        link_adjust = link_adjust_map.get(id(link), np.eye(4, dtype=float))
        wants_mesh = False
        if mesh_mode == "all":
            wants_mesh = True
        elif mesh_mode == "csg":
            wants_mesh = _shape_has_boolean(link)
        elif mesh_mode == "none":
            wants_mesh = False
        else:
            raise ValueError(f"Unknown mesh_mode: {mesh_mode}")

        if wants_mesh:
            rgba = link.get_link_rgba() if hasattr(link, "get_link_rgba") else (0.7, 0.7, 0.7, 1.0)
            mat_name = f"mat_{lname}"

            vis_map = mesh_visual_filenames or {}
            vis_file = vis_map.get(id(link), f"{default_mesh_dir}/{lname}.{visual_mesh_format}")
            vis_rpy = _mesh_origin_correction_rpy(vis_file, gltf_axis_fix=gltf_axis_fix, gltf_axis_fix_rpy=gltf_axis_fix_rpy)
            _add_mesh_visual(
                robot,
                material_defined,
                link_elem,
                vis_file,
                rgba,
                mat_name,
                origin_rpy=vis_rpy,
                semantic_path=semantic_path,
            )

            if include_collision:
                col_map = mesh_collision_filenames or {}
                col_file = col_map.get(id(link), f"{default_mesh_dir}/{lname}.{collision_mesh_format}")
                col_rpy = _mesh_origin_correction_rpy(col_file, gltf_axis_fix=gltf_axis_fix, gltf_axis_fix_rpy=gltf_axis_fix_rpy)
                _add_mesh_collision(
                    link_elem,
                    col_file,
                    origin_rpy=col_rpy,
                    semantic_path=semantic_path,
                )

        else:
            for prim in _iter_primitives_for_urdf(link, semantic_path):
                _add_visual_or_collision(robot, material_defined, link_elem, prim, "visual", link_adjust=link_adjust)
                if include_collision:
                    _add_visual_or_collision(robot, material_defined, link_elem, prim, "collision", link_adjust=link_adjust)

    for parent, jname, j, child in joints:
        joint_limit = _normalize_joint_limit(j.joint_type, j.limit)
        jtype = "fixed" if j.joint_type == "fixed" else (
            "continuous"
            if j.joint_type == "revolute" and joint_limit is None
            else ("revolute" if j.joint_type == "revolute" else "prismatic")
        )
        je = ET.SubElement(
            robot,
            "joint",
            {"name": joint_names[(id(parent), jname, id(child))], "type": jtype},
        )
        ET.SubElement(je, "parent", {"link": link_names[id(parent)]})
        ET.SubElement(je, "child", {"link": link_names[id(child)]})

        origin_mat = joint_origin_out.get((id(parent), jname), _as_mat4(j.origin))
        xyz, rpy, _scale = _decompose_mat4(origin_mat)
        origin = ET.SubElement(je, "origin")
        _add_origin(origin, xyz, rpy)

        if j.joint_type in ("revolute", "prismatic"):
            # The DSL always interprets Joint.axis in the PARENT link frame.
            # URDF expects the axis expressed in the JOINT frame.
            # Convert: axis_joint = R_out^T @ (ApR @ axis_parent_dsl)
            axis_in = np.asarray(j.axis, dtype=float).reshape(3)
            ApR = link_adjust_map[id(parent)][:3, :3]
            R_out = np.asarray(origin_mat[:3, :3], dtype=float).reshape(3, 3)
            axis = R_out.T @ (ApR @ axis_in)
            n = float(np.linalg.norm(axis))
            if n < 1e-12:
                axis = np.array([0.0, 0.0, 1.0], dtype=float)
                n = 1.0
            axis = axis / n
            axis = np.where(np.abs(axis) < 1e-12, 0.0, axis)
            ET.SubElement(je, "axis", {"xyz": f"{axis[0]:.9g} {axis[1]:.9g} {axis[2]:.9g}"})

        if j.joint_type in ("revolute", "prismatic") and joint_limit is not None:
            lo, hi = joint_limit
            attrib = {"lower": f"{lo:.9g}", "upper": f"{hi:.9g}"}
            if j.effort is not None:
                attrib["effort"] = f"{float(j.effort):.9g}"
            if j.velocity is not None:
                attrib["velocity"] = f"{float(j.velocity):.9g}"
            ET.SubElement(je, "limit", attrib)
        elif j.joint_type == "revolute" and joint_limit is None:
            # URDF continuous joints have no lower/upper, but effort/velocity may be provided.
            attrib2 = {}
            if j.effort is not None:
                attrib2["effort"] = f"{float(j.effort):.9g}"
            if j.velocity is not None:
                attrib2["velocity"] = f"{float(j.velocity):.9g}"
            if attrib2:
                ET.SubElement(je, "limit", attrib2)

    return _pretty_xml(robot)


def export_urdf(
    root: Asset,
    filepath: str | os.PathLike[str],
    *,
    robot_name: Optional[str] = None,
    include_collision: bool = True,
    mesh_mode: str = "csg",
    mesh_backend: str = "blender",
    mesh_dir: str = "meshes",
    visual_mesh_format: str = "glb",
    collision_mesh_format: str = "stl",
    flatten_prismatic_origin_rotation: bool = True,
    gltf_axis_fix: bool = True,
    gltf_axis_fix_rpy: Tuple[float, float, float] = (math.pi / 2.0, 0.0, 0.0),
    mesh_pitch: float = 0.01,
    max_voxels: int = 128**3,
    mesh_cache: bool = True,
) -> str:
    """Write URDF to `filepath`, exporting meshes alongside it when requested.

    mesh_mode:
      - "none": export URDF primitives only (CSG is approximated)
      - "csg": export a mesh for any *link* that contains CSG (boolean) geometry
      - "all": export meshes for all links

    mesh_backend:
      - "voxel": voxel sampling + marching cubes (robust, pure Python, but approximate)
      - "blender": if running inside Blender (bpy available), export exact CSG meshes

    Meshes are generated using voxel sampling + marching cubes by default for robustness.
    For OBJ visual meshes, a minimal MTL is written so colors show up in many viewers.
    """
    filepath = os.fspath(filepath)
    out_dir = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(out_dir, exist_ok=True)

    mesh_mode = mesh_mode.lower()
    visual_mesh_format = visual_mesh_format.lower()
    collision_mesh_format = collision_mesh_format.lower()

    # Note: URDF itself doesn't standardize GLB, but some simulators/viewers accept it.
    if visual_mesh_format not in ("stl", "obj", "ply", "dae", "glb"):
        raise ValueError("visual_mesh_format must be one of: stl, obj, ply, dae, glb")
    if collision_mesh_format not in ("stl", "obj", "ply"):
        raise ValueError("collision_mesh_format must be one of: stl, obj, ply")

    # Prepare mesh directory
    mesh_out_dir = mesh_dir
    if not os.path.isabs(mesh_out_dir):
        mesh_out_dir = os.path.join(out_dir, mesh_out_dir)
    if mesh_mode in ("csg", "all"):
        os.makedirs(mesh_out_dir, exist_ok=True)

    link_names = _assign_unique_link_names(root)
    links, _joints = _collect_links_and_joints(root)

    mesh_visual: Dict[int, str] = {}
    mesh_collision: Dict[int, str] = {}

    mesh_backend = mesh_backend.lower()

    if mesh_mode in ("csg", "all"):
        if mesh_backend == "voxel":
            _require_meshing_deps()
        elif mesh_backend == "blender":
            # Blender backend is only available when running inside Blender's Python.
            try:  # pragma: no cover
                from .export_glb import export_glb  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "mesh_backend='blender' requires Blender's Python environment (bpy). "
                    "Run this script via Blender, or use mesh_backend='voxel'."
                ) from e
            if export_glb is None:  # pragma: no cover
                raise RuntimeError(
                    "mesh_backend='blender' requires Blender's Python environment (bpy)."
                )
            if visual_mesh_format != "glb":
                raise ValueError("mesh_backend='blender' currently supports visual_mesh_format='glb' only")
        else:
            raise ValueError("mesh_backend must be one of: voxel, blender")

        for link in links:
            wants = (mesh_mode == "all") or _shape_has_boolean(link)
            if not wants:
                continue

            lname = link_names[id(link)]
            rgba = link.get_link_rgba() if hasattr(link, "get_link_rgba") else (0.7, 0.7, 0.7, 1.0)

            # Visual mesh
            vis_path = os.path.join(mesh_out_dir, f"{lname}.{visual_mesh_format}")
            if not (mesh_cache and os.path.exists(vis_path)):
                if mesh_backend == "voxel":
                    mesh = _mesh_from_shape_csg(link, pitch=mesh_pitch, max_voxels=max_voxels)
                    if getattr(mesh, "vertices", None) is None or mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
                        continue
                    _export_mesh_with_optional_mtl(mesh, vis_path, rgba)
                else:
                    # Blender exact boolean backend (visual only).
                    from .export_glb import export_glb  # type: ignore
                    export_glb(link, vis_path, clear_scene=True, include_joint_children=False)

            vis_rel = os.path.relpath(vis_path, out_dir).replace(os.sep, "/")
            mesh_visual[id(link)] = vis_rel

            # Collision mesh
            if include_collision:
                if mesh_backend == "blender":
                    # Fast path: reuse the same GLB for collision.
                    # (Many URDF consumers will either ignore collision meshes, or accept GLB here.)
                    mesh_collision[id(link)] = vis_rel
                else:
                    col_path = os.path.join(mesh_out_dir, f"{lname}_col.{collision_mesh_format}")
                    if not (mesh_cache and os.path.exists(col_path)):
                        mesh = _mesh_from_shape_csg(link, pitch=mesh_pitch, max_voxels=max_voxels)
                        if getattr(mesh, "vertices", None) is None or mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
                            continue
                        if collision_mesh_format == "obj":
                            _export_mesh_with_optional_mtl(mesh, col_path, (0.6, 0.6, 0.6, 1.0))
                        else:
                            mesh.export(col_path)
                    if os.path.exists(col_path):
                        col_rel = os.path.relpath(col_path, out_dir).replace(os.sep, "/")
                        mesh_collision[id(link)] = col_rel

    urdf_text = to_urdf(
        root,
        robot_name=robot_name,
        include_collision=include_collision,
        mesh_mode=mesh_mode,
        mesh_visual_filenames=mesh_visual if mesh_visual else None,
        mesh_collision_filenames=mesh_collision if mesh_collision else None,
        default_mesh_dir=os.path.relpath(mesh_out_dir, out_dir).replace(os.sep, "/")
        if mesh_mode in ("csg", "all") else "meshes",
        visual_mesh_format=visual_mesh_format,
        collision_mesh_format=collision_mesh_format,
        flatten_prismatic_origin_rotation=flatten_prismatic_origin_rotation,
        gltf_axis_fix=gltf_axis_fix,
        gltf_axis_fix_rpy=gltf_axis_fix_rpy,
    )
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(urdf_text)
    joint_names = _assign_unique_joint_names(_joints)
    joint_states = {
        joint_names[(id(parent), joint_name, id(child))]: float(joint.initial)
        for parent, joint_name, joint, child in _joints
        if joint.joint_type in ("revolute", "prismatic")
    }
    state_path = os.path.splitext(filepath)[0] + ".joint_states.json"
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "joints": joint_states}, f, indent=2)
    return filepath


__all__ = ["to_urdf", "export_urdf"]
