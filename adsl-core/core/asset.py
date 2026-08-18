from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import (
    Any,
    Dict,
    Optional,
    Iterable,
    Iterator,
    List,
    Tuple,
    Mapping,
    MutableMapping,
)

import math
import numpy as np

from .joints import Joint, _normalize_joint_limit
from .math_utils import T, _as_color, _as_mat4, _as_vec3, _parse_axis, origin_to_mat4
from .serialize import asset_to_dict


def remap_copied_value(value: Any, mapping: dict[int, Any]) -> Any:
    if isinstance(value, Asset):
        return mapping.get(id(value), value.copy())
    if isinstance(value, dict):
        return {key: remap_copied_value(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [remap_copied_value(item, mapping) for item in value]
    if isinstance(value, tuple):
        return tuple(remap_copied_value(item, mapping) for item in value)
    if isinstance(value, set):
        return {remap_copied_value(item, mapping) for item in value}
    return value


##############
# Base Asset #
##############


@dataclass
class Asset:
    # Base properties
    label: str = "Asset"
    _link_color: Optional[Tuple[float, float, float]] = field(default=None, repr=False)
    _link_alpha: Optional[float] = field(default=None, repr=False)
    _primitives: List[Dict[str, Any]] = field(default_factory=list)
    _children: Dict[str, "Asset"] = field(default_factory=dict)
    _parent: Optional["Asset"] = field(default=None, init=False, repr=False)
    # Kinematic tree (URDF-style)
    _joints: Dict[str, Joint] = field(default_factory=dict)              # joint_name -> Joint spec
    _joint_children: Dict[str, "Asset"] = field(default_factory=dict)    # joint_name -> child link Asset
    _joint_parent: Optional["Asset"] = field(default=None, init=False, repr=False)
    _parent_joint: Optional[Joint] = field(default=None, init=False, repr=False)

    # -----------------
    # Child management
    # -----------------

    def attach_part(
        self,
        name: str,
        shape: "Asset | Iterable[MutableMapping[str, Any]]",
    ) -> "Asset":
        child = ensure_shape(shape, label=name)
        if child._parent is not None and child._parent is not self:
            child = child.copy()
        previous = self._children.get(name)
        if previous is not None and previous is not child:
            self.detach_part(name)
        self._children[name] = child
        object.__setattr__(child, "_parent", self)
        object.__setattr__(self, name, child)
        return child

    def detach_part(self, name: str) -> None:
        child = self._children.pop(name, None)
        if child is not None:
            object.__setattr__(child, "_parent", None)
        if hasattr(self, name):
            object.__delattr__(self, name)

    # ---------------------
    # Kinematic management
    # ---------------------

    def _take_child_link(self, child: "Asset | str", *, copy_if_shared: bool = True) -> Tuple[Optional[str], "Asset"]:
        """Resolve a child link by name and detach it from parts if needed.

        Common pattern this enables:
            self.attach_part("arm", Arm())
            self.revolute("arm", axis=(0,0,1), origin=(...))

        Returns (child_name_if_str_else_None, child_shape).
        """
        if isinstance(child, str):
            name = child
            # Prefer geometry parts: attach_part then convert to joint
            if name in self._children:
                ch = self._children[name]
                self.detach_part(name)
                return name, ch

            # Idempotent: already a joint child
            if name in self._joint_children:
                return name, self._joint_children[name]

            raise KeyError(f"No child Asset named '{name}' found on this Asset")

        ch = child
        if copy_if_shared and getattr(ch, "_joint_parent", None) is not None and ch._joint_parent is not self:
            ch = ch.copy()
        return None, ch

    # ---------------------------
    # Ergonomic joint wrappers
    # ---------------------------


    def revolute(
        self,
        child: "Asset | str",
        *,
        axis: Any = (0.0, 0.0, 1.0),
        limit: Optional[Tuple[float, float]] = (-math.pi, math.pi),
        origin: Any = (0.0, 0.0, 0.0),
        initial: float = 0.0,
        joint_name: Optional[str] = None,
        effort: Optional[float] = None,
        velocity: Optional[float] = None,
    ) -> "Asset":
        """Attach a revolute joint with a lightweight signature.

        Required by your spec:
          - child name
          - revolve axis
          - limits
          - revolve origin (3,) translation

        Notes:
        - For ease-of-use, `axis` is interpreted in the **joint/child frame** at
          zero pose (i.e., after applying `origin`). This matches how URDF
          defines joint axes.
        - Internally we store the axis in the parent frame by rotating it with
          `origin[:3,:3]`.
        - `origin` may be a (3,) translation or full 4x4 matrix.
        """
        child_name, child_link = self._take_child_link(child)
        jname = joint_name or child_name
        if jname is None:
            raise ValueError("joint_name is required when child is passed as an Asset")

        origin_mat = origin_to_mat4(origin)
        axis_joint = _parse_axis(axis)
        axis_parent = origin_mat[:3, :3] @ axis_joint

        return self.attach_joint(
            jname,
            child_link,
            joint_type="revolute",
            axis=axis_parent,
            origin=origin_mat,
            limit=limit,
            initial=initial,
            effort=effort,
            velocity=velocity,
        )

    def prismatic(
        self,
        child: "Asset | str",
        *,
        axis: Any = (0.0, 0.0, 1.0),
        limit: Optional[Tuple[float, float]] = (0.0, 1.0),
        joint_name: Optional[str] = None,
        origin: Any = None,
        initial: float = 0.0,
        towards: Any = None,
        effort: Optional[float] = None,
        velocity: Optional[float] = None,
    ) -> "Asset":
        """Attach a prismatic joint with a lightweight signature.

        Required by your spec:
          - child name
          - motion axis
          - limits

        Notes:
        - For ease-of-use, `axis` is interpreted in the **joint/child frame** at
          zero pose (i.e., after applying `origin`).
          Internally we store the axis in the parent frame by rotating it with
          `origin[:3,:3]`.
        - `origin` is optional; if provided it may be (3,) or 4x4.
        - `towards` is optional convenience: if provided, the axis sign is chosen
          so that *positive* motion goes towards the target.

          `towards` can be:
            - (3,) point in parent frame
            - "origin" / "parent_origin"
            - name of a sibling part/joint-child on this parent (e.g. "table_fixed")
            - an Asset object (direct joint child preferred)
        """
        child_name, child_link = self._take_child_link(child)
        jname = joint_name or child_name
        if jname is None:
            raise ValueError("joint_name is required when child is passed as an Asset")

        origin_mat = origin_to_mat4(origin)
        axis_joint = _parse_axis(axis)
        axis_vec = origin_mat[:3, :3] @ axis_joint

        if towards is not None:
            from .bounds import shape_center

            def _target_point_for_shape(target_shape: "Asset") -> np.ndarray:
                center = np.asarray(shape_center(target_shape), dtype=float).reshape(3)
                if not np.all(np.isfinite(center)):
                    raise ValueError("towards target must have finite geometry bounds")
                if (
                    target_shape._joint_parent is self
                    and target_shape._parent_joint is not None
                ):
                    transform = _as_mat4(target_shape._parent_joint.origin)
                    return transform[:3, :3] @ center + transform[:3, 3]
                return center

            if isinstance(towards, str):
                key = towards.strip()
                if key.lower() in ("origin", "parent", "parent_origin"):
                    target_point = np.zeros(3, dtype=float)
                elif key in self._children:
                    target_point = _target_point_for_shape(self._children[key])
                elif key in self._joint_children:
                    target_point = _target_point_for_shape(self._joint_children[key])
                else:
                    raise KeyError(f"No target Asset named '{towards}' on this parent")
            elif isinstance(towards, Asset):
                target_point = _target_point_for_shape(towards)
            else:
                target_point = _as_vec3(towards)

            pivot = np.asarray(origin_mat[:3, 3], dtype=float).reshape(3)
            if float(np.dot(axis_vec, np.asarray(target_point) - pivot)) < 0.0:
                axis_vec = -axis_vec

        return self.attach_joint(
            jname,
            child_link,
            joint_type="prismatic",
            axis=axis_vec,
            origin=origin_mat,
            limit=limit,
            initial=initial,
            effort=effort,
            velocity=velocity,
        )

    def fixed(
        self,
        child: "Asset | str",
        *,
        joint_name: Optional[str] = None,
        origin: Any = None,
    ) -> "Asset":
        """Attach a fixed joint with a lightweight signature."""
        child_name, child_link = self._take_child_link(child)
        jname = joint_name or child_name
        if jname is None:
            raise ValueError("joint_name is required when child is passed as an Asset")
        return self.attach_joint(
            jname,
            child_link,
            joint_type="fixed",
            origin=origin_to_mat4(origin),
            limit=None,
        )

    # ----------------------
    # Low-level joint API
    # ----------------------

    def attach_joint(
        self,
        joint_name: str,
        child_link: "Asset",
        *,
        joint_type: str = "revolute",
        axis: Iterable[float] = (0.0, 0.0, 1.0),
        origin: T | None = None,
        limit: Optional[Tuple[float, float]] = None,
        initial: float = 0.0,
        effort: Optional[float] = None,
        velocity: Optional[float] = None,
    ) -> "Asset":
        """Attach `child_link` as a kinematic child connected by a joint.

        This does NOT affect geometric children attached via `attach_part`.

        Axis semantics:
        - `axis` is always treated as **parent-link frame**.
        """
        if joint_type not in ("revolute", "prismatic", "fixed"):
            raise ValueError("joint_type must be one of: 'revolute', 'prismatic', 'fixed'")

        initial_value = float(initial)
        if not math.isfinite(initial_value):
            raise ValueError("joint initial state must be finite")
        if joint_type == "fixed" and abs(initial_value) > 1e-12:
            raise ValueError("fixed joints require initial=0")
        normalized_limit = _normalize_joint_limit(joint_type, limit)
        if joint_type in ("revolute", "prismatic") and normalized_limit is not None:
            lower, upper = normalized_limit
            if initial_value < lower - 1e-12 or initial_value > upper + 1e-12:
                raise ValueError("joint initial state must stay within its limits")

        if not isinstance(child_link, Asset):
            child_link = ensure_shape(child_link, label=getattr(child_link, "label", None) or joint_name)

        # Interpret `origin` as the **joint/pivot frame** in the *parent* link frame.
        # To avoid "double translating" geometry (a very common URDF pitfall), we
        # rebase the child's geometry into its own link frame by applying the
        # inverse origin to the entire child subtree.
        origin_mat = _as_mat4(origin)
        if not np.allclose(origin_mat, np.eye(4, dtype=float), atol=1e-12):
            # Lazy import to avoid circular dependency at module import time.
            from .transforms import _transform_in_place

            _transform_in_place(child_link, np.linalg.inv(origin_mat))

        # Prevent a single Asset instance from being shared across multiple kinematic parents.
        if getattr(child_link, "_joint_parent", None) is not None and child_link._joint_parent is not self:
            child_link = child_link.copy()

        prev = self._joint_children.get(joint_name)
        if prev is not None and prev is not child_link:
            self.detach_joint(joint_name)

        j = Joint(
            name=joint_name,
            joint_type=joint_type,
            axis=_parse_axis(axis),
            origin=origin_mat,
            limit=normalized_limit,
            initial=initial_value,
            effort=effort,
            velocity=velocity,
        )

        self._joints[joint_name] = j
        self._joint_children[joint_name] = child_link
        object.__setattr__(child_link, "_joint_parent", self)
        object.__setattr__(child_link, "_parent_joint", j)
        object.__setattr__(self, joint_name, child_link)
        return child_link

    def detach_joint(self, joint_name: str) -> None:
        ch = self._joint_children.pop(joint_name, None)
        self._joints.pop(joint_name, None)
        if ch is not None:
            object.__setattr__(ch, "_joint_parent", None)
            object.__setattr__(ch, "_parent_joint", None)
        if hasattr(self, joint_name):
            object.__delattr__(self, joint_name)

    @property
    def joint_children(self) -> Dict[str, "Asset"]:
        return dict(self._joint_children)

    @property
    def joints(self) -> Dict[str, Joint]:
        return dict(self._joints)

    @property
    def _parts(self) -> Dict[str, "Asset"]:
        return dict(self._children)

    # -------------------------------
    # Primitive management & iteration
    # -------------------------------

    def add_primitive(self, primitive: Mapping[str, Any]) -> None:
        self._primitives.append(dict(primitive))

    def iter_local_primitives(self) -> Iterator[Dict[str, Any]]:
        for prim in self._primitives:
            yield dict(prim)

    def iter_primitives(self) -> Iterator[Dict[str, Any]]:
        for prim in self.iter_local_primitives():
            yield prim
        for child in self._children.values():
            yield from child.iter_primitives()

    def local_primitives(self) -> List[Dict[str, Any]]:
        return list(self.iter_local_primitives())

    def primitives(self) -> List[Dict[str, Any]]:
        return list(self.iter_primitives())

    #######################
    # Appearance / Colors #
    #######################

    def set_link_color(
        self,
        color: Iterable[float] = (1.0, 1.0, 1.0),
        *,
        alpha: float = 1.0,
    ) -> "Asset":
        """Set a default color for this link when exporting to URDF."""
        r, g, b = _as_color(color)
        a = float(alpha)
        a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
        self._link_color = (r, g, b)
        self._link_alpha = a
        return self

    def get_link_rgba(self) -> Tuple[float, float, float, float]:
        """Get the effective RGBA for this link."""
        if self._link_color is not None:
            a = 1.0 if self._link_alpha is None else float(self._link_alpha)
            return (float(self._link_color[0]), float(self._link_color[1]), float(self._link_color[2]), a)

        # Use the first non-boolean primitive color in this subtree.
        for prim in self.iter_primitives():
            if prim.get("type") == "boolean":
                continue
            c = prim.get("color", (0.7, 0.7, 0.7))
            r, g, b = (float(c[0]), float(c[1]), float(c[2]))
            a = prim.get("alpha", None)
            a = 1.0 if a is None else float(a)
            a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
            return (r, g, b, a)

        return (0.7, 0.7, 0.7, 1.0)

    # -----------------------
    # Copying / serialization
    # -----------------------

    def copy(self) -> "Asset":
        """Deep-copy an Asset tree, preserving the concrete Python class.

        Important: we avoid calling subclass __init__ (which may rebuild geometry),
        but we *do* preserve any convenience attributes that reference parts/joints.
        """
        # Create an instance of the same concrete class without invoking __init__.
        cls = self.__class__
        c = cls.__new__(cls)

        # Initialize base Asset fields (mirrors dataclass defaults)
        object.__setattr__(c, "label", self.label)
        object.__setattr__(c, "_link_color", getattr(self, "_link_color", None))
        object.__setattr__(c, "_link_alpha", getattr(self, "_link_alpha", None))
        object.__setattr__(c, "_primitives", [])
        object.__setattr__(c, "_children", {})
        object.__setattr__(c, "_parent", None)
        object.__setattr__(c, "_joints", {})
        object.__setattr__(c, "_joint_children", {})
        object.__setattr__(c, "_joint_parent", None)
        object.__setattr__(c, "_parent_joint", None)

        # Copy primitives
        for p in self._primitives:
            c.add_primitive(p)

        # Track original->copy mapping so we can remap convenience attributes.
        mapping: Dict[int, Asset] = {id(self): c}

        # Copy geometry children (parts)
        for name, ch in self._children.items():
            c.attach_part(name, ch.copy())
            mapping[id(ch)] = getattr(c, name)

        # Copy kinematic children (joints)
        # IMPORTANT: do NOT call attach_joint here.
        # attach_joint may rebase the child geometry by the joint origin, which is correct
        # when first creating a joint, but incorrect when copying an already-constructed tree.
        for jname, child in self._joint_children.items():
            j = self._joints[jname]
            child_c = child.copy()

            j_c = Joint(
                name=jname,
                joint_type=j.joint_type,
                axis=_as_vec3(j.axis),
                origin=_as_mat4(j.origin),
                limit=None if j.limit is None else (float(j.limit[0]), float(j.limit[1])),
                initial=float(j.initial),
                effort=None if j.effort is None else float(j.effort),
                velocity=None if j.velocity is None else float(j.velocity),
            )

            c._joints[jname] = j_c
            c._joint_children[jname] = child_c
            object.__setattr__(child_c, "_joint_parent", c)
            object.__setattr__(child_c, "_parent_joint", j_c)

            # Mirror attach_joint's convenience attribute behavior
            object.__setattr__(c, jname, child_c)

            mapping[id(child)] = child_c

        # Remap any extra/convenience attributes (e.g. self.seat -> self.seat_assembly).
        asset_base_fields = {item.name for item in fields(Asset)}
        for k, v in getattr(self, "__dict__", {}).items():
            if k in asset_base_fields:
                continue
            # Skip per-part/joint attributes already created by attach_part/attach_joint
            if k in self._children or k in self._joint_children:
                continue
            object.__setattr__(c, k, remap_copied_value(v, mapping))

        return c

    def to_dict(self) -> Dict[str, Any]:
        return asset_to_dict(self)


def ensure_shape(
    shape_like: "Asset | Iterable[MutableMapping[str, Any]] | None",
    *,
    label: Optional[str] = None,
) -> Asset:
    if shape_like is None:
        return Asset(label=label or "Empty")
    if isinstance(shape_like, Asset):
        return shape_like
    s = Asset(label=label or "Wrapped")
    for prim in shape_like:
        s.add_primitive(prim)
    return s


def concat_shapes(shapes: Iterable[Asset], *, label: Optional[str] = None) -> Asset:
    combined = Asset(label=label or "Concatenated")
    for i, shape in enumerate(shapes):
        combined.attach_part(f"part_{i}", shape)
    return combined
