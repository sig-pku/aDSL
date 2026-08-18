# aDSL modeling reference

Import the public API from `adsl.core`:

```python
from adsl.core import *
```

The world coordinate convention is `+x` right, `+y` inward, and `+z` up.
Lengths use caller-defined scene units. Rotation angles passed to
`rotation_matrix` and the axis/angle form of `rotate_shape` are degrees. Joint
positions use radians for revolute joints and scene units for prismatic joints.
All positive axis-angle rotations follow the right-hand rule. For a quick sign
check, a positive 90-degree rotation maps `+y` toward `+z` around `+x`, `+z`
toward `+x` around `+y`, and `+x` toward `+y` around `+z`. Negating the axis
reverses the positive rotation direction.

Transforms and layout functions return new Asset trees. `attach_part`, joint
methods, and appearance setters modify the receiving Asset and return an Asset
for fluent construction.

## Assets and hierarchy

```python
Asset(label: str = "Asset")
Asset.attach_part(name: str, shape: Asset) -> Asset
Asset.detach_part(name: str) -> None
Asset.copy() -> Asset
concat_shapes(shapes: Iterable[Asset], *, label: str | None = None) -> Asset
```

`attach_part` records a named modeling subpart and preserves the hierarchy.
Names must be unique among the direct children of one parent.

`concat_shapes` returns a container whose children are named `part_0`,
`part_1`, and so on. Use explicit `attach_part` calls when semantic names matter.

Example:

```python
desk = Asset("desk")
desk.attach_part("desktop", Cube((1.4, 0.7, 0.06), center=(0, 0, 0.73)))
legs = Asset("legs")
for index, position in enumerate(((-0.6, -0.25), (-0.6, 0.25), (0.6, -0.25), (0.6, 0.25)), 1):
    legs.attach_part(
        f"leg_{index}",
        Cube((0.06, 0.06, 0.7), center=(position[0], position[1], 0.35)),
    )
desk.attach_part("legs", legs)
scene = desk
```

## Primitives

The capitalized constructors and lowercase constructors are equivalent public
forms. A scalar cube scale creates equal x/y/z dimensions.

```python
Cube(scale: float | Sequence[float], center=(0, 0, 0), color=(1, 1, 1), alpha=None) -> Asset
Sphere(radius: float, center=(0, 0, 0), color=(1, 1, 1), alpha=None) -> Asset
Cylinder(
    radius: float,
    p0: Sequence[float] | None = None,
    p1: Sequence[float] | None = None,
    *,
    height: float | None = None,
    center=(0, 0, 0),
    axis="z",
    color=(1, 1, 1),
    alpha=None,
) -> Asset
cube(...), sphere(...), cylinder(...)
```

A cylinder requires either both endpoints `p0`/`p1`, or `height` with a
cardinal `axis`. Endpoints are the centers of the circular end caps. A cylinder
is symmetric along its length, so negating `axis` only swaps which end is
considered `p0` versus `p1`; it does not change the visible geometry.

## Boolean operations

```python
boolean_union(*shapes: Asset) -> Asset
boolean_intersection(*shapes: Asset) -> Asset
boolean_difference(base: Asset, *subtractors: Asset) -> Asset
boolean_xor(*shapes: Asset) -> Asset
```

Boolean results retain their operand hierarchy for inspection.

## Transformations

```python
translation_matrix(offset: Sequence[float]) -> T
scaling_matrix(scale: float | Sequence[float], center=(0, 0, 0)) -> T
rotation_matrix(axis: str | Sequence[float], angle: float, center=(0, 0, 0)) -> T
transform_shape(shape: Asset, matrix: T) -> Asset
translate_shape(shape: Asset, offset: Sequence[float]) -> Asset
scale_shape(shape: Asset, scale: float | Sequence[float], center=None) -> Asset
rotate_shape(shape: Asset, axis, angle: float, center=None) -> Asset
rotate_shape(shape: Asset, *, euler: Sequence[float], center=None) -> Asset
```

Signed cardinal axes are `+x`, `-x`, `+y`, `-y`, `+z`, and `-z`; bare axis
letters mean their positive direction. Axis-angle rotation follows the
right-hand rule described above. The `euler=(x, y, z)` form accepts degrees and
applies the x rotation first, then y, then z (combined matrix `Rz @ Ry @ Rx`).
When a transform center is omitted, `scale_shape` and `rotate_shape` use the
current AABB center.

## Bounds and anchors

```python
shape_aabb(shape: Asset) -> tuple[P, P]
shape_min(shape: Asset) -> P
shape_max(shape: Asset) -> P
shape_size(shape: Asset) -> P
shape_center(shape: Asset) -> P
shape_anchor(shape: Asset, anchor: str = "center") -> P
shape_support(shape: Asset, direction: str | Sequence[float]) -> P
shape_bounds_along(shape: Asset, direction) -> tuple[float, float]
shape_extent_along(shape: Asset, direction) -> float
```

The first six functions use world-space axis-aligned bounds. Anchor tokens map
to AABB sides:

- `left` / `right`: minimum / maximum x
- `front` / `back`: minimum / maximum y
- `bottom` / `top`: minimum / maximum z

Unspecified axes use the center. For example, `top` is the center of the top
face and `left_front_top` is a corner. Support and directional-bound functions
should be used for arbitrary directions and rotated contact reasoning.

## Alignment and placement

```python
align_centers(shape: Asset, target: Asset, axes=("x", "y", "z")) -> Asset
align_anchors(
    shape: Asset,
    target: Asset | Sequence[float],
    anchor: str = "center",
    target_anchor: str | None = None,
    offset=(0, 0, 0),
) -> Asset
place_on_axis(shape: Asset, target: Asset | float, axis="+z", gap=0.0) -> Asset
offset_from(
    shape: Asset,
    reference: Asset | Sequence[float | None] | None,
    offset: Sequence[float | None],
) -> Asset
```

`align_anchors` aligns one source AABB anchor with an Asset anchor or an exact
world point. `target_anchor` is valid only for an Asset target and defaults to
the same name as `anchor`.

`place_on_axis` uses the axis sign to choose direction. For `+z`, the source
bottom is placed above the target top. For `-z`, the source top is placed below
the target bottom. A numeric target is the boundary coordinate. `gap` must be
non-negative.

`offset_from` positions selected center coordinates relative to an Asset center,
a point, or the origin. A `None` coordinate leaves that source coordinate
unchanged.

## Repeated layouts

```python
distribute_along_axis(shapes, axis="+x", spacing=1.0) -> Asset
stack_shapes(shapes, axis="+z", gap=0.0) -> Asset
grid_shapes(
    shapes,
    rows=None,
    cols=None,
    spacing=(1.0, 1.0),
    plane="xy",
    center=(0, 0, 0),
    order="row-major",
) -> Asset
radial_shapes(
    shapes,
    radius,
    axis="+z",
    center=(0, 0, 0),
    start_angle=0.0,
    sweep=360.0,
    *,
    rotate_with_layout=False,
    rotation_offset=0.0,
) -> Asset
```

For `distribute_along_axis` and `stack_shapes`, the **first input shape is the
fixed base** and remains at its original center. Later shapes are placed in
sequence along the signed axis. Distribution uses center-to-center `spacing`;
stacking uses `gap` between neighboring AABB boundaries. Both distances must be
non-negative.

`grid_shapes` centers the complete grid at `center`. `spacing` is the
center-to-center pitch in the two axes named by `plane`. Columns increase along
the positive first plane axis; rows increase along the negative second plane
axis. For `plane="xy"`, columns run along `+x`; the first row is on the `+y`
side, and later rows advance toward `-y`.

`radial_shapes` uses evenly spaced slots without duplicating the first slot for
a 360-degree sweep. Partial arcs include both endpoints. With
`rotate_with_layout=False`, input orientations are unchanged. With
`rotate_with_layout=True`, every shape is first rotated around its own center by
its slot angle plus `rotation_offset`, then translated. The input orientation at
zero degrees is the pattern reference. Positive slot angles follow the
right-hand rule around the signed `axis`; negating `axis` reverses the sweep.
The zero-angle radial direction is `+x` for a z axis, `+y` for an x axis, and
`+z` for a y axis. For example, around `axis="+z"`, zero degrees lies on `+x`
and positive angles sweep toward `+y`.

Bicycle-spoke example:

```python
spokes = radial_shapes(
    [Cube((0.45, 0.015, 0.015)) for _ in range(12)],
    radius=0.225,
    axis="+z",
    rotate_with_layout=True,
)
```

## Articulation

```python
Asset.revolute(
    child: Asset | str,
    *, axis=(0, 0, 1), limit=(-pi, pi), origin=(0, 0, 0),
    initial=0.0, joint_name=None, effort=None, velocity=None,
) -> Asset
Asset.prismatic(
    child: Asset | str,
    *, axis=(0, 0, 1), limit=(0, 1), origin=None, initial=0.0,
    towards=None, joint_name=None, effort=None, velocity=None,
) -> Asset
Asset.fixed(child: Asset | str, *, joint_name=None, origin=None) -> Asset
Asset.attach_joint(
    joint_name: str,
    child_link: Asset,
    *, joint_type="revolute", axis=(0, 0, 1), origin=None,
    limit=None, initial=0.0, effort=None, velocity=None,
) -> Asset
```

The ergonomic `revolute`, `prismatic`, and `fixed` methods accept the name of an
existing direct part or an Asset. When an Asset is passed directly, `joint_name`
is required. Place the child geometry in the parent's zero-pose coordinates
before creating the joint. The method rebases the child by `inverse(origin)`
into the joint frame, so `origin` is the hinge/pivot/slide frame expressed in
the parent link. A point origin supplies translation only; a 4x4 origin may also
rotate the joint frame.

For the ergonomic methods, `axis` is expressed in the joint/child frame at the
zero pose. Positive revolute motion follows the right-hand rule around that
axis; positive prismatic motion translates along the axis. Negating the axis
reverses the meaning of positive joint values. Choose `axis`, signed `limit`,
and `initial` together so the initial and endpoint poses move the part in the
intended physical direction. The pivot location alone does not determine which
way a lid or door opens.

`attach_joint` is the low-level exception: its `axis` is expressed directly in
the **parent-link frame**, not the child/joint frame. Prefer the ergonomic
methods unless that distinction is intentional.

For a freely rotating revolute joint, use `limit=None` or
`limit=(-float("inf"), float("inf"))`.

`initial` must be finite and within a finite `limit`. `towards` on a prismatic joint may
name a direct sibling, provide an Asset, provide a point, or select the parent
origin; it flips the axis when necessary and raises if the target cannot be
resolved. It chooses the positive slide direction only; limits and initial
values remain measured along that resolved direction.

Before finalizing an articulated object, reason about both the zero pose and at
least one nonzero pose. Example: if a closed laptop lid extends from an x-axis
hinge toward `-y`, then `axis="-x"` with positive limits rotates the lid toward
`+z`:

```python
laptop.attach_part("lid", lid_in_closed_parent_coordinates)
laptop.revolute(
    "lid", axis="-x", origin=hinge_point,
    limit=(0.0, 2.18), initial=1.83,
)
```

Using `axis="+x"` for the same geometry would require equivalent negative
limits and an initial value such as `-1.83`. If the lid extends toward `+y`
instead, reverse these signs.
