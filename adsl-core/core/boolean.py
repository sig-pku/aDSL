from __future__ import annotations
from typing import Optional
from .asset import Asset, ensure_shape

def _boolean_primitive(
    mode: str,
    base: Optional[Asset],
    others: list[Asset],
    *,
    label: str
) -> Asset:
    container = Asset(label=label)
    params = {"mode": mode}
    if base is not None:
        container.attach_part("base", base)
    for i, s in enumerate(others):
        container.attach_part(f"op_{i}", s)
    container.add_primitive({
        "type": "boolean",
        "params": params,
        "xform": None,
        "color": (1.0, 1.0, 1.0),
        "alpha": None
    })
    return container

def boolean_union(
    *shapes: Asset
) -> Asset:
    present = [ensure_shape(s, label=f"Union_{i}") for i, s in enumerate(shapes)]
    if not present:
        return Asset(label="BooleanUnion")
    if len(present) == 1:
        return present[0]
    return _boolean_primitive("UNION", None, present, label="BooleanUnion")

def boolean_intersection(
    *shapes: Asset
) -> Asset:
    present = [ensure_shape(s, label=f"Intersection_{i}") for i, s in enumerate(shapes)]
    if not present:
        return Asset(label="BooleanIntersection")
    if len(present) == 1:
        return present[0]
    return _boolean_primitive("INTERSECT", None, present, label="BooleanIntersection")

def boolean_difference(
    base: Asset,
    *subtractors: Asset
) -> Asset:
    base_shape = ensure_shape(base, label="Difference_Base")
    if base_shape is None:
        return Asset(label="BooleanDifference")
    subs = [ensure_shape(s, label=f"Difference_Sub_{i}") for i, s in enumerate(subtractors)]
    if not subs:
        return base_shape
    return _boolean_primitive("DIFFERENCE", base_shape, subs, label="BooleanDifference")

def boolean_xor(
    a: Asset,
    b: Asset
) -> Asset:
    A = ensure_shape(a, label="XOR_A")
    B = ensure_shape(b, label="XOR_B")
    present = [p for p in (A, B) if p is not None]
    if not present:
        return Asset(label="BooleanXor")
    if len(present) == 1:
        return present[0]
    return boolean_union(
        boolean_difference(A, B), 
        boolean_difference(B, A)
    )
    
__all__ = [
    "boolean_union", 
    "boolean_intersection", 
    "boolean_difference", 
    "boolean_xor"
]
