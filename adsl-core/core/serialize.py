from __future__ import annotations

from typing import Any


def asset_to_dict(asset: Any) -> dict[str, Any]:
    return {
        "label": asset.label,
        "primitives": asset._primitives,
        "children": {key: child.to_dict() for key, child in asset._children.items()},
        "joints": {
            key: {
                "spec": asset._joints[key].to_dict(),
                "child": asset._joint_children[key].to_dict(),
            }
            for key in asset._joint_children.keys()
        },
    }
