from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..asset import Asset


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _primitive_to_manifest(
    primitive: Mapping[str, Any],
    *,
    include_transforms: bool,
) -> dict[str, Any]:
    payload = dict(primitive)
    if not include_transforms:
        payload.pop("xform", None)
    return _jsonable(payload)


def _joint_to_manifest(joint: Any, *, include_transforms: bool) -> dict[str, Any]:
    payload = joint.to_dict()
    if not include_transforms:
        payload.pop("origin", None)
    return _jsonable(payload)


def asset_to_manifest(
    asset: Asset,
    *,
    include_primitives: bool = True,
    include_transforms: bool = True,
    include_joints: bool = True,
    include_children: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "label": str(getattr(asset, "label", "Asset")),
    }
    if include_primitives:
        payload["primitives"] = [
            _primitive_to_manifest(primitive, include_transforms=include_transforms)
            for primitive in getattr(asset, "_primitives", [])
        ]
    if include_children:
        payload["children"] = {
            str(name): asset_to_manifest(
                child,
                include_primitives=include_primitives,
                include_transforms=include_transforms,
                include_joints=include_joints,
                include_children=include_children,
            )
            for name, child in getattr(asset, "_children", {}).items()
        }
    if include_joints:
        payload["joints"] = {
            str(name): {
                "spec": _joint_to_manifest(joint, include_transforms=include_transforms),
                "child": (
                    asset_to_manifest(
                        getattr(asset, "_joint_children", {})[name],
                        include_primitives=include_primitives,
                        include_transforms=include_transforms,
                        include_joints=include_joints,
                        include_children=include_children,
                    )
                    if include_children
                    else None
                ),
            }
            for name, joint in getattr(asset, "_joints", {}).items()
            if name in getattr(asset, "_joint_children", {})
        }
    return _jsonable(payload)


def export_manifest(
    asset: Asset,
    filepath: str | Path,
    *,
    include_primitives: bool = True,
    include_transforms: bool = True,
    include_joints: bool = True,
    include_children: bool = True,
    indent: int | None = 2,
) -> dict[str, Any]:
    payload = asset_to_manifest(
        asset,
        include_primitives=include_primitives,
        include_transforms=include_transforms,
        include_joints=include_joints,
        include_children=include_children,
    )
    path = Path(filepath).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=indent, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


__all__ = ["asset_to_manifest", "export_manifest"]
