from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
from typing import Sequence

# Blender must initialize before trimesh-backed modules on Windows.
from adsl.core import Asset, export_glb, export_urdf
from adsl.tools.render import render_video


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-view-count", type=int, default=8)
    parser.add_argument("--render-elevation", type=float, default=15.0)
    parser.add_argument("--urdf", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    namespace = runpy.run_path(str(source), run_name="__adsl_generated__")
    scene = namespace.get("scene")
    if not isinstance(scene, Asset):
        raise TypeError("Generated source must assign an adsl Asset to `scene`")
    render_root = output / "render"
    render_root.mkdir(parents=True, exist_ok=True)
    glb_path = render_root / "scene.glb"
    export_glb(
        scene,
        filepath=glb_path,
        clear_scene=True,
        apply_modifiers=True,
    )

    urdf_path = None
    if args.urdf:
        urdf_path = render_root / "scene.urdf"
        export_urdf(
            scene,
            filepath=urdf_path,
            mesh_backend="blender",
            visual_mesh_format="glb",
        )

    if args.render:
        render_video(
            output_dir=render_root,
            glb_path=glb_path,
            elevations=(args.render_elevation,),
            num_camera_per_layer=args.render_view_count,
        )

    manifest = {
        "result_path": str(output),
        "render_root": str(render_root),
        "glb_path": str(glb_path),
        "urdf_path": None if urdf_path is None else str(urdf_path),
    }
    (output / "execution.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
