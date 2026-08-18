from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import bpy
import mathutils

from bpyrenderer import SceneManager
from bpyrenderer.camera import add_camera
from bpyrenderer.camera.layout import get_camera_positions_on_sphere
from bpyrenderer.engine import init_render_engine
from bpyrenderer.render_output import enable_color_output


Background = Literal["transparent", "white", "gray"]
DEFAULT_SUN_LOCATION = (20.0, -20.0, 0.0)
DEFAULT_SUN_STRENGTH = 1.25
DEFAULT_WORLD_STRENGTH = 0.35
DEFAULT_COLOR_EXPOSURE = -0.8
DEFAULT_RENDER_ELEVATIONS = (15.0,)


@dataclass(frozen=True)
class RenderView:
    index: int
    camera_matrix: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]
    elevation: float | None = None
    azimuth: float | None = None
    label: str | None = None
    sun_location: tuple[float, float, float] | None = None


def _mat4_tuple(value: Any) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]:
    rows = [[float(item) for item in row] for row in value]
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError(f"Expected a 4x4 camera matrix, got: {value!r}")
    return tuple(tuple(row) for row in rows)  # type: ignore[return-value]


def _vec3_tuple(value: Any) -> tuple[float, float, float]:
    items = [float(item) for item in value]
    if len(items) != 3:
        raise ValueError(f"Expected a 3-vector, got: {value!r}")
    return (items[0], items[1], items[2])


def _coerce_render_view(
    value: RenderView | Mapping[str, Any],
    index: int,
) -> RenderView:
    if isinstance(value, RenderView):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(
            "Render views must be RenderView or mapping objects, "
            f"got {type(value)!r}"
        )
    camera_matrix = (
        value.get("camera_matrix")
        or value.get("transform_matrix")
        or value.get("matrix")
    )
    if camera_matrix is None:
        raise ValueError(f"Render view is missing a camera matrix: {value!r}")
    sun_location = value.get("sun_location")
    return RenderView(
        index=int(value.get("index", index)),
        camera_matrix=_mat4_tuple(camera_matrix),
        elevation=(
            None if value.get("elevation") is None else float(value["elevation"])
        ),
        azimuth=None if value.get("azimuth") is None else float(value["azimuth"]),
        label=None if value.get("label") is None else str(value["label"]),
        sun_location=None if sun_location is None else _vec3_tuple(sun_location),
    )


def _coerce_render_views(
    views: Sequence[RenderView | Mapping[str, Any]],
) -> tuple[RenderView, ...]:
    if len(views) == 0:
        raise ValueError("views must contain at least one render view")
    return tuple(
        _coerce_render_view(view, index)
        for index, view in enumerate(views)
    )


def orbit_render_views(
    *,
    elevations: Sequence[float] = DEFAULT_RENDER_ELEVATIONS,
    num_camera_per_layer: int = 8,
) -> tuple[RenderView, ...]:
    if int(num_camera_per_layer) <= 0:
        raise ValueError("num_camera_per_layer must be greater than 0")
    if len(elevations) == 0:
        raise ValueError("elevations must contain at least one value")

    _camera_positions, camera_matrices, elevation_values, azimuth_values = (
        get_camera_positions_on_sphere(
            center=(0, 0, 0),
            radius=1.5,
            elevations=list(elevations),
            num_camera_per_layer=int(num_camera_per_layer),
            azimuth_offset=-90,
        )
    )
    return tuple(
        RenderView(
            index=index,
            camera_matrix=_mat4_tuple(camera_matrices[index]),
            elevation=float(elevation_values[index]),
            azimuth=float(azimuth_values[index]),
        )
        for index in range(len(camera_matrices))
    )


def _mesh_objects() -> list[bpy.types.Object]:
    return [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH"
        if obj.data is not None and len(obj.data.vertices) > 0
    ]


def _scene_bounds() -> tuple[mathutils.Vector, mathutils.Vector]:
    infinity = float("inf")
    bounds_min = mathutils.Vector((infinity, infinity, infinity))
    bounds_max = mathutils.Vector((-infinity, -infinity, -infinity))

    meshes = _mesh_objects()
    if not meshes:
        raise RuntimeError("No mesh objects found after importing the GLB scene")

    for obj in meshes:
        for coordinate in obj.bound_box:
            world_coordinate = obj.matrix_world @ mathutils.Vector(coordinate)
            bounds_min = mathutils.Vector(
                (
                    min(bounds_min.x, world_coordinate.x),
                    min(bounds_min.y, world_coordinate.y),
                    min(bounds_min.z, world_coordinate.z),
                )
            )
            bounds_max = mathutils.Vector(
                (
                    max(bounds_max.x, world_coordinate.x),
                    max(bounds_max.y, world_coordinate.y),
                    max(bounds_max.z, world_coordinate.z),
                )
            )
    return bounds_min, bounds_max


def _normalize_scene(normalize_range: float = 0.8) -> None:
    bounds_min, bounds_max = _scene_bounds()
    max_extent = max(bounds_max - bounds_min)
    if max_extent <= 0:
        raise RuntimeError("Cannot normalize scene with zero-size bounds")

    scale = normalize_range / max_extent
    offset = -(bounds_min + bounds_max) / 2
    for obj in bpy.context.scene.objects:
        if obj.parent is not None or obj.type in {"CAMERA", "LIGHT"}:
            continue
        obj.matrix_world.translation += offset
        obj.matrix_world.translation = obj.matrix_world.translation * scale
        obj.scale = obj.scale * scale

    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")


def _smooth_scene() -> None:
    for obj in _mesh_objects():
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth()
    bpy.ops.object.select_all(action="DESELECT")


def _set_first_supported(owner: Any, attribute: str, values: Sequence[Any]) -> None:
    for value in values:
        try:
            setattr(owner, attribute, value)
        except Exception:
            continue
        if getattr(owner, attribute, None) == value:
            return


def _configure_color_management() -> None:
    scene = bpy.context.scene
    scene.display_settings.display_device = "sRGB"
    _set_first_supported(
        scene.view_settings,
        "view_transform",
        ("Filmic", "AgX", "Standard"),
    )
    _set_first_supported(
        scene.view_settings,
        "look",
        ("Medium High Contrast", "Medium Contrast", "None"),
    )
    scene.view_settings.exposure = DEFAULT_COLOR_EXPOSURE
    scene.view_settings.gamma = 1.0


def _disable_ambient_occlusion() -> None:
    view_layer = bpy.context.scene.view_layers["ViewLayer"]
    if hasattr(view_layer, "use_pass_ambient_occlusion"):
        view_layer.use_pass_ambient_occlusion = False

    cycles = bpy.context.scene.cycles
    if hasattr(cycles, "use_ambient_occlusion"):
        cycles.use_ambient_occlusion = False

    eevee = getattr(bpy.context.scene, "eevee", None)
    if eevee is not None and hasattr(eevee, "use_gtao"):
        eevee.use_gtao = False


def _import_glb(glb_path: str | Path | None) -> None:
    if glb_path is None or not Path(glb_path).is_file():
        raise ValueError(f"GLB file not found: {glb_path}")
    result = bpy.ops.import_scene.gltf(filepath=str(Path(glb_path).resolve()))
    if "FINISHED" not in result:
        raise RuntimeError(f"Failed to import GLB file: {glb_path}")


def _set_scene_background(background: Background) -> None:
    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    if scene.world.node_tree is None:
        scene.world.use_nodes = True

    scene.render.film_transparent = background == "transparent"
    nodes = scene.world.node_tree.nodes
    background_node = nodes.get("Background")
    if background_node is None:
        background_node = nodes.new(type="ShaderNodeBackground")
        output_node = nodes.get("World Output") or nodes.new(
            type="ShaderNodeOutputWorld"
        )
        scene.world.node_tree.links.new(
            background_node.outputs["Background"],
            output_node.inputs["Surface"],
        )

    color = (1.0, 1.0, 1.0, 1.0)
    if background == "gray":
        color = (0.45, 0.45, 0.45, 1.0)
    background_node.inputs["Color"].default_value = color
    background_node.inputs["Strength"].default_value = DEFAULT_WORLD_STRENGTH


def _set_key_sun(
    location: Sequence[float] = DEFAULT_SUN_LOCATION,
    *,
    strength: float = DEFAULT_SUN_STRENGTH,
) -> None:
    for obj in list(bpy.context.scene.objects):
        if obj.type != "LIGHT":
            continue
        light_data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if light_data is not None and light_data.users == 0:
            bpy.data.lights.remove(light_data, do_unlink=True)

    sun_data = bpy.data.lights.new(name="KeySun", type="SUN")
    sun_data.energy = float(strength)
    sun_data.use_shadow = True
    sun = bpy.data.objects.new(name="KeySun", object_data=sun_data)
    bpy.context.scene.collection.objects.link(sun)
    sun.location = _vec3_tuple(location)
    direction = -sun.location
    if direction.length > 1e-12:
        sun.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _write_metadata(
    output_dir: str | Path,
    width: int,
    height: int,
    cameras: Sequence[bpy.types.Object],
    views: Sequence[RenderView],
) -> None:
    metadata = {"width": width, "height": height, "locations": []}
    for camera, view in zip(cameras, views):
        record = {
            "index": f"{view.index:04d}",
            "projection_type": camera.data.type,
            "ortho_scale": camera.data.ortho_scale,
            "camera_angle_x": camera.data.angle_x,
            "elevation": view.elevation,
            "azimuth": view.azimuth,
            "transform_matrix": [
                [float(value) for value in row]
                for row in mathutils.Matrix(camera.matrix_world)
            ],
        }
        if view.label is not None:
            record["label"] = view.label
        if view.sun_location is not None:
            record["sun_location"] = [float(value) for value in view.sun_location]
        metadata["locations"].append(record)

    metadata_path = Path(output_dir) / "meta.json"
    metadata_path.write_text(json.dumps(metadata, indent=4), encoding="utf-8")


def render_multiview(
    output_dir: str | Path,
    glb_path: str | Path,
    views: Sequence[RenderView | Mapping[str, Any]],
    width: int = 1024,
    height: int = 1024,
    render_samples: int = 256,
    render_threads: int | None = None,
    background: Background = "transparent",
) -> None:
    if background not in {"transparent", "white", "gray"}:
        raise ValueError(
            f"Invalid background: {background!r}. Expected transparent, white, or gray."
        )
    resolved_views = _coerce_render_views(views)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    init_render_engine(
        "BLENDER_EEVEE",
        render_samples=max(1, int(render_samples)),
    )
    scene_manager = SceneManager()
    scene_manager.clear(reset_keyframes=True)

    _import_glb(glb_path)
    _smooth_scene()
    _normalize_scene(0.8)
    _set_scene_background(background)

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 1
    cameras = []
    for index, view in enumerate(resolved_views):
        camera = add_camera(
            mathutils.Matrix(view.camera_matrix),
            add_frame=index < len(resolved_views) - 1,
        )
        cameras.append(camera)

    enable_color_output(
        int(width),
        int(height),
        str(output_path),
        file_prefix="render_",
        file_format="PNG",
        mode="IMAGE",
        film_transparent=background == "transparent",
    )
    if render_threads is not None and int(render_threads) > 0:
        scene.render.threads_mode = "FIXED"
        scene.render.threads = int(render_threads)

    _set_key_sun()
    _configure_color_management()
    _disable_ambient_occlusion()

    scene_manager.render()
    _write_metadata(output_path, width, height, cameras, resolved_views)


def render_video(
    output_dir: str | Path,
    shape=None,
    glb_path: str | Path | None = None,
    width: int = 1024,
    height: int = 1024,
    elevations: Sequence[float] = DEFAULT_RENDER_ELEVATIONS,
    num_camera_per_layer: int = 120,
    render_samples: int = 256,
    render_threads: int | None = None,
    background: Background = "transparent",
) -> None:
    """Export an Asset when needed and render orbit-view PNG images."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if glb_path is None:
        if shape is None:
            raise ValueError("render_video requires either glb_path or shape.")
        from adsl.core import export_glb

        glb_path = output_path / "scene.glb"
        export_glb(
            shape,
            filepath=glb_path,
            clear_scene=True,
            apply_modifiers=False,
        )

    views = orbit_render_views(
        elevations=elevations,
        num_camera_per_layer=num_camera_per_layer,
    )
    render_multiview(
        output_dir=output_path,
        glb_path=glb_path,
        views=views,
        width=width,
        height=height,
        render_samples=render_samples,
        render_threads=render_threads,
        background=background,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render multi-view PNGs from a GLB file with bpyrenderer."
    )
    parser.add_argument("--glb-path", required=True, help="Path to the input GLB file.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument(
        "--elevations",
        type=float,
        nargs="+",
        default=list(DEFAULT_RENDER_ELEVATIONS),
    )
    parser.add_argument("--num-camera-per-layer", type=int, default=8)
    parser.add_argument("--render-samples", type=int, default=256)
    parser.add_argument("--render-threads", type=int, default=None)
    parser.add_argument(
        "--background",
        choices=("transparent", "white", "gray"),
        default="transparent",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    output_dir = args.output_dir or str(
        Path("render_output") / f"{Path(args.glb_path).stem}_bpyrenderer"
    )
    render_video(
        output_dir=output_dir,
        glb_path=args.glb_path,
        width=args.width,
        height=args.height,
        elevations=args.elevations,
        num_camera_per_layer=args.num_camera_per_layer,
        render_samples=args.render_samples,
        render_threads=args.render_threads,
        background=args.background,
    )


__all__ = [
    "RenderView",
    "orbit_render_views",
    "render_multiview",
    "render_video",
    "main",
]


if __name__ == "__main__":
    main()
