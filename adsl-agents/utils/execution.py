from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys


@dataclass(frozen=True)
class ExecutionResult:
    output_root: Path
    glb_path: Path
    urdf_path: Path | None
    render_paths: tuple[Path, ...]
    stdout: str
    stderr: str


class AssetExecutionError(RuntimeError):
    pass


def execute_asset_source(
    source_path: str | Path,
    output_root: str | Path,
    *,
    render: bool = True,
    render_view_count: int = 8,
    render_elevation: float = 15.0,
    export_urdf: bool = True,
    timeout: float = 300.0,
    working_directory: str | Path | None = None,
) -> ExecutionResult:
    """Execute one generated aDSL program in an isolated child process."""

    source = Path(source_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    workdir = source.parent if working_directory is None else Path(working_directory).expanduser().resolve()
    if not workdir.is_dir():
        raise NotADirectoryError(workdir)
    if render and render_view_count < 1:
        raise ValueError("render_view_count must be at least 1")
    output.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        "-m",
        "adsl.agents.utils.asset_executor",
        "--source",
        str(source),
        "--output",
        str(output),
    ]
    if render:
        command.extend(
            [
                "--render",
                "--render-view-count",
                str(render_view_count),
                "--render-elevation",
                str(render_elevation),
            ]
        )
    if export_urdf:
        command.append("--urdf")
    completed = subprocess.run(
        command,
        cwd=workdir,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise AssetExecutionError(
            f"Generated source exited with code {completed.returncode}.\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    manifest_path = output / "execution.json"
    if not manifest_path.is_file():
        raise AssetExecutionError("Generated source did not write execution.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    glb_path = Path(manifest["glb_path"]).resolve()
    if not glb_path.is_file():
        raise AssetExecutionError(f"Generated GLB is missing: {glb_path}")
    urdf_value = manifest.get("urdf_path")
    urdf_path = None if urdf_value is None else Path(urdf_value).resolve()
    if urdf_path is not None and not urdf_path.is_file():
        raise AssetExecutionError(f"Generated URDF is missing: {urdf_path}")
    render_paths = tuple(sorted((output / "render").glob("*.png")))
    if render and not render_paths:
        raise AssetExecutionError("Rendering was requested but no PNG was produced")
    return ExecutionResult(
        output_root=output,
        glb_path=glb_path,
        urdf_path=urdf_path,
        render_paths=render_paths,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


__all__ = ["AssetExecutionError", "ExecutionResult", "execute_asset_source"]
