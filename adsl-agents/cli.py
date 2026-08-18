from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from .models import ObjectRequest
from .service import ObjectWorkflow
from .utils.io import read_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adsl-run")
    parser.add_argument("--model-config", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    edit = subparsers.add_parser("edit")
    resume = subparsers.add_parser("resume")
    for command in (create, edit):
        command.add_argument("requirement")
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--task-id", default=None)
        command.add_argument("--image", type=Path, action="append", default=[])
        command.add_argument("--articulation", action="store_true")
        command.add_argument("--max-rounds", type=int, default=2)
    edit.add_argument("--source", type=Path, required=True)
    edit.add_argument("--edit-kind", choices=["continue", "extend", "variant"], default="continue")
    resume.add_argument("--output", type=Path, required=True)
    resume.add_argument("--task-id", default=None)
    resume.add_argument("--requirement", default=None)
    resume.add_argument("--max-rounds", type=int, default=2)
    return parser


def _resume_request(args: argparse.Namespace) -> ObjectRequest:
    config_path = args.output.expanduser().resolve() / "runtime_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    payload = read_json(config_path).get("request", {})
    return ObjectRequest(
        requirement=args.requirement or str(payload.get("requirement", "")),
        workspace=args.output,
        task_id=args.task_id or str(payload.get("task_id", "")),
        image_paths=tuple(Path(value) for value in payload.get("image_paths", [])),
        articulation=bool(payload.get("articulation", False)),
        max_rounds=args.max_rounds,
    )


async def _run(args: argparse.Namespace) -> dict[str, object]:
    workflow = ObjectWorkflow(args.model_config)
    if args.command == "resume":
        result = await workflow.resume(_resume_request(args))
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(result).items()
        }
    request = ObjectRequest(
        requirement=args.requirement,
        workspace=args.output,
        task_id=args.task_id or uuid4().hex,
        image_paths=tuple(args.image),
        articulation=args.articulation,
        max_rounds=args.max_rounds,
    )
    if args.command == "create":
        result = await workflow.generate(request)
    else:
        result = await workflow.edit(request, source=args.source, edit_kind=args.edit_kind)
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(result).items()}


def main_cli(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(asyncio.run(_run(args)), indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
