from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from adsl.agents.utils.config import ModelProfile, packaged_profile

from .models import ChatRequest, ChatResult
from .service import ObjectChatService
from .state import ChatSessionState
from .workspace import (
    WorkspaceSnapshot,
    build_workspace_context,
    format_workspace_table,
    scan_workspace,
)


DEFAULT_STATE_FILE = "chat_state.json"


@dataclass(frozen=True)
class SlashCommand:
    name: str
    argument: str = ""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Conversation workspace containing persistent chat state and model sessions.",
    )
    parser.add_argument("--session-file", type=Path)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--max-rounds", type=int, default=2)
    return parser


def _parse_slash_command(raw: str) -> SlashCommand | None:
    text = raw.strip()
    if not text.startswith("/"):
        return None
    body = text[1:].strip()
    if not body:
        return SlashCommand("unknown")
    name, separator, argument = body.partition(" ")
    return SlashCommand(name=name.lower(), argument=argument.strip() if separator else "")


def _print_help() -> None:
    print("Commands:")
    print("  /help            Show this help text.")
    print("  /paste           Enter a multiline message; finish with /end.")
    print("  /open PATH       Attach an existing asset workspace.")
    print("  /target PATH     Reserve PATH for the next generated or edited asset.")
    print("  /resume PATH     Resume a saved chat_state.json session.")
    print("  /config PATH     Switch the active model profile.")
    print("  /where           Show conversation, active, and pending workspaces.")
    print("  /memories        List remembered asset workspaces.")
    print("  /exit            Exit the chat CLI.")


def _read_multiline_message() -> str:
    print("Paste mode: finish with /end, or cancel with /cancel.")
    lines: list[str] = []
    while True:
        try:
            line = input("... ")
        except EOFError:
            break
        if line == "/end":
            break
        if line == "/cancel":
            return ""
        lines.append(line)
    return "\n".join(lines).strip("\n")


def _initialize_state(args: argparse.Namespace) -> tuple[ChatSessionState, Path]:
    conversation_workspace = args.workspace.expanduser().resolve()
    state_path = (
        args.session_file.expanduser().resolve()
        if args.session_file is not None
        else conversation_workspace / DEFAULT_STATE_FILE
    )
    state = ChatSessionState.load(
        state_path,
        conversation_workspace=conversation_workspace,
        task_id=args.task_id,
        model_config=args.model_config or packaged_profile(),
    )
    if args.task_id:
        state.task_id = str(args.task_id).strip()
    if args.model_config is not None:
        state.model_config = str(args.model_config.expanduser().resolve())
    if not state.model_config:
        state.model_config = str(packaged_profile().resolve())
    Path(state.conversation_workspace).mkdir(parents=True, exist_ok=True)
    state.save(state_path)
    return state, state_path


def _active_snapshot(state: ChatSessionState) -> WorkspaceSnapshot | None:
    if not state.active_workspace:
        return None
    snapshot = scan_workspace(state.active_workspace)
    return snapshot if snapshot.exists else None


def _known_snapshots(state: ChatSessionState) -> list[WorkspaceSnapshot]:
    return [
        snapshot
        for snapshot in (scan_workspace(path) for path in state.known_workspaces)
        if snapshot.exists
    ]


def _next_output_workspace(state: ChatSessionState) -> Path:
    if state.pending_workspace:
        return Path(state.pending_workspace)
    return Path(state.conversation_workspace) / "assets" / uuid4().hex


async def _handle_message(
    *,
    service: ObjectChatService,
    state: ChatSessionState,
    message: str,
    max_rounds: int,
) -> ChatResult:
    active = _active_snapshot(state)
    known = _known_snapshots(state)
    result = await service.handle(
        ChatRequest(
            message=message,
            conversation_workspace=Path(state.conversation_workspace),
            output_workspace=_next_output_workspace(state),
            source_path=None if active is None else active.source_path,
            asset_context=build_workspace_context(active, known),
            task_id=state.task_id,
            max_rounds=max_rounds,
        )
    )
    if result.asset is not None:
        state.set_pending_workspace(None)
        state.remember_workspace(result.asset.workspace)
    return result


def _result_payload(result: ChatResult) -> dict[str, object]:
    return {
        "decision": result.decision.model_dump(),
        "asset": None if result.asset is None else str(result.asset.workspace),
    }


def _print_location(state: ChatSessionState, state_path: Path) -> None:
    print(f"Session file: {state_path}")
    print(f"Conversation workspace: {state.conversation_workspace}")
    print(f"Active workspace: {state.active_workspace or 'None'}")
    print(f"Pending workspace: {state.pending_workspace or 'None'}")
    print(f"Model config: {state.model_config or 'default'}")
    active = _active_snapshot(state)
    if active is not None:
        print(active.summary_text(include_source=False))


def _command_argument(command: SlashCommand) -> str | None:
    if command.argument:
        return command.argument
    print(f"/{command.name} requires a path.")
    return None


async def _interactive(args: argparse.Namespace) -> int:
    if args.max_rounds < 1:
        raise ValueError("--max-rounds must be at least 1")
    state, state_path = _initialize_state(args)
    service = ObjectChatService(state.model_config)

    print("aDSL chat CLI")
    _print_location(state, state_path)
    print("Use /help to list commands.")

    while True:
        try:
            raw = input("you> ").strip()
        except KeyboardInterrupt:
            print("")
            continue
        except EOFError:
            print("")
            break
        if not raw:
            continue

        command = _parse_slash_command(raw)
        if command is not None:
            if command.name in {"exit", "quit"}:
                break
            if command.name == "help":
                _print_help()
                continue
            if command.name == "paste":
                raw = _read_multiline_message()
                if not raw:
                    continue
            elif command.name == "open":
                argument = _command_argument(command)
                if argument is None:
                    continue
                snapshot = scan_workspace(argument)
                if not snapshot.exists:
                    print(f"Workspace does not exist: {snapshot.path}")
                    continue
                state.set_pending_workspace(None)
                state.remember_workspace(snapshot.path)
                state.save(state_path)
                print(f"Active workspace set to {snapshot.path}")
                print(snapshot.summary_text(include_source=False))
                continue
            elif command.name == "target":
                argument = _command_argument(command)
                if argument is None:
                    continue
                target = Path(argument).expanduser().resolve()
                if target.exists():
                    print(f"Target already exists; choose a new workspace path: {target}")
                    continue
                state.set_pending_workspace(target)
                state.save(state_path)
                print(f"Pending workspace set to {target}")
                continue
            elif command.name == "resume":
                argument = _command_argument(command)
                if argument is None:
                    continue
                resume_path = Path(argument).expanduser().resolve()
                if not resume_path.is_file():
                    print(f"Session file does not exist: {resume_path}")
                    continue
                state = ChatSessionState.load(resume_path)
                state_path = resume_path
                if not state.model_config:
                    state.model_config = str(packaged_profile().resolve())
                Path(state.conversation_workspace).mkdir(parents=True, exist_ok=True)
                service = ObjectChatService(state.model_config)
                state.save(state_path)
                print(f"Resumed session: {state_path}")
                _print_location(state, state_path)
                continue
            elif command.name == "config":
                argument = _command_argument(command)
                if argument is None:
                    continue
                config_path = Path(argument).expanduser().resolve()
                if not config_path.is_file():
                    print(f"Model profile does not exist: {config_path}")
                    continue
                ModelProfile.load(config_path)
                state.model_config = str(config_path)
                service = ObjectChatService(config_path)
                state.save(state_path)
                print(f"Model config switched to {config_path}")
                continue
            elif command.name == "where":
                _print_location(state, state_path)
                continue
            elif command.name == "memories":
                print(format_workspace_table(state.known_workspaces))
                continue
            elif command.name != "paste":
                print(f"Unknown command: /{command.name}. Use /help for available commands.")
                continue

        try:
            result = await _handle_message(
                service=service,
                state=state,
                message=raw,
                max_rounds=args.max_rounds,
            )
            print(f"assistant> {result.decision.assistant_message}")
            if result.asset is not None:
                print(f"asset> {result.asset.workspace}")
        finally:
            state.save(state_path)

    state.save(state_path)
    return 0


async def _once(args: argparse.Namespace, message: str) -> dict[str, object]:
    if args.max_rounds < 1:
        raise ValueError("--max-rounds must be at least 1")
    state, state_path = _initialize_state(args)
    try:
        result = await _handle_message(
            service=ObjectChatService(state.model_config),
            state=state,
            message=message,
            max_rounds=args.max_rounds,
        )
        return _result_payload(result)
    finally:
        state.save(state_path)


def once_cli(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    parser.add_argument("message")
    args = parser.parse_args(argv)
    print(json.dumps(asyncio.run(_once(args, args.message)), indent=2, ensure_ascii=False))
    return 0


def main_cli(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return asyncio.run(_interactive(args))


if __name__ == "__main__":
    raise SystemExit(main_cli())


__all__ = [
    "DEFAULT_STATE_FILE",
    "SlashCommand",
    "_handle_message",
    "_parse_slash_command",
    "main_cli",
    "once_cli",
]
