from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from typing import Any

from agents import RunResult

from .io import read_json, write_json


_LOCKS_GUARD = Lock()
_LOCKS: dict[Path, Lock] = {}


def _lock_for(path: Path) -> Lock:
    resolved = path.resolve()
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(resolved, Lock())


@dataclass(frozen=True)
class UsageEvent:
    timestamp: str
    stage: str
    agent: str
    requests: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cached_tokens: int
    cache_write_tokens: int
    total_tokens: int
    cost: float | None = None


@dataclass(frozen=True)
class UsageTotals:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    cost: float | None = None


class UsageRecorder:
    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.events_path = self.workspace / "usage.jsonl"
        self.manifest_path = self.workspace / "run.json"

    def record(self, *, stage: str, agent: str, result: RunResult) -> UsageEvent:
        usage = result.context_wrapper.usage
        input_details = usage.input_tokens_details
        output_details = usage.output_tokens_details
        event = UsageEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            stage=stage,
            agent=agent,
            requests=int(usage.requests),
            input_tokens=int(usage.input_tokens),
            output_tokens=int(usage.output_tokens),
            reasoning_tokens=int(output_details.reasoning_tokens),
            cached_tokens=int(input_details.cached_tokens),
            cache_write_tokens=int(input_details.cache_write_tokens),
            total_tokens=int(usage.total_tokens),
        )
        lock = _lock_for(self.events_path)
        with lock:
            with self.events_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
            self._write_manifest_usage()
        return event

    def events(self) -> list[UsageEvent]:
        if not self.events_path.is_file():
            return []
        return [
            UsageEvent(**json.loads(line))
            for line in self.events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def totals(self) -> UsageTotals:
        events = self.events()
        costs = [event.cost for event in events if event.cost is not None]
        return UsageTotals(
            requests=sum(event.requests for event in events),
            input_tokens=sum(event.input_tokens for event in events),
            output_tokens=sum(event.output_tokens for event in events),
            reasoning_tokens=sum(event.reasoning_tokens for event in events),
            cached_tokens=sum(event.cached_tokens for event in events),
            cache_write_tokens=sum(event.cache_write_tokens for event in events),
            total_tokens=sum(event.total_tokens for event in events),
            cost=sum(costs) if costs else None,
        )

    def update_manifest(self, **fields: Any) -> None:
        lock = _lock_for(self.events_path)
        with lock:
            payload = read_json(self.manifest_path) if self.manifest_path.is_file() else {}
            payload.update(fields)
            payload["usage"] = asdict(self.totals())
            write_json(self.manifest_path, payload)

    def _write_manifest_usage(self) -> None:
        payload = read_json(self.manifest_path) if self.manifest_path.is_file() else {}
        payload["usage"] = asdict(self.totals())
        write_json(self.manifest_path, payload)


__all__ = ["UsageEvent", "UsageRecorder", "UsageTotals"]
