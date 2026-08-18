from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence

from agents import Agent, Runner

from .config import ModelProfile
from .io import write_json
from .sessions import SessionManager
from .usage import UsageRecorder


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class AgentRuntime:
    def __init__(
        self,
        *,
        model_profile: str | Path,
        workspace: str | Path,
        task_id: str,
    ) -> None:
        self.profile = ModelProfile.load(model_profile)
        if self.profile.api not in {"chat_completions", "responses"}:
            raise ValueError("AgentRuntime requires a chat_completions or responses profile")
        self.workspace = Path(workspace).expanduser().resolve()
        self.sessions = SessionManager(self.workspace, task_id)
        self.usage = UsageRecorder(self.workspace)
        self.model = self.profile.agent_model()
        self.model_settings = self.profile.model_settings()

    def write_runtime_config(
        self,
        *,
        workflow: str,
        request: Any,
        execution: dict[str, Any],
        context_policy: dict[str, str],
        **fields: Any,
    ) -> Path:
        """Persist the effective non-secret configuration for one workflow run."""

        profile = self.profile
        return write_json(
            self.workspace / "runtime_config.json",
            _json_value({
                "workflow": workflow,
                "model": {
                    "profile_name": profile.path.name,
                    "provider": "openai",
                    "api": profile.api,
                    "base_url": profile.base_url,
                    "model": profile.model,
                    "timeout": profile.timeout,
                    "max_retries": profile.max_retries,
                    "max_tokens": profile.max_tokens,
                    "temperature": profile.temperature,
                    "parallel_tool_calls": profile.parallel_tool_calls,
                    "include_usage": profile.include_usage,
                },
                "request": request,
                "execution": execution,
                "context_policy": context_policy,
                **fields,
            }),
        )

    def agent(
        self,
        *,
        name: str,
        instructions: str,
        tools: Sequence[Any] = (),
        output_type: type[Any] | None = None,
    ) -> Agent[Any]:
        return Agent(
            name=name,
            instructions=instructions,
            model=self.model,
            model_settings=self.model_settings,
            tools=list(tools),
            output_type=output_type,
        )

    async def run(
        self,
        *,
        agent: Agent[Any],
        input: Any,
        role: str,
        stage: str,
        context: Any | None = None,
        max_turns: int = 16,
    ):
        result = await Runner.run(
            agent,
            input,
            context=context,
            max_turns=max_turns,
            session=self.sessions.for_role(role),
        )
        self.usage.record(stage=stage, agent=agent.name, result=result)
        return result


__all__ = ["AgentRuntime"]
