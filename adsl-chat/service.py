from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from adsl.agents import ObjectRequest, ObjectWorkflow
from adsl.agents.utils.config import packaged_profile
from adsl.agents.utils.inputs import user_input
from adsl.agents.utils.runner import AgentRuntime
from adsl.agents.utils.usage import UsageTotals
from .models import ChatDecision, ChatRequest, ChatResult
from .workspace import WORKSPACE_QUERY_TOOLS


_CONTEXT_POLICY = {
    "router": "one persistent conversation session keyed by the conversation workspace",
    "workspace_queries": "read-only inspection of active, remembered, or user-specified object workspaces",
    "asset_workflow": "an independent object-agent workspace for every created or edited asset",
}


class ObjectChatService:
    """Shared conversational application service for CLI and web clients."""

    def __init__(
        self,
        model_profile: str | Path | None = None,
        *,
        product_name: str = "aDSL Chat",
    ) -> None:
        self.model_profile = Path(model_profile or packaged_profile()).resolve()
        self.product_name = product_name
        self._locks: dict[str, asyncio.Lock] = {}

    async def handle(self, request: ChatRequest) -> ChatResult:
        key = str(request.conversation_workspace.expanduser().resolve())
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            decision, conversation_usage = await self._route(request)
            return await self.execute(
                request=request,
                decision=decision,
                conversation_usage=conversation_usage,
            )

    async def execute(
        self,
        *,
        request: ChatRequest,
        decision: ChatDecision,
        conversation_usage: UsageTotals | None = None,
    ) -> ChatResult:
        usage = conversation_usage or UsageTotals()
        if decision.action == "chat":
            return ChatResult(decision=decision, asset=None, conversation_usage=usage)
        if request.output_workspace is None:
            raise ValueError("output_workspace is required for asset actions")
        object_request = ObjectRequest(
            requirement=decision.requirement or "",
            workspace=request.output_workspace,
            task_id=request.task_id,
            image_paths=request.image_paths,
            articulation=decision.articulation,
            max_rounds=request.max_rounds,
        )
        workflow = ObjectWorkflow(self.model_profile)
        if decision.action == "create":
            asset = await workflow.generate(object_request)
        else:
            if request.source_path is None:
                raise ValueError(f"source_path is required for {decision.action}")
            asset = await workflow.edit(
                object_request,
                source=request.source_path,
                edit_kind=decision.action,
            )
        return ChatResult(decision=decision, asset=asset, conversation_usage=usage)

    async def route(self, request: ChatRequest) -> ChatDecision:
        key = str(request.conversation_workspace.expanduser().resolve())
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            decision, _usage = await self._route(request)
            return decision

    async def _route(self, request: ChatRequest) -> tuple[ChatDecision, UsageTotals]:
        runtime = AgentRuntime(
            model_profile=self.model_profile,
            workspace=request.conversation_workspace,
            task_id=request.task_id,
        )
        runtime.write_runtime_config(
            workflow="object_chat",
            request=request,
            execution={"asset_workflow": "object_agent"},
            context_policy=_CONTEXT_POLICY,
            product_name=self.product_name,
        )
        runtime.usage.update_manifest(
            task_id=request.task_id,
            mode="chat",
            status="running",
        )
        router = runtime.agent(
            name="object-chat",
            instructions=f"""You are the conversational controller for {self.product_name}.
Understand the user's intent semantically and choose exactly one action:
- chat: answer or ask a focused clarification when no asset creation or edit is requested.
- create: create an independent new 3D asset.
- continue: modify the selected asset while preserving every unaffected feature.
- extend: add components and integrate them into the selected asset hierarchy.
- variant: create an alternative that preserves the selected asset's reusable structure.

Choose an asset action only when the user clearly requests creation or modification. A discussion about modeling is not itself a creation request. continue, extend, and variant require an active asset; without one choose chat or create.

Answer questions about existing assets from the supplied workspace context. When the user names a workspace path or asks for details not present in that context, use `inspect_workspace` or `read_workspace_file` before answering. These tools are read-only. Never guess about workspace contents, and do not claim that inspection changed an asset.

For every asset action, `requirement` must be a precise, standalone description of the complete target asset after the requested action, not merely the latest delta. For continue, extend, and variant, incorporate the active asset context, explicitly preserve unaffected geometry, appearance, hierarchy, behavior, and articulation, and state the requested differences. If essential intent is missing, choose chat and ask one focused question instead of inventing details.

Set `articulation=true` when the target requires functional moving parts or kinematic links, including hinges, doors, lids, drawers, sliders, telescoping parts, rotating wheels or arms, and robot joints. Set it to false for static objects or fixed poses without functional motion. For continue, extend, and variant, preserve the active asset's articulation value unless the user explicitly asks to add or remove functional motion. When articulation is enabled, include the intended moving parts and motion in `requirement`.

`assistant_message` is the concise user-facing response. Keep it grounded in the provided conversation and asset context. Return structured output only.""",
            tools=WORKSPACE_QUERY_TOOLS,
            output_type=ChatDecision,
        )
        workspace_context = request.asset_context
        if not workspace_context and request.source_path is not None:
            workspace_context = json.dumps(
                {"source_path": str(request.source_path.expanduser().resolve())},
                ensure_ascii=False,
            )
        editable_state = (
            "An editable active asset source is available."
            if request.source_path is not None
            else "There is no editable active asset source."
        )
        result = await runtime.run(
            agent=router,
            input=user_input(
                (
                    f"Workspace context:\n{workspace_context}\n\n{editable_state}\n\nLatest user message:\n{request.message}"
                    if workspace_context
                    else f"There is no active or remembered 3D asset workspace.\n{editable_state}\n\nLatest user message:\n{request.message}"
                ),
                request.image_paths,
            ),
            role="conversation",
            stage="route",
        )
        if not isinstance(result.final_output, ChatDecision):
            raise TypeError(
                f"Expected ChatDecision, got {type(result.final_output).__name__}"
            )
        runtime.usage.update_manifest(
            status="completed",
            action=result.final_output.action,
        )
        return result.final_output, runtime.usage.totals()


__all__ = ["ObjectChatService"]
