from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from .models import (
    CodeCriticDecision,
    DebuggerDecision,
    EditKind,
    EditPlan,
    ImageCriticDecision,
    ObjectPlan,
    ObjectRequest,
    ObjectRunResult,
)
from .prompts import object_prompt
from .tools import AgentToolContext, PATCH_TOOLS, READ_TOOLS, WRITE_TOOLS
from .utils.config import packaged_profile
from .utils.execution import AssetExecutionError, ExecutionResult, execute_asset_source
from .utils.inputs import user_input
from .utils.io import read_json, write_json
from .utils.runner import AgentRuntime
from .utils.usage import UsageRecorder


_CONTEXT_POLICY = {
    "planner": "isolated requirement and references",
    "coder_initial": "isolated requirement, plan, and references; source.py is written by tool",
    "coder_repair": "isolated per repair with plan and current feedback; source.py is read and patched by tool",
    "debugger": "isolated per execution failure with current error; source.py is read by tool",
    "image_critic": "isolated per round with current renders, references, and explicit text-only prior judgements",
    "code_critic": "isolated per round with current renders and explicit critic history; source.py is read by tool",
}


class ObjectWorkflow:
    def __init__(self, model_profile: str | Path | None = None) -> None:
        self.model_profile = Path(model_profile or packaged_profile()).resolve()

    async def generate(self, request: ObjectRequest) -> ObjectRunResult:
        self._validate_request(request)
        workspace = self._prepare_workspace(request.workspace)
        self._persist_user_input(workspace, request)
        source_path = workspace / "source.py"
        source_path.touch(exist_ok=False)
        runtime = self._runtime(request, workspace, mode="generate")
        plan = await self._plan_generation(runtime, request)
        write_json(workspace / "plan.json", plan.model_dump())
        self._write_checkpoint(workspace, mode="generate", stage="planned")

        context = AgentToolContext(workspace=workspace, source_path=source_path)
        coder = runtime.agent(
            name="object-coder",
            instructions=object_prompt("coder", articulation=request.articulation),
            tools=WRITE_TOOLS,
        )
        await runtime.run(
            agent=coder,
            input=user_input(
                json.dumps(
                    {
                        "requirement": request.requirement,
                        "articulation_required": request.articulation,
                        "plan": plan.model_dump(),
                        "assignment": "Write source.py with the complete initial implementation.",
                    },
                    ensure_ascii=False,
                ),
                request.image_paths,
            ),
            role="coder:initial",
            stage="initial_code",
            context=context,
        )
        self._require_tool_event(context, "write_file", "initial coder")
        self._write_checkpoint(workspace, mode="generate", stage="refining", next_round=1)
        return await self._iterate(
            runtime=runtime,
            request=request,
            workspace=workspace,
            source_path=source_path,
            mode="generate",
            plan=plan,
        )

    async def edit(
        self,
        request: ObjectRequest,
        *,
        source: str | Path,
        edit_kind: EditKind = "continue",
    ) -> ObjectRunResult:
        self._validate_request(request)
        workspace = self._prepare_workspace(request.workspace)
        self._persist_user_input(workspace, request)
        source_path = workspace / "source.py"
        source_input = Path(source).expanduser().resolve()
        if not source_input.is_file():
            raise FileNotFoundError(source_input)
        shutil.copy2(source_input, source_path)
        runtime = self._runtime(
            request,
            workspace,
            mode="edit",
            edit_kind=edit_kind,
            source_parent=str(source_input),
            initial_source_sha256=self._source_sha256(source_path),
        )
        plan = await self._plan_edit(runtime, request, source_path, edit_kind)
        write_json(workspace / "plan.json", plan.model_dump())
        self._write_checkpoint(
            workspace,
            mode="edit",
            stage="planned",
            edit_kind=edit_kind,
            source_parent=str(source_input),
        )

        context = AgentToolContext(workspace=workspace, source_path=source_path)
        coder = runtime.agent(
            name="object-coder",
            instructions=object_prompt("coder", articulation=request.articulation),
            tools=PATCH_TOOLS,
        )
        await runtime.run(
            agent=coder,
            input=user_input(
                json.dumps(
                    {
                        "edit_kind": edit_kind,
                        "requirement": request.requirement,
                        "plan": plan.model_dump(),
                        "assignment": "Read source.py and apply the requested minimal patch.",
                    },
                    ensure_ascii=False,
                ),
                request.image_paths,
            ),
            role="coder:initial",
            stage="initial_patch",
            context=context,
        )
        self._require_tool_event(context, "apply_patch", "initial edit coder")
        self._write_checkpoint(
            workspace,
            mode="edit",
            stage="refining",
            next_round=1,
            edit_kind=edit_kind,
            source_parent=str(source_input),
        )
        return await self._iterate(
            runtime=runtime,
            request=request,
            workspace=workspace,
            source_path=source_path,
            mode="edit",
            plan=plan,
        )

    async def resume(self, request: ObjectRequest) -> ObjectRunResult:
        """Continue an interrupted object run in its existing workspace.

        ``max_rounds`` is the number of additional refinement attempts allowed by
        this resume invocation. Existing plans, source, sessions, usage, and
        completed round artifacts are retained.
        """

        self._validate_request(request)
        workspace = request.workspace.expanduser().resolve()
        if not workspace.is_dir():
            raise FileNotFoundError(workspace)
        source_path = workspace / "source.py"
        plan_path = workspace / "plan.json"
        if not source_path.is_file():
            raise ValueError("object resume requires an existing source.py")

        checkpoint_path = workspace / "checkpoint.json"
        checkpoint = read_json(checkpoint_path) if checkpoint_path.is_file() else {}
        manifest_path = workspace / "run.json"
        manifest = read_json(manifest_path) if manifest_path.is_file() else {}
        mode = str(checkpoint.get("mode") or manifest.get("mode") or "generate")
        if mode not in {"generate", "edit"}:
            raise ValueError(f"unsupported object resume mode: {mode}")
        if (
            checkpoint.get("stage") == "completed"
            or manifest.get("status") == "completed"
        ) and self._published_files_exist(workspace):
            return self._load_completed_result(workspace, manifest)

        runtime = self._runtime(request, workspace, mode=mode, resume=True)
        runtime.usage.update_manifest(status="running", resumed=True)
        if plan_path.is_file():
            plan_type = ObjectPlan if mode == "generate" else EditPlan
            plan = plan_type.model_validate(read_json(plan_path))
        elif mode == "generate":
            plan = await self._plan_generation(runtime, request)
            write_json(plan_path, plan.model_dump())
            checkpoint = {**checkpoint, "mode": mode, "stage": "planned"}
            write_json(checkpoint_path, checkpoint)
        else:
            edit_kind = str(checkpoint.get("edit_kind") or manifest.get("edit_kind") or "continue")
            if edit_kind not in {"continue", "extend", "variant"}:
                raise ValueError(f"unsupported resumed edit kind: {edit_kind}")
            plan = await self._plan_edit(runtime, request, source_path, edit_kind)
            write_json(plan_path, plan.model_dump())
            checkpoint = {
                **checkpoint,
                "mode": mode,
                "stage": "planned",
                "edit_kind": edit_kind,
                "initial_source_sha256": self._source_sha256(source_path),
            }
            write_json(checkpoint_path, checkpoint)

        stage = str(checkpoint.get("stage") or "refining")
        if stage == "planned":
            if mode == "generate":
                if not source_path.read_text(encoding="utf-8").strip():
                    context = AgentToolContext(workspace=workspace, source_path=source_path)
                    coder = runtime.agent(
                        name="object-coder",
                        instructions=object_prompt("coder", articulation=request.articulation),
                        tools=WRITE_TOOLS,
                    )
                    await runtime.run(
                        agent=coder,
                        input=user_input(
                            json.dumps({
                                "requirement": request.requirement,
                                "articulation_required": request.articulation,
                                "plan": plan.model_dump(),
                                "assignment": "Write source.py with the complete initial implementation.",
                            }, ensure_ascii=False),
                            request.image_paths,
                        ),
                        role="coder:initial",
                        stage="initial_code:resume",
                        context=context,
                    )
                    self._require_tool_event(context, "write_file", "initial coder")
            else:
                initial_hash = checkpoint.get("initial_source_sha256")
                if not initial_hash or initial_hash == self._source_sha256(source_path):
                    context = AgentToolContext(workspace=workspace, source_path=source_path)
                    coder = runtime.agent(
                        name="object-coder",
                        instructions=object_prompt("coder", articulation=request.articulation),
                        tools=PATCH_TOOLS,
                    )
                    await runtime.run(
                        agent=coder,
                        input=user_input(
                            json.dumps({
                                "edit_kind": checkpoint.get("edit_kind", "continue"),
                                "requirement": request.requirement,
                                "plan": plan.model_dump(),
                                "assignment": "Read source.py and apply the requested minimal patch.",
                            }, ensure_ascii=False),
                            request.image_paths,
                        ),
                        role="coder:initial",
                        stage="initial_patch:resume",
                        context=context,
                    )
                    self._require_tool_event(context, "apply_patch", "initial edit coder")
            checkpoint = {**checkpoint, "stage": "refining", "next_round": 1}
            write_json(checkpoint_path, checkpoint)

        return await self._iterate(
            runtime=runtime,
            request=request,
            workspace=workspace,
            source_path=source_path,
            mode=mode,
            plan=plan,
            resume_state=checkpoint,
        )

    def _runtime(
        self,
        request: ObjectRequest,
        workspace: Path,
        *,
        mode: str,
        **fields: object,
    ) -> AgentRuntime:
        runtime = AgentRuntime(
            model_profile=self.model_profile,
            workspace=workspace,
            task_id=request.task_id,
        )
        runtime.write_runtime_config(
            workflow="object_agent",
            request=request,
            execution={
                "render": True,
                "render_view_count": 8,
                "render_elevation": 15.0,
                "export_urdf": True,
                "timeout": 300.0,
            },
            context_policy=_CONTEXT_POLICY,
            mode=mode,
            **fields,
        )
        runtime.usage.update_manifest(
            task_id=request.task_id,
            mode=mode,
            requirement=request.requirement,
            context_policy=_CONTEXT_POLICY,
            status="running",
            **fields,
        )
        return runtime

    async def _plan_generation(
        self,
        runtime: AgentRuntime,
        request: ObjectRequest,
    ) -> ObjectPlan:
        planner = runtime.agent(
            name="object-planner",
            instructions=object_prompt("planner", articulation=request.articulation),
            output_type=ObjectPlan,
        )
        result = await runtime.run(
            agent=planner,
            input=user_input(request.requirement, request.image_paths),
            role="planner",
            stage="plan",
        )
        return self._typed_output(result.final_output, ObjectPlan)

    async def _plan_edit(
        self,
        runtime: AgentRuntime,
        request: ObjectRequest,
        source_path: Path,
        edit_kind: EditKind,
    ) -> EditPlan:
        planner = runtime.agent(
            name="object-edit-planner",
            instructions=object_prompt("edit_planner", articulation=request.articulation),
            tools=READ_TOOLS,
            output_type=EditPlan,
        )
        context = AgentToolContext(workspace=runtime.workspace, source_path=source_path)
        result = await runtime.run(
            agent=planner,
            input=user_input(
                json.dumps(
                    {
                        "source_file": "source.py",
                        "edit_kind": edit_kind,
                        "requirement": request.requirement,
                    },
                    ensure_ascii=False,
                ),
                request.image_paths,
            ),
            role="planner",
            stage="edit_plan",
            context=context,
        )
        self._require_tool_event(context, "read_file", "edit planner")
        return self._typed_output(result.final_output, EditPlan)

    async def _iterate(
        self,
        *,
        runtime: AgentRuntime,
        request: ObjectRequest,
        workspace: Path,
        source_path: Path,
        mode: str,
        plan: ObjectPlan | EditPlan,
        resume_state: dict[str, Any] | None = None,
    ) -> ObjectRunResult:
        rounds_root = workspace / "rounds"
        rounds_root.mkdir(exist_ok=resume_state is not None)
        repairer = runtime.agent(
            name="object-coder",
            instructions=object_prompt("coder", articulation=request.articulation),
            tools=PATCH_TOOLS,
        )
        debugger = runtime.agent(
            name="object-debugger",
            instructions=object_prompt("debugger", articulation=request.articulation),
            tools=READ_TOOLS,
            output_type=DebuggerDecision,
        )
        image_critic = runtime.agent(
            name="object-image-critic",
            instructions=object_prompt("image_critic", articulation=request.articulation),
            output_type=ImageCriticDecision,
        )
        code_critic = runtime.agent(
            name="object-code-critic",
            instructions=object_prompt("code_critic", articulation=request.articulation),
            tools=READ_TOOLS,
            output_type=CodeCriticDecision,
        )
        state = resume_state or {}
        image_history: list[dict[str, object]] = list(state.get("image_history", []))
        code_history: list[dict[str, object]] = list(state.get("code_history", []))
        image_critic_corrections: list[str] = list(
            state.get("image_critic_corrections", [])
        )
        failures: list[dict[str, object]] = list(state.get("failures", []))
        final: tuple[int, ExecutionResult, bool, str] | None = None
        start_round = int(state.get("next_round", 1))
        end_round = start_round + request.max_rounds - 1

        for round_number in range(start_round, end_round + 1):
            round_root = rounds_root / f"round_{round_number:02d}"
            self._preserve_interrupted_round(round_root)
            snapshot = rounds_root / f"round_{round_number:02d}_source.py"
            shutil.copy2(source_path, snapshot)
            try:
                execution = execute_asset_source(
                    source_path,
                    round_root,
                    render=True,
                    export_urdf=True,
                    timeout=300.0,
                )
            except AssetExecutionError as exc:
                failure = {"round": round_number, "stage": "execute", "error": str(exc)}
                failures.append(failure)
                debug_context = AgentToolContext(workspace=workspace, source_path=source_path)
                debug_result = await runtime.run(
                    agent=debugger,
                    input=json.dumps(
                        {
                            "requirement": request.requirement,
                            "plan": plan.model_dump(),
                            "assigned_source": "source.py",
                            "round": round_number,
                            "execution_error": str(exc),
                        },
                        ensure_ascii=False,
                    ),
                    role=f"debugger:round:{round_number}",
                    stage=f"debugger:{round_number}",
                    context=debug_context,
                )
                self._require_tool_event(debug_context, "read_file", "debugger")
                decision = self._typed_output(debug_result.final_output, DebuggerDecision)
                round_root.mkdir(parents=True, exist_ok=True)
                write_json(round_root / "debugger.json", decision.model_dump())
                if round_number == end_round:
                    self._write_checkpoint(
                        workspace,
                        mode=mode,
                        stage="refining",
                        next_round=round_number + 1,
                        image_history=image_history,
                        code_history=code_history,
                        image_critic_corrections=image_critic_corrections,
                        failures=failures,
                    )
                    runtime.usage.update_manifest(status="failed", failures=failures)
                    raise AssetExecutionError(
                        f"Final generated source failed execution after {request.max_rounds} attempts: {exc}"
                    ) from exc
                await self._repair(
                    runtime=runtime,
                    repairer=repairer,
                    workspace=workspace,
                    source_path=source_path,
                    role=f"coder:debugger-repair:{round_number}",
                    stage=f"debugger_patch:{round_number}",
                    payload={
                        "requirement": request.requirement,
                        "plan": plan.model_dump(),
                        "assignment": "Patch the exact execution failure diagnosed by the debugger.",
                        "debugger": decision.model_dump(),
                    },
                )
                self._write_checkpoint(
                    workspace,
                    mode=mode,
                    stage="refining",
                    next_round=round_number + 1,
                    image_history=image_history,
                    code_history=code_history,
                    image_critic_corrections=image_critic_corrections,
                    failures=failures,
                )
                continue

            if round_number == end_round:
                final = (
                    round_number,
                    execution,
                    False,
                    "round_limit_after_execution",
                )
                break

            image_payload = {
                "requirement": request.requirement,
                "planner_checklist": self._critic_checklist(plan),
                "round": round_number,
                "max_rounds": end_round,
                "reference_image_count": len(request.image_paths),
                "render_image_count": len(execution.render_paths),
                "previous_image_decisions": image_history,
                "code_critic_corrections": image_critic_corrections,
            }
            image_result = await runtime.run(
                agent=image_critic,
                input=user_input(
                    json.dumps(image_payload, ensure_ascii=False),
                    (*request.image_paths, *execution.render_paths),
                ),
                role=f"image-critic:round:{round_number}",
                stage=f"image_critic:{round_number}",
            )
            image_decision = self._typed_output(
                image_result.final_output,
                ImageCriticDecision,
            )
            write_json(round_root / "image_critique.json", image_decision.model_dump())
            image_history.append(image_decision.model_dump())
            if image_decision.approved:
                final = (round_number, execution, True, "image_critic_approved")
                break

            code_context = AgentToolContext(workspace=workspace, source_path=source_path)
            code_result = await runtime.run(
                agent=code_critic,
                input=user_input(
                    json.dumps(
                        {
                            "requirement": request.requirement,
                            "plan": plan.model_dump(),
                            "assigned_source": "source.py",
                            "round": round_number,
                            "max_rounds": end_round,
                            "image_critic": image_decision.model_dump(),
                            "previous_code_decisions": code_history,
                        },
                        ensure_ascii=False,
                    ),
                    (*request.image_paths, *execution.render_paths),
                ),
                role=f"code-critic:round:{round_number}",
                stage=f"code_critic:{round_number}",
                context=code_context,
            )
            self._require_tool_event(code_context, "read_file", "code critic")
            code_decision = self._normalize_code_critic_decision(
                self._typed_output(code_result.final_output, CodeCriticDecision)
            )
            write_json(round_root / "code_critique.json", code_decision.model_dump())
            code_history.append(code_decision.model_dump())
            image_critic_corrections = code_decision.image_critic_corrections
            if code_decision.approved:
                final = (round_number, execution, True, "code_critic_approved")
                break
            await self._repair(
                runtime=runtime,
                repairer=repairer,
                workspace=workspace,
                source_path=source_path,
                role=f"coder:critic-repair:{round_number}",
                stage=f"critic_patch:{round_number}",
                payload={
                    "requirement": request.requirement,
                    "plan": plan.model_dump(),
                    "assignment": "Patch every valid required change from the Code Critic.",
                    "code_critic": code_decision.model_dump(),
                },
            )
            self._write_checkpoint(
                workspace,
                mode=mode,
                stage="refining",
                next_round=round_number + 1,
                image_history=image_history,
                code_history=code_history,
                image_critic_corrections=image_critic_corrections,
                failures=failures,
            )

        if final is None:
            raise RuntimeError("Object refinement ended without a publishable execution")
        selected_round, execution, approved, finalization_reason = final
        final_glb, final_urdf, final_renders = self._publish(workspace, execution)
        runtime.usage.update_manifest(
            status="completed",
            error_type=None,
            error=None,
            mode=mode,
            selected_round=selected_round,
            approved=approved,
            critic_skipped=finalization_reason == "round_limit_after_execution",
            finalization_reason=finalization_reason,
            failures=failures,
            image_critic_history=image_history,
            code_critic_history=code_history,
            source_path=str(source_path),
            glb_path=str(final_glb),
            urdf_path=None if final_urdf is None else str(final_urdf),
            render_paths=[str(path) for path in final_renders],
        )
        self._write_checkpoint(
            workspace,
            mode=mode,
            stage="completed",
            next_round=selected_round + 1,
            image_history=image_history,
            code_history=code_history,
            image_critic_corrections=image_critic_corrections,
            failures=failures,
            approved=approved,
            finalization_reason=finalization_reason,
        )
        return ObjectRunResult(
            workspace=workspace,
            source_path=source_path,
            glb_path=final_glb,
            urdf_path=final_urdf,
            render_paths=tuple(final_renders),
            selected_round=selected_round,
            approved=approved,
            usage=runtime.usage.totals(),
        )

    @staticmethod
    def _write_checkpoint(workspace: Path, **fields: object) -> Path:
        checkpoint_path = workspace / "checkpoint.json"
        payload = read_json(checkpoint_path) if checkpoint_path.is_file() else {
            "version": 1,
            "workflow": "object_agent",
        }
        payload.update(fields)
        return write_json(checkpoint_path, payload)

    @staticmethod
    def _source_sha256(source_path: Path) -> str:
        return hashlib.sha256(source_path.read_bytes()).hexdigest()

    @staticmethod
    def _preserve_interrupted_round(round_root: Path) -> None:
        if not round_root.exists():
            return
        index = 1
        while True:
            preserved = round_root.with_name(f"{round_root.name}_interrupted_{index:02d}")
            if not preserved.exists():
                round_root.rename(preserved)
                return
            index += 1

    @staticmethod
    def _published_files_exist(workspace: Path) -> bool:
        return (workspace / "source.py").is_file() and (workspace / "scene.glb").is_file()

    @staticmethod
    def _load_completed_result(workspace: Path, manifest: dict[str, Any]) -> ObjectRunResult:
        render_root = workspace / "render"
        urdf_path = workspace / "scene.urdf"
        usage = UsageRecorder(workspace).totals()
        return ObjectRunResult(
            workspace=workspace,
            source_path=workspace / "source.py",
            glb_path=workspace / "scene.glb",
            urdf_path=urdf_path if urdf_path.is_file() else None,
            render_paths=tuple(sorted(render_root.glob("*"))) if render_root.is_dir() else (),
            selected_round=int(manifest.get("selected_round", 0)),
            approved=bool(manifest.get("approved", False)),
            usage=usage,
        )

    async def _repair(
        self,
        *,
        runtime: AgentRuntime,
        repairer: Any,
        workspace: Path,
        source_path: Path,
        role: str,
        stage: str,
        payload: dict[str, object],
    ) -> None:
        context = AgentToolContext(workspace=workspace, source_path=source_path)
        payload = {"assigned_source": "source.py", **payload}
        await runtime.run(
            agent=repairer,
            input=json.dumps(payload, ensure_ascii=False),
            role=role,
            stage=stage,
            context=context,
        )
        self._require_tool_event(context, "apply_patch", stage)

    @staticmethod
    def _publish(
        workspace: Path,
        execution: ExecutionResult,
    ) -> tuple[Path, Path | None, list[Path]]:
        glb_path = workspace / "scene.glb"
        shutil.copy2(execution.glb_path, glb_path)
        urdf_path = None
        workspace_urdf = workspace / "scene.urdf"
        meshes_root = workspace / "meshes"
        if execution.urdf_path is not None:
            urdf_path = workspace_urdf
            shutil.copy2(execution.urdf_path, urdf_path)
            if meshes_root.exists():
                shutil.rmtree(meshes_root)
            shutil.copytree(execution.urdf_path.parent / "meshes", meshes_root)
        else:
            workspace_urdf.unlink(missing_ok=True)
            if meshes_root.exists():
                shutil.rmtree(meshes_root)
        render_root = workspace / "render"
        if render_root.exists():
            shutil.rmtree(render_root)
        render_root.mkdir()
        render_paths: list[Path] = []
        for source_render in execution.render_paths:
            destination = render_root / source_render.name
            shutil.copy2(source_render, destination)
            render_paths.append(destination)
        return glb_path, urdf_path, render_paths

    @staticmethod
    def _critic_checklist(plan: ObjectPlan | EditPlan) -> list[str]:
        if isinstance(plan, ObjectPlan):
            return plan.critic_checklist
        return [*plan.changes, *[f"preserve: {item}" for item in plan.preserved_features]]

    @staticmethod
    def _require_tool_event(context: AgentToolContext, expected: str, actor: str) -> None:
        if not any(event.tool == expected for event in context.events):
            raise RuntimeError(f"{actor} did not use required tool {expected}")

    @staticmethod
    def _validate_request(request: ObjectRequest) -> None:
        if not request.requirement.strip():
            raise ValueError("requirement must not be empty")
        if not request.task_id.strip():
            raise ValueError("task_id must not be empty")
        if request.max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        for image_path in request.image_paths:
            if not Path(image_path).expanduser().resolve().is_file():
                raise FileNotFoundError(image_path)

    @staticmethod
    def _prepare_workspace(value: Path) -> Path:
        workspace = value.expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=False)
        return workspace

    @staticmethod
    def _persist_user_input(workspace: Path, request: ObjectRequest) -> None:
        (workspace / "user_input.txt").write_text(
            request.requirement,
            encoding="utf-8",
        )
        saved_images: list[dict[str, str]] = []
        if request.image_paths:
            image_root = workspace / "user_input_images"
            image_root.mkdir(exist_ok=False)
            for index, image_value in enumerate(request.image_paths, 1):
                source = Path(image_value).expanduser().resolve()
                destination = image_root / f"{index:02d}_{source.name}"
                shutil.copy2(source, destination)
                saved_images.append(
                    {
                        "source_path": str(source),
                        "saved_path": destination.relative_to(workspace).as_posix(),
                    }
                )
        write_json(
            workspace / "user_input.json",
            {
                "requirement": request.requirement,
                "image_paths": saved_images,
                "articulation": request.articulation,
                "max_rounds": request.max_rounds,
            },
        )

    @staticmethod
    def _typed_output(value: Any, expected: type[Any]) -> Any:
        if not isinstance(value, expected):
            raise TypeError(f"Expected {expected.__name__}, got {type(value).__name__}")
        return value

    @staticmethod
    def _normalize_code_critic_decision(
        decision: CodeCriticDecision,
    ) -> CodeCriticDecision:
        if decision.approved and decision.required_changes:
            return decision.model_copy(update={"approved": False})
        return decision


__all__ = ["ObjectWorkflow"]
