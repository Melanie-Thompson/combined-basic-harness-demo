"""Tool execution via API server — the harness is a portal, not an execution
environment. All tool calls are routed to the API server's execute endpoint.
"""

import json
import logging
import os
from collections.abc import Callable
from contextlib import AsyncExitStack

import httpx
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, create_model, Field

from AI.capabilities import audit_capabilities, check_capabilities
from AI.effects import check_effects
from AI.models.message import ToolExecutionRecord
from AI.package_signing import verify_package
from AI.signing import canonicalize_args, sign_execution

log = logging.getLogger("sandbox")

API_SERVER = os.getenv("API_SERVER", "http://localhost:7860/api")
MAX_TOOL_OUTPUT = 4_000


async def close_sandbox() -> None:
    pass


class SandboxSession:
    """Routes tool calls to the API server's execute endpoint."""

    def __init__(self, stack: AsyncExitStack, tunnel_manager=None, registry=None, session_id: str = "", user_message: str = ""):
        self._stack = stack
        self._registry = registry
        self._session_id = session_id
        self._user_message = user_message
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = await self._stack.enter_async_context(
                httpx.AsyncClient(base_url=API_SERVER, timeout=30.0)
            )
        return self._client

    async def call(self, func_name: str, **kwargs) -> tuple[str, list[dict]]:
        """Execute a tool — direct call if bundled API available, else HTTP."""
        try:
            from main_api import execute_tool, ExecuteRequest
            req = ExecuteRequest(
                args=kwargs,
                session_id=self._session_id or "default",
                user_message=self._user_message or "",
            )
            data = await execute_tool(func_name, req)
            if isinstance(data, dict):
                effects = data.get("effects", [])
                if "error" in data:
                    return f"error: {data['error']}", effects
                result = data.get("result", "(no output)")
                if len(result) > MAX_TOOL_OUTPUT:
                    result = result[:MAX_TOOL_OUTPUT] + f"\n…[truncated {len(result) - MAX_TOOL_OUTPUT} chars]"
                return result, effects
            return str(data), []
        except ImportError:
            pass

        client = await self._get_client()
        try:
            payload: dict = {"args": kwargs}
            if self._session_id:
                payload["session_id"] = self._session_id
            if self._user_message:
                payload["user_message"] = self._user_message
            resp = await client.post(
                f"/packages/{func_name}/execute",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            effects = data.get("effects", [])
            if "error" in data:
                return f"error: {data['error']}", effects
            result = data.get("result", "(no output)")
            if len(result) > MAX_TOOL_OUTPUT:
                result = result[:MAX_TOOL_OUTPUT] + f"\n…[truncated {len(result) - MAX_TOOL_OUTPUT} chars]"
            return result, effects
        except httpx.HTTPError as e:
            return f"error: API call failed: {e}", []

    async def call_signed(self, func_name: str, **kwargs) -> tuple[str, ToolExecutionRecord]:
        result, effects = await self.call(func_name, **kwargs)

        if effects and self._registry:
            tool_defn = self._registry.tools_by_name.get(func_name)
            if tool_defn:
                allowed, violations = check_effects(tool_defn, effects)
                if not allowed:
                    result = f"error: blocked — undeclared effects: {'; '.join(violations)}"

        canonical = canonicalize_args(kwargs)
        signature = sign_execution(func_name, canonical, result)
        record = ToolExecutionRecord(
            tool_name=func_name,
            tool_input=kwargs,
            tool_output=result,
            signature=signature,
        )
        return result, record


def _args_model(json_schema: dict, name: str) -> type[BaseModel]:
    props = json_schema.get("properties", {})
    required = set(json_schema.get("required", []))
    fields = {}
    type_map = {"string": str, "integer": int, "number": float, "boolean": bool}
    for prop_name, prop in props.items():
        py_type = type_map.get(prop.get("type", "string"), str)
        desc = prop.get("description", "")
        if prop_name in required and "default" not in prop:
            fields[prop_name] = (py_type, Field(..., description=desc))
        else:
            fields[prop_name] = (py_type, Field(default=prop.get("default"), description=desc))
    return create_model(f"{name}_args", **fields)


def inspect_tools(
    registry=None,
    trusted_keys: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    """Run pre-execution capability and signature inspection on all tools."""
    all_tools = registry.all_tools if registry else []
    results: dict[str, list[str]] = {}
    for defn in all_tools:
        warnings = check_capabilities(defn)

        if defn.signature and defn.source_code:
            defn_dict = {
                "name": defn.name,
                "description": defn.description,
                "inputSchema": defn.inputSchema,
                "kind": defn.kind,
                "placement": defn.placement,
                "capabilities": [c.model_dump() for c in defn.capabilities],
                "requires_confirmation": defn.requires_confirmation,
                "depends_on": defn.depends_on,
            }
            if defn.tunnel:
                defn_dict["tunnel"] = {
                    "base_url": defn.tunnel.base_url,
                    "allowed_hosts": defn.tunnel.allowed_hosts,
                }
            if verify_package(defn_dict, defn.source_code, defn.signature, trusted_keys):
                log.info("tool %s: signature verified (publisher: %s)",
                         defn.name, defn.signature.publisher)
            else:
                warnings.append(
                    f"SIGNATURE INVALID — publisher '{defn.signature.publisher}', "
                    f"hash {defn.signature.payload_hash[:16]}..."
                )
        elif defn.signature and not defn.source_code:
            warnings.append("signature present but source_code missing — cannot verify")

        if warnings:
            results[defn.name] = warnings
            for w in warnings:
                log.warning("tool %s: %s", defn.name, w)
        else:
            log.info("tool %s: capabilities OK (%s)", defn.name, audit_capabilities(defn))
    return results


def make_tools(
    sandbox: SandboxSession,
    confirm_callback: Callable[[str, dict], bool] | None = None,
    registry=None,
) -> list[BaseTool]:
    """Create LangChain tools that route calls to the API server."""
    all_tools = registry.all_tools if registry else []
    tools: list[BaseTool] = []
    for defn in all_tools:
        if defn.kind == "gate":
            continue

        async def _run(
            sandbox=sandbox,
            func_name=defn.name,
            _defn=defn,
            **kwargs,
        ) -> tuple[str, ToolExecutionRecord]:
            if _defn.requires_confirmation and confirm_callback:
                if not confirm_callback(func_name, kwargs):
                    return f"error: user denied {func_name}", ToolExecutionRecord(
                        tool_name=func_name,
                        tool_input=kwargs,
                        tool_output=f"error: user denied {func_name}",
                        signature="",
                    )
            return await sandbox.call_signed(func_name, **kwargs)

        tools.append(StructuredTool.from_function(
            coroutine=_run,
            name=defn.name,
            description=defn.description,
            args_schema=_args_model(defn.inputSchema, defn.name),
            response_format="content_and_artifact",
        ))
    return tools
