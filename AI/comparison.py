"""Raw LLM + tools baseline — same model, same tools, no harness guardrails.

The baseline uses the SAME model and SAME tools as the agent but without
the harness pipeline (no faithfulness checks, no relevance gates, no
interstitial checks, no signing). This tests whether the harness adds
value over a naked tool-calling LLM.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

log = logging.getLogger("comparison")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
MAX_TOOL_ROUNDS = 3


@dataclass
class BaselineResult:
    model: str
    response_text: str
    latency_ms: float
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "response_text": self.response_text,
            "latency_ms": round(self.latency_ms, 1),
            "error": self.error,
        }


def _build_tool_schemas(registry) -> list[dict]:
    """Convert registry tool definitions to OpenAI function-calling format."""
    if not registry:
        return []
    schemas = []
    for defn in registry.all_tools:
        if defn.kind == "gate":
            continue
        schemas.append({
            "type": "function",
            "function": {
                "name": defn.name,
                "description": defn.description,
                "parameters": defn.inputSchema,
            },
        })
    return schemas


def _exec_tool(func_name: str, args: dict, registry) -> str:
    """Execute a tool from registry source code (same as sandbox direct path)."""
    defn = registry.tools_by_name.get(func_name)
    if not defn or not defn.source_code:
        return f"error: tool '{func_name}' not found"
    try:
        namespace: dict = {}
        exec(compile(defn.source_code, f"<baseline:{func_name}>", "exec"), namespace)  # noqa: S102
        fn = namespace.get(func_name)
        if fn is None:
            available = [k for k in namespace if callable(namespace[k]) and not k.startswith("_")]
            fn = namespace[available[0]] if len(available) == 1 else None
        if fn is None:
            return f"error: no callable found in '{func_name}'"
        return str(fn(**args))
    except Exception as e:
        return f"error: {e}"


async def run_inline_baseline(
    prompt: str, agent_model: str, registry=None,
) -> BaselineResult:
    """Same model + same tools, raw tool-calling loop, no harness."""
    import json

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    tools = _build_tool_schemas(registry)
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided tools to answer questions."},
        {"role": "user", "content": prompt},
    ]

    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            for _ in range(MAX_TOOL_ROUNDS):
                body: dict[str, Any] = {
                    "model": agent_model,
                    "messages": messages,
                    "temperature": 0.2,
                }
                if tools:
                    body["tools"] = tools

                resp = await client.post(
                    f"{LLM_BASE_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                    json=body,
                )
                if resp.status_code == 429:
                    latency = (time.monotonic() - start) * 1000
                    return BaselineResult(model=agent_model, response_text="", latency_ms=latency, error="rate limited")
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                msg = choice["message"]

                tool_calls = msg.get("tool_calls")
                if not tool_calls:
                    text = msg.get("content", "")
                    latency = (time.monotonic() - start) * 1000
                    return BaselineResult(model=agent_model, response_text=text, latency_ms=latency)

                messages.append(msg)
                for tc in tool_calls:
                    fn_name = tc["function"]["name"]
                    try:
                        fn_args = json.loads(tc["function"]["arguments"])
                    except (json.JSONDecodeError, KeyError):
                        fn_args = {}
                    result = _exec_tool(fn_name, fn_args, registry)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })

            latency = (time.monotonic() - start) * 1000
            return BaselineResult(model=agent_model, response_text="(max rounds reached)", latency_ms=latency)

    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        return BaselineResult(model=agent_model, response_text="", latency_ms=latency, error=str(e))


async def run_all_baselines(prompt: str, registry=None) -> list[BaselineResult]:
    """Placeholder — extend with multiple models if needed."""
    return []
