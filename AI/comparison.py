"""Run the same prompt through raw LLM baselines (no tools, no agent pipeline)
to compare against the tool-grounded agent response.

The inline baseline uses the SAME model the agent uses but without
the tool pipeline — just a direct question. This shows whether the agent's
tool grounding adds value over raw parametric knowledge.

A background sweep across all available models runs asynchronously and
attaches results when they're ready.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

log = logging.getLogger("comparison")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
BASELINE_SYSTEM = (
    "Answer the user's question directly and concisely. "
    "If you don't know the answer, say so."
)


@dataclass
class BaselineResult:
    model: str
    response_text: str
    latency_ms: float
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "response_text": self.response_text[:500],
            "latency_ms": round(self.latency_ms, 1),
            "error": self.error,
        }


async def _query_model(model: str, prompt: str, timeout: int = 30) -> BaselineResult:
    """Query a single model via OpenAI-compatible endpoint."""
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{LLM_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": BASELINE_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            latency = (time.monotonic() - start) * 1000
            return BaselineResult(model=model, response_text=text, latency_ms=latency)
    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        return BaselineResult(model=model, response_text="", latency_ms=latency, error=str(e))


async def run_inline_baseline(prompt: str, agent_model: str) -> BaselineResult:
    """Run one fast baseline using the agent's own model (already loaded in Ollama).

    This is the inline comparison — same model, no tools. Fast because
    the model weights are already resident.
    """
    return await _query_model(agent_model, prompt, timeout=30)


async def _list_models() -> list[str]:
    """Get available models. Returns empty if no local Ollama running."""
    ollama_base = os.getenv("OLLAMA_BASE", "http://localhost:11434")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{ollama_base}/api/tags")
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


async def run_all_baselines(prompt: str) -> list[BaselineResult]:
    """Run the prompt through ALL available models in parallel.

    This can be slow (Ollama serialises inference across models), so it
    should only be used as a background task, not inline.
    """
    models = await _list_models()
    if not models:
        return []

    tasks = [_query_model(m, prompt, timeout=120) for m in models]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    out: list[BaselineResult] = []
    for r in results:
        if isinstance(r, BaselineResult):
            out.append(r)
        elif isinstance(r, Exception):
            log.warning("baseline query failed: %s", r)
    return out
