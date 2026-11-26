"""LangChain side: builds the chain, runs a tool-calling loop, and yields text deltas.

Tools execute inside Pydantic Monty — a sandboxed Python interpreter written in
Rust, running in a separate worker process. Each tool defined in AI/tools/ is
exposed to the model as a LangChain tool, but executes inside the sandbox.

pip install langchain-openai pydantic-monty
"""

import json
import logging
import os
import random
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack

log = logging.getLogger("agent")

from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from AI.context import ContextManager
from AI.embeddings import semantic_similarity
from AI.evaluator import VerificationError, verify_tool_execution
from AI.faithfulness import check_faithfulness, check_relevance
from AI.models.message import InterstitialCheck, ToolExecution, ToolExecutionRecord
from AI.sandbox import SandboxSession, close_sandbox, make_tools

MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3.5-9B")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.together.ai/v1")
API_KEY = os.getenv("TOGETHER_API_KEY", "ollama")

MAX_TOOL_ROUNDS = 5                           # stops runaway call loops on small models

SYSTEM_PROMPT = (
    "You are a tool-first assistant. Every response MUST begin with a tool call. "
    "Interpret every user message as an instruction to call the most relevant tool.\n\n"
    "IMPORTANT: When the user asks about a country, region, or large area, call the tool "
    "multiple times for different major cities/locations in that area. For example, "
    "'weather in Italy' should check Rome, Milan, Naples, etc. — not just one city. "
    "You may make several tool calls before answering.\n\n"
    "After receiving tool results, write a helpful natural-language answer that:\n"
    "- Presents the tool data in a clear, conversational way\n"
    "- Uses all relevant values from the tool output (numbers, names, conditions)\n"
    "- May rephrase, reorder, or add light commentary like 'quite warm' or 'you might want an umbrella'\n"
    "- NEVER adds factual claims the tool did not provide (no Wikipedia facts, no history, no definitions)\n"
    "- NEVER invents data points the tool did not return"
)

MAX_FAITHFULNESS_RETRIES = 2


# ── Model ────────────────────────────────────────────────────────────────────

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=MODEL,
            base_url=BASE_URL,
            api_key=os.getenv("TOGETHER_API_KEY", "ollama"),
            temperature=0.2,
            timeout=120,
            max_retries=1,
        )
    return _llm


def _flatten(content) -> str:
    """Normalise AIMessageChunk.content, which is str *or* a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _check_interstitial(
    prev_executions: list[ToolExecution], reasoning: str,
) -> InterstitialCheck:
    outputs = [t.tool_output for t in prev_executions if t.tool_output]
    if not outputs:
        return InterstitialCheck(semantic_score=1.0, faithful=True)
    combined_output = " | ".join(outputs)
    score = semantic_similarity(combined_output, reasoning)
    carried: list[str] = []
    for t in prev_executions:
        for token in t.tool_output.split():
            if len(token) >= 3 and token in reasoning:
                carried.append(token)
                break
    faithful = score >= 0.25 or len(carried) > 0
    detail = ""
    if not faithful:
        detail = "reasoning does not reflect preceding tool outputs"
    return InterstitialCheck(
        semantic_score=score, values_carried=carried, faithful=faithful, detail=detail,
    )


async def _run_tool_call(call: dict, tools_by_name: dict[str, BaseTool]) -> ToolMessage:
    """Execute one parsed tool call, verify HMAC, return ToolMessage."""
    t = tools_by_name.get(call["name"])
    if t is None:
        return ToolMessage(f"error: unknown tool '{call['name']}'", tool_call_id=call["id"])
    try:
        result = await t.ainvoke(call)
        record = getattr(result, "artifact", None)
        if record is not None:
            verify_tool_execution(record)
        return result
    except VerificationError as e:
        return ToolMessage(f"error: verification failed: {e}", tool_call_id=call["id"])
    except Exception as e:  # noqa: BLE001
        return ToolMessage(f"error: {e}", tool_call_id=call["id"])


async def stream_reply(
    message: str,
    system: str = SYSTEM_PROMPT,
    history: list[BaseMessage] | None = None,
    on_tool: Callable[[str, dict, str], None] | None = None,
    turn_messages: list[BaseMessage] | None = None,
    enforce_faithfulness: bool = True,
    session_id: str | None = None,
    registry=None,
) -> AsyncIterator[dict]:
    """Yield plain text deltas, running sandboxed tool calls between model turns.

    on_tool(name, args, result) is called after each tool runs — handy for
    logging or showing a "🔧 running code…" line in the UI.

    If turn_messages is provided, this turn's new messages (human + AI + tool)
    are appended to it so the caller can persist them.
    """
    managed_history = history or []
    if session_id and managed_history:
        ctx = ContextManager(session_id)
        managed_history = await ctx.prepare(managed_history, message)

    messages: list[BaseMessage] = [SystemMessage(system), *managed_history, HumanMessage(message)]
    new_start = 1 + len(managed_history)
    all_records: list[ToolExecutionRecord] = []
    prev_executions: list[ToolExecution] = []

    def _export():
        if turn_messages is not None:
            turn_messages.extend(messages[new_start:])

    async with AsyncExitStack() as stack:
        sandbox = SandboxSession(stack, registry=registry)
        tools = make_tools(sandbox, registry=registry)
        tools_by_name = {t.name: t for t in tools}
        has_tools = bool(tools)
        llm = _get_llm()
        llm_forced = llm.bind_tools(tools, tool_choice="required") if has_tools else llm.bind_tools(tools)
        llm_auto = llm.bind_tools(tools)
        grounded = not has_tools

        faithfulness_retries = 0
        for _round_num in range(1, MAX_TOOL_ROUNDS + 1):
            llm = llm_auto if grounded else llm_forced
            full: AIMessageChunk | None = None
            buffered_tokens: list[str] = []

            async for chunk in llm.astream(messages):
                full = chunk if full is None else full + chunk

                if grounded:
                    text = _flatten(chunk.content)
                    if not text:
                        text = chunk.additional_kwargs.get("reasoning_content", "")
                    if text:
                        buffered_tokens.append(text)

            if full is None:
                _export()
                return

            bad = [
                ToolMessage(f"error: could not parse arguments: {c.get('error')}", tool_call_id=c["id"])
                for c in full.invalid_tool_calls if c.get("id")
            ]

            # ── Final response (no tool calls) ──────────────────────────
            if not full.tool_calls and not bad:
                if not grounded:
                    messages.append(full)
                    messages.append(HumanMessage(
                        "You MUST use at least one tool before answering. "
                        "Do not answer from your own knowledge — call a tool first."
                    ))
                    continue

                if all_records:
                    ai_text = _flatten(full.content)
                    faith_by_id = check_faithfulness(ai_text, all_records)
                    if faith_by_id:
                        combined = next(iter(faith_by_id.values()))
                        if len(faith_by_id) > 1:
                            scores = [r.semantic_score for r in faith_by_id.values()]
                            all_contained = all(r.contained for r in faith_by_id.values())
                            all_mismatches = [m for r in faith_by_id.values() for m in r.mismatches]
                            from AI.models.message import FaithfulnessResult
                            combined = FaithfulnessResult(
                                semantic_score=sum(scores) / len(scores),
                                contained=all_contained,
                                faithful=all_contained,
                                mismatches=all_mismatches,
                            )
                        full.additional_kwargs["faithfulness"] = combined.model_dump()

                    unfaithful = any(not r.faithful for r in faith_by_id.values())
                    if unfaithful and enforce_faithfulness:
                        faithfulness_retries += 1
                        if faithfulness_retries > MAX_FAITHFULNESS_RETRIES:
                            log.warning("faithfulness retries exhausted — blocking response")
                            deflection_prompt = [
                                SystemMessage(
                                    "The tool returned results but you failed to summarise them accurately. "
                                    "Generate a brief, polite message explaining you got data but couldn't "
                                    "summarise it reliably, and suggest the user try rephrasing. "
                                    "Be natural and conversational. Do NOT include any tool data in your response."
                                ),
                                HumanMessage(message),
                            ]
                            deflection_response = await _get_llm().ainvoke(deflection_prompt)
                            deflection_text = _flatten(deflection_response.content) or "I got results but couldn't summarise them accurately. Could you try rephrasing?"
                            yield {"type": "tool_blocked", "reason": deflection_text}
                            _export()
                            return
                        mismatches = [m for r in faith_by_id.values() for m in r.mismatches]
                        log.info("faithfulness retry %d/%d: %s", faithfulness_retries, MAX_FAITHFULNESS_RETRIES, mismatches)
                        messages.append(full)
                        messages.append(HumanMessage(
                            "Your response must be based ONLY on the tool output above. "
                            "Do not add any information from your own knowledge. "
                            "Do not add context, definitions, or explanations the tool did not provide. "
                            f"Mismatches found: {mismatches}. Rewrite using only tool data."
                        ))
                        buffered_tokens.clear()
                        continue

                for tok in buffered_tokens:
                    yield {"type": "token", "token": tok}
                messages.append(full)
                _export()
                return

            # ── Intermediate round (has tool calls) ─────────────────────
            reasoning = _flatten(full.content)

            messages.append(full)
            messages.extend(bad)

            # Pre-execution relevance gate: block before calling the API
            tool_descs = {d.name: d.description for d in (registry.all_tools if registry else [])}
            selected_tools = [call["name"] for call in full.tool_calls]
            relevant, rel_mismatches = check_relevance(message, selected_tools, tool_descs)
            log.info("pre-exec relevance: relevant=%s, mismatches=%s", relevant, rel_mismatches)
            if not relevant:
                tool_names = [d.name for d in (registry.all_tools if registry else [])]
                tool_descs = [f"{d.name}: {d.description}" for d in (registry.all_tools if registry else [])]
                tool_info = "; ".join(tool_descs) if tool_descs else "none currently loaded"
                deflection_prompt = [
                    SystemMessage(
                        "Generate a brief, polite deflection message. The user asked something "
                        "your tools cannot help with. Mention what tools you DO have and what "
                        "they can do. Be natural and conversational — vary your phrasing. "
                        "Do NOT answer the user's question. Do NOT use your own knowledge. "
                        "Just explain you can't help with that and say what you can help with.\n\n"
                        f"Your available tools: {tool_info}"
                    ),
                    HumanMessage(message),
                ]
                deflection_response = await _get_llm().ainvoke(deflection_prompt)
                deflection_text = _flatten(deflection_response.content) or "Sorry, that's outside what I can help with."
                yield {"type": "tool_blocked", "reason": deflection_text}
                _export()
                return

            round_records: list[ToolExecutionRecord] = []
            round_executions: list[ToolExecution] = []
            for call in full.tool_calls:
                yield {"type": "tool_call", "tool": call["name"], "args": call["args"]}
                result = await _run_tool_call(call, tools_by_name)
                record = getattr(result, "artifact", None)
                if isinstance(record, ToolExecutionRecord):
                    record.tool_call_id = call.get("id")
                    all_records.append(record)
                    round_records.append(record)
                    round_executions.append(ToolExecution(
                        tool_name=record.tool_name,
                        tool_call_id=record.tool_call_id,
                        tool_input=record.tool_input,
                        tool_output=record.tool_output,
                        signature=record.signature,
                    ))
                yield {"type": "tool_result", "tool": call["name"], "result": str(result.content)[:2000]}
                if on_tool:
                    on_tool(call["name"], call["args"], str(result.content))
                messages.append(result)

            if round_records:
                grounded = True

            full.additional_kwargs["tool_executions"] = [
                e.model_dump() for e in round_executions
            ]

            # Interstitial: does this reasoning reflect the previous round's outputs?
            if prev_executions and reasoning:
                interstitial = _check_interstitial(prev_executions, reasoning)
                full.additional_kwargs["interstitial"] = interstitial.model_dump()

            prev_executions = round_executions

        tool_descs = [f"{d.name}: {d.description}" for d in (registry.all_tools if registry else [])]
        tool_info = "; ".join(tool_descs) if tool_descs else "none currently loaded"
        deflection_prompt = [
            SystemMessage(
                "You tried multiple times but couldn't resolve the user's request. "
                "Generate a brief, polite message explaining you weren't able to help "
                "and suggest they try rephrasing. Mention what tools you have. "
                "Be natural and conversational. Do NOT answer the user's question.\n\n"
                f"Your available tools: {tool_info}"
            ),
            HumanMessage(message),
        ]
        try:
            deflection_response = await _get_llm().ainvoke(deflection_prompt)
            deflection_text = _flatten(deflection_response.content) or "I wasn't able to help with that. Could you try rephrasing?"
        except Exception:
            deflection_text = "I wasn't able to help with that. Could you try rephrasing?"
        yield {"type": "tool_blocked", "reason": deflection_text}
        _export()


# ── Quick manual test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio

    async def _main():
        def log(name, args, result):
            print(f"\n  🔧 {name}({json.dumps(args)}) → {result}\n")

        try:
            async for delta in stream_reply(
                "What's 17.5% of 2,340, and what time is it in UTC?", on_tool=log
            ):
                print(delta, end="", flush=True)
            print()
        finally:
            await close_sandbox()

    asyncio.run(_main())
