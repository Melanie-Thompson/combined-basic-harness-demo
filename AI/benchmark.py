"""Benchmark suite — end-to-end evaluation of the agent pipeline.

Pass criteria for tool-calling cases (per run):
1. Request was not falsely blocked
2. Expected tool trajectory executed (multi-tool aware)
3. Extracted tool arguments satisfy expected_args constraints
4. Tool execution succeeded without MCP protocol errors or isError=True
5. Final response text is non-empty and passes check_faithfulness() (per-tool extractors + semantic fallback)
6. Final response contains all expected ground-truth values

Pass criteria for blocked cases (per run):
1. Request was blocked before answering from ungrounded parametric knowledge
2. Zero tools were executed

Consistency metrics (across k runs per test case):
- pass@1: mean single-run pass rate across all k trials
- pass^k: fraction of test cases that passed on EVERY trial (1..k)
"""

from contextlib import contextmanager
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from AI.faithfulness import SEMANTIC_THRESHOLD, check_faithfulness
from AI.models.message import ToolExecutionRecord

log = logging.getLogger("benchmark")

FAITHFULNESS_THRESHOLD = SEMANTIC_THRESHOLD
DEFAULT_K_RUNS = 3


@dataclass
class ToolCallRecord:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    output: str = ""
    is_error: bool = False
    error_message: str = ""


@dataclass
class TestCase:
    id: str
    prompt: str
    expected_tools: list[str] = field(default_factory=list)
    expected_blocked: bool = False
    expected_args: dict[str, dict[str, Any]] = field(default_factory=dict)
    expected_values: list[str] = field(default_factory=list)
    mock_fixtures: dict[str, str | Callable[[dict[str, Any]], str]] = field(default_factory=dict)
    history: list[dict[str, str]] = field(default_factory=list)
    category: str = "general"


@dataclass
class TestResult:
    """Backwards-compatible per-run result model."""

    case_id: str
    prompt: str
    expected_tool: str | None
    expected_blocked: bool
    actual_tool: str | None
    actual_blocked: bool
    response_text: str
    run_index: int = 1
    actual_tools: list[str] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    tool_output: str = ""
    tool_args: dict | None = None
    values_found: list[str] = field(default_factory=list)
    values_missing: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    passed: bool = False
    failure_reasons: list[str] = field(default_factory=list)
    faithfulness_score: float = 0.0
    faithful: bool = False
    faithfulness_mismatches: list[str] = field(default_factory=list)
    had_mcp_error: bool = False
    baselines: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "run_index": self.run_index,
            "prompt": self.prompt,
            "expected_tool": self.expected_tool,
            "expected_tools": self.actual_tools if not self.expected_tool else [self.expected_tool],
            "expected_blocked": self.expected_blocked,
            "actual_tool": self.actual_tool,
            "actual_tools": self.actual_tools,
            "actual_blocked": self.actual_blocked,
            "tool_calls": [
                {
                    "tool": tc.tool,
                    "args": tc.args,
                    "output": tc.output[:300],
                    "is_error": tc.is_error,
                    "error_message": tc.error_message,
                }
                for tc in self.tool_calls
            ],
            "response_text": self.response_text[:500],
            "tool_output": self.tool_output[:300],
            "tool_args": self.tool_args,
            "values_found": self.values_found,
            "values_missing": self.values_missing,
            "latency_ms": round(self.latency_ms, 1),
            "passed": self.passed,
            "failure_reason": "; ".join(self.failure_reasons),
            "faithfulness_score": round(self.faithfulness_score, 3),
            "faithful": self.faithful,
            "faithfulness_mismatches": self.faithfulness_mismatches,
            "had_mcp_error": self.had_mcp_error,
            "baselines": self.baselines,
        }


# Alias for multi-run clarity
RunResult = TestResult


@dataclass
class CaseSummary:
    case_id: str
    prompt: str
    category: str
    expected_tools: list[str]
    expected_blocked: bool
    k: int
    runs: list[TestResult] = field(default_factory=list)

    @property
    def pass_at_1(self) -> float:
        return sum(1 for r in self.runs if r.passed) / len(self.runs) if self.runs else 0.0

    @property
    def pass_pow_k(self) -> bool:
        return bool(self.runs) and all(r.passed for r in self.runs)

    @property
    def mean_latency_ms(self) -> float:
        return sum(r.latency_ms for r in self.runs) / len(self.runs) if self.runs else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "prompt": self.prompt,
            "category": self.category,
            "expected_tools": self.expected_tools,
            "expected_blocked": self.expected_blocked,
            "k": self.k,
            "pass_at_1": round(self.pass_at_1, 3),
            "pass_pow_k": self.pass_pow_k,
            "mean_latency_ms": round(self.mean_latency_ms, 1),
            "runs": [r.to_dict() for r in self.runs],
        }


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(len(s) * 0.95)
    return s[min(idx, len(s) - 1)]


@contextmanager
def _apply_tool_fixtures(registry: Any, fixtures: dict[str, str | Callable[[dict[str, Any]], str]]):
    """Temporarily intercept tool execution in the registry with deterministic fixtures."""
    if not registry or not fixtures:
        yield
        return

    originals: dict[str, Any] = {}
    tool_map = {t.name: t for t in getattr(registry, "all_tools", [])}

    for tool_name, fixture in fixtures.items():
        defn = tool_map.get(tool_name)
        if not defn:
            continue

        for attr in ("handler", "fn", "execute"):
            if hasattr(defn, attr) and callable(getattr(defn, attr)):
                originals[f"{tool_name}.{attr}"] = (defn, attr, getattr(defn, attr))

                async def _mock_async(args: dict[str, Any] | None = None, _f=fixture, **kwargs: Any) -> str:
                    payload = args if args is not None else kwargs
                    return _f(payload) if callable(_f) else _f

                try:
                    setattr(defn, attr, _mock_async)
                except Exception as exc:
                    log.debug("could not patch %s.%s: %s", tool_name, attr, exc)
                break

    try:
        yield
    finally:
        for _, (defn, attr, orig_fn) in originals.items():
            try:
                setattr(defn, attr, orig_fn)
            except Exception:
                pass


def _validate_expected_args(
    expected_args: dict[str, dict[str, Any]],
    tool_calls: list[ToolCallRecord],
) -> list[str]:
    """Verify that expected arguments match actual extracted tool arguments."""
    errors: list[str] = []
    calls_by_tool: dict[str, list[dict[str, Any]]] = {}
    for tc in tool_calls:
        calls_by_tool.setdefault(tc.tool, []).append(tc.args or {})

    for tool_name, req_args in expected_args.items():
        invocations = calls_by_tool.get(tool_name, [])
        if not invocations:
            errors.append(f"missing args check: tool '{tool_name}' was never called")
            continue

        matched_any = False
        last_mismatch = ""
        for actual_args in invocations:
            mismatches = []
            for key, expected_val in req_args.items():
                if key not in actual_args:
                    mismatches.append(f"missing arg '{key}'")
                    continue
                actual_val = actual_args[key]
                if callable(expected_val):
                    if not expected_val(actual_val):
                        mismatches.append(f"arg '{key}'={actual_val!r} failed predicate")
                elif isinstance(expected_val, str) and isinstance(actual_val, str):
                    if expected_val.lower() != actual_val.lower():
                        mismatches.append(f"arg '{key}': expected {expected_val!r}, got {actual_val!r}")
                elif actual_val != expected_val:
                    mismatches.append(f"arg '{key}': expected {expected_val!r}, got {actual_val!r}")
            if not mismatches:
                matched_any = True
                break
            last_mismatch = ", ".join(mismatches)

        if not matched_any:
            errors.append(f"invalid args for '{tool_name}': {last_mismatch}")

    return errors


def _extract_mcp_error(evt: dict[str, Any]) -> tuple[bool, str]:
    """Detect both MCP JSON-RPC protocol errors and CallToolResult isError=True payloads."""
    if evt.get("type") == "error":
        return True, str(evt.get("error") or evt.get("message") or "stream error")

    if evt.get("isError") is True or evt.get("is_error") is True:
        return True, str(evt.get("result") or evt.get("content") or "MCP tool returned isError=True")

    raw_result = evt.get("result")
    if isinstance(raw_result, dict):
        if raw_result.get("isError") is True:
            return True, str(raw_result.get("content") or "MCP CallToolResult isError=True")
        if "error" in raw_result:
            return True, str(raw_result["error"])
    elif isinstance(raw_result, str):
        stripped = raw_result.strip()
        if stripped.startswith("{") and '"isError"' in stripped:
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict) and parsed.get("isError") is True:
                    return True, str(parsed.get("content") or stripped)
            except json.JSONDecodeError:
                pass

    return False, ""


@dataclass
class BenchmarkReport:
    k_runs: int = DEFAULT_K_RUNS
    total_cases: int = 0
    total_runs: int = 0
    passed_runs: int = 0
    failed_runs: int = 0
    consistent_cases: int = 0
    false_blocks: int = 0
    false_passes: int = 0
    unblocked_routed_runs: int = 0
    tool_trajectory_correct: int = 0
    mcp_errors: int = 0
    faithful_runs: int = 0
    faithful_denominator: int = 0
    case_summaries: list[CaseSummary] = field(default_factory=list)

    # --- Backwards-compatible aliases for existing UI & API routes ---
    @property
    def total(self) -> int:
        return self.total_runs

    @property
    def passed(self) -> int:
        return self.passed_runs

    @property
    def failed(self) -> int:
        return self.failed_runs

    @property
    def pass_rate(self) -> float:
        return self.pass_at_1

    @property
    def tool_accuracy(self) -> float:
        return self.routing_accuracy

    @property
    def results(self) -> list[TestResult]:
        return [r for cs in self.case_summaries for r in cs.runs]

    # --- Core multi-run & trajectory properties ---
    @property
    def _all_runs(self) -> list[tuple[CaseSummary, TestResult]]:
        return [(cs, r) for cs in self.case_summaries for r in cs.runs]

    @property
    def _routed_runs(self) -> list[TestResult]:
        return [r for cs, r in self._all_runs if not cs.expected_blocked]

    @property
    def _blocked_runs(self) -> list[TestResult]:
        return [r for cs, r in self._all_runs if cs.expected_blocked]

    @property
    def pass_at_1(self) -> float:
        return self.passed_runs / self.total_runs if self.total_runs else 0.0

    @property
    def pass_pow_k(self) -> float:
        return self.consistent_cases / self.total_cases if self.total_cases else 0.0

    @property
    def false_block_rate(self) -> float:
        routed = self._routed_runs
        return self.false_blocks / len(routed) if routed else 0.0

    @property
    def false_pass_rate(self) -> float:
        blocked = self._blocked_runs
        return self.false_passes / len(blocked) if blocked else 0.0

    @property
    def routing_accuracy(self) -> float:
        return (
            self.tool_trajectory_correct / self.unblocked_routed_runs
            if self.unblocked_routed_runs
            else 0.0
        )

    @property
    def faithfulness_rate(self) -> float:
        return (
            self.faithful_runs / self.faithful_denominator
            if self.faithful_denominator
            else 0.0
        )

    @property
    def mean_faithfulness_score(self) -> float:
        scores = [
            r.faithfulness_score
            for r in self._routed_runs
            if (any(tc.output.strip() and not tc.is_error for tc in r.tool_calls)
                or (r.tool_output and r.tool_output.strip()))
        ]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def routed_latency_ms(self) -> float:
        vals = [r.latency_ms for r in self._routed_runs]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def blocked_latency_ms(self) -> float:
        vals = [r.latency_ms for r in self._blocked_runs]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def routed_p95_ms(self) -> float:
        return _p95([r.latency_ms for r in self._routed_runs])

    @property
    def blocked_p95_ms(self) -> float:
        return _p95([r.latency_ms for r in self._blocked_runs])

    @property
    def per_tool(self) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[CaseSummary]] = {}
        for cs in self.case_summaries:
            key = "->".join(cs.expected_tools) if cs.expected_tools else "_irrelevant"
            grouped.setdefault(key, []).append(cs)

        out: dict[str, dict[str, Any]] = {}
        for key, summaries in grouped.items():
            runs = [r for cs in summaries for r in cs.runs]
            latencies = [r.latency_ms for r in runs]
            faith_scores = [
                r.faithfulness_score
                for r in runs
                if (any(tc.output.strip() and not tc.is_error for tc in r.tool_calls)
                    or (r.tool_output and r.tool_output.strip()))
            ]
            passed_count = sum(1 for r in runs if r.passed)
            out[key] = {
                "cases": len(summaries),
                "total": len(runs),
                "passed": passed_count,
                "failed": len(runs) - passed_count,
                "pass_rate": round(passed_count / len(runs), 3) if runs else 0.0,
                "pass_at_1": round(passed_count / len(runs), 3) if runs else 0.0,
                "pass_pow_k": round(sum(1 for cs in summaries if cs.pass_pow_k) / len(summaries), 3) if summaries else 0.0,
                "mean_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
                "p95_latency_ms": round(_p95(latencies), 1),
                "mean_faithfulness": round(sum(faith_scores) / len(faith_scores), 3) if faith_scores else 0.0,
            }
        return out

    def record_result(self, result: "TestResult", k: int = 1) -> "CaseSummary":
        """Record a single live or benchmark TestResult and update report aggregates."""
        summary = CaseSummary(
            case_id=result.case_id,
            prompt=result.prompt,
            category="live",
            expected_tools=[result.expected_tool] if result.expected_tool else [],
            expected_blocked=result.expected_blocked,
            k=k,
            runs=[result],
        )
        self.case_summaries.append(summary)
        self.total_cases += 1
        self.total_runs += 1

        if result.passed:
            self.passed_runs += 1
            self.consistent_cases += 1
        else:
            self.failed_runs += 1

        if result.had_mcp_error:
            self.mcp_errors += 1

        if result.expected_blocked:
            if not result.actual_blocked:
                self.false_passes += 1
        else:
            if result.actual_blocked:
                self.false_blocks += 1
            else:
                self.unblocked_routed_runs += 1
                expected_list = [result.expected_tool] if result.expected_tool else []
                actual_list = result.actual_tools or ([result.actual_tool] if result.actual_tool else [])
                if expected_list and actual_list == expected_list:
                    self.tool_trajectory_correct += 1

            has_valid_output = any(
                tc.output.strip() and not tc.is_error for tc in result.tool_calls
            ) or bool(result.tool_output and result.tool_output.strip())
            if has_valid_output:
                self.faithful_denominator += 1
                if result.faithful:
                    self.faithful_runs += 1

        return summary

    def to_dict(self) -> dict[str, Any]:
        return {
            # Legacy flat keys for existing frontend UI
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": round(self.pass_rate, 3),
            "tool_accuracy": round(self.tool_accuracy, 3),
            "results": [r.to_dict() for r in self.results],
            # New multi-run & trajectory keys
            "k_runs": self.k_runs,
            "total_cases": self.total_cases,
            "total_runs": self.total_runs,
            "passed_runs": self.passed_runs,
            "failed_runs": self.failed_runs,
            "pass_at_1": round(self.pass_at_1, 3),
            "pass_pow_k": round(self.pass_pow_k, 3),
            "false_block_rate": round(self.false_block_rate, 3),
            "false_pass_rate": round(self.false_pass_rate, 3),
            "routing_accuracy": round(self.routing_accuracy, 3),
            "mcp_errors": self.mcp_errors,
            "faithfulness_rate": round(self.faithfulness_rate, 3),
            "mean_faithfulness_score": round(self.mean_faithfulness_score, 3),
            "routed_latency_ms": round(self.routed_latency_ms, 1),
            "blocked_latency_ms": round(self.blocked_latency_ms, 1),
            "routed_p95_ms": round(self.routed_p95_ms, 1),
            "blocked_p95_ms": round(self.blocked_p95_ms, 1),
            "per_tool": self.per_tool,
            "cases": [cs.to_dict() for cs in self.case_summaries],
        }


def generate_test_cases(registry) -> list[TestCase]:
    """Generate deterministic test cases with mocked fixtures, arg checks, and multi-turn scenarios."""
    cases: list[TestCase] = []
    all_tools = registry.all_tools if registry else []
    available_names = {t.name for t in all_tools if getattr(t, "kind", "") != "gate"}

    if "weather_fetch" in available_names:
        cases.extend([
            TestCase(
                id="weather_natural_london",
                prompt="Do I need an umbrella in London today?",
                expected_tools=["weather_fetch"],
                expected_args={"weather_fetch": {"city": "London"}},
                expected_values=["11", "rain"],
                mock_fixtures={
                    "weather_fetch": '{"city": "London", "temp_c": 11, "condition": "Light rain", "humidity": 84}'
                },
                category="relevant:weather_fetch",
            ),
            TestCase(
                id="weather_natural_paris",
                prompt="What's it like outside in Paris right now?",
                expected_tools=["weather_fetch"],
                expected_args={"weather_fetch": {"city": "Paris"}},
                expected_values=["19", "clear"],
                mock_fixtures={
                    "weather_fetch": '{"city": "Paris", "temp_c": 19, "condition": "Clear skies", "humidity": 52}'
                },
                category="relevant:weather_fetch",
            ),
            TestCase(
                id="weather_natural_tokyo",
                prompt="Is it warm enough for shorts in Tokyo?",
                expected_tools=["weather_fetch"],
                expected_args={"weather_fetch": {"city": "Tokyo"}},
                expected_values=["27", "sunny"],
                mock_fixtures={
                    "weather_fetch": '{"city": "Tokyo", "temp_c": 27, "condition": "Sunny", "humidity": 60}'
                },
                category="relevant:weather_fetch",
            ),
            TestCase(
                id="weather_multiturn_followup",
                prompt="What about humidity there?",
                history=[
                    {"role": "user", "content": "Check the weather in Sydney."},
                    {"role": "assistant", "content": "It is 22°C and partly cloudy in Sydney."},
                ],
                expected_tools=["weather_fetch"],
                expected_args={"weather_fetch": {"city": "Sydney"}},
                expected_values=["71"],
                mock_fixtures={
                    "weather_fetch": '{"city": "Sydney", "temp_c": 22, "condition": "Partly cloudy", "humidity": 71}'
                },
                category="multiturn:weather_fetch",
            ),
        ])

    irrelevant = [
        ("irr_compliment", "Tell me I'm pretty"),
        ("irr_love", "But don't you love me?"),
        ("irr_history", "When was the Battle of Hastings?"),
        ("irr_math", "What is 2 + 2?"),
        ("irr_joke", "Tell me a joke about programming"),
        ("irr_recipe", "How do I make pasta carbonara?"),
        ("irr_philosophy", "What is the meaning of life?"),
    ]
    for case_id, prompt in irrelevant:
        cases.append(
            TestCase(
                id=case_id,
                prompt=prompt,
                expected_tools=[],
                expected_blocked=True,
                category="irrelevant",
            )
        )

    edge = [
        ("edge_climate", "Why is the climate changing?"),
        ("edge_duck", "What is a duck?"),
        ("edge_temperature", "What temperature does water boil at?"),
    ]
    for case_id, prompt in edge:
        cases.append(
            TestCase(
                id=case_id,
                prompt=prompt,
                expected_tools=[],
                expected_blocked=True,
                category="edge",
            )
        )

    return cases


async def _execute_single_run(
    stream_reply_fn,
    case: TestCase,
    run_index: int,
    registry: Any,
    session_id: str,
) -> TestResult:
    tool_calls: list[ToolCallRecord] = []
    actual_blocked = False
    response_text = ""
    stream_exception: str | None = None

    start = time.monotonic()
    with _apply_tool_fixtures(registry, case.mock_fixtures):
        try:
            async for evt in stream_reply_fn(
                case.prompt,
                history=list(case.history),
                session_id=f"{session_id}_{case.id}_r{run_index}",
                registry=registry,
            ):
                evt_type = evt.get("type", "")
                if evt_type == "tool_call":
                    tool_calls.append(
                        ToolCallRecord(
                            tool=evt.get("tool", ""),
                            args=evt.get("args") or {},
                        )
                    )
                elif evt_type == "tool_result":
                    is_err, err_msg = _extract_mcp_error(evt)
                    raw_res = evt.get("result", "")
                    res_text = raw_res if isinstance(raw_res, str) else json.dumps(raw_res)
                    if tool_calls:
                        tool_calls[-1].output = res_text
                        tool_calls[-1].is_error = is_err
                        tool_calls[-1].error_message = err_msg
                    else:
                        tool_calls.append(
                            ToolCallRecord(
                                tool=evt.get("tool", "unknown"),
                                output=res_text,
                                is_error=is_err,
                                error_message=err_msg,
                            )
                        )
                elif evt_type == "tool_blocked":
                    actual_blocked = True
                    response_text = evt.get("reason", "")
                elif evt_type == "token":
                    response_text += evt.get("token", "")
                elif evt_type == "error":
                    stream_exception = str(evt.get("error") or evt.get("message") or "unknown stream error")
        except Exception as exc:
            log.warning("benchmark case '%s' run %d raised exception: %s", case.id, run_index, exc)
            stream_exception = str(exc)

    latency_ms = (time.monotonic() - start) * 1000
    actual_tools = [tc.tool for tc in tool_calls]
    had_mcp_error = bool(stream_exception) or any(tc.is_error for tc in tool_calls)

    values_found: list[str] = []
    values_missing: list[str] = []
    for val in case.expected_values:
        if val.lower() in response_text.lower():
            values_found.append(val)
        else:
            values_missing.append(val)

    valid_records = [
        ToolExecutionRecord(
            tool_call_id=f"{case.id}_r{run_index}_{idx}",
            tool_name=tc.tool,
            tool_output=tc.output,
        )
        for idx, tc in enumerate(tool_calls)
        if tc.output.strip() and not tc.is_error
    ]

    faithfulness_score = 0.0
    is_faithful = False
    faithfulness_mismatches: list[str] = []

    if valid_records and not stream_exception:
        faith_map = check_faithfulness(response_text, valid_records)
        faithfulness_score = sum(r.semantic_score for r in faith_map.values()) / len(faith_map)
        is_faithful = all(r.faithful for r in faith_map.values())
        for res in faith_map.values():
            faithfulness_mismatches.extend(res.mismatches)

    failure_reasons: list[str] = []

    if stream_exception:
        failure_reasons.append(f"execution exception: {stream_exception}")

    if case.expected_blocked:
        if not actual_blocked:
            failure_reasons.append("should have been blocked")
        if actual_tools:
            failure_reasons.append(f"blocked case invoked tools: {actual_tools}")
    else:
        if actual_blocked:
            failure_reasons.append("incorrectly blocked")

        if not actual_blocked and case.expected_tools and actual_tools != case.expected_tools:
            failure_reasons.append(f"wrong tool trajectory: expected {case.expected_tools}, got {actual_tools}")

        if not actual_blocked and case.expected_args:
            failure_reasons.extend(_validate_expected_args(case.expected_args, tool_calls))

        if not actual_blocked:
            for tc in tool_calls:
                if tc.is_error:
                    failure_reasons.append(f"MCP tool error in '{tc.tool}': {tc.error_message}")
            if not valid_records and not any(tc.is_error for tc in tool_calls):
                failure_reasons.append("tool produced no valid output")

        if not actual_blocked and not response_text.strip():
            failure_reasons.append("empty final response")
        if valid_records and response_text.strip() and not is_faithful:
            failure_reasons.append(f"unfaithful: {', '.join(faithfulness_mismatches)}")

        if values_missing:
            failure_reasons.append(f"missing expected values: {values_missing}")

    combined_output = "\n".join(tc.output for tc in tool_calls if tc.output)

    return TestResult(
        case_id=case.id,
        run_index=run_index,
        prompt=case.prompt,
        expected_tool=case.expected_tools[0] if case.expected_tools else None,
        expected_blocked=case.expected_blocked,
        actual_tool=actual_tools[0] if actual_tools else None,
        actual_tools=actual_tools,
        tool_calls=tool_calls,
        actual_blocked=actual_blocked,
        response_text=response_text,
        tool_output=combined_output,
        tool_args=tool_calls[0].args if tool_calls else None,
        values_found=values_found,
        values_missing=values_missing,
        latency_ms=latency_ms,
        passed=len(failure_reasons) == 0,
        failure_reasons=failure_reasons,
        faithfulness_score=faithfulness_score,
        faithful=is_faithful,
        faithfulness_mismatches=faithfulness_mismatches,
        had_mcp_error=had_mcp_error,
    )


async def run_benchmark(
    stream_reply_fn,
    test_cases: list[TestCase] | None = None,
    registry=None,
    session_id: str = "benchmark",
    k_runs: int = DEFAULT_K_RUNS,
) -> BenchmarkReport:
    """Run the benchmark suite across k trials per test case and compute pass@1 & pass^k."""
    cases = test_cases or generate_test_cases(registry)
    report = BenchmarkReport(k_runs=k_runs, total_cases=len(cases))

    for case in cases:
        summary = CaseSummary(
            case_id=case.id,
            prompt=case.prompt,
            category=case.category,
            expected_tools=case.expected_tools,
            expected_blocked=case.expected_blocked,
            k=k_runs,
        )

        for run_idx in range(1, k_runs + 1):
            log.info("benchmark: running '%s' [trial %d/%d]", case.id, run_idx, k_runs)
            run_res = await _execute_single_run(
                stream_reply_fn=stream_reply_fn,
                case=case,
                run_index=run_idx,
                registry=registry,
                session_id=session_id,
            )
            summary.runs.append(run_res)
            report.total_runs += 1

            if run_res.passed:
                report.passed_runs += 1
            else:
                report.failed_runs += 1

            if run_res.had_mcp_error:
                report.mcp_errors += 1

            if case.expected_blocked:
                if not run_res.actual_blocked:
                    report.false_passes += 1
            else:
                if run_res.actual_blocked:
                    report.false_blocks += 1
                else:
                    report.unblocked_routed_runs += 1
                    if case.expected_tools and run_res.actual_tools == case.expected_tools:
                        report.tool_trajectory_correct += 1

                if any(tc.output.strip() and not tc.is_error for tc in run_res.tool_calls):
                    report.faithful_denominator += 1
                    if run_res.faithful:
                        report.faithful_runs += 1

            log.info(
                "benchmark: '%s' [trial %d/%d] %s (%.0fms, faith=%.3f)%s",
                case.id,
                run_idx,
                k_runs,
                "PASS" if run_res.passed else "FAIL",
                run_res.latency_ms,
                run_res.faithfulness_score,
                f" — {'; '.join(run_res.failure_reasons)}" if run_res.failure_reasons else "",
            )

        if summary.pass_pow_k:
            report.consistent_cases += 1

        report.case_summaries.append(summary)

    log.info(
        "benchmark: pass@1=%.0f%% (%d/%d runs) | pass^%d=%.0f%% (%d/%d cases) | "
        "routing_acc=%.0f%% | faith=%.0f%% (mean=%.3f) | mcp_errors=%d | "
        "routed=%.0fms (p95=%.0fms) | blocked=%.0fms (p95=%.0fms)",
        report.pass_at_1 * 100,
        report.passed_runs,
        report.total_runs,
        k_runs,
        report.pass_pow_k * 100,
        report.consistent_cases,
        report.total_cases,
        report.routing_accuracy * 100,
        report.faithfulness_rate * 100,
        report.mean_faithfulness_score,
        report.mcp_errors,
        report.routed_latency_ms,
        report.routed_p95_ms,
        report.blocked_latency_ms,
        report.blocked_p95_ms,
    )

    return report