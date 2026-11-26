"""Causality enforcement — validates tool execution order against the DAG.

Each tool definition may declare `depends_on: [tool_name, ...]`. Before a
tool executes, the validator checks that all its dependencies have already
been called in the current session. Violations are rejected before execution.

The data-flow layer traces whether each tool's input values came from a
preceding tool's output or from the user message, flagging ungrounded inputs.

Session state is in-memory — a session is scoped to one harness conversation.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("causality")


@dataclass
class DataFlowEdge:
    from_tool: str
    to_tool: str
    values: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"from_tool": self.from_tool, "to_tool": self.to_tool, "values": self.values}


@dataclass
class CausalityResult:
    valid: bool
    violations: list[str] = field(default_factory=list)
    data_flow: list[DataFlowEdge] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"valid": self.valid}
        if self.violations:
            d["violations"] = self.violations
        if self.data_flow:
            d["data_flow"] = [e.to_dict() for e in self.data_flow]
        return d


@dataclass
class ExecutionRecord:
    tool_name: str
    tool_input: dict[str, Any]
    tool_output: str


def _extract_values(d: dict) -> list[str]:
    values: list[str] = []
    for v in d.values():
        if isinstance(v, dict):
            values.extend(_extract_values(v))
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    values.extend(_extract_values(item))
                else:
                    values.append(str(item))
        else:
            s = str(v)
            if not s:
                continue
            if isinstance(v, (int, float)) or len(s) >= 2:
                values.append(s)
    return values


_WORD_BOUNDARY_CACHE: dict[str, re.Pattern] = {}


def _word_boundary_match(needle: str, text: str) -> bool:
    if needle not in _WORD_BOUNDARY_CACHE:
        _WORD_BOUNDARY_CACHE[needle] = re.compile(
            r"(?<![.\w])" + re.escape(needle) + r"(?![.\w])"
        )
    return _WORD_BOUNDARY_CACHE[needle].search(text) is not None


def _find_value_in_text(value: str, text: str) -> bool:
    try:
        f = float(value)
        for fmt in (f"{f:+.1f}", f"{f:.1f}", f"{f:.0f}", f"{f:g}", value):
            if _word_boundary_match(fmt, text):
                return True
        return False
    except (ValueError, TypeError):
        pass
    if len(value) <= 3:
        return _word_boundary_match(value, text)
    return value in text


class CausalityValidator:
    """Per-session causality enforcement.

    Tracks which tools have been called and validates the dependency DAG
    before each execution. Also traces data flow between tool calls.
    """

    def __init__(self):
        self._called: list[ExecutionRecord] = []
        self._called_names: set[str] = set()

    def check_before_execute(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        definition: dict[str, Any],
    ) -> CausalityResult:
        """Check the dependency DAG before a tool runs.

        Returns a CausalityResult. If valid is False, the tool should
        not execute — its dependencies haven't been met.
        """
        violations: list[str] = []

        deps = definition.get("depends_on", [])
        for dep in deps:
            if dep not in self._called_names:
                violations.append(
                    f"'{tool_name}' requires '{dep}' to run first"
                )

        if violations:
            log.warning("causality violation for '%s': %s", tool_name, violations)

        return CausalityResult(valid=len(violations) == 0, violations=violations)

    def record_execution(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: str,
    ) -> None:
        """Record a successful tool execution for future dependency checks."""
        self._called.append(ExecutionRecord(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
        ))
        self._called_names.add(tool_name)

    def trace_data_flow(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        user_message: str = "",
    ) -> tuple[list[DataFlowEdge], list[str]]:
        """Trace where this tool's input values came from."""
        input_values = _extract_values(tool_input)
        if not input_values:
            return [], []

        edges: list[DataFlowEdge] = []
        warnings: list[str] = []
        matched: dict[str, list[str]] = {}

        for val in input_values:
            if user_message and _find_value_in_text(val, user_message):
                matched.setdefault("user", []).append(val)
                continue

            found = False
            for prev in self._called:
                if _find_value_in_text(val, prev.tool_output):
                    matched.setdefault(prev.tool_name, []).append(val)
                    found = True
                    break
            if not found:
                matched.setdefault("_untraced", []).append(val)

        for source, vals in matched.items():
            if source in ("user", "_untraced"):
                continue
            edges.append(DataFlowEdge(
                from_tool=source,
                to_tool=tool_name,
                values=vals,
            ))

        untraced = matched.get("_untraced", [])
        from_user = matched.get("user", [])
        from_tools = {k for k in matched if k not in ("user", "_untraced")}

        if untraced and not from_user and not from_tools:
            warnings.append(
                f"'{tool_name}' inputs {untraced} not traceable to "
                f"any preceding tool output or user message"
            )

        return edges, warnings

    def reset(self) -> None:
        """Clear session state."""
        self._called.clear()
        self._called_names.clear()


# Per-session validators, keyed by session_id
_sessions: dict[str, CausalityValidator] = {}


def get_validator(session_id: str) -> CausalityValidator:
    if session_id not in _sessions:
        _sessions[session_id] = CausalityValidator()
    return _sessions[session_id]


def close_session(session_id: str) -> None:
    _sessions.pop(session_id, None)
