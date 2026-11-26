"""Capability enforcement — builds per-tool sandbox restrictions from declared capabilities.

Each tool declares what it needs. This module turns those declarations into
a restricted OSAccess and a filtered external_lookup for the Monty sandbox.
Anything undeclared is blocked.
"""

from __future__ import annotations

import fnmatch
import logging
import os
from collections.abc import Callable

try:
    import pydantic_monty as pm
except ImportError:
    pm = None  # type: ignore[assignment]

from AI.tools.definition import Capability, ToolDefinition

log = logging.getLogger("capabilities")


def build_os_access(tool: ToolDefinition) -> pm.OSAccess:
    """Build a restricted OSAccess for a tool based on its declared capabilities.

    - No filesystem capability: empty virtual filesystem
    - filesystem capability: MemoryFiles seeded from matching real paths
    - No env capability: no environment variables
    - env capability: only the declared env var names are visible
    """
    fs_caps = [c for c in tool.capabilities if c.kind == "filesystem"]
    env_caps = [c for c in tool.capabilities if c.kind == "env"]

    files = []
    for cap in fs_caps:
        for path in _resolve_glob(cap.target):
            try:
                content = open(path, "rb").read()
                files.append(pm.MemoryFile(path, content))
            except OSError:
                log.warning("capability: %s declared filesystem '%s' but cannot read '%s'",
                            tool.name, cap.target, path)

    environ = {}
    for cap in env_caps:
        val = os.environ.get(cap.target)
        if val is not None:
            environ[cap.target] = val

    return pm.OSAccess(
        files=files if files else None,
        environ=environ if environ else None,
    )


def build_external_lookup(
    tool: ToolDefinition,
    sandbox_functions: dict[str, Callable],
) -> dict[str, Callable]:
    """Build a filtered external_lookup that only exposes the tool's own function.

    The sandbox can only call functions that exist in this dict. Tools with no
    network capability get only their own function. Tools with network capability
    get their function plus any declared network targets as allowed hosts (enforced
    at the fetch/commit layer, not here — the sandbox itself has no raw socket access).
    """
    fn = sandbox_functions.get(tool.name)
    if fn is None:
        return {}
    return {tool.name: fn}


def check_capabilities(tool: ToolDefinition) -> list[str]:
    """Pre-execution inspection: check that declared capabilities are valid.

    Returns a list of warnings. An empty list means the tool passed inspection.
    """
    warnings: list[str] = []

    for cap in tool.capabilities:
        if cap.kind == "filesystem":
            resolved = _resolve_glob(cap.target)
            if not resolved:
                warnings.append(f"filesystem capability '{cap.target}' matches no paths")
        elif cap.kind == "env":
            if cap.target not in os.environ:
                warnings.append(f"env capability '{cap.target}' not set in environment")
        elif cap.kind == "network":
            if not cap.target:
                warnings.append("network capability has empty target")
        elif cap.kind == "subprocess":
            warnings.append(f"subprocess capability '{cap.target}' — subprocess is not supported in Monty sandbox")

    if tool.kind == "processor" and any(c.kind == "network" for c in tool.capabilities):
        warnings.append("processor declares network capability — processors should run locally without network access")

    return warnings


def audit_capabilities(tool: ToolDefinition) -> dict:
    """Return an audit-friendly summary of a tool's declared capabilities."""
    return {
        "tool": tool.name,
        "kind": tool.kind,
        "placement": tool.placement,
        "capabilities": [
            {"kind": c.kind, "target": c.target}
            for c in tool.capabilities
        ],
        "requires_confirmation": tool.requires_confirmation,
    }


def _resolve_glob(pattern: str) -> list[str]:
    """Resolve a path glob to existing file paths."""
    import glob
    return glob.glob(pattern, recursive=True)
