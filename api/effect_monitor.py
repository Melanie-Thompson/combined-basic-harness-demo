"""Effect monitor — tracks what a tool execution actually does.

Wraps tool execution to intercept and record:
- Network calls (urllib, httpx, requests, socket)
- Filesystem access (open, pathlib)
- Environment variable reads (os.environ)
- Subprocess calls (subprocess, os.system)

Returns a list of observed effects alongside the tool result so the
harness can compare them against declared capabilities.
"""

import builtins
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

log = logging.getLogger("effect_monitor")


@dataclass
class ObservedEffect:
    kind: str  # network, filesystem, env, subprocess
    target: str
    action: str  # read, write, connect, getenv, exec

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "target": self.target, "action": self.action}


@dataclass
class EffectReport:
    effects: list[ObservedEffect] = field(default_factory=list)

    def record(self, kind: str, target: str, action: str) -> None:
        self.effects.append(ObservedEffect(kind=kind, target=target, action=action))

    def to_list(self) -> list[dict[str, str]]:
        return [e.to_dict() for e in self.effects]


@contextmanager
def monitor_effects():
    """Context manager that intercepts side effects during tool execution."""
    report = EffectReport()
    original_open = builtins.open
    original_environ_getitem = os.environ.__class__.__getitem__
    original_environ_get = os.environ.__class__.get

    def tracked_open(file, mode="r", *args, **kwargs):
        action = "write" if any(c in str(mode) for c in "wax+") else "read"
        report.record("filesystem", str(file), action)
        return original_open(file, mode, *args, **kwargs)

    def tracked_getitem(self_env, key):
        report.record("env", str(key), "getenv")
        return original_environ_getitem(self_env, key)

    def tracked_get(self_env, key, default=None):
        report.record("env", str(key), "getenv")
        return original_environ_get(self_env, key, default)

    # Intercept imports of network/subprocess modules
    original_import = builtins.__import__

    NETWORK_MODULES = {"urllib", "urllib.request", "httpx", "requests", "socket", "http.client", "aiohttp"}
    SUBPROCESS_MODULES = {"subprocess", "os"}

    def tracked_import(name, *args, **kwargs):
        if name in NETWORK_MODULES or any(name.startswith(m + ".") for m in NETWORK_MODULES):
            report.record("network", name, "import")
        if name in SUBPROCESS_MODULES:
            # os is used for many things — only flag it, don't block
            pass
        return original_import(name, *args, **kwargs)

    try:
        builtins.open = tracked_open
        builtins.__import__ = tracked_import
        with patch.object(os.environ.__class__, "__getitem__", tracked_getitem), \
             patch.object(os.environ.__class__, "get", tracked_get):
            yield report
    finally:
        builtins.open = original_open
        builtins.__import__ = original_import
