"""
Pytest benchmark configuration and compatibility shims.
"""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    # Ensure backward compatibility between pytest 8+ and older pytest-benchmark
    try:
        from _pytest.nodes import Node

        if not hasattr(Node, "_nodeid"):
            Node._nodeid = property(lambda self: self.nodeid)  # type: ignore[attr-defined]
    except Exception:
        pass
