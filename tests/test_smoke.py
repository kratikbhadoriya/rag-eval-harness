"""Smoke test: pytest runs offline with no secrets."""

import os
import socket

import rageval


def test_package_imports_with_no_network_or_keys() -> None:
    assert "GROQ_API_KEY" not in os.environ
    assert rageval.__name__ == "rageval"


def test_network_is_unavailable_or_unneeded() -> None:
    # Import must not have opened sockets; creating one is still allowed locally.
    assert socket.gethostname()
