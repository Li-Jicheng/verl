# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0.
"""Exercise engine-port reservation without Ray, vLLM or accelerator imports."""

import ast
import asyncio
import importlib.util
import inspect
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_SERVER = _ROOT / "verl/workers/rollout/vllm_rollout/vllm_async_server.py"


def _load_server_methods():
    spec = importlib.util.spec_from_file_location("port_test_net_utils", _ROOT / "verl/utils/net_utils.py")
    net = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(net)
    tree = ast.parse(_SERVER.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "vLLMHttpServer")
    methods = [
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"_reserve_engine_ports", "_release_engine_ports", "launch_server", "run_server"}
    ]
    scope = dict(
        get_free_port=net.get_free_port,
        is_valid_ipv6_address=net.is_valid_ipv6_address,
        argparse=SimpleNamespace(Namespace=object),
        inspect=inspect,
    )
    # Compile the actual production methods with their standard-library dependency.
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(_SERVER), "exec"), scope)
    return type(
        "PortServer",
        (),
        {
            name: scope[name]
            for name in ("_reserve_engine_ports", "_release_engine_ports", "launch_server", "run_server")
        },
    ), scope


PortServer, _SCOPE = _load_server_methods()


@pytest.mark.parametrize("address,wildcard", [("127.0.0.1", "0.0.0.0"), ("::1", "::")])
def test_reservations_match_listener_scope_without_changing_advertised_address(address, wildcard):
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    held = socket.socket(family, socket.SOCK_STREAM)
    try:
        try:
            held.bind((address, 0))
        except OSError:
            if family == socket.AF_INET6:
                pytest.skip("IPv6 loopback is unavailable")
            raise
        held.listen(1)
        server = PortServer()
        server._server_address = address
        try:
            server._reserve_engine_ports()
            sockets = [server._master_sock, server._dp_rpc_sock, server._dp_master_sock]
            assert server._server_address == address
            assert all(sock.getsockname()[0] == wildcard for sock in sockets)
            ports = [sock.getsockname()[1] for sock in sockets]
            assert len(set(ports)) == 3
            assert held.getsockname()[1] not in ports
        finally:
            server._release_engine_ports()
        assert all(sock.fileno() == -1 for sock in sockets)
        server._release_engine_ports()
    finally:
        held.close()


def test_partial_allocation_failure_releases_earlier_sockets(monkeypatch):
    server = PortServer()
    server._server_address = "127.0.0.1"
    original = _SCOPE["get_free_port"]
    allocated = []

    def allocate(address, with_alive_sock):
        if len(allocated) == 2:
            raise OSError("allocation failed")
        result = original(address, with_alive_sock)
        allocated.append(result[1])
        return result

    monkeypatch.setitem(_SCOPE, "get_free_port", allocate)
    with pytest.raises(OSError, match="allocation failed"):
        server._reserve_engine_ports()
    assert all(sock.fileno() == -1 for sock in allocated)
    server._release_engine_ports()


@pytest.mark.parametrize("error", [RuntimeError("invalid engine configuration"), asyncio.CancelledError()])
def test_launch_failure_and_cancellation_release_reservations(error):
    server = PortServer()
    server._server_address = "127.0.0.1"
    server._reserve_engine_ports()
    sockets = [server._master_sock, server._dp_rpc_sock, server._dp_master_sock]

    async def fail(*args):
        raise error

    server._launch_server = fail
    with pytest.raises(type(error)):
        asyncio.run(server.launch_server())
    assert all(sock.fileno() == -1 for sock in sockets)


def test_non_master_without_reservations_can_clean_up():
    PortServer()._release_engine_ports()


def test_reservations_are_released_before_the_engine_binds(monkeypatch):
    server = PortServer()
    server._server_address = "127.0.0.1"
    server._reserve_engine_ports()
    sockets = [server._master_sock, server._dp_rpc_sock, server._dp_master_sock]
    config = SimpleNamespace(parallel_config=SimpleNamespace())
    args = SimpleNamespace(create_engine_config=lambda **kwargs: config)

    def create_engine(vllm_config, usage_context):
        assert all(sock.fileno() == -1 for sock in sockets)
        assert vllm_config.parallel_config.data_parallel_master_port == server._dp_master_port
        raise RuntimeError("stop after binding precondition")

    monkeypatch.setitem(_SCOPE, "AsyncEngineArgs", SimpleNamespace(from_cli_args=lambda value: args))
    monkeypatch.setitem(_SCOPE, "UsageContext", SimpleNamespace(OPENAI_API_SERVER="test"))
    monkeypatch.setitem(_SCOPE, "AsyncLLM", SimpleNamespace(from_vllm_config=create_engine))
    try:
        with pytest.raises(RuntimeError, match="stop after binding precondition"):
            asyncio.run(server.run_server(object()))
    finally:
        server._release_engine_ports()
