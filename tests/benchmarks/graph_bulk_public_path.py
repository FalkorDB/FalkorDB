#!/usr/bin/env python3
"""Measure and validate one Redis-backed GRAPH.BULK public-path shape.

The timed operation is a single GRAPH.BULK request sent over RESP to a fresh
Redis server. Payload generation and server setup are intentionally outside the
timer: callers of GRAPH.BULK supply an already materialized binary payload.
The server CPU delta and VmHWM are sampled for the same request. After timing,
the driver validates indexed edge lookup, incoming traversal, duplicate pairs,
arrival-order IDs, and a second append batch.

It has no third-party Python dependencies. Redis is the repository's documented
flow-test prerequisite and can be selected with REDIS_SERVER or --redis-server.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable


PROPERTY_INT64 = 4


class RedisReplyError(RuntimeError):
    pass


class RESPClient:
    """Minimal RESP2 client used so binary GRAPH.BULK arguments stay exact."""

    def __init__(self, host: str, port: int) -> None:
        self._socket = socket.create_connection((host, port), timeout=30)
        self._socket.settimeout(300)
        self._reader = self._socket.makefile("rb")

    def close(self) -> None:
        self._reader.close()
        self._socket.close()

    def command(self, *arguments: bytes | str | int) -> Any:
        encoded: list[bytes] = []
        for argument in arguments:
            if isinstance(argument, bytes):
                encoded.append(argument)
            else:
                encoded.append(str(argument).encode("utf-8"))

        request = bytearray(f"*{len(encoded)}\r\n".encode("ascii"))
        for argument in encoded:
            request.extend(f"${len(argument)}\r\n".encode("ascii"))
            request.extend(argument)
            request.extend(b"\r\n")

        self._socket.sendall(request)
        return self._read_reply()

    def _read_line(self) -> bytes:
        line = self._reader.readline()
        if not line.endswith(b"\r\n"):
            raise RedisReplyError(f"truncated RESP line: {line!r}")
        return line[:-2]

    def _read_reply(self) -> Any:
        marker = self._reader.read(1)
        if not marker:
            raise RedisReplyError("connection closed while reading RESP reply")

        line = self._read_line()
        if marker == b"+":
            return line
        if marker == b"-":
            raise RedisReplyError(line.decode("utf-8", "replace"))
        if marker == b":":
            return int(line)
        if marker == b"$":
            length = int(line)
            if length == -1:
                return None
            value = self._reader.read(length + 2)
            if len(value) != length + 2 or value[-2:] != b"\r\n":
                raise RedisReplyError("truncated RESP bulk string")
            return value[:-2]
        if marker == b"*":
            count = int(line)
            if count == -1:
                return None
            return [self._read_reply() for _ in range(count)]

        raise RedisReplyError(f"unsupported RESP marker: {marker!r}")


@dataclass(frozen=True)
class BulkPayload:
    node_count: int
    edge_count: int
    relation_files: tuple[bytes, ...]
    node_file: bytes
    duplicate_group_size: int
    first_pairs: tuple[tuple[int, int], ...]
    first_ids: tuple[int, ...]


def pack_int(value: int) -> bytes:
    return bytes((PROPERTY_INT64,)) + struct.pack("=q", value)


def pair_for_index(pair_index: int, width: int) -> tuple[int, int]:
    # width * width distinct pairs fit in nodes [0, width]. A nonzero
    # destination makes the first pair (0, 1), useful for transpose checks.
    return pair_index % width, 1 + pair_index // width


def node_file(node_count: int) -> bytes:
    stream = bytearray(b"N\0")
    stream.extend(struct.pack("=I", 1))
    stream.extend(b"id\0")
    for node_id in range(node_count):
        stream.extend(pack_int(node_id))
    return bytes(stream)


def edge_file(name: str, records: Iterable[tuple[int, int, int]]) -> bytes:
    stream = bytearray(name.encode("ascii"))
    stream.extend(b"\0")
    stream.extend(struct.pack("=I", 1))
    stream.extend(b"p\0")
    for src, dest, value in records:
        stream.extend(struct.pack("=QQ", src, dest))
        stream.extend(pack_int(value))
    return bytes(stream)


def build_payloads(
    edge_count: int,
    relation_count: int,
    duplicate_group_size: int,
    append_edge_count: int,
) -> tuple[BulkPayload, BulkPayload]:
    if edge_count <= 0:
        raise ValueError("--edges must be positive")
    if relation_count <= 0:
        raise ValueError("--relation-files must be positive")
    if duplicate_group_size <= 0:
        raise ValueError("--duplicate-group-size must be positive")
    if edge_count % (relation_count * duplicate_group_size) != 0:
        raise ValueError(
            "--edges must divide evenly by --relation-files times "
            "--duplicate-group-size"
        )
    if append_edge_count < 2 * relation_count:
        raise ValueError("--append-edges must provide at least two records per relation")

    groups_per_relation = edge_count // (relation_count * duplicate_group_size)
    total_initial_groups = groups_per_relation * relation_count
    # Append needs one existing pair and at most append_edge_count - relation_count
    # new pairs. Allocate all nodes up front so the append cannot resize matrices.
    total_pairs = total_initial_groups + append_edge_count
    width = math.isqrt(total_pairs - 1) + 1
    nodes = node_file(width + 1)

    initial_files: list[bytes] = []
    append_files: list[bytes] = []
    next_edge_id = 0
    next_new_pair = total_initial_groups
    append_value = edge_count

    for relation in range(relation_count):
        initial_records: list[tuple[int, int, int]] = []
        for group in range(groups_per_relation):
            src, dest = pair_for_index(relation * groups_per_relation + group, width)
            for _ in range(duplicate_group_size):
                initial_records.append((src, dest, next_edge_id))
                next_edge_id += 1
        initial_files.append(edge_file(f"R{relation}", initial_records))

        relation_append_count = append_edge_count // relation_count
        if relation < append_edge_count % relation_count:
            relation_append_count += 1

        append_records: list[tuple[int, int, int]] = []
        # The first append in each relation promotes a scalar to a vector when
        # duplicate_group_size is one, and extends a vector otherwise.
        existing_src, existing_dest = pair_for_index(
            relation * groups_per_relation, width
        )
        append_records.append((existing_src, existing_dest, append_value))
        append_value += 1
        for _ in range(1, relation_append_count):
            src, dest = pair_for_index(next_new_pair, width)
            next_new_pair += 1
            append_records.append((src, dest, append_value))
            append_value += 1
        append_files.append(edge_file(f"R{relation}", append_records))

    first_pairs = tuple(
        pair_for_index(relation * groups_per_relation, width)
        for relation in range(relation_count)
    )
    first_ids = tuple(
        relation * groups_per_relation * duplicate_group_size
        for relation in range(relation_count)
    )
    append_first_ids: list[int] = []
    append_id = edge_count
    for relation in range(relation_count):
        append_first_ids.append(append_id)
        append_id += append_edge_count // relation_count
        if relation < append_edge_count % relation_count:
            append_id += 1

    return (
        BulkPayload(
            node_count=width + 1,
            edge_count=edge_count,
            relation_files=tuple(initial_files),
            node_file=nodes,
            duplicate_group_size=duplicate_group_size,
            first_pairs=first_pairs,
            first_ids=first_ids,
        ),
        BulkPayload(
            node_count=0,
            edge_count=append_edge_count,
            relation_files=tuple(append_files),
            node_file=b"",
            duplicate_group_size=duplicate_group_size,
            first_pairs=first_pairs,
            first_ids=tuple(append_first_ids),
        ),
    )


def find_redis_server(argument: str | None) -> str:
    requested = argument or os.environ.get("REDIS_SERVER") or "redis-server"
    if os.path.sep in requested:
        path = Path(requested)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
    resolved = shutil.which(requested)
    if resolved is None:
        raise RuntimeError(
            f"Redis server {requested!r} was not found; install Redis 8+ or set "
            "REDIS_SERVER/--redis-server as documented for the flow tests"
        )
    return resolved


def unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def start_server(redis_server: str, module: Path, directory: Path) -> tuple[subprocess.Popen[bytes], int]:
    port = unused_port()
    log = (directory / "redis.log").open("wb")
    try:
        process = subprocess.Popen(
            [
                redis_server,
                "--port",
                str(port),
                "--bind",
                "127.0.0.1",
                "--save",
                "",
                "--appendonly",
                "no",
                "--dir",
                str(directory),
                "--dbfilename",
                "graph-bulk.rdb",
                "--loadmodule",
                str(module),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    finally:
        log.close()

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            detail = (directory / "redis.log").read_text(errors="replace")
            raise RuntimeError(f"redis-server exited during startup:\n{detail}")
        try:
            client = RESPClient("127.0.0.1", port)
            try:
                if client.command("PING") == b"PONG":
                    return process, port
            finally:
                client.close()
        except OSError:
            time.sleep(0.02)

    process.terminate()
    process.wait(timeout=10)
    detail = (directory / "redis.log").read_text(errors="replace")
    raise RuntimeError(f"timed out waiting for redis-server:\n{detail}")


def stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=20)


def server_cpu_ns(pid: int) -> int:
    # /proc/<pid>/stat fields 14 and 15 are user and system ticks. The process
    # name may contain spaces, so split only after its closing parenthesis.
    stat = Path(f"/proc/{pid}/stat").read_text()
    trailing = stat.rsplit(")", 1)[1].split()
    ticks = int(trailing[11]) + int(trailing[12])
    return ticks * (1_000_000_000 // os.sysconf("SC_CLK_TCK"))


def server_rss_bytes(pid: int) -> int:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS is unavailable in /proc; peak RSS requires Linux")


class RSSSampler:
    """Sample server RSS while one synchronous Redis command is in flight.

    Some Linux procfs configurations omit VmHWM. Sampling VmRSS from a separate
    client thread preserves a command-scoped peak without changing server work.
    """

    def __init__(self, pid: int) -> None:
        self._pid = pid
        self._peak = server_rss_bytes(pid)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("RSS sampler did not stop")
        self._peak = max(self._peak, server_rss_bytes(self._pid))
        return self._peak

    def _sample(self) -> None:
        while not self._stop.wait(0.001):
            self._peak = max(self._peak, server_rss_bytes(self._pid))


def assert_bulk_reply(reply: Any, node_count: int, edge_count: int) -> None:
    expected = f"{node_count} nodes created, {edge_count} edges created".encode()
    if reply != expected:
        raise AssertionError(f"unexpected GRAPH.BULK reply {reply!r}, expected {expected!r}")


def graph_bulk(client: RESPClient, graph: str, payload: BulkPayload, include_nodes: bool) -> Any:
    arguments: list[bytes | str | int] = [
        "GRAPH.BULK",
        graph,
        payload.node_count if include_nodes else 0,
        payload.edge_count,
        1 if include_nodes else 0,
        len(payload.relation_files),
    ]
    if include_nodes:
        arguments.append(payload.node_file)
    arguments.extend(payload.relation_files)
    return client.command(*arguments)


def graph_result(reply: Any) -> list[Any]:
    # GRAPH.QUERY uses an outer result-set array, while GRAPH.EXPLAIN returns
    # its lines directly. Normalize the former without assuming a single row.
    if isinstance(reply, list) and len(reply) == 1 and isinstance(reply[0], list):
        return reply[0]
    if isinstance(reply, list):
        return reply
    raise AssertionError(f"unexpected graph query reply: {reply!r}")


def rows(reply: Any) -> list[list[Any]]:
    result = graph_result(reply)
    if len(result) < 2 or not isinstance(result[1], list):
        raise AssertionError(f"unexpected graph query reply: {reply!r}")
    return result[1]


def scalar_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, bytes):
        return int(value)
    raise AssertionError(f"expected integer reply, got {value!r}")


def query_one_int(client: RESPClient, graph: str, query: str) -> int:
    result_rows = rows(client.command("GRAPH.QUERY", graph, query))
    if len(result_rows) != 1 or len(result_rows[0]) != 1:
        raise AssertionError(f"expected one scalar query result, got {result_rows!r}")
    return scalar_int(result_rows[0][0])


def assert_index_lookup(client: RESPClient, graph: str, relation: str, value: int, expected_id: int) -> None:
    query = f"MATCH ()-[e:{relation} {{p:{value}}}]->() RETURN ID(e)"
    explain = client.command("GRAPH.EXPLAIN", graph, query)
    if not isinstance(explain, list) or not any(
        isinstance(line, bytes) and b"Edge By Index Scan" in line for line in explain
    ):
        raise AssertionError(f"edge property lookup did not use its index: {explain!r}")

    result_rows = rows(client.command("GRAPH.QUERY", graph, query))
    if len(result_rows) != 1 or scalar_int(result_rows[0][0]) != expected_id:
        raise AssertionError(
            f"indexed lookup for property {value} returned {result_rows!r}; "
            f"expected edge ID {expected_id}"
        )


def validate_graph(
    client: RESPClient,
    graph: str,
    initial: BulkPayload,
    appended: BulkPayload | None,
) -> None:
    append_count = 0 if appended is None else appended.edge_count
    expected_count = initial.edge_count + append_count
    total = query_one_int(client, graph, "MATCH ()-[e]->() RETURN count(e)")
    if total != expected_count:
        raise AssertionError(f"edge count is {total}, expected {expected_count}")

    # Every relation's first decoded edge checks arrival-order IDs and index
    # insertion. Incoming traversals read each installed transpose, including
    # multiedge groups and the append's scalar/vector promotion.
    expected_incoming = initial.duplicate_group_size + (1 if appended else 0)
    for relation, first_pair in enumerate(initial.first_pairs):
        first_id = initial.first_ids[relation]
        relation_name = f"R{relation}"
        assert_index_lookup(client, graph, relation_name, first_id, first_id)
        first_src, first_dest = first_pair
        incoming = query_one_int(
            client,
            graph,
            "MATCH (target)<-[e:" + relation_name + "]-(source) "
            f"WHERE ID(target)={first_dest} AND ID(source)={first_src} "
            "RETURN count(e)",
        )
        if incoming != expected_incoming:
            raise AssertionError(
                f"incoming {relation_name} multiedge count is {incoming}, "
                f"expected {expected_incoming}"
            )

    if appended is not None:
        # Each append file has one existing pair first. Its value/ID validates
        # per-file wire-order continuity, vector promotion, and index effects.
        for relation, append_id in enumerate(appended.first_ids):
            assert_index_lookup(client, graph, f"R{relation}", append_id, append_id)


def create_edge_indices(client: RESPClient, graph: str, relation_count: int) -> None:
    for relation in range(relation_count):
        reply = graph_result(client.command(
            "GRAPH.QUERY",
            graph,
            f"CREATE INDEX FOR ()-[e:R{relation}]-() ON (e.p)",
        ))
        if not reply or reply[0] != b"Indices created: 1":
            raise AssertionError(f"failed to create R{relation} edge index: {reply!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module",
        default="bin/linux-x64-release/falkordb.so",
        help="built FalkorDB module path (default: %(default)s)",
    )
    parser.add_argument(
        "--redis-server",
        help="redis-server executable; defaults to REDIS_SERVER or redis-server",
    )
    parser.add_argument("--edges", type=int, required=True, help="initial bulk edge count")
    parser.add_argument(
        "--relation-files",
        type=int,
        required=True,
        help="number of relation descriptor files",
    )
    parser.add_argument(
        "--duplicate-group-size",
        type=int,
        required=True,
        help="number of parallel edges per initial endpoint pair",
    )
    parser.add_argument(
        "--append-edges",
        type=int,
        help="append batch size; defaults to max(64, 2 * relation files)",
    )
    parser.add_argument(
        "--mode",
        choices=("fresh", "append"),
        default="fresh",
        help="time the initial empty-topology bulk or a subsequent append bulk",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    module = Path(args.module).resolve()
    if not module.is_file():
        raise RuntimeError(f"module not found: {module}")

    append_edges = args.append_edges or max(64, 2 * args.relation_files)
    initial, appended = build_payloads(
        args.edges,
        args.relation_files,
        args.duplicate_group_size,
        append_edges,
    )
    redis_server = find_redis_server(args.redis_server)

    with tempfile.TemporaryDirectory(prefix="falkordb-graph-bulk-") as temporary:
        # Load a per-run copy from the fresh server directory. Besides avoiding
        # a server reading a concurrently rebuilt artifact, this makes Linux
        # module-page residency independent of the baseline/Candidate worktree
        # path before sampling command-scoped RSS.
        server_directory = Path(temporary)
        server_module = server_directory / "falkordb.so"
        shutil.copy2(module, server_module)
        process, port = start_server(redis_server, server_module, server_directory)
        try:
            client = RESPClient("127.0.0.1", port)
            try:
                graph = "graph_bulk_public_path"
                create_edge_indices(client, graph, args.relation_files)

                if args.mode == "append":
                    assert_bulk_reply(graph_bulk(client, graph, initial, True), initial.node_count, initial.edge_count)
                    client.command("PING")
                    timed_payload = appended
                    include_nodes = False
                else:
                    timed_payload = initial
                    include_nodes = True

                rss_sampler = RSSSampler(process.pid)
                rss_sampler.start()
                cpu_before = server_cpu_ns(process.pid)
                start = time.perf_counter_ns()
                reply = graph_bulk(client, graph, timed_payload, include_nodes)
                wall_ns = time.perf_counter_ns() - start
                cpu_ns = server_cpu_ns(process.pid) - cpu_before
                peak_rss_bytes = rss_sampler.stop()
                assert_bulk_reply(reply, timed_payload.node_count if include_nodes else 0, timed_payload.edge_count)

                if args.mode == "fresh":
                    assert_bulk_reply(graph_bulk(client, graph, appended, False), 0, appended.edge_count)

                validate_graph(client, graph, initial, appended)
            finally:
                client.close()
        finally:
            stop_server(process)

    # Perfloop's adapter requires exactly one object per declared metric.
    print(json.dumps({"metric": "wall_ns", "value": wall_ns}))
    print(json.dumps({"metric": "server_cpu_ns", "value": cpu_ns}))
    print(json.dumps({"metric": "server_peak_rss_bytes", "value": peak_rss_bytes}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"graph_bulk_public_path: {error}", file=sys.stderr)
        raise SystemExit(1)
