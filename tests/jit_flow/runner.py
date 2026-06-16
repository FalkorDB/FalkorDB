#!/usr/bin/env python3

import argparse
import os
from fnmatch import fnmatch

from falkordb import FalkorDB

from discovery import load_discovered_specs
from prerequisites import evaluate_prerequisites
from spec_types import GraphMeta, QuerySpec


def _graph_metadata(graph) -> GraphMeta:
    labels = [
        row[0]
        for row in graph.query(
            "CALL db.labels() YIELD label RETURN label ORDER BY label"
        ).result_set
    ]
    relationship_types = [
        row[0]
        for row in graph.query(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType"
        ).result_set
    ]
    node_count = graph.query("MATCH (n) RETURN count(n)").result_set[0][0]
    edge_count = graph.query("MATCH ()-[r]->() RETURN count(r)").result_set[0][0]
    return GraphMeta(
        name=graph.name,
        node_count=node_count,
        edge_count=edge_count,
        labels=labels,
        relationship_types=relationship_types,
    )


def _select_graphs(spec: QuerySpec, all_graphs: list[str]) -> list[str]:
    selected = list(all_graphs)
    if spec.graphs.allow:
        selected = [
            graph_name
            for graph_name in selected
            if any(fnmatch(graph_name, pattern) for pattern in spec.graphs.allow)
        ]
    if spec.graphs.deny:
        selected = [
            graph_name
            for graph_name in selected
            if not any(fnmatch(graph_name, pattern) for pattern in spec.graphs.deny)
        ]
    return selected


def _truncate_error(message: str, size: int = 90) -> str:
    if len(message) <= size:
        return message
    return f"{message[:size]}..."


def run_tests(host: str, port: int, specs_dir: str) -> bool:
    db = FalkorDB(host=host, port=port)
    graph_names = db.list_graphs()

    print(f"Connected to Redis at {host}:{port}")
    print(f"Found {len(graph_names)} graphs: {', '.join(graph_names)}\n")

    if len(graph_names) == 0:
        print("ERROR: No graphs found.")
        return False

    loaded_specs = load_discovered_specs(specs_dir)
    if len(loaded_specs) == 0:
        print(f"ERROR: No spec files found in {specs_dir}")
        return False

    metadata_by_graph = {}
    for graph_name in graph_names:
        graph = db.select_graph(graph_name)
        metadata_by_graph[graph_name] = _graph_metadata(graph)

    passed = 0
    failed = 0
    skipped = 0

    for spec_file, query_specs in loaded_specs:
        print(f"Spec file: {os.path.basename(spec_file)}")
        for query_spec in query_specs:
            selected_graphs = _select_graphs(query_spec, graph_names)
            executable_graphs = []
            skip_reasons = {}

            for graph_name in selected_graphs:
                is_eligible, missing = evaluate_prerequisites(
                    query_spec.prerequisites, metadata_by_graph[graph_name]
                )
                if is_eligible:
                    executable_graphs.append(graph_name)
                else:
                    skip_reasons[graph_name] = ", ".join(missing)

            print(
                f"  {query_spec.id}: using {len(executable_graphs)}/{len(selected_graphs)} graphs"
            )

            if len(selected_graphs) == 0:
                print("    SKIP all: no graphs matched filters")
                skipped += 1
                continue

            for graph_name in selected_graphs:
                if graph_name in skip_reasons:
                    print(f"    {graph_name}: SKIP ({skip_reasons[graph_name]})")
                    skipped += 1
                    continue

                try:
                    graph = db.select_graph(graph_name)
                    graph.query(query_spec.cypher, query_spec.params)
                    print(f"    {graph_name}: PASS")
                    passed += 1
                except Exception as e:
                    print(f"    {graph_name}: FAIL ({_truncate_error(str(e))})")
                    failed += 1
        print("")

    total = passed + failed + skipped
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed, {skipped} skipped (total: {total})")
    print("=" * 60)

    return failed == 0


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Run JIT flow tests from YAML specs")
    parser.add_argument("--host", default="127.0.0.1", help="Redis host")
    parser.add_argument("--port", type=int, default=6379, help="Redis port")
    parser.add_argument(
        "--specs-dir",
        default=os.path.join(here, "specs"),
        help="Directory containing spec_*.yml files",
    )
    args = parser.parse_args()

    return 0 if run_tests(args.host, args.port, args.specs_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())

