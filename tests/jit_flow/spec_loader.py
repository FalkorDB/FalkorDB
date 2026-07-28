try:
    from .spec_types import GraphFilter, QuerySpec
except ImportError:
    from spec_types import GraphFilter, QuerySpec

try:
    import yaml
except ImportError as exc:
    raise RuntimeError(
        "PyYAML is required for JIT flow YAML specs. Install it with: pip3 install pyyaml"
    ) from exc


def _as_list_of_strings(value, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    return value


def load_specs_file(path: str) -> list[QuerySpec]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{path}: YAML root must be a map")

    queries = data.get("queries")
    if not isinstance(queries, list) or len(queries) == 0:
        raise ValueError(f"{path}: 'queries' must be a non-empty list")

    spec_list = []
    for idx, query in enumerate(queries):
        if not isinstance(query, dict):
            raise ValueError(f"{path}: query at index {idx} must be a map")

        query_id = query.get("id")
        cypher = query.get("cypher")
        params = query.get("params", {})
        prerequisites = query.get("prerequisites", [])
        graphs = query.get("graphs", {})

        if not isinstance(query_id, str) or not query_id:
            raise ValueError(f"{path}: query at index {idx} has invalid 'id'")
        if not isinstance(cypher, str) or not cypher.strip():
            raise ValueError(f"{path}: query '{query_id}' has invalid 'cypher'")
        if not isinstance(params, dict):
            raise ValueError(f"{path}: query '{query_id}' has invalid 'params'")
        if not isinstance(prerequisites, list) or not all(
            isinstance(item, str) for item in prerequisites
        ):
            raise ValueError(f"{path}: query '{query_id}' has invalid 'prerequisites'")
        if not isinstance(graphs, dict):
            raise ValueError(f"{path}: query '{query_id}' has invalid 'graphs'")

        graph_filter = GraphFilter(
            allow=_as_list_of_strings(graphs.get("allow"), f"{path}:{query_id}:graphs.allow"),
            deny=_as_list_of_strings(graphs.get("deny"), f"{path}:{query_id}:graphs.deny"),
        )

        spec_list.append(
            QuerySpec(
                id=query_id,
                cypher=cypher,
                params=params,
                prerequisites=prerequisites,
                graphs=graph_filter,
            )
        )

    return spec_list
