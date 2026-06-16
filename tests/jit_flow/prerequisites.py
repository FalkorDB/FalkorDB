try:
    from .spec_types import GraphMeta
except ImportError:
    from spec_types import GraphMeta


def _has_nodes(meta: GraphMeta) -> bool:
    return meta.node_count > 0


def _has_two_nodes(meta: GraphMeta) -> bool:
    return meta.node_count >= 2


def _has_edges(meta: GraphMeta) -> bool:
    return meta.edge_count > 0


def _has_labels(meta: GraphMeta) -> bool:
    return len(meta.labels) > 0


def _has_relationship_types(meta: GraphMeta) -> bool:
    return len(meta.relationship_types) > 0


PREDICATES = {
    "has_nodes": (_has_nodes, "requires at least one node"),
    "has_two_nodes": (_has_two_nodes, "requires at least two nodes"),
    "has_edges": (_has_edges, "requires at least one edge"),
    "has_labels": (_has_labels, "requires at least one label"),
    "has_relationship_types": (
        _has_relationship_types,
        "requires at least one relationship type",
    ),
}


def evaluate_prerequisites(prerequisites: list[str], meta: GraphMeta) -> tuple[bool, list[str]]:
    missing = []
    for prerequisite in prerequisites:
        if prerequisite not in PREDICATES:
            raise ValueError(f"Unknown prerequisite: {prerequisite}")
        predicate, description = PREDICATES[prerequisite]
        if not predicate(meta):
            missing.append(description)
    return (len(missing) == 0, missing)
