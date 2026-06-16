from dataclasses import dataclass, field


@dataclass
class GraphMeta:
    name: str
    node_count: int
    edge_count: int
    labels: list[str]
    relationship_types: list[str]


@dataclass
class GraphFilter:
    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)


@dataclass
class QuerySpec:
    id: str
    cypher: str
    params: dict = field(default_factory=dict)
    prerequisites: list[str] = field(default_factory=list)
    graphs: GraphFilter = field(default_factory=GraphFilter)

