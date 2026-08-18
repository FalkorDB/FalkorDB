"""LDBC Social Network Benchmark (Interactive v1) against FalkorDB.

Scope, stated plainly: this measures the 14 Interactive **complex reads** as an
internal instrument. It is not an auditable LDBC result — that needs the
official Java driver, its validation and workload-generation phases, LDBC
membership and a commissioned audit. Nothing here should be published as an
"LDBC score".

What it is good for: catching a query-planner regression on realistic graph
shapes, and tracking how far FalkorDB's dialect is from the reference queries.
"""

from falkorbench.ldbc import dataset
from falkorbench.ldbc import loader
from falkorbench.ldbc import params
from falkorbench.ldbc import queries
from falkorbench.ldbc import runner
from falkorbench.ldbc import schema

__all__ = ["dataset", "loader", "params", "queries", "runner", "schema"]
