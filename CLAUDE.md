# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FalkorDB is a Property Graph Database implemented as a Redis module (`falkordb.so`). It uses sparse matrices (via GraphBLAS) to represent adjacency matrices and linear algebra for query execution. The query language is OpenCypher.

- **Primary language:** C (core engine), with Rust components in `deps/FalkorDB-core-rs/`
- **Build system:** Make (wrapper) + CMake (compilation) + Cargo (Rust)
- **License:** SSPLv1

## Build Commands

```bash
# Clone with submodules (required - many git submodule dependencies)
git clone --recurse-submodules -j8 https://github.com/FalkorDB/FalkorDB.git

# Build (produces bin/<arch>/src/falkordb.so)
make

# Debug build
make DEBUG=1

# Build with address sanitizer
make SAN=address

# Build for Valgrind
make VG=1

# Build with coverage
make COV=1

# Run redis-server with the module loaded
make run

# Clean build artifacts (add ALL=1 to include deps, DEPS=1 for dependency artifacts)
make clean
```

### System Dependencies

- **Ubuntu:** `apt-get install build-essential cmake m4 automake peg libtool autoconf python3 python3-pip`
- **macOS:** `brew install cmake m4 automake peg libtool autoconf`
  - macOS requires GCC for OpenMP support: `brew install gcc g++`

## Testing

```bash
# Install Python test dependencies
pip install -r tests/requirements.txt

# Run all tests (unit + flow + tck + upgrade)
make test

# Individual test suites
make unit-tests
make flow-tests
make tck-tests
make upgrade-tests
make fuzz-tests

# Run a specific test
make flow-tests TEST=test_aggregation
make unit-tests TEST=test_delta_matrix

# Parallel test execution
make flow-tests PARALLEL=4

# Run with debugger
make flow-tests GDB=1 TEST=test_aggregation

# List all tests without running
make test LIST=1
```

**Flow tests** (`tests/flow/test_*.py`): Python integration tests using the RLTest framework. Each test class gets a live Redis+FalkorDB instance. Pattern:

```python
from common import *
class testExample():
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("test")
    def test01_something(self):
        result = self.graph.query("MATCH (n) RETURN n")
        self.env.assertEqual(result.result_set, expected)
```

**Unit tests** (`tests/unit/test_*.c`): Compiled C executables, one per test file. Built when `UNIT_TESTS=1`.

**TCK tests** (`tests/tck/`): OpenCypher compliance tests using Python Behave (BDD).

## Architecture

### Query Lifecycle

```
Client Request (GRAPH.QUERY "graph" "MATCH ...")
  -> CommandDispatch (src/commands/cmd_dispatcher.c)
     -> Thread routing (main thread for MULTI/Lua/replication, worker thread otherwise)
       -> Graph_Query (src/commands/cmd_query.c)
         -> Parse Cypher to AST (libcypher-parser)
         -> Build/cache ExecutionPlan from AST
         -> Acquire read/write lock on graph
         -> ExecutionPlan_Execute (pull-based operation tree)
         -> Format ResultSet (verbose/compact/bolt)
         -> Replicate if write (query replay or effects)
         -> Release lock
```

### Core Data Structures

**Graph** (`src/graph/graph.c`): Nodes and edges properties are stored in `DataBlock` arrays. Labels and relationships represented as sparse matrices (GraphBLAS CSR format):
- `adjacency_matrix`: all connections
- `labels[]`: one diagonal matrix per node label
- `relations[]`: one matrix per relation type
- `node_labels`: maps node_id to all its labels
- Uses `Delta_Matrix` for incremental changes on small matrices before committing to the main matrix.

**GraphContext** (`src/graph/graphcontext.c`): Wraps a Graph with metadata - schemas, attribute mappings, execution plan cache, pending write queue.

**ExecutionPlan** (`src/execution_plan/`): Tree of `OpBase` operations. Execution is pull-based (root pulls records from children). Plans are cached per graph for repeated queries.

### Key Source Modules

| Directory | Purpose |
|-----------|---------|
| `src/commands/` | Redis command handlers (QUERY, RO_QUERY, DELETE, COPY, CONFIG, etc.) |
| `src/execution_plan/` | Query planner, optimizer, operation implementations |
| `src/graph/` | Graph storage, matrices, delta matrices, locking |
| `src/ast/` | AST parsing and enrichment |
| `src/arithmetic/` | Expression evaluation (string, numeric, temporal, aggregate functions) |
| `src/procedures/` | Built-in procedures (PageRank, BFS, shortest path, etc.) |
| `src/index/` | Graph indexing (backed by RediSearch) |
| `src/serializers/` | Graph persistence (encode/decode to RDB) |
| `src/bolt/` | Bolt protocol support |
| `src/udf/` | User-defined functions (QuickJS JavaScript runtime) |
| `src/filter_tree/` | Query filter optimization |
| `src/schema/` | Schema and constraint definitions |

### Threading Model

- **Main Redis thread:** Used for replicated commands, MULTI/EXEC, Lua, and loading
- **Worker thread pool:** Regular queries dispatched via `ThreadPool_AddWork()`; client is blocked until completion
- **Reader/writer locks:** Per-graph `pthread_rwlock_t` with writer preference
- **Cron tasks:** Background tasks for timeouts, indexing, maintenance

### Dependencies (git submodules in `deps/`)

- **GraphBLAS/LAGraph:** Sparse matrix operations (core of graph representation)
- **RediSearch:** Full-text search and indexing
- **libcypher-parser:** OpenCypher query parsing
- **QuickJS:** JavaScript engine for UDFs
- **FalkorDB-core-rs:** Rust components

### Entry Points

- `src/module.c`: Redis module initialization, command registration
- `src/globals.c`: Global state (thread pools, memory)
- `src/query_ctx.c`: Thread-local query context
- `src/module_event_handlers.c`: Cluster replication, Redis events

### Redis Commands

`GRAPH.QUERY`, `GRAPH.RO_QUERY`, `GRAPH.EXPLAIN`, `GRAPH.PROFILE`, `GRAPH.DELETE`, `GRAPH.COPY`, `GRAPH.RESTORE`, `GRAPH.BULK`, `GRAPH.CONFIG`, `GRAPH.INFO`, `GRAPH.DEBUG`, `GRAPH.ACL`
