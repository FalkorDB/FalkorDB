---
name: Debug FalkorDB
description: Debug FalkorDB with GDB/LLDB, AddressSanitizer, Valgrind, and GRAPH.EXPLAIN/PROFILE
---

# Debug FalkorDB

Build FalkorDB for debugging. Developers launch and attach their own debugger.

## Debug build

    # Build with debug symbols (unoptimized, full symbols)
    make DEBUG=1

The debug binary is output to `bin/<arch>/src/falkordb.so`. Load it into redis-server and attach your debugger of choice (GDB, LLDB, IDE) to the running process.

## AddressSanitizer (memory errors)

    # Build with ASAN
    make SAN=address

    # Run tests with ASAN
    make flow-tests SAN=address
    make unit-tests SAN=address

## ThreadSanitizer (data races)

    make SAN=thread

## Valgrind (memory leaks and access errors)

    # Build for Valgrind
    make VG=1

    # Run tests under Valgrind
    make flow-tests VG=1
    make unit-tests VG=1

    # Valgrind with leak checking
    make flow-tests VG=1 VG_LEAKS=1

    # Valgrind with memory access error checking
    make unit-tests VG=1 VG_ACCESS=1

## Query diagnostics

    # View execution plan without running (check index usage)
    redis-cli GRAPH.EXPLAIN social "MATCH (p:Person {age: 30}) RETURN p"

    # Execute and see per-operator runtime stats
    redis-cli GRAPH.PROFILE social "MATCH (p:Person)-[:KNOWS]->(f) RETURN f"

    # Check slow query log
    redis-cli GRAPH.SLOWLOG social

## Notes

- `make DEBUG=1` produces unoptimized binaries with full symbols suitable for any debugger
- ASAN and Valgrind builds are mutually exclusive - use one or the other
- GRAPH.EXPLAIN shows the plan without executing; look for "Index Scan" vs "Label Scan"
- GRAPH.PROFILE executes the query and returns per-operator record counts and timing
