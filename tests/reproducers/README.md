# Crash reproducers

Standalone scripts for reproducing customer-reported crashes outside the regular
test suite. These are intentionally NOT wired into CI because they intentionally
crash the server.

## issue_1823_uaf_reproducer.py

Reproduces the `QGEdge_RelationID` use-after-free reported in
[issue #1823](https://github.com/FalkorDB/FalkorDB/issues/1823).

Crash signature (from customer log):

```
SIGSEGV (signal 11), si_code 1, accessing address (nil)
falkordb.so(QGEdge_RelationID+0xb)
falkordb.so(EdgeTraverseCtx_CollectEdges+0xdf)
```

### Mechanism

When the execution-plan cache evicts a cached entry, a clone of that plan may
still be running on a thread-pool worker. `QGEdge` and `QGNode` instances held
by the clone borrow alias / label / reltype string pointers that originated in
the source plan's AST and sub-query-graph. Eviction tears those down, and the
in-flight clone dereferences the freed memory the next time it touches
`e->reltypeIDs`.

### Reproducing

Build FalkorDB with ASan:

```
./build.sh SAN=address
```

Start a server with a tiny cache and several worker threads (small cache size
maximizes eviction pressure):

```
LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libasan.so.8 /lib/x86_64-linux-gnu/libzstd.so.1" \
  redis-server --loadmodule bin/linux-x64-debug-asan/falkordb.so \
    CACHE_SIZE 1 THREAD_COUNT 8 --port 6399
```

Drive load:

```
python3 tests/reproducers/issue_1823_uaf_reproducer.py 127.0.0.1 6399
```

Expected outcome on master (commit 0f1552e24, 2026-04): SIGSEGV in
`QGEdge_RelationID` from a `thread-pool-thr` within ~30 seconds.

### Status

A first-pass fix was attempted in the closed
[PR #1824](https://github.com/FalkorDB/FalkorDB/pull/1824) (deep-copying alias
and reltype strings into `QGNode`/`QGEdge` themselves). Cherry-picking that PR
locally:

* removes the original SIGSEGV, but
* introduces silent wrong-result regressions in `tests/flow/test_cache.py`
  (test_05/06/07/08/09/10 fail) because many downstream consumers (NodeScanCtx,
  AllNodeScan, AE operands, NodeCreateCtx, EdgeCreateCtx, merge ops, ...) keep
  borrowed pointers into the per-MATCH `sub_qg` that `buildMatchOpTree` frees
  eagerly. PR #1824's deep copy moves ownership into that short-lived sub_qg,
  so all those borrowed pointers now dangle right after each MATCH is built.

A complete fix needs to either

1. extend ownership end-to-end across all op constructors and clone/free paths,
   or
2. attach `sub_qg` to the parent `ExecutionPlan` so it lives as long as the
   plan, AND keep PR #1824's ownership change so cloned cached plans survive
   eviction of the source plan.

Either route requires a careful design pass plus a broad flow-test sweep
(cache, index_scans, optimizations_plan, traversal_construction, merge, ...).
