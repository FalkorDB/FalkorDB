//! Where a graph command runs: inline on the calling thread, or blocked and handed to
//! the thread pool.

use valkey_module::{Context, ContextFlags};

/// True when this context must not block, so the command has to run inline on the
/// calling thread.
///
/// Mirrors C's dispatcher (`src/commands/cmd_dispatcher.c` on `master`), which is the
/// reference for "this context must not block":
///
/// ```c
/// bool main_thread = (is_replicated ||
///     (flags & (REDISMODULE_CTX_FLAGS_MULTI       |
///             REDISMODULE_CTX_FLAGS_LUA           |
///             REDISMODULE_CTX_FLAGS_DENY_BLOCKING |
///             REDISMODULE_CTX_FLAGS_LOADING))) ;
/// ```
///
/// Every flag earns its place:
///
/// * `REPLICATED` — a replica has to apply the command *before* the handler returns.
///   Blocking instead lets Redis advance the replication offset while the write is
///   still queued, so the master's `WAIT` reports the replica in sync when it is not.
/// * `MULTI` / `LUA` — Redis rejects blocking outright in both. `LUA` is defensive
///   only: every graph command is registered `deny-script`, so Redis rejects the call
///   before the handler runs.
/// * `DENY_BLOCKING` / `LOADING` — AOF replay drives a fake client carrying
///   `CLIENT_DENY_BLOCKING` but *not* `CLIENT_MASTER`, so `REPLICATED` is not set for
///   it. Blocking that client is fatal rather than merely wrong: Redis asserts
///   `(fakeClient->flags & CLIENT_BLOCKED) == 0` (`aof.c`) while loading, so any
///   AOF-enabled server crashed on restart (#2421).
///
/// Keep this the single definition. The bug it closes was five call sites having each
/// grown a narrower copy of C's one predicate, and drifting from it independently.
pub fn must_run_inline(ctx: &Context) -> bool {
    ctx.get_flags().intersects(
        ContextFlags::MULTI
            | ContextFlags::REPLICATED
            | ContextFlags::LUA
            | ContextFlags::DENY_BLOCKING
            | ContextFlags::LOADING,
    )
}

/// True when a query that finishes in this context is logged to the graph's telemetry
/// stream.
///
/// Telemetry is node-local: each node's stream holds the queries it served its own
/// clients. Two kinds of command run here without being served here, and C's
/// dispatcher leaves both out (`log_query` in `cmd_dispatcher.c`):
///
/// * `REPLICATED` — the master's query, re-run on this replica. It is in the master's
///   stream already; here it would show a replica "serving" writes it never accepted.
/// * `LOADING` — AOF replay. Logging it would rebuild, on every restart, a stream of
///   queries from before the restart, stamped with the time of the replay.
///
/// Both run inline (see [`must_run_inline`]), so the inline path is the only one that
/// has to ask. This engine replicates writes as `GRAPH.EFFECT`, which logs nothing, so
/// what the rule catches in practice is a `GRAPH.QUERY` from elsewhere: a C master's
/// replication stream, or an AOF an earlier version wrote. `CMD_INFO` is the other half of C's predicate, and
/// `telemetry::enqueue_entry` reads it.
pub fn logs_telemetry(ctx: &Context) -> bool {
    !ctx.get_flags()
        .intersects(ContextFlags::REPLICATED | ContextFlags::LOADING)
}
