//! Where a graph command runs: inline on the calling thread, or blocked and handed to
//! the thread pool.

use redis_module::{Context, ContextFlags};

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
