//! Core graph execution and concurrency primitives.
//!
//! This module owns the execution model used by Redis command handlers.
//!
//! ## Concurrency model
//! ```text
//! Client query
//!    |
//!    v
//! query_mut -> threadpool worker
//!    |
//!    +--> execute_query() detects write IR?
//!            |
//!            +-- no --> run on MVCC read snapshot (parallel reads)
//!            |
//!            +-- yes -> enqueue blocked client + query
//!                        |
//!                        v
//!                  process_write_queued_query()
//!                        |
//!                        +--> single writer loop
//!                              +--> execute_query_write()
//!                              +--> commit() on success / rollback() on error
//! ```
//!
//! The key design goal is predictable write ordering with high read throughput.
//! Reads are lock-light and concurrent; writes are serialized through an explicit
//! queue guarded by `write_loop`. Every query runs under a
//! [`crate::query_session::QuerySession`], which owns its locks and documents the
//! reader→writer protocol.

use crate::{
    config::{
        CONFIGURATION_IMPORT_FOLDER, EFFECTS_THRESHOLD, MAX_QUEUED_QUERIES, QUERY_MEM_CAPACITY,
        RESULTSET_SIZE, TIMEOUT, TIMEOUT_DEFAULT, TIMEOUT_MAX,
    },
    reply::{reply_compact, reply_verbose},
    slow_log::SlowLog,
    telemetry,
};
use atomic_refcell::AtomicRefCell;
use crossfire::{
    MTx, Rx,
    mpsc::{Array, bounded_blocking},
};
use graph::{
    graph::{
        graph::{Graph, Plan},
        mvcc_graph::MvccGraph,
    },
    planner::{IR, plan_is_non_deterministic},
    runtime::{
        eval::evaluate_param,
        pending::{
            EFFECT_CREATE_INDEX, EFFECT_DROP_INDEX, EFFECTS_VERSION, write_string, write_u16,
        },
        runtime::Runtime,
    },
    threadpool::{pending_count, spawn},
};
use orx_tree::{Collection, Dfs, NodeRef};
use parking_lot::RwLock;
use redis_module::{Context, ContextFlags, RedisResult, RedisValue, raw};
use std::{
    collections::HashMap,
    os::raw::{c_char, c_void},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use crate::allocator::{
    current_thread_usage, disable_tracking, enable_tracking, net_thread_usage, reset_counter,
};
use crate::dispatch::must_run_inline;
use crate::query_session::QuerySession;

/// Global registry of all live graph instances.
/// Used by the pthread_atfork prepare handler to sync all GraphBLAS matrices
/// before fork, preventing deadlocks in the BGSAVE child process.
pub static GRAPH_REGISTRY: std::sync::LazyLock<
    parking_lot::Mutex<HashMap<String, Arc<RwLock<ThreadedGraph>>>>,
> = std::sync::LazyLock::new(|| parking_lot::Mutex::new(HashMap::new()));

pub fn register_graph(
    name: String,
    arc: Arc<RwLock<ThreadedGraph>>,
) {
    // Invariant: every registered name must have been removed from the
    // registry before being re-inserted. `graph_free` removes entries on
    // Redis key delete/overwrite/expire, and the RDB multi-key load path
    // mutates the placeholder Arc in place rather than rebinding the name.
    // If this assert fires, a caller is leaking the previously-registered
    // Arc and likely racing index/teardown shutdown.
    let displaced = GRAPH_REGISTRY.lock().insert(name, arc);
    debug_assert!(
        displaced.is_none(),
        "register_graph: name already registered; missing graph_free or placeholder swap"
    );
    // Drop any displaced graph off the main Redis thread. Index::drop ->
    // RediSearch_DropIndex queues a destroyCallback to RediSearch's GC
    // thread pool when its timer can't be stopped synchronously; that
    // callback asserts gc->stopped == 1 and aborts under the resulting
    // race. Moving the drop to a background thread also routes Index::drop
    // through the GIL-acquiring path, which the synchronous main-thread
    // drop intentionally skips.
    if let Some(displaced) = displaced {
        std::thread::spawn(move || drop(displaced));
    }
}

/// True if `arc` is still the graph registered under some Redis key.
///
/// A query that has just escalated to writer mode checks this before mutating, and
/// **aborts** if it fails — mirroring C's `QueryCtx_AcquireWriteLock`, which re-opens
/// the key under WRITE and raises a runtime exception when the key is empty, is not a
/// graph, or holds a *different* value than the one the query started on (our
/// `data_ptr` comparison is that third check).
///
/// Aborting rather than committing-and-discarding is required, not just tidy: the
/// write would still be replicated, and a replica's `GRAPH.EFFECT` handler *creates* a
/// graph when the key is missing, so it would resurrect a key the primary no longer
/// has — a diverged replica.
///
/// One check is enough. The exposed window is exactly [release read lock → take GIL],
/// because every path that removes a key (`GRAPH.DELETE`, `FLUSHALL`, overwrite,
/// expiry) runs inline on the main thread, which cannot execute a command callback
/// while this query holds the GIL.
///
/// The scan compares `data_ptr` rather than looking the name up: `rename_graph`
/// re-keys the registry without updating `Graph::name()`, so an O(1) by-name lookup
/// would wrongly abort after a `RENAME`.
pub fn graph_is_registered(arc: &Arc<RwLock<ThreadedGraph>>) -> bool {
    let target = arc.data_ptr() as usize;
    GRAPH_REGISTRY
        .lock()
        .values()
        .any(|registered| registered.data_ptr() as usize == target)
}

/// Re-key a registry entry when a Redis RENAME moves a graph to a new key.
///
/// `graph_free` only runs for the *overwritten destination* value, so without
/// this the old name keeps a stale entry: the next `register_graph` under
/// that name (e.g. a concurrent write query re-creating the key) displaces
/// it and trips the invariant assert above.
pub fn rename_graph(
    old_name: &str,
    new_name: &str,
) {
    let displaced = {
        let mut reg = GRAPH_REGISTRY.lock();
        reg.remove(old_name)
            .and_then(|arc| reg.insert(new_name.to_string(), arc))
    };
    // The destination entry is normally already removed (overwriting the key
    // ran `graph_free` synchronously), but under lazy free that removal is
    // deferred, so we may displace it here. Drop off the main Redis thread
    // (see `register_graph` for the rationale).
    if let Some(displaced) = displaced {
        std::thread::spawn(move || drop(displaced));
    }
}

pub struct WriteMessage {
    pub bc: BlockedClient,
    pub query: Arc<str>,
    pub compact: bool,
    pub cached: bool,
    pub key_name: Arc<str>,
    pub timeout: Option<i64>,
    pub received_at: i64,
    pub enqueue_instant: Instant,
    pub waiting_id: u64,
    /// True if this is a GRAPH.PROFILE write: execute with profiling enabled and
    /// reply with the profile tree instead of the result set. Otherwise handled
    /// exactly like a GRAPH.QUERY write — same locking, commit and effects
    /// replication, so a profiled write does not diverge from replicas.
    pub profile: bool,
}

/// What `commit_and_replicate` needs to publish a finished write.
pub(crate) struct WriteQueryOk {
    pub(crate) graph: Arc<AtomicRefCell<Graph>>,
    pub(crate) effects_buffer: Option<Vec<u8>>,
    pub(crate) modified: bool,
}

/// Result from a read-path `execute_query` call, surfacing timing metadata
/// needed for telemetry.
pub struct ReadQueryResult {
    pub is_write: bool,
    pub cached: bool,
    pub execution_time_ms: f64,
    pub params_offset: usize,
    pub timed_out: bool,
}

/// Outcome of the GRAPH.PROFILE read/detect phase (`execute_profile`).
pub enum ProfileDetect {
    /// Read query — it was profiled and the reply was already sent on this path.
    ReadReplied,
    /// Write query — must run through the write queue like GRAPH.QUERY. The
    /// reply is the profile tree, emitted by `execute_query_write` in profile
    /// mode.
    Write,
}

/// Safe wrappers over Redis module FFI function pointers.
///
/// Redis guarantees these pointers are non-null after `RedisModule_Init` succeeds.
/// Centralising the `.expect()` here documents that invariant in one place and
/// keeps call sites free of `unwrap()` noise.
pub mod ffi {
    use redis_module::raw;
    use std::ffi::{CStr, CString};
    use std::os::raw::c_char;
    use std::ptr::null_mut;

    const MSG: &str = "Redis module FFI pointer not initialised (call RedisModule_Init first)";

    /// Wrap a user error string into a `CString`, replacing any NUL bytes so
    /// malformed strings never crash the module. NUL is extremely unlikely in
    /// practice but must be handled because `CString::new` rejects it.
    pub fn sanitise_error(err: impl Into<Vec<u8>>) -> CString {
        let mut bytes = err.into();
        for b in &mut bytes {
            if *b == 0 {
                *b = b' ';
            }
        }
        // Safety: we just stripped all interior NULs.
        CString::new(bytes).expect("interior NULs were sanitised above")
    }

    /// Block the calling client; returns the opaque blocked-client handle.
    ///
    /// # Safety
    /// `ctx` must be a valid Redis module context for the active command.
    pub unsafe fn block_client(
        ctx: *mut raw::RedisModuleCtx
    ) -> *mut raw::RedisModuleBlockedClient {
        let f = unsafe { raw::RedisModule_BlockClient }.expect(MSG);
        unsafe { f(ctx, None, None, None, 0) }
    }

    /// Unblock a client previously returned from [`block_client`].
    ///
    /// # Safety
    /// `bc` must be a valid pointer returned by `RedisModule_BlockClient`.
    pub unsafe fn unblock_client(bc: *mut raw::RedisModuleBlockedClient) {
        let f = unsafe { raw::RedisModule_UnblockClient }.expect(MSG);
        unsafe { f(bc, null_mut()) };
    }

    /// Obtain a thread-safe context bound to `bc`.
    ///
    /// # Safety
    /// `bc` must be a valid blocked-client handle.
    pub unsafe fn get_thread_safe_context(
        bc: *mut raw::RedisModuleBlockedClient
    ) -> *mut raw::RedisModuleCtx {
        let f = unsafe { raw::RedisModule_GetThreadSafeContext }.expect(MSG);
        unsafe { f(bc) }
    }

    /// Free a thread-safe context obtained from [`get_thread_safe_context`].
    ///
    /// # Safety
    /// `ctx` must be a valid thread-safe context.
    pub unsafe fn free_thread_safe_context(ctx: *mut raw::RedisModuleCtx) {
        let f = unsafe { raw::RedisModule_FreeThreadSafeContext }.expect(MSG);
        unsafe { f(ctx) };
    }

    /// Reply to the client with a NUL-terminated error string.
    ///
    /// # Safety
    /// `ctx` must be a valid reply context; `err` must outlive the call.
    pub unsafe fn reply_error(
        ctx: *mut raw::RedisModuleCtx,
        err: *const c_char,
    ) {
        raw::reply_with_error(ctx, err);
    }

    // NOTE: the GIL itself (`RedisModule_ThreadSafeContextLock` / `…Unlock`) is
    // deliberately *not* wrapped here. It is private to `crate::query_session`, whose
    // `QuerySession` and `hold_gil` are the only ways to take it, so the ordering rule
    // (global → per-graph → indexer, issue #726) cannot be bypassed.

    /// Mark `key_name` as modified so `WATCH` clients are notified. Must be
    /// called with the GIL held (see [`lock_thread_safe_ctx`]).
    ///
    /// # Safety
    /// `ctx` must be a valid Redis module context with the GIL held.
    pub unsafe fn signal_modified_key(
        ctx: *mut raw::RedisModuleCtx,
        key_name: &[u8],
    ) {
        let create = unsafe { raw::RedisModule_CreateString }.expect(MSG);
        let signal = unsafe { raw::RedisModule_SignalModifiedKey }.expect(MSG);
        let free = unsafe { raw::RedisModule_FreeString }.expect(MSG);
        unsafe {
            let rstr = create(ctx, key_name.as_ptr().cast(), key_name.len());
            signal(ctx, rstr);
            free(ctx, rstr);
        }
    }

    /// Yield execution back to Redis so it can serve other clients (e.g. PING)
    /// during long-running commands.
    ///
    /// `flags`: `REDISMODULE_YIELD_FLAG_NONE` (1) or `REDISMODULE_YIELD_FLAG_CLIENTS` (2).
    /// `busy_reply`: optional busy reply string (null pointer for default).
    ///
    /// # Safety
    /// `ctx` must be a valid Redis module context for the active command.
    pub unsafe fn yield_ctx(
        ctx: *mut raw::RedisModuleCtx,
        flags: i32,
    ) {
        let f = unsafe { raw::RedisModule_Yield }.expect(MSG);
        unsafe { f(ctx, flags, null_mut::<c_char>()) };
    }

    /// `REDISMODULE_YIELD_FLAG_CLIENTS` — allow Redis to process client
    /// commands (including PING) while we yield.
    pub const YIELD_FLAG_CLIENTS: i32 = 1 << 1;

    /// Start measuring the time a blocked client is waiting. This enables
    /// Redis to record the command in its own slowlog and latency reporting.
    ///
    /// # Safety
    /// `bc` must be a valid blocked-client handle.
    pub unsafe fn measure_time_start(bc: *mut raw::RedisModuleBlockedClient) {
        if let Some(f) = unsafe { raw::RedisModule_BlockedClientMeasureTimeStart } {
            unsafe { f(bc) };
        }
    }

    /// Stop measuring the time for a blocked client.
    ///
    /// # Safety
    /// `bc` must be a valid blocked-client handle.
    pub unsafe fn measure_time_end(bc: *mut raw::RedisModuleBlockedClient) {
        if let Some(f) = unsafe { raw::RedisModule_BlockedClientMeasureTimeEnd } {
            unsafe { f(bc) };
        }
    }

    /// Take an owned reference on `s`, independent of any command context.
    ///
    /// A NULL context is deliberate: `RM_HoldString` only registers the string in a
    /// context's auto-memory when one is passed, and such a string is freed when that
    /// command's context is destroyed — i.e. out from under a worker thread still
    /// holding it. With NULL the caller owns the reference and must
    /// [`free_string`] it.
    ///
    /// # Safety
    /// `s` must be a valid `RedisModuleString`, and the GIL must be held (the main
    /// thread holds it implicitly inside a command).
    pub unsafe fn hold_string(s: *mut raw::RedisModuleString) -> *mut raw::RedisModuleString {
        let f = unsafe { raw::RedisModule_HoldString }.expect(MSG);
        unsafe { f(null_mut(), s) }
    }

    /// Release a reference taken by [`hold_string`].
    ///
    /// # Safety
    /// `s` must have come from [`hold_string`] and not been freed already. The GIL
    /// must be held: these strings originate from client command arguments, and
    /// Redis requires the GIL for any access to those.
    pub unsafe fn free_string(s: *mut raw::RedisModuleString) {
        let f = unsafe { raw::RedisModule_FreeString }.expect(MSG);
        unsafe { f(null_mut(), s) };
    }

    /// Trim a held string's spare allocation.
    ///
    /// Mandatory, not an optimisation, for any string a background thread will
    /// reference: Redis may auto-trim retained strings when the command returns, and
    /// its own docs call that auto-trim "not thread safe … could result with data
    /// corruption" if a worker touches the string concurrently. Trimming up front
    /// leaves nothing for the auto-trim to do.
    ///
    /// # Safety
    /// `s` must be a valid string held by the caller, with the GIL held.
    pub unsafe fn trim_string_allocation(s: *mut raw::RedisModuleString) {
        let f = unsafe { raw::RedisModule_TrimStringAllocation }.expect(MSG);
        unsafe { f(s) };
    }

    /// Replicate `cmd` with a pre-built argument vector.
    ///
    /// Unlike `Context::replicate`, this does *not* build new strings from byte
    /// slices — `argv` is propagated by reference (Redis increments each refcount),
    /// so a large payload is not duplicated. `RM_Replicate`'s `"v"` format takes the
    /// vector and its length.
    ///
    /// # Safety
    /// `ctx` must be a valid module context and every entry of `argv` a valid string.
    /// The GIL must be held: propagation is flushed when it is released.
    pub unsafe fn replicate_argv(
        ctx: *mut raw::RedisModuleCtx,
        cmd: &CStr,
        argv: &[*mut raw::RedisModuleString],
    ) {
        const FMT: &[u8] = b"v\0";
        let f = unsafe { raw::RedisModule_Replicate }.expect(MSG);
        unsafe {
            f(
                ctx,
                cmd.as_ptr(),
                FMT.as_ptr().cast::<c_char>(),
                argv.as_ptr(),
                argv.len(),
            )
        };
    }
}

/// Sticky flag: set once any replica has ever attached (ReplicaChange
/// server event) and never cleared — a disconnected replica may resume
/// from the replication backlog, so effects buffers must keep being
/// built after the first attach. While false (and AOF is off) write
/// queries skip serializing effects entirely; the replication layer's
/// verbatim-query fallback is then a no-op propagation.
pub static REPLICATION_CONSUMERS: AtomicBool = AtomicBool::new(false);

pub struct ThreadedGraph {
    pub graph: MvccGraph,
    pub sender: MTx<Array<Box<WriteMessage>>>,
    pub receiver: Rx<Array<Box<WriteMessage>>>,
    pub write_loop: AtomicBool,
    pub slow_log: SlowLog,
}

unsafe impl Send for ThreadedGraph {}
unsafe impl Sync for ThreadedGraph {}

impl ThreadedGraph {
    pub fn new(
        cache_size: usize,
        name: &str,
    ) -> Self {
        let (sender, receiver) = bounded_blocking(1024);
        Self {
            graph: MvccGraph::new(16384, 16384, cache_size, name),
            sender,
            receiver,
            write_loop: AtomicBool::new(false),
            slow_log: SlowLog::new(),
        }
    }

    /// Create a `ThreadedGraph` from an existing `MvccGraph`.
    /// Used by the RDB load path.
    pub fn from_mvcc(graph: MvccGraph) -> Self {
        let (sender, receiver) = bounded_blocking(1024);
        Self {
            graph,
            sender,
            receiver,
            write_loop: AtomicBool::new(false),
            slow_log: SlowLog::new(),
        }
    }
}

/// Execute a read query, or detect that it is a write.
///
/// Takes `session` rather than a `&ThreadedGraph`: the session owns the per-graph
/// lock and may swap its read guard for a write guard mid-query, so a long-lived
/// borrow would be invalidated. See `crate::query_session`.
pub fn execute_query(
    session: &QuerySession,
    ctx: &Context,
    query: &str,
    compact: bool,
    write: bool,
    cmd: &str,
    per_query_timeout: Option<i64>,
) -> Result<ReadQueryResult, String> {
    let wall_start = Instant::now();
    let Plan {
        plan,
        cached,
        parameters,
        params_offset,
        ..
    } = session.with_graph(|tg| tg.graph.read().borrow().get_plan(query))?;
    let parameters = parameters
        .into_iter()
        .map(|(k, v)| Ok((k, evaluate_param(&v.root())?)))
        .collect::<Result<HashMap<_, _>, String>>()?;
    // Single pass over the plan: index DDL (CREATE/DROP INDEX) and a Commit
    // node both mean this is a write. Writes escalate to writer mode lazily
    // during execution (see `crate::query_session`), so no further plan
    // classification is needed here.
    let is_write = plan.iter().any(|n| {
        matches!(
            n,
            IR::Commit | IR::CreateIndex { .. } | IR::DropIndex { .. }
        )
    });
    let g = if is_write {
        if !write {
            return Err(String::from(
                "graph.RO_QUERY is to be executed only on read-only queries",
            ));
        }
        return Ok(ReadQueryResult {
            is_write: true,
            cached,
            execution_time_ms: 0.0,
            params_offset,
            timed_out: false,
        });
    } else {
        session.with_graph(|tg| tg.graph.read())
    };
    let timeout_ms = compute_effective_timeout(per_query_timeout, is_write)?;
    let runtime = Runtime::new(
        g,
        parameters,
        is_write,
        plan,
        false,
        (*CONFIGURATION_IMPORT_FOLDER.lock(ctx)).clone(),
        RESULTSET_SIZE.load(Ordering::Relaxed),
        false,
        timeout_ms,
        QUERY_MEM_CAPACITY.load(Ordering::Relaxed),
        Some(net_thread_usage),
        session,
    );
    let mut result = runtime.query()?;
    result.stats.cached = cached;
    if compact {
        reply_compact(ctx, &runtime, &result);
    } else {
        reply_verbose(ctx, &runtime, &result);
    }
    let latency = wall_start.elapsed().as_secs_f64() * 1000.0;
    let execution_time_ms = result.stats.execution_time;
    session.with_graph(|tg| tg.slow_log.add(cmd, query, params_offset, latency));
    Ok(ReadQueryResult {
        is_write: false,
        cached,
        execution_time_ms,
        params_offset,
        timed_out: false,
    })
}

/// GRAPH.PROFILE read/detect phase.
///
/// For a read query it runs with profiling and replies here; for a write it
/// returns [`ProfileDetect::Write`] so the caller routes it through the write
/// queue (exactly like GRAPH.QUERY), where `execute_query_write` executes and
/// replies it in profile mode.
pub fn execute_profile(
    session: &QuerySession,
    ctx: &Context,
    query: &str,
    per_query_timeout: Option<i64>,
) -> Result<ProfileDetect, String> {
    let Plan {
        plan, parameters, ..
    } = session.with_graph(|tg| tg.graph.read().borrow().get_plan(query))?;
    // Detect writes (mirrors `execute_query`). A write is NOT executed here — it
    // must go through the serialized write queue to get the locking protocol and
    // effects replication.
    let is_write = plan.iter().any(|n| {
        matches!(
            n,
            IR::Commit | IR::CreateIndex { .. } | IR::DropIndex { .. }
        )
    });
    if is_write {
        return Ok(ProfileDetect::Write);
    }
    let parameters = parameters
        .into_iter()
        .map(|(k, v)| Ok((k, evaluate_param(&v.root())?)))
        .collect::<Result<HashMap<_, _>, String>>()?;
    let g = session.with_graph(|tg| tg.graph.read());
    let timeout_ms = compute_effective_timeout(per_query_timeout, false)?;
    let runtime = Runtime::new(
        g,
        parameters,
        false,
        plan.clone(),
        false,
        String::new(),
        -1,
        true,
        timeout_ms,
        QUERY_MEM_CAPACITY.load(Ordering::Relaxed),
        Some(net_thread_usage),
        session,
    );
    let _ = runtime.query()?;
    reply_profile(ctx, &runtime, &plan);
    Ok(ProfileDetect::ReadReplied)
}

/// Execute a write query end-to-end: run it, commit, replicate, reply.
///
/// Takes the session **by value** so the writer window can be closed as early as
/// possible. The GIL is held from the query's first mutation through commit and
/// replication, and dropped *before* the result set is serialized — a large reply
/// would otherwise stall every other client for the whole serialization (the reply is
/// built from the query's own MVCC version, so it needs no lock).
///
/// The session swaps its read guard for a write guard when the query escalates, so a
/// long-lived `&ThreadedGraph` would be invalidated mid-query; every access here is a
/// short-lived `session.with_graph` borrow instead.
///
/// Returns `(params_offset, execution_time_ms)` for telemetry.
#[allow(clippy::too_many_arguments)]
pub fn execute_query_write(
    session: QuerySession,
    ctx: &Context,
    key_name: &Arc<str>,
    query: &str,
    compact: bool,
    first_cached: bool,
    per_query_timeout: Option<i64>,
    profile: bool,
) -> Result<(usize, f64), String> {
    let wall_start = Instant::now();
    let Plan {
        plan,
        parameters,
        params_offset,
        ..
    } = session.with_graph(|tg| tg.graph.read().borrow().get_plan(query))?;
    let cached = first_cached;
    let parameters = parameters
        .into_iter()
        .map(|(k, v)| Ok((k, evaluate_param(&v.root())?)))
        .collect::<Result<HashMap<_, _>, String>>()?;
    debug_assert!(plan.iter().any(|n| matches!(
        n,
        IR::Commit | IR::CreateIndex { .. } | IR::DropIndex { .. }
    )));

    let is_non_deterministic = plan_is_non_deterministic(&plan);
    // Compute the timeout before taking the MVCC write slot, so an error
    // here cannot leak the slot.
    let timeout_ms = compute_effective_timeout(per_query_timeout, true)?;

    // Claim the MVCC write slot: `MvccGraph::write` CASes the single-writer flag and
    // hands back a fresh COW version to build into — it is not a lock, so a writer
    // that already owns the slot makes this fail rather than block. We are still a
    // reader here, so an inline writer can be holding it; return a retryable error.
    // On this branch we did NOT claim it, so callers must not roll back — the slot and
    // its version belong to the other writer.
    let Some(g) = session.with_graph(|tg| tg.graph.write()) else {
        return Err("another write is in progress, retry the query".to_string());
    };
    let runtime = Runtime::new(
        g.clone(),
        parameters,
        true,
        plan,
        false,
        (*CONFIGURATION_IMPORT_FOLDER.lock(ctx)).clone(),
        RESULTSET_SIZE.load(Ordering::Relaxed),
        profile,
        timeout_ms,
        QUERY_MEM_CAPACITY.load(Ordering::Relaxed),
        Some(net_thread_usage),
        &session,
    );
    runtime.build_effects.set(
        REPLICATION_CONSUMERS.load(Ordering::Relaxed)
            || ctx.get_flags().contains(ContextFlags::AOF),
    );
    let mut result = match runtime.query() {
        Ok(r) => r,
        Err(err) => {
            // Resync the index against committed state before discarding the
            // private version: documents published by earlier `Commit`s are deleted
            // (entity never existed) or rewritten from committed values. Writer
            // mode, so no reader observes the interim state.
            let committed = session.with_graph(|tg| tg.graph.read());
            runtime.resync_published_indexes(&committed);
            // Release the MVCC write slot we took above (dropping the private
            // version with all its writes). Slot ownership lives here — callers
            // must NOT roll back.
            session.with_graph(|tg| tg.graph.rollback());
            return Err(err);
        }
    };

    // If any CreateIndex carries OPTIONS, the binary effect format can't
    // currently round-trip them — fall back to verbatim GRAPH.QUERY
    // replication by skipping the effects buffer entirely.
    let has_unencodable_index = runtime
        .plan
        .iter()
        .any(|node| matches!(node, IR::CreateIndex { options, .. } if options.is_some()));

    // Capture effects buffer before replying (pending data is still available)
    let mut effects_buffer = if has_unencodable_index {
        None
    } else {
        should_use_effects(is_non_deterministic, &runtime, result.stats.execution_time)
    };

    // Build index effects for CreateIndex / DropIndex IR nodes (not tracked by Pending)
    if !has_unencodable_index {
        effects_buffer = build_index_effects(&runtime, effects_buffer);
    }

    result.stats.cached = cached;
    let execution_time_ms = result.stats.execution_time;
    let modified = result.stats.nodes_created > 0
        || result.stats.nodes_deleted > 0
        || result.stats.relationships_created > 0
        || result.stats.relationships_deleted > 0
        || result.stats.properties_set > 0
        || result.stats.properties_removed > 0
        || result.stats.labels_added > 0
        || result.stats.labels_removed > 0
        || result.stats.indexes_created > 0
        || result.stats.indexes_dropped > 0
        || runtime.effects_count.get() > 0;

    // Commit → signal WATCH → replicate, the last work that needs the GIL.
    let wq = WriteQueryOk {
        graph: g,
        effects_buffer,
        modified,
    };
    let graph = session.graph_arc();
    if session
        .with_graph_mut(|tg| commit_and_replicate(tg, ctx, key_name, query, wq))
        .is_none()
    {
        // Never escalated, so the plan's `Commit` never ran and nothing was mutated —
        // `LIMIT 0` short-circuits above `Commit`, for instance. There is nothing to
        // publish or replicate; just release the MVCC write slot (only
        // commit/rollback clears it) and reply as usual.
        session.with_graph(|tg| tg.graph.rollback());
    }
    session.release_locks();

    // Writer window is closed. Serializing the reply reads this query's own MVCC
    // version through the runtime, so it needs no lock — and holding the GIL across it
    // would stall the server for as long as the result set takes to write.
    if profile {
        // GRAPH.PROFILE: reply with the annotated plan tree (per-operator records +
        // timing collected because the runtime ran profile=true), not the result set.
        // Everything else — effects, commit, replication — is identical to a
        // GRAPH.QUERY write.
        reply_profile(ctx, &runtime, &runtime.plan);
    } else if compact {
        reply_compact(ctx, &runtime, &result);
    } else {
        reply_verbose(ctx, &runtime, &result);
    }

    let latency = wall_start.elapsed().as_secs_f64() * 1000.0;
    let cmd_name = if profile {
        "GRAPH.PROFILE"
    } else {
        "GRAPH.QUERY"
    };
    graph
        .read()
        .slow_log
        .add(cmd_name, query, params_offset, latency);
    Ok((params_offset, execution_time_ms))
}

/// Reply with profile output: DFS walk of the plan tree, each line annotated
/// with `Records produced: N, Execution time: T.TTTTTT ms`.
/// Skips `Commit` nodes (internal implementation detail).
fn reply_profile(
    ctx: &Context,
    runtime: &Runtime,
    plan: &orx_tree::DynTree<IR>,
) {
    let all_ops: Vec<_> = plan.root().indices::<Dfs>().collect();
    // Filter out Commit nodes and adjust depth accordingly.
    let ops: Vec<_> = all_ops
        .iter()
        .filter(|idx| !matches!(plan.node(**idx).data(), IR::Commit))
        .collect();
    let profile_data = runtime.profile_data.borrow();
    raw::reply_with_array(ctx.ctx, ops.len() as _);
    for idx in ops {
        let node = plan.node(*idx);
        // Calculate effective depth (subtract number of Commit ancestors).
        let mut depth = node.depth();
        let mut cur = *idx;
        while let Some(parent) = plan.node(cur).parent() {
            if matches!(parent.data(), IR::Commit) {
                depth -= 1;
            }
            cur = parent.idx();
        }
        let (records, time) = profile_data
            .get(idx)
            .copied()
            .unwrap_or((0, std::time::Duration::ZERO));
        let time_ms = time.as_secs_f64() * 1000.0;
        let line = format!(
            "{}{} | Records produced: {records}, Execution time: {time_ms:.6} ms",
            "    ".repeat(depth),
            node.data()
        );
        raw::reply_with_string_buffer(ctx.ctx, line.as_ptr().cast::<c_char>(), line.len());
    }
}

pub struct BlockedClient {
    pub inner: *mut raw::RedisModuleBlockedClient,
}

unsafe impl Send for BlockedClient {}
unsafe impl Sync for BlockedClient {}

impl BlockedClient {
    /// Block the calling client and start measuring time for Redis slowlog.
    ///
    /// # Safety
    /// `ctx` must be a valid Redis module context for the active command.
    pub unsafe fn new(ctx: *mut raw::RedisModuleCtx) -> Self {
        let inner = unsafe { ffi::block_client(ctx) };
        unsafe { ffi::measure_time_start(inner) };
        Self { inner }
    }
}

impl Drop for BlockedClient {
    fn drop(&mut self) {
        unsafe {
            ffi::measure_time_end(self.inner);
            ffi::unblock_client(self.inner);
        }
    }
}

/// Compute the effective timeout in milliseconds for a query.
///
/// Rules:
/// - Per-query timeout cannot exceed TIMEOUT_MAX (if set).
/// - Per-query timeout is only applied to read queries (write queries ignore it).
/// - Falls back to TIMEOUT_DEFAULT, then deprecated TIMEOUT.
/// - Returns None for unlimited.
fn compute_effective_timeout(
    per_query_timeout: Option<i64>,
    is_write: bool,
) -> Result<Option<u64>, String> {
    let timeout_max = TIMEOUT_MAX.load(Ordering::Relaxed);
    let timeout_default = TIMEOUT_DEFAULT.load(Ordering::Relaxed);
    let timeout_legacy = TIMEOUT.load(Ordering::Relaxed);

    // Per-query timeout: enforce TIMEOUT_MAX limit, skip for writes
    if let Some(pq) = per_query_timeout {
        if timeout_max > 0 && pq > timeout_max {
            return Err("The query TIMEOUT parameter value cannot exceed the TIMEOUT_MAX configuration parameter value".to_string());
        }
        // Per-query timeout is ignored for write queries
        if !is_write && pq > 0 {
            return Ok(Some(pq as u64));
        }
    }

    // Global config fallback
    if timeout_default > 0 {
        return Ok(Some(timeout_default as u64));
    }
    if timeout_max > 0 {
        return Ok(Some(timeout_max as u64));
    }
    if timeout_legacy > 0 {
        // Legacy TIMEOUT: only apply to read queries
        if !is_write {
            return Ok(Some(timeout_legacy as u64));
        }
    }
    Ok(None)
}

/// Replies `[error, current_version]` for a client-supplied version that does
/// not match the graph's current schema version.
pub(crate) fn reply_invalid_graph_version(
    ctx: &Context,
    current_version: i64,
) {
    const ERR: &std::ffi::CStr = c"ERR invalid graph version";
    raw::reply_with_array(ctx.ctx, 2);
    raw::reply_with_error(ctx.ctx, ERR.as_ptr());
    raw::reply_with_long_long(ctx.ctx, current_version);
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn query_mut(
    ctx: &Context,
    graph: &Arc<RwLock<ThreadedGraph>>,
    query: &str,
    compact: bool,
    write: bool,
    track_mem: bool,
    key_name: Arc<str>,
    per_query_timeout: Option<i64>,
    version_check: Option<u64>,
) -> RedisResult {
    // Reject a stale client-supplied schema version before doing any work.
    if let Some(provided_version) = version_check {
        let current_schema_version = graph.read().graph.read().borrow().schema_version;
        if provided_version != current_schema_version {
            reply_invalid_graph_version(ctx, current_schema_version as i64);
            return Ok(RedisValue::NoReply);
        }
    }

    // Contexts that cannot block run inline on this thread — see `must_run_inline` for
    // why each flag is in the set. Two of them are load-bearing here: a replica has to
    // apply the query before the handler returns, or Redis advances the replication
    // offset while the write is still queued and the master's WAIT reports the replica
    // in-sync when it is not; and an AOF-replay client must never be blocked, which
    // used to crash the server on restart (#2421).
    if must_run_inline(ctx) {
        return query_sync(
            ctx,
            graph,
            query,
            compact,
            write,
            &key_name,
            per_query_timeout,
        );
    }

    // Check pending queries limit before dispatching.
    let max = MAX_QUEUED_QUERIES.load(Ordering::Relaxed) as usize;
    if pending_count() >= max {
        let bc = BlockedClient {
            inner: unsafe { ffi::block_client(ctx.ctx) },
        };
        let err_ctx = unsafe { ffi::get_thread_safe_context(bc.inner) };
        let err_ctx = Context::new(err_ctx);
        let cerr = ffi::sanitise_error("Max pending queries exceeded");
        unsafe { ffi::reply_error(err_ctx.ctx, cerr.as_ptr()) };
        drop(bc);
        unsafe { ffi::free_thread_safe_context(err_ctx.ctx) };
        return Ok(RedisValue::NoReply);
    }

    let bc = unsafe { BlockedClient::new(ctx.ctx) };
    let graph = graph.clone();
    let query: Arc<str> = Arc::from(query);
    let received_at = telemetry::unix_now_secs();
    // Monotonic clock captured at dispatch time so the worker can attribute the
    // thread-pool queueing delay (receipt → worker start) to "Wait duration".
    let dispatch_instant = Instant::now();
    // The query is *waiting* from now until a worker picks it up: that queueing
    // delay is exactly what `GRAPH.INFO WaitingQueries` reports, and in C it is
    // read straight off the thread pool's task queue. Our jobs are opaque
    // closures, so the registry stands in for that queue — register here, on the
    // dispatching thread, and promote to "running" inside the worker.
    // Guarded, not a bare id: `spawn` drops the job when the pool is shutting
    // down, and a worker can panic before promoting. Either would leave the
    // query in `GRAPH.INFO WaitingQueries` forever, so the entry's lifetime is
    // tied to the closure that owns it.
    let mut pool_waiting = telemetry::WaitingEntry::register(received_at, &key_name, &query);
    spawn(
        move || {
            let mem_capacity = QUERY_MEM_CAPACITY.load(Ordering::Relaxed);
            let enforce_mem = track_mem || mem_capacity > 0;
            if enforce_mem {
                reset_counter();
                enable_tracking();
            }
            // Sync query timeout to UDF JS runtime
            graph::udf::js_context::JS_TIMEOUT_MS
                .store(TIMEOUT_DEFAULT.load(Ordering::Relaxed), Ordering::Relaxed);

            let g = graph.clone();
            let binding = graph.clone();
            // Every query runs under a session: a read never escalates, a write
            // escalates at its first `Commit`.
            let session = QuerySession::begin(&binding);
            let bc = bc;
            let ctx = unsafe { ffi::get_thread_safe_context(bc.inner) };
            let ctx = Context::new(ctx);

            let cmd = if write {
                "GRAPH.QUERY"
            } else {
                "GRAPH.RO_QUERY"
            };

            // Leaves the waiting list and joins the running list in one step. The
            // fallback cannot normally happen (only this worker consumes the id);
            // it keeps the running report correct rather than silently empty.
            let running_id = pool_waiting.promote().unwrap_or_else(|| {
                telemetry::register_running(received_at, &key_name, &query, false)
            });
            let wall_start = Instant::now();
            // Time spent waiting in the thread pool before this worker started.
            let wait_ms = wall_start.duration_since(dispatch_instant).as_secs_f64() * 1000.0;
            let res = execute_query(
                &session,
                &ctx,
                &query,
                compact,
                write,
                cmd,
                per_query_timeout,
            );
            let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
            telemetry::unregister_running(running_id);

            // Log memory tracking BEFORE freeing the context.
            if enforce_mem {
                if track_mem {
                    let (allocated, deallocated) = current_thread_usage();
                    ctx.log(
                        redis_module::logging::RedisLogLevel::Notice,
                        &format!(
                            "Allocated: {allocated} bytes, Deallocated: {deallocated} bytes, Net: {}",
                            allocated as isize - deallocated as isize
                        ),
                    );
                }
                disable_tracking();
            }

            match res {
                Ok(read_result) => {
                    if read_result.is_write {
                        let waiting_id =
                            telemetry::register_waiting(received_at, &key_name, &query);
                        let msg = Box::new(WriteMessage {
                            bc,
                            query,
                            compact,
                            cached: read_result.cached,
                            key_name,
                            timeout: per_query_timeout,
                            received_at,
                            enqueue_instant: Instant::now(),
                            waiting_id,
                            profile: false,
                        });
                        let send = session.with_graph(|tg| tg.sender.send(msg));
                        if let Err(send_err) = send {
                            let msg = send_err.0;
                            telemetry::unregister_waiting(msg.waiting_id);
                            let cerr = ffi::sanitise_error(
                                "ERR graph write queue unavailable".to_string(),
                            );
                            unsafe { ffi::reply_error(ctx.ctx, cerr.as_ptr()) };
                            drop(msg.bc);
                            unsafe { ffi::free_thread_safe_context(ctx.ctx) };
                            return;
                        }
                        // Drop our reader session BEFORE the write loop: it takes its
                        // own, and escalating would block forever behind this read
                        // lock.
                        drop(session);
                        // BlockedClient now lives in the queued WriteMessage; this
                        // worker's thread-safe context is no longer needed.
                        unsafe { ffi::free_thread_safe_context(ctx.ctx) };
                        process_write_queued_query(&g);
                    } else {
                        // Read query completed — write telemetry
                        let query_text = &query[read_result.params_offset..];
                        let params_text = &query[..read_result.params_offset];
                        let exec_ms = read_result.execution_time_ms;
                        let report_ms = (wall_ms - exec_ms).max(0.0);

                        let entry = telemetry::TelemetryEntry {
                            received_at,
                            query: telemetry::truncate(query_text.trim_start()),
                            params: telemetry::truncate(params_text.trim()),

                            wait_duration_ms: wait_ms,
                            execution_duration_ms: exec_ms,
                            report_duration_ms: report_ms,
                            utilized_cache: read_result.cached,
                            is_write: false,
                            timed_out: read_result.timed_out,
                        };
                        telemetry::enqueue_entry(&key_name, entry);
                        drop(bc);
                        unsafe { ffi::free_thread_safe_context(ctx.ctx) };
                    }
                }
                Err(err) => {
                    let cerr = ffi::sanitise_error(err);
                    unsafe { ffi::reply_error(ctx.ctx, cerr.as_ptr()) };
                    drop(bc);
                    unsafe { ffi::free_thread_safe_context(ctx.ctx) };
                }
            }
        },
        None,
    );
    Ok(RedisValue::NoReply)
}

/// Execute a query synchronously on the calling thread.
/// Used when inside MULTI/EXEC where blocking is not allowed.
fn query_sync(
    ctx: &Context,
    graph: &Arc<RwLock<ThreadedGraph>>,
    query: &str,
    compact: bool,
    write: bool,
    key_name: &Arc<str>,
    per_query_timeout: Option<i64>,
) -> RedisResult {
    // First pass: parse + detect if write, execute reads inline.
    // Sync query timeout to UDF JS runtime
    graph::udf::js_context::JS_TIMEOUT_MS
        .store(TIMEOUT_DEFAULT.load(Ordering::Relaxed), Ordering::Relaxed);

    let mem_capacity = QUERY_MEM_CAPACITY.load(Ordering::Relaxed);
    if mem_capacity > 0 {
        reset_counter();
        enable_tracking();
    }

    let received_at = telemetry::unix_now_secs();
    let cmd = if write {
        "GRAPH.QUERY"
    } else {
        "GRAPH.RO_QUERY"
    };
    let wall_start = Instant::now();
    // Telemetry stores the query as an `Arc<str>`; build it once and share it
    // across the running-registry registrations on this (rare) sync path.
    let query_arc: Arc<str> = Arc::from(query);
    let running_id = telemetry::register_running(received_at, key_name, &query_arc, false);
    let res = {
        let session = QuerySession::begin(graph);
        execute_query(&session, ctx, query, compact, write, cmd, per_query_timeout)
    };
    let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
    telemetry::unregister_running(running_id);
    match res {
        Ok(read_result) => {
            if read_result.is_write {
                // Write path: acquire exclusive lock and execute.
                let write_start = Instant::now();
                let running_id2 =
                    telemetry::register_running(received_at, key_name, &query_arc, false);
                // On the main thread the GIL is already held, so start as a writer:
                // the order is satisfied and there is nothing to escalate.
                let res = execute_query_write(
                    QuerySession::begin_writer(graph),
                    ctx,
                    key_name,
                    query,
                    compact,
                    read_result.cached,
                    per_query_timeout,
                    false,
                );
                let write_wall_ms = write_start.elapsed().as_secs_f64() * 1000.0;
                telemetry::unregister_running(running_id2);
                match res {
                    Ok((params_offset, exec_ms)) => {
                        // Write telemetry
                        let query_text = &query[params_offset..];
                        let params_text = &query[..params_offset];
                        let report_ms = (write_wall_ms - exec_ms).max(0.0);

                        let entry = telemetry::TelemetryEntry {
                            received_at,
                            query: telemetry::truncate(query_text.trim_start()),
                            params: telemetry::truncate(params_text.trim()),

                            wait_duration_ms: 0.0,
                            execution_duration_ms: exec_ms,
                            report_duration_ms: report_ms,
                            utilized_cache: read_result.cached,
                            is_write: true,
                            timed_out: false,
                        };
                        telemetry::enqueue_entry(key_name, entry);
                    }
                    Err(err) => {
                        // execute_query_write already released the MVCC write
                        // slot on failure (rollback lives there now); just
                        // surface the error. Dropping the session releases the
                        // write lock.
                        return Err(redis_module::RedisError::String(err));
                    }
                }
            } else {
                // Read completed — write telemetry
                let query_text = &query[read_result.params_offset..];
                let params_text = &query[..read_result.params_offset];
                let exec_ms = read_result.execution_time_ms;
                let report_ms = (wall_ms - exec_ms).max(0.0);
                let entry = telemetry::TelemetryEntry {
                    received_at,
                    query: telemetry::truncate(query_text.trim_start()),
                    params: telemetry::truncate(params_text.trim()),

                    wait_duration_ms: 0.0,
                    execution_duration_ms: exec_ms,
                    report_duration_ms: report_ms,
                    utilized_cache: read_result.cached,
                    is_write: false,
                    timed_out: read_result.timed_out,
                };
                telemetry::enqueue_entry(key_name, entry);
            }
        }
        Err(err) => {
            return Err(redis_module::RedisError::String(err));
        }
    }
    Ok(RedisValue::NoReply)
}

#[inline]
pub fn profile_mut(
    ctx: &Context,
    graph: &Arc<RwLock<ThreadedGraph>>,
    query: &str,
    key_name: &Arc<str>,
    per_query_timeout: Option<i64>,
) -> RedisResult {
    // Contexts that cannot block run inline (see `query_mut` for the rationale, and
    // `must_run_inline` for the flag set).
    if must_run_inline(ctx) {
        return profile_sync(ctx, graph, query, key_name, per_query_timeout);
    }

    let bc = unsafe { BlockedClient::new(ctx.ctx) };
    let graph = graph.clone();
    let query: Arc<str> = Arc::from(query);
    let key_name = key_name.clone();
    let received_at = telemetry::unix_now_secs();
    spawn(
        move || {
            let mem_capacity = QUERY_MEM_CAPACITY.load(Ordering::Relaxed);
            if mem_capacity > 0 {
                reset_counter();
                enable_tracking();
            }
            let g = graph.clone();
            let binding = graph.clone();
            let session = QuerySession::begin(&binding);
            let bc = bc;
            let ctx = unsafe { ffi::get_thread_safe_context(bc.inner) };
            let ctx = Context::new(ctx);

            let res = execute_profile(&session, &ctx, &query, per_query_timeout);

            // Read-phase memory tracking ends here; a write is tracked
            // separately inside process_write_queued_query.
            if mem_capacity > 0 {
                disable_tracking();
            }

            match res {
                // Read query — already profiled and replied on this path.
                Ok(ProfileDetect::ReadReplied) => {
                    drop(bc);
                    unsafe { ffi::free_thread_safe_context(ctx.ctx) };
                }
                // Write query — enqueue exactly like GRAPH.QUERY so it runs through
                // the write loop; `execute_query_write` in profile mode replies with
                // the profile tree instead of the result set.
                Ok(ProfileDetect::Write) => {
                    let waiting_id = telemetry::register_waiting(received_at, &key_name, &query);
                    let msg = Box::new(WriteMessage {
                        bc,
                        query,
                        compact: false,
                        cached: false,
                        key_name,
                        timeout: per_query_timeout,
                        received_at,
                        enqueue_instant: Instant::now(),
                        waiting_id,
                        profile: true,
                    });
                    let send = session.with_graph(|tg| tg.sender.send(msg));
                    if let Err(send_err) = send {
                        let msg = send_err.0;
                        telemetry::unregister_waiting(msg.waiting_id);
                        let cerr =
                            ffi::sanitise_error("ERR graph write queue unavailable".to_string());
                        unsafe { ffi::reply_error(ctx.ctx, cerr.as_ptr()) };
                        drop(msg.bc);
                        unsafe { ffi::free_thread_safe_context(ctx.ctx) };
                        return;
                    }
                    // Release the reader session before the write loop — see
                    // `query_mut`.
                    drop(session);
                    // BlockedClient now lives in the queued WriteMessage.
                    unsafe { ffi::free_thread_safe_context(ctx.ctx) };
                    process_write_queued_query(&g);
                }
                Err(err) => {
                    let cerr = ffi::sanitise_error(err);
                    unsafe { ffi::reply_error(ctx.ctx, cerr.as_ptr()) };
                    drop(bc);
                    unsafe { ffi::free_thread_safe_context(ctx.ctx) };
                }
            }
        },
        None,
    );
    Ok(RedisValue::NoReply)
}

fn profile_sync(
    ctx: &Context,
    graph: &Arc<RwLock<ThreadedGraph>>,
    query: &str,
    key_name: &Arc<str>,
    per_query_timeout: Option<i64>,
) -> RedisResult {
    let mem_capacity = QUERY_MEM_CAPACITY.load(Ordering::Relaxed);
    if mem_capacity > 0 {
        reset_counter();
        enable_tracking();
    }
    let res = {
        let session = QuerySession::begin(graph);
        execute_profile(&session, ctx, query, per_query_timeout)
    };
    match res {
        // Read query — already profiled and replied.
        Ok(ProfileDetect::ReadReplied) => {}
        // Write query. On the main thread, which already holds the GIL implicitly,
        // so start as a writer and every global-lock acquire below is a no-op.
        // Mirrors query_sync's write branch, so a profiled write reaches replicas.
        Ok(ProfileDetect::Write) => {
            let res = execute_query_write(
                QuerySession::begin_writer(graph),
                ctx,
                key_name,
                query,
                false,
                false,
                per_query_timeout,
                true,
            );
            match res {
                Ok(_) => {}
                Err(err) => {
                    // execute_query_write already released the MVCC slot on
                    // failure (rollback lives there); just surface the error.
                    if mem_capacity > 0 {
                        disable_tracking();
                    }
                    return Err(redis_module::RedisError::String(err));
                }
            }
        }
        Err(err) => {
            if mem_capacity > 0 {
                disable_tracking();
            }
            return Err(redis_module::RedisError::String(err));
        }
    }
    if mem_capacity > 0 {
        disable_tracking();
    }
    Ok(RedisValue::NoReply)
}

/// Commit a successfully-executed write and replicate it.
///
/// Runs in writer mode — GIL *and* per-graph write lock — so the `commit` Arc-swap
/// is fork-safe (#452) and commit+replicate are atomic against inline main-thread
/// writers. The last work a write does under the GIL; its caller releases the locks
/// immediately afterwards and only then serializes the reply.
pub(crate) fn commit_and_replicate(
    g: &mut ThreadedGraph,
    ctx: &Context,
    key_name: &Arc<str>,
    query: &str,
    wq: WriteQueryOk,
) {
    // Index document changes were already applied by each `CommitOp` while this
    // query held the write lock (so a later operator in the same query could see
    // them); nothing left to publish but the matrix version.
    g.graph.commit(Arc::clone(&wq.graph));
    // Signal the key as modified so WATCH gets triggered.
    unsafe { ffi::signal_modified_key(ctx.ctx, key_name.as_bytes()) };
    // Send replication while the GIL is held.
    if wq.modified {
        replicate_effects(ctx, key_name, wq.effects_buffer, query);
    }
}

pub fn process_write_queued_query(graph: &Arc<RwLock<ThreadedGraph>>) {
    if graph
        .read()
        .write_loop
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Acquire)
        .is_err()
    {
        return;
    }
    loop {
        // Dequeue under a read lock — the channel and `write_loop` are
        // self-synchronized, and holding only L1-read here lets reads and
        // BGSAVE interleave between messages.
        let msg = {
            let g = graph.read();
            match g.receiver.try_recv() {
                Ok(msg) => msg,
                Err(_) => {
                    g.write_loop.store(false, Ordering::Release);
                    if g.receiver.is_empty() {
                        return;
                    }
                    if g.write_loop
                        .compare_exchange(false, true, Ordering::Acquire, Ordering::Acquire)
                        .is_err()
                    {
                        return;
                    }
                    // A producer raced an enqueue after we saw empty; keep draining.
                    continue;
                }
            }
        };
        let WriteMessage {
            bc,
            query,
            compact,
            cached,
            key_name,
            timeout: per_query_timeout,
            received_at,
            enqueue_instant,
            waiting_id,
            profile,
        } = *msg;
        let running_id = telemetry::transition_waiting_to_running(waiting_id);
        let write_start = Instant::now();
        let wait_ms = write_start.duration_since(enqueue_instant).as_secs_f64() * 1000.0;
        let ctx = unsafe { ffi::get_thread_safe_context(bc.inner) };
        let ctx = Context::new(ctx);
        let mem_capacity = QUERY_MEM_CAPACITY.load(Ordering::Relaxed);
        if mem_capacity > 0 {
            reset_counter();
            enable_tracking();
        }

        // Start as a reader and let the query escalate on its first mutation —
        // `crate::query_session` documents the protocol and why the order matters
        // (#726). Two consequences here: the writer window covers everything from
        // the first mutation through commit+replicate, so `Commit` publishes index
        // documents inline (visible to later operators in *this* query, readers
        // excluded) and `commit`'s Arc-swap still happens under the GIL (#452); and
        // index DDL needs no special path, since creating a spec is just a mutation.
        //
        // The session lives only for this block, so its locks are released — the read
        // lock, or the GIL + write lock if the query escalated — on every path out,
        // including the error returns, and before the reply below.
        //
        // Plain `begin`: a client sent this write to this instance, so escalation
        // re-authorizes it against the pause state and role that are live *then* — both
        // can have changed since it was admitted and queued. Replicated commands never
        // reach here; they run inline (see `query_mut`).
        let res = execute_query_write(
            QuerySession::begin(graph),
            &ctx,
            &key_name,
            &query,
            compact,
            cached,
            per_query_timeout,
            profile,
        );

        if mem_capacity > 0 {
            disable_tracking();
        }
        let write_wall_ms = write_start.elapsed().as_secs_f64() * 1000.0;
        if let Some(rid) = running_id {
            telemetry::unregister_running(rid);
        }
        match res {
            Ok((params_offset, exec_ms)) => {
                unsafe { ffi::free_thread_safe_context(ctx.ctx) };
                let query_text = &query[params_offset..];
                let params_text = &query[..params_offset];
                let report_ms = (write_wall_ms - exec_ms).max(0.0);
                let entry = telemetry::TelemetryEntry {
                    received_at,
                    query: telemetry::truncate(query_text.trim_start()),
                    params: telemetry::truncate(params_text.trim()),

                    wait_duration_ms: wait_ms,
                    execution_duration_ms: exec_ms,
                    report_duration_ms: report_ms,
                    utilized_cache: cached,
                    is_write: true,
                    timed_out: false,
                };
                telemetry::enqueue_entry(&key_name, entry);
                drop(bc);
            }
            Err(err) => {
                let cerr = ffi::sanitise_error(err);
                // reply_error on a blocked-client ThreadSafeContext writes into
                // the client's reply buffer; the buffer is synchronized by the
                // blocked-client machinery, so the module GIL is not required.
                unsafe {
                    ffi::reply_error(ctx.ctx, cerr.as_ptr());
                    ffi::free_thread_safe_context(ctx.ctx);
                };
                drop(bc);
            }
        }
    }
}

/// Decide whether to use effects replication and get the pre-built buffer.
/// The buffer was built in `CommitOp` before pending was cleared.
/// Returns Some(buffer) if effects should be sent, None for verbatim replication.
pub(crate) fn should_use_effects(
    is_non_deterministic: bool,
    runtime: &Runtime,
    exec_time_ms: f64,
) -> Option<Vec<u8>> {
    let threshold = EFFECTS_THRESHOLD.load(Ordering::Relaxed);

    let buf = runtime.effects_buffer.borrow_mut().take();
    let buf = match buf {
        Some(b) if b.len() > 1 => b, // > 1 because version byte alone means empty
        _ => return None,
    };

    let n_effects = runtime.effects_count.get();

    let use_effects = if is_non_deterministic || threshold == 0 {
        true
    } else if n_effects == 0 {
        false
    } else {
        let avg_mod_time_us = (exec_time_ms / n_effects as f64) * 1000.0;
        avg_mod_time_us > threshold as f64
    };

    if use_effects { Some(buf) } else { None }
}

/// Send replication: GRAPH.EFFECT with binary buffer, or verbatim query replay.
fn replicate_effects(
    ctx: &Context,
    key_name: &str,
    effects_buffer: Option<Vec<u8>>,
    query: &str,
) {
    if let Some(buf) = effects_buffer {
        let args: &[&[u8]] = &[key_name.as_bytes(), &buf];
        ctx.replicate("GRAPH.EFFECT", args);
    } else {
        let args: &[&[u8]] = &[key_name.as_bytes(), query.as_bytes()];
        ctx.replicate("GRAPH.QUERY", args);
    }
}

/// Encode IndexType as u8 tag for effects buffer.
const fn index_type_tag(it: &graph::index::IndexType) -> u8 {
    use graph::index::IndexType;
    match it {
        IndexType::Range => 0,
        IndexType::Fulltext => 1,
        IndexType::Vector => 2,
    }
}

/// Encode EntityType as u8 tag for effects buffer.
const fn entity_type_tag(et: &graph::entity_type::EntityType) -> u8 {
    use graph::entity_type::EntityType;
    match et {
        EntityType::Node => 0,
        EntityType::Relationship => 1,
    }
}

/// Scan the plan for CreateIndex / DropIndex IR nodes and append their
/// effects to the buffer. Returns the (possibly new) effects buffer.
/// Caller must ensure no CreateIndex carries OPTIONS — those can't currently
/// round-trip in the binary effect format and require verbatim replication.
pub(crate) fn build_index_effects(
    runtime: &Runtime,
    mut effects_buffer: Option<Vec<u8>>,
) -> Option<Vec<u8>> {
    for node in runtime.plan.iter() {
        match node {
            IR::CreateIndex {
                label,
                attrs,
                index_type,
                entity_type,
                options: _,
            } => {
                let buf = effects_buffer.get_or_insert_with(|| vec![EFFECTS_VERSION]);
                buf.push(EFFECT_CREATE_INDEX);
                buf.push(index_type_tag(index_type));
                buf.push(entity_type_tag(entity_type));
                write_string(buf, label);
                write_u16(buf, attrs.len() as u16);
                for attr in attrs {
                    write_string(buf, attr);
                }
            }
            IR::DropIndex {
                label,
                attrs,
                index_type,
                entity_type,
            } => {
                let buf = effects_buffer.get_or_insert_with(|| vec![EFFECTS_VERSION]);
                buf.push(EFFECT_DROP_INDEX);
                buf.push(index_type_tag(index_type));
                buf.push(entity_type_tag(entity_type));
                write_string(buf, label);
                write_u16(buf, attrs.len() as u16);
                for attr in attrs {
                    write_string(buf, attr);
                }
            }
            _ => {}
        }
    }
    effects_buffer
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn graph_free(value: *mut c_void) {
    unsafe {
        // `value` is the Box pointer Redis received from `Box::into_raw`; it
        // points to the heap slot holding the Arc, NOT to the RwLock inside
        // ArcInner. Compare against the boxed Arc's `data_ptr()` so the
        // registry entry whose clone shares this inner allocation is removed.
        let boxed = Box::from_raw(value.cast::<Arc<RwLock<ThreadedGraph>>>());
        let data_ptr = boxed.data_ptr() as usize;
        let removed: Vec<Arc<RwLock<ThreadedGraph>>> = {
            let mut reg = GRAPH_REGISTRY.lock();
            #[allow(clippy::needless_collect)]
            let keys: Vec<String> = reg
                .iter()
                .filter(|(_, arc)| arc.data_ptr() as usize == data_ptr)
                .map(|(k, _)| k.clone())
                .collect();
            keys.into_iter().filter_map(|k| reg.remove(&k)).collect()
        };
        // Drop off the main Redis thread so Index::drop routes through the
        // GIL-acquiring path (see register_graph for the rationale).
        std::thread::spawn(move || {
            drop(boxed);
            drop(removed);
        });
    }
}
