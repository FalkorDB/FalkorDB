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
//! queue guarded by `write_loop`.

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
/// `graph_free` removes a graph from the registry when its key is deleted,
/// overwritten or expired, so a missing entry means the key is gone. A query that
/// escalated to writer mode checks this before committing: the key can disappear
/// (`GRAPH.DELETE`, `FLUSHALL`, overwrite) during the window where escalation has
/// released the read lock and not yet taken the write lock. Our `Arc` keeps the
/// object alive, so this is not a use-after-free — but committing into a graph
/// nobody can reach any more would silently discard the write. Mirrors the
/// re-open-and-verify step in C's `QueryCtx_AcquireWriteLock`.
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
    /// exactly like a GRAPH.QUERY write — same two-phase commit, DDL routing, and
    /// effects replication (so a profiled write no longer diverges from replicas).
    pub profile: bool,
}

pub struct WriteQueryOk {
    pub graph: Arc<AtomicRefCell<Graph>>,
    pub effects_buffer: Option<Vec<u8>>,
    pub modified: bool,
    pub execution_time_ms: f64,
    pub params_offset: usize,
}
type WriteQueryResult = Result<WriteQueryOk, String>;

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
    use std::ffi::CString;
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

    /// Acquire the global Redis lock for a thread-safe context.
    ///
    /// # Safety
    /// `ctx` must be a valid thread-safe context.
    pub unsafe fn lock_thread_safe_ctx(ctx: *mut raw::RedisModuleCtx) {
        // Lock-order guard (#726): must not hold L1 when acquiring the GIL.
        crate::query_lock::assert_safe_to_take_host_lock();
        let f = unsafe { raw::RedisModule_ThreadSafeContextLock }.expect(MSG);
        unsafe { f(ctx) };
    }

    /// Release the global Redis lock previously acquired with
    /// [`lock_thread_safe_ctx`].
    ///
    /// # Safety
    /// `ctx` must be a valid thread-safe context whose lock the current
    /// thread already holds.
    pub unsafe fn unlock_thread_safe_ctx(ctx: *mut raw::RedisModuleCtx) {
        let f = unsafe { raw::RedisModule_ThreadSafeContextUnlock }.expect(MSG);
        unsafe { f(ctx) };
    }

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

/// Execute a read query (or detect that it is a write) for the current thread's
/// lock session.
///
/// Free function, not a method: the session owns the per-graph lock and may swap
/// its read guard for a write guard mid-query, so a long-lived `&ThreadedGraph`
/// would be invalidated. See `crate::query_lock`.
pub fn execute_query(
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
    } = crate::query_lock::with_graph(|tg| tg.graph.read().borrow().get_plan(query))?;
    let parameters = parameters
        .into_iter()
        .map(|(k, v)| Ok((k, evaluate_param(&v.root())?)))
        .collect::<Result<HashMap<_, _>, String>>()?;
    // Single pass over the plan: index DDL (CREATE/DROP INDEX) and a Commit
    // node both mean this is a write. Writes escalate to writer mode lazily
    // during execution (see `crate::query_lock`), so no further plan
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
        crate::query_lock::with_graph(|tg| tg.graph.read())
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
        &crate::query_lock::RedisQueryLock,
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
    drop(result);
    drop(runtime);
    crate::query_lock::with_graph(|tg| tg.slow_log.add(cmd, query, params_offset, latency));
    Ok(ReadQueryResult {
        is_write: false,
        cached,
        execution_time_ms,
        params_offset,
        timed_out: false,
    })
}

/// GRAPH.PROFILE read/detect phase for the current thread's lock session.
///
/// For a read query it runs with profiling and replies here; for a write it
/// returns [`ProfileDetect::Write`] so the caller routes it through the write
/// queue (exactly like GRAPH.QUERY), where `execute_query_write` executes and
/// replies it in profile mode. See `execute_query` for why this is a free
/// function rather than a method.
pub fn execute_profile(
    ctx: &Context,
    query: &str,
    per_query_timeout: Option<i64>,
) -> Result<ProfileDetect, String> {
    let Plan {
        plan, parameters, ..
    } = crate::query_lock::with_graph(|tg| tg.graph.read().borrow().get_plan(query))?;
    // Detect writes (mirrors `execute_query`). A write is NOT executed here —
    // it must go through the serialized write queue so it gets the two-phase
    // locking and effects replication.
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
    let g = crate::query_lock::with_graph(|tg| tg.graph.read());
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
        &crate::query_lock::RedisQueryLock,
    );
    let _ = runtime.query()?;
    reply_profile(ctx, &runtime, &plan);
    Ok(ProfileDetect::ReadReplied)
}

/// Execute a write query end-to-end for the **current thread's lock session**.
///
/// Not a method on `ThreadedGraph` on purpose: the session owns the per-graph
/// lock and *swaps* the read guard for a write guard when the query escalates to
/// writer mode (see `crate::query_lock`). A long-lived `&ThreadedGraph` would
/// therefore be invalidated mid-query, so every access to the locked graph here
/// is a short-lived `with_graph` borrow instead.
pub fn execute_query_write(
    ctx: &Context,
    query: &str,
    compact: bool,
    first_cached: bool,
    per_query_timeout: Option<i64>,
    profile: bool,
) -> WriteQueryResult {
    let wall_start = Instant::now();
    let Plan {
        plan,
        parameters,
        params_offset,
        ..
    } = crate::query_lock::with_graph(|tg| tg.graph.read().borrow().get_plan(query))?;
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

    // Acquire the MVCC write slot (single-writer serialization). Under the
    // two-phase write loop (issue #726) a data write runs here holding only
    // the outer L1-READ lock, so a concurrent inline writer racing in the
    // commit gap can momentarily hold the slot. Return a retryable error
    // rather than panicking. NOTE: on this branch we did NOT acquire the
    // slot, so callers must not roll back — it belongs to the other writer.
    let Some(g) = crate::query_lock::with_graph(|tg| tg.graph.write()) else {
        return Err("write lock unavailable, retry the query".to_string());
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
        &crate::query_lock::RedisQueryLock,
    );
    runtime.build_effects.set(
        REPLICATION_CONSUMERS.load(Ordering::Relaxed)
            || ctx.get_flags().contains(ContextFlags::AOF),
    );
    let mut result = match runtime.query() {
        Ok(r) => r,
        Err(err) => {
            // Bring the index back in line with committed state before discarding
            // the private version: documents published by earlier `Commit`s in this
            // query must be deleted (entity never existed) or rewritten from
            // committed values (entity still exists with its old values). Runs in
            // writer mode, so no reader can observe the interim state.
            let committed = crate::query_lock::with_graph(|tg| tg.graph.read());
            runtime.resync_published_indexes(&committed);
            // Query failed after we took the MVCC write slot: release it
            // here (the private new version is dropped with all its writes).
            // Slot ownership lives in this function now — callers must NOT
            // roll back.
            crate::query_lock::with_graph(|tg| tg.graph.rollback());
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
    if profile {
        // GRAPH.PROFILE: reply with the annotated plan tree (per-operator
        // records + timing collected because the runtime ran profile=true),
        // not the result set. Everything else — effects, deferred indexes,
        // commit, replication — is identical to a GRAPH.QUERY write.
        reply_profile(ctx, &runtime, &runtime.plan);
    } else if compact {
        reply_compact(ctx, &runtime, &result);
    } else {
        reply_verbose(ctx, &runtime, &result);
    }
    let latency = wall_start.elapsed().as_secs_f64() * 1000.0;
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
    let cmd_name = if profile {
        "GRAPH.PROFILE"
    } else {
        "GRAPH.QUERY"
    };
    crate::query_lock::with_graph(|tg| tg.slow_log.add(cmd_name, query, params_offset, latency));
    Ok(WriteQueryOk {
        graph: g,
        effects_buffer,
        modified,
        execution_time_ms,
        params_offset,
    })
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
) -> RedisResult {
    // Inside MULTI/EXEC: execute synchronously (blocking commands not allowed).
    // Also run replicated commands synchronously on the main thread (matches
    // FalkorDB C): otherwise the replica's handler returns NoReply before the
    // query actually executes, Redis advances the replication offset, and
    // master's WAIT reports the replica in-sync while writes are still queued.
    if ctx.get_flags().contains(ContextFlags::MULTI)
        || ctx.get_flags().contains(ContextFlags::REPLICATED)
    {
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
            // Every query — read or write — runs under a lock session, so all
            // lock transitions go through `crate::query_lock`. A read query simply
            // never escalates; a write query escalates at its first `Commit`.
            let _session = crate::query_lock::ScopedSession::begin(&binding, false);
            let bc = bc;
            let ctx = unsafe { ffi::get_thread_safe_context(bc.inner) };
            let ctx = Context::new(ctx);

            let cmd = if write {
                "GRAPH.QUERY"
            } else {
                "GRAPH.RO_QUERY"
            };

            let running_id = telemetry::register_running(received_at, &key_name, &query, false);
            let wall_start = Instant::now();
            // Time spent waiting in the thread pool before this worker started.
            let wait_ms = wall_start.duration_since(dispatch_instant).as_secs_f64() * 1000.0;
            let res = execute_query(&ctx, &query, compact, write, cmd, per_query_timeout);
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
                        let send = crate::query_lock::with_graph(|tg| tg.sender.send(msg));
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
                        // Release our reader session BEFORE entering the write loop:
                        // it installs its own session and escalates to the write
                        // lock, which would block forever behind the read lock we
                        // are still holding.
                        drop(_session);
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
        let _session = crate::query_lock::ScopedSession::begin(graph, true);
        execute_query(ctx, query, compact, write, cmd, per_query_timeout)
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
                // Runs on the main thread, which already holds the GIL
                // implicitly, so start as a writer: GIL → L1-write is already the
                // correct order and there is nothing to escalate.
                let _session = crate::query_lock::ScopedSession::begin_writer(graph);
                let res = execute_query_write(
                    ctx,
                    query,
                    compact,
                    read_result.cached,
                    per_query_timeout,
                    false,
                );
                let write_wall_ms = write_start.elapsed().as_secs_f64() * 1000.0;
                telemetry::unregister_running(running_id2);
                match res {
                    Ok(wq) => {
                        // Commit (index docs already applied inline), signal WATCH,
                        // and replicate under this L1-write guard — the same path
                        // the async write loop uses.
                        let (params_offset, exec_ms) = crate::query_lock::with_current(|s| {
                            let g = s
                                .graph_mut()
                                .expect("writer-mode session on the sync write path");
                            commit_and_replicate(g, ctx, key_name, query, wq)
                        })
                        .expect("lock session installed");
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
    // Inside MULTI/EXEC: execute synchronously.
    // Also run replicated commands synchronously (see query_mut for rationale).
    if ctx.get_flags().contains(ContextFlags::MULTI)
        || ctx.get_flags().contains(ContextFlags::REPLICATED)
    {
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
            let _session = crate::query_lock::ScopedSession::begin(&binding, false);
            let bc = bc;
            let ctx = unsafe { ffi::get_thread_safe_context(bc.inner) };
            let ctx = Context::new(ctx);

            let res = execute_profile(&ctx, &query, per_query_timeout);

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
                // Write query — enqueue exactly like GRAPH.QUERY so it runs
                // through the two-phase write loop (GIL→L1-write commit + effects
                // replication); `execute_query_write` in profile mode replies with
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
                    let send = crate::query_lock::with_graph(|tg| tg.sender.send(msg));
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
                    drop(_session);
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
        let _session = crate::query_lock::ScopedSession::begin(graph, true);
        execute_profile(ctx, query, per_query_timeout)
    };
    match res {
        // Read query — already profiled and replied.
        Ok(ProfileDetect::ReadReplied) => {}
        // Write query. This runs on the main thread, which already holds the
        // implicit GIL, so DDL's `GilGuard` no-ops and there is no L1→GIL
        // inversion. Execute + commit under L1-write, then replicate — mirroring
        // query_sync's write branch, so a profiled write reaches replicas.
        Ok(ProfileDetect::Write) => {
            let _session = crate::query_lock::ScopedSession::begin_writer(graph);
            let res = execute_query_write(ctx, query, false, false, per_query_timeout, true);
            match res {
                Ok(wq) => {
                    // Same commit → signal WATCH → replicate path as query_sync
                    // and the async write loop (commit_and_replicate); the main
                    // thread holds the implicit GIL, satisfying its GIL→L1
                    // contract.
                    crate::query_lock::with_current(|s| {
                        let g = s
                            .graph_mut()
                            .expect("writer-mode session on the sync profile path");
                        commit_and_replicate(g, ctx, key_name, query, wq);
                    })
                    .expect("lock session installed");
                }
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

/// Commit a successfully-executed write and replicate it. Runs under the
/// GIL AND the outer L1-write guard `g` (GIL→L1 order), so the `commit`
/// Arc-swap is fork-safe (#452) and commit+replicate are atomic vs. inline
/// main-thread writers. Returns `(params_offset, execution_time_ms)` for
/// telemetry.
fn commit_and_replicate(
    g: &mut ThreadedGraph,
    ctx: &Context,
    key_name: &Arc<str>,
    query: &str,
    wq: WriteQueryOk,
) -> (usize, f64) {
    let params_offset = wq.params_offset;
    let execution_time_ms = wq.execution_time_ms;
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
    (params_offset, execution_time_ms)
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

        // Two-phase locking, mirroring FalkorDB C's `QueryCtx_AcquireWriteLock`
        // (issue #726). The query starts as a **reader** (per-graph read lock, no
        // GIL) so its match phase runs concurrently with other readers, and
        // escalates to **writer** (GIL → L1-write) on its first mutation, staying
        // a writer until it finishes. Escalation always releases the read lock
        // *before* taking the GIL — the reverse order is the AB-BA deadlock
        // against an inline main-thread command holding the GIL and waiting for
        // L1-write.
        //
        // Because the writer window covers everything from the first mutation
        // through commit+replicate, index document writes are applied inline as
        // each `Commit` operator runs: they are visible to the rest of *this*
        // query (a later index scan sees what an earlier subquery wrote) while
        // concurrent readers stay excluded by L1-write. `commit` also still runs
        // under the GIL, keeping the Arc-swap fork-safe (#452).
        //
        // DDL (CREATE/DROP INDEX) needs no special path any more: creating the
        // RediSearch spec is a mutation, so it escalates like any other write and
        // finds the GIL already held.
        let _session = crate::query_lock::ScopedSession::begin(graph, false);
        let exec = execute_query_write(&ctx, &query, compact, cached, per_query_timeout, profile);
        let res: Result<(usize, f64), String> = match exec {
            Ok(wq) => {
                // The query escalated (every write plan ends in a Commit), so we
                // hold GIL + L1-write here: commit, signal and replicate without
                // any further lock juggling.
                crate::query_lock::with_current(|s| match s.graph_mut() {
                    Some(g) => Ok(commit_and_replicate(g, &ctx, &key_name, &query, wq)),
                    // Defensive: a write that somehow never escalated cannot be
                    // committed safely under a read lock. Release the MVCC slot
                    // (only commit/rollback clears it) so the graph stays writable.
                    None => {
                        s.graph().graph.rollback();
                        Err("write query did not acquire the graph write lock".to_string())
                    }
                })
                .expect("lock session installed")
            }
            // execute_query_write released the MVCC slot on failure; dropping the
            // session releases the read lock (and the GIL if it had escalated).
            Err(err) => Err(err),
        };
        // `_session` releases the read lock — or the GIL + write lock if the query
        // escalated — on every path, including the error returns.
        drop(_session);

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
fn should_use_effects(
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
fn build_index_effects(
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
