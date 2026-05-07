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
    Rx, Tx,
    spsc::{Array, bounded_blocking},
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
        pool::Pool,
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

pub(crate) struct WriteMessage {
    pub bc: BlockedClient,
    pub query: Arc<str>,
    pub compact: bool,
    pub cached: bool,
    pub key_name: Arc<str>,
    pub timeout: Option<i64>,
    pub received_at: i64,
    pub enqueue_instant: Instant,
    pub waiting_id: u64,
}

pub(crate) struct WriteQueryOk {
    pub graph: Arc<AtomicRefCell<Graph>>,
    pub effects_buffer: Option<Vec<u8>>,
    pub modified: bool,
    pub execution_time_ms: f64,
    pub params_offset: usize,
}
type WriteQueryResult = Result<WriteQueryOk, String>;

/// Result from a read-path `execute_query` call, surfacing timing metadata
/// needed for telemetry.
pub(crate) struct ReadQueryResult {
    pub is_write: bool,
    pub cached: bool,
    pub execution_time_ms: f64,
    pub params_offset: usize,
    pub timed_out: bool,
}

/// Safe wrappers over Redis module FFI function pointers.
///
/// Redis guarantees these pointers are non-null after `RedisModule_Init` succeeds.
/// Centralising the `.expect()` here documents that invariant in one place and
/// keeps call sites free of `unwrap()` noise.
pub(crate) mod ffi {
    use redis_module::raw;
    use std::ffi::CString;
    use std::os::raw::c_char;
    use std::ptr::null_mut;

    const MSG: &str = "Redis module FFI pointer not initialised (call RedisModule_Init first)";

    /// Wrap a user error string into a CString, replacing any NUL bytes so
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
        unsafe { f(ctx, flags, null_mut() as *const c_char) };
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

pub struct ThreadedGraph {
    pub graph: MvccGraph,
    pub sender: Tx<Array<Box<WriteMessage>>>,
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

    /// Returns the graph name.
    pub fn name(&self) -> String {
        let g = self.graph.read();
        g.borrow().name().to_string()
    }

    pub fn execute_query(
        &self,
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
        } = self.graph.read().borrow().get_plan(query)?;
        let parameters = parameters
            .into_iter()
            .map(|(k, v)| Ok((k, evaluate_param(&v.root())?)))
            .collect::<Result<HashMap<_, _>, String>>()?;
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
            self.graph.read()
        };
        let env_pool = Pool::new();
        let timeout_ms = compute_effective_timeout(per_query_timeout, is_write)?;
        let runtime = Runtime::new(
            g,
            parameters,
            is_write,
            plan,
            false,
            (*CONFIGURATION_IMPORT_FOLDER.lock(ctx)).clone(),
            &env_pool,
            RESULTSET_SIZE.load(Ordering::Relaxed),
            false,
            timeout_ms,
            QUERY_MEM_CAPACITY.load(Ordering::Relaxed),
            Some(net_thread_usage),
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
        self.slow_log.add(cmd, query, params_offset, latency);
        Ok(ReadQueryResult {
            is_write: false,
            cached,
            execution_time_ms,
            params_offset,
            timed_out: false,
        })
    }

    pub fn execute_query_write(
        &self,
        ctx: &Context,
        query: &str,
        compact: bool,
        first_cached: bool,
        per_query_timeout: Option<i64>,
    ) -> WriteQueryResult {
        let wall_start = Instant::now();
        let Plan {
            plan,
            parameters,
            params_offset,
            ..
        } = self.graph.read().borrow().get_plan(query)?;
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

        // Invariant: callers hold the outer `RwLock<ThreadedGraph>` write guard
        // (via `process_write_queued_query` or `query_sync`), so the MVCC write
        // slot must be free here. If it is not, the write queue has been bypassed
        // and the MVCC single-writer invariant is violated.
        let g = self
            .graph
            .write()
            .expect("MVCC write slot busy: single-writer invariant violated");
        let env_pool = Pool::new();
        let timeout_ms = compute_effective_timeout(per_query_timeout, true)?;
        let runtime = Runtime::new(
            g.clone(),
            parameters,
            true,
            plan,
            false,
            (*CONFIGURATION_IMPORT_FOLDER.lock(ctx)).clone(),
            &env_pool,
            RESULTSET_SIZE.load(Ordering::Relaxed),
            false,
            timeout_ms,
            QUERY_MEM_CAPACITY.load(Ordering::Relaxed),
            Some(net_thread_usage),
        );
        let mut result = match runtime.query() {
            Ok(r) => r,
            Err(err) => {
                // Clean up dirty cache entries before the graph is dropped.
                g.borrow_mut().rollback_cache();
                return Err(err);
            }
        };

        // Query succeeded — now commit deferred index operations to RediSearch.
        runtime.commit_deferred_indexes();

        // Capture effects buffer before replying (pending data is still available)
        let mut effects_buffer =
            should_use_effects(is_non_deterministic, &runtime, result.stats.execution_time);

        // Build index effects for CreateIndex / DropIndex IR nodes (not tracked by Pending)
        effects_buffer = build_index_effects(&runtime, effects_buffer);

        result.stats.cached = cached;
        if compact {
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
        self.slow_log
            .add("GRAPH.QUERY", query, params_offset, latency);
        Ok(WriteQueryOk {
            graph: g,
            effects_buffer,
            modified,
            execution_time_ms,
            params_offset,
        })
    }

    /// Execute a query with profiling enabled (read path).
    pub fn execute_profile(
        &self,
        ctx: &Context,
        query: &str,
        per_query_timeout: Option<i64>,
    ) -> Result<bool, String> {
        let Plan {
            plan, parameters, ..
        } = self.graph.read().borrow().get_plan(query)?;
        let parameters = parameters
            .into_iter()
            .map(|(k, v)| Ok((k, evaluate_param(&v.root())?)))
            .collect::<Result<HashMap<_, _>, String>>()?;
        let is_write = plan.iter().any(|n| {
            matches!(
                n,
                IR::Commit | IR::CreateIndex { .. } | IR::DropIndex { .. }
            )
        });
        if is_write {
            return Ok(true);
        }
        let g = self.graph.read();
        let env_pool = Pool::new();
        let timeout_ms = compute_effective_timeout(per_query_timeout, false)?;
        let runtime = Runtime::new(
            g,
            parameters,
            false,
            plan.clone(),
            false,
            String::new(),
            &env_pool,
            -1,
            true,
            timeout_ms,
            QUERY_MEM_CAPACITY.load(Ordering::Relaxed),
            Some(net_thread_usage),
        );
        let _ = runtime.query()?;
        reply_profile(ctx, &runtime, &plan);
        Ok(false)
    }

    /// Execute a write query with profiling enabled.
    pub fn execute_profile_write(
        &self,
        ctx: &Context,
        query: &str,
        per_query_timeout: Option<i64>,
    ) -> WriteQueryResult {
        let Plan {
            plan, parameters, ..
        } = self.graph.read().borrow().get_plan(query)?;
        let parameters = parameters
            .into_iter()
            .map(|(k, v)| Ok((k, evaluate_param(&v.root())?)))
            .collect::<Result<HashMap<_, _>, String>>()?;

        // Invariant: see `execute_query_write` — outer RwLock guarantees the
        // MVCC write slot is free at this point.
        let g = self
            .graph
            .write()
            .expect("MVCC write slot busy: single-writer invariant violated");
        let env_pool = Pool::new();
        let timeout_ms = compute_effective_timeout(per_query_timeout, true)?;
        let runtime = Runtime::new(
            g.clone(),
            parameters,
            true,
            plan.clone(),
            false,
            String::new(),
            &env_pool,
            -1,
            true,
            timeout_ms,
            QUERY_MEM_CAPACITY.load(Ordering::Relaxed),
            Some(net_thread_usage),
        );
        match runtime.query() {
            Ok(_) => {
                runtime.commit_deferred_indexes();
                reply_profile(ctx, &runtime, &plan);
                Ok(WriteQueryOk {
                    graph: g,
                    effects_buffer: None,
                    modified: false,
                    execution_time_ms: 0.0,
                    params_offset: 0,
                })
            }
            Err(err) => {
                g.borrow_mut().rollback_cache();
                Err(err)
            }
        }
    }
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
    if ctx.get_flags().contains(ContextFlags::MULTI) {
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
            let graph = binding.read();
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
            let res = graph.execute_query(&ctx, &query, compact, write, cmd, per_query_timeout);
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
                        });
                        if let Err(send_err) = graph.sender.send(msg) {
                            let msg = send_err.0;
                            telemetry::unregister_waiting(msg.waiting_id);
                            let cerr = ffi::sanitise_error(
                                "ERR graph write queue unavailable".to_string(),
                            );
                            unsafe { ffi::reply_error(ctx.ctx, cerr.as_ptr()) };
                            drop(msg.bc);
                            drop(graph);
                            unsafe { ffi::free_thread_safe_context(ctx.ctx) };
                            return;
                        }
                        drop(graph);
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

                            wait_duration_ms: 0.0,
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
    let running_id = telemetry::register_running(received_at, key_name, query, false);
    let res = {
        let g = graph.read();
        g.execute_query(ctx, query, compact, write, cmd, per_query_timeout)
    };
    let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
    telemetry::unregister_running(running_id);
    match res {
        Ok(read_result) => {
            if read_result.is_write {
                // Write path: acquire exclusive lock and execute.
                let write_start = Instant::now();
                let running_id2 = telemetry::register_running(received_at, key_name, query, false);
                let mut g = graph.write();
                let res = g.execute_query_write(
                    ctx,
                    query,
                    compact,
                    read_result.cached,
                    per_query_timeout,
                );
                let write_wall_ms = write_start.elapsed().as_secs_f64() * 1000.0;
                telemetry::unregister_running(running_id2);
                match res {
                    Ok(wq) => {
                        g.graph.commit(wq.graph);
                        if wq.modified {
                            replicate_effects(ctx, key_name, wq.effects_buffer, query);
                        }
                        // Flush dirty cache entries to fjall if over budget.
                        let value = g.graph.read().borrow().maybe_flush_caches();
                        if let Err(e) = value {
                            ctx.log_warning(&format!("FalkorDB: cache flush failed: {e}"));
                        }
                        // Write telemetry
                        let query_text = &query[wq.params_offset..];
                        let params_text = &query[..wq.params_offset];
                        let exec_ms = wq.execution_time_ms;
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
                        g.graph.rollback();
                        return Err(redis_module::RedisError::String(err));
                    }
                }
                drop(g);
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
    if ctx.get_flags().contains(ContextFlags::MULTI) {
        return profile_sync(ctx, graph, query, key_name, per_query_timeout);
    }

    let bc = unsafe { BlockedClient::new(ctx.ctx) };
    let graph = graph.clone();
    let query: Arc<str> = Arc::from(query);
    spawn(
        move || {
            let mem_capacity = QUERY_MEM_CAPACITY.load(Ordering::Relaxed);
            if mem_capacity > 0 {
                reset_counter();
                enable_tracking();
            }
            let g = graph.clone();
            let binding = graph.clone();
            let graph_read = binding.read();
            let bc = bc;
            let ctx = unsafe { ffi::get_thread_safe_context(bc.inner) };
            let ctx = Context::new(ctx);

            let res = graph_read.execute_profile(&ctx, &query, per_query_timeout);

            match res {
                Ok(is_write) => {
                    if is_write {
                        // Write path: drop read lock, acquire write lock.
                        // Free the read-phase context before creating a new one.
                        drop(graph_read);
                        unsafe { ffi::free_thread_safe_context(ctx.ctx) };
                        let mut graph_write = g.write();
                        let ctx2 = unsafe { ffi::get_thread_safe_context(bc.inner) };
                        let ctx2 = Context::new(ctx2);
                        let res =
                            graph_write.execute_profile_write(&ctx2, &query, per_query_timeout);
                        match res {
                            Ok(wq) => {
                                graph_write.graph.commit(wq.graph);
                                let value = graph_write.graph.read().borrow().maybe_flush_caches();
                                if let Err(e) = value {
                                    ctx2.log_warning(&format!("FalkorDB: cache flush failed: {e}"));
                                }
                            }
                            Err(err) => {
                                let cerr = ffi::sanitise_error(err);
                                unsafe { ffi::reply_error(ctx2.ctx, cerr.as_ptr()) };
                                graph_write.graph.rollback();
                            }
                        }
                        drop(bc);
                        unsafe { ffi::free_thread_safe_context(ctx2.ctx) };
                    } else {
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
            if mem_capacity > 0 {
                disable_tracking();
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
    _key_name: &Arc<str>,
    per_query_timeout: Option<i64>,
) -> RedisResult {
    let mem_capacity = QUERY_MEM_CAPACITY.load(Ordering::Relaxed);
    if mem_capacity > 0 {
        reset_counter();
        enable_tracking();
    }
    let res = {
        let g = graph.read();
        g.execute_profile(ctx, query, per_query_timeout)
    };
    match res {
        Ok(is_write) => {
            if is_write {
                let mut g = graph.write();
                let res = g.execute_profile_write(ctx, query, per_query_timeout);
                match res {
                    Ok(wq) => {
                        g.graph.commit(wq.graph);
                        let value = g.graph.read().borrow().maybe_flush_caches();
                        if let Err(e) = value {
                            ctx.log_warning(&format!("FalkorDB: cache flush failed: {e}"));
                        }
                    }
                    Err(err) => {
                        g.graph.rollback();
                        if mem_capacity > 0 {
                            disable_tracking();
                        }
                        return Err(redis_module::RedisError::String(err));
                    }
                }
                drop(g);
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

pub fn process_write_queued_query(graph: &Arc<RwLock<ThreadedGraph>>) {
    let g = graph.read();
    if g.write_loop
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Acquire)
        .is_ok()
    {
        drop(g);
        let mut graph = graph.write();
        loop {
            while let Ok(msg) = { graph.receiver.try_recv() } {
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
                } = *msg;
                // Transition from waiting to running
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
                let res =
                    graph.execute_query_write(&ctx, &query, compact, cached, per_query_timeout);
                if mem_capacity > 0 {
                    disable_tracking();
                }
                let write_wall_ms = write_start.elapsed().as_secs_f64() * 1000.0;
                if let Some(rid) = running_id {
                    telemetry::unregister_running(rid);
                }
                match res {
                    Ok(wq) => {
                        // Signal the key as modified so WATCH gets triggered.
                        unsafe {
                            ffi::lock_thread_safe_ctx(ctx.ctx);
                            ffi::signal_modified_key(ctx.ctx, key_name.as_bytes());
                        };
                        // Send replication while GIL is held
                        if wq.modified {
                            replicate_effects(&ctx, &key_name, wq.effects_buffer, &query);
                        }
                        unsafe {
                            ffi::unlock_thread_safe_ctx(ctx.ctx);
                            ffi::free_thread_safe_context(ctx.ctx);
                        };
                        let query_text = &query[wq.params_offset..];
                        let params_text = &query[..wq.params_offset];
                        let exec_ms = wq.execution_time_ms;
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
                        // Enqueue telemetry entry for background flusher
                        telemetry::enqueue_entry(&key_name, entry);
                        drop(bc);
                        graph.graph.commit(wq.graph);
                        let value = graph.graph.read().borrow().maybe_flush_caches();
                        if let Err(e) = value {
                            redis_module::logging::log_warning(format!(
                                "FalkorDB: cache flush failed: {e}"
                            ));
                        }
                    }
                    Err(err) => {
                        let cerr = ffi::sanitise_error(err);
                        unsafe { ffi::reply_error(ctx.ctx, cerr.as_ptr()) };
                        drop(bc);
                        unsafe { ffi::free_thread_safe_context(ctx.ctx) };
                        graph.graph.rollback();
                    }
                }
                std::thread::yield_now();
            }
            graph.write_loop.store(false, Ordering::Release);
            if graph.receiver.is_empty() {
                return;
            }
            if graph
                .write_loop
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Acquire)
                .is_err()
            {
                return;
            }
        }
    }
}

/// Decide whether to use effects replication and get the pre-built buffer.
/// The buffer was built in CommitOp before pending was cleared.
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
                ..
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
        drop(Box::from_raw(value.cast::<Arc<RwLock<ThreadedGraph>>>()));
    }
}
