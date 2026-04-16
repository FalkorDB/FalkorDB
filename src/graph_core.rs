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
        CONFIGURATION_IMPORT_FOLDER, EFFECTS_THRESHOLD, MAX_QUEUED_QUERIES, RESULTSET_SIZE,
        TIMEOUT_DEFAULT,
    },
    reply::{reply_compact, reply_verbose},
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
    ffi::CString,
    os::raw::{c_char, c_void},
    ptr::null_mut,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use crate::allocator::{current_thread_usage, disable_tracking, enable_tracking, reset_counter};

type WriteMessage = (BlockedClient, Arc<String>, bool, bool, Arc<String>);
type WriteQueryResult = Result<(Arc<AtomicRefCell<Graph>>, Option<Vec<u8>>, bool), String>;

pub struct ThreadedGraph {
    pub graph: MvccGraph,
    pub sender: Tx<Array<WriteMessage>>,
    pub receiver: Rx<Array<WriteMessage>>,
    pub write_loop: AtomicBool,
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
    ) -> Result<(bool, bool), String> {
        let Plan {
            plan,
            cached,
            parameters,
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
            return Ok((is_write, cached));
        } else {
            self.graph.read()
        };
        let env_pool = Pool::new();
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
        );
        let mut result = runtime.query()?;
        result.stats.cached = cached;
        if compact {
            reply_compact(ctx, &runtime, &result);
        } else {
            reply_verbose(ctx, &runtime, &result);
        }
        drop(result);
        drop(runtime);
        Ok((is_write, cached))
    }

    pub fn execute_query_write(
        &self,
        ctx: &Context,
        query: &str,
        compact: bool,
        first_cached: bool,
    ) -> WriteQueryResult {
        let Plan {
            plan, parameters, ..
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

        let g = self.graph.write().unwrap();
        let env_pool = Pool::new();
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
        );
        let mut result = match runtime.query() {
            Ok(r) => r,
            Err(err) => {
                // Clean up dirty cache entries before the graph is dropped.
                g.borrow_mut().rollback_cache();
                return Err(err);
            }
        };

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
        let modified = result.stats.nodes_created > 0
            || result.stats.nodes_deleted > 0
            || result.stats.relationships_created > 0
            || result.stats.relationships_deleted > 0
            || result.stats.properties_set > 0
            || result.stats.properties_removed > 0
            || result.stats.labels_added > 0
            || result.stats.labels_removed > 0
            || result.stats.indexes_created > 0
            || result.stats.indexes_dropped > 0;
        Ok((g, effects_buffer, modified))
    }

    /// Execute a query with profiling enabled (read path).
    pub fn execute_profile(
        &self,
        ctx: &Context,
        query: &str,
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
    ) -> WriteQueryResult {
        let Plan {
            plan, parameters, ..
        } = self.graph.read().borrow().get_plan(query)?;
        let parameters = parameters
            .into_iter()
            .map(|(k, v)| Ok((k, evaluate_param(&v.root())?)))
            .collect::<Result<HashMap<_, _>, String>>()?;

        let g = self.graph.write().unwrap();
        let env_pool = Pool::new();
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
        );
        match runtime.query() {
            Ok(_) => {
                reply_profile(ctx, &runtime, &plan);
                Ok((g, None, false))
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

impl Drop for BlockedClient {
    fn drop(&mut self) {
        unsafe { raw::RedisModule_UnblockClient.unwrap()(self.inner, null_mut()) };
    }
}

#[inline]
pub fn query_mut(
    ctx: &Context,
    graph: &Arc<RwLock<ThreadedGraph>>,
    query: &str,
    compact: bool,
    write: bool,
    track_mem: bool,
    key_name: Arc<String>,
) -> RedisResult {
    // Inside MULTI/EXEC: execute synchronously (blocking commands not allowed).
    if ctx.get_flags().contains(ContextFlags::MULTI) {
        return query_sync(ctx, graph, query, compact, write, &key_name);
    }

    // Check pending queries limit before dispatching.
    let max = MAX_QUEUED_QUERIES.load(Ordering::Relaxed) as usize;
    if pending_count() >= max {
        let bc = BlockedClient {
            inner: unsafe { raw::RedisModule_BlockClient.unwrap()(ctx.ctx, None, None, None, 0) },
        };
        let err_ctx = unsafe { raw::RedisModule_GetThreadSafeContext.unwrap()(bc.inner) };
        let err_ctx = Context::new(err_ctx);
        let cerr = CString::new("Max pending queries exceeded").unwrap();
        raw::reply_with_error(err_ctx.ctx, cerr.as_ptr());
        drop(bc);
        unsafe { raw::RedisModule_FreeThreadSafeContext.unwrap()(err_ctx.ctx) };
        return Ok(RedisValue::NoReply);
    }

    let bc = BlockedClient {
        inner: unsafe { raw::RedisModule_BlockClient.unwrap()(ctx.ctx, None, None, None, 0) },
    };
    let graph = graph.clone();
    let query = Arc::new(query.to_string());
    spawn(
        move || {
            if track_mem {
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
            let ctx = unsafe { raw::RedisModule_GetThreadSafeContext.unwrap()(bc.inner) };
            let ctx = Context::new(ctx);

            let res = graph.execute_query(&ctx, &query, compact, write);

            // Log memory tracking BEFORE freeing the context.
            if track_mem {
                let (allocated, deallocated) = current_thread_usage();
                disable_tracking();
                ctx.log(
                    redis_module::logging::RedisLogLevel::Notice,
                    &format!(
                        "Allocated: {allocated} bytes, Deallocated: {deallocated} bytes, Net: {}",
                        allocated as isize - deallocated as isize
                    ),
                );
            }

            match res {
                Ok((is_write, cached)) => {
                    if is_write {
                        graph
                            .sender
                            .send((bc, query, compact, cached, key_name))
                            .unwrap();
                        drop(graph);
                        process_write_queued_query(&g);
                    } else {
                        drop(bc);
                        unsafe { raw::RedisModule_FreeThreadSafeContext.unwrap()(ctx.ctx) };
                    }
                }
                Err(err) => {
                    let cerr = CString::new(err).unwrap();
                    raw::reply_with_error(ctx.ctx, cerr.as_ptr());
                    drop(bc);
                    unsafe { raw::RedisModule_FreeThreadSafeContext.unwrap()(ctx.ctx) };
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
    key_name: &Arc<String>,
) -> RedisResult {
    // First pass: parse + detect if write, execute reads inline.
    // Sync query timeout to UDF JS runtime
    graph::udf::js_context::JS_TIMEOUT_MS
        .store(TIMEOUT_DEFAULT.load(Ordering::Relaxed), Ordering::Relaxed);

    let res = {
        let g = graph.read();
        g.execute_query(ctx, query, compact, write)
    };
    match res {
        Ok((is_write, cached)) => {
            if is_write {
                // Write path: acquire exclusive lock and execute.
                let mut g = graph.write();
                let res = g.execute_query_write(ctx, query, compact, cached);
                match res {
                    Ok((new_graph, effects_buffer, modified)) => {
                        g.graph.commit(new_graph);
                        if modified {
                            replicate_effects(ctx, key_name, effects_buffer, query);
                        }
                        // Flush dirty cache entries to fjall if over budget.
                        let value = g.graph.read().borrow().maybe_flush_caches();
                        if let Err(e) = value {
                            eprintln!("FalkorDB: cache flush failed: {e}");
                        }
                    }
                    Err(err) => {
                        g.graph.rollback();
                        return Err(redis_module::RedisError::String(err));
                    }
                }
                drop(g);
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
    key_name: &Arc<String>,
) -> RedisResult {
    // Inside MULTI/EXEC: execute synchronously.
    if ctx.get_flags().contains(ContextFlags::MULTI) {
        return profile_sync(ctx, graph, query, key_name);
    }

    let bc = BlockedClient {
        inner: unsafe { raw::RedisModule_BlockClient.unwrap()(ctx.ctx, None, None, None, 0) },
    };
    let graph = graph.clone();
    let query = Arc::new(query.to_string());
    spawn(
        move || {
            let g = graph.clone();
            let binding = graph.clone();
            let graph_read = binding.read();
            let bc = bc;
            let ctx = unsafe { raw::RedisModule_GetThreadSafeContext.unwrap()(bc.inner) };
            let ctx = Context::new(ctx);

            let res = graph_read.execute_profile(&ctx, &query);

            match res {
                Ok(is_write) => {
                    if is_write {
                        // Write path: drop read lock, acquire write lock.
                        // Free the read-phase context before creating a new one.
                        drop(graph_read);
                        unsafe { raw::RedisModule_FreeThreadSafeContext.unwrap()(ctx.ctx) };
                        let mut graph_write = g.write();
                        let ctx2 =
                            unsafe { raw::RedisModule_GetThreadSafeContext.unwrap()(bc.inner) };
                        let ctx2 = Context::new(ctx2);
                        let res = graph_write.execute_profile_write(&ctx2, &query);
                        match res {
                            Ok((new_graph, _, _)) => {
                                graph_write.graph.commit(new_graph);
                                let value = graph_write.graph.read().borrow().maybe_flush_caches();
                                if let Err(e) = value {
                                    eprintln!("FalkorDB: cache flush failed: {e}");
                                }
                            }
                            Err(err) => {
                                let cerr = CString::new(err).unwrap();
                                raw::reply_with_error(ctx2.ctx, cerr.as_ptr());
                                graph_write.graph.rollback();
                            }
                        }
                        drop(bc);
                        unsafe { raw::RedisModule_FreeThreadSafeContext.unwrap()(ctx2.ctx) };
                    } else {
                        drop(bc);
                        unsafe { raw::RedisModule_FreeThreadSafeContext.unwrap()(ctx.ctx) };
                    }
                }
                Err(err) => {
                    let cerr = CString::new(err).unwrap();
                    raw::reply_with_error(ctx.ctx, cerr.as_ptr());
                    drop(bc);
                    unsafe { raw::RedisModule_FreeThreadSafeContext.unwrap()(ctx.ctx) };
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
    _key_name: &Arc<String>,
) -> RedisResult {
    let res = {
        let g = graph.read();
        g.execute_profile(ctx, query)
    };
    match res {
        Ok(is_write) => {
            if is_write {
                let mut g = graph.write();
                let res = g.execute_profile_write(ctx, query);
                match res {
                    Ok((new_graph, _, _)) => {
                        g.graph.commit(new_graph);
                        let value = g.graph.read().borrow().maybe_flush_caches();
                        if let Err(e) = value {
                            eprintln!("FalkorDB: cache flush failed: {e}");
                        }
                    }
                    Err(err) => {
                        g.graph.rollback();
                        return Err(redis_module::RedisError::String(err));
                    }
                }
                drop(g);
            }
        }
        Err(err) => {
            return Err(redis_module::RedisError::String(err));
        }
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
        while let Ok((bc, query, compact, cached, key_name)) = { graph.receiver.try_recv() } {
            let ctx = unsafe { raw::RedisModule_GetThreadSafeContext.unwrap()(bc.inner) };
            let ctx = Context::new(ctx);
            let res = graph.execute_query_write(&ctx, &query, compact, cached);
            match res {
                Ok((g, effects_buffer, modified)) => {
                    // Signal the key as modified so WATCH gets triggered.
                    unsafe {
                        raw::RedisModule_ThreadSafeContextLock.unwrap()(ctx.ctx);
                        let rstr = raw::RedisModule_CreateString.unwrap()(
                            ctx.ctx,
                            key_name.as_ptr().cast(),
                            key_name.len(),
                        );
                        raw::RedisModule_SignalModifiedKey.unwrap()(ctx.ctx, rstr);
                        raw::RedisModule_FreeString.unwrap()(ctx.ctx, rstr);
                    };
                    // Send replication while GIL is held
                    if modified {
                        replicate_effects(&ctx, &key_name, effects_buffer, &query);
                    }
                    unsafe {
                        raw::RedisModule_ThreadSafeContextUnlock.unwrap()(ctx.ctx);
                        raw::RedisModule_FreeThreadSafeContext.unwrap()(ctx.ctx);
                    };
                    drop(bc);
                    graph.graph.commit(g);
                    // Flush dirty cache entries to fjall if over budget.
                    let value = graph.graph.read().borrow().maybe_flush_caches();
                    if let Err(e) = value {
                        eprintln!("FalkorDB: cache flush failed: {e}");
                    }
                }
                Err(err) => {
                    let cerr = CString::new(err).unwrap();
                    raw::reply_with_error(ctx.ctx, cerr.as_ptr());
                    drop(bc);
                    unsafe { raw::RedisModule_FreeThreadSafeContext.unwrap()(ctx.ctx) };
                    graph.graph.rollback();
                }
            }
            // Yield between batched writes so other graph write loops
            // (on different threadpool workers) can make progress.
            std::thread::yield_now();
        }
        graph.write_loop.store(false, Ordering::Release);
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
