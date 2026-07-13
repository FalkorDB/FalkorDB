//! Redis module initialization and startup wiring.
//!
//! Handles RediSearch bootstrap, GraphBLAS allocator setup, Redis event
//! subscription, and function registry initialization.
//!
//! ## Startup sequence
//! ```text
//! Redis loads module
//!      |
//!      v
//! graph_init()
//!   1) (optional) start profiler backend
//!   2) initialize RediSearch API bindings
//!   3) install Redis allocators into GraphBLAS layer
//!   4) subscribe to FlushDB event hook
//!   5) register built-in runtime functions
//!      |
//!      v
//! module ready to accept GRAPH.* commands
//! ```
//!
//! Any hard failure during critical init steps returns `Status::Err` so Redis
//! can reject loading an incomplete module.

use crate::config::{
    CONFIGURATION_INDEX_WORKER_THREADS, CONFIGURATION_JS_HEAP_SIZE, CONFIGURATION_JS_STACK_SIZE,
    CONFIGURATION_NODE_CREATION_BUFFER, CONFIGURATION_TEMP_FOLDER, DELTA_MAX_PENDING_CHANGES,
    EFFECTS_THRESHOLD, MAX_QUEUED_QUERIES, OMP_THREAD_COUNT, QUERY_MEM_CAPACITY, RESULTSET_SIZE,
    TIMEOUT, TIMEOUT_DEFAULT, TIMEOUT_MAX, get_thread_count, normalize_node_creation_buffer,
};
use crate::redis_type::on_persistence;
use crate::telemetry;
use graph::{
    graph::graphblas::matrix::init,
    index::redisearch::{
        REDISEARCH_INIT_LIBRARY, RediSearch_CleanupModule, RediSearch_Init,
        RediSearch_SetDefaultScorer, RediSearch_SetNumWorkerThreads,
    },
    runtime::functions::{init_functions, init_udf_functions},
    threadpool::{self, init_thread_pool},
    udf,
};
use redis_module::{
    Context, ContextFlags, REDISMODULE_OK, RedisModule_Alloc, RedisModule_Calloc, RedisModule_Free,
    RedisModule_Realloc, RedisModule_SubscribeToServerEvent, RedisModuleCtx, RedisModuleEvent,
    Status,
};
use std::{os::raw::c_int, os::raw::c_void, panic, sync::atomic::AtomicI64};

/// Redis event ID for FlushDB event (database flush/clear).
#[allow(non_upper_case_globals)]
static RedisModuleEvent_FlushDB: RedisModuleEvent = RedisModuleEvent { id: 2, dataver: 1 };

/// Redis event ID for Persistence events (RDB save start/end).
#[allow(non_upper_case_globals)]
static RedisModuleEvent_Persistence: RedisModuleEvent = RedisModuleEvent { id: 1, dataver: 1 };

/// Redis event ID for Loading events (RDB/AOF/replication load lifecycle).
#[allow(non_upper_case_globals)]
static RedisModuleEvent_Loading: RedisModuleEvent = RedisModuleEvent { id: 3, dataver: 1 };

/// Subevent: loading completed successfully.
const REDISMODULE_SUBEVENT_LOADING_ENDED: u64 = 3;
/// Subevent: loading failed.
const REDISMODULE_SUBEVENT_LOADING_FAILED: u64 = 4;

/// Redis event ID for replication role changes (master <-> replica).
#[allow(non_upper_case_globals)]
static RedisModuleEvent_ReplicationRoleChanged: RedisModuleEvent =
    RedisModuleEvent { id: 0, dataver: 1 };

/// Redis event ID for replica attach/detach (REDISMODULE_EVENT_REPLICA_CHANGE).
#[allow(non_upper_case_globals)]
static RedisModuleEvent_ReplicaChange: RedisModuleEvent = RedisModuleEvent { id: 6, dataver: 1 };

/// Redis event ID for shutdown. Only wired up under sanitizer/valgrind
/// runs (gated by `RS_GLOBAL_DTORS`) so workers join cleanly and per-thread
/// + module-level RediSearch/LAGraph state is released — otherwise these
///   allocations are reported as leaks at process exit.
#[allow(non_upper_case_globals)]
static RedisModuleEvent_Shutdown: RedisModuleEvent = RedisModuleEvent { id: 5, dataver: 1 };

/// Subevent: this instance is now a replica.
const REDISMODULE_EVENT_REPLROLECHANGED_NOW_REPLICA: u64 = 1;

unsafe extern "C" {
    fn pthread_atfork(
        prepare: Option<unsafe extern "C" fn()>,
        parent: Option<unsafe extern "C" fn()>,
        child: Option<unsafe extern "C" fn()>,
    ) -> c_int;
}

/// Called in the forked child process (via `pthread_atfork`). Mirrors
/// the C port's `_AfterForkChild` (`module_event_handlers.c:586`).
///
/// Three responsibilities, in order:
///
/// 1. Mark this process as a fork child so downstream code can detect
///    it isn't the original parent.
/// 2. Force GraphBLAS/OpenMP to single-threaded; the parent's OMP thread
///    pool handles are shared parent state and invalid in the child.
///    Safe (and required) for every fork path — BGSAVE *and* RediSearch
///    ForkGC.
/// 3. Only on the main-thread fork (BGSAVE): validate every graph is
///    fully synced. If `pre_fork_prepare` failed to materialize any
///    matrix (e.g. a writer slipped in after the registry walk), abort
///    the BGSAVE child via `std::process::abort()` rather than emit a
///    corrupt RDB. `abort()` is async-signal-safe (SIGABRT) and skips
///    Rust/C atexit chains — important here because libomp registers
///    an atexit (`__kmp_internal_end_atexit`) that can deadlock on
///    parent-held mutexes in the fork child, and ASAN registers a
///    leak-check atexit that takes seconds to walk the heap.
unsafe extern "C" fn on_fork_child() {
    graph::thread_id::set_process_is_child(true);
    graph::graph::graphblas::matrix::set_nthreads(1);

    if !graph::thread_id::is_main_thread() {
        // RediSearch ForkGC fork — graph state is irrelevant to its
        // RediSearch-only work.
        return;
    }

    let registry = crate::graph_core::GRAPH_REGISTRY.lock();
    for graph_arc in registry.values() {
        let tg: &crate::graph_core::ThreadedGraph = unsafe { &*graph_arc.data_ptr() };
        let g = tg.graph.read();
        let graph = g.borrow();
        if !graph.is_synced() {
            std::process::abort();
        }
    }
}

pub fn graph_init(
    ctx: &Context,
    args: &[redis_module::RedisString],
) -> Status {
    graph::thread_id::set_main_thread();
    panic::set_hook(Box::new(|info| {
        // Route the panic message + backtrace through RedisModule_Log so
        // it lands in the Redis log file uploaded by CI on test failure.
        // Redis itself writes its log to stderr/stdout when no `logfile`
        // is configured, so this path also covers interactive runs — no
        // need for an extra eprintln! that would double-print there.
        let msg = format!(
            "FalkorDB panic: {info}\nBacktrace:\n{}",
            std::backtrace::Backtrace::force_capture()
        )
        .replace('\0', " ");
        unsafe {
            if let Some(log) = graph::index::redisearch::redis::RedisModule_Log
                && let Ok(c_msg) = std::ffi::CString::new(msg)
            {
                log(
                    std::ptr::null_mut(),
                    c"warning".as_ptr(),
                    c"%s".as_ptr(),
                    c_msg.as_ptr(),
                );
            }
        }
        std::process::exit(1);
    }));

    // Parse module args for AtomicI64/AtomicU64 statics not registered in the
    // redis_module! config section.
    {
        let args_str: Vec<String> = args
            .iter()
            .map(redis_module::RedisString::to_string_lossy)
            .collect();
        let mut i = 0;
        while i < args_str.len() {
            let name = args_str[i].to_uppercase();
            let target_i64: Option<&AtomicI64> = match name.as_str() {
                "TIMEOUT" => Some(&TIMEOUT),
                "TIMEOUT_DEFAULT" => Some(&TIMEOUT_DEFAULT),
                "TIMEOUT_MAX" => Some(&TIMEOUT_MAX),
                "RESULTSET_SIZE" => Some(&RESULTSET_SIZE),
                "QUERY_MEM_CAPACITY" => Some(&QUERY_MEM_CAPACITY),
                "DELTA_MAX_PENDING_CHANGES" => Some(&DELTA_MAX_PENDING_CHANGES),
                "EFFECTS_THRESHOLD" => Some(&EFFECTS_THRESHOLD),
                "OMP_THREAD_COUNT" => Some(&OMP_THREAD_COUNT),
                _ => None,
            };
            if let Some(target) = target_i64 {
                if i + 1 < args_str.len()
                    && let Ok(v) = args_str[i + 1].parse::<i64>()
                {
                    target.store(v, std::sync::atomic::Ordering::Relaxed);
                    i += 2;
                    continue;
                }
                ctx.log_warning(&format!("Invalid value for {name} module argument"));
                return Status::Err;
            }
            if name == "MAX_QUEUED_QUERIES" {
                if i + 1 < args_str.len()
                    && let Ok(v) = args_str[i + 1].parse::<u64>()
                {
                    MAX_QUEUED_QUERIES.store(v, std::sync::atomic::Ordering::Relaxed);
                    i += 2;
                    continue;
                }
                ctx.log_warning("Invalid value for MAX_QUEUED_QUERIES module argument");
                return Status::Err;
            }
            i += 1;
        }
    }
    unsafe {
        // Disable OpenMP's pthread_atfork handlers. Without this, the
        // libomp atfork child handler crashes (SIGSEGV in __kmpc_set_lock)
        // when Redis forks for bgsave because the OMP thread pool state
        // is invalid in the child process.
        std::env::set_var("KMP_INIT_AT_FORK", "FALSE");

        // libomp's default KMP_BLOCKTIME (200ms) keeps OpenMP workers
        // spin-waiting after every GraphBLAS parallel region. With many
        // concurrent read queries the spinners starve real work — measured
        // 2.7x lower throughput at 20 parallel clients on 4-hop expansions.
        // 0 = sleep immediately, matching libgomp's effectively-short spin
        // (what the FalkorDB C image links). Respect an explicit override.
        if std::env::var_os("KMP_BLOCKTIME").is_none() {
            std::env::set_var("KMP_BLOCKTIME", "0");
        }

        let result = RediSearch_Init(ctx.ctx.cast(), REDISEARCH_INIT_LIBRARY as c_int);
        if result == REDISMODULE_OK as c_int {
            ctx.log_notice("RediSearch initialized successfully.");
        } else {
            ctx.log_notice("Failed initializing RediSearch.");
            return Status::Err;
        }

        // RediSearch 8.6 changed the default scorer from TFIDF to BM25STD.
        // FalkorDB compares absolute fulltext scores against the legacy TFIDF
        // magnitudes, so opt back in (non-fatal). Mirrors FalkorDB/FalkorDB#2021.
        if RediSearch_SetDefaultScorer(c"TFIDF".as_ptr()) != REDISMODULE_OK as c_int {
            ctx.log_warning("Failed to set RediSearch default scorer to TFIDF");
        }
        if let Err(err) = init(
            RedisModule_Alloc,
            RedisModule_Calloc,
            RedisModule_Realloc,
            RedisModule_Free,
        ) {
            ctx.log_warning(&format!("Failed to initialize GraphBLAS/LAGraph: {err}"));
            return Status::Err;
        }

        // Register fork handlers:
        // - PREPARE: on the main thread (BGSAVE path), call `Matrix::wait`
        //            on all graphs so the forked child sees a fully
        //            materialized GraphBLAS state. On non-main threads
        //            (RediSearch's ForkGC) we return immediately, mirroring
        //            the C port's `_ForkPrepare`. See
        //            [`crate::redis_type::pre_fork_prepare`].
        // - PARENT:  nothing — writers serialize against BGSAVE via the GIL
        //            during their commit phase; BGSAVE forks on the main
        //            thread which holds the GIL, so no per-fork release.
        // - CHILD:   force GraphBLAS/OpenMP to single-threaded mode so they
        //            don't touch the parent's (now-invalid) thread pools.
        pthread_atfork(
            Some(crate::redis_type::pre_fork_prepare),
            None,
            Some(on_fork_child),
        );

        let res = RedisModule_SubscribeToServerEvent.unwrap()(
            ctx.ctx,
            RedisModuleEvent_FlushDB,
            Some(on_flush),
        );
        debug_assert_eq!(res, REDISMODULE_OK as c_int);

        // Subscribe to persistence events for virtual key management.
        let res = RedisModule_SubscribeToServerEvent.unwrap()(
            ctx.ctx,
            RedisModuleEvent_Persistence,
            Some(on_persistence),
        );
        if res != REDISMODULE_OK as c_int {
            eprintln!("FalkorDB: failed to subscribe to persistence events: code {res}");
            return Status::Err;
        }

        // Subscribe to loading events to clean up virtual keys after the
        // slave finishes a full RDB resync (mirrors C's
        // ModuleEventHandler_AUXAfterKeyspaceEvent path).
        let res = RedisModule_SubscribeToServerEvent.unwrap()(
            ctx.ctx,
            RedisModuleEvent_Loading,
            Some(on_loading),
        );
        if res != REDISMODULE_OK as c_int {
            eprintln!("FalkorDB: failed to subscribe to loading events: code {res}");
            return Status::Err;
        }
    }
    match init_functions() {
        Ok(()) => {}
        Err(_) => return Status::Err,
    }
    init_udf_functions();
    udf::init_udf_repo();

    // Sync JS config values to atomics accessible without Redis GIL
    {
        let heap_size = *CONFIGURATION_JS_HEAP_SIZE.lock(ctx);
        let stack_size = *CONFIGURATION_JS_STACK_SIZE.lock(ctx);
        graph::udf::js_context::JS_HEAP_SIZE.store(heap_size, std::sync::atomic::Ordering::Relaxed);
        graph::udf::js_context::JS_STACK_SIZE
            .store(stack_size, std::sync::atomic::Ordering::Relaxed);
    }
    // Validate TEMP_FOLDER: must be an existing writable directory.
    {
        let tf_guard = CONFIGURATION_TEMP_FOLDER.lock(ctx);
        let tf = tf_guard.as_str();
        let path = std::path::Path::new(tf);
        if !path.is_dir() {
            ctx.log_warning(&format!("TEMP_FOLDER '{tf}' is not a valid directory"));
            return Status::Err;
        }
        // Check write access by attempting to create a temp file.
        let test_path = path.join(".falkordb_temp_test");
        if std::fs::File::create(&test_path).is_ok() {
            let _ = std::fs::remove_file(&test_path);
        } else {
            ctx.log_warning(&format!("TEMP_FOLDER '{tf}' is not writable"));
            return Status::Err;
        }
    }

    // Validate timeout mutual exclusion: cannot use deprecated TIMEOUT
    // together with TIMEOUT_DEFAULT / TIMEOUT_MAX.
    {
        let timeout = TIMEOUT.load(std::sync::atomic::Ordering::Relaxed);
        let timeout_default = TIMEOUT_DEFAULT.load(std::sync::atomic::Ordering::Relaxed);
        let timeout_max = TIMEOUT_MAX.load(std::sync::atomic::Ordering::Relaxed);
        if timeout > 0 && (timeout_default > 0 || timeout_max > 0) {
            ctx.log_warning("Cannot specify TIMEOUT together with TIMEOUT_DEFAULT or TIMEOUT_MAX");
            return Status::Err;
        }
        if timeout_default > 0 && timeout_max > 0 && timeout_default > timeout_max {
            ctx.log_warning("TIMEOUT_DEFAULT cannot exceed TIMEOUT_MAX");
            return Status::Err;
        }
    }

    // Initialize the thread pool with the configured thread count.
    // THREAD_COUNT may come from module args (parsed by redis_module macro).
    let tc = get_thread_count(ctx) as usize;
    let _ = init_thread_pool(tc);

    // Publish the normalized NODE_CREATION_BUFFER to the graph engine: it is
    // the chunk size matrix capacities grow by (see `Graph::grow_cap`).
    graph::graph::graph::NODE_CREATION_BUFFER.store(
        normalize_node_creation_buffer(*CONFIGURATION_NODE_CREATION_BUFFER.lock(ctx)) as u64,
        std::sync::atomic::Ordering::Relaxed,
    );

    // If OMP_THREAD_COUNT was given as a module arg, cap GraphBLAS/OpenMP
    // parallelism per operation (mirrors the C module's GxB_NTHREADS setup).
    // Otherwise GraphBLAS keeps its default (all cores) and we report the
    // thread pool size, matching prior behavior.
    let omp_tc = OMP_THREAD_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    if omp_tc > 0 {
        graph::graph::graphblas::matrix::set_nthreads(omp_tc as i32);
    } else {
        OMP_THREAD_COUNT.store(tc as i64, std::sync::atomic::Ordering::Relaxed);
    }

    // Enable RediSearch background worker threads (TIERED vector-index HNSW
    // migration jobs) when configured. Left disabled (0) by default, in which
    // case tiered inserts stay in the flat buffer (KNN still correct) and we
    // make no call — preserving prior single-threaded behavior.
    let index_workers = (*CONFIGURATION_INDEX_WORKER_THREADS.lock(ctx)).max(0) as usize;
    if index_workers > 0 {
        // RediSearch_SetNumWorkerThreads returns REDISEARCH_OK (0) on success.
        unsafe {
            if RediSearch_SetNumWorkerThreads(index_workers) != 0 {
                ctx.log_warning(
                    "Failed to set RediSearch worker thread count; \
                     RediSearch keeps its previous (disabled) worker pool. \
                     Reporting INDEX_WORKER_THREADS as 0 to match what is in effect.",
                );
                // Reset the stored config to the effective value so
                // `GRAPH.CONFIG GET INDEX_WORKER_THREADS` doesn't report a count
                // RediSearch rejected (which would hide that tiered-index
                // background work is effectively disabled).
                *CONFIGURATION_INDEX_WORKER_THREADS.lock(ctx) = 0;
            }
        }
    }

    // Start the background telemetry flusher: workers enqueue entries
    // lock-free; this thread batches them and writes XADDs under a single
    // GIL acquisition per batch.
    telemetry::start_flusher_thread();

    // Initialize cached replica state and subscribe to role-change events
    // so telemetry is suppressed when this instance is a replica (master's
    // XADDs replicate to us automatically).
    telemetry::set_is_replica(ctx.get_flags().contains(ContextFlags::SLAVE));
    unsafe {
        let res = RedisModule_SubscribeToServerEvent.unwrap()(
            ctx.ctx,
            RedisModuleEvent_ReplicationRoleChanged,
            Some(on_role_change),
        );
        debug_assert_eq!(res, REDISMODULE_OK as c_int);
    }

    // Latch effects-buffer building as soon as any replica attaches (or if
    // this instance already changed roles); until then standalone masters
    // skip serializing per-commit replication effects.
    unsafe {
        let res = RedisModule_SubscribeToServerEvent.unwrap()(
            ctx.ctx,
            RedisModuleEvent_ReplicaChange,
            Some(on_replica_change),
        );
        debug_assert_eq!(res, REDISMODULE_OK as c_int);
    }
    if ctx.get_flags().contains(ContextFlags::SLAVE) {
        // Loaded on a replica: replication topology already exists.
        crate::graph_core::REPLICATION_CONSUMERS.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    // Subscribe to keyspace notifications for graph key rename handling.
    unsafe {
        let res = redis_module::raw::RedisModule_SubscribeToKeyspaceEvents.unwrap()(
            ctx.ctx,
            4, // REDISMODULE_NOTIFY_GENERIC (covers RENAME)
            Some(on_keyspace_event),
        );
        if res != REDISMODULE_OK as c_int {
            eprintln!("FalkorDB: failed to subscribe to keyspace events: code {res}");
            return Status::Err;
        }
    }

    // Wire shutdown cleanup only when the runner sets `RS_GLOBAL_DTORS`
    // (sanitizer/valgrind). The handler joins worker threads, finalizes
    // LAGraph, and frees module-level RediSearch state — work that is
    // pointless when the kernel is about to reap the process anyway.
    if std::env::var_os("RS_GLOBAL_DTORS").is_some() {
        unsafe {
            let res = RedisModule_SubscribeToServerEvent.unwrap()(
                ctx.ctx,
                RedisModuleEvent_Shutdown,
                Some(on_shutdown),
            );
            if res != REDISMODULE_OK as c_int {
                eprintln!("FalkorDB: failed to subscribe to shutdown event: code {res}");
                return Status::Err;
            }
        }
    }

    Status::Ok
}

const unsafe extern "C" fn on_flush(
    _ctx: *mut RedisModuleCtx,
    _eid: RedisModuleEvent,
    _subevent: u64,
    _data: *mut c_void,
) {
}

/// Shutdown event handler — runs only under `RS_GLOBAL_DTORS` (sanitizer
/// or valgrind). Mirrors the C implementation's `_ShutdownEventHandler`:
/// join worker threads (so their TLS destructors fire), finalize LAGraph,
/// then free RediSearch module-level state. Skipped in normal production
/// runs to avoid spending time on cleanup the kernel will do for us.
unsafe extern "C" fn on_shutdown(
    _ctx: *mut RedisModuleCtx,
    _eid: RedisModuleEvent,
    _subevent: u64,
    _data: *mut c_void,
) {
    // Stop the telemetry flusher first: it issues RM_Call("XADD") on a
    // thread-safe context, which races with Redis tearing down server state
    // and produces a SIGSEGV under ASAN if left running.
    telemetry::shutdown_flusher_thread();
    threadpool::shutdown();
    graph::graph::graphblas::matrix::shutdown();
    unsafe { RediSearch_CleanupModule() };
}

unsafe extern "C" fn on_role_change(
    _ctx: *mut RedisModuleCtx,
    _eid: RedisModuleEvent,
    subevent: u64,
    _data: *mut c_void,
) {
    telemetry::set_is_replica(subevent == REDISMODULE_EVENT_REPLROLECHANGED_NOW_REPLICA);
    // A role change means a replication topology exists (or is being set
    // up); keep building effects buffers from here on.
    crate::graph_core::REPLICATION_CONSUMERS.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Any replica attach/detach latches the sticky "replication has
/// consumers" flag: even after the last replica detaches it may resume
/// from the replication backlog, so effects buffers must keep being built.
unsafe extern "C" fn on_replica_change(
    _ctx: *mut RedisModuleCtx,
    _eid: RedisModuleEvent,
    _subevent: u64,
    _data: *mut c_void,
) {
    crate::graph_core::REPLICATION_CONSUMERS.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Loading event callback. After a slave finishes a full RDB resync from
/// the master, drop any virtual keys that came along in the snapshot —
/// their content has already been merged into the main graph key.
unsafe extern "C" fn on_loading(
    ctx: *mut RedisModuleCtx,
    _eid: RedisModuleEvent,
    subevent: u64,
    _data: *mut c_void,
) {
    if subevent == REDISMODULE_SUBEVENT_LOADING_ENDED
        || subevent == REDISMODULE_SUBEVENT_LOADING_FAILED
    {
        crate::redis_type::finalize_pending_graphs();
        unsafe { crate::redis_type::delete_stale_virtual_keys(ctx) };
    }
}

/// Tracks the old key name during a two-phase RENAME notification.
static RENAME_OLD_NAME: parking_lot::Mutex<Option<String>> = parking_lot::Mutex::new(None);

/// Keyspace event callback for handling graph key renames.
/// Redis RENAME fires two sequential events on the same thread:
/// 1. `rename_from` with the old key name
/// 2. `rename_to` with the new key name
unsafe extern "C" fn on_keyspace_event(
    ctx: *mut RedisModuleCtx,
    _type: c_int,
    event: *const std::os::raw::c_char,
    key: *mut redis_module::raw::RedisModuleString,
) -> c_int {
    let event_str = unsafe { std::ffi::CStr::from_ptr(event) }
        .to_str()
        .unwrap_or("");

    let mut key_len: usize = 0;
    let key_ptr =
        unsafe { redis_module::raw::RedisModule_StringPtrLen.unwrap()(key, &raw mut key_len) };
    let key_bytes = unsafe { std::slice::from_raw_parts(key_ptr.cast(), key_len) };
    let Ok(key_name) = std::str::from_utf8(key_bytes) else {
        return 0;
    };

    match event_str {
        "rename_from" => {
            *RENAME_OLD_NAME.lock() = Some(key_name.to_string());
        }
        "rename_to" => {
            let old = RENAME_OLD_NAME.lock().take();
            if let Some(old_name) = old {
                crate::graph_core::rename_graph(&old_name, key_name);
                let context = Context::new(ctx);
                telemetry::delete_stream(&context, &old_name);
            }
        }
        _ => {}
    }
    0
}
