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
    CONFIGURATION_JS_HEAP_SIZE, CONFIGURATION_JS_STACK_SIZE, CONFIGURATION_TEMP_FOLDER,
    OMP_THREAD_COUNT, TIMEOUT, TIMEOUT_DEFAULT, TIMEOUT_MAX, get_thread_count,
};
use crate::redis_type::on_persistence;
use crate::telemetry;
use graph::{
    graph::graphblas::matrix::init,
    index::redisearch::{REDISEARCH_INIT_LIBRARY, RediSearch_Init},
    runtime::functions::{init_functions, init_udf_functions},
    threadpool::init_thread_pool,
    udf,
};
use redis_module::{
    Context, ContextFlags, REDISMODULE_OK, RedisModule_Alloc, RedisModule_Calloc, RedisModule_Free,
    RedisModule_Realloc, RedisModule_SubscribeToServerEvent, RedisModuleCtx, RedisModuleEvent,
    Status,
};
use std::{os::raw::c_int, os::raw::c_void, panic};

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

/// Subevent: this instance is now a replica.
const REDISMODULE_EVENT_REPLROLECHANGED_NOW_REPLICA: u64 = 1;

unsafe extern "C" {
    fn pthread_atfork(
        prepare: Option<unsafe extern "C" fn()>,
        parent: Option<unsafe extern "C" fn()>,
        child: Option<unsafe extern "C" fn()>,
    ) -> c_int;
}

/// Called in the forked child process (via `pthread_atfork`).
/// Forces GraphBLAS/OpenMP to single-threaded mode so they don't
/// touch the parent's (now-invalid) thread pool handles.
unsafe extern "C" fn on_fork_child() {
    graph::graph::graphblas::matrix::set_nthreads(1);
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
        );
        unsafe {
            if let Some(log) = graph::index::redisearch::redis::RedisModule_Log {
                // Strip any internal NUL bytes so CString::new succeeds.
                let sanitized: String =
                    msg.chars().map(|c| if c == '\0' { ' ' } else { c }).collect();
                if let Ok(c_msg) = std::ffi::CString::new(sanitized) {
                    log(std::ptr::null_mut(), c"warning".as_ptr(), c"%s".as_ptr(), c_msg.as_ptr());
                }
            }
        }
        std::process::exit(1);
    }));

    // Parse timeout-related module args (TIMEOUT, TIMEOUT_DEFAULT, TIMEOUT_MAX).
    // These are AtomicI64 statics not registered in the redis_module! config section,
    // so we parse them manually here.
    {
        let args_str: Vec<String> = args
            .iter()
            .map(redis_module::RedisString::to_string_lossy)
            .collect();
        let mut i = 0;
        while i < args_str.len() {
            match args_str[i].to_uppercase().as_str() {
                "TIMEOUT" => {
                    if i + 1 < args_str.len()
                        && let Ok(v) = args_str[i + 1].parse::<i64>()
                    {
                        TIMEOUT.store(v, std::sync::atomic::Ordering::Relaxed);
                        i += 2;
                        continue;
                    }
                    ctx.log_warning("Invalid value for TIMEOUT module argument");
                    return Status::Err;
                }
                "TIMEOUT_DEFAULT" => {
                    if i + 1 < args_str.len()
                        && let Ok(v) = args_str[i + 1].parse::<i64>()
                    {
                        TIMEOUT_DEFAULT.store(v, std::sync::atomic::Ordering::Relaxed);
                        i += 2;
                        continue;
                    }
                    ctx.log_warning("Invalid value for TIMEOUT_DEFAULT module argument");
                    return Status::Err;
                }
                "TIMEOUT_MAX" => {
                    if i + 1 < args_str.len()
                        && let Ok(v) = args_str[i + 1].parse::<i64>()
                    {
                        TIMEOUT_MAX.store(v, std::sync::atomic::Ordering::Relaxed);
                        i += 2;
                        continue;
                    }
                    ctx.log_warning("Invalid value for TIMEOUT_MAX module argument");
                    return Status::Err;
                }
                _ => {
                    i += 1;
                }
            }
        }
    }
    unsafe {
        // Disable OpenMP's pthread_atfork handlers. Without this, the
        // libomp atfork child handler crashes (SIGSEGV in __kmpc_set_lock)
        // when Redis forks for bgsave because the OMP thread pool state
        // is invalid in the child process.
        std::env::set_var("KMP_INIT_AT_FORK", "FALSE");

        let result = RediSearch_Init(ctx.ctx.cast(), REDISEARCH_INIT_LIBRARY as c_int);
        if result == REDISMODULE_OK as c_int {
            ctx.log_notice("RediSearch initialized successfully.");
        } else {
            ctx.log_notice("Failed initializing RediSearch.");
            return Status::Err;
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

        // Register fork child handler to make GraphBLAS/OpenMP single-threaded
        // in bgsave child processes. Prepare handler materializes all GraphBLAS
        // matrices so the child doesn't hit held internal locks.
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
    OMP_THREAD_COUNT.store(tc as i64, std::sync::atomic::Ordering::Relaxed);

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

    Status::Ok
}

const unsafe extern "C" fn on_flush(
    _ctx: *mut RedisModuleCtx,
    _eid: RedisModuleEvent,
    _subevent: u64,
    _data: *mut c_void,
) {
}

unsafe extern "C" fn on_role_change(
    _ctx: *mut RedisModuleCtx,
    _eid: RedisModuleEvent,
    subevent: u64,
    _data: *mut c_void,
) {
    telemetry::set_is_replica(subevent == REDISMODULE_EVENT_REPLROLECHANGED_NOW_REPLICA);
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
    let key_ptr = unsafe {
        redis_module::raw::RedisModule_StringPtrLen.unwrap()(key, &mut key_len as *mut usize)
    };
    let key_bytes = unsafe { std::slice::from_raw_parts(key_ptr.cast(), key_len) };
    let key_name = match std::str::from_utf8(key_bytes) {
        Ok(s) => s,
        Err(_) => return 0,
    };

    match event_str {
        "rename_from" => {
            *RENAME_OLD_NAME.lock() = Some(key_name.to_string());
        }
        "rename_to" => {
            if let Some(old_name) = RENAME_OLD_NAME.lock().take() {
                let context = Context::new(ctx);
                telemetry::delete_stream(&context, &old_name);
            }
        }
        _ => {}
    }
    0
}
