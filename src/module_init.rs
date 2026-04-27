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
use graph::{
    graph::graphblas::matrix::init,
    index::redisearch::{REDISEARCH_INIT_LIBRARY, RediSearch_Init},
    runtime::functions::{init_functions, init_udf_functions},
    threadpool::init_thread_pool,
    udf,
};
use redis_module::{
    Context, REDISMODULE_OK, RedisModule_Alloc, RedisModule_Calloc, RedisModule_Free,
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
    args: &Vec<redis_module::RedisString>,
) -> Status {
    panic::set_hook(Box::new(|info| {
        eprintln!("FalkorDB panic: {info}");
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
        // in bgsave child processes.
        pthread_atfork(None, None, Some(on_fork_child));

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
    Status::Ok
}

const unsafe extern "C" fn on_flush(
    _ctx: *mut RedisModuleCtx,
    _eid: RedisModuleEvent,
    _subevent: u64,
    _data: *mut c_void,
) {
}
