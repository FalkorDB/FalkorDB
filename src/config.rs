//! Global FalkorDB configuration values.
//!
//! Configuration is split into two tiers based on mutability and access
//! requirements:
//!
//! ```text
//! +-------------------------------+    +----------------------------+
//! | RedisGILGuard configs         |    | Atomic configs             |
//! | (set at module load)          |    | (changeable at runtime)    |
//! |-------------------------------|    |----------------------------|
//! | CACHE_SIZE                    |    | TIMEOUT / TIMEOUT_DEFAULT  |
//! | THREAD_COUNT                  |    | TIMEOUT_MAX                |
//! | NODE_CREATION_BUFFER          |    | MAX_QUEUED_QUERIES         |
//! | VKEY_MAX_ENTITY_COUNT         |    | RESULTSET_SIZE             |
//! | IMPORT_FOLDER / TEMP_FOLDER   |    | QUERY_MEM_CAPACITY         |
//! | JS_HEAP_SIZE / JS_STACK_SIZE  |    | DELTA_MAX_PENDING_CHANGES  |
//! | DELAY_INDEXING                |    | CMD_INFO                   |
//! |                               |    | OMP_THREAD_COUNT ...       |
//! +-------------------------------+    +----------------------------+
//!       |                                      |
//!       | requires Redis GIL to read           | lock-free atomic reads
//!       v                                      v
//!   module arg parsing                    GRAPH.CONFIG SET / GET
//! ```
//!
//! - **GIL-guarded configs** are declared via `lazy_static` and wrapped in
//!   `RedisGILGuard`, ensuring safe access from the Redis main thread.
//!   Most are immutable after module load (flagged `IMMUTABLE` in the
//!   `redis_module!` macro); the ones C also lets change at run-time are
//!   flagged `DEFAULT` and accepted by `GRAPH.CONFIG SET`.
//!
//! - **Atomic configs** use `AtomicI64` / `AtomicU64` / `AtomicBool` for
//!   lock-free concurrent reads from worker threads. Some are
//!   runtime-configurable through `GRAPH.CONFIG SET`; others are read-only
//!   after init.
//!
//! `CONFIG_NAMES` provides the ordered list of all configuration keys,
//! used by `GRAPH.CONFIG GET *` to enumerate settings.

use lazy_static::lazy_static;
use redis_module::RedisGILGuard;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64};

// ── Redis module-level configurations (set via moduleArgs) ──

lazy_static! {
    pub static ref CONFIGURATION_IMPORT_FOLDER: RedisGILGuard<String> =
        RedisGILGuard::new("/var/lib/FalkorDB/import/".into());
    pub static ref CONFIGURATION_CACHE_SIZE: RedisGILGuard<i64> = RedisGILGuard::new(25.into());
    pub static ref CONFIGURATION_THREAD_COUNT: RedisGILGuard<i64> = RedisGILGuard::new(0.into());
    /// Number of RediSearch background worker threads (e.g. TIERED vector-index
    /// HNSW migration jobs). 0 = disabled: tiered inserts stay in the flat
    /// buffer (KNN still correct), no async migration.
    pub static ref CONFIGURATION_INDEX_WORKER_THREADS: RedisGILGuard<i64> =
        RedisGILGuard::new(0.into());
    pub static ref CONFIGURATION_NODE_CREATION_BUFFER: RedisGILGuard<i64> =
        RedisGILGuard::new(graph::graph::graph::DEFAULT_NODE_CREATION_BUFFER as i64);
    pub static ref CONFIGURATION_VKEY_MAX_ENTITY_COUNT: RedisGILGuard<i64> =
        RedisGILGuard::new(100_000.into());
    pub static ref CONFIGURATION_DELAY_INDEXING: RedisGILGuard<bool> = RedisGILGuard::new(false);
    pub static ref CONFIGURATION_TEMP_FOLDER: RedisGILGuard<String> =
        RedisGILGuard::new("/tmp".into());
    pub static ref CONFIGURATION_JS_HEAP_SIZE: RedisGILGuard<i64> =
        RedisGILGuard::new(256 * 1024 * 1024_i64);
    pub static ref CONFIGURATION_JS_STACK_SIZE: RedisGILGuard<i64> =
        RedisGILGuard::new(1024 * 1024_i64);
}

// ── Runtime-configurable atomics ──

pub static MAX_QUEUED_QUERIES: AtomicU64 = AtomicU64::new(u32::MAX as u64);
pub static TIMEOUT: AtomicI64 = AtomicI64::new(0);
pub static TIMEOUT_DEFAULT: AtomicI64 = AtomicI64::new(0);
pub static TIMEOUT_MAX: AtomicI64 = AtomicI64::new(0);
pub static RESULTSET_SIZE: AtomicI64 = AtomicI64::new(-1);
pub static QUERY_MEM_CAPACITY: AtomicI64 = AtomicI64::new(0);
pub static DELTA_MAX_PENDING_CHANGES: AtomicI64 = AtomicI64::new(10000);
/// Smallest v3 effects payload worth compressing, in bytes; 0 disables it.
/// Re-exported because the decision is made in the graph crate, and two statics
/// would drift.
pub use graph::effects::v3::EFFECTS_COMPRESSION;
/// **Deprecated and ignored.** v3 is the only wire format and effects are the
/// only mechanism, so there is nothing left for either knob to select.
///
/// Kept settable rather than removed. Removing them made `GRAPH.CONFIG SET
/// EFFECTS_THRESHOLD` answer `Unknown configuration field`, which is a startup
/// failure for any config file that names one and broke five flow suites that
/// set it. A value is accepted, stored, reported back, and read by nothing.
pub static EFFECTS_VERSION_DEPRECATED: AtomicI64 = AtomicI64::new(3);
pub static EFFECTS_THRESHOLD_DEPRECATED: AtomicI64 = AtomicI64::new(0);
/// Toggles the telemetry stream that backs `GRAPH.INFO`. Atomic rather than
/// GIL-guarded because the query hot path reads it off the main thread, and
/// because C lets `GRAPH.CONFIG SET CMD_INFO yes|no` flip it at runtime.
pub static CONFIGURATION_CMD_INFO: AtomicBool = AtomicBool::new(true);
/// Cap on the telemetry stream's length, applied by the flusher's stream trim.
/// 0 means "keep nothing", the way C's unconditional `StreamTrimByLength` does.
pub static MAX_INFO_QUERIES: AtomicI64 = AtomicI64::new(MAX_INFO_QUERIES_CAP);
/// Whether graph teardown may happen off the calling thread. Settable at run-time as
/// in C, but not yet read: this engine always frees off-thread (see
/// `graph_core::graph_free`).
pub static ASYNC_DELETE: AtomicI64 = AtomicI64::new(0);

// ── Read-only runtime configs ──

pub static OMP_THREAD_COUNT: AtomicI64 = AtomicI64::new(0);
pub static BOLT_PORT: AtomicI64 = AtomicI64::new(65535);

/// Highest accepted `MAX_INFO_QUERIES`, and its default. Larger values are
/// clamped rather than rejected, matching C's `Config_cmd_info_max_queries_set`.
pub const MAX_INFO_QUERIES_CAP: i64 = 1000;

pub fn get_thread_count(ctx: &redis_module::Context) -> i64 {
    let val = *CONFIGURATION_THREAD_COUNT.lock(ctx);
    if val > 0 {
        val
    } else {
        std::thread::available_parallelism().map_or(4, |n| n.get() as i64)
    }
}

/// Round up to the next power of 2, with a minimum of 128.
pub fn normalize_node_creation_buffer(val: i64) -> i64 {
    let val = val.max(128) as u64;
    val.next_power_of_two() as i64
}

/// Ordered list of all configuration names for `GET *`.
pub const CONFIG_NAMES: &[&str] = &[
    "TIMEOUT",
    "TIMEOUT_DEFAULT",
    "TIMEOUT_MAX",
    "CACHE_SIZE",
    "ASYNC_DELETE",
    "OMP_THREAD_COUNT",
    "THREAD_COUNT",
    "INDEX_WORKER_THREADS",
    "RESULTSET_SIZE",
    "VKEY_MAX_ENTITY_COUNT",
    "MAX_QUEUED_QUERIES",
    "QUERY_MEM_CAPACITY",
    "DELTA_MAX_PENDING_CHANGES",
    "NODE_CREATION_BUFFER",
    "CMD_INFO",
    "MAX_INFO_QUERIES",
    "EFFECTS_COMPRESSION",
    "EFFECTS_VERSION",
    "EFFECTS_THRESHOLD",
    "BOLT_PORT",
    "DELAY_INDEXING",
    "IMPORT_FOLDER",
    "TEMP_FOLDER",
    "JS_HEAP_SIZE",
    "JS_STACK_SIZE",
];
