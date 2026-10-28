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

use crate::commands::config_cmd::{
    ConfigValue, UNKNOWN_FIELD, apply_config_set, check_timeout_constraints, validate_config_set,
};
use lazy_static::lazy_static;
use redis_module::{Context, RedisGILGuard};
use std::fs::{self, File};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};

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
    /// Workers in the native index's background build pool (feature `index-falkordb`), which runs
    /// the indexer batches of every index `CREATE INDEX` leaves `Building`. Load-time only. Not
    /// listed by `GRAPH.CONFIG`, whose list mirrors the C engine's.
    pub static ref CONFIGURATION_INDEX_BUILD_THREAD_COUNT: RedisGILGuard<i64> =
        RedisGILGuard::new(1.into());
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

/// Queries that may wait at once — in the thread pool plus every graph's write
/// queue — before new ones get "Max pending queries exceeded". Deliberately
/// finite, unlike C's unlimited default, so a stalled graph cannot queue
/// without bound.
pub static MAX_QUEUED_QUERIES: AtomicU64 = AtomicU64::new(10_000);
pub static TIMEOUT: AtomicI64 = AtomicI64::new(0);
pub static TIMEOUT_DEFAULT: AtomicI64 = AtomicI64::new(0);
pub static TIMEOUT_MAX: AtomicI64 = AtomicI64::new(0);
pub static RESULTSET_SIZE: AtomicI64 = AtomicI64::new(-1);
pub static QUERY_MEM_CAPACITY: AtomicI64 = AtomicI64::new(0);
pub static DELTA_MAX_PENDING_CHANGES: AtomicI64 = AtomicI64::new(10000);
/// Smallest v3 effects payload worth compressing, in bytes; 0 disables it.
/// Re-exported because the decision is made in the graph crate, and two statics
/// would drift.
pub use graph::effects::EFFECTS_COMPRESSION;
/// **Deprecated and ignored.** Effects are the only replication mechanism now,
/// so there is nothing left for the threshold to select.
///
/// Kept settable rather than removed. Removing it made `GRAPH.CONFIG SET
/// EFFECTS_THRESHOLD` answer `Unknown configuration field`, which is a startup
/// failure for any config file that names it and broke five flow suites that
/// set it. A value is accepted, stored, reported back, and read by nothing.
///
/// Still 300, as it always was. The knob is ignored, but its *reported* value
/// is observable — changing it would break a client or config check that
/// asserts the default, for no benefit, which is the opposite of keeping it for
/// compatibility.
///
/// There is deliberately no `EFFECTS_VERSION` beside it. That one never
/// shipped in either engine — `main` has 23 configs and only this one, and C's
/// `config.h` has `Config_EFFECTS_THRESHOLD` and no version knob — so a
/// deprecated no-op for it would be compatibility with nothing.
pub static EFFECTS_THRESHOLD_DEPRECATED: AtomicI64 = AtomicI64::new(300);
/// Toggles the telemetry stream that backs `GRAPH.INFO`. Atomic rather than
/// GIL-guarded because the query hot path reads it off the main thread, and
/// because C lets `GRAPH.CONFIG SET CMD_INFO yes|no` flip it at runtime.
pub static CONFIGURATION_CMD_INFO: AtomicBool = AtomicBool::new(true);
/// Cap on the telemetry stream's length, applied by the flusher's stream trim.
/// 0 means "keep nothing", the way C's unconditional `StreamTrimByLength` does.
pub static MAX_INFO_QUERIES: AtomicI64 = AtomicI64::new(MAX_INFO_QUERIES_CAP);
/// Whether graph teardown may happen off the calling thread. Settable at run-time as
/// in C, but not yet read: this engine always frees off-thread (see
/// `graph_core::graph_free`), so it defaults to 1 — C's default, and what happens.
pub static ASYNC_DELETE: AtomicI64 = AtomicI64::new(1);

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
    "EFFECTS_THRESHOLD",
    "BOLT_PORT",
    "DELAY_INDEXING",
    "IMPORT_FOLDER",
    "TEMP_FOLDER",
    "JS_HEAP_SIZE",
    "JS_STACK_SIZE",
];

// ── Module load arguments ──

/// Configs C has that this engine accepts and stores but does not act on yet.
/// Setting one at load time logs that the value is accepted and ignored.
const IGNORED_CONFIGS: &[&str] = &[
    "ASYNC_DELETE",
    "DELAY_INDEXING",
    "DELTA_MAX_PENDING_CHANGES",
    "EFFECTS_THRESHOLD",
];

/// A validated module load argument, ready to apply.
enum LoadArg {
    CacheSize(i64),
    ThreadCount(i64),
    OmpThreadCount(i64),
    NodeCreationBuffer(i64),
    IndexWorkerThreads(i64),
    IndexBuildThreadCount(i64),
    ImportFolder(String),
    TempFolder(String),
    /// Not validated at load time (yet); accepted and unused, as before.
    BoltPort,
    /// A config `GRAPH.CONFIG SET` also accepts, applied through its setter;
    /// the raw value is kept for the log line.
    Runtime {
        name: String,
        value: String,
        parsed: ConfigValue,
    },
}

/// C's `TEMP_FOLDER` checks, with its error texts.
fn validate_temp_folder(path: &str) -> Result<(), String> {
    let p = Path::new(path);
    if !p.exists() {
        return Err("TEMP_FOLDER path does not exist".to_string());
    }
    if !p.is_dir() {
        return Err("TEMP_FOLDER path is not a directory".to_string());
    }
    let probe = p.join(".falkordb_temp_test");
    if File::create(&probe).is_err() {
        return Err("No write access to TEMP_FOLDER path".to_string());
    }
    let _ = fs::remove_file(&probe);
    Ok(())
}

/// Validate one load-only config (one `GRAPH.CONFIG SET` refuses) with C's
/// load-time rules. Returns `None` when `name` is not a load-only config.
fn validate_load_only(
    name: &str,
    value: &str,
) -> Option<Result<LoadArg, String>> {
    // C's `_Config_ParsePositiveInteger` / `_Config_ParseNonNegativeInteger`:
    // a decimal `long long` (so `-0` is 0 and 2^63 is out of range), then a sign
    // check.
    let int = value.parse::<i64>().ok();
    let positive = int.filter(|v| v.is_positive());
    let non_negative = int.filter(|v| !v.is_negative());
    let checked = match name {
        "CACHE_SIZE" => positive.map(LoadArg::CacheSize),
        "THREAD_COUNT" => positive.map(LoadArg::ThreadCount),
        "OMP_THREAD_COUNT" => positive.map(LoadArg::OmpThreadCount),
        "NODE_CREATION_BUFFER" => non_negative.map(LoadArg::NodeCreationBuffer),
        "INDEX_WORKER_THREADS" => non_negative.map(LoadArg::IndexWorkerThreads),
        // This engine's own: the native index's build pool, 1 to 1024 workers as registered.
        "INDEX_BUILD_THREAD_COUNT" => positive
            .filter(|&v| v <= 1024)
            .map(LoadArg::IndexBuildThreadCount),
        "IMPORT_FOLDER" => Some(LoadArg::ImportFolder(value.to_string())),
        "TEMP_FOLDER" => {
            return Some(
                validate_temp_folder(value).map(|()| LoadArg::TempFolder(value.to_string())),
            );
        }
        "BOLT_PORT" => Some(LoadArg::BoltPort),
        _ => return None,
    };
    Some(checked.ok_or_else(|| format!("Failed to set config value {name} to {value}")))
}

/// Apply one validated load argument.
fn apply_load_arg(
    ctx: &Context,
    arg: LoadArg,
) {
    match arg {
        LoadArg::CacheSize(v) => *CONFIGURATION_CACHE_SIZE.lock(ctx) = v,
        LoadArg::ThreadCount(v) => *CONFIGURATION_THREAD_COUNT.lock(ctx) = v,
        LoadArg::OmpThreadCount(v) => OMP_THREAD_COUNT.store(v, Ordering::Relaxed),
        LoadArg::NodeCreationBuffer(v) => *CONFIGURATION_NODE_CREATION_BUFFER.lock(ctx) = v,
        LoadArg::IndexWorkerThreads(v) => *CONFIGURATION_INDEX_WORKER_THREADS.lock(ctx) = v,
        LoadArg::IndexBuildThreadCount(v) => *CONFIGURATION_INDEX_BUILD_THREAD_COUNT.lock(ctx) = v,
        LoadArg::ImportFolder(v) => *CONFIGURATION_IMPORT_FOLDER.lock(ctx) = v,
        LoadArg::TempFolder(v) => *CONFIGURATION_TEMP_FOLDER.lock(ctx) = v,
        LoadArg::BoltPort => {}
        LoadArg::Runtime {
            name,
            value,
            parsed,
        } => {
            apply_config_set(ctx, &name, &parsed);
            if IGNORED_CONFIGS.contains(&name.as_str()) {
                ctx.log_warning(&format!(
                    "FalkorDB: configuration field '{name}' is deprecated and has no effect in \
                     this version; the value '{value}' is accepted and ignored"
                ));
            }
        }
    }
}

/// Apply the module's load arguments (`--loadmodule falkordb.so NAME VALUE ...`)
/// the way C's `Config_Init` does: names are case-insensitive, every value is
/// validated, and an odd argument count, an unknown name or an invalid value
/// refuses the load. Run-time configs go through the same validation and
/// setter as `GRAPH.CONFIG SET`; load-only configs get C's load-time rules.
///
/// Every pair is validated before any is applied, so a refused load leaves no
/// partial configuration behind. The pairs are still checked in order, each
/// against the timeouts the pairs before it set, so the first failure and its
/// message are the ones C's one-pair-at-a-time `Config_Init` reports.
///
/// Runs in `graph_init`, after Redis has applied any `graph.*` lines from
/// redis.conf, so a module argument wins over the same `graph.*` config.
///
/// Returns the message to log on failure.
pub fn apply_module_args(
    ctx: &Context,
    args: &[String],
) -> Result<(), String> {
    if args.len() % 2 == 1 {
        return Err(format!(
            "FalkorDB received {} arguments, all configurations should be key-value pairs",
            args.len()
        ));
    }

    let mut timeout_default = TIMEOUT_DEFAULT.load(Ordering::Relaxed);
    let mut timeout_max = TIMEOUT_MAX.load(Ordering::Relaxed);
    let mut old_timeout_specified = false;
    let mut new_timeout_specified = false;
    let mut validated = Vec::with_capacity(args.len() / 2);
    for [field, value] in args.as_chunks::<2>().0 {
        let name = field.to_ascii_uppercase();
        let failed = |err: String| format!("Failed setting field '{field}' with error: {err}");

        if let Some(arg) = validate_load_only(&name, value) {
            validated.push(arg.map_err(failed)?);
            continue;
        }
        let parsed = validate_config_set(&name, value).map_err(|err| {
            if err == UNKNOWN_FIELD {
                format!("Encountered unknown configuration field '{field}'")
            } else {
                failed(err)
            }
        })?;
        let pair = [(name.as_str(), parsed)];
        check_timeout_constraints(&mut timeout_default, &mut timeout_max, &pair).map_err(failed)?;
        let [(_, parsed)] = pair;
        match name.as_str() {
            "TIMEOUT" => old_timeout_specified = true,
            "TIMEOUT_DEFAULT" | "TIMEOUT_MAX" => new_timeout_specified = true,
            _ => {}
        }
        validated.push(LoadArg::Runtime {
            name,
            value: value.clone(),
            parsed,
        });
    }

    if old_timeout_specified && new_timeout_specified {
        return Err(
            "The TIMEOUT configuration parameter should be removed when specifying \
             TIMEOUT_DEFAULT and/or TIMEOUT_MAX"
                .to_string(),
        );
    }

    for arg in validated {
        apply_load_arg(ctx, arg);
    }
    Ok(())
}
