//! Telemetry subsystem: stream writing and running/waiting query registry.
//!
//! After each query execution, a telemetry entry is written to a Redis stream
//! named `telemetry{graph_name}`. The GRAPH.INFO command reads the global
//! query registry to report currently running and waiting queries.

use crossfire::mpmc::{self, List};
use crossfire::{MRx, MTx};
use parking_lot::{Mutex, RwLock};
use redis_module::key::KeyFlags;
use redis_module::logging::log_warning;
use redis_module::{Context, ContextFlags, RedisString, RedisValue, raw};
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Weak};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::config::{CONFIGURATION_CMD_INFO, MAX_INFO_QUERIES};
use crate::graph_core::{GRAPH_REGISTRY, ThreadedGraph, up_to_nul};
use crate::redis_type::GRAPH_TYPE;

/// Maximum stored string length for query/params in telemetry entries.
const STR_MAX_LEN: usize = 2048;

// ---------------------------------------------------------------------------
// String truncation
// ---------------------------------------------------------------------------

/// Truncate `s` to at most `STR_MAX_LEN` characters, appending `"..."` if
/// truncated.
pub fn truncate(s: &str) -> String {
    if let Some((byte_idx, _)) = s.char_indices().nth(STR_MAX_LEN) {
        let mut t = String::with_capacity(byte_idx + 3);
        t.push_str(&s[..byte_idx]);
        t.push_str("...");
        t
    } else {
        s.to_owned()
    }
}

/// Truncate an `Arc<str>` to at most `STR_MAX_LEN` characters.
///
/// The common case (query already within the limit) clones the `Arc`
/// (a refcount bump, no byte copy) instead of reallocating the string.
/// Only over-long queries pay an allocation, and only to bound stored memory.
fn truncate_arc(s: &Arc<str>) -> Arc<str> {
    if let Some((byte_idx, _)) = s.char_indices().nth(STR_MAX_LEN) {
        let mut t = String::with_capacity(byte_idx + 3);
        t.push_str(&s[..byte_idx]);
        t.push_str("...");
        Arc::from(t)
    } else {
        Arc::clone(s)
    }
}

// ---------------------------------------------------------------------------
// Telemetry stream writing
// ---------------------------------------------------------------------------

/// Build the Redis stream key name for a graph's telemetry.
pub fn stream_name(graph_name: &str) -> String {
    // Named after the graph, not after the key it lives at: C builds this with
    // `RM_CreateStringPrintf(NULL, "telemetry{%s}", gc->graph_name)`, so a graph whose
    // key holds an interior NUL — which a `RENAME` can leave behind — streams under the
    // part before it. Truncating here rather than at the call sites keeps the callers
    // free to pass the key, which is what the rest of the flush path addresses Redis by.
    format!("telemetry{{{}}}", up_to_nul(graph_name))
}

/// Data for one telemetry stream entry (10 fields).
pub struct TelemetryEntry {
    pub received_at: i64,
    pub query: String,
    pub params: String,
    pub wait_duration_ms: f64,
    pub execution_duration_ms: f64,
    pub report_duration_ms: f64,
    pub utilized_cache: bool,
    pub is_write: bool,
    pub timed_out: bool,
}

/// Field names of a telemetry entry, in the order they are written.
///
/// Kept as one list so the names and the values written beside them cannot drift
/// apart, and so the `RedisModuleString`s built from them can be created once for
/// the whole process rather than per entry.
const FIELD_NAMES: [&str; FIELD_COUNT] = [
    "Received at",
    "Query",
    "Query parameters",
    "Total duration",
    "Wait duration",
    "Execution duration",
    "Report duration",
    "Utilized cache",
    "Write",
    "Timeout",
];

const FIELD_COUNT: usize = 10;

/// The ten constant field-name `RedisModuleString`s, shared by every write.
///
/// A pointer is not `Send`, but these are: created once on the main thread during
/// module init, then only ever read. `RM_StreamAdd` copies the field bytes into
/// the stream's listpack rather than retaining the object, so no refcount is
/// touched and no synchronisation is needed past the initial publication.
struct FieldNameStrings([*mut raw::RedisModuleString; FIELD_COUNT]);

// SAFETY: see the type's documentation — write-once on the init thread, read-only
// thereafter, and the pointees are immutable for their whole lifetime.
unsafe impl Send for FieldNameStrings {}
unsafe impl Sync for FieldNameStrings {}

/// Created by [`start_flusher_thread`] and freed by [`shutdown_flusher_thread`],
/// so an unload/reload cycle neither leaks them nor reuses freed ones.
static FIELD_NAME_STRINGS: Mutex<Option<FieldNameStrings>> = Mutex::new(None);

/// The reusable half of a stream write: field names paired with per-entry values.
///
/// This is the port of the C engine's `_event` template
/// (`cron/tasks/stream_finished_queries.c`), and the reason it exists is cost.
/// Writing an entry with `RM_Call("XADD", ...)` pays Redis's command lookup,
/// argument vector, call dispatch and reply machinery — 25,534 instructions a
/// query measured on an M3 Pro — plus 2,395 more to propagate each XADD to
/// replicas, plus a fresh `RedisModuleString` for all 25 tokens including the
/// ten constant field names. Writing through the key API pays none of that.
///
/// Strings are created with a NULL context, so Redis does not tie them to a
/// command's auto-memory pool.
struct StreamTemplate {
    /// `[name0, value0, name1, value1, …]` — `StreamAdd` takes field/value pairs
    /// flattened. The name slots are borrowed from [`FIELD_NAME_STRINGS`] and
    /// never freed here; the value slots are refilled and freed per entry.
    argv: [*mut raw::RedisModuleString; FIELD_COUNT * 2],
    /// `StreamAdd` calls that returned `REDISMODULE_ERR`, reported once per batch
    /// rather than per entry so a broken stream cannot flood the log.
    failed: usize,
}

impl StreamTemplate {
    /// Borrows the shared field-name strings. Returns `None` before
    /// [`start_flusher_thread`] has published them.
    fn new() -> Option<Self> {
        let names = FIELD_NAME_STRINGS.lock();
        let names = names.as_ref()?;
        let mut argv = [ptr::null_mut(); FIELD_COUNT * 2];
        for (i, name) in names.0.iter().enumerate() {
            argv[i * 2] = *name;
        }
        Some(Self { argv, failed: 0 })
    }

    /// Format and append one entry to `key`. The GIL must be held.
    ///
    /// Formatting happens here rather than in a pre-pass outside the GIL, so that a
    /// queued entry has exactly one representation all the way to the stream. It is
    /// seven `format!` calls per entry, beside the ten `RM_CreateString`s that were
    /// inside the critical section either way. Measured: `redis-benchmark -n 20000
    /// -c 1 GRAPH.QUERY g "RETURN 1"` on an M3 Pro gives a median 28,694 ops/s with
    /// the pre-pass and 28,450 without it (spread within each ≈4%), p50/p95/p99
    /// identical at 0.031/0.039/0.047 ms — so the move is not measurable, while the
    /// feature as a whole costs ~3% against the 29,300 ops/s of `CMD_INFO no`.
    fn add(
        &mut self,
        key: *mut raw::RedisModuleKey,
        pe: &PendingEntry,
    ) {
        let entry = &pe.entry;
        let received = entry.received_at.to_string();
        let wait = format!("{:.6}", entry.wait_duration_ms);
        let exec = format!("{:.6}", entry.execution_duration_ms);
        let report = format!("{:.6}", entry.report_duration_ms);
        // Add half-ULP (5e-7) so that after formatting to 6 decimal places,
        // float(total_str) >= float(exec_str) + float(report_str) always
        // holds despite floating-point addition rounding in consumers.
        let total_raw = exec.parse::<f64>().unwrap_or(0.0)
            + report.parse::<f64>().unwrap_or(0.0)
            + entry.wait_duration_ms
            + 5e-7;
        let total = format!("{total_raw:.6}");
        let flag = |b: bool| if b { "1" } else { "0" };
        // Same order as `FIELD_NAMES`.
        let values: [&str; FIELD_COUNT] = [
            &received,
            &entry.query,
            &entry.params,
            &total,
            &wait,
            &exec,
            &report,
            flag(entry.utilized_cache),
            flag(entry.is_write),
            flag(entry.timed_out),
        ];
        for (i, value) in values.iter().enumerate() {
            self.argv[i * 2 + 1] = create_detached_string(value);
        }
        let status = unsafe {
            let f = raw::RedisModule_StreamAdd.expect("RedisModule_StreamAdd");
            f(
                key,
                raw::REDISMODULE_STREAM_ADD_AUTOID as c_int,
                ptr::null_mut(),
                self.argv.as_mut_ptr(),
                FIELD_COUNT as i64,
            )
        };
        if status != raw::REDISMODULE_OK as c_int {
            self.failed += 1;
        }
        // Values are per-entry; the names are shared and outlive us. Freeing here
        // keeps the peak at one entry's worth of strings however long the batch is.
        for i in 0..FIELD_COUNT {
            let slot = &mut self.argv[i * 2 + 1];
            free_detached_string(*slot);
            *slot = ptr::null_mut();
        }
    }

    /// Report the batch's write failures, if any, and reset the count.
    fn report_failures(
        &mut self,
        graph_name: &str,
    ) {
        if self.failed > 0 {
            log_warning(format!(
                "telemetry: {} entr{} for graph '{graph_name}' could not be appended to its stream",
                self.failed,
                if self.failed == 1 { "y" } else { "ies" },
            ));
            self.failed = 0;
        }
    }
}

/// A `RedisModuleString` owned by us rather than by a command's auto-memory.
fn create_detached_string(s: &str) -> *mut raw::RedisModuleString {
    unsafe {
        let f = raw::RedisModule_CreateString.expect("RedisModule_CreateString");
        f(ptr::null_mut(), s.as_ptr().cast::<c_char>(), s.len())
    }
}

fn free_detached_string(s: *mut raw::RedisModuleString) {
    unsafe {
        let f = raw::RedisModule_FreeString.expect("RedisModule_FreeString");
        f(ptr::null_mut(), s);
    }
}

/// Append every entry of one graph to its telemetry stream, then cap it.
///
/// One key open per graph per batch, mirroring the C cron task: it opens the
/// stream once, adds the whole circular buffer, trims once, closes.
///
/// Unlike the `RM_Call(..., replicate)` this replaces, a key-API write is **not
/// propagated to replicas** — which is also true of the C engine, whose cron
/// task writes the same way. Telemetry is therefore local to the instance that
/// wrote it; [`enqueue_entry`] still declines to write any on a replica, so a
/// replica's stream is whatever its last full resync brought over.
fn stream_entries(
    ctx: *mut raw::RedisModuleCtx,
    template: &mut StreamTemplate,
    graph_name: &str,
    entries: &[PendingEntry],
    max_len: i64,
) {
    // Owned rather than borrowed from a command's auto-memory, and freed by its own
    // `Drop` — `RedisString` is the same NULL-context, binary-safe string
    // `create_detached_string` builds, minus the frees this function would otherwise
    // have to place on each of its three exits.
    let key_name =
        RedisString::create_from_slice(ptr::null_mut(), stream_name(graph_name).as_bytes());
    let key = unsafe {
        let f = raw::RedisModule_OpenKey.expect("RedisModule_OpenKey");
        f(ctx, key_name.inner, raw::REDISMODULE_WRITE as c_int)
    };
    if key.is_null() {
        return;
    }
    // A key of some other type is not ours to write to. The C engine makes the
    // same check and leaves such a key alone.
    let key_type = unsafe {
        let f = raw::RedisModule_KeyType.expect("RedisModule_KeyType");
        f(key)
    };
    let writable = key_type == raw::REDISMODULE_KEYTYPE_STREAM as c_int
        || key_type == raw::REDISMODULE_KEYTYPE_EMPTY as c_int;
    if writable {
        for entry in entries {
            template.add(key, entry);
        }
        template.report_failures(graph_name);
        // Unconditionally, `max_len` of 0 included, which is what C does:
        // `CronTask_streamFinishedQueries` passes `Config_CMD_INFO_MAX_QUERY_COUNT`
        // straight to `RedisModule_StreamTrimByLength`. Skipping the call for 0 turns
        // the one value that means "keep nothing" into "keep everything", and 0 is
        // accepted by both `GRAPH.CONFIG SET MAX_INFO_QUERIES` and the module
        // argument — so the stream would then grow for the lifetime of the server
        // with no way to bound it.
        //
        // Returns the number of entries deleted, or -1 with `errno` set.
        let deleted = unsafe {
            let f = raw::RedisModule_StreamTrimByLength.expect("RedisModule_StreamTrimByLength");
            f(key, raw::REDISMODULE_STREAM_TRIM_APPROX as c_int, max_len)
        };
        // A failed trim is not a lost entry, but it does mean the stream is
        // growing past MAX_INFO_QUERIES unbounded, which is worth saying out
        // loud rather than discovering as memory growth.
        if deleted < 0 {
            log_warning(format!(
                "telemetry: failed to trim the stream of graph '{graph_name}' to {max_len} entries"
            ));
        }
    }
    unsafe {
        let f = raw::RedisModule_CloseKey.expect("RedisModule_CloseKey");
        f(key);
    }
}

/// Delete the telemetry stream for a graph.
///
/// `graph_name` may be either the graph's name or the key it lives at; [`stream_name`]
/// truncates, so both name the same stream. The key is built from ptr+len rather than
/// through `create_string`, whose `CString::new(..).unwrap()` is what made a stray NUL
/// abort the process (#2490) — this path should not be one caller's mistake away from
/// that again.
pub fn delete_stream(
    ctx: &Context,
    graph_name: &str,
) {
    let key = stream_name(graph_name);
    let args = [RedisString::create_from_slice(
        ctx.get_raw(),
        key.as_bytes(),
    )];
    let _ = ctx.call("DEL", args.iter().collect::<Vec<_>>().as_slice());
}

// ---------------------------------------------------------------------------
// Running / Waiting query registry
// ---------------------------------------------------------------------------

static NEXT_QUERY_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    NEXT_QUERY_ID.fetch_add(1, Ordering::Relaxed)
}

/// Current UNIX timestamp in seconds.
pub fn unix_now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

#[derive(Clone)]
pub struct RunningQueryInfo {
    pub id: u64,
    pub received_at: i64,
    pub graph_name: Arc<str>,
    pub query: Arc<str>,
    pub start: Instant,
    pub is_replicated: bool,
}

#[derive(Clone)]
pub struct WaitingQueryInfo {
    pub id: u64,
    pub received_at: i64,
    pub graph_name: Arc<str>,
    pub query: Arc<str>,
    pub enqueued: Instant,
}

struct QueryRegistry {
    running: Vec<RunningQueryInfo>,
    waiting: Vec<WaitingQueryInfo>,
}

/// Number of registry shards. Each query is routed to a shard by its id, so
/// concurrent worker threads register/unregister on different mutexes instead
/// of contending on one global lock. Must be a power of two for the mask.
const REGISTRY_SHARDS: usize = 8;

static REGISTRY: [Mutex<QueryRegistry>; REGISTRY_SHARDS] = [const {
    Mutex::new(QueryRegistry {
        running: Vec::new(),
        waiting: Vec::new(),
    })
}; REGISTRY_SHARDS];

/// Route a query id to its registry shard.
#[inline]
fn shard_for(id: u64) -> &'static Mutex<QueryRegistry> {
    &REGISTRY[(id as usize) & (REGISTRY_SHARDS - 1)]
}

/// Register a query as currently running. Returns its unique ID.
pub fn register_running(
    received_at: i64,
    graph_name: &Arc<str>,
    query: &Arc<str>,
    is_replicated: bool,
) -> u64 {
    let id = next_id();
    let info = RunningQueryInfo {
        id,
        received_at,
        graph_name: Arc::clone(graph_name),
        query: truncate_arc(query),
        start: Instant::now(),
        is_replicated,
    };
    shard_for(id).lock().running.push(info);
    id
}

/// Unregister a running query.
pub fn unregister_running(id: u64) {
    let mut reg = shard_for(id).lock();
    if let Some(pos) = reg.running.iter().position(|q| q.id == id) {
        reg.running.swap_remove(pos);
    }
}

/// Unregister a waiting query.
pub fn unregister_waiting(id: u64) {
    let mut reg = shard_for(id).lock();
    if let Some(pos) = reg.waiting.iter().position(|q| q.id == id) {
        reg.waiting.swap_remove(pos);
    }
}

/// Register a query as **waiting**: accepted but not yet executing, because it
/// sits either in the thread-pool queue (before a worker picks it up) or in a
/// graph's write queue (before the write loop drains it). Mirrors C, where
/// `GRAPH.INFO WaitingQueries` reports the thread pool's queued tasks.
///
/// Returns its unique ID, to be passed to [`transition_waiting_to_running`] once
/// the query starts, or to [`unregister_waiting`] if it never does.
pub fn register_waiting(
    received_at: i64,
    graph_name: &Arc<str>,
    query: &Arc<str>,
) -> u64 {
    let id = next_id();
    let info = WaitingQueryInfo {
        id,
        received_at,
        graph_name: Arc::clone(graph_name),
        query: truncate_arc(query),
        enqueued: Instant::now(),
    };
    shard_for(id).lock().waiting.push(info);
    id
}

/// Owns a waiting-registry entry and removes it on drop unless
/// [`Self::promote`] consumes it.
///
/// The entry is created on the dispatching thread but consumed by a worker, and
/// two paths never reach the worker: `threadpool::spawn` deliberately drops the
/// job when the channel has disconnected during shutdown, and a worker can
/// panic before promoting. Either would otherwise leave the query in
/// `GRAPH.INFO WaitingQueries` forever. Moving this guard into the job closure
/// ties the entry's lifetime to the closure instead.
pub struct WaitingEntry(Option<u64>);

impl WaitingEntry {
    /// Registers a waiting query and returns the guard owning its entry.
    pub fn register(
        received_at: i64,
        graph_name: &Arc<str>,
        query: &Arc<str>,
    ) -> Self {
        Self(Some(register_waiting(received_at, graph_name, query)))
    }

    /// Moves the query from waiting to running, disarming the guard. Returns
    /// the running id, or `None` if the entry was already taken.
    pub fn promote(&mut self) -> Option<u64> {
        self.0.take().and_then(transition_waiting_to_running)
    }
}

impl Drop for WaitingEntry {
    fn drop(&mut self) {
        if let Some(id) = self.0.take() {
            unregister_waiting(id);
        }
    }
}

/// Transition a waiting query to running, keeping its id — and therefore its
/// shard, so the move costs a single lock acquisition instead of two.
///
/// Returns the id to pass to [`unregister_running`], or `None` if the query was
/// never registered as waiting.
pub fn transition_waiting_to_running(waiting_id: u64) -> Option<u64> {
    let mut reg = shard_for(waiting_id).lock();
    let pos = reg.waiting.iter().position(|q| q.id == waiting_id)?;
    let waiting = reg.waiting.swap_remove(pos);
    reg.running.push(RunningQueryInfo {
        id: waiting.id,
        received_at: waiting.received_at,
        graph_name: waiting.graph_name,
        query: waiting.query,
        start: Instant::now(),
        is_replicated: false,
    });
    Some(waiting.id)
}

/// Snapshot of all currently running queries.
pub fn snapshot_running() -> Vec<RunningQueryInfo> {
    let mut out = Vec::new();
    for shard in &REGISTRY {
        out.extend(shard.lock().running.iter().cloned());
    }
    out
}

/// Snapshot of all currently waiting queries.
pub fn snapshot_waiting() -> Vec<WaitingQueryInfo> {
    let mut out = Vec::new();
    for shard in &REGISTRY {
        out.extend(shard.lock().waiting.iter().cloned());
    }
    out
}

/// Build a `RedisValue` for the GRAPH.INFO `RunningQueries` section.
pub fn running_queries_reply() -> Vec<RedisValue> {
    let now = Instant::now();
    let queries = snapshot_running();
    queries
        .into_iter()
        .map(|q| {
            let duration_ms = now.duration_since(q.start).as_secs_f64() * 1000.0;
            RedisValue::Array(vec![
                RedisValue::BulkString("Received at".into()),
                RedisValue::Integer(q.received_at),
                RedisValue::BulkString("Graph name".into()),
                RedisValue::BulkString(q.graph_name.to_string()),
                RedisValue::BulkString("Query".into()),
                RedisValue::BulkString(q.query.to_string()),
                RedisValue::BulkString("Execution duration".into()),
                RedisValue::Float(duration_ms),
                RedisValue::BulkString("Replicated command".into()),
                RedisValue::Integer(i64::from(q.is_replicated)),
            ])
        })
        .collect()
}

/// Build a `RedisValue` for the GRAPH.INFO `WaitingQueries` section.
pub fn waiting_queries_reply() -> Vec<RedisValue> {
    let now = Instant::now();
    let queries = snapshot_waiting();
    queries
        .into_iter()
        .map(|q| {
            let duration_ms = now.duration_since(q.enqueued).as_secs_f64() * 1000.0;
            RedisValue::Array(vec![
                RedisValue::BulkString("Received at".into()),
                RedisValue::Integer(q.received_at),
                RedisValue::BulkString("Graph name".into()),
                RedisValue::BulkString(q.graph_name.to_string()),
                RedisValue::BulkString("Query".into()),
                RedisValue::BulkString(q.query.to_string()),
                RedisValue::BulkString("Wait duration".into()),
                RedisValue::Float(duration_ms),
            ])
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Batched telemetry flusher
// ---------------------------------------------------------------------------
//
// Worker threads enqueue telemetry entries onto a lock-free MPMC channel
// instead of performing an XADD under the Redis module lock. A single
// background thread drains the channel and issues all pending XADDs under one
// GIL acquisition per batch. This amortizes the cost of acquiring the global
// module lock across many entries and removes lock contention from the read
// query hot path. Mirrors the FalkorDB C implementation.

/// Maximum entries drained per flush iteration. Caps the time the GIL is held.
const FLUSH_BATCH_MAX: usize = 256;

/// How long the flusher waits for the first entry before parking again.
const FLUSH_INTERVAL: Duration = Duration::from_millis(5);

/// How long the flusher keeps collecting after the *first* entry of a batch
/// arrives, before writing what it has.
///
/// Without it the batching amortizes nothing under the load it was built for:
/// `recv_timeout` returns as soon as one entry is queued, and a serial client
/// produces exactly one entry per query, so every query got its own wakeup, its
/// own GIL acquisition and its own single-entry batch. It costs up to this much
/// delay before an entry is visible in the stream — invisible next to the
/// consumer-side polling in `GRAPH.INFO`'s own tests (10ms interval, 30s budget),
/// and next to the C engine, whose cron task streams finished queries on its own
/// schedule rather than per query.
const FLUSH_LINGER: Duration = Duration::from_millis(5);

/// How long `shutdown_flusher_thread` waits for the flusher to stop before giving up on
/// it and letting shutdown continue without it. See there for why it is not an
/// unconditional join. Far longer than an iteration of the loop takes, so the normal
/// path returns the moment the flusher signals rather than after any part of this.
const FLUSHER_STOP_GRACE: Duration = Duration::from_millis(500);

/// Cap on entries held back while replica traffic is paused (see `flusher_loop`).
/// Beyond this the oldest are dropped: the telemetry stream is already lossy —
/// `MAX_INFO_QUERIES` trims it on every write — so bounding memory matters more
/// than keeping every entry across a long failover pause.
const DEFERRED_XADD_MAX: usize = 4 * FLUSH_BATCH_MAX;

struct PendingEntry {
    /// The Redis key this query addressed — replaced by the key the graph is
    /// registered under *now* just before the write, see `resolve_current_names`.
    graph_name: Arc<str>,
    /// The graph this query ran on, weakly.
    ///
    /// The graph and not the name is what an entry belongs to: a name can be
    /// flushed and rebound to a *different* graph while an entry is still queued,
    /// and writing it then attributes one graph's query to another's stream — and
    /// resurrects a stream key that `FLUSHALL` removed. The C engine cannot get this
    /// wrong because its queries log is owned by the `GraphContext`, so flushing the
    /// graph frees the pending entries with it; this handle reproduces that
    /// ownership, and is also what a `RENAME`d entry is re-keyed by.
    graph: Weak<RwLock<ThreadedGraph>>,
    entry: TelemetryEntry,
}

/// One distinct graph of a flush batch.
struct BatchGraph {
    /// A strong reference, not just an address: an address only identifies a graph for
    /// as long as the allocation behind it lives, and holding one is what stops a graph
    /// being freed and its address reused by another between resolution and the write.
    /// `Arc::as_ptr` of this is also what the batch's entries are matched against.
    graph: Arc<RwLock<ThreadedGraph>>,
    /// The key this batch's entries addressed. Tried against the registry first: it is
    /// still the right answer for every graph whose key has not moved.
    captured: Arc<str>,
    /// The key the graph is registered under *now*, once resolved.
    current: Option<Arc<str>>,
}

/// Re-key every entry to the Redis key its graph is registered under *now*, dropping
/// entries whose graph is no longer registered at all.
///
/// Entries carry the key the query addressed, and a linger window plus a replica pause
/// can pass before they are written — long enough for a `RENAME` to move the graph and
/// delete the old key's stream with it. C follows the graph: its cron task takes the
/// stream name from the `GraphContext`, and `GraphContext_Rename` deletes the old
/// stream and rebuilds that name, so a renamed graph's in-flight entries land in its
/// new stream. Matching a stale name against the registry instead just discards them,
/// which loses every entry in flight across a blue/green key swap with no diagnostic.
///
/// One registry lock for the whole batch rather than one per entry: at flush rates the
/// per-entry version was tens of thousands of acquisitions a second on a mutex that
/// `register_graph`, `rename_graph`, `graph_free` and the pre-fork sync all need.
///
/// Called with the GIL held, so that the name this settles on and the keyspace
/// [`key_holds_graph`] then checks it against cannot disagree: a `RENAME` runs on the
/// main thread, which cannot run a command callback while the flusher holds the GIL.
/// Resolving outside it left a window where a rename between resolution and the write
/// made the confirmation fail, and the entry was dropped — the very case this exists
/// to carry across a rename.
fn resolve_current_names(
    deferred: &mut Vec<PendingEntry>,
    graphs: &mut [BatchGraph],
) {
    if !graphs.is_empty() {
        let registry = GRAPH_REGISTRY.lock();
        // One hash lookup per graph settles the overwhelming majority: the key an entry
        // addressed is still the key its graph answers to unless a `RENAME` moved it, or
        // the name was rebound to a different graph, in the milliseconds since.
        let mut unresolved = 0;
        for b in &mut *graphs {
            match registry.get(&*b.captured) {
                Some(arc) if arc.data_ptr() == b.graph.data_ptr() => {
                    b.current = Some(Arc::clone(&b.captured));
                }
                _ => unresolved += 1,
            }
        }
        // Only the graphs that missed need the scan, and it stops as soon as the last of
        // them is placed. Scanning unconditionally is O(R·G) in the number of graphs in
        // the keyspace, with the GIL held, every 5-10ms: on a multi-tenant keyspace of
        // tens of thousands of graphs that is all command processing stalled behind
        // telemetry bookkeeping.
        if unresolved > 0 {
            for (name, arc) in registry.iter() {
                let Some(b) = graphs
                    .iter_mut()
                    .find(|b| b.current.is_none() && b.graph.data_ptr() == arc.data_ptr())
                else {
                    continue;
                };
                b.current = Some(Arc::from(name.as_str()));
                unresolved -= 1;
                if unresolved == 0 {
                    break;
                }
            }
        }
    }
    deferred.retain_mut(|pe| {
        // Matched by address rather than by upgrading: `graphs` holds a strong reference
        // to every graph in it, so an address found there is alive by construction, and
        // a `Weak` keeps its own allocation alive for as long as it exists — so no dead
        // entry can borrow a live graph's address and be filed under its name. Upgrading
        // here instead put one contended atomic per entry, a thousand of them a flush,
        // on the single counter every query thread is already touching.
        let Some(name) = graphs
            .iter()
            .find(|b| Arc::as_ptr(&b.graph) == pe.graph.as_ptr())
            .and_then(|b| b.current.as_ref())
        else {
            // Either the graph is gone, or it is alive and registered under no key —
            // deleted, or flushed and replaced. Neither has a stream to be written to.
            return false;
        };
        if pe.graph_name != *name {
            pe.graph_name = Arc::clone(name);
        }
        true
    });
}

/// The distinct graphs of `deferred`, each held by a strong reference.
///
/// Built *before* the GIL is taken and kept until after it is released, so every graph
/// the flush touches is already held here: nothing done under the GIL can be the last
/// release of one, and no `ThreadedGraph` destructor — matrices, RediSearch indexes —
/// can run on this thread while it holds the GIL.
///
/// Entries are grouped by `Weak::as_ptr`, a pointer read, and only one upgrade is done
/// per distinct graph. A `Weak` keeps its own allocation alive, so two live `Weak`s
/// address the same allocation only if they are the same graph, which makes the pointer
/// a sound identity here. The obvious version — upgrade every entry and compare — is
/// one contended atomic increment per entry on a single cache line, a thousand of them
/// per flush, against the same counter the query threads are hammering; it cost about
/// half the flusher's throughput under a 16-client load.
fn hold_batch_graphs(deferred: &[PendingEntry]) -> Vec<BatchGraph> {
    let mut graphs: Vec<BatchGraph> = Vec::new();
    // Addresses already considered, including any whose upgrade failed, so a batch of
    // one dead graph's entries does not retry the upgrade a thousand times.
    let mut seen: Vec<*const RwLock<ThreadedGraph>> = Vec::new();
    for pe in deferred {
        let addr = pe.graph.as_ptr();
        if seen.contains(&addr) {
            continue;
        }
        seen.push(addr);
        // A batch is one or two distinct graphs in practice, so the linear scans here
        // and at the match sites beat hashing.
        if let Some(graph) = pe.graph.upgrade() {
            graphs.push(BatchGraph {
                graph,
                captured: Arc::clone(&pe.graph_name),
                current: None,
            });
        }
    }
    graphs
}

/// Release what [`hold_batch_graphs`] took, off this thread if it turns out to be the
/// last holder.
///
/// `register_graph`, `rename_graph` and `graph_free` all hand teardown to a background
/// thread rather than run it inline, because `Index::drop` reaches RediSearch and takes
/// the GIL. The flusher is a latency-critical thread for the same reason they are: the
/// next batch waits behind whatever runs here.
fn release_batch_graphs(graphs: Vec<BatchGraph>) {
    for b in graphs {
        let arc = b.graph;
        if Arc::strong_count(&arc) == 1 {
            thread::spawn(move || drop(arc));
        }
    }
}

/// True if the Redis key `name` currently holds `graph`. The GIL must be held.
///
/// The registry is the module's view of the keyspace; this is Redis's own, and the two
/// can disagree. `graph_free` is the module type's free callback, so under lazy free
/// (`lazyfree-lazy-user-flush yes`, `UNLINK`, an async `FLUSHALL`) it runs on the
/// lazyfree thread — arbitrarily later than the key left the keyspace. Writing on the
/// registry's word alone would then recreate the stream key of a graph Redis has
/// already dropped, and because a key-API write is not propagated, that key exists on
/// the master and nowhere else: the master/replica keyspace mismatch this whole check
/// chain is here to prevent. One key open per graph per batch.
fn key_holds_graph(
    ctx: &Context,
    name: &str,
    graph: &Arc<RwLock<ThreadedGraph>>,
) -> bool {
    // NULL context, like every other string this module builds off the command path,
    // so Redis does not tie it to an auto-memory pool that outlives the call.
    let key_name = RedisString::create_from_slice(ptr::null_mut(), name.as_bytes());
    // NOTOUCH: reporting on a query must not make the graph look more recently used
    // than the query already made it.
    let key = ctx.open_key_with_flags(&key_name, KeyFlags::NOTOUCH);
    matches!(
        key.get_value::<Arc<RwLock<ThreadedGraph>>>(&GRAPH_TYPE),
        Ok(Some(registered)) if registered.data_ptr() == graph.data_ptr()
    )
}

/// Move everything already queued into `batch`, up to [`FLUSH_BATCH_MAX`].
///
/// `true` if the channel has disconnected. `false` does *not* mean it has not: the loop
/// also falls out with a full batch, having asked nothing, and `try_recv` reports
/// `Disconnected` only once the channel is *empty* as well as closed. Callers that are
/// about to block on something the shutdown path holds must re-check with
/// [`MRx::is_disconnected`], which answers the question the sender's fate alone decides.
fn drain_queued(
    rx: &MRx<List<PendingEntry>>,
    batch: &mut Vec<PendingEntry>,
) -> bool {
    while batch.len() < FLUSH_BATCH_MAX {
        match rx.try_recv() {
            Ok(pe) => {
                QUEUED.fetch_sub(1, Ordering::Relaxed);
                batch.push(pe);
            }
            Err(crossfire::TryRecvError::Empty) => return false,
            Err(crossfire::TryRecvError::Disconnected) => return true,
        }
    }
    false
}

/// Producer side of the telemetry channel. `None` before
/// `start_flusher_thread` and after `shutdown_flusher_thread`; entries
/// enqueued outside that window are silently dropped. Wrapped in a `Mutex`
/// so shutdown can drop the sender, closing the channel and letting the
/// flusher loop observe `Disconnected` and exit cleanly.
static SENDER: Mutex<Option<MTx<List<PendingEntry>>>> = Mutex::new(None);
/// Receiver, parked here only between `start_flusher_thread` being called and
/// the flusher thread taking ownership.
static RECEIVER: Mutex<Option<MRx<List<PendingEntry>>>> = Mutex::new(None);
/// Handle for the flusher thread, joined during `shutdown_flusher_thread`.
static FLUSHER: Mutex<Option<JoinHandle<()>>> = Mutex::new(None);
/// Signalled when the flusher thread has left its loop, so shutdown can wait for it
/// with a deadline instead of joining it unconditionally. The sender lives in the
/// spawned closure, so it also drops — waking the receiver — if the thread panics.
static FLUSHER_EXIT: Mutex<Option<mpsc::Receiver<()>>> = Mutex::new(None);

/// Entries sent but not yet taken off the channel.
///
/// The channel is unbounded and has to be: `enqueue_entry` runs on the query hot
/// path, where blocking a worker until the flusher catches up would charge
/// telemetry the very latency it exists to measure. Unbounded *and* uncapped is the
/// other failure — the flusher writes at most [`FLUSH_BATCH_MAX`] entries per GIL
/// acquisition, so a burst that outruns it grows the queue for as long as it lasts:
/// 400k queries at 140k ops/s left the flusher 8.2s and ~50MB behind. C cannot grow
/// here at all, because its per-graph queries log is a fixed-size circular buffer
/// that overwrites its oldest entry; dropping arrivals past [`QUEUE_MAX`] is the
/// same trade, bounded and lossy in the same way.
static QUEUED: AtomicUsize = AtomicUsize::new(0);

/// Entries dropped because the channel was at [`QUEUE_MAX`], reported by the flusher.
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Cap on entries waiting on the channel — four batches, the same bound
/// [`DEFERRED_XADD_MAX`] puts on entries held back through a replica pause.
const QUEUE_MAX: usize = 4 * FLUSH_BATCH_MAX;

/// Push a telemetry entry onto the background channel.
///
/// Off the GIL, but not lock-free: the sender lives behind a `Mutex` so that
/// [`shutdown_flusher_thread`] can drop it and disconnect the channel. The critical
/// section is a pointer read and a queue push, so workers serialise for the length of
/// the push and nothing else — the entry's real cost (formatting, string creation,
/// the stream write) is all on the flusher.
pub fn enqueue_entry(
    graph_name: &Arc<str>,
    graph: &Arc<RwLock<ThreadedGraph>>,
    entry: TelemetryEntry,
) {
    // `CMD_INFO no` means "do not log finished queries", which is exactly this
    // path, and it is the only way to opt out of what logging one costs. C gates
    // the equivalent cron task on the same config, so entries produced while off
    // are dropped rather than buffered, and turning it back on resumes streaming.
    if !CONFIGURATION_CMD_INFO.load(Ordering::Relaxed) {
        return;
    }
    // Skip on replicas: a replica must not create keys of its own, and the stream
    // it has is the master's, brought over by the last full resync.
    //
    // Note this is no longer about duplicates. The writes were replicated when they
    // went through `RM_Call("XADD", ..., replicate)`; they no longer are, so having
    // each instance log the queries it actually served — which is what the C engine
    // does — is now merely a behaviour change rather than a source of double
    // entries. Making it is deliberately out of scope here.
    if IS_REPLICA.load(Ordering::Relaxed) {
        return;
    }
    // Bounded queue: see `QUEUED`. Checked before the send and with a plain load, so a
    // burst can overshoot the cap by however many workers race here at once — which is
    // fine, the cap is a memory bound and not a quota.
    if QUEUED.load(Ordering::Relaxed) >= QUEUE_MAX {
        DROPPED.fetch_add(1, Ordering::Relaxed);
        return;
    }
    if let Some(tx) = SENDER.lock().as_ref() {
        // Counted before the send so the flusher's decrement can never run first and
        // wrap the counter through zero.
        QUEUED.fetch_add(1, Ordering::Relaxed);
        if tx
            .send(PendingEntry {
                graph_name: Arc::clone(graph_name),
                graph: Arc::downgrade(graph),
                entry,
            })
            .is_err()
        {
            QUEUED.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

/// Tracks whether this Redis instance is currently a replica. Updated on
/// module load and on `RedisModuleEvent_ReplicationRoleChanged` notifications.
static IS_REPLICA: AtomicBool = AtomicBool::new(false);

/// Update the cached replica state. Called from module init and the role
/// change event handler.
pub fn set_is_replica(is_replica: bool) {
    IS_REPLICA.store(is_replica, Ordering::Relaxed);
}

/// Spawn the background flusher thread. Must be called once at module init.
pub fn start_flusher_thread() {
    let (tx, rx) = mpmc::unbounded_blocking::<PendingEntry>();
    {
        let mut sender = SENDER.lock();
        if sender.is_some() {
            // Already initialized.
            return;
        }
        *sender = Some(tx);
    }
    *RECEIVER.lock() = Some(rx);

    // Here rather than on the flusher thread: this runs on the main thread during
    // module init, where calling into the module API is unambiguously safe, and the
    // ten strings are then shared by every write for the module's whole lifetime
    // instead of being rebuilt per flusher thread.
    {
        let mut names = FIELD_NAME_STRINGS.lock();
        if names.is_none() {
            *names = Some(FieldNameStrings(FIELD_NAMES.map(create_detached_string)));
        }
    }

    let (exit_tx, exit_rx) = mpsc::channel();
    *FLUSHER_EXIT.lock() = Some(exit_rx);
    let handle = thread::Builder::new()
        .name("falkordb-telemetry".to_string())
        .spawn(move || {
            flusher_loop();
            // Sent, and in any case dropped, as the thread ends: either way the wait in
            // `shutdown_flusher_thread` returns.
            let _ = exit_tx.send(());
        })
        .expect("failed to spawn telemetry flusher thread");
    *FLUSHER.lock() = Some(handle);
}

/// Stop the background flusher: drop the sender so the channel disconnects, then wait
/// for the thread. Must be called on module unload before tearing down Redis state the
/// flusher's writes touch.
///
/// The wait is bounded, and that is the point. This runs on the main thread inside
/// Redis's shutdown event callback, which holds the module GIL for its whole duration —
/// so a flusher parked on `hold_gil()` can never wake, and an unconditional join hangs
/// the server. The flusher checks for the disconnect immediately before it acquires the
/// GIL, which closes that door in every ordering but one: the sender dropping in the
/// instant between its check and its acquire. Rather than pay for that window on the
/// hot path — a polling try-lock keeps missing the brief windows in which Redis
/// releases the GIL, which cost the flusher about half its throughput in measurement —
/// the window is made harmless here. A flusher still parked when the grace expires is
/// left parked: it is blocked acquiring a GIL that this thread holds and will hold
/// until the process exits, so it cannot touch Redis state, which is all the join was
/// ever protecting.
pub fn shutdown_flusher_thread() {
    // Drop the sender to close the channel; the flusher observes the disconnect and
    // leaves *without* writing, dropping whatever is still queued.
    drop(SENDER.lock().take());
    let handle = FLUSHER.lock().take();
    let exit = FLUSHER_EXIT.lock().take();
    // `Err(Timeout)` is the only outcome that means "still running": a send and a
    // dropped sender (the thread panicked) both say it is done.
    let stopped = exit.is_none_or(|rx| {
        !matches!(
            rx.recv_timeout(FLUSHER_STOP_GRACE),
            Err(mpsc::RecvTimeoutError::Timeout)
        )
    });
    if !stopped {
        log_warning(
            "telemetry: the flush thread did not stop within the shutdown grace period \
             and is being left parked; it holds no Redis state",
        );
        // Dropped rather than joined, which detaches it.
        drop(handle);
        // Leaked deliberately: the parked thread's stream template still borrows them.
        return;
    }
    if let Some(h) = handle {
        let _ = h.join();
    }
    // Only after the join: the flusher's template borrows these.
    if let Some(names) = FIELD_NAME_STRINGS.lock().take() {
        for name in names.0 {
            free_detached_string(name);
        }
    }
}

fn flusher_loop() {
    let rx = RECEIVER
        .lock()
        .take()
        .expect("flusher started without a receiver");

    // Detached thread-safe context: no associated client, used purely to
    // hold the module lock while issuing XADDs.
    let tsc = unsafe {
        let f = raw::RedisModule_GetThreadSafeContext.expect("RedisModule_GetThreadSafeContext");
        f(ptr::null_mut())
    };

    let mut batch: Vec<PendingEntry> = Vec::with_capacity(FLUSH_BATCH_MAX);
    let mut template =
        StreamTemplate::new().expect("flusher started before the field names were published");
    let mut disconnected = false;
    // Entries not yet written because replica traffic was paused when their turn
    // came. Retried on a later iteration — see the pause check below.
    let mut deferred: Vec<PendingEntry> = Vec::new();
    // Drops already reported, so each warning covers only what is new.
    let mut reported_drops = 0;

    loop {
        // Block until at least one entry is available (or the channel closes).
        match rx.recv_timeout(FLUSH_INTERVAL) {
            Ok(first) => {
                QUEUED.fetch_sub(1, Ordering::Relaxed);
                batch.push(first);
            }
            // Nothing new — but entries held back by a pause still need a retry, so
            // only park again once there is genuinely nothing to do.
            Err(crossfire::RecvTimeoutError::Timeout) => {
                if deferred.is_empty() {
                    continue;
                }
            }
            Err(crossfire::RecvTimeoutError::Disconnected) => break,
        }
        // Skipped when nothing arrived: this iteration is then only a retry of a
        // held batch, and there is no arrival to collect neighbours for.
        if !batch.is_empty() {
            // Take what is already queued *before* deciding to wait. A full batch is
            // the signal that the flusher is behind: lingering then buys no batching
            // it does not already have, and caps throughput at FLUSH_BATCH_MAX per
            // FLUSH_LINGER — ~51k entries/s in theory, 36k measured — while the
            // channel keeps accepting at whatever rate the clients manage. That is
            // how 400k queries at 140k ops/s left the flusher 8.2s behind.
            disconnected |= drain_queued(&rx, &mut batch);
            // Let the window elapse, then take what arrived during it.
            //
            // A *blocking* wait here would return the moment an entry is queued, so
            // the thread would still wake once per query and only the flush would be
            // batched — and waking is most of what an entry costs. Asleep, a sender's
            // push is an atomic with nobody to wake, which is the position the C
            // engine is in: its cron task runs on a timer and the query thread only
            // appends to a per-graph buffer.
            if batch.len() < FLUSH_BATCH_MAX && !disconnected {
                thread::sleep(FLUSH_LINGER);
                disconnected |= drain_queued(&rx, &mut batch);
            }
        }

        // Once per flush rather than once per drop: at the rate a full queue drops
        // entries, a line each would be the more expensive half of the problem.
        let drops = DROPPED.load(Ordering::Relaxed);
        if drops > reported_drops {
            log_warning(format!(
                "telemetry: dropped {} finished-query entries — the queue was at its \
                 {QUEUE_MAX}-entry cap",
                drops - reported_drops,
            ));
            reported_drops = drops;
        }

        // Asked of the channel itself rather than inferred from the drains: `try_recv`
        // reports `Disconnected` only when the channel is *empty* as well as closed, and
        // `drain_queued` stops asking altogether once the batch is full. A backlog —
        // precisely the state the linger-skip above exists to handle — would otherwise
        // hide the shutdown for as long as it lasted. `is_disconnected` is the sender's
        // fate alone, so a queue with entries still in it cannot mask it.
        disconnected |= rx.is_disconnected();

        if disconnected {
            // Leave *without* writing. `shutdown_flusher_thread` runs on the main
            // thread inside Redis's shutdown event callback, which holds the module
            // GIL for its whole duration (Redis releases it in `beforeSleep`), and
            // blocks there joining this thread. Taking the GIL here would deadlock
            // the two and hang the server on shutdown after any query burst. Entries
            // still queued are dropped, which is the same thing the `Disconnected`
            // arm above does when the channel closes while the flusher is parked.
            break;
        }

        let max_len = MAX_INFO_QUERIES.load(Ordering::Relaxed);
        deferred.append(&mut batch);
        // Oldest-first drop, so a long pause bounds memory instead of growing without
        // limit. The stream is trimmed on every write anyway, so the entries most worth
        // keeping are the newest.
        if deferred.len() > DEFERRED_XADD_MAX {
            deferred.drain(..deferred.len() - DEFERRED_XADD_MAX);
        }

        // Entries whose graph has been dropped outright have nowhere to go. Checking it
        // here costs one atomic load each and takes no lock at all, so a batch that is
        // entirely dead — a graph deleted while its last queries were still queued —
        // never reaches the point of acquiring the GIL to write nothing.
        deferred.retain(|pe| pe.graph.strong_count() > 0);
        if deferred.is_empty() {
            continue;
        }

        // Upgraded here, outside the GIL, and held until it has been released again, so
        // that nothing done under the GIL can be the last release of a graph. This is
        // also the batch's only pass of upgrades: everything below matches entries to
        // these by address. See `hold_batch_graphs`.
        let mut graphs = hold_batch_graphs(&deferred);

        // Single GIL acquisition for the whole batch, through the same guard queries
        // use, so every acquisition in the process funnels through one place.
        //
        // Blocking, and deliberately: the main thread can be waiting on this thread
        // while holding the GIL — `shutdown_flusher_thread` waits for it from inside the
        // shutdown event callback — but the disconnect check above is what keeps this
        // thread from arriving here after that has begun. The one ordering it cannot
        // cover, a sender dropped between that check and this line, is handled by
        // bounding the wait over there instead of bounding this. A try-lock here would
        // cover it, and was measured: polling misses the brief windows in which Redis
        // releases the GIL, where a blocked waiter is handed it, and the flusher fell
        // far enough behind to drop several times as many entries at the queue cap.
        {
            let _gil = crate::query_session::hold_gil();
            let ctx = Context::new(tsc);
            // Both checks are read here, under the GIL, rather than trusted from when the
            // entries were enqueued: a role change arrives as a server event on the main
            // thread and a pause is opened from it, so neither can move while we hold the
            // GIL. `enqueue_entry`'s `IS_REPLICA` check is on the producing thread and
            // says nothing about what is true now, at dispatch.
            if ctx.get_flags().contains(ContextFlags::SLAVE) {
                // We have become a replica since these were enqueued. Discard rather than
                // hold: writing here would create stream keys directly on a replica, which
                // is exactly what `enqueue_entry` refuses to do on the hot path — and they
                // would not survive the next full resync anyway.
                //
                // This interleaving is the *likely* one, not a corner case, and the pause
                // check below is what makes it so: a FAILOVER opens a pause window, the
                // batch is held for its duration, and the window closes at the moment this
                // instance has become a replica. Without this the whole held batch would
                // then be written, on a replica.
                deferred.clear();
            } else if !ctx.avoid_replication_traffic() {
                // Held for the pause window even though `stream_entries` writes through
                // the key API, which is *not* propagated the way the `RM_Call(..., replicate)`
                // it replaced was — so it can no longer trip the `propagateNow()` invariant
                // that killed the master in #2359 (the crash `query_session::reauthorize_write`
                // guards the write path against). What a replica pause still asks for is
                // that the dataset stay fixed until it lapses, and a telemetry entry is
                // dataset; holding is also what makes the demotion above a reliable
                // interleaving rather than a race, which is how `test_role_change_race`
                // pins this behaviour.
                //
                // Address each entry to the key its graph answers to *now*, and drop the
                // entries whose graph is registered under no key at all: the graph a
                // query ran on is what an entry belongs to, and its key can have moved
                // (`RENAME`) or been rebound to another graph (`FLUSHALL` then `RESTORE`)
                // in the milliseconds since.
                //
                // Under the GIL, and immediately before the write, so that nothing can
                // move between resolving a name and confirming it: the pause branch can
                // hold a batch for a whole pause window, and every path that deletes or
                // re-keys a graph runs on the main thread, which cannot execute a command
                // callback while this thread holds the GIL.
                resolve_current_names(&mut deferred, &mut graphs);

                // Group by graph so each stream key is opened and trimmed once per batch
                // rather than once per entry. Sorting is enough: entries of one graph end
                // up in one run, and being stable it keeps arrival order within a graph,
                // which is the order consumers read.
                deferred.sort_by(|a, b| a.graph_name.cmp(&b.graph_name));
                for run in deferred.chunk_by(|a, b| a.graph_name == b.graph_name) {
                    let name = &run[0].graph_name;
                    // The registry named this key; Redis has to agree that it still
                    // holds this graph. See `key_holds_graph`. One graph per run, so
                    // any entry of it answers for the rest.
                    let Some(b) = graphs
                        .iter()
                        .find(|b| Arc::as_ptr(&b.graph) == run[0].graph.as_ptr())
                    else {
                        continue;
                    };
                    if !key_holds_graph(&ctx, name, &b.graph) {
                        continue;
                    }
                    stream_entries(tsc, &mut template, name, run, max_len);
                }
                deferred.clear();
            }
        }
        release_batch_graphs(graphs);
    }

    unsafe {
        raw::RedisModule_FreeThreadSafeContext.expect("RedisModule_FreeThreadSafeContext")(tsc);
    }
}
