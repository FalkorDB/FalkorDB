//! Telemetry subsystem: stream writing and running/waiting query registry.
//!
//! After each query execution, a telemetry entry is written to a Redis stream
//! named `telemetry{graph_name}`. The GRAPH.INFO command reads the global
//! query registry to report currently running and waiting queries.

use crossfire::mpmc::{self, List};
use crossfire::{MRx, MTx};
use parking_lot::Mutex;
use redis_module::{CallOptions, CallOptionsBuilder, Context, RedisString, RedisValue, raw};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::config::MAX_INFO_QUERIES;

/// Maximum stored string length for query/params in telemetry entries.
const STR_MAX_LEN: usize = 2048;

// ---------------------------------------------------------------------------
// String truncation
// ---------------------------------------------------------------------------

/// Truncate `s` to at most `STR_MAX_LEN` characters, appending `"..."` if
/// truncated.
pub(crate) fn truncate(s: &str) -> String {
    if let Some((byte_idx, _)) = s.char_indices().nth(STR_MAX_LEN) {
        let mut t = String::with_capacity(byte_idx + 3);
        t.push_str(&s[..byte_idx]);
        t.push_str("...");
        t
    } else {
        s.to_owned()
    }
}

// ---------------------------------------------------------------------------
// Telemetry stream writing
// ---------------------------------------------------------------------------

/// Build the Redis stream key name for a graph's telemetry.
pub(crate) fn stream_name(graph_name: &str) -> String {
    format!("telemetry{{{graph_name}}}")
}

/// Data for one telemetry stream entry (10 fields).
pub(crate) struct TelemetryEntry {
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

/// Write a telemetry entry to the graph's stream via XADD.
///
/// The stream is trimmed to approximately `MAX_INFO_QUERIES` entries.
/// Must be called with the GIL held (or from the main thread).
fn write_to_stream(
    ctx: &Context,
    graph_name: &str,
    entry: &TelemetryEntry,
) {
    let stream_key = stream_name(graph_name);
    let max_len = MAX_INFO_QUERIES.load(Ordering::Relaxed).to_string();
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
    let total = format!("{:.6}", total_raw);
    let cache_flag = if entry.utilized_cache { "1" } else { "0" };
    let write_flag = if entry.is_write { "1" } else { "0" };
    let timeout_flag = if entry.timed_out { "1" } else { "0" };

    // XADD telemetry{graph} MAXLEN ~ <max> * field value ...
    let args: Vec<RedisString> = [
        &stream_key,
        "MAXLEN",
        "~",
        &max_len,
        "*",
        "Received at",
        &received,
        "Query",
        &entry.query,
        "Query parameters",
        &entry.params,
        "Total duration",
        &total,
        "Wait duration",
        &wait,
        "Execution duration",
        &exec,
        "Report duration",
        &report,
        "Utilized cache",
        cache_flag,
        "Write",
        write_flag,
        "Timeout",
        timeout_flag,
    ]
    .iter()
    .map(|s| ctx.create_string(*s))
    .collect();

    let _: redis_module::CallResult = ctx.call_ext(
        "XADD",
        &replicated_call_options(),
        args.iter().collect::<Vec<_>>().as_slice(),
    );
}

/// CallOptions used by the flusher: replicate the XADD to attached replicas
/// so the telemetry stream stays mirrored across master/replica.
fn replicated_call_options() -> CallOptions {
    CallOptionsBuilder::new().replicate().build()
}

/// Delete the telemetry stream for a graph.
pub(crate) fn delete_stream(
    ctx: &Context,
    graph_name: &str,
) {
    let key = stream_name(graph_name);
    let args = [ctx.create_string(key.as_str())];
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
pub(crate) fn unix_now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

#[derive(Clone)]
pub(crate) struct RunningQueryInfo {
    pub id: u64,
    pub received_at: i64,
    pub graph_name: String,
    pub query: String,
    pub start: Instant,
    pub is_replicated: bool,
}

#[derive(Clone)]
pub(crate) struct WaitingQueryInfo {
    pub id: u64,
    pub received_at: i64,
    pub graph_name: String,
    pub query: String,
    pub enqueued: Instant,
}

struct QueryRegistry {
    running: Vec<RunningQueryInfo>,
    waiting: Vec<WaitingQueryInfo>,
}

static REGISTRY: Mutex<QueryRegistry> = Mutex::new(QueryRegistry {
    running: Vec::new(),
    waiting: Vec::new(),
});

/// Register a query as currently running. Returns its unique ID.
pub(crate) fn register_running(
    received_at: i64,
    graph_name: &str,
    query: &str,
    is_replicated: bool,
) -> u64 {
    let id = next_id();
    let info = RunningQueryInfo {
        id,
        received_at,
        graph_name: graph_name.to_string(),
        query: truncate(query),
        start: Instant::now(),
        is_replicated,
    };
    REGISTRY.lock().running.push(info);
    id
}

/// Unregister a running query.
pub(crate) fn unregister_running(id: u64) {
    let mut reg = REGISTRY.lock();
    reg.running.retain(|q| q.id != id);
}

/// Unregister a waiting query.
pub(crate) fn unregister_waiting(id: u64) {
    let mut reg = REGISTRY.lock();
    reg.waiting.retain(|q| q.id != id);
}

/// Register a query as waiting in the write queue. Returns its unique ID.
pub(crate) fn register_waiting(
    received_at: i64,
    graph_name: &str,
    query: &str,
) -> u64 {
    let id = next_id();
    let info = WaitingQueryInfo {
        id,
        received_at,
        graph_name: graph_name.to_string(),
        query: truncate(query),
        enqueued: Instant::now(),
    };
    REGISTRY.lock().waiting.push(info);
    id
}

/// Transition a waiting query to running. Returns the waiting info if found.
pub(crate) fn transition_waiting_to_running(waiting_id: u64) -> Option<u64> {
    let mut reg = REGISTRY.lock();
    let pos = reg.waiting.iter().position(|q| q.id == waiting_id)?;
    let waiting = reg.waiting.remove(pos);
    let running_id = next_id();
    reg.running.push(RunningQueryInfo {
        id: running_id,
        received_at: waiting.received_at,
        graph_name: waiting.graph_name,
        query: waiting.query,
        start: Instant::now(),
        is_replicated: false,
    });
    Some(running_id)
}

/// Snapshot of all currently running queries.
pub(crate) fn snapshot_running() -> Vec<RunningQueryInfo> {
    REGISTRY.lock().running.clone()
}

/// Snapshot of all currently waiting queries.
pub(crate) fn snapshot_waiting() -> Vec<WaitingQueryInfo> {
    REGISTRY.lock().waiting.clone()
}

/// Build a RedisValue for the GRAPH.INFO RunningQueries section.
pub(crate) fn running_queries_reply() -> Vec<RedisValue> {
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
                RedisValue::BulkString(q.graph_name),
                RedisValue::BulkString("Query".into()),
                RedisValue::BulkString(q.query),
                RedisValue::BulkString("Execution duration".into()),
                RedisValue::Float(duration_ms),
                RedisValue::BulkString("Replicated command".into()),
                RedisValue::Integer(if q.is_replicated { 1 } else { 0 }),
            ])
        })
        .collect()
}

/// Build a RedisValue for the GRAPH.INFO WaitingQueries section.
pub(crate) fn waiting_queries_reply() -> Vec<RedisValue> {
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
                RedisValue::BulkString(q.graph_name),
                RedisValue::BulkString("Query".into()),
                RedisValue::BulkString(q.query),
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

struct PendingEntry {
    graph_name: String,
    entry: TelemetryEntry,
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

/// Push a telemetry entry to the background channel. Lock-free hot path.
pub(crate) fn enqueue_entry(
    graph_name: &str,
    entry: TelemetryEntry,
) {
    // Skip on replicas: the master's XADDs are replicated to us, so writing
    // here would duplicate entries (and direct writes to a replica must not
    // create a stream).
    if IS_REPLICA.load(Ordering::Relaxed) {
        return;
    }
    if let Some(tx) = SENDER.lock().as_ref() {
        let _ = tx.send(PendingEntry {
            graph_name: graph_name.to_string(),
            entry,
        });
    }
}

/// Tracks whether this Redis instance is currently a replica. Updated on
/// module load and on `RedisModuleEvent_ReplicationRoleChanged` notifications.
static IS_REPLICA: AtomicBool = AtomicBool::new(false);

/// Update the cached replica state. Called from module init and the role
/// change event handler.
pub(crate) fn set_is_replica(is_replica: bool) {
    IS_REPLICA.store(is_replica, Ordering::Relaxed);
}

/// Spawn the background flusher thread. Must be called once at module init.
pub(crate) fn start_flusher_thread() {
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

    let handle = thread::Builder::new()
        .name("falkordb-telemetry".to_string())
        .spawn(flusher_loop)
        .expect("failed to spawn telemetry flusher thread");
    *FLUSHER.lock() = Some(handle);
}

/// Stop the background flusher: drop the sender so the channel disconnects,
/// then join the thread. Must be called on module unload before tearing down
/// Redis state the flusher's `RM_Call("XADD")` touches.
pub(crate) fn shutdown_flusher_thread() {
    // Drop the sender to close the channel; the flusher loop exits on
    // `Disconnected` after draining any pending entries.
    drop(SENDER.lock().take());
    let handle = FLUSHER.lock().take();
    if let Some(h) = handle {
        let _ = h.join();
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
        f(std::ptr::null_mut())
    };

    let mut batch: Vec<PendingEntry> = Vec::with_capacity(FLUSH_BATCH_MAX);

    loop {
        // Block until at least one entry is available (or the channel closes).
        match rx.recv_timeout(FLUSH_INTERVAL) {
            Ok(first) => batch.push(first),
            Err(crossfire::RecvTimeoutError::Timeout) => continue,
            Err(crossfire::RecvTimeoutError::Disconnected) => break,
        }
        // Drain any additional entries non-blockingly, up to the batch cap.
        while batch.len() < FLUSH_BATCH_MAX {
            match rx.try_recv() {
                Ok(pe) => batch.push(pe),
                Err(_) => break,
            }
        }

        // Single GIL acquisition for the whole batch.
        unsafe {
            raw::RedisModule_ThreadSafeContextLock.expect("ThreadSafeContextLock")(tsc);
        }
        let ctx = Context::new(tsc);
        for pe in batch.drain(..) {
            write_to_stream(&ctx, &pe.graph_name, &pe.entry);
        }
        unsafe {
            raw::RedisModule_ThreadSafeContextUnlock.expect("ThreadSafeContextUnlock")(tsc);
        }
    }

    unsafe {
        raw::RedisModule_FreeThreadSafeContext.expect("RedisModule_FreeThreadSafeContext")(tsc);
    }
}
