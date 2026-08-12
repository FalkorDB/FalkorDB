//! Telemetry subsystem: stream writing and running/waiting query registry.
//!
//! After each query execution, a telemetry entry is written to a Redis stream
//! named `telemetry{graph_name}`. The GRAPH.INFO command reads the global
//! query registry to report currently running and waiting queries.

use crossfire::mpmc::{self, List};
use crossfire::{MRx, MTx};
use parking_lot::Mutex;
use redis_module::{CallOptions, CallOptionsBuilder, Context, RedisString, RedisValue, raw};
use std::sync::Arc;
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
    format!("telemetry{{{graph_name}}}")
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

/// Pre-formatted XADD arguments for one telemetry entry.
///
/// Built **outside** the Redis module lock so that the only work performed
/// while the GIL is held is `create_string` + `RM_Call("XADD")`. The pure-Rust
/// formatting (`format!`, float parsing, stream-key construction) happens off
/// the critical section, shrinking the window during which the main thread is
/// stalled.
struct PreparedXadd {
    /// Alternating XADD argument tokens (stream key, MAXLEN, fields, values).
    /// Field names are `'static` borrows; only values are owned.
    args: Vec<std::borrow::Cow<'static, str>>,
}

/// Build the XADD argument list for one entry. Pure Rust — no GIL required.
/// Consumes the entry so the query/params strings are moved, not cloned.
fn prepare_xadd(
    pe: PendingEntry,
    max_len: &str,
) -> PreparedXadd {
    use std::borrow::Cow;
    let entry = pe.entry;
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
    let cache_flag = if entry.utilized_cache { "1" } else { "0" };
    let write_flag = if entry.is_write { "1" } else { "0" };
    let timeout_flag = if entry.timed_out { "1" } else { "0" };

    // XADD telemetry{graph} MAXLEN ~ <max> * field value ...
    PreparedXadd {
        args: vec![
            Cow::Owned(stream_name(&pe.graph_name)),
            Cow::Borrowed("MAXLEN"),
            Cow::Borrowed("~"),
            Cow::Owned(max_len.to_owned()),
            Cow::Borrowed("*"),
            Cow::Borrowed("Received at"),
            Cow::Owned(received),
            Cow::Borrowed("Query"),
            Cow::Owned(entry.query),
            Cow::Borrowed("Query parameters"),
            Cow::Owned(entry.params),
            Cow::Borrowed("Total duration"),
            Cow::Owned(total),
            Cow::Borrowed("Wait duration"),
            Cow::Owned(wait),
            Cow::Borrowed("Execution duration"),
            Cow::Owned(exec),
            Cow::Borrowed("Report duration"),
            Cow::Owned(report),
            Cow::Borrowed("Utilized cache"),
            Cow::Borrowed(cache_flag),
            Cow::Borrowed("Write"),
            Cow::Borrowed(write_flag),
            Cow::Borrowed("Timeout"),
            Cow::Borrowed(timeout_flag),
        ],
    }
}

/// Issue the XADD for a pre-formatted entry. Must be called with the GIL held.
fn dispatch_xadd(
    ctx: &Context,
    prepared: &PreparedXadd,
    call_options: &CallOptions,
) {
    let args: Vec<RedisString> = prepared
        .args
        .iter()
        .map(|s| ctx.create_string(s.as_ref()))
        .collect();
    let _: redis_module::CallResult = ctx.call_ext(
        "XADD",
        call_options,
        args.iter().collect::<Vec<_>>().as_slice(),
    );
}

/// CallOptions used by the flusher: replicate the XADD to attached replicas
/// so the telemetry stream stays mirrored across master/replica.
fn replicated_call_options() -> CallOptions {
    CallOptionsBuilder::new().replicate().build()
}

/// Delete the telemetry stream for a graph.
pub fn delete_stream(
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

struct PendingEntry {
    graph_name: Arc<str>,
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
pub fn enqueue_entry(
    graph_name: &Arc<str>,
    entry: TelemetryEntry,
) {
    // `CMD_INFO no` means "do not log finished queries", which is exactly this
    // path — and it is the only way to opt out of what logging one costs.
    if !LOG_QUERIES.load(Ordering::Relaxed) {
        return;
    }
    // Skip on replicas: the master's XADDs are replicated to us, so writing
    // here would duplicate entries (and direct writes to a replica must not
    // create a stream).
    if IS_REPLICA.load(Ordering::Relaxed) {
        return;
    }
    if let Some(tx) = SENDER.lock().as_ref() {
        let _ = tx.send(PendingEntry {
            graph_name: Arc::clone(graph_name),
            entry,
        });
    }
}

/// Tracks whether this Redis instance is currently a replica. Updated on
/// module load and on `RedisModuleEvent_ReplicationRoleChanged` notifications.
static IS_REPLICA: AtomicBool = AtomicBool::new(false);

/// Mirror of the `CMD_INFO` configuration, which decides whether finished
/// queries are logged to a graph's telemetry stream.
///
/// A mirror rather than a read of `CONFIGURATION_CMD_INFO` itself, because that
/// one lives behind a `RedisGILGuard` and this is read on a worker thread that
/// holds no GIL — taking it there is exactly the contention the whole telemetry
/// channel exists to avoid. `CMD_INFO` is registered `IMMUTABLE`, so the value
/// is settled at module load and one store at init is enough.
static LOG_QUERIES: AtomicBool = AtomicBool::new(true);

/// Publish the effective `CMD_INFO` value for the query path to read.
pub fn set_log_queries(enabled: bool) {
    LOG_QUERIES.store(enabled, Ordering::Relaxed);
}

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

    let handle = thread::Builder::new()
        .name("falkordb-telemetry".to_string())
        .spawn(flusher_loop)
        .expect("failed to spawn telemetry flusher thread");
    *FLUSHER.lock() = Some(handle);
}

/// Stop the background flusher: drop the sender so the channel disconnects,
/// then join the thread. Must be called on module unload before tearing down
/// Redis state the flusher's `RM_Call("XADD")` touches.
pub fn shutdown_flusher_thread() {
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

        // Format every entry's XADD arguments *outside* the GIL so the
        // critical section below only does string creation + RM_Call.
        let max_len = MAX_INFO_QUERIES.load(Ordering::Relaxed).to_string();
        let call_options = replicated_call_options();
        #[allow(clippy::iter_with_drain)]
        let prepared: Vec<PreparedXadd> = batch
            .drain(..)
            .map(|pe| prepare_xadd(pe, &max_len))
            .collect();

        // Single GIL acquisition for the whole batch, through the same guard queries
        // use, so every acquisition in the process funnels through one place.
        {
            let _gil = crate::query_session::hold_gil();
            let ctx = Context::new(tsc);
            for p in &prepared {
                dispatch_xadd(&ctx, p, &call_options);
            }
        }
    }

    unsafe {
        raw::RedisModule_FreeThreadSafeContext.expect("RedisModule_FreeThreadSafeContext")(tsc);
    }
}
