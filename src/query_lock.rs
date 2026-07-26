//! Redis implementation of [`graph::query_lock::QueryLock`].
//!
//! This is the *only* place that knows the commit lock is "module GIL + per-graph
//! `RwLock`". The `graph` crate calls
//! [`graph::query_lock::upgrade_to_write`] and never sees Redis.
//!
//! ## Model (mirrors FalkorDB C's `QueryCtx` 2PL)
//!
//! A write query runs in two modes, tracked per thread by [`LockSession`]:
//!
//! * **Reader** — holds only the per-graph read lock (L1-read). The match phase
//!   runs here, concurrently with other readers, and without the GIL.
//! * **Writer** — holds the GIL *and* the per-graph write lock (L1-write).
//!   Entered on the query's first mutation and held until the query ends, so
//!   every later step (including reads of the shared, non-MVCC index) sees an
//!   exclusively locked graph.
//!
//! Escalation order is **release read → GIL → write**, never GIL-while-holding-L1
//! — that inversion is the issue #726 deadlock. Because the guards are
//! `Arc`-owned (`parking_lot`'s `arc_lock`), the session can hold them across the
//! whole query and swap read for write mid-flight, which a scope-bound
//! `RwLockReadGuard` in a caller frame could never allow.

use crate::graph_core::{ThreadedGraph, ffi};
use graph::query_lock::{AccessMode, QueryLock};
use parking_lot::{ArcRwLockReadGuard, ArcRwLockWriteGuard, RawRwLock, RwLock};
use redis_module::raw;
use std::{
    cell::{Cell, RefCell},
    sync::Arc,
};

/// Which lock(s) the current query holds.
enum Mode {
    /// Per-graph read lock only; no GIL.
    Reader(ArcRwLockReadGuard<RawRwLock, ThreadedGraph>),
    /// GIL + per-graph write lock. `ts_ctx` is the thread-safe context whose
    /// GIL we took (null when the GIL was already held by this thread, e.g. the
    /// main thread during command dispatch — then we must not unlock it).
    Writer {
        guard: ArcRwLockWriteGuard<RawRwLock, ThreadedGraph>,
        ts_ctx: *mut raw::RedisModuleCtx,
    },
}

/// Per-query lock state for one thread — the analogue of C's `QueryCtx`
/// `read_locked`/`write_locked` flags plus the guards themselves.
pub struct LockSession {
    graph: Arc<RwLock<ThreadedGraph>>,
    mode: Option<Mode>,
    /// True when this thread already held the GIL before the session started
    /// (main-thread command dispatch, or the write loop's DDL window), so
    /// escalation must not re-acquire the non-recursive module GIL.
    gil_already_held: bool,
}

impl LockSession {
    /// Start a session in **reader** mode, taking the per-graph read lock.
    ///
    /// `gil_already_held` must be true iff the calling thread already holds the
    /// module GIL (implicitly on the main thread, or explicitly in a
    /// GIL-then-L1 window). In that case escalation skips the GIL acquire —
    /// the lock order is already satisfied.
    /// Register `graph` as this query's target **without locking yet**. The lock
    /// itself is taken by [`QueryLock::acquire_read`] so that the whole
    /// read → write → release lifecycle lives behind the trait.
    pub fn new_target(
        graph: &Arc<RwLock<ThreadedGraph>>,
        gil_already_held: bool,
    ) -> Self {
        Self {
            graph: Arc::clone(graph),
            mode: None,
            gil_already_held,
        }
    }

    /// Take the per-graph read lock. Idempotent; a no-op once locked.
    fn acquire_read(&mut self) -> Result<(), String> {
        if self.mode.is_none() {
            self.mode = Some(Mode::Reader(RwLock::read_arc(&self.graph)));
            MODE.with(|m| m.set(AccessMode::Read));
        }
        Ok(())
    }

    /// Start a session already in **writer** mode, for callers that legitimately
    /// hold the GIL first and want the write lock for the whole query (index
    /// DDL, which calls GIL-requiring RediSearch spec FFI during execution).
    pub fn new_writer_gil_held(graph: &Arc<RwLock<ThreadedGraph>>) -> Self {
        MODE.with(|m| m.set(AccessMode::Write));
        Self {
            graph: Arc::clone(graph),
            mode: Some(Mode::Writer {
                guard: RwLock::write_arc(graph),
                ts_ctx: std::ptr::null_mut(),
            }),
            gil_already_held: true,
        }
    }

    /// Borrow the locked graph, whichever mode we are in.
    pub fn graph(&self) -> &ThreadedGraph {
        match self
            .mode
            .as_ref()
            .expect("query lock not acquired (call acquire_read first)")
        {
            Mode::Reader(g) => g,
            Mode::Writer { guard, .. } => guard,
        }
    }

    /// Mutably borrow the locked graph. Only available in writer mode.
    pub fn graph_mut(&mut self) -> Option<&mut ThreadedGraph> {
        match self.mode.as_mut()? {
            Mode::Writer { guard, .. } => Some(guard),
            Mode::Reader(_) => None,
        }
    }

    pub const fn holds_write(&self) -> bool {
        matches!(self.mode, Some(Mode::Writer { .. }))
    }

    /// Escalate reader → writer: **release read, take the GIL, take write**.
    ///
    /// Idempotent. The brief window where no per-graph lock is held is safe: the
    /// query owns the single MVCC write slot for its whole lifetime, so no other
    /// writer can commit, and it has not yet mutated the shared index (that is
    /// what it is escalating in order to do) — a reader entering the window sees
    /// a wholly consistent older state.
    fn escalate(&mut self) -> Result<(), String> {
        if self.holds_write() {
            return Ok(());
        }
        // 1. Drop the read lock FIRST — holding it across the GIL acquire is the
        //    #726 inversion. Publish `Unlocked` too, so the lock-order assertion
        //    inside the host-lock acquire below sees the truth.
        self.mode = None;
        MODE.with(|m| m.set(AccessMode::Unlocked));

        // 2. Acquire the GIL (unless this thread already holds it).
        let ts_ctx = if self.gil_already_held {
            std::ptr::null_mut()
        } else {
            let ctx = unsafe { ffi::get_thread_safe_context(std::ptr::null_mut()) };
            unsafe { ffi::lock_thread_safe_ctx(ctx) };
            ctx
        };

        // 3. Now take the per-graph write lock: GIL → L1-write, the canonical
        //    order shared with every inline main-thread command.
        let guard = RwLock::write_arc(&self.graph);
        self.mode = Some(Mode::Writer { guard, ts_ctx });
        MODE.with(|m| m.set(AccessMode::Write));

        // 4. The key could have been deleted while we held no per-graph lock —
        //    re-verify the graph is still reachable before mutating it, as C does
        //    by re-opening the key under WRITE. Locks stay held; the caller aborts
        //    the query and the session releases them on drop.
        if !crate::graph_core::graph_is_registered(&self.graph) {
            return Err("graph was deleted while the query was running, aborting".to_string());
        }
        Ok(())
    }

    /// Release everything held by the session (write lock then GIL, reverse
    /// order of acquisition). Called on query completion.
    pub fn release(&mut self) {
        MODE.with(|m| m.set(AccessMode::Unlocked));
        match self.mode.take() {
            Some(Mode::Writer { guard, ts_ctx }) => {
                drop(guard);
                if !ts_ctx.is_null() {
                    unsafe {
                        ffi::unlock_thread_safe_ctx(ts_ctx);
                        ffi::free_thread_safe_context(ts_ctx);
                    }
                }
            }
            Some(Mode::Reader(guard)) => drop(guard),
            None => {}
        }
    }
}

impl Drop for LockSession {
    fn drop(&mut self) {
        self.release();
    }
}

thread_local! {
    /// The lock session of the query currently executing on this thread, if any.
    /// Mirrors C's thread-local `QueryCtx`: the runtime deep in the plan can
    /// escalate without any of the plumbing knowing about locks.
    static CURRENT: RefCell<Option<LockSession>> = const { RefCell::new(None) };

    /// Mirror of this thread's [`AccessMode`], kept outside [`CURRENT`] so that
    /// `mode()` / `holds_write()` — and the debug assertions built on them — never
    /// borrow the session `RefCell`. This is load-bearing: `escalate` runs *inside*
    /// a `with_current` borrow and takes the host lock, whose lock-order assertion
    /// reads the mode; reading it through the session would panic with
    /// "RefCell already borrowed".
    static MODE: Cell<AccessMode> = const { Cell::new(AccessMode::Unlocked) };

    /// Nesting depth of explicit [`QueryLock::lock_host`] acquisitions on this
    /// thread (background index tasks), and the thread-safe context whose GIL we
    /// actually took (null when the acquisition was a no-op because the GIL was
    /// already held).
    static HOST_DEPTH: Cell<usize> = const { Cell::new(0) };
    static HOST_CTX: Cell<*mut raw::RedisModuleCtx> = const { Cell::new(std::ptr::null_mut()) };
}

/// Install `session` as the current thread's query lock session.
///
/// Query code then reaches the locked graph only through short-lived borrows
/// ([`with_graph`]) and escalates through
/// [`graph::query_lock::upgrade_to_write`]. Nothing may hold a borrow across
/// query execution: escalation *swaps* the read guard for a write guard, which
/// would invalidate any outstanding `&ThreadedGraph`. Keeping every borrow brief
/// is what makes the mid-query upgrade sound.
fn install(session: LockSession) {
    // A nested install would already have blocked inside `read_arc`/`write_arc`
    // above (the per-graph lock is not recursive), so this fires only in the
    // constructor-reordered case — but it documents the invariant loudly.
    debug_assert!(
        CURRENT.with(|c| c.borrow().is_none()),
        "a lock session is already installed on this thread; nesting them \
         deadlocks on the non-recursive per-graph lock"
    );
    CURRENT.with(|c| *c.borrow_mut() = Some(session));
}

/// Remove and return the current thread's session (releasing its locks when the
/// returned value is dropped).
fn take() -> Option<LockSession> {
    CURRENT.with(|c| c.borrow_mut().take())
}

/// RAII wrapper around [`install`] / [`take`].
///
/// Binding one guarantees the session — and therefore the per-graph lock and any
/// host lock it escalated to — is released on **every** exit path, including
/// early returns and unwinds. A leaked session would return a pool worker to the
/// queue still holding the read lock, wedging the next writer, so installation is
/// only exposed this way.
#[must_use]
pub struct ScopedSession(());

impl ScopedSession {
    /// Register `graph` as the current query's target and enter reader mode
    /// through the trait, so acquisition follows the same path as escalation and
    /// release.
    /// Panics if the host cannot take the read lock. This implementation simply
    /// acquires a `parking_lot` read guard and so never fails; a host whose
    /// `acquire_read` can fail must surface it at these call sites instead.
    pub fn begin(
        graph: &Arc<RwLock<ThreadedGraph>>,
        gil_already_held: bool,
    ) -> Self {
        install(LockSession::new_target(graph, gil_already_held));
        // Bind the guard first so a failure still releases the session.
        let guard = Self(());
        graph::query_lock::acquire_read().expect("failed to acquire the per-graph read lock");
        guard
    }

    /// Register `graph` and enter **writer** mode immediately, for callers that
    /// already hold the host lock and write for the whole query.
    pub fn begin_writer(graph: &Arc<RwLock<ThreadedGraph>>) -> Self {
        install(LockSession::new_writer_gil_held(graph));
        Self(())
    }
}

impl Drop for ScopedSession {
    fn drop(&mut self) {
        drop(take());
    }
}

/// Run `f` with the current thread's session, if one is installed.
///
/// The borrow is confined to `f`, which must not itself escalate (that would
/// re-enter the `RefCell`).
pub fn with_current<R>(f: impl FnOnce(&mut LockSession) -> R) -> Option<R> {
    CURRENT.with(|c| c.borrow_mut().as_mut().map(f))
}

/// Run `f` against the locked graph in a **short-lived** borrow.
///
/// # Panics
/// If no session is installed — every write path installs one before executing.
pub fn with_graph<R>(f: impl FnOnce(&ThreadedGraph) -> R) -> R {
    with_current(|s| f(s.graph())).expect("no lock session installed for this query")
}

/// The registered [`QueryLock`] implementation: escalates the calling thread's
/// [`LockSession`].
pub struct RedisQueryLock;

impl QueryLock for RedisQueryLock {
    fn acquire_read(&self) -> Result<(), String> {
        with_current(LockSession::acquire_read).unwrap_or(Ok(()))
    }

    fn release(&self) {
        if let Some(mut session) = take() {
            session.release();
        }
    }

    fn mode(&self) -> AccessMode {
        MODE.with(Cell::get)
    }

    fn upgrade_to_write(&self) -> Result<(), String> {
        // Already a writer: return without touching the session `RefCell`, so
        // this stays safe to call from anywhere (including while the session is
        // borrowed or has been taken out for the commit phase).
        if MODE.with(Cell::get) == AccessMode::Write {
            return Ok(());
        }
        // No session installed => not running inside a managed query (e.g. a
        // background index task that already ordered its own locks): nothing to
        // escalate.
        with_current(LockSession::escalate).unwrap_or(Ok(()))
    }

    fn lock_host(&self) {
        // Already covered: the main thread holds the GIL implicitly during
        // command dispatch, and an escalated query holds it explicitly. The
        // module GIL is NOT recursive, so re-acquiring would self-deadlock.
        if graph::thread_id::is_main_thread() || holds_gil() {
            HOST_DEPTH.with(|d| d.set(d.get() + 1));
            return;
        }
        let ctx = unsafe { ffi::get_thread_safe_context(std::ptr::null_mut()) };
        unsafe { ffi::lock_thread_safe_ctx(ctx) };
        HOST_CTX.with(|c| c.set(ctx));
        HOST_DEPTH.with(|d| d.set(d.get() + 1));
    }

    fn unlock_host(&self) {
        let depth = HOST_DEPTH.with(|d| {
            let n = d.get().saturating_sub(1);
            d.set(n);
            n
        });
        if depth > 0 {
            return;
        }
        let ctx = HOST_CTX.with(|c| c.replace(std::ptr::null_mut()));
        if !ctx.is_null() {
            unsafe {
                ffi::unlock_thread_safe_ctx(ctx);
                ffi::free_thread_safe_context(ctx);
            }
        }
    }

    fn holds_host_lock(&self) -> bool {
        graph::thread_id::is_main_thread() || holds_gil()
    }
}

/// True if this thread holds the module GIL through a query escalation or an
/// explicit [`QueryLock::lock_host`]. Both must be consulted: the module GIL is
/// **not recursive**, so a nested acquire would self-deadlock.
fn holds_gil() -> bool {
    MODE.with(Cell::get) == AccessMode::Write || HOST_DEPTH.with(Cell::get) > 0
}

/// Register the Redis commit lock with the `graph` crate. Called once at module
/// init, before any query runs.
pub fn register() {
    graph::query_lock::set_query_lock(Arc::new(RedisQueryLock));
}
