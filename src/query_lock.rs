//! Redis implementation of [`graph::query_lock::QueryLock`] and
//! [`graph::host_lock::HostLock`].
//!
//! This is the *only* place that knows those locks are "module GIL + per-graph
//! `RwLock`". The `graph` crate calls [`QueryLock::upgrade_to_write`] and
//! [`graph::host_lock::HostLockGuard::acquire`], and never sees Redis.
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
use graph::{host_lock::HostLock, query_lock::QueryLock};
use parking_lot::{ArcRwLockReadGuard, ArcRwLockWriteGuard, RawRwLock, RwLock};
use redis_module::raw;
use std::{
    cell::{Cell, RefCell},
    marker::PhantomData,
    ptr::NonNull,
    sync::Arc,
};

/// Lock mode held by the current query on this thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AccessMode {
    /// No per-graph lock held.
    Unlocked,
    /// Per-graph read lock only (the match phase).
    Read,
    /// GIL + per-graph write lock (from the first mutation on).
    Write,
}

/// RAII guard for the module GIL, taken straight from [`RedisQueryLock`].
///
/// Deliberately *not* routed through [`graph::host_lock::HostLockGuard`]: that
/// one silently no-ops when no host has registered an implementation, which is
/// the right behaviour for `graph`-crate unit tests but the wrong one here —
/// escalating without the GIL would be a silent lock-order violation.
struct Gil;

impl Gil {
    fn acquire() -> Self {
        RedisQueryLock.lock();
        Self
    }
}

impl Drop for Gil {
    fn drop(&mut self) {
        RedisQueryLock.unlock();
    }
}

/// Which lock(s) the current query holds.
enum Mode {
    /// Per-graph read lock only; no GIL.
    Reader(ArcRwLockReadGuard<RawRwLock, ThreadedGraph>),
    /// GIL + per-graph write lock.
    ///
    /// Field order is load-bearing: enum fields drop in declaration order, so the
    /// write lock is released before the GIL — the reverse of acquisition. `_gil`
    /// is never read; it is held for its `Drop`.
    Writer {
        guard: ArcRwLockWriteGuard<RawRwLock, ThreadedGraph>,
        _gil: Gil,
    },
}

impl Mode {
    const fn access(&self) -> AccessMode {
        match self {
            Self::Reader(_) => AccessMode::Read,
            Self::Writer { .. } => AccessMode::Write,
        }
    }
}

/// Per-query lock state for one thread — the analogue of C's `QueryCtx`
/// `read_locked`/`write_locked` flags plus the guards themselves.
pub struct LockSession {
    graph: Arc<RwLock<ThreadedGraph>>,
    mode: Option<Mode>,
}

impl LockSession {
    /// Start a session in **reader** mode on `graph`, taking the per-graph read
    /// lock.
    fn new_reader(graph: &Arc<RwLock<ThreadedGraph>>) -> Self {
        let session = Self {
            graph: Arc::clone(graph),
            mode: Some(Mode::Reader(RwLock::read_arc(graph))),
        };
        MODE.set(AccessMode::Read);
        session
    }

    /// Start a session already in **writer** mode, for callers that write for the
    /// whole query (index DDL, which calls GIL-requiring RediSearch spec FFI
    /// during execution) and so have nothing to escalate.
    ///
    /// Takes GIL → write, the same order as [`Self::escalate`]; on the main thread
    /// the GIL acquire is the no-op it must be.
    fn new_writer(graph: &Arc<RwLock<ThreadedGraph>>) -> Self {
        let gil = Gil::acquire();
        let session = Self {
            graph: Arc::clone(graph),
            mode: Some(Mode::Writer {
                guard: RwLock::write_arc(graph),
                _gil: gil,
            }),
        };
        MODE.set(AccessMode::Write);
        session
    }

    /// Borrow the locked graph, whichever mode we are in.
    pub fn graph(&self) -> &ThreadedGraph {
        match self.mode.as_ref().expect("query lock not acquired") {
            Mode::Reader(g) => g,
            Mode::Writer { guard, .. } => guard,
        }
    }

    /// Mutably borrow the locked graph. `Some` only in writer mode, and therefore
    /// proof that the write lock is held.
    pub fn graph_mut(&mut self) -> Option<&mut ThreadedGraph> {
        match self.mode.as_mut()? {
            Mode::Writer { guard, .. } => Some(guard),
            Mode::Reader(_) => None,
        }
    }

    /// Escalate reader → writer: **release read, take the GIL, take write**.
    ///
    /// Idempotent. The brief window where no per-graph lock is held is safe: the
    /// query owns the single MVCC write slot for its whole lifetime, so no other
    /// writer can commit, and it has not yet mutated the shared index (that is
    /// what it is escalating in order to do) — a reader entering the window sees
    /// a wholly consistent older state.
    fn escalate(&mut self) -> Result<(), String> {
        self.debug_check_mirror();
        if matches!(self.mode, Some(Mode::Writer { .. })) {
            return Ok(());
        }
        // 1. Drop the read lock FIRST — holding it across the GIL acquire is the
        //    #726 inversion. Publish `Unlocked` too, so the lock-order assertion
        //    inside the GIL acquire below sees the truth.
        self.mode = None;
        MODE.set(AccessMode::Unlocked);

        // 2. GIL, then 3. the per-graph write lock: the canonical order shared
        //    with every inline main-thread command.
        let gil = Gil::acquire();
        let guard = RwLock::write_arc(&self.graph);
        self.mode = Some(Mode::Writer { guard, _gil: gil });
        MODE.set(AccessMode::Write);

        // 4. The key could have been deleted while we held no per-graph lock —
        //    re-verify the graph is still reachable before mutating it, as C does
        //    by re-opening the key under WRITE. Locks stay held; the caller aborts
        //    the query and the session releases them on drop.
        if crate::graph_core::graph_is_registered(&self.graph) {
            Ok(())
        } else {
            Err("graph was deleted or replaced while the query was running, aborting".to_string())
        }
    }

    /// The [`MODE`] mirror must always agree with the guards actually held.
    fn debug_check_mirror(&self) {
        debug_assert_eq!(
            MODE.get(),
            self.mode
                .as_ref()
                .map_or(AccessMode::Unlocked, Mode::access),
            "the MODE mirror drifted from the locks actually held"
        );
    }
}

impl Drop for LockSession {
    /// Releases the write lock then the GIL (reverse order of acquisition, by
    /// field declaration order in [`Mode::Writer`]), or the read lock.
    ///
    /// Release happens only here — of the session itself, or of the
    /// [`ScopedSession`] that owns it — so no caller can forget it.
    fn drop(&mut self) {
        MODE.set(AccessMode::Unlocked);
        self.mode = None;
    }
}

thread_local! {
    /// The lock session of the query currently executing on this thread, if any.
    /// Mirrors C's thread-local `QueryCtx`: the runtime deep in the plan can
    /// escalate without any of the plumbing knowing about locks.
    static CURRENT: RefCell<Option<LockSession>> = const { RefCell::new(None) };

    /// Mirror of this thread's [`AccessMode`], kept outside [`CURRENT`] so that
    /// mode checks — and the debug assertions built on them — never borrow the
    /// session `RefCell`. This is load-bearing: `escalate` runs *inside* a
    /// `with_current` borrow and takes the GIL, whose lock-order assertion reads
    /// the mode; reading it through the session would panic with "RefCell already
    /// borrowed". [`LockSession::debug_check_mirror`] guards against drift.
    static MODE: Cell<AccessMode> = const { Cell::new(AccessMode::Unlocked) };

    /// Nesting depth of GIL acquisitions on this thread, and the thread-safe
    /// context whose GIL we actually took (`None` when the acquisition was a
    /// no-op because the GIL was already held).
    static HOST_DEPTH: Cell<usize> = const { Cell::new(0) };
    static HOST_CTX: Cell<Option<NonNull<raw::RedisModuleCtx>>> = const { Cell::new(None) };
}

/// RAII installation of the current thread's query lock session.
///
/// Binding one guarantees the session — and therefore the per-graph lock and any
/// GIL it escalated to — is released on **every** exit path, including early
/// returns and unwinds. A leaked session would return a pool worker to the queue
/// still holding the read lock, wedging the next writer, so installation is only
/// exposed this way.
///
/// Query code then reaches the locked graph through short-lived borrows
/// ([`with_graph`], [`with_graph_mut`]) and escalates through
/// [`upgrade_to_write`]. Nothing may hold a borrow across query execution:
/// escalation *swaps* the read guard for a write guard, which would invalidate any
/// outstanding `&ThreadedGraph`. Keeping every borrow brief is what makes the
/// mid-query upgrade sound.
///
/// Not `Send`: the session lives in a thread-local, so it must be dropped on the
/// thread that installed it.
#[must_use]
pub struct ScopedSession(PhantomData<*const ()>);

impl ScopedSession {
    /// Enter **reader** mode on `graph` for the current query.
    pub fn begin(graph: &Arc<RwLock<ThreadedGraph>>) -> Self {
        Self::install(LockSession::new_reader(graph))
    }

    /// Enter **writer** mode on `graph` immediately, for callers that write for
    /// the whole query.
    pub fn begin_writer(graph: &Arc<RwLock<ThreadedGraph>>) -> Self {
        Self::install(LockSession::new_writer(graph))
    }

    fn install(session: LockSession) -> Self {
        // A nested install would already have blocked in `read_arc`/`write_arc`
        // inside the constructor (the per-graph lock is not recursive), so this
        // fires only if that ever changes — but it documents the invariant loudly.
        debug_assert!(
            CURRENT.with_borrow(Option::is_none),
            "a lock session is already installed on this thread; nesting them \
             deadlocks on the non-recursive per-graph lock"
        );
        CURRENT.set(Some(session));
        Self(PhantomData)
    }
}

impl Drop for ScopedSession {
    fn drop(&mut self) {
        drop(CURRENT.take());
    }
}

/// Run `f` with the current thread's session, if one is installed.
///
/// The borrow is confined to `f`, which must not itself escalate (that would
/// re-enter the `RefCell`).
pub fn with_current<R>(f: impl FnOnce(&mut LockSession) -> R) -> Option<R> {
    CURRENT.with_borrow_mut(|c| c.as_mut().map(f))
}

/// Run `f` against the locked graph in a **short-lived** borrow.
///
/// # Panics
/// If no session is installed — every query path installs one before executing.
pub fn with_graph<R>(f: impl FnOnce(&ThreadedGraph) -> R) -> R {
    with_current(|s| f(s.graph())).expect("no lock session installed for this query")
}

/// Run `f` against the **mutably** locked graph in a short-lived borrow.
///
/// `None` if this thread has no session, or has one that has not escalated to
/// writer mode; `Some` is proof the write lock is held.
pub fn with_graph_mut<R>(f: impl FnOnce(&mut ThreadedGraph) -> R) -> Option<R> {
    with_current(|s| s.graph_mut().map(f))?
}

/// Escalate this thread's query to writer mode (GIL + per-graph write lock).
///
/// Free-function form of [`QueryLock::upgrade_to_write`] for host-side callers,
/// which have no `Runtime` to reach the trait object through.
pub fn upgrade_to_write() -> Result<(), String> {
    QueryLock::upgrade_to_write(&RedisQueryLock)
}

/// The host's lock implementation: per-query escalation ([`QueryLock`]) and the
/// process-wide module GIL ([`HostLock`]).
pub struct RedisQueryLock;

impl QueryLock for RedisQueryLock {
    fn upgrade_to_write(&self) -> Result<(), String> {
        // Already a writer: return without touching the session `RefCell`, so
        // this stays safe to call from anywhere (including while the session is
        // borrowed or has been taken out for the commit phase).
        if MODE.get() == AccessMode::Write {
            return Ok(());
        }
        match with_current(LockSession::escalate) {
            Some(res) => res,
            // No session on this thread. Reaching here means a caller is about to
            // mutate shared state believing it escalated, when in fact nothing is
            // locked — a silent lost-lock, not a benign no-op. Every real caller
            // runs on its query's own thread, where a `ScopedSession` is installed.
            None => {
                debug_assert!(
                    false,
                    "upgrade_to_write() with no lock session installed on this \
                     thread: the caller would mutate shared state unlocked"
                );
                Err("no lock session for this query; refusing to escalate".to_string())
            }
        }
    }
}

impl HostLock for RedisQueryLock {
    fn lock(&self) {
        HOST_DEPTH.set(HOST_DEPTH.get() + 1);
        // Nothing to do for a nested acquire, or on the main thread which holds the
        // GIL implicitly for the whole command callback: the module GIL is NOT
        // recursive, so re-acquiring would self-deadlock. `HOST_DEPTH` is a
        // complete record of this thread's explicit acquisitions because this
        // function is the only place in the module that takes the GIL.
        if HOST_DEPTH.get() > 1 || graph::thread_id::is_main_thread() {
            return;
        }
        // SAFETY: a null blocked-client yields a detached context, which is what
        // we want — we need the GIL, not a client to reply to.
        let ctx = unsafe { ffi::get_thread_safe_context(std::ptr::null_mut()) };
        let ctx = NonNull::new(ctx).expect("RedisModule_GetThreadSafeContext returned null");
        // SAFETY: `ctx` was just created and is owned by this thread until the
        // matching `unlock` frees it.
        unsafe { ffi::lock_thread_safe_ctx(ctx.as_ptr()) };
        HOST_CTX.set(Some(ctx));
    }

    fn unlock(&self) {
        let depth = HOST_DEPTH.get().saturating_sub(1);
        HOST_DEPTH.set(depth);
        if depth > 0 {
            return;
        }
        let Some(ctx) = HOST_CTX.take() else {
            return; // the acquire was a no-op (GIL already held)
        };
        // SAFETY: `ctx` is the context this thread locked in `lock`, taken out of
        // the thread-local so it is released exactly once. `release_` (not
        // `unlock_`) because this runs from a `Drop` and must not panic.
        unsafe { ffi::release_thread_safe_ctx(ctx.as_ptr()) };
    }
}

/// Debug-only guard for taking the GIL: the current query must not be holding the
/// per-graph **read** lock.
///
/// That is the issue #726 AB-BA inversion — a worker holding the read lock and
/// waiting for the GIL, versus the main thread holding the GIL and waiting for the
/// write lock. Escalation avoids it by releasing the read lock first, so this
/// catches any *other* path reaching for the GIL mid-query.
#[inline]
pub fn assert_safe_to_take_host_lock() {
    debug_assert!(
        MODE.get() != AccessMode::Read,
        "taking the host lock while holding the per-graph read lock is the #726 \
         deadlock; release the read lock first (upgrade_to_write does this)"
    );
}

/// Register the process-wide host lock with the `graph` crate. Called once at
/// module init.
///
/// Only the *host* lock is global — there is one module GIL per process. The
/// per-query lock is handed to each `Runtime` instead (see
/// [`graph::query_lock::QueryLock`]), so it needs no registration.
pub fn register() {
    graph::host_lock::set_host_lock(Box::new(RedisQueryLock));
}
