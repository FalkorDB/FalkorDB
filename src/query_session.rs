//! A query's locks, and the Redis implementation of the two seams in
//! [`graph::locks`].
//!
//! [`QuerySession`] is the value that **owns** the locks a query holds. The host
//! creates one, hands `&session` to the `Runtime`, reaches the locked graph
//! through it, and drops it when the query ends. There is no hidden state: if you
//! cannot see a session, no lock is held.
//!
//! ## Modes (mirrors FalkorDB C's `QueryCtx` 2PL)
//!
//! * **Reader** — the per-graph read lock (L1-read) only. The match phase runs
//!   here, concurrently with other readers, without the GIL.
//! * **Writer** — the module GIL *and* the per-graph write lock (L1-write).
//!   Entered on the query's first mutation ([`QuerySession::upgrade_to_write`])
//!   and held until the query ends, so every later step — including reads of the
//!   shared, non-MVCC index — sees an exclusively locked graph.
//!
//! Escalation order is **release read → GIL → write**; the reverse is the issue
//! #726 deadlock. The guards are `Arc`-owned (`parking_lot`'s `arc_lock`) so the
//! session can hold them across the whole query and swap read for write
//! mid-flight, which a scope-bound `RwLockReadGuard` never could.
//!
//! ## The GIL
//!
//! [`RedisGil`] implements [`GlobalLock`] and is registered once at module init.
//! The FFI that takes the GIL is **private to this module**, so the only way to
//! acquire it anywhere in the process is [`GlobalLockGuard`] — which is what makes
//! the ordering rule enforceable rather than merely documented.

use crate::graph_core::{ThreadedGraph, ffi};
use graph::locks::{GlobalLock, GlobalLockGuard, WriteEscalation};
use parking_lot::{ArcRwLockReadGuard, ArcRwLockWriteGuard, RawRwLock, RwLock};
use redis_module::raw;
use std::{cell::Cell, cell::RefCell, ptr::NonNull, sync::Arc};

/// Which lock(s) a query holds.
enum Mode {
    /// Per-graph read lock only; no GIL.
    Reader(ArcRwLockReadGuard<RawRwLock, ThreadedGraph>),
    /// GIL + per-graph write lock.
    ///
    /// Field order is load-bearing: struct fields drop in declaration order, so
    /// the write lock is released before the GIL — the reverse of acquisition.
    /// `_gil` is never read; it is held for its `Drop`.
    Writer {
        guard: ArcRwLockWriteGuard<RawRwLock, ThreadedGraph>,
        _gil: GlobalLockGuard,
    },
}

/// The locks held by one query, on one thread.
///
/// Not `Send` (the `GlobalLockGuard` inside isn't): the GIL is released by the
/// thread that took it.
pub struct QuerySession {
    graph: Arc<RwLock<ThreadedGraph>>,
    /// `RefCell` because escalation happens through `&self` — plan operators
    /// reach the session as `&dyn WriteEscalation`. Every borrow is confined to
    /// one method call, so escalating from *inside* a [`Self::with_graph`]
    /// closure is the one misuse, and it panics loudly instead of dangling.
    mode: RefCell<Option<Mode>>,
}

impl QuerySession {
    /// Enter **reader** mode: take the per-graph read lock.
    pub fn begin(graph: &Arc<RwLock<ThreadedGraph>>) -> Self {
        let session = Self {
            graph: Arc::clone(graph),
            mode: RefCell::new(Some(Mode::Reader(RwLock::read_arc(graph)))),
        };
        ACCESS.set(AccessMode::Read);
        session
    }

    /// Enter **writer** mode immediately — GIL, then the per-graph write lock —
    /// for callers that write for the whole query and have nothing to escalate
    /// (index DDL, which calls GIL-requiring RediSearch spec FFI as it executes).
    ///
    /// On the main thread the GIL acquire is the no-op it must be, since command
    /// dispatch already holds it.
    pub fn begin_writer(graph: &Arc<RwLock<ThreadedGraph>>) -> Self {
        let gil = GlobalLockGuard::acquire();
        let session = Self {
            graph: Arc::clone(graph),
            mode: RefCell::new(Some(Mode::Writer {
                guard: RwLock::write_arc(graph),
                _gil: gil,
            })),
        };
        ACCESS.set(AccessMode::Write);
        session
    }

    /// Run `f` against the locked graph in a **short-lived** borrow.
    ///
    /// Short-lived is the point: escalation *replaces* the read guard with a write
    /// guard, so a `&ThreadedGraph` that outlived it would be invalid. `f` must
    /// therefore not escalate.
    ///
    /// # Panics
    /// If the session has no lock — impossible for a live session, since both
    /// constructors take one and only `Drop` gives it up.
    pub fn with_graph<R>(
        &self,
        f: impl FnOnce(&ThreadedGraph) -> R,
    ) -> R {
        match self.mode.borrow().as_ref().expect("session holds no lock") {
            Mode::Reader(graph) => f(graph),
            Mode::Writer { guard, .. } => f(guard),
        }
    }

    /// Run `f` against the **mutably** locked graph in a short-lived borrow.
    ///
    /// `None` if this query has not escalated; `Some` is proof the write lock is
    /// held and mutation is allowed.
    pub fn with_graph_mut<R>(
        &self,
        f: impl FnOnce(&mut ThreadedGraph) -> R,
    ) -> Option<R> {
        match self.mode.borrow_mut().as_mut()? {
            Mode::Writer { guard, .. } => Some(f(guard)),
            Mode::Reader(_) => None,
        }
    }

    /// Escalate reader → writer: **release read, take the GIL, take write**.
    ///
    /// Idempotent. The brief window where no per-graph lock is held is safe: the
    /// query owns the single MVCC write slot for its whole lifetime, so no other
    /// writer can commit, and it has not yet mutated the shared index (that is
    /// what it is escalating in order to do) — a reader entering the window sees a
    /// wholly consistent older state.
    ///
    /// The inherent method plan operators reach through
    /// [`WriteEscalation::upgrade_to_write`].
    pub fn upgrade_to_write(&self) -> Result<(), String> {
        let mut mode = self.mode.borrow_mut();
        if matches!(*mode, Some(Mode::Writer { .. })) {
            return Ok(());
        }
        // 1. Drop the read lock FIRST — holding it across the GIL acquire is the
        //    #726 inversion. Publish `Unlocked` too, so the lock-order assertion
        //    inside the GIL acquire below sees the truth.
        *mode = None;
        ACCESS.set(AccessMode::Unlocked);

        // 2. GIL, then 3. the per-graph write lock: the canonical order, shared
        //    with every inline main-thread command.
        let gil = GlobalLockGuard::acquire();
        let guard = RwLock::write_arc(&self.graph);
        *mode = Some(Mode::Writer { guard, _gil: gil });
        ACCESS.set(AccessMode::Write);
        drop(mode);

        // 4. The key could have been deleted while we held no per-graph lock —
        //    re-verify the graph is still reachable before mutating it, as C does
        //    by re-opening the key under WRITE. The locks stay held; the caller
        //    aborts the query and `Drop` releases them.
        if crate::graph_core::graph_is_registered(&self.graph) {
            Ok(())
        } else {
            Err("graph was deleted or replaced while the query was running, aborting".to_string())
        }
    }
}

impl WriteEscalation for QuerySession {
    fn upgrade_to_write(&self) -> Result<(), String> {
        Self::upgrade_to_write(self)
    }
}

impl Drop for QuerySession {
    /// Releases the write lock then the GIL — reverse order of acquisition, by
    /// field order in [`Mode::Writer`] — or the read lock.
    fn drop(&mut self) {
        ACCESS.set(AccessMode::Unlocked);
        *self.mode.borrow_mut() = None;
    }
}

/// `WriteEscalation` for query paths that must never write.
///
/// `GRAPH.RECORD` replays a plan against an MVCC snapshot for debugging, holding
/// no session. A write there has nowhere to commit, so refuse rather than let it
/// mutate shared state unlocked.
pub struct NoEscalation;

impl WriteEscalation for NoEscalation {
    fn upgrade_to_write(&self) -> Result<(), String> {
        Err("this command cannot execute write queries".to_string())
    }
}

/// Which locks the query on *this thread* holds.
///
/// Exists for exactly one job: [`RedisGil::lock`] must reject a GIL acquire made
/// while the per-graph read lock is held (#726), and it is reached from index FFI
/// deep in the graph crate with no session in sight. Written only by
/// [`QuerySession`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AccessMode {
    Unlocked,
    Read,
    Write,
}

thread_local! {
    static ACCESS: Cell<AccessMode> = const { Cell::new(AccessMode::Unlocked) };

    /// Nesting depth of GIL acquisitions on this thread, and the thread-safe
    /// context whose GIL we actually took (`None` when the acquire was a no-op
    /// because the GIL was already held).
    static GIL_DEPTH: Cell<usize> = const { Cell::new(0) };
    static GIL_CTX: Cell<Option<NonNull<raw::RedisModuleCtx>>> = const { Cell::new(None) };
}

/// The process-wide global lock — Redis's module GIL.
pub struct RedisGil;

impl GlobalLock for RedisGil {
    fn lock(&self) {
        GIL_DEPTH.set(GIL_DEPTH.get() + 1);
        // Nothing to do for a nested acquire, or on the main thread which holds
        // the GIL implicitly for the whole command callback: the module GIL is NOT
        // recursive, so re-acquiring would self-deadlock. `GIL_DEPTH` is a
        // complete record of this thread's acquisitions because `lock_gil` below
        // is the only place in the module that takes the GIL.
        if GIL_DEPTH.get() > 1 || graph::thread_id::is_main_thread() {
            return;
        }
        // The lock-order rule, checked at the one funnel every acquire goes
        // through: never the GIL while holding the per-graph read lock (#726).
        // Escalation satisfies this by releasing the read lock first; this catches
        // anything else — e.g. index FFI reached from a read path.
        debug_assert_ne!(
            ACCESS.get(),
            AccessMode::Read,
            "taking the global lock while holding the per-graph read lock is the \
             #726 deadlock; escalate to writer mode first"
        );
        // SAFETY: a null blocked-client yields a detached context, which is what
        // we want — we need the GIL, not a client to reply to.
        let ctx = unsafe { ffi::get_thread_safe_context(std::ptr::null_mut()) };
        let ctx = NonNull::new(ctx).expect("RedisModule_GetThreadSafeContext returned null");
        // SAFETY: `ctx` was just created and is owned by this thread until the
        // matching `unlock` releases it.
        unsafe { lock_gil(ctx.as_ptr()) };
        GIL_CTX.set(Some(ctx));
    }

    fn unlock(&self) {
        let depth = GIL_DEPTH.get().saturating_sub(1);
        GIL_DEPTH.set(depth);
        if depth > 0 {
            return;
        }
        let Some(ctx) = GIL_CTX.take() else {
            return; // the acquire was a no-op (GIL already held)
        };
        // SAFETY: `ctx` is the context this thread locked in `lock`, taken out of
        // the thread-local so it is released exactly once.
        unsafe { release_gil(ctx.as_ptr()) };
    }
}

/// Acquire the module GIL through `ctx`.
///
/// Private on purpose: [`GlobalLockGuard`] is the only way to take the GIL
/// anywhere in the process, so lock order cannot be violated by a stray caller.
///
/// # Safety
/// `ctx` must be a valid thread-safe context this thread does not already hold.
unsafe fn lock_gil(ctx: *mut raw::RedisModuleCtx) {
    let f = unsafe { raw::RedisModule_ThreadSafeContextLock }.expect("missing Redis API symbol");
    unsafe { f(ctx) };
}

/// Release the GIL taken by [`lock_gil`] and free the context.
///
/// Resolves the symbols instead of `expect`ing them: this runs from a `Drop`,
/// where a panic aborts the process. If Redis no longer exposes the API, leaking
/// the context beats killing the server.
///
/// # Safety
/// `ctx` must be a valid thread-safe context whose GIL this thread holds, and must
/// not be used again.
unsafe fn release_gil(ctx: *mut raw::RedisModuleCtx) {
    unsafe {
        if let Some(unlock) = raw::RedisModule_ThreadSafeContextUnlock {
            unlock(ctx);
        }
        if let Some(free) = raw::RedisModule_FreeThreadSafeContext {
            free(ctx);
        }
    }
}

/// Register the module GIL as the `graph` crate's global lock. Called once at
/// module init, before any query can run.
pub fn register() {
    graph::locks::set_global_lock(Box::new(RedisGil));
}
