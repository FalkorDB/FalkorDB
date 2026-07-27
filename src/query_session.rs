//! A query's locks, and the Redis side of [`graph::locks::WriteEscalation`].
//!
//! [`QuerySession`] is the value that **owns** the locks a query holds: the host
//! creates one, hands `&session` to the `Runtime`, reaches the locked graph through
//! it, and drops it when the query ends. No hidden state — if you cannot see a
//! session, no lock is held.
//!
//! * **Reader** — the per-graph read lock only. The match phase runs here,
//!   concurrently with other readers, without the GIL.
//! * **Writer** — the GIL *and* the per-graph write lock, entered on the first
//!   mutation ([`QuerySession::upgrade_to_write`]) and held to the end of the
//!   query, so later steps — including reads of the shared index — are exclusive.
//!
//! The guards are `Arc`-owned (`parking_lot`'s `arc_lock`), which is what lets one
//! session hold them across a whole query and swap read for write mid-flight.
//!
//! The GIL never leaves this module: the FFI that takes it is private, so the only
//! ways to acquire it anywhere in the process are a [`QuerySession`] and
//! [`hold_gil`] (telemetry's flush thread). That is what makes the ordering rule
//! enforceable rather than merely documented — the `graph` crate cannot reach it at
//! all.

use crate::graph_core::{ThreadedGraph, ffi};
use graph::locks::WriteEscalation;
use parking_lot::{ArcRwLockReadGuard, ArcRwLockWriteGuard, RawRwLock, RwLock};
use redis_module::raw;
use std::{cell::Cell, cell::RefCell, marker::PhantomData, ptr::NonNull, sync::Arc};

/// Which lock(s) a query holds.
enum Mode {
    /// Per-graph read lock only; no GIL.
    Reader(ArcRwLockReadGuard<RawRwLock, ThreadedGraph>),
    /// GIL + per-graph write lock. Field order is load-bearing: fields drop in
    /// declaration order, releasing the write lock before the GIL — the reverse of
    /// acquisition. `_gil` is never read, only held for its `Drop`.
    Writer {
        guard: ArcRwLockWriteGuard<RawRwLock, ThreadedGraph>,
        _gil: Gil,
    },
}

/// The locks held by one query, on one thread. Not `Send`, because the GIL must be
/// released by the thread that took it.
pub struct QuerySession {
    graph: Arc<RwLock<ThreadedGraph>>,
    /// `RefCell` because escalation goes through `&self` — operators hold the
    /// session as `&dyn WriteEscalation`. Every borrow is confined to one method, so
    /// the only misuse is escalating inside a [`Self::with_graph`] closure, which
    /// panics loudly instead of dangling.
    mode: RefCell<Option<Mode>>,
}

impl QuerySession {
    /// Enter **reader** mode: take the per-graph read lock.
    pub fn begin(graph: &Arc<RwLock<ThreadedGraph>>) -> Self {
        Self {
            graph: Arc::clone(graph),
            mode: RefCell::new(Some(Mode::Reader(RwLock::read_arc(graph)))),
        }
    }

    /// Enter **writer** mode immediately — GIL, then the per-graph write lock — for
    /// callers that write for the whole query and have nothing to escalate. On the
    /// main thread the GIL acquire is the no-op it must be.
    pub fn begin_writer(graph: &Arc<RwLock<ThreadedGraph>>) -> Self {
        let gil = Gil::acquire();
        Self {
            graph: Arc::clone(graph),
            mode: RefCell::new(Some(Mode::Writer {
                guard: RwLock::write_arc(graph),
                _gil: gil,
            })),
        }
    }

    /// Release the locks now, before the session itself goes out of scope.
    ///
    /// For the one case where the two must differ: a write's reply is serialized from
    /// the runtime, which borrows this session for its whole life, so the session
    /// cannot be dropped before the reply — but the GIL and the write lock should be.
    /// Idempotent, and `Drop` still covers every other path.
    ///
    /// Afterwards this session holds nothing, so [`Self::with_graph`] would panic:
    /// call it last.
    pub fn release_locks(&self) {
        *self.mode.borrow_mut() = None;
    }

    /// The graph this session locks, as an owned handle.
    ///
    /// Lets a caller reach the graph *after* dropping the session — the slow log is
    /// written once the writer window is closed, so it takes its own brief read lock.
    pub fn graph_arc(&self) -> Arc<RwLock<ThreadedGraph>> {
        Arc::clone(&self.graph)
    }

    /// Run `f` against the locked graph in a **short-lived** borrow.
    ///
    /// Short-lived is the point: escalation *replaces* the read guard with a write
    /// guard, so a `&ThreadedGraph` outliving it would be invalid. `f` must not
    /// escalate.
    pub fn with_graph<R>(
        &self,
        f: impl FnOnce(&ThreadedGraph) -> R,
    ) -> R {
        // Both arms hold a guard that derefs to `&ThreadedGraph`; only the variant
        // differs.
        match self.mode.borrow().as_ref().expect("session holds no lock") {
            Mode::Reader(guard) => f(guard),
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

    /// Become a writer, if not one already.
    ///
    /// Idempotent. The brief window holding no per-graph lock is safe: this query
    /// owns the single MVCC write slot for its whole lifetime, so no other writer
    /// can commit, and it has not yet touched the shared index — a reader entering
    /// the window sees a consistent older state.
    pub fn upgrade_to_write(&self) -> Result<(), String> {
        if self.escalate() {
            // The key could have been deleted in the window where we held no
            // per-graph lock, so re-verify the graph is still reachable before
            // mutating it — as C does by re-opening the key under WRITE. The locks
            // stay held; the caller aborts the query and `Drop` releases them.
            if !crate::graph_core::graph_is_registered(&self.graph) {
                return Err(
                    "graph was deleted or replaced while the query was running, aborting"
                        .to_string(),
                );
            }
        }
        Ok(())
    }

    /// Release the read lock, take the GIL, take the write lock — in that order.
    /// `false` if this query was already a writer, so there was nothing to do.
    ///
    /// Separate from [`Self::upgrade_to_write`] so the `mode` borrow ends here, by
    /// scope, instead of spanning that method's call out to the graph registry.
    fn escalate(&self) -> bool {
        let mut mode = self.mode.borrow_mut();
        if matches!(*mode, Some(Mode::Writer { .. })) {
            return false;
        }
        // Drop the read lock FIRST: holding it across the GIL acquire below is the
        // #726 inversion, and this assignment is what guarantees we never do.
        *mode = None;

        // GIL, then the write lock — the order every inline main-thread command
        // already has.
        let gil = Gil::acquire();
        let guard = RwLock::write_arc(&self.graph);
        *mode = Some(Mode::Writer { guard, _gil: gil });
        true
    }
}

impl WriteEscalation for QuerySession {
    fn upgrade_to_write(&self) -> Result<(), String> {
        Self::upgrade_to_write(self)
    }
}

impl Drop for QuerySession {
    /// Releases the write lock then the GIL (see [`Mode::Writer`]), or the read lock.
    fn drop(&mut self) {
        *self.mode.borrow_mut() = None;
    }
}

thread_local! {
    /// The thread-safe context whose GIL this thread holds, if any. `None` also means
    /// "not held by us", which is how [`Gil::acquire`] detects a nested acquire.
    static GIL_CTX: Cell<Option<NonNull<raw::RedisModuleCtx>>> = const { Cell::new(None) };
}

/// Take the module GIL for the duration of the returned guard.
///
/// The one entry point for code that needs the GIL outside a query — telemetry's
/// flush thread. Keeping it the *only* other caller is what lets the ordering rule be
/// enforced by construction rather than by convention.
///
/// Bind the guard: dropping it immediately would release the GIL in the same statement
/// that took it.
#[must_use]
pub fn hold_gil() -> impl Drop {
    Gil::acquire()
}

/// RAII guard for the host's global lock: Redis's module GIL.
///
/// `locked` records whether *this* guard is the one that took the GIL, so a nested
/// guard releases nothing.
///
/// Not `Send`: the GIL is released by the thread that took it, which is also what
/// keeps [`QuerySession`] `!Send`.
struct Gil {
    locked: bool,
    _not_send: PhantomData<*const ()>,
}

impl Gil {
    #[must_use]
    /// Take the GIL, unless this thread already holds it — the module GIL is not
    /// recursive, so re-acquiring would self-deadlock. Two ways it can already be
    /// held: the main thread holds it implicitly for the whole command callback, and a
    /// nested acquire finds our own context in [`GIL_CTX`].
    fn acquire() -> Self {
        if graph::thread_id::is_main_thread() || GIL_CTX.get().is_some() {
            return Self {
                locked: false,
                _not_send: PhantomData,
            };
        }
        // SAFETY: a null blocked-client yields a detached context, which is what we
        // want — we need the GIL, not a client to reply to. It is ours, so `Drop`
        // frees it.
        let ctx = unsafe { ffi::get_thread_safe_context(std::ptr::null_mut()) };
        let ctx = NonNull::new(ctx).expect("RedisModule_GetThreadSafeContext returned null");
        // SAFETY: `ctx` was just created and is owned by this thread until `Drop`.
        unsafe { lock_gil(ctx.as_ptr()) };
        GIL_CTX.set(Some(ctx));
        Self {
            locked: true,
            _not_send: PhantomData,
        }
    }
}

impl Drop for Gil {
    fn drop(&mut self) {
        if !self.locked {
            return; // a nested guard, or the main thread's implicit hold
        }
        // `take` rather than `get`: the context leaves the thread-local as it is
        // released, so no later guard can release it again. Only the guard that locked
        // gets here, so `None` is unreachable — but this runs from a `Drop`, where a
        // panic aborts, so leak rather than assert.
        if let Some(ctx) = GIL_CTX.take() {
            // SAFETY: `ctx` is the context this thread locked in `acquire`, taken out
            // of the thread-local so it is released exactly once.
            unsafe { release_gil(ctx.as_ptr()) };
        }
    }
}

/// Acquire the module GIL through `ctx`. Private on purpose, so [`Gil`] is the only
/// way to take the GIL anywhere in the process.
///
/// # Safety
/// `ctx` must be a valid thread-safe context this thread does not already hold.
unsafe fn lock_gil(ctx: *mut raw::RedisModuleCtx) {
    let f = unsafe { raw::RedisModule_ThreadSafeContextLock }.expect("missing Redis API symbol");
    unsafe { f(ctx) };
}

/// Release the GIL taken by [`lock_gil`] and free the context.
///
/// Resolves the symbols rather than `expect`ing them: this runs from a `Drop`, where
/// a panic aborts, and leaking a context beats killing the server.
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
