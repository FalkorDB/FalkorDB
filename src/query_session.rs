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
use redis_module::{Context, ContextFlags, raw};
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

/// Why an escalation to writer mode was refused.
///
/// The first two `Display` strings are byte-identical to C's `EMSG_REPLICA_TRAFFIC_PAUSED`
/// and `EMSG_NOT_MASTER` (`src/errors/error_msgs.h`, PR #2372), so a client or a test can
/// match either engine. Keeping them in one place is the point of the type: they are a
/// compatibility contract, not incidental text, and they were previously written inline at
/// their single use site each.
#[derive(Debug, thiserror::Error)]
pub enum WriteAbort {
    /// A `CLIENT PAUSE` / `FAILOVER` window is open. Committing would propagate inside it.
    #[error("Write query aborted: replica traffic is currently paused")]
    ReplicaTrafficPaused,
    /// This instance is a read-only replica now, whatever it was at admission.
    #[error("Write query aborted: this instance is not a master")]
    NotAMaster,
    /// The graph key was deleted or replaced while no per-graph lock was held.
    #[error("graph was deleted or replaced while the query was running, aborting")]
    GraphUnregistered,
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
    /// What this write does and where it came from — see [`WriteFacts`]. Moot unless
    /// the session escalates, so it is unread for a reader and for
    /// [`Self::begin_writer`].
    facts: WriteFacts,
}

/// The two facts about a write that decide which re-checks apply when it escalates.
///
/// Both are properties of the *write*, not of the checks, so a call site can answer them
/// without knowing what the guard does with them — and a path that later starts
/// replicating flips the field named after replication.
///
/// They have to be carried rather than read at the check: escalation runs on a worker
/// whose only context is the detached one [`Gil::acquire`] made, which has no client
/// behind it, so `REPLICATED` always reads false there. C reads it straight off the
/// command's own context instead.
///
/// `docs/write-authorization.md` has the states, the coverage table and the call sites.
#[derive(Clone, Copy, Debug)]
pub struct WriteFacts {
    /// It can reach the replication stream from this thread, so a replica-pause window
    /// open at commit time would be propagated into.
    pub replicates: bool,
    /// This instance decided this write, rather than replaying one decided elsewhere.
    /// Only our own decisions require that we are still a writable master; rejecting a
    /// replayed one would diverge us from our master.
    pub originated_here: bool,
}

impl WriteFacts {
    /// A client's own write, which will replicate: both re-checks apply. The default,
    /// and the right answer for every path that has no reason to differ.
    pub const CLIENT: Self = Self {
        replicates: true,
        originated_here: true,
    };
}

impl QuerySession {
    /// Enter **reader** mode: take the per-graph read lock.
    ///
    /// Assumes [`WriteFacts::CLIENT`] — if this session escalates, the write is
    /// re-authorized against live server state. Use [`Self::begin_with`] for a write
    /// that differs on either fact.
    pub fn begin(graph: &Arc<RwLock<ThreadedGraph>>) -> Self {
        Self::begin_with(graph, WriteFacts::CLIENT)
    }

    /// Enter **reader** mode for a write that is not a plain client write — see
    /// [`WriteFacts`] for the two questions and `docs/write-authorization.md` for how
    /// each caller answers them.
    pub fn begin_with(
        graph: &Arc<RwLock<ThreadedGraph>>,
        facts: WriteFacts,
    ) -> Self {
        Self {
            graph: Arc::clone(graph),
            mode: RefCell::new(Some(Mode::Reader(RwLock::read_arc(graph)))),
            facts,
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
            // Moot: already a writer, so `escalate` never runs and this is never read.
            // These are the inline main-thread paths anyway, where Redis's own dispatch
            // gated the command and no pause or role-change window can open mid-command.
            facts: WriteFacts::CLIENT,
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
    pub fn upgrade_to_write(&self) -> Result<(), WriteAbort> {
        if self.escalate() {
            // The key could have been deleted in the window where we held no
            // per-graph lock, so re-verify the graph is still reachable before
            // mutating it — as C does by re-opening the key under WRITE. The locks
            // stay held; the caller aborts the query and `Drop` releases them.
            if !crate::graph_core::graph_is_registered(&self.graph) {
                return Err(WriteAbort::GraphUnregistered);
            }
            // Same window, the other half of what C re-validates here: this instance
            // may no longer be allowed to originate a write at all.
            reauthorize_write(self.facts)?;
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

/// Re-check, under the GIL, that this instance may still originate a write.
///
/// A write is admitted on the Redis main thread, where the `write` command flag makes
/// Redis's own dispatch postpone it while `CLIENT PAUSE ... WRITE` is in effect and
/// reject it on a read-only replica. But it mutates and replicates much later, from a
/// worker thread — so a `FAILOVER` / `CLIENT PAUSE` window can have opened, or this
/// instance can have been demoted, in between. Replicating then trips Redis's
/// `propagateNow()` invariant (`server.c`,
/// `!(isPausedActions(PAUSE_ACTION_REPLICA) && !server.client_pause_in_transaction)`)
/// and kills the master — the crash reported in #2359.
///
/// Holding the GIL is what makes this race-free: neither the pause state nor the role
/// can change until it is released. Mirrors C's `QueryCtx_AcquireWriteLock`
/// (`src/query_ctx.c` on `master`, PR #2372); the first two messages are byte-identical
/// to its `EMSG_REPLICA_TRAFFIC_PAUSED` / `EMSG_NOT_MASTER` so clients and tests can
/// match either engine.
fn reauthorize_write(facts: WriteFacts) -> Result<(), WriteAbort> {
    // Off the main thread the GIL is held through the detached context `Gil::acquire`
    // just created. On the main thread there is none — and nothing to re-check either,
    // since Redis gated the command at dispatch and no window can open mid-command.
    let Some(ctx) = GIL_CTX.get() else {
        return Ok(());
    };
    // SAFETY: `ctx` is the context this thread locked in `Gil::acquire` and still
    // holds. `Context` is a borrowing wrapper with no `Drop`, so it frees nothing —
    // the same construction `telemetry`'s flush thread uses under `hold_gil`.
    let ctx = Context::new(ctx.as_ptr());
    let flags = ctx.get_flags();

    // `LOADING` cannot reach here: `dispatch::must_run_inline` contains it, so a command
    // replaying from AOF/RDB runs inline and never escalates. Assert rather than bypass,
    // so narrowing that predicate surfaces here instead of silently changing which writes
    // get authorized.
    debug_assert!(
        !flags.contains(ContextFlags::LOADING),
        "escalated with LOADING set: must_run_inline no longer routes replay inline"
    );

    if facts.replicates && ctx.avoid_replication_traffic() {
        return Err(WriteAbort::ReplicaTrafficPaused);
    }
    // `READONLY`, not `MASTER`: a replica running `replica-read-only no` accepts writes,
    // and rejecting them there would break replica-divergence testing. C corrected this
    // the same way in PR #2372.
    //
    // No check for a role that flipped away and back (master -> replica -> master)
    // between admission and here: the only such flip that matters resynced from the new
    // master, which frees and re-registers the graph key, so `graph_is_registered` above
    // has already aborted this write. A flip that resynced nothing left the data
    // unchanged and the write is still valid.
    if facts.originated_here && flags.contains(ContextFlags::READONLY) {
        return Err(WriteAbort::NotAMaster);
    }
    Ok(())
}

impl WriteEscalation for QuerySession {
    fn upgrade_to_write(&self) -> Result<(), String> {
        // `graph::locks::WriteEscalation` is defined in the `graph` crate, which knows
        // nothing about Redis roles or pauses, so its contract stays a plain message and
        // the typed reason is flattened here at the boundary.
        Self::upgrade_to_write(self).map_err(|e| e.to_string())
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
/// `need_release` records whether *this* guard is the one that took the GIL — a
/// nested guard, or one on the main thread, releases nothing.
///
/// Not `Send`: the GIL is released by the thread that took it, which is also what
/// keeps [`QuerySession`] `!Send`.
struct Gil {
    need_release: bool,
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
                need_release: false,
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
            need_release: true,
            _not_send: PhantomData,
        }
    }
}

/// The context this thread locked the GIL through.
///
/// Borrowed, never owned: it is valid only while a [`Gil`] guard is alive on
/// this thread, must not be freed, and a `Context` built over it must not be
/// treated as owning — `Context` has no `Drop`, so nothing in the type says so.
///
/// Exists so a background thread can replicate **inside** the GIL hold it
/// already has. Taking a second context and locking again is what re-crashed
/// the master in #2371: `RM_ThreadSafeContextUnlock` runs
/// `postExecutionUnitOperations`, which calls `propagateNow`, and the pause
/// check that makes escalation safe is only sound while the GIL is held
/// continuously from that check through the commit and the replicate.
///
/// # Panics
///
/// If this thread holds no GIL guard. That is a caller bug rather than a state
/// to branch on, and it used to be one: the single caller tested the `Option`
/// and skipped replicating when it was `None`, which would have left the
/// replica's constraint UNDER CONSTRUCTION forever with nothing logged. There
/// is no recovery from a missing context at that point, so failing loudly beats
/// a branch that silently drops a replicated write.
///
/// [`Gil::acquire`] records the context only when it actually locks — the main
/// thread's implicit hold sets nothing — so the precondition is "a worker
/// thread that has escalated", which is what `QuerySession::upgrade_to_write`
/// establishes.
#[must_use]
pub(crate) fn gil_context() -> NonNull<raw::RedisModuleCtx> {
    GIL_CTX
        .get()
        .expect("gil_context requires a GIL guard held by this thread")
}

impl Drop for Gil {
    fn drop(&mut self) {
        if !self.need_release {
            return; // a nested guard, or the main thread's implicit hold
        }
        // `take` rather than `get`: the context leaves the thread-local as it is
        // released, so no later guard can release it again. Only the guard that took
        // the GIL gets here, so `None` is unreachable — but this runs from a `Drop`,
        // where a panic aborts, so leak rather than assert.
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
