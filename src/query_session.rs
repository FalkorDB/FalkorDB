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
//! [`hold_gil`] (telemetry's flush thread, and the native index builder's host). That
//! is what makes the ordering rule enforceable rather than merely documented — the
//! `graph` crate cannot reach it at all.
//!
//! ## Waiting for the write lock
//!
//! A writer waits for the write lock until every read already running on the graph
//! has ended, and it waits holding the GIL, because GIL-then-write-lock is the order
//! the inline main-thread commands have (#726). Waiting in one go would keep the
//! server from answering anything, PING included, for as long as the longest of
//! those reads (#3613). So a writer off the main thread waits in slices of
//! `WRITE_LOCK_POLL_INTERVAL` ms and gives the GIL back between them
//! ([`acquire_write`]), and readers that would slip in during those hand-offs wait
//! at the graph's [`WriterGate`] instead. This is C's `QueryCtx_AcquireWriteLock`
//! (#3422, #3685).

use crate::config::WRITE_LOCK_POLL_INTERVAL;
use crate::graph_core::{ThreadedGraph, ffi, graph_is_registered};
use graph::graph::{graph::Graph, mvcc_graph::CommitError};
use graph::locks::WriteEscalation;
use graph::thread_id::{is_main_thread, is_pool_thread};
use parking_lot::{ArcRwLockReadGuard, ArcRwLockWriteGuard, RawRwLock, RwLock, RwLockWriteGuard};
use std::{
    cell::Cell,
    cell::RefCell,
    marker::PhantomData,
    ptr::NonNull,
    sync::{Arc, atomic::Ordering},
    time::Duration,
};
use valkey_module::{Context, ContextFlags, raw};

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
    /// Another writer committed while this one was still matching as a reader,
    /// so the version it built no longer extends the committed graph. Nothing
    /// was mutated or published yet: the caller discards it and runs the write
    /// again from the start.
    #[error("Write query aborted: the graph was written while this query was matching")]
    Stale,
    /// The version this write produced failed `Graph::validate`, so it was not
    /// published. An engine fault rather than a reason to retry.
    #[error("{0}")]
    Invalid(String),
}

impl From<CommitError> for WriteAbort {
    fn from(e: CommitError) -> Self {
        match e {
            CommitError::Stale => Self::Stale,
            CommitError::Invalid(e) => Self::Invalid(e.to_string()),
        }
    }
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
    /// Where this session's write version stands — see [`Self::write_version`].
    version: RefCell<Version>,
}

/// Where a session's write version stands against the published graph.
///
/// A write that starts as a reader forks while it matches and checks the fork
/// when it escalates. Forking takes nothing, so an inline writer, which holds
/// the GIL and the write lock from the start, is never held up by a query that
/// is still matching (#3093): if it commits meanwhile, that query finds its
/// fork stale instead, and re-runs. A fork made under the write lock is current
/// from the start, since nothing else can commit before its own session does.
enum Version {
    /// No write version.
    None,
    /// Forked as a reader; the fork is checked when the session escalates.
    Unchecked,
    /// Extends the published graph, and stays so: this session holds the
    /// write lock until it commits.
    Current,
    /// Another writer committed after the fork was made; the fork is gone.
    Stale,
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
        let guard = read_past_gate(graph);
        Self {
            graph: Arc::clone(graph),
            mode: RefCell::new(Some(Mode::Reader(guard))),
            facts,
            version: RefCell::new(Version::None),
        }
    }

    /// Enter **writer** mode immediately — GIL, then the per-graph write lock — for
    /// callers that write for the whole query and have nothing to escalate. On the
    /// main thread the GIL acquire is the no-op it must be.
    ///
    /// Main thread only: it waits for the write lock in one go, which off the main
    /// thread would hold the GIL for the whole wait. A worker becomes a writer
    /// through [`Self::upgrade_to_write`] instead.
    pub fn begin_writer(graph: &Arc<RwLock<ThreadedGraph>>) -> Self {
        debug_assert!(
            holds_gil(),
            "begin_writer holds the GIL while it waits for the write lock: main thread only"
        );
        let gil = Gil::acquire();
        let guard = RwLock::write_arc(graph);
        Self {
            graph: Arc::clone(graph),
            mode: RefCell::new(Some(Mode::Writer { guard, _gil: gil })),
            // Moot: already a writer, so `escalate` never runs and this is never read.
            // These are the inline main-thread paths anyway, where Redis's own dispatch
            // gated the command and no pause or role-change window can open mid-command.
            facts: WriteFacts::CLIENT,
            version: RefCell::new(Version::None),
        }
    }

    /// The facts this session was begun with, to begin another for the same
    /// write.
    pub const fn facts(&self) -> WriteFacts {
        self.facts
    }

    /// The private version a write builds into: a fork of the published graph.
    ///
    /// In writer mode it is current now, and stays so. In reader mode it is
    /// checked when the query escalates ([`Self::upgrade_to_write`]); if
    /// another writer committed in between, escalation fails with
    /// [`WriteAbort::Stale`] and [`Self::is_stale`] reports it.
    ///
    /// The caller owns the fork, and hands it to [`Self::upgrade_to_write`]
    /// to be checked. Nothing has to be released afterwards: an uncommitted
    /// version is just dropped.
    pub fn write_version(&self) -> Graph {
        debug_assert!(
            matches!(*self.version.borrow(), Version::None),
            "one write version per session"
        );
        let fork = self.with_graph(|tg| tg.graph.fork());
        *self.version.borrow_mut() = if self.is_writer() {
            Version::Current
        } else {
            Version::Unchecked
        };
        fork
    }

    /// Whether the write lost its version to a writer that committed while it
    /// was matching, so it must be run again from the start.
    pub fn is_stale(&self) -> bool {
        matches!(*self.version.borrow(), Version::Stale)
    }

    /// Whether this session's version is known to extend the published graph
    /// under the write lock — the only state in which a `Commit` can have
    /// published index documents from it. Still matching, stale, or refused at
    /// escalation, it published none.
    pub fn version_is_current(&self) -> bool {
        matches!(*self.version.borrow(), Version::Current)
    }

    fn is_writer(&self) -> bool {
        matches!(*self.mode.borrow(), Some(Mode::Writer { .. }))
    }

    /// Check the fork made in reader mode against the published graph, and
    /// rebase it onto the indexer batches published since it was made. Writer
    /// mode only, so nothing can commit between this check and this session's
    /// own commit.
    ///
    /// # Panics
    ///
    /// If the session forked as a reader and `fork` is `None`.
    fn check_fork(
        &self,
        fork: Option<&mut Graph>,
    ) -> Result<(), WriteAbort> {
        let mut state = self.version.borrow_mut();
        if !matches!(*state, Version::Unchecked) {
            return Ok(());
        }
        let fork = fork.expect("a session that forked as a reader escalates with its fork");
        let current = self.with_graph(|tg| {
            if tg.graph.is_stale(fork) {
                return false;
            }
            // An indexer batch changes no data, so it leaves the fork current,
            // but the fork takes its index progress now: the index documents
            // this query publishes, and the indexes it reads from here on,
            // have it. Without the native index there are no batches.
            #[cfg(feature = "index-falkordb")]
            tg.graph.rebase_onto_indexer_batches(fork);
            true
        });
        if !current {
            *state = Version::Stale;
            return Err(WriteAbort::Stale);
        }
        *state = Version::Current;
        Ok(())
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
    /// Idempotent. While it waits for the write lock the query holds no per-graph
    /// lock, and between slices not the GIL either, so the wait can be as long as
    /// the longest read already running on the graph. That is safe: the query has
    /// not yet touched the shared index, so a reader sees a consistent older
    /// state; a writer that commits meanwhile makes this query's version stale,
    /// which the check below detects before anything is published; and the key,
    /// the role and the pause state are re-checked under the same GIL hold the
    /// lock is finally taken under ([`acquire_write`]).
    ///
    /// `fork` is the version from [`Self::write_version`], if the session took
    /// one: it is checked against the published graph here.
    ///
    /// Refused before the lock is taken, the session is a reader again, so
    /// calling this again runs every check again. Refused by the fork check, it
    /// is a writer holding both locks, and the caller drops it.
    pub fn upgrade_to_write(
        &self,
        fork: Option<&mut Graph>,
    ) -> Result<(), WriteAbort> {
        if self.escalate()? {
            // Only now, under the write lock, find out whether an inline writer
            // committed while this query waited, making the version built while
            // matching stale. Here rather than at commit: the caller publishes
            // index documents next, which dropping a stale fork cannot take back.
            self.check_fork(fork)?;
        }
        Ok(())
    }

    /// Release the read lock, then take the GIL and the write lock through
    /// [`acquire_write`], with [`authorize_write`] as the check before each try.
    /// `Ok(false)` if this query was already a writer, so there was nothing to do.
    ///
    /// On a refusal the read lock is taken back, as C's `QueryCtx_AcquireWriteLock`
    /// does, so the session still holds what [`Self::with_graph`] needs. Directly, as
    /// in C, not through the gate: every caller drops the session next, and a refused
    /// write should not wait behind another writer's gate just to report the refusal.
    fn escalate(&self) -> Result<bool, WriteAbort> {
        let mut mode = self.mode.borrow_mut();
        if matches!(*mode, Some(Mode::Writer { .. })) {
            return Ok(false);
        }
        let Some(Mode::Reader(guard)) = mode.take() else {
            unreachable!("session holds no lock");
        };
        // The wait below holds no lock to reach the gate through, so it keeps its own
        // handle: the one `Arc` clone a write pays for the gate.
        let gate = Arc::clone(&guard.writer_gate);
        // Drop the read lock FIRST: holding it across the GIL acquire below is the
        // #726 inversion, and this drop is what guarantees we never do.
        drop(guard);

        match acquire_write(&self.graph, &gate, || {
            authorize_write(&self.graph, self.facts)
        }) {
            Ok((guard, gil)) => {
                *mode = Some(Mode::Writer { guard, _gil: gil });
                Ok(true)
            }
            Err(abort) => {
                *mode = Some(Mode::Reader(RwLock::read_arc(&self.graph)));
                Err(abort)
            }
        }
    }
}

/// Missed slices after which a waiting writer closes the gate. C's `GATE_THRESHOLD`.
const GATE_AFTER_SLICES: u32 = 2;

/// How long a writer that missed a slice sleeps, holding nothing, before it tries
/// again. The sleep is the hand-off: without it this thread would usually take the
/// GIL straight back before the main thread, blocked on it, could. C's 1 ms
/// `nanosleep`.
const GIL_HANDOFF: Duration = Duration::from_millis(1);

/// Take the GIL and then `graph`'s write lock, holding the GIL for at most one
/// slice of the wait at a time.
///
/// Each attempt takes the GIL, runs `check_before_try` under it, and waits up to
/// `WRITE_LOCK_POLL_INTERVAL` ms for the lock. The check runs on every attempt
/// because the GIL was given back since the last one, so whatever it checks may
/// have changed; its error ends the wait. On a miss the GIL is given back and the
/// thread sleeps [`GIL_HANDOFF`] so the main thread can serve commands. On success
/// the GIL has been held since the check, so it still holds, and both locks are
/// returned for the session to keep.
///
/// What the writer waits for is the reads already running. New ones are held back
/// by the lock itself during a slice, but when a slice ends without the lock,
/// parking_lot wakes every reader it held back, and a steady flow of reads would
/// keep the writer out indefinitely. So after [`GATE_AFTER_SLICES`] misses the
/// writer closes the [`WriterGate`], and the gate opens again when the writer has
/// the lock or gives up. Closing it only after a few misses keeps it out of the
/// common short wait, as C's #3685 does.
///
/// A thread that already holds the GIL — the main thread, or one inside
/// [`hold_gil`] — cannot give it back mid-wait, so it waits in one go. So does every
/// writer when the interval is 0, which is C's way back to that behaviour.
fn acquire_write(
    graph: &Arc<RwLock<ThreadedGraph>>,
    gate: &WriterGate,
    mut check_before_try: impl FnMut() -> Result<(), WriteAbort>,
) -> Result<(ArcRwLockWriteGuard<RawRwLock, ThreadedGraph>, Gil), WriteAbort> {
    let interval = WRITE_LOCK_POLL_INTERVAL.load(Ordering::Relaxed);
    if interval <= 0 || holds_gil() {
        let gil = Gil::acquire();
        check_before_try()?;
        return Ok((RwLock::write_arc(graph), gil));
    }

    let slice = Duration::from_millis(interval as u64);
    let mut missed = 0;
    let mut closed = None;
    loop {
        let gil = Gil::acquire();
        check_before_try()?;
        if let Some(guard) = RwLock::try_write_arc_for(graph, slice) {
            // `closed` drops on the way out: the gate opens with the lock taken,
            // so the readers it held go on to queue behind this write.
            return Ok((guard, gil));
        }
        drop(gil);
        missed += 1;
        if closed.is_none() && missed >= GATE_AFTER_SLICES {
            // Holds no lock here, so it can wait for another writer to open the
            // gate.
            closed = Some(gate.close());
        }
        std::thread::sleep(GIL_HANDOFF);
    }
}

/// What a write must still be allowed to do, checked under the GIL it takes the
/// write lock under.
fn authorize_write(
    graph: &Arc<RwLock<ThreadedGraph>>,
    facts: WriteFacts,
) -> Result<(), WriteAbort> {
    // The key could have been deleted or replaced while this query held no
    // per-graph lock, so re-verify the graph is still reachable before mutating
    // it — as C does by re-opening the key under WRITE.
    if !graph_is_registered(graph) {
        return Err(WriteAbort::GraphUnregistered);
    }
    // The other half of what C re-validates here: this instance may no longer be
    // allowed to originate a write at all.
    reauthorize_write(facts)
}

/// Take `graph`'s read lock as a reader, first waiting out a writer that closed the
/// gate.
///
/// The gate is checked after the lock is taken rather than before, because it lives
/// inside the lock: parking_lot's `Arc` guards need the `RwLock` to be all of the
/// `Arc`, so there is nowhere outside it to keep a gate. A reader that finds the
/// gate closed lets go of the lock at once and waits, holding nothing, so the
/// writer's next slice does not wait for it. With no writer kept waiting, the cost
/// is an atomic load; only a reader that waits clones the gate's `Arc`, to hold it
/// once it has let go of the lock.
///
/// Only a thread that [`may_wait_at_gate`] waits; any other takes the lock as it
/// finds it.
fn read_past_gate(
    graph: &Arc<RwLock<ThreadedGraph>>
) -> ArcRwLockReadGuard<RawRwLock, ThreadedGraph> {
    loop {
        let guard = RwLock::read_arc(graph);
        if !guard.writer_gate.is_closed() || !may_wait_at_gate() {
            return guard;
        }
        let gate = Arc::clone(&guard.writer_gate);
        drop(guard);
        gate.wait_until_open();
    }
}

/// Whether this thread can wait at a writer gate without stopping the writer: a
/// pool worker that does not hold the GIL.
///
/// The writer needs the GIL for every slice, so a reader that waits holding it
/// waits forever. A pool worker holds the GIL only inside a [`Gil`] of its own,
/// which [`holds_gil`] sees. Any other thread may hold it unseen: the main thread
/// does for every command, and another module's thread can take it and call a
/// graph command, which then runs on that thread. Such a call can read; a write
/// there reaches [`Gil::acquire`], which cannot see that GIL either. C, likewise,
/// gates only the queries its reader pool runs.
fn may_wait_at_gate() -> bool {
    is_pool_thread() && !holds_gil()
}

/// Where a graph's readers wait while a writer is being kept from its write lock.
///
/// A lock of its own. A writer that has missed [`GATE_AFTER_SLICES`] slices
/// ([`acquire_write`]) closes the gate by holding its write guard, and drops it once
/// it has the graph's lock or gives up; being a guard, it cannot be left shut by an
/// early return or a panic. A reader waits at the gate by taking a read guard, which
/// it gets once no writer holds the gate or is queued for it ahead of the reader.
///
/// Two writers can wait on the same graph at once: the write queue and a
/// constraint settling, say. Only one holds the gate. The other blocks in
/// [`Self::close`], holding no lock, until the first lets go, and closes the gate
/// again as soon as it runs; the readers queued behind it wait for it too.
#[derive(Default)]
pub struct WriterGate(RwLock<()>);

impl WriterGate {
    /// One atomic load: what a reader pays when no writer is being kept waiting.
    fn is_closed(&self) -> bool {
        self.0.is_locked_exclusive()
    }

    fn close(&self) -> RwLockWriteGuard<'_, ()> {
        self.0.write()
    }

    fn wait_until_open(&self) {
        drop(self.0.read());
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
    fn upgrade_to_write(
        &self,
        fork: &mut Graph,
    ) -> Result<(), String> {
        // `graph::locks::WriteEscalation` is defined in the `graph` crate, which knows
        // nothing about Redis roles or pauses, so its contract stays a plain message and
        // the typed reason is flattened here at the boundary.
        Self::upgrade_to_write(self, Some(fork)).map_err(|e| e.to_string())
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
/// flush thread, `GRAPH.BULK` freeing its arguments and cleaning up off the main thread
/// (`commands::bulk_insert`), and the native index builder publishing a batch
/// (`graph_core::RedisBuildHost`). Keeping these the *only* other callers is what lets the
/// ordering rule be enforced by construction rather than by convention.
///
/// Bind the guard: dropping it immediately would release the GIL in the same statement
/// that took it.
#[must_use]
pub fn hold_gil() -> impl Drop {
    Gil::acquire()
}

/// Whether this thread holds the GIL. Two ways it can: the main thread holds it
/// implicitly for the whole command callback, and any other thread holds it while
/// its own context is in [`GIL_CTX`].
fn holds_gil() -> bool {
    is_main_thread() || GIL_CTX.get().is_some()
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
    /// Take the GIL, unless this thread already holds it ([`holds_gil`]) — the module
    /// GIL is not recursive, so re-acquiring would self-deadlock.
    fn acquire() -> Self {
        if holds_gil() {
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
