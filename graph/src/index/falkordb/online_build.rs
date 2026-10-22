//! The online index build's driver: when build steps run, and how each one commits.
//!
//! A build step reads one batch of a `Building` column (`graph_writes::read_build_batch`). This
//! module runs those steps on a pool of its own — [`start`]'s workers, which serve no queries — one
//! turn at a time, for every graph that has `Building` columns. Nothing here knows about Redis: a
//! host registers a [`BuildHost`] for each graph ([`MvccGraph::set_build_host`]), which is how the
//! builder reads the graph and how it publishes as a writer. The Redis module's host takes the GIL
//! and the graph's write lock to publish; an embedded host would take its own lock.
//!
//! **A turn never makes a writer run again.** Holding no lock, it takes one build step on a new
//! version of the published graph (`Graph::step_index_build`). Then it publishes through the host
//! ([`MvccGraph::commit_index_build`]): if nothing has committed since, that version is published
//! as it is, so the step never holds the host's locks. If a writer committed meanwhile, the step
//! is taken again, under those locks, on a new version of the writer's, reading the values it
//! holds, and that one is published. A writer whose fork the commit makes stale merges the other
//! way when it claims (`Graph::merge_build_progress`). The turn holds the versions a commit leaves
//! behind until the host lets go of its locks, so freeing them never holds those.
//!
//! **When the builder runs.** [`MvccGraph::commit`] queues its graph when the version it publishes
//! starts a build — it has a `Building` column, and the version it replaces has none — which covers
//! every way a `CREATE INDEX` can arrive. Each turn is a task on the builder's own [`ThreadPool`],
//! whose workers serve no queries. The pool only runs tasks; a turn decides what comes next: it
//! submits the graph's next turn, until the graph has no `Building` column left, so the commits
//! made meanwhile need not queue it. A graph is therefore queued once per build and has one turn
//! at a time, unless every building column is dropped and a new one created while a turn runs:
//! that commit starts a build and queues the graph a second time, and two turns at once publish
//! nothing wrong ([`run_turn`]). The work per `CREATE INDEX` is the same however many graphs there
//! are.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::Duration;

use atomic_refcell::AtomicRefCell;

use super::graph_writes::BUILD_STEP_ENTITIES;
use crate::graph::graph::Graph;
use crate::graph::mvcc_graph::{BuildCommit, MvccGraph};
use crate::threadpool::ThreadPool;

/// How the builder reaches one graph. Implemented by the host that owns the graph, and registered
/// with it through [`MvccGraph::set_build_host`].
pub trait BuildHost: Send + Sync {
    /// Run `f` on the graph's coordinator with the host's read access. `false` if the graph is
    /// gone.
    fn read(
        &self,
        f: &mut dyn FnMut(&MvccGraph),
    ) -> bool;

    /// Run `f` with whatever a writer holds to publish a version: exclusive access to the
    /// coordinator, and anything the host must hold while one is published (the Redis host takes
    /// the GIL, then the graph's write lock). `false` if the host refuses — the graph is gone, or
    /// the builder is [`stopping`].
    ///
    /// Must exclude [`read`](Self::read) while `f` runs.
    fn publish(
        &self,
        f: &mut dyn FnMut(&mut MvccGraph),
    ) -> bool;
}

/// What one build turn did.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Turn {
    /// The graph has no `Building` column.
    Idle,
    /// The step was committed; `more` if a column is still `Building`.
    Committed { more: bool },
    /// The host refused: the graph is gone, or the builder is stopping.
    Gone,
    /// The step panicked, or its version was refused.
    Failed,
}

/// One build turn on the graph `host` names, with the default batch size.
///
/// The builder runs at most one turn per graph at a time. Two at once would still publish nothing
/// wrong — the later one's commit finds the earlier one's published, and takes its step again on
/// it — but would read a batch twice.
fn run_turn(host: &dyn BuildHost) -> Turn {
    run_turn_within(host, BUILD_STEP_ENTITIES)
}

/// [`run_turn`] with an explicit batch size, for tests.
pub(crate) fn run_turn_within(
    host: &dyn BuildHost,
    entities: usize,
) -> Turn {
    // One read for the published version to build on. The step needs no lock: a published version
    // never changes, and the step goes into a new version of it that only this turn holds.
    let mut published = None;
    if !host.read(&mut |mvcc| published = Some(mvcc.read())) {
        return Turn::Gone;
    }
    let Some(published) = published else {
        return Turn::Gone;
    };
    let prepared = Arc::new(AtomicRefCell::new(published.borrow().new_version()));
    drop(published);
    if !prepared.borrow_mut().step_index_build(entities) {
        return Turn::Idle;
    }
    // Whoever stops the builder may hold what publishing needs (the Redis shutdown handler holds
    // the GIL) while it waits for this thread.
    if stopping() {
        return Turn::Gone;
    }

    let mut turn = Turn::Gone;
    // Held, with `prepared`, until the host has let go of its locks, so that freeing the version
    // the commit replaces, or `prepared` if a writer made it stale, never happens under them.
    let mut replaced = None;
    let published = host.publish(&mut |mvcc| {
        replaced = Some(mvcc.read());
        let step = |g: &mut Graph| g.step_index_build(entities);
        turn = match mvcc.commit_index_build(&prepared, step) {
            Some(BuildCommit::Prepared | BuildCommit::Redone) => Turn::Committed {
                more: mvcc.read().borrow().falkordb_index.has_building_columns(),
            },
            Some(BuildCommit::Nothing) => Turn::Idle,
            // Writers take the slot only with publish access, so it was free; retried all the
            // same, as a refused version is.
            None => Turn::Failed,
        };
    });
    if published { turn } else { Turn::Gone }
}

// ---- Running the turns ----

/// The builder's pool while it runs, and before [`start`] the graphs a commit queued, which wait
/// for it.
struct Builder {
    pool: Option<Arc<ThreadPool>>,
    waiting: Vec<Arc<dyn BuildHost>>,
}

static BUILDER: Mutex<Builder> = Mutex::new(Builder {
    pool: None,
    waiting: Vec::new(),
});

/// Turns submitted to the pool that have not started yet.
static QUEUED: AtomicUsize = AtomicUsize::new(0);

/// Set while the builder is stopping: a turn that has not started gives up, and one that has
/// gives up rather than publish.
static STOPPING: AtomicBool = AtomicBool::new(false);

/// Whether the builder is stopping. A host that has waited for what publishing needs checks it
/// before publishing: whoever stops the builder may have held that while waiting for the workers.
#[must_use]
pub fn stopping() -> bool {
    STOPPING.load(Ordering::Relaxed)
}

fn builder() -> MutexGuard<'static, Builder> {
    BUILDER.lock().unwrap_or_else(|e| e.into_inner())
}

/// Start the builder's pool with `threads` workers (at least one), and submit the graphs queued
/// before it started. Idempotent while it runs.
pub fn start(threads: usize) {
    let waiting = {
        let mut builder = builder();
        if builder.pool.is_some() {
            return;
        }
        STOPPING.store(false, Ordering::Relaxed);
        builder.pool = Some(Arc::new(ThreadPool::named(
            threads.max(1),
            "falkordb-index-build",
        )));
        std::mem::take(&mut builder.waiting)
    };
    for host in waiting {
        submit(host);
    }
}

/// Stop the builder's pool, waiting up to [`STOP_WAIT`] for its workers to exit
/// ([`ThreadPool::shutdown_within`]). A turn that has not started publishing drops its batch. One
/// waiting for what its host needs to publish may be waiting on the caller — the Redis shutdown
/// handler calls this holding the GIL — so it is left behind; its host gives up once it gets the
/// lock ([`stopping`]). Graphs still queued are dropped from the build for good.
pub fn stop() {
    STOPPING.store(true, Ordering::Relaxed);
    let pool = {
        let mut builder = builder();
        builder.waiting.clear();
        builder.pool.take()
    };
    if let Some(pool) = pool {
        pool.shutdown_within(STOP_WAIT);
    }
    QUEUED.store(0, Ordering::Relaxed);
}

/// How long [`stop`] waits for the workers.
const STOP_WAIT: Duration = Duration::from_secs(2);

/// Graphs waiting for a turn. For tests.
#[must_use]
pub fn pending() -> usize {
    builder().waiting.len() + QUEUED.load(Ordering::Relaxed)
}

/// Queue `host`'s graph for its first turn. Never blocks for long: it is called from
/// [`MvccGraph::commit`], which holds the host's locks.
pub(crate) fn enqueue(host: Arc<dyn BuildHost>) {
    submit(host);
}

/// Submit `host`'s next turn. Before the builder starts, the graph waits for it; once it has
/// stopped, the graph is dropped.
fn submit(host: Arc<dyn BuildHost>) {
    let pool = {
        let mut builder = builder();
        match &builder.pool {
            Some(pool) => Arc::clone(pool),
            None => {
                if !stopping() {
                    builder.waiting.push(host);
                }
                return;
            }
        }
    };
    QUEUED.fetch_add(1, Ordering::Relaxed);
    pool.spawn(move || take_turn(host), None);
}

/// One turn of `host`'s graph, on a builder worker, then the decision on its next one.
fn take_turn(host: Arc<dyn BuildHost>) {
    QUEUED.fetch_sub(1, Ordering::Relaxed);
    if stopping() {
        return;
    }
    let turn = catch_unwind(AssertUnwindSafe(|| run_turn(host.as_ref()))).unwrap_or(Turn::Failed);
    if turn == Turn::Failed {
        eprintln!("index build: a build step failed; it will be retried");
    }
    let Some(wait) = next_turn(&turn) else {
        return;
    };
    // Only after a failure, which is a bug: holding this worker for it costs nothing that
    // matters.
    if !wait.is_zero() {
        thread::sleep(wait);
    }
    submit(host);
}

/// How long to wait before a graph's next turn, after a turn that went `turn`; `None` when the
/// graph is done — a commit that starts another build queues it again.
fn next_turn(turn: &Turn) -> Option<Duration> {
    match turn {
        Turn::Committed { more: true } => Some(Duration::ZERO),
        Turn::Failed => Some(FAILED_RETRY),
        Turn::Committed { more: false } | Turn::Idle | Turn::Gone => None,
    }
}

/// How long a graph waits before retrying after a turn failed: its step panicked, or its version
/// was refused.
const FAILED_RETRY: Duration = Duration::from_secs(1);

#[cfg(test)]
mod tests {
    use super::*;

    /// After a turn that committed with more to build, the next one runs right away; after a
    /// failed one, a second later; after any other, there is none.
    #[test]
    fn a_turn_decides_when_the_next_one_runs() {
        assert_eq!(
            next_turn(&Turn::Committed { more: true }),
            Some(Duration::ZERO)
        );
        assert_eq!(next_turn(&Turn::Failed), Some(FAILED_RETRY));
        for done in [Turn::Committed { more: false }, Turn::Idle, Turn::Gone] {
            assert_eq!(next_turn(&done), None, "{done:?}");
        }
    }
}
