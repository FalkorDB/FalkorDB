//! The online index build's driver: when build steps run, and how each one commits.
//!
//! A build step reads one batch of a `Building` column (`graph_writes::read_build_batch`). This
//! module runs those steps on a pool of its own — [`start`]'s workers, which serve no queries — one
//! turn at a time, for every graph that has `Building` columns. Nothing here knows about Redis: a
//! host registers a [`BuildHost`] for each graph ([`MvccGraph::set_build_host`]), which is how the
//! builder reads the graph and how it publishes as a writer. The Redis module's host takes the GIL
//! and the graph's write lock to publish; an embedded host would take its own lock.
//!
//! **A turn neither makes a writer run again nor loses its batch to one.** Holding no lock, it
//! reads its batch from the published version and merges it into a new version of that one
//! (`Graph::merge_index_build`). Then it publishes through the host
//! ([`MvccGraph::commit_index_build`]): if nothing has committed since, that version is published
//! as it is, so the batch's insert never holds the host's locks; if a writer committed meanwhile,
//! the batch is merged into a new version of the current one instead, skipping the entities
//! written since, and that one is published. A writer whose fork the commit makes stale merges the
//! other way when it claims (`Graph::merge_build_progress`). Only a column dropped or re-created
//! meanwhile costs a turn its batch.
//!
//! **When the builder runs.** [`MvccGraph::commit`] queues its graph when the version it publishes
//! starts a build — it has a `Building` column, and the version it replaces has none — which covers
//! every way a `CREATE INDEX` can arrive. A worker takes the graph with the earliest start time,
//! runs one turn, and queues it again — not before as long as the turn took — until it has no
//! `Building` column left, so the commits made meanwhile need not queue it. A graph is therefore
//! queued once per build and has one turn at a time, unless every building column is dropped and a
//! new one created while a turn runs: that commit starts a build and queues the graph a second
//! time, and two turns at once publish nothing wrong ([`run_turn`]). The work per `CREATE INDEX`
//! is the same however many graphs there are.

use std::cmp::{Ordering as CmpOrdering, Reverse};
use std::collections::BinaryHeap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use atomic_refcell::AtomicRefCell;

use super::graph_writes::{BUILD_STEP_EDGES, BUILD_STEP_NODES, read_build_batch};
use crate::graph::mvcc_graph::{BuildCommit, MvccGraph};

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
    /// The column the batch was read from was dropped or re-created meanwhile; the batch was
    /// dropped.
    Dropped,
    /// The host refused: the graph is gone, or the builder is stopping.
    Gone,
    /// The step panicked, or its version was refused.
    Failed,
}

/// One build turn on the graph `host` names, with the default batch sizes.
///
/// The builder runs at most one turn per graph at a time. Two at once would still publish nothing
/// wrong — the later one's commit finds the earlier one's published, and its merge refuses a batch
/// read before that — but would read the same batch twice.
fn run_turn(host: &dyn BuildHost) -> Turn {
    run_turn_within(host, BUILD_STEP_NODES, BUILD_STEP_EDGES)
}

/// [`run_turn`] with explicit batch sizes, for tests.
pub(crate) fn run_turn_within(
    host: &dyn BuildHost,
    nodes: usize,
    edges: usize,
) -> Turn {
    // One read for the published version to build on. The rest needs no lock: a published version
    // never changes, and the batch goes into a new version of it that only this turn holds.
    let mut published = None;
    if !host.read(&mut |mvcc| published = Some(mvcc.read())) {
        return Turn::Gone;
    }
    let Some(published) = published else {
        return Turn::Gone;
    };
    let Some(mut batch) = read_build_batch(&published.borrow(), nodes, edges) else {
        return Turn::Idle;
    };
    let prepared = Arc::new(AtomicRefCell::new(published.borrow().new_version()));
    drop(published);
    if !prepared.borrow_mut().merge_index_build(&mut batch) {
        // Read from the same version: cannot happen, and harmless if it did.
        return Turn::Dropped;
    }
    // Whoever stops the builder may hold what publishing needs (the Redis shutdown handler holds
    // the GIL) while it waits for this thread.
    if stopping() {
        return Turn::Gone;
    }

    let mut turn = Turn::Gone;
    let mut prepared = Some(prepared);
    let published = host.publish(&mut |mvcc| {
        let prepared = prepared.take().expect("published once");
        turn = match mvcc.commit_index_build(prepared, &mut batch) {
            Some(BuildCommit::Prepared | BuildCommit::Merged) => Turn::Committed {
                more: mvcc.read().borrow().falkordb_index.has_building_columns(),
            },
            Some(BuildCommit::Dropped) => Turn::Dropped,
            // Writers take the slot only with publish access, so it was free; retried all the
            // same, as a refused version is.
            None => Turn::Failed,
        };
    });
    if published { turn } else { Turn::Gone }
}

// ---- The worker pool ----

/// One graph waiting for its next turn.
struct Queued {
    /// When its next turn may start.
    not_before: Instant,
    /// Breaks ties in queue order.
    seq: u64,
    host: Arc<dyn BuildHost>,
}

impl PartialEq for Queued {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        (self.not_before, self.seq) == (other.not_before, other.seq)
    }
}

impl Eq for Queued {}

impl PartialOrd for Queued {
    fn partial_cmp(
        &self,
        other: &Self,
    ) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl Ord for Queued {
    fn cmp(
        &self,
        other: &Self,
    ) -> CmpOrdering {
        (self.not_before, self.seq).cmp(&(other.not_before, other.seq))
    }
}

struct Pool {
    /// Graphs waiting for a turn, earliest start first.
    queue: BinaryHeap<Reverse<Queued>>,
    next_seq: u64,
    workers: Vec<JoinHandle<()>>,
}

static POOL: Mutex<Pool> = Mutex::new(Pool {
    queue: BinaryHeap::new(),
    next_seq: 0,
    workers: Vec::new(),
});

/// Signalled when a graph is queued, and at stop.
static WORK: Condvar = Condvar::new();

/// Set while the pool is stopping: workers exit, and a turn gives up rather than publish.
static STOPPING: AtomicBool = AtomicBool::new(false);

/// Whether the builder is stopping. A host that has waited for what publishing needs checks it
/// before publishing: whoever stops the builder may have held that while waiting for the workers.
#[must_use]
pub fn stopping() -> bool {
    STOPPING.load(Ordering::Relaxed)
}

fn pool() -> MutexGuard<'static, Pool> {
    POOL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Start the pool's `threads` workers (at least one). Idempotent while they run.
pub fn start(threads: usize) {
    let mut pool = pool();
    if !pool.workers.is_empty() {
        return;
    }
    STOPPING.store(false, Ordering::Relaxed);
    for i in 0..threads.max(1) {
        pool.workers.push(
            thread::Builder::new()
                .name(format!("falkordb-index-build-{i}"))
                .spawn(work)
                .expect("spawn an index build worker"),
        );
    }
}

/// Stop the pool's workers, waiting up to [`STOP_WAIT`] for them to exit. A turn that has not
/// started publishing drops its batch. One waiting for what its host needs to publish may be
/// waiting on the caller — the Redis shutdown handler calls this holding the GIL — so it is left
/// behind; its host gives up once it gets the lock ([`stopping`]). Graphs still queued are
/// dropped from the build for good.
pub fn stop() {
    STOPPING.store(true, Ordering::Relaxed);
    let workers = std::mem::take(&mut pool().workers);
    WORK.notify_all();
    let deadline = Instant::now() + STOP_WAIT;
    while Instant::now() < deadline && !workers.iter().all(JoinHandle::is_finished) {
        thread::sleep(Duration::from_millis(1));
    }
    for worker in workers {
        if worker.is_finished() {
            drop(worker.join());
        }
    }
}

/// How long [`stop`] waits for the workers.
const STOP_WAIT: Duration = Duration::from_secs(2);

/// Graphs waiting for a turn. For tests.
#[must_use]
pub fn pending() -> usize {
    pool().queue.len()
}

/// Queue `host`'s graph for its first turn. Never blocks for long: it is called from
/// [`MvccGraph::commit`], which holds the host's locks.
pub(crate) fn enqueue(host: Arc<dyn BuildHost>) {
    requeue(
        Queued {
            not_before: Instant::now(),
            seq: 0,
            host,
        },
        &mut pool(),
    );
}

fn requeue(
    mut graph: Queued,
    pool: &mut Pool,
) {
    graph.seq = pool.next_seq;
    pool.next_seq += 1;
    pool.queue.push(Reverse(graph));
    WORK.notify_one();
}

/// How long a graph waits for its next turn after one that committed or lost: as long as the turn
/// took, so a build takes at most half of one worker's time per graph, and the graph's write lock
/// at most once per that long.
fn pace(turn: Duration) -> Duration {
    turn.max(Duration::from_millis(1))
}

/// How long a graph waits before retrying after a turn failed: its step panicked, or its version
/// was refused.
const FAILED_RETRY: Duration = Duration::from_secs(1);

/// A worker: take the graph whose turn is due first, run it, queue it again if it has more.
fn work() {
    loop {
        let mut graph = {
            let mut pool = pool();
            loop {
                if stopping() {
                    return;
                }
                let now = Instant::now();
                let due = match pool.queue.peek() {
                    Some(Reverse(next)) if next.not_before <= now => None,
                    Some(Reverse(next)) => Some(next.not_before - now),
                    None => {
                        pool = WORK.wait(pool).unwrap_or_else(|e| e.into_inner());
                        continue;
                    }
                };
                match due {
                    None => {
                        let Reverse(graph) = pool.queue.pop().expect("peeked");
                        break graph;
                    }
                    Some(wait) => {
                        pool = WORK
                            .wait_timeout(pool, wait)
                            .unwrap_or_else(|e| e.into_inner())
                            .0;
                    }
                }
            }
        };

        let started = Instant::now();
        let turn = catch_unwind(AssertUnwindSafe(|| run_turn(graph.host.as_ref())))
            .unwrap_or(Turn::Failed);
        let again = match turn {
            Turn::Committed { more: true } | Turn::Dropped => Some(pace(started.elapsed())),
            Turn::Failed => {
                eprintln!("index build: a build step failed; it will be retried");
                Some(FAILED_RETRY)
            }
            // Done: a commit that starts another build queues the graph again.
            Turn::Committed { more: false } | Turn::Idle => None,
            Turn::Gone => None,
        };

        if let Some(wait) = again {
            graph.not_before = Instant::now() + wait;
            requeue(graph, &mut pool());
        }
    }
}
