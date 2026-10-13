//! The online index build's driver: when build steps run, and how each one commits.
//!
//! A build step reads one batch of a `Building` column (`graph_writes::read_build_batch`). This
//! module runs those steps on a pool of its own — [`start`]'s workers, which serve no queries — one
//! turn at a time, for every graph that has `Building` columns. Nothing here knows about Redis: a
//! host registers a [`BuildHost`] for each graph ([`MvccGraph::set_build_host`]), which is how the
//! builder reads the graph and how it publishes as a writer. The Redis module's host takes the GIL
//! and the graph's write lock to publish; an embedded host would take its own lock.
//!
//! **A turn neither makes a writer run again nor loses its batch to one.** It reads its batch on a
//! private version of the published graph and applies it there, holding no lock. Then it publishes
//! through the host: it claims the write slot ([`MvccGraph::claim`]) and commits the version as an
//! index-build step ([`MvccGraph::commit_index_build`]). If a writer committed while the batch was
//! read, the claim is stale, and the turn applies the batch to a fork of what that writer published
//! instead, skipping the entities written since
//! ([`FalkorDbIndex::apply_build`](super::falkordb_index::FalkorDbIndex::apply_build)). A queued
//! write whose fork the turn makes stale merges the other way when it claims. Only a column dropped
//! or re-created meanwhile drops a batch.
//!
//! **When the builder runs.** [`MvccGraph::commit`] queues its graph when the version it publishes
//! has a `Building` column and the graph is not queued yet, which covers every way a
//! `CREATE INDEX` can arrive. A worker takes the graph with the earliest start time, runs one turn,
//! and queues it again — not before as long as the turn took — until it has no `Building` column
//! left. A graph is either queued or on one worker, never both, so it never has two turns at once.
//! The work per `CREATE INDEX` is the same however many graphs there are.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use atomic_refcell::AtomicRefCell;

use super::falkordb_index::BuildBatch;
use super::graph_writes::{self, BUILD_STEP_EDGES, BUILD_STEP_NODES};
use crate::graph::graph::Graph;
use crate::graph::mvcc_graph::{Claim, MvccGraph};

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
    /// the GIL, then the graph's write lock). `false` if the host refuses — the graph is gone.
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
    Lost,
    /// The host refused: the graph is gone, or the builder is stopping.
    Gone,
    /// The step panicked, or its version was refused.
    Failed,
}

/// One build turn on the graph `host` names, with the default batch sizes.
///
/// The builder runs at most one turn per graph at a time. Two at once would still publish nothing
/// wrong — each merges into what the other committed — but would read the same batch twice.
pub(crate) fn run_turn(host: &dyn BuildHost) -> Turn {
    run_turn_within(host, BUILD_STEP_NODES, BUILD_STEP_EDGES)
}

/// [`run_turn`] with explicit batch sizes, for tests.
pub(crate) fn run_turn_within(
    host: &dyn BuildHost,
    nodes: usize,
    edges: usize,
) -> Turn {
    // One read to see whether there is anything to build and to fork the version to read the
    // batch on. The step itself needs no lock: it reads and writes only its own version.
    let mut version = None;
    if !host.read(&mut |mvcc| {
        if graph_writes::has_building_columns(&mvcc.read().borrow()) {
            version = Some(mvcc.fork());
        }
    }) {
        return Turn::Gone;
    }
    let Some(version) = version else {
        return Turn::Idle;
    };
    let batch = match step(&version, nodes, edges) {
        Some(Some(batch)) => batch,
        Some(None) => return Turn::Idle,
        None => return Turn::Failed,
    };
    // At shutdown the host may hold what publishing needs (the Redis host's GIL) while it waits
    // for this thread.
    if stopping() {
        return Turn::Gone;
    }

    let mut turn = Turn::Gone;
    let mut version = Some(version);
    let published = host.publish(&mut |mvcc| {
        let version = version.take().expect("published once");
        turn = match mvcc.claim(&version) {
            Ok(()) => commit(mvcc, version),
            // A writer committed while the batch was read: apply the batch to what it published.
            Err(Claim::Stale) => {
                let merged = mvcc.fork();
                if merged.borrow_mut().falkordb_index.apply_build(&batch)
                    && mvcc.claim(&merged).is_ok()
                {
                    commit(mvcc, merged)
                } else {
                    Turn::Lost
                }
            }
            // Writers claim the slot only with publish access, so it is free; handled all the same.
            Err(Claim::Busy) => Turn::Lost,
        };
    });
    if published { turn } else { Turn::Gone }
}

/// Read one build step's batch on `version` and apply it there. `None` if it panicked;
/// `Some(None)` if no column is `Building`.
fn step(
    version: &AtomicRefCell<Graph>,
    nodes: usize,
    edges: usize,
) -> Option<Option<BuildBatch>> {
    catch_unwind(AssertUnwindSafe(|| {
        let mut g = version.borrow_mut();
        let batch = graph_writes::read_build_batch(&g, nodes, edges)?;
        g.falkordb_index.apply_build(&batch);
        Some(batch)
    }))
    .ok()
}

/// Commit a build step, holding the slot.
fn commit(
    mvcc: &mut MvccGraph,
    version: Arc<AtomicRefCell<Graph>>,
) -> Turn {
    let more = graph_writes::has_building_columns(&version.borrow());
    // A refused version is rolled back by the commit itself; the column stays `Building` and a
    // later turn retries.
    match mvcc.commit_index_build(version) {
        Ok(()) => Turn::Committed { more },
        Err(_) => Turn::Failed,
    }
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
    ) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Queued {
    fn cmp(
        &self,
        other: &Self,
    ) -> std::cmp::Ordering {
        (self.not_before, self.seq).cmp(&(other.not_before, other.seq))
    }
}

struct Pool {
    /// Graphs waiting for a turn, earliest start first.
    queue: BinaryHeap<Reverse<Queued>>,
    /// Graphs a worker is running a turn for.
    running: usize,
    next_seq: u64,
    workers: Vec<JoinHandle<()>>,
}

static POOL: Mutex<Pool> = Mutex::new(Pool {
    queue: BinaryHeap::new(),
    running: 0,
    next_seq: 0,
    workers: Vec::new(),
});

/// Signalled when a graph is queued, and at stop.
static WORK: Condvar = Condvar::new();

/// Set while the pool is stopping: workers exit, and a turn gives up rather than publish.
static STOPPING: AtomicBool = AtomicBool::new(false);

fn stopping() -> bool {
    STOPPING.load(Ordering::Relaxed)
}

fn pool() -> MutexGuard<'static, Pool> {
    POOL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Start the pool's `threads` workers (at least one). Idempotent while they run; can start again
/// after [`stop`].
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

/// Stop the pool's workers and wait for them to exit. A turn in progress finishes its step and
/// drops it instead of publishing. Graphs still queued stay queued.
pub fn stop() {
    STOPPING.store(true, Ordering::Relaxed);
    let workers = std::mem::take(&mut pool().workers);
    WORK.notify_all();
    for worker in workers {
        drop(worker.join());
    }
}

/// Graphs queued for a turn or in the middle of one — what is left to build.
#[must_use]
pub fn pending() -> usize {
    let pool = pool();
    pool.queue.len() + pool.running
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

/// How long a graph waits before retrying after its step panicked.
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
                        pool.running += 1;
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
            Turn::Committed { more: true } | Turn::Lost => Some(pace(started.elapsed())),
            Turn::Failed => {
                eprintln!("index build: a build step failed; it will be retried");
                Some(FAILED_RETRY)
            }
            // Done — unless a `Building` column came in while the graph was still queued.
            Turn::Committed { more: false } | Turn::Idle => {
                let mut more = false;
                graph.host.read(&mut |mvcc| more = mvcc.build_done());
                more.then_some(Duration::ZERO)
            }
            Turn::Gone => None,
        };

        let mut pool = pool();
        pool.running -= 1;
        if let Some(wait) = again {
            graph.not_before = Instant::now() + wait;
            requeue(graph, &mut pool);
        }
    }
}
