//! The online index build's driver: when build steps run, and how each one takes the write slot.
//!
//! `graph_writes::build_step` indexes one batch on a private version. This module runs those
//! steps on a pool of its own — [`start`]'s workers, which serve no queries — one turn at a time,
//! for every graph that has `Building` columns. Nothing here knows about Redis: a host registers a
//! [`BuildHost`] for each graph ([`MvccGraph::set_build_host`]), which is how the builder reads the
//! graph and how it publishes as a writer. The Redis module's host takes the GIL and the graph's
//! write lock to publish; an embedded host would take its own lock.
//!
//! **A turn is a writer that gives way.** It takes the write slot as a build step
//! ([`MvccGraph::write_for_build`]) and indexes its batch holding no lock. A writer that wants the
//! slot meanwhile takes it over instead of failing ([`MvccGraph::write`]). Then the turn publishes
//! through the host: it claims the slot as a writer and commits ([`MvccGraph::claim_build`]), or
//! finds it lost the slot and drops its batch for a later turn.
//!
//! **Starvation.** After two lost turns in a row (`TURNS_LOST_BEFORE_HOLDING`), the next turn holds:
//! it publishes through the host first, takes the slot as an ordinary writer, and keeps both for
//! the whole step, so nothing can take the slot from it. A committed turn resets the count.
//!
//! **When the builder runs.** [`MvccGraph::commit`] queues its graph when the version it publishes
//! has a `Building` column and the graph is not queued yet, which covers every way a
//! `CREATE INDEX` can arrive. A worker takes the graph with the earliest start time, runs one turn,
//! and queues it again — not before as long as the turn took, so the graph's writers get the slot
//! and the GIL between its turns — until it has no `Building` column left. A graph is either queued
//! or on one worker, never both, so it never has two turns at once. The work per `CREATE INDEX` is
//! the same however many graphs there are.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use atomic_refcell::AtomicRefCell;

use super::graph_writes::{self, BUILD_STEP_EDGES, BUILD_STEP_NODES};
use crate::graph::graph::Graph;
use crate::graph::mvcc_graph::MvccGraph;

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

/// Lost turns in a row after which a graph's next build turn holds the slot instead of giving way.
pub(crate) const TURNS_LOST_BEFORE_HOLDING: u32 = 2;

/// What one build turn did.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Turn {
    /// The graph has no `Building` column.
    Idle,
    /// The step was committed; `more` if a column is still `Building`. `held` if the turn held
    /// the slot rather than giving way.
    Committed { more: bool, held: bool },
    /// A writer took the slot over; the batch was dropped.
    Lost,
    /// A writer held the slot, so the turn did not start.
    Busy,
    /// The host refused: the graph is gone, or the builder is stopping.
    Gone,
    /// The step panicked. The slot was released.
    Failed,
}

/// One build turn on the graph `host` names, with the default batch sizes. `losses` is the
/// graph's count of turns lost in a row, kept by the caller across turns.
///
/// The builder runs at most one turn per graph at a time. Two at once would still publish nothing
/// stale — a turn checks its version is the next one before committing — but would waste work.
pub(crate) fn run_turn(
    host: &dyn BuildHost,
    losses: &mut u32,
) -> Turn {
    run_turn_within(host, losses, BUILD_STEP_NODES, BUILD_STEP_EDGES)
}

/// [`run_turn`] with explicit batch sizes, for tests.
pub(crate) fn run_turn_within(
    host: &dyn BuildHost,
    losses: &mut u32,
    nodes: usize,
    edges: usize,
) -> Turn {
    // One read to see whether there is anything to build and, for a turn that gives way, to take
    // the slot. The step itself needs no lock: it reads and writes only its own version.
    let holding = *losses >= TURNS_LOST_BEFORE_HOLDING;
    let mut building = false;
    let mut fork = None;
    if !host.read(&mut |mvcc| {
        building = graph_writes::has_building_columns(&mvcc.read().borrow());
        if building && !holding {
            fork = mvcc.write_for_build();
        }
    }) {
        return Turn::Gone;
    }
    if !building {
        return Turn::Idle;
    }
    if holding {
        let turn = held_turn(host, nodes, edges);
        if matches!(turn, Turn::Committed { .. }) {
            *losses = 0;
        }
        return turn;
    }
    let Some(fork) = fork else {
        return Turn::Busy;
    };
    let Some(more) = step(&fork, nodes, edges) else {
        host.read(&mut |mvcc| mvcc.abandon_build());
        return Turn::Failed;
    };
    // At shutdown the host may hold what publishing needs (the Redis host's GIL) while it waits
    // for this thread. Giving up the slot needs only read access.
    if stopping() {
        host.read(&mut |mvcc| mvcc.abandon_build());
        return Turn::Gone;
    }

    let mut turn = Turn::Gone;
    let mut fork = Some(fork);
    let published = host.publish(&mut |mvcc| {
        if !mvcc.claim_build() {
            // A writer took the slot over. The batch was read from the version that writer
            // replaces, so it is dropped and read again on a later turn.
            turn = Turn::Lost;
            return;
        }
        let fork = fork.take().expect("published once");
        // The claim succeeding is not proof enough on its own: had a writer taken the slot, then
        // committed, and another build step taken it again, the claim would succeed on that
        // step's hold with this step's stale version. Only the version right after the published
        // one may be committed.
        if fork.borrow().version != mvcc.read().borrow().version + 1 {
            mvcc.rollback();
            turn = Turn::Lost;
            return;
        }
        turn = commit(mvcc, fork, more, false);
    });
    if !published {
        host.read(&mut |mvcc| mvcc.abandon_build());
        return Turn::Gone;
    }
    match turn {
        Turn::Lost => *losses += 1,
        Turn::Committed { .. } => *losses = 0,
        _ => {}
    }
    turn
}

/// A turn that holds: publish access first, then the slot as an ordinary writer, which nobody can
/// take over, for the whole step.
fn held_turn(
    host: &dyn BuildHost,
    nodes: usize,
    edges: usize,
) -> Turn {
    if stopping() {
        return Turn::Gone;
    }
    let mut turn = Turn::Gone;
    let published = host.publish(&mut |mvcc| {
        // Another writer holds the slot without the host's locks: a client write between taking
        // it and publishing, or `GRAPH.BULK` building its version.
        let Some(fork) = mvcc.write() else {
            turn = Turn::Busy;
            return;
        };
        turn = match step(&fork, nodes, edges) {
            Some(more) => commit(mvcc, fork, more, true),
            None => {
                mvcc.rollback();
                Turn::Failed
            }
        };
    });
    if published { turn } else { Turn::Gone }
}

/// Run one build step on `fork`. `None` if it panicked.
fn step(
    fork: &AtomicRefCell<Graph>,
    nodes: usize,
    edges: usize,
) -> Option<bool> {
    catch_unwind(AssertUnwindSafe(|| {
        graph_writes::build_step(&mut fork.borrow_mut(), nodes, edges)
    }))
    .ok()
}

/// Commit a build step, holding the slot as a writer.
fn commit(
    mvcc: &mut MvccGraph,
    fork: Arc<AtomicRefCell<Graph>>,
    more: bool,
    held: bool,
) -> Turn {
    // A refused version is rolled back by `commit` itself; the column stays `Building` and a
    // later turn retries.
    match mvcc.commit(fork) {
        Ok(()) => Turn::Committed { more, held },
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
    /// Turns lost in a row.
    losses: u32,
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
/// gives up the slot instead of publishing. Graphs still queued stay queued.
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
            losses: 0,
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
/// took, so the graph's writers get the slot, and on a held turn the GIL, at least as long again.
fn pace(turn: Duration) -> Duration {
    turn.max(Duration::from_millis(1))
}

/// How long a graph waits before retrying when a writer held its slot.
const BUSY_RETRY: Duration = Duration::from_millis(10);

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
        let turn = catch_unwind(AssertUnwindSafe(|| {
            run_turn(graph.host.as_ref(), &mut graph.losses)
        }))
        .unwrap_or(Turn::Failed);
        let again = match turn {
            Turn::Committed { more: true, .. } | Turn::Lost => Some(pace(started.elapsed())),
            Turn::Busy => Some(BUSY_RETRY),
            Turn::Failed => {
                eprintln!("index build: a build step failed; it will be retried");
                Some(FAILED_RETRY)
            }
            // Done — unless a `Building` column came in while the graph was still queued.
            Turn::Committed { more: false, .. } | Turn::Idle => {
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
