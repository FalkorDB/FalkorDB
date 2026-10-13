//! The online index build's driver: when build steps run, and how each one takes the write slot.
//!
//! `graph_writes::build_step` indexes one batch on a private version. This module runs those
//! steps on a background thread, one turn at a time, for every graph that has `Building` columns.
//! Nothing here knows about Redis: a host gives the builder its graphs through [`BuildHost`],
//! which is how the builder reads a graph and how it publishes as a writer. The Redis module's
//! host takes the GIL and the graph's write lock to publish; an embedded host would take its own
//! lock.
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
//! **When the builder runs.** [`MvccGraph::commit`] wakes it whenever a version brings in a
//! `Building` column its predecessor did not have; the builder then asks the host for its graphs
//! and builds every one with `Building` columns, a turn each in rotation, until none has any left.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use atomic_refcell::AtomicRefCell;

use super::graph_writes::{self, BUILD_STEP_EDGES, BUILD_STEP_NODES};
use crate::graph::graph::Graph;
use crate::graph::mvcc_graph::MvccGraph;

/// How the builder reaches one graph. Implemented by the host that owns the graph.
pub trait BuildHost: Send + Sync {
    /// Identifies the graph while it lives, so the builder can tell graphs apart when it asks the
    /// host for them again.
    fn id(&self) -> usize;

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
/// Runs at most one build turn per graph at a time: [`MvccGraph::write_for_build`] relies on it.
/// The builder thread does; a host that calls this itself must too.
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
        turn = commit(mvcc, fork.take().expect("published once"), more, false);
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

// ---- The builder thread ----

/// The graphs a host has, each as a [`BuildHost`].
type Hosts = Arc<dyn Fn() -> Vec<Arc<dyn BuildHost>> + Send + Sync>;

struct Shared {
    state: Mutex<State>,
    wakeup: Condvar,
}

struct State {
    /// A version brought in a `Building` column since the builder last asked for the graphs.
    woken: bool,
    hosts: Option<Hosts>,
    thread: Option<JoinHandle<()>>,
}

static SHARED: Shared = Shared {
    state: Mutex::new(State {
        woken: false,
        hosts: None,
        thread: None,
    }),
    wakeup: Condvar::new(),
};

/// How many times the builder has asked the host for its graphs — so a test can tell the scan at
/// start from one a commit woke.
#[cfg(test)]
pub(crate) static SCANS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Set while the builder is stopping: turns give up rather than publish.
static STOPPING: AtomicBool = AtomicBool::new(false);

fn stopping() -> bool {
    STOPPING.load(Ordering::Relaxed)
}

/// Start the builder thread, which builds the `Building` columns of the graphs `hosts` returns.
/// Idempotent while it runs; can start again after [`stop`].
pub fn start(hosts: impl Fn() -> Vec<Arc<dyn BuildHost>> + Send + Sync + 'static) {
    let mut state = SHARED.state.lock().unwrap_or_else(|e| e.into_inner());
    if state.thread.is_some() {
        return;
    }
    STOPPING.store(false, Ordering::Relaxed);
    state.hosts = Some(Arc::new(hosts));
    // Look once at start: a graph may already have `Building` columns.
    state.woken = true;
    state.thread = Some(
        thread::Builder::new()
            .name("falkordb-index-build".into())
            .spawn(run_builder)
            .expect("spawn the index builder thread"),
    );
}

/// Stop the builder thread and wait for it to exit. A turn in progress finishes its step and gives
/// up the slot instead of publishing.
pub fn stop() {
    STOPPING.store(true, Ordering::Relaxed);
    let thread = {
        let mut state = SHARED.state.lock().unwrap_or_else(|e| e.into_inner());
        state.hosts = None;
        state.thread.take()
    };
    SHARED.wakeup.notify_all();
    if let Some(thread) = thread {
        drop(thread.join());
    }
}

/// Tell the builder a version brought in a `Building` column. Never blocks for long: it is called
/// from [`MvccGraph::commit`], which holds the host's locks.
pub(crate) fn wake() {
    SHARED.state.lock().unwrap_or_else(|e| e.into_inner()).woken = true;
    SHARED.wakeup.notify_all();
}

/// How long the builder waits after a turn: as long as the turn took, so a build uses at most
/// about half of one core, and a turn that held the host's locks leaves them free at least as
/// long again.
fn pace(turn: Duration) {
    thread::sleep(turn.max(Duration::from_millis(1)));
}

/// How long the builder waits before retrying a graph whose slot a writer held.
const BUSY_RETRY: Duration = Duration::from_millis(10);

/// How long the builder waits before retrying a graph whose step panicked.
const FAILED_RETRY: Duration = Duration::from_secs(1);

struct Building {
    host: Arc<dyn BuildHost>,
    losses: u32,
}

fn run_builder() {
    let mut building: Vec<Building> = Vec::new();
    let mut next = 0usize;
    loop {
        // Wait until a version brings in a `Building` column, unless a build is under way.
        let hosts = {
            let mut state = SHARED.state.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if stopping() {
                    return;
                }
                if state.woken {
                    state.woken = false;
                    break state.hosts.clone();
                }
                if !building.is_empty() {
                    break None;
                }
                state = SHARED.wakeup.wait(state).unwrap_or_else(|e| e.into_inner());
            }
        };
        if let Some(hosts) = hosts {
            let hosts = hosts();
            #[cfg(test)]
            SCANS.fetch_add(1, Ordering::Relaxed);
            for host in hosts {
                if building.iter().any(|b| b.host.id() == host.id()) {
                    continue;
                }
                let mut has = false;
                host.read(&mut |mvcc| {
                    has = graph_writes::has_building_columns(&mvcc.read().borrow());
                });
                if has {
                    building.push(Building { host, losses: 0 });
                }
            }
        }
        if building.is_empty() {
            continue;
        }

        // One turn for the next graph in rotation.
        next %= building.len();
        let started = Instant::now();
        let entry = &mut building[next];
        let turn = catch_unwind(AssertUnwindSafe(|| {
            run_turn(entry.host.as_ref(), &mut entry.losses)
        }))
        .unwrap_or(Turn::Failed);
        match turn {
            Turn::Committed { more: true, .. } | Turn::Lost => {
                next += 1;
                pace(started.elapsed());
            }
            Turn::Committed { more: false, .. } | Turn::Idle | Turn::Gone => {
                building.swap_remove(next);
            }
            Turn::Busy => {
                next += 1;
                thread::sleep(BUSY_RETRY);
            }
            Turn::Failed => {
                eprintln!("index build: a build step failed; it will be retried");
                next += 1;
                thread::sleep(FAILED_RETRY);
            }
        }
    }
}
