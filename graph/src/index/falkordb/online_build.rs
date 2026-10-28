//! The background index builder: runs the indexer batches of every `Building` index, in every
//! graph, on a pool of its own.
//!
//! An indexer batch reads the next entities of a `Building` index ([`Graph::indexer_prepare`]). The
//! builder runs one chain of batches per index, each batch a turn on a worker of its own pool,
//! which serves no queries, so no two batches ever run on one index, and the indexes of a graph, or
//! of several, build side by side. A turn decides what comes next: its chain submits the index's
//! next turn until the index is `Ready`. Nothing here knows about Redis: a host registers a
//! [`BuildHost`] with each graph ([`MvccGraph::set_build_host`]), which is how the builder reads
//! the graph and how it publishes as a writer does.
//!
//! **A turn holds no lock while it reads.** It takes its batch on a new build version of the
//! published one ([`Graph::new_build_version`]), which only it holds, and then publishes through
//! the host ([`MvccGraph::commit_indexer_batch`]). What committed meanwhile decides how the batch
//! goes in; the rules writers and batches follow so that every published version stays correct, in
//! whatever order they commit, are on [`BuildState`](super::versioned::BuildState). A batch's
//! commit keeps the data version, so it never makes a writer's fork stale.
//!
//! **When the builder runs.** A commit that brings in a `Building` index starts its chain
//! ([`MvccGraph::commit`]): a `CREATE INDEX`, however it arrives, or an index dropped and created
//! again. So does registering a graph's host, for every index its published version holds
//! `Building`: one a commit brought in before any host was registered, or one a stop left
//! unfinished. A chain already running for the index takes one more turn when it would have ended,
//! in case it ended on the index that was dropped.
//!
//! **Stopping.** [`stop`] waits a bounded time for the pool's workers, since a turn waiting for
//! what its host needs to publish may be waiting on the caller (the Redis shutdown handler holds
//! the GIL). A turn checks again that the builder is not stopping once it holds what publishing
//! needs, so one that waited publishes nothing. Then `stop` waits, with no limit, for every turn
//! still holding a version of a graph with no lock: each one ends, and the caller tears GraphBLAS
//! down next. Each start of the pool is a [`Run`] of its own, so a worker a stop left behind,
//! finishing after a restart, counts and checks against its own stopped run, never the new one.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};
use std::thread;
use std::time::Duration;

use atomic_refcell::AtomicRefCell;

use super::versioned::{IndexId, IndexerCommit};
use crate::graph::graph::Graph;
use crate::graph::mvcc_graph::MvccGraph;
use crate::threadpool::ThreadPool;

/// How the builder reaches one graph. Implemented by the host that owns the graph, and registered
/// with it through [`MvccGraph::set_build_host`].
pub trait BuildHost: Send + Sync {
    /// The graph's published version, loaded under whatever guards the host's graph and released
    /// before the version is borrowed: a published version is never changed in place, so it may be
    /// read with no lock held. `None` if the graph is gone.
    fn read(&self) -> Option<Arc<AtomicRefCell<Graph>>>;

    /// Run `f` on the graph's coordinator with what a writer holds to publish a version: the host's
    /// global lock (the Redis host takes the GIL, then the graph's read lock to reach the
    /// coordinator). `false` if the host refuses: the graph is gone.
    fn publish(
        &self,
        f: &mut dyn FnMut(&MvccGraph),
    ) -> bool;
}

/// How many entities one indexer batch reads, as the earlier online build's step did (0742bcefb).
pub const BATCH_ENTITIES: usize = 10_000;

/// What one turn did.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Turn {
    /// The index was not `Building`: it was built, or dropped.
    Idle,
    /// The batch was published; `more` if the index is still `Building`.
    Committed { more: bool },
    /// The batch published nothing: its index was dropped or created again since the batch read
    /// it, or another batch committed on it. `more` if the index, created again or not, is
    /// `Building`: the next batch starts from its published cursor.
    Dropped { more: bool },
    /// The host refused, or the builder is stopping.
    Gone,
    /// The batch panicked, or its version was refused. Under the Redis module a panic ends the
    /// process (its panic hook exits), so there only a refused version comes back as this.
    Failed,
}

/// One start of an indexer's pool, to its stop. A turn belongs to the run that submitted it.
pub(crate) struct Run {
    /// Set by the stop: a turn not started gives up, and one started publishes nothing.
    stopping: AtomicBool,
    /// Turns submitted that have not started.
    queued: AtomicUsize,
    /// Turns holding a version of a graph with no lock held: from inside the read until the batch
    /// is applied, and again while they free what they held. The stop waits for none to be left.
    reading: AtomicUsize,
}

impl Run {
    pub(crate) fn new() -> Self {
        Self {
            stopping: AtomicBool::new(false),
            queued: AtomicUsize::new(0),
            reading: AtomicUsize::new(0),
        }
    }

    pub(crate) fn stopping(&self) -> bool {
        self.stopping.load(Ordering::SeqCst)
    }

    /// Stop the run: a turn checks this before it reads and again once it holds what publishing
    /// needs.
    pub(crate) fn stop(&self) {
        self.stopping.store(true, Ordering::SeqCst);
    }

    /// Wait, with no limit, for every turn still holding a version with no lock held. Each one
    /// ends: none waits on a lock while counted.
    fn wait_for_readers(&self) {
        while self.reading.load(Ordering::SeqCst) > 0 {
            thread::sleep(Duration::from_millis(1));
        }
    }
}

/// A turn's place in [`Run::reading`], given up when dropped, however the turn ends.
struct Reading<'a>(&'a Run);

impl<'a> Reading<'a> {
    /// `None`, counting nothing, once the run is stopping: the stop may already have seen no turn
    /// left. Ordered against the stop flag (both `SeqCst`), so either the turn sees the flag or the
    /// stop sees the turn.
    fn enter(run: &'a Run) -> Option<Self> {
        run.reading.fetch_add(1, Ordering::SeqCst);
        let reading = Self(run);
        (!run.stopping()).then_some(reading)
    }
}

impl Drop for Reading<'_> {
    fn drop(&mut self) {
        self.0.reading.fetch_sub(1, Ordering::SeqCst);
    }
}

/// One turn of index `id` of the graph `host` reaches, in `run`: one indexer batch of at most
/// `entities` entities, published.
///
/// The builder runs at most one turn per index at a time ([`Indexer`]). Two at once would still
/// publish nothing wrong (the later commit finds the earlier batch on the index and publishes
/// nothing), but a debug build stops on it as a bug ([`report_batch_race`]).
///
/// [`report_batch_race`]: super::versioned::report_batch_race
pub(crate) fn run_turn(
    host: &dyn BuildHost,
    id: &IndexId,
    run: &Run,
    entities: usize,
) -> Turn {
    let Some(reading) = Reading::enter(run) else {
        return Turn::Gone;
    };
    let Some(published) = host.read() else {
        return Turn::Gone;
    };
    let prepared = Arc::new(AtomicRefCell::new(published.borrow().new_build_version()));
    drop(published);
    let batch = prepared.borrow_mut().indexer_prepare(id, entities);
    let Some(batch) = batch else {
        // Freed while still counted, so GraphBLAS is not torn down under it.
        drop(prepared);
        drop(reading);
        return Turn::Idle;
    };
    if run.stopping() {
        drop(prepared);
        drop(reading);
        return Turn::Gone;
    }
    // Not counted while publishing: whoever stops the builder may hold what publishing needs (the
    // Redis shutdown handler holds the GIL) while it waits for the turns reading.
    drop(reading);

    let mut turn = Turn::Gone;
    // Held, with `prepared`, until the host has let go of what it holds, so that freeing the
    // version the commit replaces, or `prepared` if the batch did not publish it, never happens
    // under the host's locks.
    let mut replaced = None;
    let reached = host.publish(&mut |mvcc| {
        // Again, now that the host holds what publishing needs: a turn that waited for it while
        // the builder stopped publishes nothing.
        if run.stopping() {
            return;
        }
        replaced = Some(mvcc.read());
        turn = match mvcc.commit_indexer_batch(&prepared, &batch) {
            Some(IndexerCommit::Same | IndexerCommit::Swap | IndexerCommit::CatchUp(_)) => {
                Turn::Committed {
                    more: mvcc.read().borrow().index_building(id),
                }
            }
            Some(IndexerCommit::Dropped(_)) => Turn::Dropped {
                more: mvcc.read().borrow().index_building(id),
            },
            // Writers claim the slot only holding what the host's publish holds, so it was free;
            // retried all the same, as a refused version is.
            None => Turn::Failed,
        };
    });
    // The versions this turn still holds are freed counted again, or, once the builder is
    // stopping, left to the exiting process: GraphBLAS may be torn down by then.
    match Reading::enter(run) {
        Some(_reading) => {
            drop(replaced);
            drop(prepared);
        }
        None => {
            std::mem::forget(replaced);
            std::mem::forget(prepared);
        }
    }
    if reached { turn } else { Turn::Gone }
}

// ---- Running the turns ----

/// A turn as an [`Indexer`] runs it: [`run_turn`] for the builder; a stand-in in the tests of how
/// turns are scheduled.
type TurnFn = dyn Fn(&dyn BuildHost, &IndexId, &Run) -> Turn + Send + Sync;

/// An index's chain of turns: the graph, by its host, and the index. It builds whatever `Building`
/// index the graph has under that name, so an index dropped and created again while its chain runs
/// is built by the same chain, never by a second one.
struct Chain {
    host: Arc<dyn BuildHost>,
    id: IndexId,
}

/// Which index of which graph a [`Chain`] builds: its host's address, which the chain keeps alive,
/// so no other host can take it while the chain is known, and the index.
type ChainId = (usize, IndexId);

fn chain_id(
    host: &Arc<dyn BuildHost>,
    id: &IndexId,
) -> ChainId {
    (Arc::as_ptr(host).cast::<()>() as usize, id.clone())
}

/// A chain the indexer knows: its host, and whether a commit created its index again while a turn
/// ran, in which case the chain takes one more turn when it would have ended.
struct Known {
    host: Arc<dyn BuildHost>,
    again: bool,
}

/// Runs the turns of every index being built, on a pool of its own: one chain of turns per index,
/// so at most one turn per index at a time, and the indexes side by side. The builder is one of
/// these ([`start`], [`stop`], [`enqueue`]); tests make their own.
pub(crate) struct Indexer {
    state: Mutex<IndexerState>,
    turn: Box<TurnFn>,
}

struct IndexerState {
    /// The pool and its run, while the indexer runs.
    running: Option<(Arc<ThreadPool>, Arc<Run>)>,
    /// Every chain not ended: queued or running while the indexer runs, and waiting for the next
    /// start while it does not.
    chains: HashMap<ChainId, Known>,
}

impl Indexer {
    pub(crate) fn new(turn: Box<TurnFn>) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(IndexerState {
                running: None,
                chains: HashMap::new(),
            }),
            turn,
        })
    }

    fn state(&self) -> MutexGuard<'_, IndexerState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Start the pool with `threads` workers (at least one), as a new run, and submit every chain
    /// not ended: those started before it, and those a stop interrupted. Idempotent while it runs.
    pub(crate) fn start(
        self: &Arc<Self>,
        threads: usize,
    ) {
        let mut state = self.state();
        if state.running.is_some() {
            return;
        }
        let pool = Arc::new(ThreadPool::named(threads.max(1), "falkordb-index-build"));
        let run = Arc::new(Run::new());
        let chains: Vec<Chain> = state
            .chains
            .iter_mut()
            .map(|((_, id), known)| {
                known.again = false;
                Chain {
                    host: Arc::clone(&known.host),
                    id: id.clone(),
                }
            })
            .collect();
        state.running = Some((Arc::clone(&pool), Arc::clone(&run)));
        drop(state);
        for chain in chains {
            self.spawn(&pool, &run, chain);
        }
    }

    /// Stop the pool, waiting up to `wait` for its workers ([`ThreadPool::shutdown_within`]); a
    /// worker still running a turn then is left behind, and its turn publishes nothing. Then wait,
    /// with no limit, for every turn still holding a version with no lock held. The chains not
    /// ended wait for the next start.
    pub(crate) fn stop_within(
        &self,
        wait: Duration,
    ) {
        let Some((pool, run)) = self.state().running.take() else {
            return;
        };
        run.stop();
        pool.shutdown_within(wait);
        run.wait_for_readers();
    }

    /// Chains waiting for a turn: queued on the pool while it runs, every chain not ended while it
    /// does not.
    pub(crate) fn pending(&self) -> usize {
        let state = self.state();
        match &state.running {
            Some((_, run)) => run.queued.load(Ordering::SeqCst),
            None => state.chains.len(),
        }
    }

    /// Whether every chain has ended.
    #[cfg(test)]
    fn idle(&self) -> bool {
        self.state().chains.is_empty()
    }

    /// Start the chain of index `id` of `host`'s graph, or, if the index has one, have that chain
    /// take one more turn when it would end. Never blocks for long: the builder's is called from
    /// [`MvccGraph::commit`], under the host's global lock.
    pub(crate) fn enqueue(
        self: &Arc<Self>,
        host: Arc<dyn BuildHost>,
        id: IndexId,
    ) {
        let mut state = self.state();
        match state.chains.entry(chain_id(&host, &id)) {
            Entry::Occupied(mut known) => {
                known.get_mut().again = true;
                return;
            }
            Entry::Vacant(new) => {
                new.insert(Known {
                    host: Arc::clone(&host),
                    again: false,
                });
            }
        }
        // Before the pool starts, the chain waits for it.
        let Some((pool, run)) = state.running.clone() else {
            return;
        };
        drop(state);
        self.spawn(&pool, &run, Chain { host, id });
    }

    /// Submit `chain`'s next turn to `run`'s pool.
    fn spawn(
        self: &Arc<Self>,
        pool: &ThreadPool,
        run: &Arc<Run>,
        chain: Chain,
    ) {
        run.queued.fetch_add(1, Ordering::SeqCst);
        let indexer = Arc::clone(self);
        let run = Arc::clone(run);
        pool.spawn(move || indexer.take_turn(&run, chain), None);
    }

    /// One turn of `chain`'s index, on a worker of `run`'s pool, then the decision on its next one.
    fn take_turn(
        self: &Arc<Self>,
        run: &Arc<Run>,
        chain: Chain,
    ) {
        run.queued.fetch_sub(1, Ordering::SeqCst);
        if run.stopping() {
            return;
        }
        // A panic in a turn is caught here only where no panic hook ends the process first, which
        // the Redis module's does.
        let turn = catch_unwind(AssertUnwindSafe(|| {
            (self.turn)(chain.host.as_ref(), &chain.id, run)
        }))
        .unwrap_or(Turn::Failed);
        if turn == Turn::Failed {
            eprintln!("native index: an indexer batch failed; it will be retried");
        }
        let next = next_turn(&turn);
        if let Some(wait) = next
            && !wait.is_zero()
        {
            // Only after a failure, which is a bug: holding this worker for it costs nothing that
            // matters.
            thread::sleep(wait);
        }
        self.after_turn(run, chain, next.is_some());
    }

    /// Submit `chain`'s next turn if `more`, or if a commit created its index again while this turn
    /// ran; otherwise end the chain, and the next commit that creates its index starts a new one.
    /// Nothing, if `run` was stopped: its chains wait for the next start, which submits them.
    fn after_turn(
        self: &Arc<Self>,
        run: &Arc<Run>,
        chain: Chain,
        more: bool,
    ) {
        let mut state = self.state();
        let Some((pool, current)) = state.running.clone() else {
            return;
        };
        if !Arc::ptr_eq(&current, run) {
            return;
        }
        let id = chain_id(&chain.host, &chain.id);
        let go = more
            || match state.chains.get_mut(&id) {
                Some(known) if known.again => {
                    known.again = false;
                    true
                }
                Some(_) => {
                    state.chains.remove(&id);
                    false
                }
                None => false,
            };
        drop(state);
        if go {
            self.spawn(&pool, run, chain);
        }
    }
}

/// How long to wait before an index's next turn, after a turn that went `turn`; `None` when the
/// index is done, and a commit that creates it again starts it again.
fn next_turn(turn: &Turn) -> Option<Duration> {
    match turn {
        Turn::Committed { more: true } | Turn::Dropped { more: true } => Some(Duration::ZERO),
        Turn::Failed => Some(FAILED_RETRY),
        Turn::Committed { more: false }
        | Turn::Dropped { more: false }
        | Turn::Idle
        | Turn::Gone => None,
    }
}

/// How long an index waits before retrying after a turn failed: its batch panicked, or its version
/// was refused.
const FAILED_RETRY: Duration = Duration::from_secs(1);

/// How long [`stop`] waits for the builder's workers.
const STOP_WAIT: Duration = Duration::from_secs(2);

/// The builder: the indexer every graph's commits start chains on.
static BUILDER: LazyLock<Arc<Indexer>> = LazyLock::new(|| {
    Indexer::new(Box::new(|host, id, run| {
        run_turn(host, id, run, BATCH_ENTITIES)
    }))
});

/// Start the builder's pool with `threads` workers (at least one), and submit every chain not
/// ended. Idempotent while it runs.
pub fn start(threads: usize) {
    BUILDER.start(threads);
}

/// Stop the builder. Its pool's workers get [`STOP_WAIT`]: one waiting for what its host needs to
/// publish may be waiting on the caller (the Redis shutdown handler calls this holding the GIL),
/// so it is left behind, and publishes nothing once it gets that. Then waits, with no limit, for
/// every turn still holding a version of a graph with no lock held, since the caller tears
/// GraphBLAS down next.
pub fn stop() {
    BUILDER.stop_within(STOP_WAIT);
}

/// Chains waiting for a turn. For tests.
#[must_use]
pub fn pending() -> usize {
    BUILDER.pending()
}

/// Start the build of index `id` of `host`'s graph ([`Indexer::enqueue`]).
pub(crate) fn enqueue(
    host: Arc<dyn BuildHost>,
    id: IndexId,
) {
    BUILDER.enqueue(host, id);
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::mpsc;
    use std::time::Instant;

    use super::*;
    use crate::entity_type::EntityType;

    /// After a turn that published or dropped its batch with more to build, the next one runs
    /// right away; after a failed one, a second later; after any other, there is none.
    #[test]
    fn a_turn_decides_when_the_next_one_runs() {
        for more in [Turn::Committed { more: true }, Turn::Dropped { more: true }] {
            assert_eq!(next_turn(&more), Some(Duration::ZERO), "{more:?}");
        }
        assert_eq!(next_turn(&Turn::Failed), Some(FAILED_RETRY));
        for done in [
            Turn::Committed { more: false },
            Turn::Dropped { more: false },
            Turn::Idle,
            Turn::Gone,
        ] {
            assert_eq!(next_turn(&done), None, "{done:?}");
        }
    }

    /// A graph for the stand-in turns, which never reach it.
    struct NoGraph;

    impl BuildHost for NoGraph {
        fn read(&self) -> Option<Arc<AtomicRefCell<Graph>>> {
            None
        }

        fn publish(
            &self,
            _: &mut dyn FnMut(&MvccGraph),
        ) -> bool {
            false
        }
    }

    /// A stand-in turn's chain, as the turn sees it: its graph's address and the index.
    fn chain_of(
        host: &dyn BuildHost,
        id: &IndexId,
    ) -> ChainId {
        (std::ptr::from_ref(host).cast::<()>() as usize, id.clone())
    }

    fn column(
        entity: EntityType,
        attr: &str,
    ) -> IndexId {
        IndexId::Column(
            entity,
            (Arc::new("L".to_string()), Arc::new(attr.to_string())),
        )
    }

    fn wait_until(
        what: &str,
        mut done: impl FnMut() -> bool,
    ) {
        let deadline = Instant::now() + Duration::from_secs(20);
        while !done() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            thread::sleep(Duration::from_millis(1));
        }
    }

    /// What the stand-in turns of [`one_turn_per_index_at_a_time_and_indexes_side_by_side`] saw.
    #[derive(Default)]
    struct Seen {
        /// Turns running now, per chain and in all.
        running: HashMap<ChainId, usize>,
        running_all: usize,
        /// The most turns ever running at once on one chain, and in all.
        most_on_a_chain: usize,
        most_at_once: usize,
        /// Turns taken, per chain.
        turns: HashMap<ChainId, usize>,
    }

    /// With more workers than indexes, and every index's build started again and again while it
    /// runs, as commits that create an index anew do, no index ever has two turns at once, several
    /// indexes do run at once, and every index gets all its turns. The turns are stand-ins that
    /// only record what runs, so this checks the indexer alone: the one thing that keeps two turns
    /// off one index.
    #[test]
    fn one_turn_per_index_at_a_time_and_indexes_side_by_side() {
        const TURNS: usize = 30;
        let seen = Arc::new(Mutex::new(Seen::default()));
        let in_turn = Arc::clone(&seen);
        let indexer = Indexer::new(Box::new(move |host, id, _| {
            let chain = chain_of(host, id);
            {
                let mut seen = in_turn.lock().unwrap();
                let running = seen.running.entry(chain.clone()).or_default();
                *running += 1;
                let running = *running;
                seen.running_all += 1;
                seen.most_on_a_chain = seen.most_on_a_chain.max(running);
                seen.most_at_once = seen.most_at_once.max(seen.running_all);
            }
            thread::sleep(Duration::from_micros(300));
            let mut seen = in_turn.lock().unwrap();
            *seen.running.get_mut(&chain).unwrap() -= 1;
            seen.running_all -= 1;
            let turns = seen.turns.entry(chain).or_default();
            *turns += 1;
            Turn::Committed {
                more: *turns < TURNS,
            }
        }));
        let hosts: Vec<Arc<dyn BuildHost>> = (0..3)
            .map(|_| Arc::new(NoGraph) as Arc<dyn BuildHost>)
            .collect();
        let ids = [
            column(EntityType::Node, "a"),
            column(EntityType::Node, "b"),
            column(EntityType::Relationship, "a"),
        ];
        indexer.start(8);
        thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    for _ in 0..40 {
                        for host in &hosts {
                            for id in &ids {
                                indexer.enqueue(Arc::clone(host), id.clone());
                            }
                        }
                        thread::sleep(Duration::from_micros(200));
                    }
                });
            }
        });
        wait_until("every chain to end", || indexer.idle());
        indexer.stop_within(STOP_WAIT);

        let seen = seen.lock().unwrap();
        assert_eq!(seen.most_on_a_chain, 1, "two turns on one index at once");
        assert!(seen.most_at_once > 1, "the indexes never ran side by side");
        let all: HashSet<&ChainId> = seen.turns.keys().collect();
        assert_eq!(all.len(), hosts.len() * ids.len(), "an index never ran");
        assert!(
            seen.turns.values().all(|&turns| turns >= TURNS),
            "an index stopped short: {:?}",
            seen.turns.values().collect::<Vec<_>>()
        );
    }

    /// An index created again while its chain's last turn runs (the turn finds it done, and the
    /// commit that creates it again finds the chain still there) is built: the chain takes one more
    /// turn instead of ending.
    #[test]
    fn an_index_created_again_as_its_chain_ends_is_built() {
        let (in_turn, wait_in_turn) = mpsc::channel::<()>();
        let (go, wait_go) = mpsc::channel::<()>();
        let wait_go = Mutex::new(wait_go);
        let turns = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&turns);
        let indexer = Indexer::new(Box::new(move |_, _, _| {
            if counted.fetch_add(1, Ordering::Relaxed) == 0 {
                in_turn.send(()).unwrap();
                wait_go.lock().unwrap().recv().unwrap();
            }
            Turn::Idle
        }));
        let host: Arc<dyn BuildHost> = Arc::new(NoGraph);
        indexer.start(2);
        indexer.enqueue(Arc::clone(&host), column(EntityType::Node, "a"));
        wait_in_turn.recv().unwrap();
        indexer.enqueue(Arc::clone(&host), column(EntityType::Node, "a"));
        go.send(()).unwrap();
        wait_until("the chain to end", || indexer.idle());
        indexer.stop_within(STOP_WAIT);
        assert_eq!(turns.load(Ordering::Relaxed), 2, "one more turn");
    }

    /// A chain started before the pool, or while it is stopped, waits for it and runs once it
    /// starts. So does a chain a stop interrupted: its index is still `Building`, and the next
    /// start builds it.
    #[test]
    fn a_chain_waits_for_the_pool_and_a_stop_interrupts_it_until_the_next_start() {
        let (in_turn, wait_in_turn) = mpsc::channel::<()>();
        let (go, wait_go) = mpsc::channel::<()>();
        let wait_go = Mutex::new(wait_go);
        let turns = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&turns);
        let indexer = Indexer::new(Box::new(move |_, _, run| {
            match counted.fetch_add(1, Ordering::SeqCst) {
                // The first turn of the second start runs until the stop.
                1 => {
                    in_turn.send(()).unwrap();
                    wait_go.lock().unwrap().recv().unwrap();
                    assert!(run.stopping(), "released only once stopped");
                    Turn::Gone
                }
                _ => Turn::Idle,
            }
        }));
        let host: Arc<dyn BuildHost> = Arc::new(NoGraph);
        indexer.enqueue(Arc::clone(&host), column(EntityType::Node, "a"));
        assert_eq!(indexer.pending(), 1, "waits for the pool");
        indexer.start(1);
        wait_until("the first chain to end", || indexer.idle());
        indexer.stop_within(STOP_WAIT);

        indexer.enqueue(Arc::clone(&host), column(EntityType::Node, "b"));
        assert_eq!(indexer.pending(), 1, "waits for the next start");
        indexer.start(1);
        wait_in_turn.recv().unwrap();
        thread::scope(|scope| {
            scope.spawn(|| indexer.stop_within(Duration::from_millis(50)));
            thread::sleep(Duration::from_millis(100));
            go.send(()).unwrap();
        });
        assert_eq!(indexer.pending(), 1, "the interrupted chain waits");
        indexer.start(1);
        wait_until("the interrupted chain to end", || indexer.idle());
        indexer.stop_within(STOP_WAIT);
        assert_eq!(turns.load(Ordering::SeqCst), 3);
    }

    /// A worker a stop left behind, with a turn running and another queued on its stopped pool,
    /// never touches the next run. Its turn ends after a new start has submitted the same index's
    /// chain again: the stopped run's turn leaves that chain alone, so an index created again
    /// meanwhile still gets one more turn of that chain, not a second chain beside it. Its queued
    /// turn gives up and counts against its own run, so the new run's count of queued turns never
    /// wraps.
    #[test]
    fn a_worker_a_stop_left_behind_never_touches_the_next_run() {
        /// What each turn of [`a_worker_a_stop_left_behind_never_touches_the_next_run`] saw: turns
        /// per index, and the most at once on one index in one run. Across a stop, the stopped
        /// run's turn may still run beside the new run's: it publishes nothing.
        #[derive(Default)]
        struct Seen {
            turns: HashMap<IndexId, usize>,
            running: HashMap<(usize, IndexId), usize>,
            most_on_one: usize,
        }
        let (a, b) = (column(EntityType::Node, "a"), column(EntityType::Node, "b"));
        let (in_turn, wait_in_turn) = mpsc::channel::<usize>();
        let gates: Vec<_> = (0..2).map(|_| mpsc::channel::<()>()).collect();
        let (open, gates): (Vec<_>, Vec<_>) = gates.into_iter().unzip();
        let gates = Mutex::new(gates.into_iter().map(Some).collect::<Vec<_>>());
        let seen = Arc::new(Mutex::new(Seen::default()));
        let in_seen = Arc::clone(&seen);
        let held = a.clone();
        let indexer = Indexer::new(Box::new(move |_, id, run| {
            let in_run = (std::ptr::from_ref(run) as usize, id.clone());
            let nth = {
                let mut seen = in_seen.lock().unwrap();
                let running = seen.running.entry(in_run.clone()).or_default();
                *running += 1;
                let running = *running;
                seen.most_on_one = seen.most_on_one.max(running);
                let turns = seen.turns.entry(id.clone()).or_default();
                *turns += 1;
                *turns
            };
            // The index's first two turns, one in each run, hold until the test opens their gate.
            if *id == held && nth <= 2 {
                let gate = gates.lock().unwrap()[nth - 1].take().unwrap();
                in_turn.send(nth).unwrap();
                gate.recv().unwrap();
            }
            *in_seen.lock().unwrap().running.get_mut(&in_run).unwrap() -= 1;
            Turn::Idle
        }));
        let host: Arc<dyn BuildHost> = Arc::new(NoGraph);

        indexer.start(1);
        indexer.enqueue(Arc::clone(&host), a.clone());
        assert_eq!(wait_in_turn.recv().unwrap(), 1);
        // The pool's only worker holds a's turn, so b's waits behind it on the pool.
        indexer.enqueue(Arc::clone(&host), b.clone());
        assert_eq!(indexer.pending(), 1);
        indexer.stop_within(Duration::from_millis(20));
        assert_eq!(indexer.pending(), 2, "both chains wait for the next start");

        indexer.start(2);
        assert_eq!(wait_in_turn.recv().unwrap(), 2, "the new run's turn of a");
        // The left-behind worker ends the stopped run's turn of a, then takes b's queued turn of
        // the stopped run.
        open[0].send(()).unwrap();
        thread::sleep(Duration::from_millis(100));
        assert_eq!(indexer.pending(), 0, "the count did not wrap");
        // A commit creates a again while the new run's turn of a still runs.
        indexer.enqueue(Arc::clone(&host), a.clone());
        thread::sleep(Duration::from_millis(50));
        open[1].send(()).unwrap();
        wait_until("every chain to end", || indexer.idle());
        indexer.stop_within(STOP_WAIT);

        let seen = seen.lock().unwrap();
        assert_eq!(
            seen.most_on_one, 1,
            "two chains of one index at once in one run"
        );
        assert_eq!(
            (seen.turns.get(&a).copied(), seen.turns.get(&b).copied()),
            (Some(3), Some(1)),
            "a: the stopped run's, the new run's, and its one more; b: the new run's only"
        );
    }
}
