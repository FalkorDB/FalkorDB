//! The background index builder: builds every `Building` index, in every graph, in batches, on a
//! pool of its own.
//!
//! A commit that brings in a `Building` index starts a [`Job`] for it ([`MvccGraph::commit`]). A
//! job builds one generation of one index ([`IndexEntry::generation`]): it runs one batch on the
//! pool, and submits itself again while that generation is `Building`, unless the batch's version
//! was refused ([`Step::Failed`]). An index dropped, or dropped and created again, is another
//! generation, so its job ends, and the commit that created the new one started a job of its own.
//! Each generation is started once, so no two jobs build one index. A job holds its graph weakly:
//! once the graph is gone, so is the job. Nothing here knows about Redis: a batch publishes
//! through the builder's [`BuildHost`].
//!
//! **A batch holds no lock while it reads.** It reads on a new build version of the published one
//! ([`Graph::new_build_version`]), which only it holds, and then publishes through the host
//! ([`MvccGraph::commit_indexer_batch`]). What committed meanwhile decides how the batch goes in;
//! the rules writers and batches follow so that every published version stays correct, in whatever
//! order they commit, are on [`BuildState`](super::versioned::BuildState). A batch's commit keeps
//! the data version, so it never makes a writer's fork stale.
//!
//! **Stopping.** Only the Redis shutdown handler stops the builder, and only where the server tears
//! its modules down (`RS_GLOBAL_DTORS`: the sanitizer runs). [`stop`] waits a bounded time for the
//! pool's workers, since one may be waiting for the GIL, which the shutdown handler holds, and says
//! whether they all exited: the handler tears GraphBLAS down only then, so a worker left behind may
//! still free what it holds. A batch checks for the stop before it reads and again once it holds
//! what publishing needs: after the stop, one that has not started reading does not, and one that
//! waited to publish publishes nothing. One already reading goes on, inside the stop's wait or
//! left behind by it. The stop is for good.
//!
//! [`IndexEntry::generation`]: super::versioned::IndexEntry::generation
//! [`MvccGraph::commit`]: crate::graph::mvcc_graph::MvccGraph::commit
//! [`MvccGraph::commit_indexer_batch`]: crate::graph::mvcc_graph::MvccGraph::commit_indexer_batch

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::Duration;

use super::versioned::{Discarded, IndexId, IndexerCommit, PreparedBatch};
use crate::graph::graph::Graph;
use crate::graph::mvcc_graph::{CommitError, MvccGraph};
use crate::threadpool::ThreadPool;
#[cfg(test)]
use parking_lot::{FairMutex, FairMutexGuard, Mutex};

/// What the builder needs of the host its graphs live in: one for every graph, given to the builder
/// as it starts ([`start`]). The Redis module's takes the GIL; this crate's tests have one of
/// their own.
pub trait BuildHost: Send + Sync {
    /// Run `f`, which publishes a batch, holding what a writer holds to publish a version: the
    /// host's global lock.
    fn publish(
        &self,
        f: &mut dyn FnMut(),
    );

    /// Report what an operator should see: a batch whose version was refused, after which its
    /// index's build stops. The default writes to stderr; the Redis host logs it.
    fn warn(
        &self,
        message: &str,
    ) {
        eprintln!("{message}");
    }
}

/// The host of this crate's tests, without Redis: its global lock is a mutex of its own, in the
/// place of the GIL. A writer holds it to commit ([`Self::lock`]), and a batch publishes holding
/// it. Fair, so that writers committing back to back do not keep a batch out. Only tests use it,
/// until the crate runs on its own somewhere.
#[cfg(test)]
#[derive(Default)]
pub(crate) struct LocalHost {
    global: FairMutex<()>,
}

#[cfg(test)]
impl LocalHost {
    /// The global lock, which a writer holds to commit.
    pub(crate) fn lock(&self) -> FairMutexGuard<'_, ()> {
        self.global.lock()
    }
}

#[cfg(test)]
impl BuildHost for LocalHost {
    fn publish(
        &self,
        f: &mut dyn FnMut(),
    ) {
        let _global = self.lock();
        f();
    }
}

/// How a job sizes its batches ([`Sizing::batch`]).
///
/// A build goes in batches, not in one pass, because it runs while writers write. Each batch's
/// commit catches up, under the host's global lock, on what writers changed in its index while it
/// read, and holds what it read in memory until then: a batch bounds both. A job's batches grow:
/// each reads as many entities as the index's batches have read so far, between `floor` and the
/// job's limit ([`Job::limit`]). With no writers they go 10,000, 10,000, 20,000, 40,000, ...,
/// doubling, and reach the 4M cap on the 11th, once 5.12M entities are read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Sizing {
    /// The fewest entities a batch reads.
    pub(crate) floor: usize,
    /// The most entities a batch reads.
    pub(crate) cap: usize,
    /// The most writers' changes a batch larger than `floor` catches up on at its commit, under
    /// the host's locks ([`PreparedBatch::limit_catch_up`]). A batch that would catch up on more
    /// publishes nothing, and its job's next batch is half its size ([`Step::TooLarge`]).
    ///
    /// [`PreparedBatch::limit_catch_up`]: super::versioned::PreparedBatch::limit_catch_up
    pub(crate) catch_up_max: u64,
}

/// The builder's sizing.
///
/// A batch costs about the pages of its column it touches, and a batch of scattered values
/// touches most of them once the column is large. Batches of a fixed size make the build
/// quadratic: 4,988 and 11,798 instructions per entity at 1M and 4M nodes, with 10,000-entity
/// batches. A batch as large as what the build has read so far spreads each one over as many
/// entities as the column holds: 1,472, 1,488 and 1,499 at 1M, 4M and 8M.
/// - `floor`: 10,000 entities, the earlier online build's step (0742bcefb).
/// - `cap`: 4M entities. A batch holds its tuples until it commits, 16 bytes each, so 64 MB. At
///   8M it costs 1,499 per entity against 1,501 uncapped; a cap of 1M cost 1,736.
/// - `catch_up_max`: 10,000. Inserting 10,000 scattered tuples into an 8M-entity column holds
///   the locks for about 13 ms, and a batch at the floor never does more than that.
pub(crate) const SIZING: Sizing = Sizing {
    floor: 10_000,
    cap: 4_000_000,
    catch_up_max: 10_000,
};

impl Sizing {
    /// How many entities a batch reads, when its index's batches have read `read` and its job's
    /// [`Job::limit`] is `limit`: `read`, between `floor` and `limit`.
    pub(crate) fn batch(
        self,
        read: u64,
        limit: usize,
    ) -> usize {
        usize::try_from(read)
            .unwrap_or(usize::MAX)
            .min(limit)
            .max(self.floor)
    }
}

/// What a job's batch did, and so what the job does next ([`Job::step`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Step {
    /// The batch was published, and the job's generation is still `Building`. The next batch may
    /// be twice as large. Also when another batch on the index was published instead
    /// ([`Discarded::AnotherBatch`]): a bug in the one job per generation, which leaves nothing
    /// wrong.
    More,
    /// The batch published nothing: writers changed more of its index while it was read than its
    /// commit may catch up on ([`Sizing::catch_up_max`]). The next batch is half its size.
    TooLarge,
    /// The batch's version was refused, which only a bug makes happen, and the job ends, leaving
    /// its generation `Building`. Reads on the index keep scanning, as they do while it is built,
    /// and never reach another index. Writers keep adding the entities they write in it to the set
    /// a batch clears, and no batch comes any more: the set grows until the index is dropped, or
    /// the graph is deleted or loaded again, and each writer's commit that writes the index copies
    /// it. An RDB load builds the index without the builder. A `CREATE INDEX` after a drop, or
    /// replayed from an AOF or a replication stream, starts another job, which the same bug may
    /// refuse too. A job that ran again instead could fail forever, reading a batch and logging a
    /// warning each time.
    Failed,
    /// The job's generation is `Ready`: its build is over.
    Done,
    /// The job is over before its generation is `Ready`, for a reason outside it: the builder has
    /// stopped, the graph is gone, or the index was dropped, or dropped and created again, which
    /// is another generation that a job of its own builds.
    Cancelled,
}

/// Runs jobs on a pool of its own. The builder is one ([`start`], [`stop`], [`enqueue`]); tests
/// make their own.
pub(crate) struct Builder {
    sizing: Sizing,
    host: Arc<dyn BuildHost>,
    pool: OnceLock<ThreadPool>,
    /// Set by the stop, for good. A batch checks it before it reads and again before it publishes:
    /// one that has not started reading does not, and one that waited to publish publishes
    /// nothing.
    stopped: AtomicBool,
    /// Run once by the next batch while it reads, holding what it holds then: for a test that
    /// needs a batch held there.
    #[cfg(test)]
    pub(crate) while_reading: Mutex<Option<Box<dyn FnOnce() + Send>>>,
    /// Once set, the next batch's version is refused, as only a bug refuses one: for a test of
    /// what a job does then ([`Step::Failed`]).
    #[cfg(test)]
    pub(crate) refuse_publish: AtomicBool,
}

impl Builder {
    pub(crate) fn new(
        sizing: Sizing,
        host: Arc<dyn BuildHost>,
    ) -> Arc<Self> {
        Arc::new(Self {
            sizing,
            host,
            pool: OnceLock::new(),
            stopped: AtomicBool::new(false),
            #[cfg(test)]
            while_reading: Mutex::new(None),
            #[cfg(test)]
            refuse_publish: AtomicBool::new(false),
        })
    }

    /// Start the pool with `threads` workers (at least one). Once: a later start does nothing, and
    /// a start after the stop runs nothing.
    pub(crate) fn start(
        &self,
        threads: usize,
    ) {
        self.pool
            .get_or_init(|| ThreadPool::named(threads.max(1), "falkordb-index-build"));
    }

    /// Stop, for good, and wait up to `wait` for the pool's workers
    /// ([`ThreadPool::shutdown_within`]): one waiting for what its host needs to publish may be
    /// waiting on the caller, and is left behind. Whether every worker exited: one left behind may
    /// still be reading a graph version, or freeing what its batch holds.
    pub(crate) fn stop_within(
        &self,
        wait: Duration,
    ) -> bool {
        self.stopped.store(true, Ordering::SeqCst);
        self.pool
            .get()
            .is_none_or(|pool| pool.shutdown_within(wait))
    }

    /// Start the job that builds generation `generation` of index `id` of `graph`. Never blocks:
    /// the builder's is called from [`MvccGraph::commit`], under the host's global lock.
    ///
    /// [`MvccGraph::commit`]: crate::graph::mvcc_graph::MvccGraph::commit
    pub(crate) fn enqueue(
        self: &Arc<Self>,
        graph: Weak<MvccGraph>,
        id: IndexId,
        generation: u64,
    ) {
        self.submit(Job::new(Arc::clone(self), graph, id, generation));
    }

    /// Run `job`'s next batch on the pool. A builder takes jobs once started: the module's is
    /// started before any graph exists ([`start`]), and a test starts its own.
    fn submit(
        &self,
        job: Job,
    ) {
        debug_assert!(
            self.pool.get().is_some(),
            "a job was submitted before the builder started"
        );
        if let Some(pool) = self.pool.get() {
            pool.spawn(move || job.run(), None);
        }
    }

    /// Whether the builder was stopped ([`Self::stop_within`]).
    fn stopped(&self) -> bool {
        self.stopped.load(Ordering::SeqCst)
    }

    /// Publish a batch's version ([`MvccGraph::commit_indexer_batch`]). A test can have it
    /// refused.
    fn commit(
        &self,
        mvcc: &MvccGraph,
        prepared: Graph,
        batch: &PreparedBatch,
    ) -> Result<IndexerCommit, CommitError> {
        #[cfg(test)]
        if self.refuse_publish.swap(false, Ordering::SeqCst) {
            return Err(CommitError::Stale);
        }
        mvcc.commit_indexer_batch(prepared, batch)
    }
}

/// The build of one generation of one index: one batch at a time on its builder's pool, until that
/// generation is `Ready` or gone, or a batch's version is refused ([`Step::Failed`]).
pub(crate) struct Job {
    builder: Arc<Builder>,
    /// Held weakly, so the job does not keep a deleted graph alive: once the key is deleted and no
    /// query or batch still holds the graph, it is freed, and the job's next batch finds it gone
    /// and ends. A strong handle would keep the whole graph in memory, and the job building it,
    /// until its index is `Ready`.
    graph: Weak<MvccGraph>,
    id: IndexId,
    generation: u64,
    /// The most entities the next batch reads: the cap at first, half the size of a batch writers
    /// made too large to commit, and twice as many after each batch that published, up to the cap.
    pub(crate) limit: usize,
}

impl Job {
    pub(crate) fn new(
        builder: Arc<Builder>,
        graph: Weak<MvccGraph>,
        id: IndexId,
        generation: u64,
    ) -> Self {
        let limit = builder.sizing.cap;
        Self {
            builder,
            graph,
            id,
            generation,
            limit,
        }
    }

    /// One batch, on a worker of the pool, then the job submits itself again unless it is over.
    fn run(mut self) {
        match self.step() {
            Step::More | Step::TooLarge => {
                let builder = Arc::clone(&self.builder);
                builder.submit(self);
            }
            Step::Done | Step::Cancelled | Step::Failed => {}
        }
    }

    /// One batch of the job's index, published: as many entities as the index's batches have read,
    /// between the floor and [`Self::limit`] ([`Sizing::batch`]), which the batch then moves. A
    /// batch larger than the floor publishes nothing if its commit would catch up on more than
    /// [`Sizing::catch_up_max`] of writers' changes ([`Step::TooLarge`]).
    pub(crate) fn step(&mut self) -> Step {
        let sizing = self.builder.sizing;
        let Read {
            mvcc,
            prepared,
            batch,
            entities,
        } = match self.read() {
            Ok(read) => read,
            Err(step) => return step,
        };
        let mut step = Step::Cancelled;
        let mut prepared = Some(prepared);
        let (builder, host, id, generation) =
            (&self.builder, &self.builder.host, &self.id, self.generation);
        host.publish(&mut || {
            // A batch that waited for what publishing needs while the builder stopped publishes
            // nothing.
            if builder.stopped() {
                return;
            }
            let Some(prepared) = prepared.take() else {
                return;
            };
            step = match builder.commit(&mvcc, prepared, &batch) {
                Ok(IndexerCommit::Discarded(Discarded::CatchUpTooLarge)) => Step::TooLarge,
                Ok(_) => standing(&mvcc.read(), id, generation).unwrap_or(Step::More),
                // Validation refused the version, or another was published while this commit ran,
                // which only a host publishing without its global lock allows.
                Err(e) => {
                    host.warn(&format!(
                        "native index: an indexer batch of {id:?} was not published ({e}); its \
                         build has stopped, and the index is left under construction, so reads \
                         on it scan. Loading the graph from an RDB builds it."
                    ));
                    Step::Failed
                }
            };
        });
        self.limit = match step {
            Step::More => self.limit.saturating_mul(2).min(sizing.cap),
            Step::TooLarge => (entities / 2).max(sizing.floor),
            Step::Failed | Step::Done | Step::Cancelled => self.limit,
        };
        step
    }

    /// The batch's read, with no lock of the host's held, or what the job does instead if there is
    /// nothing to read.
    fn read(&self) -> Result<Read, Step> {
        let sizing = self.builder.sizing;
        if self.builder.stopped() {
            return Err(Step::Cancelled);
        }
        let Some(mvcc) = self.graph.upgrade() else {
            return Err(Step::Cancelled);
        };
        let mut prepared = mvcc.read().new_build_version();
        if let Some(step) = standing(&prepared, &self.id, self.generation) {
            return Err(step);
        }
        let read = prepared.index_build_read(&self.id).unwrap_or(0);
        let entities = sizing.batch(read, self.limit);
        // Never `None`: the index is `Building` in this version.
        let Some(mut batch) = prepared.indexer_prepare(&self.id, entities) else {
            return Err(Step::Cancelled);
        };
        // A batch at the floor catches up on whatever writers changed: none can be smaller.
        if entities > sizing.floor {
            batch.limit_catch_up(sizing.catch_up_max);
        }
        #[cfg(test)]
        if let Some(hold) = self.builder.while_reading.lock().take() {
            hold();
        }
        Ok(Read {
            mvcc,
            prepared,
            batch,
            entities,
        })
    }
}

/// What a batch's read hands its publish ([`Job::read`]).
struct Read {
    /// The job's graph, held until the batch is over.
    mvcc: Arc<MvccGraph>,
    /// A new build version of the graph's published one, with the batch applied.
    prepared: Graph,
    batch: PreparedBatch,
    /// How many entities the batch was to read ([`Sizing::batch`]): the last batch reads fewer.
    entities: usize,
}

/// What a job for generation `generation` of index `id` does next, as `graph` holds it: `None`
/// while that generation is `Building`; [`Step::Done`] once it is `Ready`; [`Step::Cancelled`] if
/// the index was dropped, or holds another generation.
fn standing(
    graph: &Graph,
    id: &IndexId,
    generation: u64,
) -> Option<Step> {
    match graph.index_generation(id) {
        Some((g, true)) if g == generation => None,
        Some((g, false)) if g == generation => Some(Step::Done),
        _ => Some(Step::Cancelled),
    }
}

/// How long [`stop`] waits for the builder's workers.
const STOP_WAIT: Duration = Duration::from_secs(2);

/// The builder: the one every graph's commits start jobs on, once started.
static BUILDER: OnceLock<Arc<Builder>> = OnceLock::new();

/// Start the builder: its pool, with `threads` workers (at least one), and the host its batches
/// publish through. The module calls it as it loads, before any graph exists. Once: a later start
/// does nothing.
pub fn start(
    threads: usize,
    host: Arc<dyn BuildHost>,
) {
    BUILDER
        .get_or_init(|| Builder::new(SIZING, host))
        .start(threads);
}

/// Stop the builder, for good ([`Builder::stop_within`]): its workers get [`STOP_WAIT`], since the
/// Redis shutdown handler calls this holding the GIL. Whether every worker exited: GraphBLAS may be
/// torn down only then.
#[must_use]
pub fn stop() -> bool {
    BUILDER
        .get()
        .is_none_or(|builder| builder.stop_within(STOP_WAIT))
}

/// Start the build of generation `generation` of index `id` of `graph` ([`Builder::enqueue`]).
/// Before the builder starts, nothing: the module starts it before any graph exists, and no unit
/// test starts it. Tests share a process, so its jobs would build other tests' indexes while they
/// run; a test that runs jobs makes a builder of its own.
pub(crate) fn enqueue(
    graph: Weak<MvccGraph>,
    id: IndexId,
    generation: u64,
) {
    if let Some(builder) = BUILDER.get() {
        builder.enqueue(graph, id, generation);
    }
}

#[cfg(test)]
impl Sizing {
    /// Every batch reads `n` entities, and none limits its catch-up: for tests that set where
    /// batches stop.
    pub(crate) const fn fixed(n: usize) -> Self {
        Self {
            floor: n,
            cap: n,
            catch_up_max: u64::MAX,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;

    /// A batch reads what its index's batches have read, between the floor and its job's limit.
    /// A fixed sizing reads its size whatever the count.
    #[test]
    fn a_batch_reads_what_its_index_has_read_between_the_floor_and_its_limit() {
        let sizing = Sizing {
            floor: 10,
            cap: 1_000,
            catch_up_max: 0,
        };
        for (read, limit, batch) in [
            (0, 1_000, 10),
            (9, 1_000, 10),
            (400, 1_000, 400),
            (400, 200, 200),
            (400, 10, 10),
            (5_000, 1_000, 1_000),
            (5_000, 500, 500),
            (u64::MAX, 1_000, 1_000),
        ] {
            assert_eq!(
                sizing.batch(read, limit),
                batch,
                "{read} read, limit {limit}"
            );
        }
        assert_eq!(Sizing::fixed(7).batch(5_000, 7), 7);
        assert_eq!(Sizing::fixed(7).batch(0, 7), 7);
    }

    /// A batch publishes on a local host holding its global lock: not while a writer holds it.
    #[test]
    fn a_local_host_publishes_holding_its_global_lock() {
        let host = LocalHost::default();
        let published = AtomicBool::new(false);
        thread::scope(|scope| {
            let writer = host.lock();
            scope.spawn(|| host.publish(&mut || published.store(true, Ordering::SeqCst)));
            thread::sleep(Duration::from_millis(100));
            assert!(
                !published.load(Ordering::SeqCst),
                "published while a writer held the lock"
            );
            drop(writer);
        });
        assert!(published.load(Ordering::SeqCst));
    }
}
