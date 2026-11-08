//! The background index builder: builds every `Building` index, in every graph, in batches, on a
//! pool of its own.
//!
//! A commit that brings in a `Building` index starts a [`Job`] for it ([`MvccGraph::commit`]), and
//! so does registering a graph's host, for each index its published version holds `Building`: one
//! a commit brought in before any host was registered ([`MvccGraph::set_build_host`]). A job builds
//! one generation of one index ([`IndexEntry::generation`]): it runs one batch on the pool, and
//! submits itself again while that generation is `Building`. An index dropped, or dropped and
//! created again, is another generation, so its job ends, and the commit that created the new one
//! started a job of its own. Each generation is started once, so no two jobs build one index.
//! Nothing here knows about Redis: a job reaches its graph through a [`BuildHost`].
//!
//! **A batch holds no lock while it reads.** It reads on a new build version of the published one
//! ([`Graph::new_build_version`]), which only it holds, and then publishes through the host
//! ([`MvccGraph::commit_indexer_batch`]). What committed meanwhile decides how the batch goes in;
//! the rules writers and batches follow so that every published version stays correct, in whatever
//! order they commit, are on [`BuildState`](super::versioned::BuildState). A batch's commit keeps
//! the data version, so it never makes a writer's fork stale.
//!
//! **Stopping.** Only the Redis shutdown handler stops the builder, and only where the server tears
//! its modules down (`RS_GLOBAL_DTORS`: the sanitizer runs), and GraphBLAS is torn down next.
//! [`stop`] waits for every batch still holding a graph version with no lock held, then a bounded
//! time for the pool's workers: one may be waiting for the GIL, which the shutdown handler holds.
//! A batch checks the builder is still running once it holds what publishing needs, so one that
//! waited publishes nothing. The stop is for good.
//!
//! [`IndexEntry::generation`]: super::versioned::IndexEntry::generation

use std::mem;
use std::sync::{Arc, LazyLock, OnceLock};
use std::thread;
use std::time::Duration;

use parking_lot::{RwLock, RwLockReadGuard};

use super::versioned::{Discarded, IndexId, IndexerCommit};
use crate::graph::graph::Graph;
use crate::graph::mvcc_graph::MvccGraph;
use crate::threadpool::ThreadPool;

/// How the builder reaches one graph. Implemented by the host that owns the graph, and registered
/// with it through [`MvccGraph::set_build_host`].
pub trait BuildHost: Send + Sync {
    /// The graph's published version, loaded under whatever guards the host's graph and released
    /// before the version is read: a published version is never changed in place, so it may be
    /// read with no lock held. `None` if the graph is gone.
    fn read(&self) -> Option<Arc<Graph>>;

    /// Run `f` on the graph's coordinator with what a writer holds to publish a version: the host's
    /// global lock (the Redis host takes the GIL, then the graph's read lock to reach the
    /// coordinator). `false` if the host refuses: the graph is gone.
    fn publish(
        &self,
        f: &mut dyn FnMut(&MvccGraph),
    ) -> bool;

    /// Report what an operator should see: a batch that failed, which the builder retries. The
    /// default writes to stderr; the Redis host logs it.
    fn warn(
        &self,
        message: &str,
    ) {
        eprintln!("{message}");
    }
}

/// How a job sizes its batches ([`Sizing::batch`]).
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
    /// The batch's version was refused, which only a bug makes happen. The job runs again after
    /// [`FAILED_RETRY`].
    Retry,
    /// The job is over: its generation is `Ready`, dropped, or replaced by another, its graph is
    /// gone, or the builder has stopped.
    Done,
}

/// Runs jobs on a pool of its own. The builder is one ([`start`], [`stop`], [`enqueue`]); tests
/// make their own.
pub(crate) struct Builder {
    sizing: Sizing,
    pool: OnceLock<ThreadPool>,
    /// `false` once stopped, after which GraphBLAS may be torn down. A batch holds it read while it
    /// holds a graph version with no lock held, so the stop, which takes it to write, waits for
    /// those batches.
    running: RwLock<bool>,
}

impl Builder {
    pub(crate) fn new(sizing: Sizing) -> Arc<Self> {
        Arc::new(Self {
            sizing,
            pool: OnceLock::new(),
            running: RwLock::new(true),
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

    /// Stop, for good. Waits, with no limit, for every batch still holding a graph version with no
    /// lock held: none waits on a lock meanwhile. Then waits up to `wait` for the pool's workers
    /// ([`ThreadPool::shutdown_within`]): one waiting for what its host needs to publish may be
    /// waiting on the caller, and is left behind.
    pub(crate) fn stop_within(
        &self,
        wait: Duration,
    ) {
        *self.running.write() = false;
        if let Some(pool) = self.pool.get() {
            pool.shutdown_within(wait);
        }
    }

    /// Start the job that builds generation `generation` of index `id` of `host`'s graph. Never
    /// blocks: the builder's is called from [`MvccGraph::commit`], under the host's global lock.
    pub(crate) fn enqueue(
        self: &Arc<Self>,
        host: Arc<dyn BuildHost>,
        id: IndexId,
        generation: u64,
    ) {
        self.submit(Job::new(Arc::clone(self), host, id, generation));
    }

    /// Run `job`'s next batch on the pool. The module starts the builder as it loads, before any
    /// graph exists, so a job always finds the pool.
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

    /// What a batch holds while it holds a graph version with no lock held: `None` once stopped.
    fn running(&self) -> Option<RwLockReadGuard<'_, bool>> {
        let running = self.running.read();
        (*running).then_some(running)
    }
}

/// The build of one generation of one index: one batch at a time on its builder's pool, until that
/// generation is `Ready` or gone.
pub(crate) struct Job {
    builder: Arc<Builder>,
    host: Arc<dyn BuildHost>,
    id: IndexId,
    generation: u64,
    /// The most entities the next batch reads: the cap at first, half the size of a batch writers
    /// made too large to commit, and twice as many after each batch that published, up to the cap.
    pub(crate) limit: usize,
}

impl Job {
    pub(crate) fn new(
        builder: Arc<Builder>,
        host: Arc<dyn BuildHost>,
        id: IndexId,
        generation: u64,
    ) -> Self {
        let limit = builder.sizing.cap;
        Self {
            builder,
            host,
            id,
            generation,
            limit,
        }
    }

    /// One batch, on a worker of the pool, then the job submits itself again unless it is done.
    fn run(mut self) {
        let step = self.step();
        if step == Step::Done {
            return;
        }
        if step == Step::Retry {
            // Only after a bug: holding this worker for it costs nothing that matters.
            thread::sleep(FAILED_RETRY);
        }
        let builder = Arc::clone(&self.builder);
        builder.submit(self);
    }

    /// One batch of the job's index, published: as many entities as the index's batches have read,
    /// between the floor and [`Self::limit`] ([`Sizing::batch`]), which the batch then moves. A
    /// batch larger than the floor publishes nothing if its commit would catch up on more than
    /// [`Sizing::catch_up_max`] of writers' changes ([`Step::TooLarge`]).
    pub(crate) fn step(&mut self) -> Step {
        let sizing = self.builder.sizing;
        // Declared first, so dropped last: on an early return, the versions this batch holds are
        // freed while it still holds this, so GraphBLAS is not torn down under them.
        let Some(running) = self.builder.running() else {
            return Step::Done;
        };
        let Some(published) = self.host.read() else {
            return Step::Done;
        };
        let mut prepared = published.new_build_version();
        drop(published);
        if prepared.index_build_generation(&self.id) != Some(self.generation) {
            return Step::Done;
        }
        let read = prepared.index_build_read(&self.id).unwrap_or(0);
        let entities = sizing.batch(read, self.limit);
        let Some(mut batch) = prepared.indexer_prepare(&self.id, entities) else {
            return Step::Done;
        };
        // A batch at the floor catches up on whatever writers changed: none can be smaller.
        if entities > sizing.floor {
            batch.limit_catch_up(sizing.catch_up_max);
        }
        // Not held while publishing: the Redis shutdown handler stops the builder holding the GIL.
        drop(running);

        let mut step = Step::Done;
        // Held until the host has let go of what it holds, so that freeing the version the commit
        // replaces, or what the batch built if it did not publish it, never happens under the
        // host's locks. The commit takes `prepared`, and frees it there unless it publishes it as
        // it is; `built` shares the index pages the batch made with it (a clone of an index copies
        // no pages), so those are freed with `built`, after.
        let built = prepared.falkordb_index.clone();
        let mut prepared = Some(prepared);
        let mut replaced = None;
        let (builder, host, id, generation) =
            (&self.builder, &self.host, &self.id, self.generation);
        host.publish(&mut |mvcc| {
            // Again, now that the host holds what publishing needs: a batch that waited for it
            // while the builder stopped publishes nothing.
            if builder.running().is_none() {
                return;
            }
            let Some(prepared) = prepared.take() else {
                return;
            };
            replaced = Some(mvcc.read());
            step = match mvcc.commit_indexer_batch(prepared, &batch) {
                Ok(IndexerCommit::Discarded(Discarded::CatchUpTooLarge)) => Step::TooLarge,
                Ok(_) if mvcc.read().index_build_generation(id) == Some(generation) => Step::More,
                Ok(_) => Step::Done,
                // Validation refused the version, or another was published while this commit ran,
                // which only a host publishing without its global lock allows.
                Err(e) => {
                    host.warn(&format!(
                        "native index: an indexer batch of {id:?} was not published ({e}); it \
                         will be retried"
                    ));
                    Step::Retry
                }
            };
        });
        // The versions this batch still holds are freed holding `running` again, or, once the
        // builder has stopped, left to the exiting process: GraphBLAS may be torn down by then.
        match self.builder.running() {
            Some(_running) => {
                drop(replaced);
                drop(built);
                drop(prepared);
            }
            None => {
                mem::forget(replaced);
                mem::forget(built);
                mem::forget(prepared);
            }
        }
        self.limit = match step {
            Step::More => self.limit.saturating_mul(2).min(sizing.cap),
            Step::TooLarge => (entities / 2).max(sizing.floor),
            Step::Retry | Step::Done => self.limit,
        };
        step
    }
}

/// How long a job waits before running again after a batch whose version was refused.
const FAILED_RETRY: Duration = Duration::from_secs(1);

/// How long [`stop`] waits for the builder's workers.
const STOP_WAIT: Duration = Duration::from_secs(2);

/// The builder: the one every graph's commits start jobs on.
static BUILDER: LazyLock<Arc<Builder>> = LazyLock::new(|| Builder::new(SIZING));

/// Start the builder's pool with `threads` workers (at least one). The module calls it as it
/// loads, before any graph exists.
pub fn start(threads: usize) {
    BUILDER.start(threads);
}

/// Stop the builder, for good ([`Builder::stop_within`]): its workers get [`STOP_WAIT`], since the
/// Redis shutdown handler calls this holding the GIL.
pub fn stop() {
    BUILDER.stop_within(STOP_WAIT);
}

/// Start the build of generation `generation` of index `id` of `host`'s graph
/// ([`Builder::enqueue`]).
pub(crate) fn enqueue(
    host: Arc<dyn BuildHost>,
    id: IndexId,
    generation: u64,
) {
    BUILDER.enqueue(host, id, generation);
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::entity_type::EntityType;

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

    /// A host that counts the times a batch reads its graph, which it never has.
    #[derive(Default)]
    struct Counted(AtomicUsize);

    impl BuildHost for Counted {
        fn read(&self) -> Option<Arc<Graph>> {
            self.0.fetch_add(1, Ordering::SeqCst);
            None
        }

        fn publish(
            &self,
            _: &mut dyn FnMut(&MvccGraph),
        ) -> bool {
            false
        }
    }

    /// A stop is for good: a job started after it, or run by hand, never reads its graph, and a
    /// start after it runs nothing.
    #[test]
    fn a_stopped_builder_runs_nothing() {
        let builder = Builder::new(Sizing::fixed(1));
        builder.start(1);
        builder.stop_within(STOP_WAIT);
        let host = Arc::new(Counted::default());
        let id = IndexId::Column(
            EntityType::Node,
            (Arc::new("L".to_string()), Arc::new("v".to_string())),
        );
        let mut job = Job::new(
            Arc::clone(&builder),
            Arc::clone(&host) as Arc<dyn BuildHost>,
            id.clone(),
            0,
        );
        assert_eq!(job.step(), Step::Done);
        builder.start(1);
        builder.enqueue(Arc::clone(&host) as Arc<dyn BuildHost>, id, 0);
        thread::sleep(Duration::from_millis(100));
        assert_eq!(
            host.0.load(Ordering::SeqCst),
            0,
            "a stopped builder read a graph"
        );
    }
}
