//! Bookkeeping for background (online) index builds: which columns have a job in flight, and when
//! a column whose last attempt failed may be tried again.
//!
//! The sweep thread and the thread pool that drive these builds live in the module crate, next to
//! the graph registry. This half lives here because the module crate installs Redis's allocator and
//! therefore cannot run a unit test at all — and the rule this encodes (do not re-run a full BASE
//! scan on every sweep for a column that keeps losing the version slot) is worth testing.
//!
//! Every method takes `now` rather than reading the clock, so the backoff schedule can be asserted
//! without sleeping.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::entity_type::EntityType;

/// One in-flight build, identified by graph name + column + build epoch.
///
/// The **epoch** is part of the identity on purpose: a column dropped and re-created mid-build gets
/// a fresh epoch, so the new build is not suppressed by the old job's marker, and the old job
/// cannot publish into the new column.
pub type BuildKey = (String, EntityType, String, String, u64);

/// What the registry remembers about one column's build.
#[derive(Debug)]
enum BuildSlot {
    /// A job is dispatched or running; whoever dispatched it owns the entry until it exits.
    Running { attempts: u32 },
    /// The last attempt did not install — it lost the version slot, the graph was tearing down, or
    /// the epoch was stale. Re-dispatch is suppressed until `until`.
    Backoff { until: Instant, attempts: u32 },
}

/// Ceiling on the backoff doublings, so a permanently-contended column settles at
/// `interval << 5` (16s at a 500ms sweep) rather than drifting towards never.
const MAX_DOUBLINGS: u32 = 5;

/// Which columns are being built, and which are waiting out a retry backoff.
pub struct BuildRegistry {
    slots: Mutex<HashMap<BuildKey, BuildSlot>>,
    /// The dispatch cadence the backoff is expressed in multiples of — one doubling per failure.
    interval: Duration,
}

impl BuildRegistry {
    #[must_use]
    pub fn new(interval: Duration) -> Self {
        Self {
            slots: Mutex::new(HashMap::new()),
            interval,
        }
    }

    /// Claim the keys that should be dispatched now, marking each `Running`. Returns the subset to
    /// spawn: a key already running, or still inside its backoff window, is left out.
    ///
    /// The attempt count carries across a claim, so a column that keeps failing keeps widening its
    /// interval instead of restarting at one sweep.
    pub fn claim(
        &self,
        builds: Vec<BuildKey>,
        now: Instant,
    ) -> Vec<BuildKey> {
        let mut slots = self.slots.lock();
        builds
            .into_iter()
            .filter(|key| {
                let attempts = match slots.get(key) {
                    Some(BuildSlot::Running { .. }) => return false,
                    Some(BuildSlot::Backoff { until, .. }) if *until > now => return false,
                    Some(BuildSlot::Backoff { attempts, .. }) => *attempts,
                    None => 0,
                };
                slots.insert(key.clone(), BuildSlot::Running { attempts });
                true
            })
            .collect()
    }

    /// Give a key back when its job exits.
    ///
    /// `installed == true` means the base landed and the column is `Ready`; it will never be in the
    /// work list again, so the entry goes. Anything else — bail, stale epoch, lost version slot,
    /// panic, or a job the pool dropped without running — arms the backoff, because the sweep will
    /// otherwise re-dispatch the same full BASE scan on its very next pass.
    pub fn release(
        &self,
        key: &BuildKey,
        installed: bool,
        now: Instant,
    ) {
        let mut slots = self.slots.lock();
        if installed {
            slots.remove(key);
            return;
        }
        let attempts = match slots.get(key) {
            Some(BuildSlot::Running { attempts }) => attempts.saturating_add(1),
            _ => 1,
        };
        let until = now + self.interval * 2u32.pow(attempts.min(MAX_DOUBLINGS));
        slots.insert(key.clone(), BuildSlot::Backoff { until, attempts });
    }

    /// Forget the backoff of columns nobody is `Building` any more — a dropped index, a finished
    /// build, or an epoch superseded by a fresh one.
    ///
    /// `Running` entries are never pruned: their job owns the key and rewrites the entry when it
    /// exits. Without this the map keeps one entry per (column, epoch) that ever failed to install,
    /// for the life of the process. Call it from whoever can see *all* pending builds at once —
    /// pruning against one graph's work list would drop every other graph's entries.
    pub fn prune(
        &self,
        pending: &HashSet<BuildKey>,
    ) {
        self.slots
            .lock()
            .retain(|key, slot| matches!(slot, BuildSlot::Running { .. }) || pending.contains(key));
    }

    /// Number of tracked keys, running and backing off. Diagnostics and tests.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.lock().len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(epoch: u64) -> BuildKey {
        (
            "g".to_string(),
            EntityType::Node,
            "L".to_string(),
            "a".to_string(),
            epoch,
        )
    }

    const INTERVAL: Duration = Duration::from_millis(500);

    #[test]
    fn claims_once_then_suppresses_while_running() {
        let reg = BuildRegistry::new(INTERVAL);
        let now = Instant::now();
        assert_eq!(reg.claim(vec![key(1)], now), vec![key(1)]);
        assert!(
            reg.claim(vec![key(1)], now).is_empty(),
            "a running build must not be dispatched twice"
        );
    }

    #[test]
    fn a_fresh_epoch_is_not_suppressed_by_the_old_job() {
        let reg = BuildRegistry::new(INTERVAL);
        let now = Instant::now();
        reg.claim(vec![key(1)], now);
        assert_eq!(
            reg.claim(vec![key(2)], now),
            vec![key(2)],
            "dropping and re-creating the column must build again"
        );
    }

    /// The point of the backoff: a build that cannot take the version slot under sustained write
    /// load would otherwise re-run its whole BASE scan and encode every single sweep.
    #[test]
    fn failure_defers_the_retry_and_the_delay_doubles() {
        let reg = BuildRegistry::new(INTERVAL);
        let t0 = Instant::now();
        reg.claim(vec![key(1)], t0);
        reg.release(&key(1), false, t0);

        assert!(
            reg.claim(vec![key(1)], t0 + INTERVAL).is_empty(),
            "one sweep later is still inside the first backoff"
        );
        // First failure waits 2 intervals.
        assert_eq!(reg.claim(vec![key(1)], t0 + INTERVAL * 2), vec![key(1)]);

        // Second failure waits 4 — the count carried across the re-claim.
        let t1 = t0 + INTERVAL * 2;
        reg.release(&key(1), false, t1);
        assert!(reg.claim(vec![key(1)], t1 + INTERVAL * 3).is_empty());
        assert_eq!(reg.claim(vec![key(1)], t1 + INTERVAL * 4), vec![key(1)]);
    }

    #[test]
    fn backoff_is_capped() {
        let reg = BuildRegistry::new(INTERVAL);
        let mut t = Instant::now();
        // Well past MAX_DOUBLINGS.
        for _ in 0..12 {
            reg.claim(vec![key(1)], t);
            reg.release(&key(1), false, t);
            t += INTERVAL * (1 << MAX_DOUBLINGS);
        }
        assert_eq!(
            reg.claim(vec![key(1)], t + INTERVAL * (1 << MAX_DOUBLINGS)),
            vec![key(1)],
            "a permanently contended column must keep retrying at the capped interval"
        );
    }

    #[test]
    fn success_releases_the_key_outright() {
        let reg = BuildRegistry::new(INTERVAL);
        let now = Instant::now();
        reg.claim(vec![key(1)], now);
        reg.release(&key(1), true, now);
        assert!(reg.is_empty(), "a Ready column leaves nothing behind");
    }

    #[test]
    fn prune_reclaims_backoff_but_never_a_running_build() {
        let reg = BuildRegistry::new(INTERVAL);
        let now = Instant::now();
        reg.claim(vec![key(1), key(2)], now);
        reg.release(&key(1), false, now); // key(1) backing off, key(2) still running

        let pending: HashSet<BuildKey> = [key(1)].into_iter().collect();
        reg.prune(&pending);
        assert_eq!(reg.len(), 2, "still pending, and still running: both kept");

        // The index was dropped: neither is pending any more.
        reg.prune(&HashSet::new());
        assert_eq!(
            reg.len(),
            1,
            "the backoff entry is reclaimed; the running job still owns its key"
        );
    }
}
