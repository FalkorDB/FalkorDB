//! Global string pool for `intern()` Cypher values.
//!
//! Returns a shared `Arc<String>` for any given string so duplicate values
//! across nodes/edges/graphs occupy a single heap allocation. Reference
//! counting is provided by the `Arc` itself; the pool prunes entries whose
//! only remaining strong ref is the pool's own.

use std::{
    collections::HashSet,
    sync::{Arc, Mutex, OnceLock},
};

pub struct StringPool {
    inner: Mutex<HashSet<Arc<String>>>,
}

impl StringPool {
    fn new() -> Self {
        Self {
            inner: Mutex::new(HashSet::new()),
        }
    }

    /// Intern by string content. The returned Arc is owned by the pool;
    /// the caller's input is not retained, so external strong references
    /// (e.g. plan-cache constants) do not pin pool entries.
    #[allow(clippy::needless_pass_by_value)]
    pub fn intern(
        &self,
        a: Arc<String>,
    ) -> Arc<String> {
        let mut m = self.inner.lock().unwrap();
        if let Some(existing) = m.get(&*a) {
            return existing.clone();
        }
        let owned = Arc::new((*a).clone());
        m.insert(owned.clone());
        owned
    }

    /// True if `arc` is the canonical pool entry for its contents.
    pub fn is_interned(
        &self,
        arc: &Arc<String>,
    ) -> bool {
        let m = self.inner.lock().unwrap();
        m.get(&**arc).is_some_and(|k| Arc::ptr_eq(k, arc))
    }

    /// Snapshot of (`unique_count`, `avg_external_refs`). Prunes orphaned
    /// entries (those whose only strong ref is the pool's own) before counting.
    pub fn stats(&self) -> (u64, f64) {
        let mut m = self.inner.lock().unwrap();
        m.retain(|k| Arc::strong_count(k) > 1);
        let count = m.len() as u64;
        if count == 0 {
            return (0, 0.0);
        }
        let total: u64 = m.iter().map(|k| (Arc::strong_count(k) - 1) as u64).sum();
        (count, total as f64 / count as f64)
    }
}

static POOL: OnceLock<StringPool> = OnceLock::new();

pub fn global() -> &'static StringPool {
    POOL.get_or_init(StringPool::new)
}
