//! Global string pool for `intern()` Cypher values.
//!
//! Returns a shared `ArcStr` for any given string so duplicate values
//! across nodes/edges/graphs occupy a single heap allocation. Reference
//! counting is provided by the `Arc` itself; the pool prunes entries whose
//! only remaining strong ref is the pool's own.

use arcstr::ArcStr;
use std::{
    collections::HashSet,
    sync::{Mutex, OnceLock},
};

pub struct StringPool {
    inner: Mutex<HashSet<ArcStr>>,
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
        a: ArcStr,
    ) -> ArcStr {
        // A static (the empty string, or a `literal!`) is already shared and
        // allocation-free, and has no count to prune by: nothing to pool.
        if ArcStr::is_static(&a) {
            return a;
        }
        let mut m = self.inner.lock().unwrap();
        if let Some(existing) = m.get(&*a) {
            return existing.clone();
        }
        let owned = ArcStr::from(a.as_str());
        m.insert(owned.clone());
        owned
    }

    /// True if `arc` is the canonical pool entry for its contents.
    pub fn is_interned(
        &self,
        arc: &ArcStr,
    ) -> bool {
        let m = self.inner.lock().unwrap();
        m.get(arc.as_str()).is_some_and(|k| ArcStr::ptr_eq(k, arc))
    }

    /// Snapshot of (`unique_count`, `avg_external_refs`). Prunes orphaned
    /// entries (those whose only strong ref is the pool's own) before counting.
    pub fn stats(&self) -> (u64, f64) {
        let mut m = self.inner.lock().unwrap();
        // `intern` never inserts a static, so every entry has a count.
        m.retain(|k| ArcStr::strong_count(k).is_some_and(|n| n > 1));
        let count = m.len() as u64;
        if count == 0 {
            return (0, 0.0);
        }
        let total: u64 = m
            .iter()
            .map(|k| ArcStr::strong_count(k).map_or(0, |n| n - 1) as u64)
            .sum();
        (count, total as f64 / count as f64)
    }
}

static POOL: OnceLock<StringPool> = OnceLock::new();

pub fn global() -> &'static StringPool {
    POOL.get_or_init(StringPool::new)
}

#[cfg(test)]
mod tests {
    use super::StringPool;
    use arcstr::ArcStr;

    #[test]
    fn the_empty_string_is_not_pooled_or_counted() {
        // `""` is arcstr's allocation-free static: pooling it saves nothing,
        // and it has no reference count, so a pool entry for it would never be
        // pruned and would count as an entry with no users.
        let pool = StringPool::new();
        let e = pool.intern(ArcStr::from(""));
        assert_eq!(e, "");
        assert_eq!(pool.stats(), (0, 0.0));
        drop(e);
        assert_eq!(pool.stats(), (0, 0.0));

        let s = pool.intern(ArcStr::from("tim"));
        assert!(pool.is_interned(&s));
        assert_eq!(pool.stats(), (1, 1.0));
        drop(s);
        assert_eq!(pool.stats(), (0, 0.0), "a dropped entry is pruned");
    }
}
