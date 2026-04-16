//! In-memory concurrent cache for entity attributes.
//!
//! Provides a shared, version-stamped, entity-level cache that sits between
//! [`super::attribute_store::AttributeStore`] and fjall.  Writes go to the
//! cache first (marked dirty); fjall is updated asynchronously when the
//! memory threshold is exceeded.
//!
//! ## Cache Entry Structure
//!
//! ```text
//!  CachedEntity
//!  ┌─────────────────────────────────────────────┐
//!  │ attrs: [(0, "Alice"), (1, 30), (2, "NYC")]  │  sorted by attr_idx
//!  │ version: 5                                  │  MVCC version stamp
//!  │ dirty: true                                 │  not yet flushed to fjall
//!  └─────────────────────────────────────────────┘
//! ```
//!
//! ## Lookup Flow
//!
//! ```text
//!  get_attr(entity_id=42, attr_idx=1, reader_version=4)
//!       │
//!       ▼
//!  quick_cache lookup by entity_id
//!       │
//!       ├── miss ──▶ return None (caller falls back to fjall)
//!       │
//!       ▼
//!  entry.version > reader_version?
//!       │
//!       ├── yes ──▶ return None (uncommitted write, invisible to reader)
//!       │
//!       ▼
//!  binary search attrs for attr_idx
//!       │
//!       ├── found ──▶ return Some(Some(value))
//!       └── not found ──▶ return Some(None)  (entity cached, attr absent)
//! ```
//!
//! ## Dirty Pinning
//!
//! Dirty entries (writes not yet flushed to fjall) are pinned by the
//! `DirtyPinLifecycle` so that `quick_cache` will not evict them during
//! normal LRU eviction. This prevents data loss -- dirty entries can only
//! be removed explicitly via `collect_dirty_lru` (which hands them to the
//! caller for flushing) or `invalidate` (used during rollback).
//!
//! ## Sharing Across MVCC Versions
//!
//! The cache is shared across MVCC versions via `Arc<AttributeCache>` so
//! that `AttributeStore::new_version()` costs only a pointer increment.
//! Version stamps on each entry ensure that readers with older MVCC
//! versions do not see uncommitted writes from a newer transaction.
//!
//! ## Implementation
//!
//! Uses [`quick_cache`] internally -- a sharded, lock-free concurrent cache
//! with CLOCK-based approximate LRU eviction and byte-weighted capacity.
//! The default budget is 2 GiB per attribute store (nodes and relationships
//! each get their own cache).

use std::sync::Arc;

use quick_cache::sync::Cache;
use quick_cache::{DefaultHashBuilder, Lifecycle, Weighter};

use crate::runtime::value::Value;

/// Per-entity cached attributes.
#[derive(Clone)]
struct CachedEntity {
    /// Sorted by `attr_idx` for O(log n) binary-search lookups.
    /// Wrapped in Arc so `quick_cache::get()` clone is O(1) refcount bump
    /// instead of a full Vec heap allocation.
    attrs: Arc<Vec<(u16, Value)>>,
    /// Graph version when this entry was written/populated.
    version: u64,
    /// `true` when the entry has not yet been flushed to fjall.
    dirty: bool,
}

/// Weighter that estimates the heap footprint of each cached entity.
#[derive(Clone)]
struct EntityWeighter;

impl Weighter<u64, CachedEntity> for EntityWeighter {
    fn weight(
        &self,
        _key: &u64,
        val: &CachedEntity,
    ) -> u64 {
        let base = val.attrs.len() * (std::mem::size_of::<u16>() + std::mem::size_of::<Value>());
        let heap: usize = val.attrs.iter().map(|(_, v)| v.heap_size()).sum();
        // Minimum weight of 1 to satisfy quick_cache invariant.
        // Include Arc overhead.
        (base
            + heap
            + std::mem::size_of::<CachedEntity>()
            + std::mem::size_of::<Arc<Vec<(u16, Value)>>>())
        .max(1) as u64
    }
}

/// Lifecycle that pins dirty entries so they cannot be evicted before flushing.
#[derive(Clone, Default)]
struct DirtyPinLifecycle;

impl Lifecycle<u64, CachedEntity> for DirtyPinLifecycle {
    type RequestState = [Option<(u64, CachedEntity)>; 2];

    #[inline]
    fn is_pinned(
        &self,
        _key: &u64,
        val: &CachedEntity,
    ) -> bool {
        val.dirty
    }

    #[inline]
    fn begin_request(&self) -> Self::RequestState {
        [None, None]
    }

    #[inline]
    fn on_evict(
        &self,
        state: &mut Self::RequestState,
        key: u64,
        val: CachedEntity,
    ) {
        if state[0].is_none() {
            state[0] = Some((key, val));
        } else if state[1].is_none() {
            state[1] = Some((key, val));
        }
    }
}

/// Shared, version-stamped, entity-level attribute cache.
///
/// Thread-safety is handled internally by `quick_cache` via sharded locks —
/// concurrent reads do not block each other.
pub struct AttributeCache {
    entries: Cache<u64, CachedEntity, EntityWeighter, DefaultHashBuilder, DirtyPinLifecycle>,
}

impl AttributeCache {
    /// Create a new cache with the given byte budget.
    #[must_use]
    pub fn new(max_bytes: usize) -> Self {
        // Use a modest item estimate — quick_cache grows internally as needed.
        // The weight_capacity is the actual byte budget that governs eviction.
        Self {
            entries: Cache::with(
                1024,
                max_bytes as u64,
                EntityWeighter,
                DefaultHashBuilder::default(),
                DirtyPinLifecycle,
            ),
        }
    }

    /// Look up a single attribute for an entity by index.
    ///
    /// Returns `Some(Some(value))` on cache hit with the attribute present,
    /// `Some(None)` on cache hit but attribute absent, and `None` on cache
    /// miss.
    #[must_use]
    pub fn get_attr(
        &self,
        entity_id: u64,
        attr_idx: u16,
        version: u64,
    ) -> Option<Option<Value>> {
        let entry = self.entries.get(&entity_id)?;
        if entry.version > version {
            return None; // newer uncommitted data — ignore
        }
        Some(
            entry
                .attrs
                .binary_search_by_key(&attr_idx, |(idx, _)| *idx)
                .ok()
                .map(|pos| entry.attrs[pos].1.clone()),
        )
    }

    /// Return all cached attributes for an entity.
    ///
    /// Returns `None` on cache miss or version mismatch.
    /// The returned `Arc` avoids cloning the underlying Vec.
    #[must_use]
    pub fn get_entity(
        &self,
        entity_id: u64,
        version: u64,
    ) -> Option<Arc<Vec<(u16, Value)>>> {
        let entry = self.entries.get(&entity_id)?;
        if entry.version > version {
            return None;
        }
        Some(entry.attrs)
    }

    /// Return all cached attributes for an entity along with the dirty flag.
    ///
    /// Returns `None` on cache miss or version mismatch.
    #[must_use]
    pub fn get_entity_with_dirty(
        &self,
        entity_id: u64,
        version: u64,
    ) -> Option<(Arc<Vec<(u16, Value)>>, bool)> {
        let entry = self.entries.get(&entity_id)?;
        if entry.version > version {
            return None;
        }
        Some((entry.attrs, entry.dirty))
    }

    /// Check whether an entity has *any* cached attributes.
    #[must_use]
    pub fn has_entity(
        &self,
        entity_id: u64,
        version: u64,
    ) -> Option<bool> {
        let entry = self.entries.get(&entity_id)?;
        if entry.version > version {
            return None;
        }
        Some(!entry.attrs.is_empty())
    }

    /// Insert (or replace) the full attribute set for an entity.
    ///
    /// The incoming `attrs` are sorted by `attr_idx` before storing to maintain
    /// the invariant required by binary searches in `get_attr`, `contains_attr`,
    /// and other methods that rely on sorted access.
    pub fn insert_entity(
        &self,
        entity_id: u64,
        mut attrs: Vec<(u16, Value)>,
        version: u64,
        dirty: bool,
    ) {
        // Ensure attrs are sorted by attr_idx to support binary searches.
        attrs.sort_by_key(|item| item.0);
        let entry = CachedEntity {
            attrs: Arc::new(attrs),
            version,
            dirty,
        };
        self.entries.insert(entity_id, entry);
    }

    /// Insert (or replace) the full attribute set for an entity when the
    /// caller guarantees the attrs are already sorted by `attr_idx`.
    ///
    /// This avoids the O(n log n) sort in [`insert_entity`] for callers
    /// (like `insert_attrs` merge) that produce sorted output.
    pub fn insert_entity_presorted(
        &self,
        entity_id: u64,
        attrs: Vec<(u16, Value)>,
        version: u64,
        dirty: bool,
    ) {
        debug_assert!(
            attrs.windows(2).all(|w| w[0].0 <= w[1].0),
            "insert_entity_presorted: attrs not sorted"
        );
        let entry = CachedEntity {
            attrs: Arc::new(attrs),
            version,
            dirty,
        };
        self.entries.insert(entity_id, entry);
    }

    /// Insert or update a cache entry only if not overwriting a newer or dirty entry.
    ///
    /// This is used by `populate_cache_from_fjall` to safely cache fjall reads without
    /// overwriting in-flight dirty writes. The insert only proceeds if:
    /// - No cache entry exists for this entity, OR
    /// - The existing entry is from an older version AND not dirty
    ///
    /// Returns `true` if the insert proceeded, `false` if it was skipped due to
    /// a newer/dirty entry already in cache.
    ///
    /// **Note:** There is a narrow TOCTOU window between the `get` check and the
    /// subsequent `insert_entity` call.  A concurrent writer could insert a dirty
    /// entry in that gap, which this clean insert would then overwrite.  In
    /// practice this is benign because (a) the window is sub-microsecond,
    /// (b) writers are serialized so only one write transaction is active, and
    /// (c) readers always hold an older MVCC version so the version check
    /// (`>=`) catches the common case.
    #[must_use]
    pub fn insert_entity_if_older(
        &self,
        entity_id: u64,
        attrs: Vec<(u16, Value)>,
        version: u64,
    ) -> bool {
        // Check if a newer or dirty entry already exists
        if let Some(existing) = self.entries.get(&entity_id) {
            // Skip if existing entry is newer or dirty (in-flight write)
            if existing.version >= version || existing.dirty {
                return false;
            }
        }
        // Safe to insert or update with this fjall read
        self.insert_entity(entity_id, attrs, version, false);
        true
    }

    /// Remove a single entity from the cache.
    pub fn invalidate(
        &self,
        entity_id: u64,
    ) {
        self.entries.remove(&entity_id);
    }

    /// Batch-invalidate entities (used during rollback).
    pub fn invalidate_batch(
        &self,
        entity_ids: &roaring::RoaringTreemap,
    ) {
        for id in entity_ids {
            self.entries.remove(&id);
        }
    }

    /// Check whether an attr already exists for an entity in the cache.
    ///
    /// Returns `Some(true/false)` on cache hit, `None` on miss.
    #[must_use]
    pub fn contains_attr(
        &self,
        entity_id: u64,
        attr_idx: u16,
        version: u64,
    ) -> Option<bool> {
        let entry = self.entries.get(&entity_id)?;
        if entry.version > version {
            return None;
        }
        Some(
            entry
                .attrs
                .binary_search_by_key(&attr_idx, |(idx, _)| *idx)
                .is_ok(),
        )
    }

    /// Remove a single attribute from a cached entity, keeping the entry dirty.
    ///
    /// This is used by `remove_attr` to remove only a specific attribute while
    /// preserving other cached attributes. The entry remains marked dirty and
    /// will be flushed to fjall along with other pending changes.
    ///
    /// Returns `true` if the attribute was found and removed, `false` if the
    /// entity was not in cache or the attribute didn't exist.
    #[must_use]
    pub fn remove_attr_from_entity(
        &self,
        entity_id: u64,
        attr_idx: u16,
    ) -> bool {
        if let Some(mut entry) = self.entries.get(&entity_id)
            && let Ok(pos) = entry.attrs.binary_search_by_key(&attr_idx, |(idx, _)| *idx)
        {
            let mut new_attrs = (*entry.attrs).clone();
            new_attrs.remove(pos);
            entry.attrs = Arc::new(new_attrs);
            entry.dirty = true;
            // Update the cache with the modified entry
            self.entries.insert(entity_id, entry);
            return true;
        }
        false
    }

    /// Returns `true` when the cache is over its memory budget.
    #[must_use]
    pub fn over_budget(&self) -> bool {
        self.entries.weight() > self.entries.capacity()
    }

    /// Collect up to `n` dirty entries for flushing.
    ///
    /// Collected entries are **removed** from the cache.  The caller is
    /// responsible for writing them to fjall.
    #[must_use]
    pub fn collect_dirty_lru(
        &self,
        n: usize,
    ) -> Vec<(u64, Arc<Vec<(u16, Value)>>)> {
        let mut result = Vec::with_capacity(n);
        // Iterate and collect dirty entries.
        for (entity_id, entry) in self.entries.iter() {
            if result.len() >= n {
                break;
            }
            if entry.dirty {
                result.push((entity_id, entry.attrs.clone()));
            }
        }
        // Remove collected entries from cache.
        for (id, _) in &result {
            self.entries.remove(id);
        }
        result
    }

    /// Total estimated bytes currently in the cache.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        let mem = self.entries.memory_used();
        mem.entries + mem.map
    }
}
