//! In-memory concurrent cache for entity attributes.
//!
//! Custom sharded `RwLock<Vec<Option<CachedEntity>>>` implementation tuned
//! for the attribute-store workload (sequential `u64` entity ids).
//!
//! ## Why a Vec instead of a HashMap
//!
//! Entity ids are allocated monotonically (`max_node_id = node_count +
//! deleted.len() - 1`), so direct slot indexing is faster than hashing +
//! probing.  Cost: gaps from deleted entities sit in memory as `None`
//! slots (≈32 bytes each); for dense ids — the common case — this is
//! cheaper than the per-entry HashMap bucket overhead.
//!
//! ## Shape
//!
//! ```text
//!  AttributeCache
//!   ├── shards: [Shard; 64]
//!   │     └── RwLock<Vec<Option<CachedEntity>>>   // slot = id >> 6
//!   │     └── AtomicU64 bytes
//!   └── dirty_shards: [Mutex<FxHashSet<u64>>; 64] // dirty index
//! ```
//!
//! ## Eviction
//!
//! Lazy: when an insert pushes the per-shard byte counter past its slice
//! of the budget, we walk the Vec and drop a few clean entries until
//! under.  Dirty entries are never evicted — caller drains them via
//! [`AttributeCache::collect_dirty_lru`].

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::{Mutex, RwLock};
use rustc_hash::FxHashSet;

use crate::runtime::value::Value;

/// Per-entity cached attributes.
#[derive(Clone)]
struct CachedEntity {
    /// Sorted by `attr_idx` for O(log n) binary-search lookups.
    attrs: Arc<Vec<(u16, Value)>>,
    /// Pre-computed byte weight (heap footprint estimate).
    weight: u32,
    /// Graph version when this entry was written/populated.
    version: u64,
    /// `true` when the entry has not yet been flushed to fjall.
    dirty: bool,
}

impl CachedEntity {
    fn compute_weight(attrs: &[(u16, Value)]) -> u32 {
        let base = attrs.len() * (std::mem::size_of::<u16>() + std::mem::size_of::<Value>());
        let heap: usize = attrs.iter().map(|(_, v)| v.heap_size()).sum();
        let total = base
            + heap
            + std::mem::size_of::<CachedEntity>()
            + std::mem::size_of::<Arc<Vec<(u16, Value)>>>();
        u32::try_from(total).unwrap_or(u32::MAX).max(1)
    }
}

const SHARDS: usize = 64;
const SHARD_BITS: u32 = 6; // log2(SHARDS)
const SHARD_MASK: u64 = (SHARDS as u64) - 1;
// Chunks of CHUNK consecutive ids share a shard, so sequential id batches
// (label scans) reuse one read lock per chunk instead of one per id.
const CHUNK_BITS: u32 = 6;
const CHUNK: u64 = 1 << CHUNK_BITS;
const CHUNK_MASK: u64 = CHUNK - 1;

struct Shard {
    /// Layout: shard = `(id >> CHUNK_BITS) & SHARD_MASK`, so `CHUNK`
    /// consecutive ids share one shard — sequential id batches (e.g.
    /// label scans) reuse a single read lock per chunk.
    entries: RwLock<Vec<Option<CachedEntity>>>,
    bytes: AtomicU64,
}

const DIRTY_SHARDS: usize = 64;

#[inline]
fn shard_idx(entity_id: u64) -> usize {
    ((entity_id >> CHUNK_BITS) & SHARD_MASK) as usize
}

#[inline]
fn slot_idx(entity_id: u64) -> usize {
    // Bijection with shard_idx. Reconstruction:
    //   id = ((slot >> CHUNK_BITS) << (CHUNK_BITS + SHARD_BITS))
    //         | (shard << CHUNK_BITS) | (slot & CHUNK_MASK)
    let high = entity_id >> (CHUNK_BITS + SHARD_BITS);
    let low = entity_id & CHUNK_MASK;
    ((high << CHUNK_BITS) | low) as usize
}

#[inline]
fn dirty_shard(entity_id: u64) -> usize {
    (entity_id as usize) & (DIRTY_SHARDS - 1)
}

/// Shared, version-stamped, entity-level attribute cache.
pub struct AttributeCache {
    shards: Box<[Shard; SHARDS]>,
    /// Total byte budget across the whole cache.
    capacity_bytes: u64,
    /// Sharded index of entity IDs whose cache entry currently has
    /// `dirty == true`.  Lets `collect_dirty_lru` scan only dirty entries
    /// instead of the entire cache.
    dirty_shards: [Mutex<FxHashSet<u64>>; DIRTY_SHARDS],
}

impl AttributeCache {
    /// Create a new cache with the given byte budget.
    #[must_use]
    pub fn new(max_bytes: usize) -> Self {
        let shards: Vec<Shard> = (0..SHARDS)
            .map(|_| Shard {
                entries: RwLock::new(Vec::new()),
                bytes: AtomicU64::new(0),
            })
            .collect();
        let shards: Box<[Shard; SHARDS]> = shards
            .into_boxed_slice()
            .try_into()
            .unwrap_or_else(|_| unreachable!("vec built with exactly SHARDS items"));
        Self {
            shards,
            capacity_bytes: max_bytes as u64,
            dirty_shards: std::array::from_fn(|_| Mutex::new(FxHashSet::default())),
        }
    }

    #[inline]
    fn shard(
        &self,
        entity_id: u64,
    ) -> &Shard {
        // SAFETY: shard_idx returns 0..SHARDS.
        unsafe { self.shards.get_unchecked(shard_idx(entity_id)) }
    }

    #[inline]
    fn mark_dirty(
        &self,
        entity_id: u64,
    ) {
        let mut shard = self.dirty_shards[dirty_shard(entity_id)].lock();
        shard.insert(entity_id);
    }

    #[inline]
    fn unmark_dirty(
        &self,
        entity_id: u64,
    ) {
        let mut shard = self.dirty_shards[dirty_shard(entity_id)].lock();
        shard.remove(&entity_id);
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
        let shard = self.shard(entity_id);
        let slot = slot_idx(entity_id);
        let entries = shard.entries.read();
        let entry = entries.get(slot)?.as_ref()?;
        if entry.version > version {
            return None;
        }
        Some(
            entry
                .attrs
                .binary_search_by_key(&attr_idx, |(idx, _)| *idx)
                .ok()
                .map(|pos| entry.attrs[pos].1.clone()),
        )
    }

    /// Batch variant of [`get_attr`] for many ids sharing the same `attr_idx`.
    ///
    /// Walks the input in order and reuses the current shard's read lock for
    /// any run of consecutive keys that hash to the same shard.
    pub fn get_attrs_batch(
        &self,
        keys: &[u64],
        attr_idx: u16,
        version: u64,
        out: &mut Vec<Option<Option<Value>>>,
    ) {
        out.clear();
        out.resize(keys.len(), None);
        if keys.is_empty() {
            return;
        }
        let mut current_shard_idx = shard_idx(keys[0]);
        let mut guard = self.shards[current_shard_idx].entries.read();
        for (pos, &id) in keys.iter().enumerate() {
            let s = shard_idx(id);
            if s != current_shard_idx {
                drop(guard);
                current_shard_idx = s;
                guard = self.shards[s].entries.read();
            }
            let slot = slot_idx(id);
            let Some(Some(entry)) = guard.get(slot) else {
                continue;
            };
            if entry.version > version {
                continue;
            }
            out[pos] = Some(
                entry
                    .attrs
                    .binary_search_by_key(&attr_idx, |(idx, _)| *idx)
                    .ok()
                    .map(|p| entry.attrs[p].1.clone()),
            );
        }
    }

    /// Return all cached attributes for an entity.
    #[must_use]
    pub fn get_entity(
        &self,
        entity_id: u64,
        version: u64,
    ) -> Option<Arc<Vec<(u16, Value)>>> {
        let shard = self.shard(entity_id);
        let slot = slot_idx(entity_id);
        let entries = shard.entries.read();
        let entry = entries.get(slot)?.as_ref()?;
        if entry.version > version {
            return None;
        }
        Some(Arc::clone(&entry.attrs))
    }

    /// Return all cached attributes for an entity along with the dirty flag.
    #[must_use]
    pub fn get_entity_with_dirty(
        &self,
        entity_id: u64,
        version: u64,
    ) -> Option<(Arc<Vec<(u16, Value)>>, bool)> {
        let shard = self.shard(entity_id);
        let slot = slot_idx(entity_id);
        let entries = shard.entries.read();
        let entry = entries.get(slot)?.as_ref()?;
        if entry.version > version {
            return None;
        }
        Some((Arc::clone(&entry.attrs), entry.dirty))
    }

    /// Check whether an entity has *any* cached attributes.
    #[must_use]
    pub fn has_entity(
        &self,
        entity_id: u64,
        version: u64,
    ) -> Option<bool> {
        let shard = self.shard(entity_id);
        let slot = slot_idx(entity_id);
        let entries = shard.entries.read();
        let entry = entries.get(slot)?.as_ref()?;
        if entry.version > version {
            return None;
        }
        Some(!entry.attrs.is_empty())
    }

    /// Check whether an attr already exists for an entity in the cache.
    #[must_use]
    pub fn contains_attr(
        &self,
        entity_id: u64,
        attr_idx: u16,
        version: u64,
    ) -> Option<bool> {
        let shard = self.shard(entity_id);
        let slot = slot_idx(entity_id);
        let entries = shard.entries.read();
        let entry = entries.get(slot)?.as_ref()?;
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

    /// Insert (or replace) the full attribute set for an entity.
    pub fn insert_entity(
        &self,
        entity_id: u64,
        mut attrs: Vec<(u16, Value)>,
        version: u64,
        dirty: bool,
    ) {
        attrs.sort_by_key(|item| item.0);
        let weight = CachedEntity::compute_weight(&attrs);
        self.insert_internal(entity_id, attrs, weight, version, dirty);
    }

    /// Insert (or replace) the full attribute set for an entity when the
    /// caller guarantees the attrs are already sorted by `attr_idx`.
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
        let weight = CachedEntity::compute_weight(&attrs);
        self.insert_internal(entity_id, attrs, weight, version, dirty);
    }

    fn insert_internal(
        &self,
        entity_id: u64,
        attrs: Vec<(u16, Value)>,
        weight: u32,
        version: u64,
        dirty: bool,
    ) {
        let entry = CachedEntity {
            attrs: Arc::new(attrs),
            weight,
            version,
            dirty,
        };
        let new_w = u64::from(weight);
        let shard = self.shard(entity_id);
        let slot = slot_idx(entity_id);
        let prev_w = {
            let mut entries = shard.entries.write();
            if entries.len() <= slot {
                entries.resize(slot + 1, None);
            }
            // SAFETY: just resized to cover `slot`.
            let cell = unsafe { entries.get_unchecked_mut(slot) };
            let prev = cell.replace(entry);
            prev.map_or(0, |p| u64::from(p.weight))
        };
        if new_w >= prev_w {
            shard.bytes.fetch_add(new_w - prev_w, Ordering::Relaxed);
        } else {
            shard.bytes.fetch_sub(prev_w - new_w, Ordering::Relaxed);
        }
        if dirty {
            self.mark_dirty(entity_id);
        }
        self.maybe_evict_clean(shard);
    }

    /// Insert or update a cache entry only if not overwriting a newer or dirty entry.
    #[must_use]
    pub fn insert_entity_if_older(
        &self,
        entity_id: u64,
        attrs: Vec<(u16, Value)>,
        version: u64,
    ) -> bool {
        let shard = self.shard(entity_id);
        let slot = slot_idx(entity_id);
        // Fast path: read-lock peek.
        {
            let entries = shard.entries.read();
            if let Some(Some(existing)) = entries.get(slot)
                && (existing.version >= version || existing.dirty)
            {
                return false;
            }
        }
        self.insert_entity(entity_id, attrs, version, false);
        true
    }

    /// Remove a single entity from the cache.
    pub fn invalidate(
        &self,
        entity_id: u64,
    ) {
        let shard = self.shard(entity_id);
        let slot = slot_idx(entity_id);
        let removed_w = {
            let mut entries = shard.entries.write();
            if let Some(cell) = entries.get_mut(slot) {
                cell.take().map_or(0, |e| u64::from(e.weight))
            } else {
                0
            }
        };
        if removed_w > 0 {
            shard.bytes.fetch_sub(removed_w, Ordering::Relaxed);
        }
        self.unmark_dirty(entity_id);
    }

    /// Batch-invalidate entities (used during rollback and commit).
    ///
    /// Groups ids by shard so each shard's write lock is taken at most once.
    pub fn invalidate_batch(
        &self,
        entity_ids: &roaring::RoaringTreemap,
    ) {
        let mut by_shard: [Vec<u64>; SHARDS] = std::array::from_fn(|_| Vec::new());
        for id in entity_ids {
            by_shard[shard_idx(id)].push(id);
        }
        for (i, ids) in by_shard.iter().enumerate() {
            if ids.is_empty() {
                continue;
            }
            let shard = &self.shards[i];
            let mut freed: u64 = 0;
            {
                let mut entries = shard.entries.write();
                for &id in ids {
                    let slot = slot_idx(id);
                    if let Some(cell) = entries.get_mut(slot)
                        && let Some(e) = cell.take()
                    {
                        freed += u64::from(e.weight);
                    }
                }
            }
            if freed > 0 {
                shard.bytes.fetch_sub(freed, Ordering::Relaxed);
            }
        }
        for id in entity_ids {
            self.unmark_dirty(id);
        }
    }

    /// Remove a single attribute from a cached entity, keeping the entry dirty.
    #[must_use]
    pub fn remove_attr_from_entity(
        &self,
        entity_id: u64,
        attr_idx: u16,
    ) -> bool {
        let shard = self.shard(entity_id);
        let slot = slot_idx(entity_id);
        let mut entries = shard.entries.write();
        let Some(Some(existing)) = entries.get_mut(slot) else {
            return false;
        };
        let Ok(pos) = existing
            .attrs
            .binary_search_by_key(&attr_idx, |(idx, _)| *idx)
        else {
            return false;
        };
        let removed_heap = existing.attrs[pos].1.heap_size();
        let removed_weight =
            (std::mem::size_of::<u16>() + std::mem::size_of::<Value>() + removed_heap) as u32;
        let new_weight = existing.weight.saturating_sub(removed_weight).max(1);
        let prev_w = u64::from(existing.weight);
        let new_w = u64::from(new_weight);
        let mut new_attrs = (*existing.attrs).clone();
        new_attrs.remove(pos);
        existing.attrs = Arc::new(new_attrs);
        existing.weight = new_weight;
        existing.dirty = true;
        drop(entries);
        if new_w >= prev_w {
            shard.bytes.fetch_add(new_w - prev_w, Ordering::Relaxed);
        } else {
            shard.bytes.fetch_sub(prev_w - new_w, Ordering::Relaxed);
        }
        self.mark_dirty(entity_id);
        true
    }

    /// Returns `true` when the cache contains any entries.
    #[must_use]
    pub fn has_entries(&self) -> bool {
        self.shards
            .iter()
            .any(|s| s.bytes.load(Ordering::Relaxed) > 0)
    }

    /// Returns `true` when the cache is over its memory budget.
    #[must_use]
    pub fn over_budget(&self) -> bool {
        self.memory_usage_atomic() > self.capacity_bytes
    }

    fn memory_usage_atomic(&self) -> u64 {
        self.shards
            .iter()
            .map(|s| s.bytes.load(Ordering::Relaxed))
            .sum()
    }

    /// Total estimated bytes currently in the cache.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.memory_usage_atomic() as usize
    }

    /// Best-effort lazy eviction of *clean* entries from the given shard
    /// when its slice of the byte budget is exhausted.
    #[inline]
    fn maybe_evict_clean(
        &self,
        shard: &Shard,
    ) {
        let per_shard_budget = self.capacity_bytes / (SHARDS as u64);
        if shard.bytes.load(Ordering::Relaxed) <= per_shard_budget {
            return;
        }
        let mut entries = shard.entries.write();
        let mut freed: u64 = 0;
        let target = per_shard_budget;
        for cell in entries.iter_mut() {
            if let Some(entry) = cell
                && !entry.dirty
            {
                let w = u64::from(entry.weight);
                *cell = None;
                freed += w;
                if shard.bytes.load(Ordering::Relaxed) - freed <= target {
                    break;
                }
            }
        }
        if freed > 0 {
            shard.bytes.fetch_sub(freed, Ordering::Relaxed);
        }
    }

    /// Collect up to `n` dirty entries for flushing.
    ///
    /// Cost is O(min(n, dirty_count)) — independent of total cache size.
    #[must_use]
    pub fn collect_dirty_lru(
        &self,
        n: usize,
    ) -> Vec<(u64, Arc<Vec<(u16, Value)>>)> {
        let mut result = Vec::with_capacity(n);
        for dirty_shard in &self.dirty_shards {
            if result.len() >= n {
                break;
            }
            let mut dshard = dirty_shard.lock();
            let take = (n - result.len()).min(dshard.len());
            if take == 0 {
                continue;
            }
            let collected: Vec<u64> = dshard.iter().copied().take(take).collect();
            for id in &collected {
                dshard.remove(id);
            }
            drop(dshard);
            for id in collected {
                let shard = self.shard(id);
                let slot = slot_idx(id);
                let mut entries = shard.entries.write();
                let Some(cell) = entries.get_mut(slot) else {
                    continue;
                };
                let take_it = matches!(cell, Some(e) if e.dirty);
                if take_it {
                    let entry = cell.take().expect("matched Some above");
                    let weight = u64::from(entry.weight);
                    drop(entries);
                    shard.bytes.fetch_sub(weight, Ordering::Relaxed);
                    result.push((id, entry.attrs));
                }
            }
        }
        result
    }
}
