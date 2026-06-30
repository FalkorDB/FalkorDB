//! Attribute storage for graph entities (nodes and relationships).
//!
//! This module provides [`AttributeStore`], an in-memory columnar key-value
//! store for entity properties backed by a concurrent [`AttributeStorage`].
//! The cache is the sole source of truth: a cache miss means the entity has
//! no attributes.
//!
//! ## Read Path
//!
//! `get_attr` checks the cache; a miss returns `None` (no attributes).
//!
//! ## Write Path
//!
//! `insert_attrs` resolves attribute names to column indices, merges with the
//! current cached state, and writes the result back to the cache as dirty.
//! `commit` applies deletions and accumulates the transaction's dirty set;
//! `rollback_cache` restores or invalidates dirty entries; `clear_rollback_state`
//! drops tracking after a successful query.
//!
//! ## MVCC Integration
//!
//! The storage is shared across MVCC versions via `Arc<AttributeStorage>`. Each
//! `AttributeStore` carries its own MVCC `version`; cache entries newer than
//! the reader's version are ignored. `new_version()` clones the store cheaply
//! (Arc bump + bitmap clear) with a fresh `dirty_entities` tracker.

use std::alloc::{self, Layout};
use std::ptr::{self, NonNull};
use std::slice;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};

use std::cmp::Ordering;

use parking_lot::RwLock;
use rustc_hash::FxHashMap;

use roaring::RoaringTreemap;

use super::graphblas::serialization::{Decode, Encode, Reader, Writer};
use crate::runtime::{ordermap::OrderMap, value::Value};

/// Insertion-ordered map of attribute names to attribute indices.
///
/// Maintains both a `Vec<Arc<String>>` (for stable index → name lookup and
/// deterministic iteration order) and a `FxHashMap<Arc<String>, u16>` for
/// O(1) name → index resolution on the hot read path.
#[derive(Default, Clone)]
pub struct AttrNameMap {
    vec: Vec<Arc<String>>,
    index: FxHashMap<Arc<String>, u16>,
}

impl AttrNameMap {
    #[must_use]
    pub const fn len(&self) -> usize {
        self.vec.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.vec.is_empty()
    }

    #[must_use]
    pub fn get(
        &self,
        idx: usize,
    ) -> Option<&Arc<String>> {
        self.vec.get(idx)
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Arc<String>> {
        self.vec.iter()
    }

    #[must_use]
    pub fn get_index_of(
        &self,
        name: &Arc<String>,
    ) -> Option<usize> {
        self.index.get(name).map(|&i| i as usize)
    }

    pub fn insert(
        &mut self,
        name: Arc<String>,
    ) {
        if self.index.contains_key(&name) {
            return;
        }
        let idx = self.vec.len() as u16;
        self.vec.push(name.clone());
        self.index.insert(name, idx);
    }
}

impl std::ops::Index<usize> for AttrNameMap {
    type Output = Arc<String>;

    fn index(
        &self,
        idx: usize,
    ) -> &Arc<String> {
        &self.vec[idx]
    }
}

/// Shared empty attribute vector to avoid per-entity allocations when an
/// entity has no properties.
static EMPTY_ATTRS: once_cell::sync::Lazy<AttrArray> = once_cell::sync::Lazy::new(AttrArray::empty);

/// Columnar attribute storage for graph entities.
///
/// Uses a shared [`AttributeStorage`] as the sole, in-memory source of truth for
/// entity attributes.  A cache miss means the entity has no attributes.
pub struct AttributeStore {
    /// Attribute names in insertion order (name → column index)
    pub attrs_name: AttrNameMap,
    /// Shared in-memory storage (cheap Arc clone across MVCC versions).
    storage: Arc<AttributeStorage>,
    /// MVCC version of this store's snapshot.
    version: u64,
    /// Entity IDs written during the current write transaction (across all
    /// operators since the last `new_version`/`clear_rollback_state`). Used by
    /// `rollback_cache` to invalidate cache entries that aren't restored.
    dirty_entities: RoaringTreemap,
    /// Entity IDs pending full deletion (all attributes) — applied on commit, cleared on rollback.
    pending_deletes: RoaringTreemap,
    /// Saved original cache entries captured before the first modification
    /// within a write transaction. On rollback, these are restored to undo
    /// cache mutations, since the cache is shared with the read version.
    saved_for_rollback: FxHashMap<u64, AttrArray>,
}

impl Clone for AttributeStore {
    fn clone(&self) -> Self {
        Self {
            attrs_name: self.attrs_name.clone(),
            storage: self.storage.clone(),
            version: self.version,
            dirty_entities: self.dirty_entities.clone(),
            pending_deletes: self.pending_deletes.clone(),
            saved_for_rollback: self.saved_for_rollback.clone(),
        }
    }
}

impl AttributeStore {
    #[must_use]
    pub fn new(version: u64) -> Self {
        Self {
            attrs_name: AttrNameMap::default(),
            storage: Arc::new(AttributeStorage::new()),
            version,
            dirty_entities: RoaringTreemap::new(),
            pending_deletes: RoaringTreemap::new(),
            saved_for_rollback: FxHashMap::default(),
        }
    }

    #[must_use]
    pub fn new_version(
        &self,
        version: u64,
    ) -> Self {
        Self {
            attrs_name: self.attrs_name.clone(),
            storage: self.storage.clone(),
            version,
            dirty_entities: RoaringTreemap::new(),
            pending_deletes: RoaringTreemap::new(),
            saved_for_rollback: FxHashMap::default(),
        }
    }

    // ---- helpers --------------------------------------------------------

    #[must_use]
    pub fn get_attr(
        &self,
        key: u64,
        attr: &Arc<String>,
    ) -> Option<Value> {
        let idx = self.attrs_name.get_index_of(attr)? as u16;
        self.get_attr_by_idx(key, idx)
    }

    #[must_use]
    pub fn get_attr_by_idx(
        &self,
        key: u64,
        attr_idx: u16,
    ) -> Option<Value> {
        if self.pending_deletes.contains(key) {
            return None;
        }
        // 1. Check cache.
        if let Some(result) = self.storage.get_attr(key, attr_idx, self.version) {
            return result;
        }
        // 2. Storage miss — entity has no attributes.
        None
    }

    /// Batch variant of `get_attr_by_idx` for a list of keys with the same
    /// `attr_idx`. Avoids re-doing the per-call setup (function dispatch,
    /// `pending_deletes` check) for every key when the deletion set is empty.
    /// Pushes one `Value` per key into `out`, substituting `default` for
    /// missing or pending-deleted entries.
    pub fn get_attrs_by_idx_batch_into(
        &self,
        keys: &[u64],
        attr_idx: u16,
        default: &Value,
        out: &mut Vec<Value>,
    ) {
        out.reserve(keys.len());
        let version = self.version;
        if self.pending_deletes.is_empty() {
            // Hot read path: fused single pass that writes `Value`s directly
            // into `out`, avoiding the intermediate `Vec<Option<Option<_>>>`
            // and a second pass. `missing` stays empty when every key hits the
            // cache (the common case), so no cold-store work is done.
            // Cache is the sole store: misses already carry `default`, so the
            // single cache pass fully populates `out`.
            let mut missing: Vec<usize> = Vec::new();
            self.storage
                .get_attrs_batch_into(keys, attr_idx, version, default, out, &mut missing);
            return;
        }
        // Slow path: some entities are pending deletion, so a cache hit must be
        // overridden with `default`. Fall back to the two-pass logic.
        let mut cache_results: Vec<Option<Option<Value>>> = Vec::with_capacity(keys.len());
        self.storage
            .get_attrs_batch(keys, attr_idx, version, &mut cache_results);
        for (i, &key) in keys.iter().enumerate() {
            if self.pending_deletes.contains(key) {
                out.push(default.clone());
                continue;
            }
            let v = cache_results[i].take().unwrap_or(None);
            out.push(v.unwrap_or_else(|| default.clone()));
        }
    }

    #[must_use]
    pub fn has_attributes(
        &self,
        key: u64,
    ) -> bool {
        // Entity pending full deletion has no attributes.
        if self.pending_deletes.contains(key) {
            return false;
        }
        if let Some(has) = self.storage.has_entity(key, self.version) {
            return has;
        }
        // Storage miss — entity has no attributes.
        false
    }

    pub fn get_attrs(
        &self,
        key: u64,
    ) -> impl Iterator<Item = Arc<String>> + '_ {
        if self.pending_deletes.contains(key) {
            return Vec::new().into_iter();
        }
        // Try cache first.
        let cached = self.storage.get_entity(key, self.version);
        let attrs = cached.unwrap_or_else(|| EMPTY_ATTRS.clone());
        attrs
            .iter()
            .filter_map(move |(idx, _)| {
                let i = idx as usize;
                if i < self.attrs_name.len() {
                    Some(self.attrs_name[i].clone())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .into_iter()
    }

    pub fn get_all_attrs(
        &self,
        key: u64,
    ) -> Vec<(Arc<String>, Value)> {
        if self.pending_deletes.contains(key) {
            return Vec::new();
        }
        let cached = self.storage.get_entity(key, self.version);
        let attrs = cached.unwrap_or_else(|| EMPTY_ATTRS.clone());
        attrs
            .iter()
            .filter_map(move |(idx, value)| {
                let i = idx as usize;
                if i < self.attrs_name.len() {
                    Some((self.attrs_name[i].clone(), value.clone()))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    }

    pub fn get_all_attrs_by_id(
        &self,
        key: u64,
    ) -> AttrArray {
        if self.pending_deletes.contains(key) {
            return EMPTY_ATTRS.clone();
        }
        self.storage
            .get_entity(key, self.version)
            .unwrap_or_else(|| EMPTY_ATTRS.clone())
    }

    // ---- write path (cache only) ----------------------------------------

    pub fn remove_attr(
        &mut self,
        key: u64,
        attr: &Arc<String>,
    ) -> Result<bool, String> {
        if let Some(idx) = self.attrs_name.get_index_of(attr) {
            let attr_idx = idx as u16;
            // The cache is the sole store: if the attr isn't cached it doesn't exist.
            if self
                .storage
                .contains_attr(key, attr_idx, self.version)
                .unwrap_or(false)
            {
                let _ = self.storage.remove_attr_from_entity(key, attr_idx);
                self.dirty_entities.insert(key);
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn remove_all(
        &mut self,
        keys: &RoaringTreemap,
    ) {
        // Save original cache entries for rollback so that bulk deletes
        // don't lose cache-only attributes when rollback_cache() runs.
        for key in keys {
            if !self.saved_for_rollback.contains_key(&key) {
                let current = self
                    .storage
                    .get_entity(key, self.version)
                    .unwrap_or_else(|| EMPTY_ATTRS.clone());
                self.saved_for_rollback.insert(key, current);
            }
        }
        self.pending_deletes |= keys;
    }

    /// Batch insert/update multiple attributes for entities.
    ///
    /// Writes go to the in-memory cache (`dirty = true`).  Returns `(nremoved, nset)`:
    /// the number of attributes *replaced* and the number of non-null attributes *set*.
    pub fn insert_attrs(
        &mut self,
        attrs: &FxHashMap<u64, OrderMap<Arc<String>, Value>>,
    ) -> Result<(usize, usize), String> {
        let mut nremoved = 0;
        let mut nset = 0;

        // Pre-resolve all unique attribute names → indices ONCE.
        // Uses Arc pointer identity as key to avoid rehashing strings.
        let mut name_to_idx: FxHashMap<*const String, u16> = FxHashMap::default();
        for entity_attrs in attrs.values() {
            for (attr, _) in entity_attrs.iter() {
                let ptr = Arc::as_ptr(attr);
                if let std::collections::hash_map::Entry::Vacant(e) = name_to_idx.entry(ptr) {
                    let idx = self.attrs_name.get_index_of(attr).unwrap_or_else(|| {
                        self.attrs_name.insert(attr.clone());
                        self.attrs_name.len() - 1
                    }) as u16;
                    e.insert(idx);
                }
            }
        }

        // Reusable buffer to avoid per-entity allocation.
        let mut merged: Vec<(u16, Value)> = Vec::new();

        for (key, entity_attrs) in attrs {
            // Skip entities whose pending map is empty: no entries to write, no nulls
            // to remove. Avoids creating an empty pinned
            // dirty cache entry per entity (matches C's NULL AttributeSet behaviour).
            if entity_attrs.is_empty() {
                continue;
            }
            // Resolve attribute indices using pre-resolved map.
            let mut new_entries: Vec<(u16, Value)> = Vec::with_capacity(entity_attrs.len());
            let mut null_indices: Vec<u16> = Vec::new();

            for (attr, value) in entity_attrs.iter() {
                let idx = name_to_idx[&Arc::as_ptr(attr)];
                if matches!(value, Value::Null) {
                    null_indices.push(idx);
                } else {
                    new_entries.push((idx, value.clone()));
                    nset += 1;
                }
            }

            // Get current stored state. Keep the shared array for rollback
            // and materialize a Vec for the in-place merge below.
            let current_arr = self.storage.get_entity(*key, self.version);
            let current: Vec<(u16, Value)> = current_arr
                .as_ref()
                .map_or_else(Vec::new, AttrArray::to_pairs);

            // Sort new entries for O(n+m) merge.
            new_entries.sort_unstable_by_key(|(idx, _)| *idx);

            // Fast path: if no nulls AND all new entries come after all current entries,
            // we can just clone current + append new without a full merge.
            if null_indices.is_empty()
                && !new_entries.is_empty()
                && (current.is_empty() || current.last().unwrap().0 < new_entries[0].0)
            {
                merged.clear();
                merged.reserve(current.len() + new_entries.len());
                merged.extend_from_slice(&current);
                merged.append(&mut new_entries);
            } else {
                null_indices.sort_unstable();

                // Single-pass sorted merge of current + new_entries, skipping nulls.
                merged.clear();
                merged.reserve(current.len() + new_entries.len());
                let mut ci = 0;
                let mut ni = 0;
                let mut di = 0;

                while ci < current.len() && ni < new_entries.len() {
                    let cur_idx = current[ci].0;
                    let new_idx = new_entries[ni].0;
                    match cur_idx.cmp(&new_idx) {
                        Ordering::Less => {
                            if di < null_indices.len() && cur_idx == null_indices[di] {
                                nremoved += 1;
                                di += 1;
                            } else {
                                merged.push(current[ci].clone());
                            }
                            ci += 1;
                        }
                        Ordering::Equal => {
                            nremoved += 1;
                            merged.push((new_idx, new_entries[ni].1.clone()));
                            ci += 1;
                            ni += 1;
                        }
                        Ordering::Greater => {
                            merged.push((new_idx, new_entries[ni].1.clone()));
                            ni += 1;
                        }
                    }
                }
                while ci < current.len() {
                    let cur_idx = current[ci].0;
                    if di < null_indices.len() && cur_idx == null_indices[di] {
                        nremoved += 1;
                        di += 1;
                    } else {
                        merged.push(current[ci].clone());
                    }
                    ci += 1;
                }
                while ni < new_entries.len() {
                    merged.push((new_entries[ni].0, new_entries[ni].1.clone()));
                    ni += 1;
                }
            }

            // Save original entry for rollback (only the first time).
            if !self.saved_for_rollback.contains_key(key) {
                let saved = current_arr.clone().unwrap_or_else(|| EMPTY_ATTRS.clone());
                self.saved_for_rollback.insert(*key, saved);
            }

            // Write merged attrs to cache as dirty (already sorted, skip re-sort).
            self.storage
                .insert_entity_presorted(*key, std::mem::take(&mut merged), self.version);
            self.dirty_entities.insert(*key);
        }

        Ok((nremoved, nset))
    }

    /// Bulk import attributes for entities known to be new (no prior state).
    ///
    /// Optimized for RDB decode: skips cache lookups since entities
    /// don't exist yet. Attributes are written directly to cache.
    /// Returns the number of non-null attributes imported.
    pub fn import_attrs(
        &mut self,
        attrs: &FxHashMap<u64, OrderMap<Arc<String>, Value>>,
    ) -> usize {
        // Pre-resolve all unique attribute names → indices ONCE.
        let mut name_to_idx: FxHashMap<*const String, u16> = FxHashMap::default();
        for entity_attrs in attrs.values() {
            for (attr, _) in entity_attrs.iter() {
                let ptr = Arc::as_ptr(attr);
                if let std::collections::hash_map::Entry::Vacant(e) = name_to_idx.entry(ptr) {
                    let idx = self.attrs_name.get_index_of(attr).unwrap_or_else(|| {
                        self.attrs_name.insert(attr.clone());
                        self.attrs_name.len() - 1
                    }) as u16;
                    e.insert(idx);
                }
            }
        }

        let mut nset = 0;
        for (key, entity_attrs) in attrs {
            let mut entries: Vec<(u16, Value)> = Vec::with_capacity(entity_attrs.len());

            for (attr, value) in entity_attrs.iter() {
                if matches!(value, Value::Null) {
                    continue;
                }
                let idx = name_to_idx[&Arc::as_ptr(attr)];
                entries.push((idx, value.clone()));
                nset += 1;
            }

            entries.sort_by_key(|(idx, _)| *idx);
            // Skip empty entities to avoid per-entity cache overhead (matches C's
            // NULL AttributeSet behaviour for prop-less entities).
            if entries.is_empty() {
                continue;
            }
            self.storage
                .insert_entity_presorted(*key, entries, self.version);
            self.dirty_entities.insert(*key);
        }
        nset
    }

    /// Import pre-resolved attribute data directly into the cache.
    /// Skips name resolution and OrderMap construction; used by bulk insert.
    pub fn import_attrs_resolved(
        &mut self,
        data: &mut Vec<(u64, Vec<(u16, Value)>)>,
    ) -> usize {
        let mut nset = 0;
        for (entity_id, entries) in data.drain(..) {
            if entries.is_empty() {
                continue;
            }
            nset += entries.len();
            self.storage
                .insert_entity_presorted(entity_id, entries, self.version);
            self.dirty_entities.insert(entity_id);
        }
        nset
    }

    /// Resolve an attribute name to its index, creating a new mapping if needed.
    pub fn get_or_create_attr_id(
        &mut self,
        attr: &Arc<String>,
    ) -> u16 {
        self.attrs_name.get_index_of(attr).unwrap_or_else(|| {
            self.attrs_name.insert(attr.clone());
            self.attrs_name.len() - 1
        }) as u16
    }

    #[must_use]
    pub fn get_attr_id(
        &self,
        attr: &Arc<String>,
    ) -> Option<usize> {
        self.attrs_name.get_index_of(attr)
    }

    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.storage.memory_usage()
    }

    /// Structural slot-storage overhead, excluding attribute payload heap.
    #[must_use]
    pub fn structural_memory_usage(&self) -> usize {
        self.storage.structural_memory_usage()
    }

    pub fn commit(&mut self) -> Result<(), String> {
        // Invalidate fully-deleted entities from the shared cache to prevent
        // stale reads. The cache is the sole store, so this is the only state
        // that must change on commit.
        if !self.pending_deletes.is_empty() {
            self.storage.invalidate_batch(&self.pending_deletes);
        }
        // Fold this operator's deletions into the transaction-wide dirty set so
        // rollback_cache() can still invalidate them if a later operator in the
        // same query fails. Attribute writes are already tracked in
        // `dirty_entities`, which persists across operators until the query
        // commits (clear_rollback_state) or rolls back.
        self.dirty_entities |= &self.pending_deletes;
        self.pending_deletes.clear();
        Ok(())
    }

    /// Drop rollback-saved state after a query commits successfully.
    /// Called by graph_core once the query reaches its final commit boundary,
    /// not per-operator (intermediate `commit_attrs` calls must preserve
    /// `saved_for_rollback` so a later failing operator can roll the whole
    /// transaction back to the pre-query state).
    pub fn clear_rollback_state(&mut self) {
        self.saved_for_rollback.clear();
        self.dirty_entities.clear();
    }

    // ---- flush / rollback -----------------------------------------------

    /// Invalidate all dirty entities from the shared cache.
    /// Called on write-transaction rollback.
    pub fn rollback_cache(&mut self) {
        // Restore saved original cache entries for entities that were
        // modified during this write transaction. This is needed because the
        // cache is shared between MVCC versions — simply invalidating would
        // lose data that is only present in the shared in-memory cache.
        let mut restored = RoaringTreemap::new();

        for (entity_id, original_attrs) in self.saved_for_rollback.drain() {
            if original_attrs.is_empty() {
                // Entity had no prior attrs — skip re-inserting an empty dirty
                // entry. Let to_invalidate evict the dirty write instead.
                continue;
            }
            self.storage.insert_entity_presorted(
                entity_id,
                original_attrs.to_pairs(),
                self.version.saturating_sub(1),
            );
            restored.insert(entity_id);
        }
        // Invalidate any remaining dirty entities not covered by saved entries
        // (e.g., newly created entities that had no prior cache entry).
        let to_invalidate = (&self.dirty_entities | &self.pending_deletes) - &restored;
        self.storage.invalidate_batch(&to_invalidate);
        self.dirty_entities.clear();
        self.pending_deletes.clear();
    }

    /// Encode a range of entities, reading attributes from the in-memory cache.
    pub fn encode_with_range(
        &self,
        w: &mut dyn Writer,
        deleted: &RoaringTreemap,
        max_id: u64,
        global_attrs: &[Arc<String>],
        count: u64,
        offset: u64,
    ) {
        // Build attr remap inline.
        let global_index: FxHashMap<&Arc<String>, usize> = global_attrs
            .iter()
            .enumerate()
            .map(|(i, n)| (n, i))
            .collect();

        let mut remap = vec![u16::MAX; self.attrs_name.len()];
        for (local_id, local_name) in self.attrs_name.iter().enumerate() {
            if let Some(&global_id) = global_index.get(local_name) {
                remap[local_id] = global_id as u16;
            }
        }

        let mut skipped = 0u64;
        let mut encoded = 0u64;

        for id in 0..=max_id {
            if deleted.contains(id) {
                continue;
            }
            if skipped < offset {
                skipped += 1;
                continue;
            }

            w.write_unsigned(id);

            let props = self.get_all_attrs_by_id(id);
            w.write_unsigned(props.len() as u64);

            for (local_attr_id, value) in props.iter() {
                let global_attr_id = if (local_attr_id as usize) < remap.len() {
                    remap[local_attr_id as usize]
                } else {
                    local_attr_id
                };
                w.write_unsigned(global_attr_id as u64);
                value.encode(w);
            }

            encoded += 1;
            if encoded >= count {
                break;
            }
        }
    }
}

// SAFETY: AttributeStore is Send+Sync because:
// - `AttributeStorage` is wrapped in `Arc` and uses sharded locks internally
// - All other owned fields are not shared across threads
unsafe impl Send for AttributeStore {}
unsafe impl Sync for AttributeStore {}

impl Decode<19> for AttributeStore {
    fn decode(_r: &mut dyn Reader) -> Result<Self, String> {
        unimplemented!("use decode_with_count for AttributeStore")
    }

    fn decode_with_count(
        &mut self,
        r: &mut dyn Reader,
        count: u64,
    ) -> Result<(), String> {
        for _ in 0..count {
            let entity_id = r.read_unsigned()?;
            let attr_count = r.read_unsigned()?;

            let mut entries: Vec<(u16, Value)> = Vec::with_capacity(attr_count as usize);
            for _ in 0..attr_count {
                let attr_id = r.read_unsigned()? as u16;
                let value = Value::decode(r)?;

                if (attr_id as usize) < self.attrs_name.len() && !matches!(value, Value::Null) {
                    entries.push((attr_id, value));
                }
            }

            if !entries.is_empty() {
                entries.sort_by_key(|(idx, _)| *idx);
                self.storage.insert_entity(entity_id, entries, self.version);
                self.dirty_entities.insert(entity_id);
            }
        }
        Ok(())
    }
}

// ============================================================================
// AttributeStorage: in-memory concurrent backing store
// ============================================================================
//
// Custom sharded `RwLock<Vec<Option<AttrArray>>>` tuned for the attribute
// workload (sequential `u64` entity ids). Ids are allocated monotonically, so
// direct slot indexing beats hashing; gaps from deleted entities sit as `None`
// slots. Shared across MVCC versions via `Arc`; per-entry version stamps give
// snapshot visibility. It is the sole source of truth (no cold tier), so it
// is unbounded: every live entity stays resident and nothing is evicted.

/// Reference-counted, single-allocation struct-of-arrays for one entity's
/// attributes.
///
/// Replaces the previous fat `Arc<[_]>` representation to remove two sources
/// of per-entity waste: the 6 bytes of padding inside each pair (the `u16` is
/// padded up to `Value`'s 8-byte alignment), and the 16-byte fat pointer plus
/// the 16-byte `Arc` header. Indices and values live as two contiguous columns
/// in **one** heap block, referenced by a thin (8-byte) pointer:
///
/// ```text
///   [ strong: AtomicU32 | len: u32 ]   8-byte header
///   [ values:  [Value; len] ]          16 B each, 8-aligned
///   [ indices: [u16;   len] ]          2 B each
/// ```
///
/// Contents are immutable after construction, so sharing across threads and
/// MVCC versions is sound with an atomic strong count — exactly like `Arc`.
pub struct AttrArray {
    ptr: NonNull<u8>,
}

/// Header stored at the front of every [`AttrArray`] allocation.
///
/// Carrying the MVCC `version` here (rather than alongside the slot pointer)
/// keeps the per-entity slot a single 8-byte `NonNull`, so an empty/absent
/// slot costs 8 bytes instead of 16 — matching the C engine's layout where
/// each entity slot is just a pointer and all metadata lives in the heap
/// allocation it points to.
#[repr(C)]
struct AttrHeader {
    strong: AtomicU32,
    len: u32,
    /// Graph version when this entity's attributes were written. A reader at
    /// snapshot version `v` ignores entries whose `version > v`.
    version: u64,
}

impl AttrArray {
    fn layout_of(len: usize) -> (Layout, usize, usize) {
        let header = Layout::new::<AttrHeader>();
        let values = Layout::array::<Value>(len).expect("values layout");
        let indices = Layout::array::<u16>(len).expect("indices layout");
        let (l, values_off) = header.extend(values).expect("extend values");
        let (l, indices_off) = l.extend(indices).expect("extend indices");
        (l.pad_to_align(), values_off, indices_off)
    }

    /// Build from attribute pairs already sorted by `attr_idx`, stamped with
    /// the writing `version`. Consumes the `Vec`, moving each `Value` into the
    /// new block (no extra clones).
    #[must_use]
    fn from_sorted(
        pairs: Vec<(u16, Value)>,
        version: u64,
    ) -> Self {
        let len = pairs.len();
        let (layout, values_off, indices_off) = Self::layout_of(len);
        // SAFETY: layout has non-zero size (the header is 16 bytes).
        let raw = unsafe { alloc::alloc(layout) };
        let Some(ptr) = NonNull::new(raw) else {
            alloc::handle_alloc_error(layout);
        };
        // SAFETY: `ptr` is a fresh allocation matching `layout`; we initialize
        // the header and exactly `len` values and indices before any read.
        unsafe {
            ptr::write(
                ptr.as_ptr().cast::<AttrHeader>(),
                AttrHeader {
                    strong: AtomicU32::new(1),
                    len: len as u32,
                    version,
                },
            );
            let vptr = ptr.as_ptr().add(values_off).cast::<Value>();
            let iptr = ptr.as_ptr().add(indices_off).cast::<u16>();
            for (i, (idx, val)) in pairs.into_iter().enumerate() {
                ptr::write(vptr.add(i), val);
                ptr::write(iptr.add(i), idx);
            }
        }
        Self { ptr }
    }

    /// Shared empty instance (used for prop-less entities). Its `version` is
    /// irrelevant: the empty array is only ever returned as a fallback payload,
    /// never stored in a slot and never version-checked.
    #[must_use]
    fn empty() -> Self {
        Self::from_sorted(Vec::new(), 0)
    }

    #[inline]
    fn header(&self) -> &AttrHeader {
        // SAFETY: `ptr` always references a valid header for the lifetime of self.
        unsafe { &*self.ptr.as_ptr().cast::<AttrHeader>() }
    }

    /// MVCC version stamp recorded when this entity's attributes were written.
    #[inline]
    #[must_use]
    fn version(&self) -> u64 {
        self.header().version
    }

    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.header().len as usize
    }

    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    #[must_use]
    pub fn indices(&self) -> &[u16] {
        let len = self.len();
        let (_, _, indices_off) = Self::layout_of(len);
        // SAFETY: the index column holds `len` initialized `u16`s.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr().add(indices_off).cast::<u16>(), len) }
    }

    #[inline]
    #[must_use]
    pub fn values(&self) -> &[Value] {
        let len = self.len();
        let (_, values_off, _) = Self::layout_of(len);
        // SAFETY: the value column holds `len` initialized `Value`s.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr().add(values_off).cast::<Value>(), len) }
    }

    /// Position of `attr_idx` within the sorted index column, if present.
    #[inline]
    #[must_use]
    pub fn position(
        &self,
        attr_idx: u16,
    ) -> Option<usize> {
        self.indices().binary_search(&attr_idx).ok()
    }

    /// Value for `attr_idx`, if the attribute is present.
    #[inline]
    #[must_use]
    pub fn get(
        &self,
        attr_idx: u16,
    ) -> Option<&Value> {
        self.position(attr_idx).map(|pos| &self.values()[pos])
    }

    /// Iterate `(attr_idx, &Value)` pairs in index order.
    pub fn iter(&self) -> impl Iterator<Item = (u16, &Value)> + '_ {
        self.indices().iter().copied().zip(self.values().iter())
    }

    /// Materialize an owned `Vec` of `(attr_idx, Value)` pairs.
    #[must_use]
    pub fn to_pairs(&self) -> Vec<(u16, Value)> {
        self.indices()
            .iter()
            .copied()
            .zip(self.values().iter().cloned())
            .collect()
    }

    /// Estimated heap bytes of this entity's single allocation, including the
    /// out-of-line heap owned by each `Value`.
    #[must_use]
    fn heap_bytes(&self) -> usize {
        let (layout, _, _) = Self::layout_of(self.len());
        layout.size() + self.values().iter().map(Value::heap_size).sum::<usize>()
    }
}

impl Clone for AttrArray {
    fn clone(&self) -> Self {
        // Relaxed suffices on clone (matches std `Arc`): the cloned reference is
        // published through existing happens-before edges of the shared store.
        let old = self.header().strong.fetch_add(1, AtomicOrdering::Relaxed);
        assert!(old != u32::MAX, "AttrArray refcount overflow");
        Self { ptr: self.ptr }
    }
}

impl Drop for AttrArray {
    fn drop(&mut self) {
        if self.header().strong.fetch_sub(1, AtomicOrdering::Release) != 1 {
            return;
        }
        // Acquire fence: ensure all prior writes/reads are visible before free.
        std::sync::atomic::fence(AtomicOrdering::Acquire);
        let len = self.len();
        let (layout, values_off, _) = Self::layout_of(len);
        // SAFETY: last reference — drop each `Value` in place, then free.
        unsafe {
            let vptr = self.ptr.as_ptr().add(values_off).cast::<Value>();
            for i in 0..len {
                ptr::drop_in_place(vptr.add(i));
            }
            alloc::dealloc(self.ptr.as_ptr(), layout);
        }
    }
}

// SAFETY: contents are immutable after construction and the strong count is
// atomic, so `AttrArray` shares across threads exactly like an `Arc<[_]>`
// (sound because `u16` and `Value` are both `Send + Sync`).
unsafe impl Send for AttrArray {}
unsafe impl Sync for AttrArray {}

const SHARDS: usize = 64;
const SHARD_BITS: u32 = 6; // log2(SHARDS)
const SHARD_MASK: u64 = (SHARDS as u64) - 1;
// Chunks of CHUNK consecutive ids share a shard, so sequential id batches
// (label scans) reuse one read lock per chunk instead of one per id.
const CHUNK_BITS: u32 = 6;
const CHUNK: u64 = 1 << CHUNK_BITS;
const CHUNK_MASK: u64 = CHUNK - 1;

struct Shard {
    /// Per-entity stored attributes, indexed by slot. Each entry is a single
    /// shared [`AttrArray`] whose heap header carries the MVCC version, so a
    /// slot is one 8-byte `NonNull` and an empty/absent slot (`None`) costs
    /// 8 bytes via the niche.
    ///
    /// Layout: shard = `(id >> CHUNK_BITS) & SHARD_MASK`, so `CHUNK`
    /// consecutive ids share one shard — sequential id batches (e.g.
    /// label scans) reuse a single read lock per chunk.
    entries: RwLock<Vec<Option<AttrArray>>>,
}

#[inline]
const fn shard_idx(entity_id: u64) -> usize {
    ((entity_id >> CHUNK_BITS) & SHARD_MASK) as usize
}

#[inline]
const fn slot_idx(entity_id: u64) -> usize {
    // Bijection with shard_idx. Reconstruction:
    //   id = ((slot >> CHUNK_BITS) << (CHUNK_BITS + SHARD_BITS))
    //         | (shard << CHUNK_BITS) | (slot & CHUNK_MASK)
    let high = entity_id >> (CHUNK_BITS + SHARD_BITS);
    let low = entity_id & CHUNK_MASK;
    ((high << CHUNK_BITS) | low) as usize
}

/// Shared, version-stamped, entity-level attribute storage.
///
/// This is the sole source of truth (no cold tier), so it is unbounded:
/// every live entity stays resident and entries are never evicted.
pub struct AttributeStorage {
    shards: Box<[Shard; SHARDS]>,
}

impl Default for AttributeStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl AttributeStorage {
    /// Create an empty attribute storage.
    #[must_use]
    pub fn new() -> Self {
        let shards: Vec<Shard> = (0..SHARDS)
            .map(|_| Shard {
                entries: RwLock::new(Vec::new()),
            })
            .collect();
        let shards: Box<[Shard; SHARDS]> = shards
            .into_boxed_slice()
            .try_into()
            .unwrap_or_else(|_| unreachable!("vec built with exactly SHARDS items"));
        Self { shards }
    }

    #[inline]
    fn shard(
        &self,
        entity_id: u64,
    ) -> &Shard {
        // SAFETY: shard_idx returns 0..SHARDS.
        unsafe { self.shards.get_unchecked(shard_idx(entity_id)) }
    }

    /// Look up a single attribute for an entity by index.
    ///
    /// Returns `Some(Some(value))` on a hit with the attribute present,
    /// `Some(None)` on a hit but attribute absent, and `None` on a miss.
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
        if entry.version() > version {
            return None;
        }
        Some(entry.get(attr_idx).cloned())
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
            if entry.version() > version {
                continue;
            }
            out[pos] = Some(entry.get(attr_idx).cloned());
        }
    }

    /// Fused batch lookup that writes resolved `Value`s straight into `out`.
    ///
    /// For each key this pushes exactly one `Value`:
    /// - hit with the attribute present  -> the cloned value,
    /// - hit with the attribute absent    -> `default`,
    /// - miss (entity not stored / newer) -> `default`, and the absolute index
    ///   into `out` is recorded in `missing` for caller fallback.
    pub fn get_attrs_batch_into(
        &self,
        keys: &[u64],
        attr_idx: u16,
        version: u64,
        default: &Value,
        out: &mut Vec<Value>,
        missing: &mut Vec<usize>,
    ) {
        if keys.is_empty() {
            return;
        }
        let base = out.len();
        let mut current_shard_idx = shard_idx(keys[0]);
        let mut guard = self.shards[current_shard_idx].entries.read();
        missing.reserve(keys.len() / 8);
        for (pos, &id) in keys.iter().enumerate() {
            let s = shard_idx(id);
            if s != current_shard_idx {
                drop(guard);
                current_shard_idx = s;
                guard = self.shards[s].entries.read();
            }
            let slot = slot_idx(id);
            if let Some(Some(entry)) = guard.get(slot)
                && entry.version() <= version
            {
                match entry.get(attr_idx) {
                    Some(v) => out.push(v.clone()),
                    None => out.push(default.clone()),
                }
                continue;
            }
            out.push(default.clone());
            missing.push(base + pos);
        }
    }

    /// Return all stored attributes for an entity (cheap shared clone).
    #[must_use]
    pub fn get_entity(
        &self,
        entity_id: u64,
        version: u64,
    ) -> Option<AttrArray> {
        let shard = self.shard(entity_id);
        let slot = slot_idx(entity_id);
        let entries = shard.entries.read();
        let entry = entries.get(slot)?.as_ref()?;
        if entry.version() > version {
            return None;
        }
        Some(entry.clone())
    }

    /// Check whether an entity has *any* stored attributes.
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
        if entry.version() > version {
            return None;
        }
        Some(!entry.is_empty())
    }

    /// Check whether an attr already exists for an entity.
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
        if entry.version() > version {
            return None;
        }
        Some(entry.get(attr_idx).is_some())
    }

    /// Insert (or replace) the full attribute set for an entity.
    pub fn insert_entity(
        &self,
        entity_id: u64,
        mut attrs: Vec<(u16, Value)>,
        version: u64,
    ) {
        attrs.sort_by_key(|item| item.0);
        self.insert_internal(entity_id, AttrArray::from_sorted(attrs, version));
    }

    /// Insert (or replace) the full attribute set when the caller guarantees
    /// the attrs are already sorted by `attr_idx`.
    pub fn insert_entity_presorted(
        &self,
        entity_id: u64,
        attrs: Vec<(u16, Value)>,
        version: u64,
    ) {
        debug_assert!(
            attrs.windows(2).all(|w| w[0].0 <= w[1].0),
            "insert_entity_presorted: attrs not sorted"
        );
        self.insert_internal(entity_id, AttrArray::from_sorted(attrs, version));
    }

    fn insert_internal(
        &self,
        entity_id: u64,
        attrs: AttrArray,
    ) {
        let shard = self.shard(entity_id);
        let slot = slot_idx(entity_id);
        let mut entries = shard.entries.write();
        if entries.len() <= slot {
            entries.resize(slot + 1, None);
        }
        // SAFETY: just resized to cover `slot`.
        let cell = unsafe { entries.get_unchecked_mut(slot) };
        *cell = Some(attrs);
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
            let mut entries = self.shards[i].entries.write();
            for &id in ids {
                let slot = slot_idx(id);
                if let Some(cell) = entries.get_mut(slot) {
                    *cell = None;
                }
            }
        }
    }

    /// Remove a single attribute from a stored entity.
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
        let Some(pos) = existing.position(attr_idx) else {
            return false;
        };
        // Preserve the entry's MVCC version when rebuilding the array.
        let version = existing.version();
        let mut pairs = existing.to_pairs();
        pairs.remove(pos);
        *existing = AttrArray::from_sorted(pairs, version);
        true
    }

    /// Estimated heap bytes of stored attribute payloads.
    ///
    /// Walks every live entry; intended for the (cold) memory-report path.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.shards
            .iter()
            .map(|s| {
                s.entries
                    .read()
                    .iter()
                    .filter_map(Option::as_ref)
                    .map(AttrArray::heap_bytes)
                    .sum::<usize>()
            })
            .sum()
    }

    /// Structural overhead of the slot vectors, excluding attribute payload.
    /// Grows monotonically as ids are allocated; does not shrink on removal.
    #[must_use]
    pub fn structural_memory_usage(&self) -> usize {
        let slot_size = std::mem::size_of::<Option<AttrArray>>();
        self.shards
            .iter()
            .map(|s| s.entries.read().len() * slot_size)
            .sum()
    }
}
