//! Attribute storage for graph entities (nodes and relationships).
//!
//! This module provides [`AttributeStore`], a two-tier key-value store for
//! entity properties. It uses an in-memory LRU cache as its primary hot-path
//! store and falls back to [`fjall`] (an LSM-tree disk store) for cold data.
//!
//! ## Two-Tier Storage Architecture
//!
//! ```text
//!  ┌─────────────────────────────────────────────────────────┐
//!  │                   AttributeStore                        │
//!  │                                                         │
//!  │  attrs_name: ["name", "age", "city"]                    │
//!  │              idx:  0      1      2                      │
//!  │                                                         │
//!  │  ┌───────────────────────────────────────────────────┐  │
//!  │  │         AttributeCache (shared via Arc)           │  │
//!  │  │  entity_id -> [(attr_idx, Value), ...]            │  │
//!  │  │  Each entry: version + dirty flag                 │  │
//!  │  │  Dirty entries pinned (cannot be evicted)         │  │
//!  │  └───────────────────────┬───────────────────────────┘  │
//!  │                          │                              │
//!  │            flush (when over budget)                     │
//!  │                          │                              │
//!  │  ┌───────────────────────▼───────────────────────────┐  │
//!  │  │              fjall Keyspace                       │  │
//!  │  │  Key: entity_id (8B BE) + attr_idx (2B BE)       │  │
//!  │  │  Value: serialized Value bytes                    │  │
//!  │  │  Snapshot isolation via fjall::Snapshot           │  │
//!  │  └───────────────────────────────────────────────────┘  │
//!  └─────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Read Path
//!
//! ```text
//!  get_attr(entity_id, "name")
//!       │
//!       ▼
//!  [1] Check cache ──hit──▶ return value (or None if absent)
//!       │
//!      miss
//!       │
//!       ▼
//!  [2] Scan fjall snapshot (prefix = entity_id)
//!       │
//!       ▼
//!  [3] Populate cache (dirty=false, version-guarded insert)
//!       │
//!       ▼
//!  [4] Return value from fetched attributes
//! ```
//!
//! ## Write Path
//!
//! ```text
//!  insert_attrs(entity_id, {name: "Alice", age: 30})
//!       │
//!       ▼
//!  [1] Resolve attr names to column indices (create if new)
//!       │
//!       ▼
//!  [2] Merge with current state (cache or fjall)
//!       │
//!       ▼
//!  [3] Write merged result to cache (dirty=true)
//!       │
//!       ▼
//!  [4] On commit: take new fjall snapshot, clear dirty tracking
//!      On rollback: invalidate dirty cache entries
//! ```
//!
//! ## MVCC Integration
//!
//! The cache is shared across MVCC versions via `Arc<AttributeCache>`.
//! Each `AttributeStore` carries its own MVCC `version` number. Cache
//! entries with a version newer than the reader's are ignored (invisible
//! uncommitted writes). On `new_version()`, the store is cloned cheaply
//! (Arc bump + bitmap clear) with a fresh `dirty_entities` tracker.
//!
//! ## Key Format (fjall)
//!
//! Each attribute is stored as a separate fjall entry:
//! `entity_id (8 bytes big-endian) + attr_idx (2 bytes big-endian)`

use std::{collections::HashMap, process, sync::Arc};

use std::cmp::Ordering;
use std::collections::HashMap as StdHashMap;

use fjall::{
    Database, Keyspace, KeyspaceCreateOptions, Readable, Snapshot, config::HashRatioPolicy,
};
use once_cell::sync::OnceCell;
use roaring::RoaringTreemap;

use super::attribute_cache::AttributeCache;
use super::graphblas::serialization::{Decode, Encode, Reader, Writer};
use crate::runtime::{ordermap::OrderMap, orderset::OrderSet, value::Value};

/// Create a composite key from entity ID and attribute index.
fn make_key(
    entity_id: u64,
    attr_idx: u16,
) -> [u8; 10] {
    let mut key = [0u8; 10];
    key[..8].copy_from_slice(&entity_id.to_be_bytes());
    key[8..].copy_from_slice(&attr_idx.to_be_bytes());
    key
}

/// Extract attribute index from a composite key.
fn extract_attr_idx(key: &[u8]) -> Option<u16> {
    if key.len() >= 10 {
        Some(u16::from_be_bytes([key[8], key[9]]))
    } else {
        None
    }
}

/// Columnar attribute storage for graph entities.
///
/// Uses a shared [`AttributeCache`] as the primary hot store and fjall as the
/// durable cold store.  The fjall keyspace is created lazily on first access
/// to avoid I/O overhead for graphs that fit entirely in cache.
pub struct AttributeStore {
    snapshot: OnceCell<Snapshot>,
    keyspace: OnceCell<Keyspace>,
    keyspace_name: Arc<String>,
    /// Attribute names in insertion order (name → column index)
    pub attrs_name: OrderSet<Arc<String>>,
    /// Shared in-memory LRU cache (cheap Arc clone across MVCC versions).
    cache: Arc<AttributeCache>,
    /// MVCC version of this store's snapshot.
    version: u64,
    /// Entity IDs dirtied during the current write tx (for rollback).
    dirty_entities: RoaringTreemap,
    /// Entity IDs pending full deletion (all attributes) — applied on commit, cleared on rollback.
    pending_deletes: RoaringTreemap,
}

impl Clone for AttributeStore {
    fn clone(&self) -> Self {
        Self {
            snapshot: self.snapshot.clone(),
            keyspace: self.keyspace.clone(),
            keyspace_name: self.keyspace_name.clone(),
            attrs_name: self.attrs_name.clone(),
            cache: self.cache.clone(),
            version: self.version,
            dirty_entities: self.dirty_entities.clone(),
            pending_deletes: self.pending_deletes.clone(),
        }
    }
}

/// Default memory budget per attribute cache (2 GiB).
const DEFAULT_ATTR_CACHE_BYTES: usize = 2 * 1024 * 1024 * 1024;

static DATABASE: OnceCell<Database> = OnceCell::new();

/// Get or initialize the shared fjall database for attribute stores.
fn get_database() -> Database {
    DATABASE
        .get_or_init(|| {
            Database::builder(format!("./attrs/{}", process::id()))
                .temporary(true)
                .manual_journal_persist(true)
                .cache_size(128 * 1_024 * 1_024)
                .open()
                .expect("failed to open fjall database")
        })
        .clone()
}

impl AttributeStore {
    #[must_use]
    pub fn new(
        keyspace: &str,
        version: u64,
    ) -> Self {
        Self {
            snapshot: OnceCell::new(),
            keyspace: OnceCell::new(),
            keyspace_name: Arc::new(keyspace.to_owned()),
            attrs_name: OrderSet::default(),
            cache: Arc::new(AttributeCache::new(DEFAULT_ATTR_CACHE_BYTES)),
            version,
            dirty_entities: RoaringTreemap::new(),
            pending_deletes: RoaringTreemap::new(),
        }
    }

    /// Get-or-create the fjall keyspace lazily, clearing stale data if present.
    ///
    /// Also initializes the snapshot so it is always taken *after* the keyspace
    /// has been cleared, preventing reads of stale data from a previously-deleted
    /// graph that reused the same keyspace name.
    ///
    /// # Panics
    ///
    /// Panics if the fjall keyspace cannot be created or cleared. This is
    /// intentional: a failure here means the storage backend is broken and
    /// the process cannot continue safely.
    fn keyspace(&self) -> &Keyspace {
        self.keyspace.get_or_init(|| {
            let db = get_database();
            let ks_exists = db.keyspace_exists(&self.keyspace_name);
            let ks = db
                .keyspace(&self.keyspace_name, || {
                    KeyspaceCreateOptions::default()
                        .data_block_hash_ratio_policy(HashRatioPolicy::all(0.75))
                        .expect_point_read_hits(true)
                        .manual_journal_persist(true)
                })
                .expect("failed to create fjall keyspace");
            if ks_exists && ks.approximate_len() > 0 {
                ks.clear().expect("failed to clear existing fjall keyspace");
            }
            ks
        })
    }

    /// Get the fjall snapshot, taking one lazily if needed.
    ///
    /// On a freshly-constructed store the snapshot is initialized by the first
    /// `keyspace()` call (which clears stale data first).  Subsequent MVCC
    /// versions (`new_version`) and commits set it eagerly.
    fn snapshot(&self) -> &Snapshot {
        self.snapshot.get_or_init(|| {
            // Ensure the keyspace is initialized (and stale data cleared) before
            // taking a snapshot, so the new version never sees data from a
            // previously-deleted graph that reused the same keyspace name.
            let _ = self.keyspace();
            get_database().snapshot()
        })
    }

    #[must_use]
    pub fn new_version(
        &self,
        version: u64,
    ) -> Self {
        Self {
            snapshot: self.snapshot.clone(),
            keyspace: self.keyspace.clone(),
            keyspace_name: self.keyspace_name.clone(),
            attrs_name: self.attrs_name.clone(),
            cache: self.cache.clone(),
            version,
            dirty_entities: RoaringTreemap::new(),
            pending_deletes: RoaringTreemap::new(),
        }
    }

    // ---- helpers --------------------------------------------------------

    /// Fetch ALL attributes for `entity_id` from the fjall snapshot and
    /// populate the cache as a clean entry.
    ///
    /// Uses a version-aware insert to avoid overwriting in-flight dirty writes:
    /// the cache entry is only updated if no newer/dirty entry already exists.
    /// Empty entries are cached to prevent repeated fjall scans for non-existent
    /// entities. Returns empty if the entity is pending full deletion.
    fn populate_cache_from_fjall(
        &self,
        entity_id: u64,
    ) -> Arc<Vec<(u16, Value)>> {
        // If this entity is pending full deletion, return empty regardless of fjall state.
        if self.pending_deletes.contains(entity_id) {
            return Arc::new(Vec::new());
        }
        let prefix = entity_id.to_be_bytes();
        let attrs: Vec<(u16, Value)> = self
            .snapshot()
            .prefix(self.keyspace(), prefix)
            .filter_map(|entry| {
                let (k, data) = entry.into_inner().ok()?;
                let idx = extract_attr_idx(&k)?;
                let (value, _) = Value::from_bytes(&data)?;
                Some((idx, value))
            })
            .collect();
        // Always cache the result (even empty entries) using safe insert that
        // respects in-flight writes: only insert if no newer/dirty entry exists.
        let _ = self
            .cache
            .insert_entity_if_older(entity_id, attrs.clone(), self.version);
        Arc::new(attrs)
    }

    // ---- read path (cache → fjall) --------------------------------------

    pub fn remove(
        &mut self,
        key: u64,
    ) -> Result<(), String> {
        // Don't invalidate cache — older MVCC versions sharing this cache may
        // still need the dirty entry. pending_deletes guards reads on this
        // version; the cache entry is harmless to older/newer readers because
        // the version check in the cache handles visibility.
        self.dirty_entities.insert(key);
        self.pending_deletes.insert(key);
        Ok(())
    }

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
        if let Some(result) = self.cache.get_attr(key, attr_idx, self.version) {
            return result;
        }
        // 2. Cache miss — populate from fjall.
        let attrs = self.populate_cache_from_fjall(key);
        attrs
            .binary_search_by_key(&attr_idx, |(idx, _)| *idx)
            .ok()
            .map(|pos| attrs[pos].1.clone())
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
        if let Some(has) = self.cache.has_entity(key, self.version) {
            return has;
        }
        // Fallback to fjall, populating cache to avoid repeated scans.
        let attrs = self.populate_cache_from_fjall(key);
        !attrs.is_empty()
    }

    pub fn get_attrs(
        &self,
        key: u64,
    ) -> impl Iterator<Item = Arc<String>> + '_ {
        if self.pending_deletes.contains(key) {
            return Vec::new().into_iter();
        }
        // Try cache first.
        let cached = self.cache.get_entity(key, self.version);
        let attrs = cached.unwrap_or_else(|| self.populate_cache_from_fjall(key));
        attrs
            .iter()
            .filter_map(move |(idx, _)| {
                let i = *idx as usize;
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
        let cached = self.cache.get_entity(key, self.version);
        let attrs = cached.unwrap_or_else(|| self.populate_cache_from_fjall(key));
        attrs
            .iter()
            .filter_map(move |(idx, value)| {
                let i = *idx as usize;
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
    ) -> Arc<Vec<(u16, Value)>> {
        if self.pending_deletes.contains(key) {
            return Arc::new(Vec::new());
        }
        self.cache
            .get_entity(key, self.version)
            .unwrap_or_else(|| self.populate_cache_from_fjall(key))
    }

    // ---- write path (cache only) ----------------------------------------

    pub fn remove_attr(
        &mut self,
        key: u64,
        attr: &Arc<String>,
    ) -> Result<bool, String> {
        if let Some(idx) = self.attrs_name.get_index_of(attr) {
            let attr_idx = idx as u16;
            // Check if the attr exists (cache or fjall).
            let exists = self
                .cache
                .contains_attr(key, attr_idx, self.version)
                .unwrap_or_else(|| {
                    let composite_key = make_key(key, attr_idx);
                    self.snapshot()
                        .contains_key(self.keyspace(), composite_key)
                        .unwrap_or(false)
                });
            if exists {
                // Try to remove from cache. If not in cache, populate from fjall first.
                let removed = self.cache.remove_attr_from_entity(key, attr_idx);
                if !removed {
                    // Attr is in fjall but not in cache. Populate cache from fjall,
                    // then remove the attr from the cached entry.
                    let _ = self.populate_cache_from_fjall(key);
                    let _ = self.cache.remove_attr_from_entity(key, attr_idx);
                }
                self.dirty_entities.insert(key);
                // Don't immediately delete from fjall; let the flush logic persist the removal
                // when the entity is flushed with its updated attribute set.
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn remove_all(
        &mut self,
        keys: &RoaringTreemap,
    ) {
        for key in keys {
            self.dirty_entities.insert(key);
            self.pending_deletes.insert(key);
        }
    }

    /// Batch insert/update multiple attributes for entities.
    ///
    /// Writes go to the in-memory cache (`dirty = true`).  Returns `(nremoved, nset)`:
    /// the number of attributes *replaced* and the number of non-null attributes *set*.
    pub fn insert_attrs(
        &mut self,
        attrs: &HashMap<u64, OrderMap<Arc<String>, Value>>,
    ) -> Result<(usize, usize), String> {
        let mut nremoved = 0;
        let mut nset = 0;

        // Pre-resolve all unique attribute names → indices ONCE.
        // Uses Arc pointer identity as key to avoid rehashing strings.
        let mut name_to_idx: StdHashMap<*const String, u16> =
            StdHashMap::with_capacity(attrs.values().next().map_or(0, |v| v.len()));
        for entity_attrs in attrs.values() {
            for (attr, _) in entity_attrs.iter() {
                let ptr = Arc::as_ptr(attr);
                if !name_to_idx.contains_key(&ptr) {
                    let idx = self.attrs_name.get_index_of(attr).unwrap_or_else(|| {
                        self.attrs_name.insert(attr.clone());
                        self.attrs_name.len() - 1
                    }) as u16;
                    name_to_idx.insert(ptr, idx);
                }
            }
        }

        // Reusable buffer to avoid per-entity allocation.
        let mut merged: Vec<(u16, Value)> = Vec::new();

        for (key, entity_attrs) in attrs {
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

            // Get current state: cache first, then fjall.
            let current = self
                .cache
                .get_entity(*key, self.version)
                .unwrap_or_else(|| self.populate_cache_from_fjall(*key));

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
                merged.extend(new_entries.drain(..));
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

            // Write merged attrs to cache as dirty (already sorted, skip re-sort).
            self.cache.insert_entity_presorted(
                *key,
                std::mem::take(&mut merged),
                self.version,
                true,
            );
            self.dirty_entities.insert(*key);
        }

        Ok((nremoved, nset))
    }

    /// Bulk import attributes for entities known to be new (no prior state).
    ///
    /// Optimized for RDB decode: skips cache/fjall lookups since entities
    /// don't exist yet. Attributes are written directly to cache.
    /// Returns the number of non-null attributes imported.
    pub fn import_attrs(
        &mut self,
        attrs: &HashMap<u64, OrderMap<Arc<String>, Value>>,
    ) -> usize {
        // Pre-resolve all unique attribute names → indices ONCE.
        let mut name_to_idx: StdHashMap<*const String, u16> =
            StdHashMap::with_capacity(attrs.values().next().map_or(0, |v| v.len()));
        for entity_attrs in attrs.values() {
            for (attr, _) in entity_attrs.iter() {
                let ptr = Arc::as_ptr(attr);
                if !name_to_idx.contains_key(&ptr) {
                    let idx = self.attrs_name.get_index_of(attr).unwrap_or_else(|| {
                        self.attrs_name.insert(attr.clone());
                        self.attrs_name.len() - 1
                    }) as u16;
                    name_to_idx.insert(ptr, idx);
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
            self.cache
                .insert_entity_presorted(*key, entries, self.version, true);
            self.dirty_entities.insert(*key);
        }
        nset
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
        self.cache.memory_usage()
    }

    pub fn commit(&mut self) -> Result<(), String> {
        // Apply pending full entity deletions to fjall.
        if !self.pending_deletes.is_empty() {
            // Only scan fjall if the keyspace was already initialized AND has entries.
            // For freshly created entities that were never flushed to fjall,
            // the keyspace was never accessed, so we skip the expensive
            // keyspace initialization + prefix scans entirely.
            if self
                .keyspace
                .get()
                .is_some_and(|ks| ks.approximate_len() > 0)
            {
                let mut batch = get_database().batch();
                // Targeted prefix scans: O(pending_deletes × attrs_per_entity)
                // instead of scanning the entire keyspace.
                for entity_id in &self.pending_deletes {
                    let prefix = entity_id.to_be_bytes();
                    for entry in self.keyspace().prefix(prefix) {
                        if let Ok(k) = entry.key() {
                            batch.remove(self.keyspace(), k);
                        }
                    }
                }
                batch.durability(None).commit().map_err(|e| e.to_string())?;
            }
            // Invalidate deleted entities from the shared cache to prevent stale reads.
            self.cache.invalidate_batch(&self.pending_deletes);
        }
        // Only refresh the fjall snapshot if the database/keyspace was already
        // initialized. For stores that never touched fjall (all data in cache),
        // skip the expensive database + keyspace initialization.
        if self.keyspace.get().is_some() {
            let new_snapshot = OnceCell::new();
            let _ = new_snapshot.set(get_database().snapshot());
            self.snapshot = new_snapshot;
        }
        self.dirty_entities.clear();
        self.pending_deletes.clear();
        Ok(())
    }

    // ---- flush / rollback -----------------------------------------------

    /// Invalidate all dirty entities from the shared cache.
    /// Called on write-transaction rollback.
    pub fn rollback_cache(&mut self) {
        self.cache.invalidate_batch(&self.dirty_entities);
        self.dirty_entities.clear();
        self.pending_deletes.clear();
    }

    /// Flush dirty cache entries to fjall.
    ///
    /// Collects up to `n` least-recently-used dirty entries, writes them to
    /// fjall in a single batch, then evicts clean entries until memory is
    /// within budget.
    pub fn flush_dirty_to_fjall(
        &self,
        n: usize,
    ) -> Result<(), String> {
        let dirty_entries = self.cache.collect_dirty_lru(n);
        if dirty_entries.is_empty() {
            return Ok(());
        }

        let mut batch = get_database().batch();
        for (entity_id, attrs) in &dirty_entries {
            // Delete all existing fjall keys for this entity first, so that
            // removed attributes don't reappear after cache eviction.
            let prefix = entity_id.to_be_bytes();
            for entry in self.keyspace().prefix(prefix) {
                if let Ok(k) = entry.key() {
                    batch.remove(self.keyspace(), k);
                }
            }
            // Then insert the current attribute set.
            for &(attr_idx, ref value) in attrs.iter() {
                let composite_key = make_key(*entity_id, attr_idx);
                batch.insert(self.keyspace(), composite_key, value.to_bytes());
            }
        }
        batch.durability(None).commit().map_err(|e| {
            // Re-insert entries to prevent data loss on commit failure.
            for (entity_id, attrs) in dirty_entries {
                self.cache
                    .insert_entity(entity_id, (*attrs).clone(), self.version, true);
            }
            e.to_string()
        })?;

        Ok(())
    }

    /// Flush an entity's pending dirty attributes to fjall, then invalidate from cache.
    ///
    /// This ensures that any unflushed writes to the cache are persisted to fjall
    /// before the cache entry is removed, preventing data loss when the entry is
    /// about to be deleted from fjall.
    ///
    /// However, if the entity was modified by the current transaction
    /// (`dirty_entities`), the flush is skipped — those writes are uncommitted
    /// and must not be persisted to fjall until `commit()`.  This prevents
    /// rollback from leaving current-tx inserts in the durable store.
    fn flush_and_invalidate(
        &self,
        entity_id: u64,
    ) -> Result<(), String> {
        if !self.dirty_entities.contains(entity_id)
            && let Some((cached, dirty)) = self.cache.get_entity_with_dirty(entity_id, self.version)
            && dirty
            && !cached.is_empty()
        {
            // Write dirty cached attributes to fjall before losing the cache entry.
            // Safe to flush: these are pre-existing dirty entries from prior
            // transactions, not from the active one.
            let mut batch = get_database().batch();
            for &(attr_idx, ref value) in cached.iter() {
                let composite_key = make_key(entity_id, attr_idx);
                batch.insert(self.keyspace(), composite_key, value.to_bytes());
            }
            batch.durability(None).commit().map_err(|e| e.to_string())?;
        }
        self.cache.invalidate(entity_id);
        Ok(())
    }

    /// Access the shared cache (for background flush scheduling).
    #[must_use]
    pub const fn cache(&self) -> &Arc<AttributeCache> {
        &self.cache
    }

    /// Encode a range of entities, borrowing the deleted bitmap directly.
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
        let global_index: std::collections::HashMap<&Arc<String>, usize> = global_attrs
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

            for &(local_attr_id, ref value) in props.iter() {
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
// - `Database`, `Snapshot`, `Keyspace` are thread-safe (fjall guarantees)
// - `AttributeCache` is wrapped in `Arc` and uses sharded locks internally
// - `OnceCell<Keyspace>` is `Sync` (interior init is thread-safe)
// - All other fields (`RoaringTreemap`, `OrderSet`, etc.) are owned and not shared
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
                self.cache
                    .insert_entity(entity_id, entries, self.version, true);
                self.dirty_entities.insert(entity_id);
            }
        }
        Ok(())
    }
}
