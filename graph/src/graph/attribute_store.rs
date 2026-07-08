//! Attribute storage for graph entities (nodes and relationships).
//!
//! This module provides [`AttributeStore`], an in-memory key-value store for
//! entity properties modeled on the C engine's DataBlock: entity ids index
//! into fixed-capacity blocks of slots, and each slot is a single thin
//! pointer to the entity's attribute set ([`AttrArray`]) — one contiguous
//! allocation of `(attr_id, value)` pairs sorted by attribute id.
//!
//! ## Concurrency / MVCC
//!
//! The store contains no locks and no unsafe code. Isolation comes from the
//! MVCC design: a write transaction operates on a *private* clone of the
//! graph (`Graph::new_version`), which is published atomically on commit or
//! simply discarded on rollback. Cloning the store is cheap — blocks are
//! shared via `Arc` and copied on first write per block (`Arc::make_mut`),
//! so older snapshots keep reading their own blocks untouched.

use std::cmp::Ordering;
use std::sync::Arc;

use roaring::RoaringTreemap;
use rustc_hash::FxHashMap;
use triomphe::ThinArc;

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

/// Shared empty attribute set to avoid per-call allocations when an entity
/// has no properties.
static EMPTY_ATTRS: once_cell::sync::Lazy<AttrArray> = once_cell::sync::Lazy::new(AttrArray::empty);

/// Reference-counted, single-allocation attribute set for one entity.
///
/// A thin (8-byte) pointer to one heap block holding the refcount, the
/// length, and the `(attr_id, value)` pairs sorted by attribute id —
/// mirroring the C engine's `AttributeSet` where the entity slot is a single
/// pointer and all metadata lives in the pointee. Contents are immutable
/// after construction; mutation builds a new array.
#[derive(Clone)]
pub struct AttrArray(ThinArc<(), (u16, Value)>);

impl AttrArray {
    /// Build from attribute pairs already sorted by attr id.
    #[must_use]
    fn from_sorted(pairs: Vec<(u16, Value)>) -> Self {
        Self(ThinArc::from_header_and_iter((), pairs.into_iter()))
    }

    #[must_use]
    fn empty() -> Self {
        Self::from_sorted(Vec::new())
    }

    #[inline]
    fn pairs(&self) -> &[(u16, Value)] {
        &self.0.slice
    }

    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.pairs().len()
    }

    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pairs().is_empty()
    }

    /// Position of `attr_idx` within the sorted pairs, if present.
    #[inline]
    #[must_use]
    pub fn position(
        &self,
        attr_idx: u16,
    ) -> Option<usize> {
        self.pairs()
            .binary_search_by_key(&attr_idx, |&(idx, _)| idx)
            .ok()
    }

    /// Value for `attr_idx`, if the attribute is present.
    #[inline]
    #[must_use]
    pub fn get(
        &self,
        attr_idx: u16,
    ) -> Option<&Value> {
        self.position(attr_idx).map(|pos| &self.pairs()[pos].1)
    }

    /// Iterate `(attr_idx, &Value)` pairs in index order.
    pub fn iter(&self) -> impl Iterator<Item = (u16, &Value)> + '_ {
        self.pairs().iter().map(|(idx, value)| (*idx, value))
    }

    /// Materialize an owned `Vec` of `(attr_idx, Value)` pairs.
    #[must_use]
    pub fn to_pairs(&self) -> Vec<(u16, Value)> {
        self.pairs().to_vec()
    }

    /// Estimated heap bytes of this entity's single allocation, including the
    /// out-of-line heap owned by each `Value`.
    #[must_use]
    fn heap_bytes(&self) -> usize {
        // ThinArc allocation: refcount (usize) + length (usize) + pairs.
        std::mem::size_of::<usize>() * 2
            + self.len() * std::mem::size_of::<(u16, Value)>()
            + self
                .pairs()
                .iter()
                .map(|(_, v)| v.heap_size())
                .sum::<usize>()
    }
}

// ============================================================================
// DataBlock: block-allocated slot storage (C engine's DataBlock, COW)
// ============================================================================

/// Slots per block, matching the C engine's DataBlock granularity.
const BLOCK_CAP: usize = 16384;

/// One block of entity slots. Slots grow lazily up to [`BLOCK_CAP`]; each
/// slot is a single 8-byte thin pointer (`None` = entity has no attributes).
#[derive(Clone, Default)]
struct Block {
    slots: Vec<Option<AttrArray>>,
}

/// Block-allocated attribute storage indexed by entity id.
///
/// Blocks are shared across MVCC versions via `Arc` and copied on first
/// write per version (`Arc::make_mut`), so snapshots never observe writes.
#[derive(Clone, Default)]
struct DataBlock {
    blocks: Vec<Arc<Block>>,
}

impl DataBlock {
    #[inline]
    const fn locate(entity_id: u64) -> (usize, usize) {
        let id = entity_id as usize;
        (id / BLOCK_CAP, id % BLOCK_CAP)
    }

    #[inline]
    fn get(
        &self,
        entity_id: u64,
    ) -> Option<&AttrArray> {
        let (b, o) = Self::locate(entity_id);
        self.blocks.get(b)?.slots.get(o)?.as_ref()
    }

    fn set(
        &mut self,
        entity_id: u64,
        attrs: AttrArray,
    ) {
        let (b, o) = Self::locate(entity_id);
        if self.blocks.len() <= b {
            self.blocks.resize_with(b + 1, Arc::default);
        }
        let block = Arc::make_mut(&mut self.blocks[b]);
        if block.slots.len() <= o {
            block.slots.resize_with(o + 1, || None);
        }
        block.slots[o] = Some(attrs);
    }

    fn remove(
        &mut self,
        entity_id: u64,
    ) {
        let (b, o) = Self::locate(entity_id);
        // Check occupancy before make_mut so clearing an already-empty slot
        // doesn't deep-copy a shared block.
        let Some(block) = self.blocks.get(b) else {
            return;
        };
        if !block.slots.get(o).is_some_and(Option::is_some) {
            return;
        }
        Arc::make_mut(&mut self.blocks[b]).slots[o] = None;
    }

    /// Estimated heap bytes of stored attribute payloads.
    fn memory_usage(&self) -> usize {
        self.blocks
            .iter()
            .map(|block| {
                block
                    .slots
                    .iter()
                    .filter_map(Option::as_ref)
                    .map(AttrArray::heap_bytes)
                    .sum::<usize>()
            })
            .sum()
    }

    /// Structural overhead of the slot storage, excluding attribute payload.
    fn structural_memory_usage(&self) -> usize {
        let slot_size = std::mem::size_of::<Option<AttrArray>>();
        self.blocks.len() * std::mem::size_of::<Arc<Block>>()
            + self
                .blocks
                .iter()
                .map(|block| block.slots.len() * slot_size)
                .sum::<usize>()
    }
}

// ============================================================================
// AttributeStore
// ============================================================================

/// Attribute storage for graph entities, keyed by entity id.
///
/// Holds the attribute-name table and a copy-on-write [`DataBlock`]. A slot
/// miss means the entity has no attributes.
#[derive(Clone)]
pub struct AttributeStore {
    /// Attribute names in insertion order (name → column index)
    pub attrs_name: AttrNameMap,
    /// Block-allocated per-entity attribute sets (COW across MVCC versions).
    data: DataBlock,
}

impl AttributeStore {
    #[must_use]
    pub fn new(_version: u64) -> Self {
        Self {
            attrs_name: AttrNameMap::default(),
            data: DataBlock::default(),
        }
    }

    /// Cheap snapshot clone for a new MVCC version (one `Arc` bump per block).
    #[must_use]
    pub fn new_version(
        &self,
        _version: u64,
    ) -> Self {
        self.clone()
    }

    // ---- read path --------------------------------------------------------

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
        self.data.get(key)?.get(attr_idx).cloned()
    }

    /// Batch variant of `get_attr_by_idx` for a list of keys with the same
    /// `attr_idx`. Pushes one `Value` per key into `out`, substituting
    /// `default` for missing entries.
    pub fn get_attrs_by_idx_batch_into(
        &self,
        keys: &[u64],
        attr_idx: u16,
        default: &Value,
        out: &mut Vec<Value>,
    ) {
        out.reserve(keys.len());
        for &key in keys {
            let value = self
                .data
                .get(key)
                .and_then(|attrs| attrs.get(attr_idx))
                .cloned();
            out.push(value.unwrap_or_else(|| default.clone()));
        }
    }

    #[must_use]
    pub fn has_attributes(
        &self,
        key: u64,
    ) -> bool {
        self.data.get(key).is_some_and(|attrs| !attrs.is_empty())
    }

    pub fn get_attrs(
        &self,
        key: u64,
    ) -> impl Iterator<Item = Arc<String>> + '_ {
        let attrs = self
            .data
            .get(key)
            .cloned()
            .unwrap_or_else(|| EMPTY_ATTRS.clone());
        attrs
            .iter()
            .filter_map(move |(idx, _)| self.attrs_name.get(idx as usize).cloned())
            .collect::<Vec<_>>()
            .into_iter()
    }

    pub fn get_all_attrs(
        &self,
        key: u64,
    ) -> Vec<(Arc<String>, Value)> {
        let attrs = self
            .data
            .get(key)
            .cloned()
            .unwrap_or_else(|| EMPTY_ATTRS.clone());
        attrs
            .iter()
            .filter_map(move |(idx, value)| {
                self.attrs_name
                    .get(idx as usize)
                    .map(|name| (name.clone(), value.clone()))
            })
            .collect::<Vec<_>>()
    }

    pub fn get_all_attrs_by_id(
        &self,
        key: u64,
    ) -> AttrArray {
        self.data
            .get(key)
            .cloned()
            .unwrap_or_else(|| EMPTY_ATTRS.clone())
    }

    // ---- write path --------------------------------------------------------

    /// Remove all attributes for the given entities (applied immediately).
    pub fn remove_all(
        &mut self,
        keys: &RoaringTreemap,
    ) {
        for key in keys {
            self.data.remove(key);
        }
    }

    /// Batch insert/update multiple attributes for entities.
    ///
    /// Returns `(nremoved, nset)`: the number of attributes *replaced* and
    /// the number of non-null attributes *set*.
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
                    e.insert(self.get_or_create_attr_id(attr));
                }
            }
        }

        // Reusable buffer to avoid per-entity allocation.
        let mut merged: Vec<(u16, Value)> = Vec::new();

        for (key, entity_attrs) in attrs {
            // Skip entities whose pending map is empty: no entries to write,
            // no nulls to remove. Avoids creating an empty slot per entity
            // (matches C's NULL AttributeSet behaviour).
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

            // Current stored state (cheap Arc clone detaches the borrow).
            let current_arr = self.data.get(*key).cloned();
            let current: &[(u16, Value)] = current_arr.as_ref().map_or(&[], AttrArray::pairs);

            // Sort new entries for O(n+m) merge.
            new_entries.sort_unstable_by_key(|(idx, _)| *idx);

            // Fast path: if no nulls AND all new entries come after all current
            // entries, just clone current + append new without a full merge.
            if null_indices.is_empty()
                && !new_entries.is_empty()
                && (current.is_empty() || current.last().unwrap().0 < new_entries[0].0)
            {
                merged.clear();
                merged.reserve(current.len() + new_entries.len());
                merged.extend_from_slice(current);
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

            self.data
                .set(*key, AttrArray::from_sorted(std::mem::take(&mut merged)));
        }

        Ok((nremoved, nset))
    }

    /// Bulk import attributes for entities known to be new (no prior state).
    ///
    /// Optimized for RDB decode: skips slot lookups since entities don't
    /// exist yet. Returns the number of non-null attributes imported.
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
                    e.insert(self.get_or_create_attr_id(attr));
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

            // Skip empty entities to avoid per-entity slot overhead (matches
            // C's NULL AttributeSet behaviour for prop-less entities).
            if entries.is_empty() {
                continue;
            }
            entries.sort_by_key(|(idx, _)| *idx);
            self.data.set(*key, AttrArray::from_sorted(entries));
        }
        nset
    }

    /// Import pre-resolved attribute data directly.
    /// Skips name resolution and OrderMap construction; used by bulk insert.
    pub fn import_attrs_resolved(
        &mut self,
        data: &mut Vec<(u64, Vec<(u16, Value)>)>,
    ) -> usize {
        let mut nset = 0;
        for (entity_id, mut entries) in data.drain(..) {
            if entries.is_empty() {
                continue;
            }
            nset += entries.len();
            entries.sort_by_key(|(idx, _)| *idx);
            self.data.set(entity_id, AttrArray::from_sorted(entries));
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
        self.data.memory_usage()
    }

    /// Structural slot-storage overhead, excluding attribute payload heap.
    #[must_use]
    pub fn structural_memory_usage(&self) -> usize {
        self.data.structural_memory_usage()
    }

    // ---- serialization -----------------------------------------------------

    /// Encode a range of entities.
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
                self.data.set(entity_id, AttrArray::from_sorted(entries));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn name(s: &str) -> Arc<String> {
        Arc::new(s.to_string())
    }

    fn store_with(entries: &[(u64, &[(&str, Value)])]) -> AttributeStore {
        let mut store = AttributeStore::new(0);
        let mut attrs: FxHashMap<u64, OrderMap<Arc<String>, Value>> = FxHashMap::default();
        for (id, pairs) in entries {
            let mut map = OrderMap::default();
            for (attr, value) in *pairs {
                map.insert(name(attr), value.clone());
            }
            attrs.insert(*id, map);
        }
        store.insert_attrs(&attrs).unwrap();
        store
    }

    #[test]
    fn slot_is_pointer_sized() {
        assert_eq!(std::mem::size_of::<Option<AttrArray>>(), 8);
        assert_eq!(std::mem::size_of::<AttrArray>(), 8);
    }

    #[test]
    fn insert_and_get() {
        let store = store_with(&[(3, &[("a", Value::Int(1)), ("b", Value::Int(2))])]);
        assert_eq!(store.get_attr(3, &name("a")), Some(Value::Int(1)));
        assert_eq!(store.get_attr(3, &name("b")), Some(Value::Int(2)));
        assert_eq!(store.get_attr(3, &name("c")), None);
        assert_eq!(store.get_attr(4, &name("a")), None);
        assert!(store.has_attributes(3));
        assert!(!store.has_attributes(4));
    }

    #[test]
    fn insert_counts_and_null_removal() {
        let mut store = store_with(&[(1, &[("a", Value::Int(1)), ("b", Value::Int(2))])]);

        // Overwrite `a`, remove `b` via null, add `c`.
        let mut map = OrderMap::default();
        map.insert(name("a"), Value::Int(10));
        map.insert(name("b"), Value::Null);
        map.insert(name("c"), Value::Int(3));
        let mut attrs = FxHashMap::default();
        attrs.insert(1u64, map);
        let (nremoved, nset) = store.insert_attrs(&attrs).unwrap();

        assert_eq!(nremoved, 2); // `a` replaced + `b` removed
        assert_eq!(nset, 2); // `a` and `c` set
        assert_eq!(store.get_attr(1, &name("a")), Some(Value::Int(10)));
        assert_eq!(store.get_attr(1, &name("b")), None);
        assert_eq!(store.get_attr(1, &name("c")), Some(Value::Int(3)));
    }

    #[test]
    fn snapshot_isolation_on_new_version() {
        let v1 = store_with(&[(1, &[("a", Value::Int(1))])]);
        let mut v2 = v1.new_version(1);

        // Mutate v2: overwrite entity 1, add entity 2, delete entity 1 attrs.
        let mut map = OrderMap::default();
        map.insert(name("a"), Value::Int(99));
        let mut attrs = FxHashMap::default();
        attrs.insert(1u64, map.clone());
        attrs.insert(2u64, map);
        v2.insert_attrs(&attrs).unwrap();

        assert_eq!(v2.get_attr(1, &name("a")), Some(Value::Int(99)));
        assert_eq!(v2.get_attr(2, &name("a")), Some(Value::Int(99)));
        // v1 unchanged.
        assert_eq!(v1.get_attr(1, &name("a")), Some(Value::Int(1)));
        assert_eq!(v1.get_attr(2, &name("a")), None);

        let mut deleted = RoaringTreemap::new();
        deleted.insert(1);
        v2.remove_all(&deleted);
        assert!(!v2.has_attributes(1));
        assert_eq!(v1.get_attr(1, &name("a")), Some(Value::Int(1)));
    }

    #[test]
    fn remove_all_is_immediate() {
        let mut store = store_with(&[(5, &[("a", Value::Int(1))])]);
        let mut keys = RoaringTreemap::new();
        keys.insert(5);
        store.remove_all(&keys);
        assert_eq!(store.get_attr(5, &name("a")), None);
        assert!(!store.has_attributes(5));
        assert!(store.get_all_attrs(5).is_empty());
    }

    #[test]
    fn remove_absent_slot_does_not_copy_block() {
        let mut store = store_with(&[(1, &[("a", Value::Int(1))])]);
        let block_before = store.data.blocks[0].clone();

        // Entity 2 shares block 0 but has no attributes; removing it must not
        // trigger a COW copy of the block.
        let mut keys = RoaringTreemap::new();
        keys.insert(2);
        // Also an id in a block that doesn't exist at all.
        keys.insert(BLOCK_CAP as u64 * 10);
        store.remove_all(&keys);

        assert!(Arc::ptr_eq(&block_before, &store.data.blocks[0]));
    }

    #[test]
    fn batch_get_with_default() {
        let store = store_with(&[(1, &[("a", Value::Int(1))]), (2, &[("b", Value::Int(2))])]);
        let idx = store.get_attr_id(&name("a")).unwrap() as u16;
        let mut out = Vec::new();
        store.get_attrs_by_idx_batch_into(&[1, 2, 3], idx, &Value::Null, &mut out);
        assert_eq!(out, vec![Value::Int(1), Value::Null, Value::Null]);
    }

    #[test]
    fn cross_block_ids() {
        let far_id = BLOCK_CAP as u64 * 3 + 7;
        let store = store_with(&[(far_id, &[("a", Value::Int(42))])]);
        assert_eq!(store.get_attr(far_id, &name("a")), Some(Value::Int(42)));
        assert_eq!(store.get_attr(far_id - 1, &name("a")), None);
    }
}
