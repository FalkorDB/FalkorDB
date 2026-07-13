//! Attribute storage for graph entities (nodes and relationships).
//!
//! This module provides [`AttributeStore`], an in-memory key-value store for
//! entity properties modeled on the C engine's DataBlock: entity ids index
//! into fixed-capacity blocks, and each block stores all of its entities'
//! attributes in one contiguous arena of packed 12-byte
//! `(attr_id, tag, payload)` entries sorted by attribute id. A slot is an
//! 8-byte `(offset, len, cap)` span descriptor into the arena, so entities
//! cost no individual allocations at all. Heap values (strings, lists,
//! vectors) are stored once in a block-level side array and referenced by
//! index from the packed payload.
//!
//! ## Memory layout
//!
//! ```text
//! DataBlock
//! ┌────────────────────────────────────────────────────────────────────┐
//! │ blocks: Vec<Arc<Block>>   (entity_id / block_cap() picks a block)  │
//! │   [ Arc<Block 0> ][ Arc<Block 1> ][ Arc<Block 2> ] ...             │
//! └───────────┬────────────────────────────────────────────────────────┘
//!             │ shared across MVCC snapshots, COW on first write
//!             ▼
//! Block
//! ┌────────────────────────────────────────────────────────────────────┐
//! │ slots: Vec<Slot>          one 8-byte span descriptor per entity    │
//! │   idx (entity_id % block_cap())                                    │
//! │   ┌─────0─────┬─────1─────┬─────2─────┬─────3─────┐                │
//! │   │ off:0     │ off:3     │ cap:0     │ off:5     │ ...            │
//! │   │ len:3 cap:3 len:2 cap:2 (no attrs)│ len:1 cap:4                │
//! │   └────┬──────┴────┬──────┴───────────┴────┬──────┘                │
//! │        │           │                       │                       │
//! │ arena: Vec<PackedAttr>   12 B each: (attr_id u16, tag u8, [u8;8])  │
//! │        ▼           ▼                       ▼                       │
//! │   ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬───────────────┐   │
//! │   │ e0 │ e0 │ e0 │ e1 │ e1 │ e3 │ ~~ │ ~~ │ ~~ │ ← bump alloc  │   │
//! │   └────┴────┴────┴────┴─┬──┴────┴────┴────┴────┴───────────────┘   │
//! │     sorted by attr_id   │  ~~ = slack (cap > len) / dead spans,    │
//! │     within each span    │  reclaimed by compact() when >50% dead   │
//! │                         │                                          │
//! │                         │ payload of Tag::Heap = u32 index         │
//! │ heap: Vec<Value>        ▼  out-of-line String/List/VecF32/...      │
//! │   ┌──────────┬──────────┬──────────┐                               │
//! │   │ "alice"  │ Null(hole)│ [1,2,3] │   heap_free: [1]              │
//! │   └──────────┴──────────┴──────────┘   (holes recycled)            │
//! └────────────────────────────────────────────────────────────────────┘
//!
//! Per-entity cost: 8 B slot + 12 B x attrs in the shared arena —
//! no per-entity allocation. Scalars (Int/Float/Bool/...) live entirely
//! in the 8-byte packed payload; only heap values touch `heap`.
//! ```
//!
//! ## Concurrency / MVCC
//!
//! The store contains no locks and no unsafe code. Isolation comes from the
//! MVCC design: a write transaction operates on a *private* clone of the
//! graph (`Graph::new_version`), which is published atomically on commit or
//! simply discarded on rollback. Cloning the store is cheap — blocks are
//! shared via `Arc` and copied on first write per block (`Arc::make_mut`),
//! so older snapshots keep reading their own blocks untouched. In-place
//! arena writes are safe because they only ever happen after `make_mut`.

use std::cmp::Ordering;
use std::sync::Arc;

use roaring::RoaringTreemap;
use rustc_hash::FxHashMap;

use super::graphblas::serialization::{Decode, Encode, Reader, Writer};
use crate::runtime::value::Value;

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

/// Packed value type tag. Scalars inline their payload; [`Tag::Heap`] values
/// live in the block-level side array (`Block::heap`) and the payload is a
/// `u32` index.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(u8)]
enum Tag {
    #[default]
    Null = 0,
    Bool = 1,
    Int = 2,
    Float = 3,
    Point = 4,
    Datetime = 5,
    Date = 6,
    Time = 7,
    Duration = 8,
    Heap = 9,
}

/// One packed attribute: 12 bytes, matching the C engine's per-attribute
/// footprint (`AttributeID` + packed `AttrValue_t`).
#[derive(Clone, Copy, Default)]
struct PackedAttr {
    id: u16,
    tag: Tag,
    payload: [u8; 8],
}

impl PackedAttr {
    #[inline]
    fn heap_index(&self) -> usize {
        debug_assert_eq!(self.tag, Tag::Heap);
        u32::from_le_bytes(self.payload[..4].try_into().unwrap()) as usize
    }
}

// ============================================================================
// Block / DataBlock: arena-based slot storage (C engine's DataBlock, COW)
// ============================================================================

/// Slots per block, matching the C engine's DataBlock granularity. Driven
/// by the NODE_CREATION_BUFFER configuration, which is normalized to a
/// power of two at module init and immutable afterwards.
#[inline]
fn block_cap() -> usize {
    super::graph::NODE_CREATION_BUFFER.load(std::sync::atomic::Ordering::Relaxed) as usize
}

/// Arena span descriptor for one entity. `cap == 0` means the entity has no
/// attribute span (the equivalent of C's NULL AttributeSet); live spans have
/// `1 <= len <= cap`.
#[derive(Clone, Copy, Default)]
struct Slot {
    /// Start index into `Block::arena`, in entries.
    offset: u32,
    /// Live entries.
    len: u16,
    /// Reserved entries (in-place overwrite allowed while `len <= cap`).
    cap: u16,
}

/// One block of entity slots plus the arena holding their packed attributes.
/// Slots grow lazily up to [`block_cap`].
#[derive(Clone, Default)]
struct Block {
    slots: Vec<Slot>,
    /// All attribute spans, bump-allocated; abandoned spans are tracked in
    /// `dead` and reclaimed by [`Block::compact`].
    arena: Vec<PackedAttr>,
    /// Out-of-line values (String/List/VecF32) referenced by index from
    /// packed payloads. Holes hold `Value::Null` and are recycled.
    heap: Vec<Value>,
    heap_free: Vec<u32>,
    /// Arena entries no longer referenced by any slot.
    dead: u32,
    /// Arena entries reserved but unused inside live spans (`cap - len`).
    slack: u32,
}

impl Block {
    /// Pack a value into a tag + 8-byte payload. Out-of-line values are
    /// pushed into `self.heap` (recycling `heap_free` holes) and the payload
    /// is their heap index. Packing therefore has a side effect that must be
    /// paired with an arena write — only [`Block::store_packed_value`] may
    /// call this.
    fn pack_value(
        &mut self,
        value: Value,
    ) -> (Tag, [u8; 8]) {
        match value {
            Value::Null => (Tag::Null, [0; 8]),
            Value::Bool(b) => {
                let mut p = [0; 8];
                p[0] = u8::from(b);
                (Tag::Bool, p)
            }
            Value::Int(i) => (Tag::Int, i.to_le_bytes()),
            Value::Float(f) => (Tag::Float, f.to_le_bytes()),
            Value::Point(point) => {
                let mut p = [0; 8];
                p[..4].copy_from_slice(&point.latitude.to_le_bytes());
                p[4..].copy_from_slice(&point.longitude.to_le_bytes());
                (Tag::Point, p)
            }
            Value::Datetime(t) => (Tag::Datetime, t.to_le_bytes()),
            Value::Date(t) => (Tag::Date, t.to_le_bytes()),
            Value::Time(t) => (Tag::Time, t.to_le_bytes()),
            Value::Duration(t) => (Tag::Duration, t.to_le_bytes()),
            other => {
                let idx = if let Some(i) = self.heap_free.pop() {
                    self.heap[i as usize] = other;
                    i
                } else {
                    self.heap.push(other);
                    (self.heap.len() - 1) as u32
                };
                let mut p = [0; 8];
                p[..4].copy_from_slice(&idx.to_le_bytes());
                (Tag::Heap, p)
            }
        }
    }

    /// Pack `value` and write the entry to the arena at `index` (appending
    /// when `index == self.arena.len()`) in one step, so the heap side effect
    /// of packing can never be separated from the arena write.
    fn store_packed_value(
        &mut self,
        index: usize,
        id: u16,
        value: Value,
    ) {
        let (tag, payload) = self.pack_value(value);
        let entry = PackedAttr { id, tag, payload };
        if index == self.arena.len() {
            self.arena.push(entry);
        } else {
            self.arena[index] = entry;
        }
    }

    fn unpack(
        &self,
        attr: &PackedAttr,
    ) -> Value {
        match attr.tag {
            Tag::Null => Value::Null,
            Tag::Bool => Value::Bool(attr.payload[0] != 0),
            Tag::Int => Value::Int(i64::from_le_bytes(attr.payload)),
            Tag::Float => Value::Float(f64::from_le_bytes(attr.payload)),
            Tag::Point => Value::Point(crate::runtime::value::Point {
                latitude: f32::from_le_bytes(attr.payload[..4].try_into().unwrap()),
                longitude: f32::from_le_bytes(attr.payload[4..].try_into().unwrap()),
            }),
            Tag::Datetime => Value::Datetime(i64::from_le_bytes(attr.payload)),
            Tag::Date => Value::Date(i64::from_le_bytes(attr.payload)),
            Tag::Time => Value::Time(i64::from_le_bytes(attr.payload)),
            Tag::Duration => Value::Duration(i64::from_le_bytes(attr.payload)),
            Tag::Heap => self.heap[attr.heap_index()].clone(),
        }
    }

    /// Release one entry's heap value (no-op for inline scalars). The hole
    /// keeps `Value::Null` and its index is recycled via `heap_free`.
    #[inline]
    fn release_heap_value(
        &mut self,
        entry: PackedAttr,
    ) {
        if entry.tag == Tag::Heap {
            let heap_index = entry.heap_index();
            self.heap[heap_index] = Value::Null;
            self.heap_free.push(heap_index as u32);
        }
    }

    /// Release the heap values referenced by a span's live entries.
    fn release_span_values(
        &mut self,
        slot: Slot,
    ) {
        let start = slot.offset as usize;
        for i in start..start + slot.len as usize {
            self.release_heap_value(self.arena[i]);
        }
    }

    /// Mark a span abandoned: its whole `cap` becomes dead arena space and
    /// stops counting as slack.
    #[inline]
    fn retire_span(
        &mut self,
        slot: Slot,
    ) {
        self.dead += u32::from(slot.cap);
        self.slack -= u32::from(slot.cap - slot.len);
    }

    /// Adjust `slack` after resizing a span in place within its `cap`.
    /// `slack` already contains `cap - old_len` for this span, so
    /// `slack + old_len >= cap >= new_len` and the subtraction cannot
    /// underflow.
    #[inline]
    fn resize_span_slack(
        &mut self,
        old_len: u16,
        new_len: usize,
    ) {
        self.slack = self.slack + u32::from(old_len) - new_len as u32;
    }

    /// Grow `slots` to cover `slot_idx` with capacity capped at
    /// [`block_cap`]. Plain `Vec::resize` doubling would overshoot the
    /// block's fixed maximum, and since COW clones reset capacity to `len`,
    /// a clone-then-grow cycle otherwise leaves up to 2x slack per block.
    fn grow_slots(
        &mut self,
        slot_idx: usize,
    ) {
        if self.slots.len() <= slot_idx {
            let target = (self.slots.len() * 2).clamp(slot_idx + 1, block_cap().max(slot_idx + 1));
            self.slots.reserve_exact(target - self.slots.len());
            self.slots.resize(slot_idx + 1, Slot::default());
        }
    }

    /// Write an entity's full attribute set, replacing any previous span.
    /// `pairs` must be sorted by attribute id; it is drained.
    fn set_span(
        &mut self,
        slot_idx: usize,
        pairs: &mut Vec<(u16, Value)>,
    ) {
        self.grow_slots(slot_idx);
        // Setting an empty attribute set clears the entity's span entirely
        // (the entity keeps no attributes).
        if pairs.is_empty() {
            self.free_span(slot_idx);
            return;
        }
        let old = self.slots[slot_idx];
        if old.cap != 0 {
            self.release_span_values(old);
        }

        let n = pairs.len();
        if n <= old.cap as usize {
            for (k, (id, value)) in pairs.drain(..).enumerate() {
                self.store_packed_value(old.offset as usize + k, id, value);
            }
            self.resize_span_slack(old.len, n);
            self.slots[slot_idx] = Slot {
                offset: old.offset,
                len: n as u16,
                cap: old.cap,
            };
        } else {
            self.retire_span(old);
            let offset = self.arena.len() as u32;
            for (id, value) in pairs.drain(..) {
                self.store_packed_value(self.arena.len(), id, value);
            }
            self.slots[slot_idx] = Slot {
                offset,
                len: n as u16,
                cap: n as u16,
            };
        }
    }

    /// Free an entity's span (entity keeps no attributes).
    fn free_span(
        &mut self,
        slot_idx: usize,
    ) {
        let slot = self.slots[slot_idx];
        if slot.cap == 0 {
            return;
        }
        self.release_span_values(slot);
        self.retire_span(slot);
        self.slots[slot_idx] = Slot::default();
    }

    /// Merge `pairs` (sorted by attribute id; `Null` value = removal) into
    /// the entity's span at the packed level: untouched attributes are
    /// copied as raw entries — no unpack/re-pack, and their heap values keep
    /// their index. Returns `(nremoved, nset)` with [`AttributeStore::insert_attrs`]
    /// semantics; the counts feed the query's `properties set` / `properties
    /// removed` statistics. `scratch` is a reusable snapshot buffer of the
    /// old span.
    fn merge_span(
        &mut self,
        slot_idx: usize,
        pairs: &[(u16, Value)],
        scratch: &mut Vec<PackedAttr>,
    ) -> (usize, usize) {
        self.grow_slots(slot_idx);
        let old = self.slots[slot_idx];

        // Fast path: every pair replaces an existing attribute (non-null,
        // id already present). Patch entries in place — the dominant shape
        // for `SET n.x = ...` on existing attributes — without snapshotting
        // or rewriting the untouched rest of the span.
        if old.cap != 0 {
            let s = old.offset as usize;
            let span = &self.arena[s..s + old.len as usize];
            if pairs.iter().all(|(id, v)| {
                !matches!(v, Value::Null) && span.binary_search_by_key(id, |e| e.id).is_ok()
            }) {
                for (id, v) in pairs {
                    let span = &self.arena[s..s + old.len as usize];
                    let pos = span.binary_search_by_key(id, |e| e.id).unwrap();
                    self.release_heap_value(self.arena[s + pos]);
                    self.store_packed_value(s + pos, *id, v.clone());
                }
                return (pairs.len(), pairs.len());
            }

            // Fast path: pure removal (every pair null). Shift surviving
            // entries left within the span, releasing removed heap values.
            if pairs.iter().all(|(_, v)| matches!(v, Value::Null)) {
                let mut w = s;
                let mut ni = 0;
                let mut nremoved = 0;
                for r in s..s + old.len as usize {
                    let e = self.arena[r];
                    while ni < pairs.len() && pairs[ni].0 < e.id {
                        ni += 1;
                    }
                    if ni < pairs.len() && pairs[ni].0 == e.id {
                        nremoved += 1;
                        self.release_heap_value(e);
                        continue;
                    }
                    self.arena[w] = e;
                    w += 1;
                }
                let new_len = w - s;
                if new_len == 0 {
                    self.retire_span(old);
                    self.slots[slot_idx] = Slot::default();
                } else {
                    self.resize_span_slack(old.len, new_len);
                    self.slots[slot_idx].len = new_len as u16;
                }
                return (nremoved, 0);
            }
        }

        scratch.clear();
        if old.cap != 0 {
            let s = old.offset as usize;
            scratch.extend_from_slice(&self.arena[s..s + old.len as usize]);
        }

        // First pass: merged length, to pick in-place vs relocation.
        let mut new_len = 0usize;
        {
            let (mut ci, mut ni) = (0usize, 0usize);
            while ci < scratch.len() && ni < pairs.len() {
                match scratch[ci].id.cmp(&pairs[ni].0) {
                    Ordering::Less => {
                        new_len += 1;
                        ci += 1;
                    }
                    Ordering::Equal => {
                        new_len += usize::from(!matches!(pairs[ni].1, Value::Null));
                        ci += 1;
                        ni += 1;
                    }
                    Ordering::Greater => {
                        new_len += usize::from(!matches!(pairs[ni].1, Value::Null));
                        ni += 1;
                    }
                }
            }
            new_len += scratch.len() - ci;
            new_len += pairs[ni..]
                .iter()
                .filter(|(_, v)| !matches!(v, Value::Null))
                .count();
        }

        let in_place = new_len != 0 && new_len <= old.cap as usize;
        let dst = if new_len == 0 {
            0 // no entry is ever written
        } else if in_place {
            old.offset as usize
        } else {
            if old.cap != 0 {
                self.retire_span(old);
            }
            let d = self.arena.len();
            self.arena.resize(d + new_len, PackedAttr::default());
            d
        };

        // Second pass: emit merged entries, releasing replaced/removed heap
        // values. Untouched heap entries move with their raw copy.
        let mut nremoved = 0usize;
        let mut nset = 0usize;
        let mut w = dst;
        let (mut ci, mut ni) = (0usize, 0usize);
        while ci < scratch.len() || ni < pairs.len() {
            let take_old =
                ni >= pairs.len() || (ci < scratch.len() && scratch[ci].id < pairs[ni].0);
            if take_old {
                self.arena[w] = scratch[ci];
                w += 1;
                ci += 1;
                continue;
            }
            if ci < scratch.len() && scratch[ci].id == pairs[ni].0 {
                nremoved += 1;
                self.release_heap_value(scratch[ci]);
                ci += 1;
            }
            let (id, v) = &pairs[ni];
            if !matches!(v, Value::Null) {
                nset += 1;
                self.store_packed_value(w, *id, v.clone());
                w += 1;
            }
            ni += 1;
        }
        debug_assert!(new_len == 0 || w - dst == new_len);

        if new_len == 0 {
            if old.cap != 0 {
                self.retire_span(old);
            }
            self.slots[slot_idx] = Slot::default();
        } else if in_place {
            self.resize_span_slack(old.len, new_len);
            self.slots[slot_idx] = Slot {
                offset: old.offset,
                len: new_len as u16,
                cap: old.cap,
            };
        } else {
            self.slots[slot_idx] = Slot {
                offset: dst as u32,
                len: new_len as u16,
                cap: new_len as u16,
            };
        }
        (nremoved, nset)
    }

    /// Rebuild the arena and heap from live spans, dropping dead spans,
    /// `cap > len` slack, and heap holes (heap indices are remapped).
    fn compact(&mut self) {
        let live = self.arena.len() - self.dead as usize - self.slack as usize;
        let mut new_arena = Vec::with_capacity(live);
        let mut new_heap = Vec::with_capacity(self.heap.len() - self.heap_free.len());
        for slot in &mut self.slots {
            if slot.cap == 0 {
                continue;
            }
            let start = slot.offset as usize;
            slot.offset = new_arena.len() as u32;
            for entry in &self.arena[start..start + slot.len as usize] {
                let mut entry = *entry;
                if entry.tag == Tag::Heap {
                    let value = std::mem::replace(&mut self.heap[entry.heap_index()], Value::Null);
                    entry.payload[..4].copy_from_slice(&(new_heap.len() as u32).to_le_bytes());
                    new_heap.push(value);
                }
                new_arena.push(entry);
            }
            slot.cap = slot.len;
        }
        debug_assert_eq!(new_arena.len(), live);
        self.arena = new_arena;
        self.heap = new_heap;
        self.heap_free = Vec::new();
        self.dead = 0;
        self.slack = 0;
    }

    /// Compact when abandoned entries dominate the arena.
    fn maybe_compact(&mut self) {
        let waste = self.dead as usize + self.slack as usize;
        if waste * 2 > self.arena.len() && self.arena.len() > 1024 {
            self.compact();
        }
    }
}

/// Borrowed view of one entity's attribute span.
#[derive(Clone, Copy)]
struct SpanRef<'a> {
    block: &'a Block,
    slot: Slot,
}

impl<'a> SpanRef<'a> {
    #[inline]
    fn entries(self) -> &'a [PackedAttr] {
        let start = self.slot.offset as usize;
        &self.block.arena[start..start + self.slot.len as usize]
    }

    #[inline]
    fn len(self) -> usize {
        self.slot.len as usize
    }

    #[inline]
    fn get(
        self,
        attr_idx: u16,
    ) -> Option<Value> {
        let entries = self.entries();
        entries
            .binary_search_by_key(&attr_idx, |attr| attr.id)
            .ok()
            .map(|pos| self.block.unpack(&entries[pos]))
    }

    fn iter(self) -> impl Iterator<Item = (u16, Value)> + 'a {
        let block = self.block;
        self.entries()
            .iter()
            .map(move |attr| (attr.id, block.unpack(attr)))
    }

    /// Estimated heap bytes attributable to this entity: its arena span plus
    /// its share of the block-level heap array.
    fn heap_bytes(self) -> usize {
        let mut bytes = self.len() * std::mem::size_of::<PackedAttr>();
        for attr in self.entries() {
            if attr.tag == Tag::Heap {
                let value = &self.block.heap[attr.heap_index()];
                bytes += std::mem::size_of::<Value>() + value.heap_size();
            }
        }
        bytes
    }
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
    fn locate(entity_id: u64) -> (usize, usize) {
        let cap = block_cap();
        debug_assert!(cap.is_power_of_two());
        let id = entity_id as usize;
        (id >> cap.trailing_zeros(), id & (cap - 1))
    }

    #[inline]
    fn get(
        &self,
        entity_id: u64,
    ) -> Option<SpanRef<'_>> {
        let (block_idx, slot_idx) = Self::locate(entity_id);
        let block = self.blocks.get(block_idx)?;
        let slot = *block.slots.get(slot_idx)?;
        if slot.cap == 0 {
            return None;
        }
        Some(SpanRef { block, slot })
    }

    /// Replace an entity's attribute set with `pairs` (sorted by attribute
    /// id, drained). An empty `pairs` frees the span.
    fn set_span(
        &mut self,
        entity_id: u64,
        pairs: &mut Vec<(u16, Value)>,
    ) {
        if pairs.is_empty() {
            self.remove(entity_id);
            return;
        }
        let (block_idx, slot_idx) = Self::locate(entity_id);
        if self.blocks.len() <= block_idx {
            self.blocks.resize_with(block_idx + 1, Arc::default);
        }
        let block = Arc::make_mut(&mut self.blocks[block_idx]);
        block.set_span(slot_idx, pairs);
        block.maybe_compact();
    }

    /// Merge `pairs` (sorted by attribute id; `Null` = removal) into an
    /// entity's span. Returns `(nremoved, nset)`.
    fn merge_span(
        &mut self,
        entity_id: u64,
        pairs: &[(u16, Value)],
        scratch: &mut Vec<PackedAttr>,
    ) -> (usize, usize) {
        let (block_idx, slot_idx) = Self::locate(entity_id);
        // Avoid COW (and empty-slot creation) when there is nothing to do:
        // no existing span and only removals.
        let has_span = self
            .blocks
            .get(block_idx)
            .is_some_and(|block| block.slots.get(slot_idx).is_some_and(|slot| slot.cap != 0));
        if !has_span && pairs.iter().all(|(_, v)| matches!(v, Value::Null)) {
            return (0, 0);
        }
        if self.blocks.len() <= block_idx {
            self.blocks.resize_with(block_idx + 1, Arc::default);
        }
        let block = Arc::make_mut(&mut self.blocks[block_idx]);
        let counts = block.merge_span(slot_idx, pairs, scratch);
        block.maybe_compact();
        counts
    }

    /// Trim arena growth slop on blocks this version owns exclusively
    /// (i.e. the blocks touched since the last snapshot). Called at commit
    /// time: mid-fill shrinking would break `Vec`'s doubling sequence and
    /// leave up to 2x over-allocation, so slop is reclaimed here instead.
    ///
    /// Only fully-populated blocks are shrunk: reallocating a still-growing
    /// tail block on every commit churns the allocator (RSS fragmentation)
    /// for slop that the next batch reclaims anyway. Residual slop is
    /// bounded by one partially-filled tail block per store.
    fn trim(&mut self) {
        let cap = block_cap();
        for arc in &mut self.blocks {
            if let Some(block) = Arc::get_mut(arc) {
                // Commit-time compaction: reclaim update churn (dead spans +
                // in-span slack) once it reaches half the arena. Steady-state
                // waste stays bounded at 2x live payload — still below the C
                // engine's realloc churn — without recopying every block on
                // each commit.
                let waste = block.dead as usize + block.slack as usize;
                if waste * 2 >= block.arena.len() && block.arena.len() > 1024 {
                    block.compact();
                }
                if block.slots.len() == cap {
                    block.arena.shrink_to_fit();
                    block.heap.shrink_to_fit();
                    // Slots fill in hash order, so `resize` can double past
                    // the block capacity from a non-power-of-two start.
                    block.slots.shrink_to_fit();
                }
                block.heap_free.shrink_to_fit();
            }
        }
    }

    fn remove(
        &mut self,
        entity_id: u64,
    ) {
        let (block_idx, slot_idx) = Self::locate(entity_id);
        // Check occupancy before make_mut so clearing an already-empty slot
        // doesn't deep-copy a shared block.
        if let Some(block) = self.blocks.get(block_idx)
            && block.slots.get(slot_idx).is_some_and(|slot| slot.cap != 0)
        {
            let block = Arc::make_mut(&mut self.blocks[block_idx]);
            block.free_span(slot_idx);
            block.maybe_compact();
        }
    }

    /// Structural overhead of the slot storage, excluding attribute payload.
    fn memory_usage(&self) -> usize {
        self.blocks.len() * (std::mem::size_of::<Arc<Block>>() + std::mem::size_of::<Block>())
            + self
                .blocks
                .iter()
                .map(|block| block.slots.capacity() * std::mem::size_of::<Slot>())
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
#[derive(Clone, Default)]
pub struct AttributeStore {
    /// Attribute names in insertion order (name → column index)
    pub attrs_name: AttrNameMap,
    /// Block-allocated per-entity attribute spans (COW across MVCC versions).
    data: DataBlock,
}

impl AttributeStore {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Cheap snapshot clone for a new MVCC version (one `Arc` bump per block).
    #[must_use]
    pub fn new_version(&self) -> Self {
        self.clone()
    }

    /// Reclaim arena growth slop on exclusively-owned blocks. Call at commit.
    pub fn trim(&mut self) {
        self.data.trim();
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
        self.data.get(key)?.get(attr_idx)
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
            let value = self.data.get(key).and_then(|span| span.get(attr_idx));
            out.push(value.unwrap_or_else(|| default.clone()));
        }
    }

    #[must_use]
    pub fn has_attributes(
        &self,
        key: u64,
    ) -> bool {
        self.data.get(key).is_some()
    }

    /// Number of attributes stored for an entity (0 if none).
    #[must_use]
    pub fn attr_count(
        &self,
        key: u64,
    ) -> usize {
        self.data.get(key).map_or(0, SpanRef::len)
    }

    /// Estimated heap bytes of one entity's attribute set (0 if none).
    #[must_use]
    pub fn entity_memory_usage(
        &self,
        key: u64,
    ) -> usize {
        self.data.get(key).map_or(0, SpanRef::heap_bytes)
    }

    pub fn get_attrs(
        &self,
        key: u64,
    ) -> impl Iterator<Item = Arc<String>> + '_ {
        self.data.get(key).into_iter().flat_map(move |span| {
            span.entries()
                .iter()
                .filter_map(move |attr| self.attrs_name.get(attr.id as usize).cloned())
        })
    }

    pub fn get_all_attrs(
        &self,
        key: u64,
    ) -> impl Iterator<Item = (Arc<String>, Value)> + '_ {
        self.data.get(key).into_iter().flat_map(move |span| {
            span.iter().filter_map(|(idx, value)| {
                self.attrs_name
                    .get(idx as usize)
                    .map(|name| (name.clone(), value))
            })
        })
    }

    pub fn get_all_attrs_by_id(
        &self,
        key: u64,
    ) -> impl Iterator<Item = (u16, Value)> + '_ {
        self.data.get(key).into_iter().flat_map(SpanRef::iter)
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
        attrs: &FxHashMap<u64, Vec<(u16, Value)>>,
    ) -> Result<(usize, usize), String> {
        let mut nremoved = 0;
        let mut nset = 0;

        // Reusable span snapshot buffer to avoid per-entity allocation.
        let mut scratch: Vec<PackedAttr> = Vec::new();

        // Apply in entity-id order: spans are laid out in id order, so this
        // turns the hash map's random arena access into a sequential sweep.
        // Empty pending maps are skipped: no entries to write, no nulls to
        // remove (matches C's NULL AttributeSet behaviour).
        let mut items: Vec<(u64, &Vec<(u16, Value)>)> = attrs
            .iter()
            .filter(|(_, entity_attrs)| !entity_attrs.is_empty())
            .map(|(key, entity_attrs)| (*key, entity_attrs))
            .collect();
        items.sort_unstable_by_key(|&(key, _)| key);

        for (key, entity_attrs) in items {
            let (r, s) = self.data.merge_span(key, entity_attrs, &mut scratch);
            nremoved += r;
            nset += s;
        }

        Ok((nremoved, nset))
    }

    /// Bulk import attributes for entities known to be new (no prior state).
    ///
    /// Input values are attribute-id-resolved and sorted by id; null values
    /// are skipped. Returns the number of non-null attributes imported.
    pub fn import_attrs(
        &mut self,
        attrs: &FxHashMap<u64, Vec<(u16, Value)>>,
    ) -> usize {
        let mut nset = 0;
        // Reusable scratch, drained into the arena per entity.
        let mut scratch: Vec<(u16, Value)> = Vec::new();
        for (key, entity_attrs) in attrs {
            scratch.clear();
            scratch.extend(
                entity_attrs
                    .iter()
                    .filter(|(_, value)| !matches!(value, Value::Null))
                    .cloned(),
            );
            // Skip empty entities to avoid per-entity slot overhead (matches
            // C's NULL AttributeSet behaviour for prop-less entities).
            if scratch.is_empty() {
                continue;
            }
            nset += scratch.len();
            self.data.set_span(*key, &mut scratch);
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
            self.data.set_span(entity_id, &mut entries);
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

    /// Structural slot-storage overhead, excluding attribute payload heap.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.data.memory_usage()
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

            let span = self.data.get(id);
            w.write_unsigned(span.map_or(0, |s| s.len() as u64));

            if let Some(span) = span {
                for (local_attr_id, value) in span.iter() {
                    let global_attr_id = if (local_attr_id as usize) < remap.len() {
                        remap[local_attr_id as usize]
                    } else {
                        local_attr_id
                    };
                    w.write_unsigned(u64::from(global_attr_id));
                    value.encode(w);
                }
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
                self.data.set_span(entity_id, &mut entries);
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
        let mut store = AttributeStore::new();
        let mut attrs: FxHashMap<u64, Vec<(u16, Value)>> = FxHashMap::default();
        for (id, pairs) in entries {
            let mut vec: Vec<(u16, Value)> = pairs
                .iter()
                .map(|(attr, value)| (store.get_or_create_attr_id(&name(attr)), value.clone()))
                .collect();
            vec.sort_unstable_by_key(|(k, _)| *k);
            attrs.insert(*id, vec);
        }
        store.insert_attrs(&attrs).unwrap();
        store
    }

    #[test]
    fn layout_matches_c() {
        assert_eq!(std::mem::size_of::<Slot>(), 8);
        assert_eq!(std::mem::size_of::<PackedAttr>(), 12);
    }

    #[test]
    fn packed_round_trip_all_storable_variants() {
        use crate::runtime::value::Point;
        let values = vec![
            Value::Bool(true),
            Value::Bool(false),
            Value::Int(i64::MIN),
            Value::Int(-1),
            Value::Int(i64::MAX),
            Value::Float(-0.0),
            Value::Float(f64::NAN),
            Value::Float(f64::MAX),
            Value::Point(Point {
                latitude: 32.07,
                longitude: 34.78,
            }),
            Value::Datetime(-62_135_596_800),
            Value::Date(19_723),
            Value::Time(-1),
            Value::Duration(i64::MAX),
            Value::String(Arc::new("hello".to_string())),
            Value::List(Arc::new(
                [Value::Int(1), Value::String(Arc::new("x".to_string()))]
                    .into_iter()
                    .collect(),
            )),
            Value::VecF32(Arc::new([1.0f32, -2.5].into_iter().collect())),
        ];
        let mut store = AttributeStore::new();
        let pairs: Vec<(u16, Value)> = values
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, v)| {
                store.get_or_create_attr_id(&name(&format!("a{i:02}")));
                (i as u16, v)
            })
            .collect();
        let mut attrs = FxHashMap::default();
        attrs.insert(7u64, pairs.clone());
        store.insert_attrs(&attrs).unwrap();

        for (i, expected) in values.iter().enumerate() {
            let got = store.get_attr_by_idx(7, i as u16).unwrap();
            match (expected, &got) {
                // NaN != NaN under PartialEq; compare bit patterns.
                (Value::Float(a), Value::Float(b)) => assert_eq!(a.to_bits(), b.to_bits()),
                _ => assert_eq!(*expected, got),
            }
        }
        assert_eq!(store.get_attr_by_idx(7, values.len() as u16), None);

        let round_tripped: Vec<_> = store.get_all_attrs_by_id(7).collect();
        assert_eq!(round_tripped.len(), pairs.len());
        for ((ei, ev), (gi, gv)) in pairs.iter().zip(round_tripped.iter()) {
            assert_eq!(ei, gi);
            if let (Value::Float(a), Value::Float(b)) = (ev, gv) {
                assert_eq!(a.to_bits(), b.to_bits());
            } else {
                assert_eq!(ev, gv);
            }
        }
    }

    #[test]
    fn scalar_entity_memory_matches_c_layout() {
        // All-scalar entity: exactly 12 bytes per attribute, nothing else.
        let pairs: Vec<(&str, Value)> = vec![
            ("a", Value::Int(1)),
            ("b", Value::Int(2)),
            ("c", Value::Int(3)),
            ("d", Value::Int(4)),
        ];
        let store = store_with(&[(1, &pairs)]);
        assert_eq!(store.entity_memory_usage(1), 4 * 12);
        assert_eq!(store.entity_memory_usage(2), 0);
        assert!(store.data.blocks[0].heap.is_empty());
    }

    #[test]
    fn mixed_scalar_and_heap_interleaved() {
        let s1 = Value::String(Arc::new("first".to_string()));
        let s2 = Value::String(Arc::new("second".to_string()));
        let store = store_with(&[(
            1,
            &[
                ("a", Value::Int(7)),
                ("b", s1.clone()),
                ("c", Value::Float(1.5)),
                ("d", s2.clone()),
            ],
        )]);
        assert_eq!(store.get_attr(1, &name("a")), Some(Value::Int(7)));
        assert_eq!(store.get_attr(1, &name("b")), Some(s1));
        assert_eq!(store.get_attr(1, &name("c")), Some(Value::Float(1.5)));
        assert_eq!(store.get_attr(1, &name("d")), Some(s2));
        assert_eq!(store.data.blocks[0].heap.len(), 2);
    }

    #[test]
    fn heap_free_list_reuse_on_overwrite() {
        let mut store = store_with(&[(1, &[("s", Value::String(Arc::new("old".to_string())))])]);
        let s = store.get_or_create_attr_id(&name("s"));
        let mut attrs = FxHashMap::default();
        attrs.insert(1u64, vec![(s, Value::String(Arc::new("new".to_string())))]);
        store.insert_attrs(&attrs).unwrap();

        assert_eq!(
            store.get_attr(1, &name("s")),
            Some(Value::String(Arc::new("new".to_string())))
        );
        // The old string's heap slot must be recycled, not leaked.
        let block = &store.data.blocks[0];
        assert_eq!(block.heap.len(), 1);
        assert!(block.heap_free.is_empty());
    }

    #[test]
    fn compaction_reclaims_abandoned_spans() {
        let mut store = AttributeStore::new();
        for i in 0..200u16 {
            store.get_or_create_attr_id(&name(&format!("a{i:03}")));
        }
        // Grow one entity's attr set one attribute at a time: every write
        // relocates the span, abandoning the previous one.
        for k in 1..=100u16 {
            let pairs: Vec<(u16, Value)> = (0..k).map(|i| (i, Value::Int(i64::from(i)))).collect();
            let mut attrs = FxHashMap::default();
            attrs.insert(1u64, pairs);
            store.insert_attrs(&attrs).unwrap();
        }
        let block = &store.data.blocks[0];
        // Compaction must have run: dead entries bounded by live ones.
        assert!(block.dead as usize * 2 <= block.arena.len() || block.arena.len() <= 1024);
        // Values intact after compaction.
        for i in 0..100u16 {
            assert_eq!(
                store.get_attr_by_idx(1, i),
                Some(Value::Int(i64::from(i))),
                "attr {i}"
            );
        }
        assert_eq!(store.get_all_attrs_by_id(1).count(), 100);
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
        let a = store.get_or_create_attr_id(&name("a"));
        let b = store.get_or_create_attr_id(&name("b"));
        let c = store.get_or_create_attr_id(&name("c"));
        let mut vec = vec![(a, Value::Int(10)), (b, Value::Null), (c, Value::Int(3))];
        vec.sort_unstable_by_key(|(k, _)| *k);
        let mut attrs = FxHashMap::default();
        attrs.insert(1u64, vec);
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
        let mut v2 = v1.new_version();

        // Mutate v2: overwrite entity 1 (in-place candidate!), add entity 2,
        // then delete entity 1's attrs. v1 must never observe any of it.
        let a = v2.get_or_create_attr_id(&name("a"));
        let vec = vec![(a, Value::Int(99))];
        let mut attrs = FxHashMap::default();
        attrs.insert(1u64, vec.clone());
        attrs.insert(2u64, vec);
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
    fn snapshot_isolation_heap_values() {
        let v1 = store_with(&[(1, &[("s", Value::String(Arc::new("v1".to_string())))])]);
        let mut v2 = v1.new_version();

        let s = v2.get_or_create_attr_id(&name("s"));
        let mut attrs = FxHashMap::default();
        attrs.insert(1u64, vec![(s, Value::String(Arc::new("v2".to_string())))]);
        v2.insert_attrs(&attrs).unwrap();

        assert_eq!(
            v1.get_attr(1, &name("s")),
            Some(Value::String(Arc::new("v1".to_string())))
        );
        assert_eq!(
            v2.get_attr(1, &name("s")),
            Some(Value::String(Arc::new("v2".to_string())))
        );
    }

    #[test]
    fn remove_all_is_immediate() {
        let mut store = store_with(&[(5, &[("a", Value::Int(1))])]);
        let mut keys = RoaringTreemap::new();
        keys.insert(5);
        store.remove_all(&keys);
        assert_eq!(store.get_attr(5, &name("a")), None);
        assert!(!store.has_attributes(5));
        assert!(store.get_all_attrs(5).next().is_none());
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
        keys.insert(block_cap() as u64 * 10);
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
        let far_id = block_cap() as u64 * 3 + 7;
        let store = store_with(&[(far_id, &[("a", Value::Int(42))])]);
        assert_eq!(store.get_attr(far_id, &name("a")), Some(Value::Int(42)));
        assert_eq!(store.get_attr(far_id - 1, &name("a")), None);
    }
}
