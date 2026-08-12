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
//! Blocks are reached through a **two-level radix directory** rather than one
//! flat vector, so a write copies only the root→page→block path it touches
//! (see *Copy-on-write* below) and the block can stay small enough to double as
//! a future RAM↔disk residency unit.
//!
//! ## Addressing
//!
//! The dense entity id *is* the address: it splits into radix digits that index
//! each level directly — no key comparisons anywhere in the descent. Shown for
//! the constants (`BLOCK_CAP` slots/block, `DIR_FANOUT`
//! blocks/page ⇒ one directory page spans 4096 ids):
//!
//! ```text
//!  entity_id   bits 63 ............ 12 │ 11 ...... 6 │ 5 ...... 0
//!                   └──── dir_idx ────┘ └ page_slot ┘ └ slot_idx ┘
//!                    id / (cap*FANOUT)   (id/cap)%64     id % cap
//! ```
//!
//! ## Memory layout
//!
//! ```text
//! DataBlock
//! ┌────────────────────────────────────────────────────────────────────┐
//! │ root: Arc<Vec<Option<Arc<DirPage>>>>   8 B/slot, grows with max id │
//! │   [ DirPage 0 ][ None ][ DirPage 2 ] ...                           │
//! └───────────┬────────────────────────────────────────────────────────┘
//!             │ dir_idx           (shared across MVCC snapshots)
//!             ▼
//! DirPage — 512 B, the directory's COW unit
//! ┌────────────────────────────────────────────────────────────────────┐
//! │ blocks: [Option<Arc<Block>>; DIR_FANOUT]                           │
//! │   [ Arc<Block> ][ Arc<Block> ][ None ] ...                         │
//! └───────────┬────────────────────────────────────────────────────────┘
//!             │ page_slot         (untouched pages stay Arc-shared)
//!             ▼
//! Block — `BLOCK_CAP` entities (~4 KB), the data COW unit
//! ┌────────────────────────────────────────────────────────────────────┐
//! │ slots: Vec<Slot>          one 8-byte span descriptor per entity    │
//! │   idx = slot_idx                                                   │
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
//! A read is two dependent loads (root → page → block), then a binary
//! search over the entity's tiny sorted span.
//! ```
//!
//! ## Copy-on-write: what one write copies
//!
//! ```text
//!  ✎ SET one attribute of one entity
//!
//!    root ──────────── path-copied      root_len x 8 B
//!      └─ DirPage[i] ── path-copied      512 B
//!           └─ Block ── path-copied      slots + arena (~4 KB default)
//!
//!    every OTHER DirPage and Block stays SHARED with the reader's
//!    snapshot — one Arc refcount bump each, no data copied
//! ```
//!
//! A flat `Vec<Arc<Block>>` directory would instead clone **every** block
//! pointer per write, which is why the block could not be made small: the
//! pointer vector grows as the block shrinks. The radix split decouples the
//! two, so blocks stay ~4 KB *and* the copied directory stays tiny.
//!
//! ## Concurrency / MVCC
//!
//! The store contains no locks and no unsafe code. Isolation comes from the
//! MVCC design: a write transaction operates on a *private* clone of the
//! graph (`Graph::new_version`), which is published atomically on commit or
//! simply discarded on rollback. Cloning the store is cheap — it bumps the
//! root `Arc`, sharing every directory page and block. The first write to a
//! block then `Arc::make_mut`s just its root→page→block path, so older
//! snapshots keep reading their own pages untouched. In-place arena writes are
//! safe because they only ever happen after `make_mut`.
//!
//! Note that this is why the directory is *not* built from interior-mutable
//! cells (e.g. `arc_swap::ArcSwapOption`): `Arc::make_mut` is what structurally
//! guarantees a writer cannot disturb a snapshot a reader still holds. Such a
//! cell is the right primitive for a future RAM↔disk *residency* swizzle
//! (`Resident ⇄ Cold` is cache state over identical bytes, and must be mutable
//! from outside the versioned tree) — not for versioned state.

use std::cmp::Ordering;
use std::sync::Arc;

use roaring::RoaringTreemap;
use rustc_hash::FxHashMap;

use super::graphblas::serialization::{Decode, Encode, Reader, Writer};
use crate::runtime::value::Value;

/// Highest `u16`, reserved as "no attribute" — C's `ATTRIBUTE_ID_NONE`
/// (`USHRT_MAX`, `src/graph/entities/attribute_set.h`). Never a valid id, so an id
/// this large in an RDB is malformed rather than merely out of our range.
pub const ATTRIBUTE_ID_NONE: u16 = u16::MAX;

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

impl<'a> IntoIterator for &'a AttrNameMap {
    type Item = &'a Arc<String>;
    type IntoIter = std::slice::Iter<'a, Arc<String>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
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

    /// Resolve `name` to its id, interning it if this graph has not seen it.
    ///
    /// The single place ids are minted. There is exactly one of these tables per
    /// graph — shared by the node and relationship stores — so an id means the same
    /// attribute wherever it appears: in a span, in the RDB, in a `GRAPH.EFFECT`, or
    /// in a compact reply. Per-store tables would number the same name differently
    /// for nodes and relationships, and a bare id on the wire would then land on the
    /// wrong attribute on a replica (#2457).
    ///
    /// Matches C, whose `GraphContext_FindOrAddAttribute` likewise takes no entity
    /// type and mints from one array.
    pub fn get_or_create(
        &mut self,
        name: &Arc<String>,
    ) -> u16 {
        if let Some(&idx) = self.index.get(name) {
            return idx;
        }
        let idx = self.vec.len() as u16;
        self.vec.push(Arc::clone(name));
        self.index.insert(Arc::clone(name), idx);
        idx
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

/// Block capacity (slots per block) — ~4 KB blocks at a few attributes/entity,
/// matching the native index's leaf-page size. The grain is the copy-on-write
/// (and future residency) unit: a smaller grain bounds the bytes a single write
/// copies via `Arc::make_mut`, while the radix directory keeps the *directory*
/// copy small at the same time (see the module docs). Deliberately decoupled
/// from `NODE_CREATION_BUFFER`, which sizes the C-style allocation block.
const BLOCK_CAP: usize = 64;

/// `entity_id` → block / slot split, folded at compile time.
const BLOCK_SHIFT: u32 = BLOCK_CAP.trailing_zeros();
const BLOCK_MASK: usize = BLOCK_CAP - 1;

const _: () = assert!(
    BLOCK_CAP.is_power_of_two(),
    "BLOCK_CAP must be a power of two: the id split is a shift + mask"
);

/// Arena floor below which a block is never compacted: the relative rule
/// (`waste * 2 > len`) already bounds steady-state waste at ~2x the live
/// payload, and this only stops us re-compacting a trivially small arena.
///
/// It must stay *below* a block's live arena (`BLOCK_CAP` x attrs/entity) or it
/// becomes the binding constraint and waste is bounded by
/// `floor / live` instead of 2x — at the 64-slot grain a 1024-entry floor let a
/// 2-attribute block bloat 5x (measured) before compaction could fire.
const COMPACT_MIN_ARENA: usize = 256;

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
/// Slots grow lazily up to [`BLOCK_CAP`].
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

    /// Grow `slots` to cover `slot_idx`, capped at the block's capacity `cap`.
    /// Plain `Vec::resize` doubling would overshoot the block's fixed maximum,
    /// and since COW clones reset capacity to `len`, a clone-then-grow cycle
    /// otherwise leaves up to 2x slack per block.
    fn grow_slots(
        &mut self,
        slot_idx: usize,
    ) {
        if self.slots.len() <= slot_idx {
            let target = (self.slots.len() * 2).clamp(slot_idx + 1, BLOCK_CAP.max(slot_idx + 1));
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
        if waste * 2 > self.arena.len() && self.arena.len() > COMPACT_MIN_ARENA {
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
    const fn len(self) -> usize {
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

    /// Bytes attributable to this entity: its live arena entries, plus the
    /// block-level heap slot and amortized payload of each out-of-line value it
    /// references.
    ///
    /// This is the *live* half of the store's memory. Everything else the store
    /// has allocated - slot tables, reserved-but-unused arena and heap capacity,
    /// abandoned spans - is reported by
    /// [`AttributeStore::structural_memory_usage`], so the two together account
    /// for the whole store exactly once.
    fn heap_bytes(self) -> usize {
        let mut bytes = self.len() * std::mem::size_of::<PackedAttr>();
        for attr in self.entries() {
            if attr.tag == Tag::Heap {
                let value = &self.block.heap[attr.heap_index()];
                bytes += std::mem::size_of::<Value>() + value.amortized_heap_size();
            }
        }
        bytes
    }
}

/// Block-allocated attribute storage indexed by entity id.
///
/// Blocks are shared across MVCC versions via `Arc` and copied on first
/// write per version (`Arc::make_mut`), so snapshots never observe writes.
/// Blocks per directory page — the middle radix level's fanout (power of two).
/// A single-entity write path-copies the root vector plus one `DirPage` (this
/// many block pointers), not the whole flat directory — turning the per-write
/// directory copy from `O(num_blocks)` into `O(root_len + DIR_FANOUT)`, so
/// untouched dir pages stay shared across MVCC versions.
const DIR_FANOUT: usize = 64;

/// One middle radix level: `DIR_FANOUT` block pointers. The copy-on-write unit
/// for the directory's leaf level.
#[derive(Clone)]
struct DirPage {
    blocks: [Option<Arc<Block>>; DIR_FANOUT],
}

impl Default for DirPage {
    fn default() -> Self {
        Self {
            blocks: std::array::from_fn(|_| None),
        }
    }
}

#[derive(Clone, Default)]
struct DataBlock {
    /// Two-level radix directory over `block_idx = entity_id / BLOCK_CAP`:
    /// `root[block_idx / DIR_FANOUT]` → `DirPage.blocks[block_idx % DIR_FANOUT]`
    /// → `Block`. `root` is `Arc`-wrapped so `new_version` is one `Arc` bump; a
    /// write path-copies only the root vector, the touched `DirPage`, and the
    /// touched `Block`, sharing every other dir page across MVCC versions.
    root: Arc<Vec<Option<Arc<DirPage>>>>,
}

impl DataBlock {
    #[inline]
    const fn locate(entity_id: u64) -> (usize, usize) {
        let id = entity_id as usize;
        (id >> BLOCK_SHIFT, id & BLOCK_MASK)
    }

    /// The block for `block_idx`, if present — two dependent loads
    /// (`root` → `DirPage` → `Block`), zero comparisons.
    #[inline]
    fn block(
        &self,
        block_idx: usize,
    ) -> Option<&Block> {
        self.root.get(block_idx / DIR_FANOUT)?.as_ref()?.blocks[block_idx % DIR_FANOUT].as_deref()
    }

    /// The `Arc<Block>` for `block_idx` (for tests that assert sharing identity).
    #[cfg(test)]
    fn block_arc(
        &self,
        block_idx: usize,
    ) -> Option<&Arc<Block>> {
        self.root.get(block_idx / DIR_FANOUT)?.as_ref()?.blocks[block_idx % DIR_FANOUT].as_ref()
    }

    /// Copy-on-write path-copy to the block for `block_idx`, creating the
    /// directory levels as needed. Clones only the root vector, the touched
    /// `DirPage`, and the touched `Block` when they are shared with another
    /// version.
    #[inline]
    fn block_mut(
        &mut self,
        block_idx: usize,
    ) -> &mut Block {
        let (dir_idx, page_slot) = (block_idx / DIR_FANOUT, block_idx % DIR_FANOUT);
        let root = Arc::make_mut(&mut self.root);
        if root.len() <= dir_idx {
            root.resize(dir_idx + 1, None);
        }
        let page = Arc::make_mut(root[dir_idx].get_or_insert_with(|| Arc::new(DirPage::default())));
        Arc::make_mut(page.blocks[page_slot].get_or_insert_with(|| Arc::new(Block::default())))
    }

    #[inline]
    fn get(
        &self,
        entity_id: u64,
    ) -> Option<SpanRef<'_>> {
        let (block_idx, slot_idx) = Self::locate(entity_id);
        let block = self.block(block_idx)?;
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
        let block = self.block_mut(block_idx);
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
            .block(block_idx)
            .is_some_and(|block| block.slots.get(slot_idx).is_some_and(|slot| slot.cap != 0));
        if !has_span && pairs.iter().all(|(_, v)| matches!(v, Value::Null)) {
            return (0, 0);
        }
        let block = self.block_mut(block_idx);
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
        // Only exclusively-owned levels are shrunk: a shared root / dir page /
        // block belongs to another version and must not be mutated. `get_mut`
        // returns `None` for shared, so the walk skips it.
        let Some(root) = Arc::get_mut(&mut self.root) else {
            return;
        };
        for page in root.iter_mut().flatten() {
            let Some(page) = Arc::get_mut(page) else {
                continue;
            };
            for arc in page.blocks.iter_mut().flatten() {
                let Some(block) = Arc::get_mut(arc) else {
                    continue;
                };
                // Commit-time compaction: reclaim update churn (dead spans +
                // in-span slack) once it reaches half the arena. Steady-state
                // waste stays bounded at 2x live payload — still below the C
                // engine's realloc churn — without recopying every block on
                // each commit.
                let waste = block.dead as usize + block.slack as usize;
                if waste * 2 >= block.arena.len() && block.arena.len() > COMPACT_MIN_ARENA {
                    block.compact();
                }
                if block.slots.len() == BLOCK_CAP {
                    block.arena.shrink_to_fit();
                    block.heap.shrink_to_fit();
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
        if self
            .block(block_idx)
            .is_some_and(|block| block.slots.get(slot_idx).is_some_and(|slot| slot.cap != 0))
        {
            let block = self.block_mut(block_idx);
            block.free_span(slot_idx);
            block.maybe_compact();
        }
    }

    /// Bytes the store has allocated that are *not* attributable to a live
    /// entity: the directory, the slot tables, and the reserved-but-unused parts
    /// of each block's arena and heap.
    ///
    /// Complements [`SpanRef::heap_bytes`], which covers the live payload. Every
    /// allocated byte falls in exactly one of the two, so a caller that adds the
    /// sampled per-entity estimate to this figure accounts for the whole store
    /// without double counting.
    ///
    /// Each directory level is counted exactly once: the root vector's own
    /// storage holds the page pointers, and a `DirPage`'s size already includes
    /// its block pointers — so only the *pointed-to* allocation (plus its `Arc`
    /// header) is added per level, never the pointer again.
    fn structural_memory_usage(&self) -> usize {
        // An `Arc` allocation carries strong + weak refcounts ahead of its value.
        const ARC_HDR: usize = 2 * std::mem::size_of::<usize>();
        let mut total = self.root.capacity() * std::mem::size_of::<Option<Arc<DirPage>>>();
        for page in self.root.iter().flatten() {
            total += ARC_HDR + std::mem::size_of::<DirPage>();
            for block in page.blocks.iter().flatten() {
                // `dead` (abandoned spans) and `slack` (reserved tail of a live
                // span) are allocated but referenced by no live entry, so they
                // belong here rather than in the per-entity payload; the same
                // holds for recycled heap holes and spare vector capacity.
                let live_arena = block.arena.len() - block.dead as usize - block.slack as usize;
                let live_heap = block.heap.len() - block.heap_free.len();

                total += ARC_HDR
                    + std::mem::size_of::<Block>()
                    + block.slots.capacity() * std::mem::size_of::<Slot>()
                    + (block.arena.capacity() - live_arena) * std::mem::size_of::<PackedAttr>()
                    + (block.heap.capacity() - live_heap) * std::mem::size_of::<Value>()
                    + block.heap_free.capacity() * std::mem::size_of::<u32>();
            }
        }
        total
    }
}

// ============================================================================
// AttributeStore
// ============================================================================

/// Attribute storage for graph entities, keyed by entity id.
///
/// A copy-on-write [`DataBlock`] of per-entity spans; a slot miss means the entity
/// has no attributes.
///
/// **Deliberately does not own an attribute-name table.** The name → id dictionary
/// lives once on [`crate::graph::graph::Graph`] and is passed to the methods that
/// need it, so the node and relationship stores share one id space — the same shape
/// as C, where `GraphContext` holds one `attributes` array for two `DataBlock`s.
/// A table per store numbered the same name differently for nodes and relationships,
/// and since a bare id travels on the wire in `GRAPH.EFFECT`, an RDB-seeded replica
/// resolved it against different numbering and wrote to the wrong attribute (#2457).
#[derive(Clone, Default)]
pub struct AttributeStore {
    /// Block-allocated per-entity attribute spans (COW across MVCC versions).
    data: DataBlock,
}

impl AttributeStore {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Cheap snapshot clone for a new MVCC version: one root `Arc` bump, which
    /// shares every directory page and block until a write path-copies one.
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

    /// The attribute ids an entity carries.
    ///
    /// Reads only the span's ids, so it stays cheaper than [`Self::get_all_attrs_by_id`]
    /// when the values are not wanted.
    pub fn get_attr_ids(
        &self,
        key: u64,
    ) -> impl Iterator<Item = u16> + '_ {
        self.data
            .get(key)
            .into_iter()
            .flat_map(|span| span.entries().iter().map(|attr| attr.id))
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

    /// Allocated bytes not attributable to any live entity — see
    /// [`DataBlock::structural_memory_usage`].
    #[must_use]
    pub fn structural_memory_usage(&self) -> usize {
        self.data.structural_memory_usage()
    }

    // ---- serialization -----------------------------------------------------

    /// Encode a range of entities.
    ///
    /// Ids are written as they are stored: the graph has one attribute dictionary, so a
    /// span's id is already the id the RDB means. This used to build a local → global
    /// remap per call, which only existed because each store numbered names itself.
    pub fn encode_with_range(
        &self,
        w: &mut dyn Writer,
        deleted: &RoaringTreemap,
        max_id: u64,
        count: u64,
        offset: u64,
    ) {
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
                for (attr_id, value) in span.iter() {
                    w.write_unsigned(u64::from(attr_id));
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

impl AttributeStore {
    /// Decode `count` entity spans, dropping any attribute whose id is not in the
    /// graph's dictionary.
    ///
    /// `attr_limit` is the dictionary's length, and it is the only way to decode a
    /// store: `AttributeStore` deliberately does **not** implement [`Decode`].
    ///
    /// It cannot implement it honestly. `Decode::decode_with_count`'s signature has
    /// nowhere to put the dictionary length, and an earlier revision satisfied the
    /// trait by passing `usize::MAX` — which silently disabled this bound on two of
    /// the three load paths, including the multi-key one, i.e. every graph large
    /// enough to be split across virtual keys. Without a real bound a malformed RDB
    /// stores an id resolving to no name, and that reads back as a **silently absent
    /// attribute** rather than an error.
    ///
    /// Omitting the impl makes that mistake a compile error rather than something a
    /// reviewer has to notice.
    pub fn decode_entities(
        &mut self,
        r: &mut dyn Reader,
        count: u64,
        attr_limit: usize,
    ) -> Result<(), String> {
        for _ in 0..count {
            let entity_id = r.read_unsigned()?;
            let attr_count = r.read_unsigned()?;

            let mut entries: Vec<(u16, Value)> = Vec::with_capacity(attr_count as usize);
            for _ in 0..attr_count {
                // Validate before narrowing. Casting first defeats the bound: an
                // encoded id of 65536 truncates to 0, which then passes any nonempty
                // `attr_limit` and lands on whatever attribute 0 happens to be.
                let raw_attr_id = r.read_unsigned()?;
                let value = Value::decode(r)?;

                let in_range =
                    raw_attr_id < attr_limit as u64 && raw_attr_id < u64::from(ATTRIBUTE_ID_NONE);
                if in_range && !matches!(value, Value::Null) {
                    entries.push((raw_attr_id as u16, value));
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

    /// A store paired with the attribute dictionary that now lives on `Graph`.
    ///
    /// These tests exercise span storage — copy-on-write, compaction, snapshot isolation
    /// across versions — not where the dictionary lives, so pairing the two here keeps
    /// them in their original shape. `Deref` forwards everything that does not need the
    /// dictionary; the handful of methods that do are shadowed below.
    #[derive(Clone, Default)]
    struct TestStore {
        names: AttrNameMap,
        store: AttributeStore,
    }

    impl TestStore {
        fn new() -> Self {
            Self::default()
        }

        /// Clones the dictionary alongside the store, which is what `Graph::new_version`
        /// does — so snapshot-isolation tests still see a version's own name table.
        fn new_version(&self) -> Self {
            Self {
                names: self.names.clone(),
                store: self.store.new_version(),
            }
        }

        fn get_or_create_attr_id(
            &mut self,
            attr: &Arc<String>,
        ) -> u16 {
            self.names.get_or_create(attr)
        }

        fn get_attr_id(
            &self,
            attr: &Arc<String>,
        ) -> Option<usize> {
            self.names.get_index_of(attr)
        }

        fn get_attr(
            &self,
            key: u64,
            attr: &Arc<String>,
        ) -> Option<Value> {
            let idx = self.names.get_index_of(attr)? as u16;
            self.store.get_attr_by_idx(key, idx)
        }

        fn get_all_attrs(
            &self,
            key: u64,
        ) -> impl Iterator<Item = (Arc<String>, Value)> + '_ {
            self.store
                .get_all_attrs_by_id(key)
                .filter_map(|(id, value)| self.names.get(id as usize).map(|n| (n.clone(), value)))
                .collect::<Vec<_>>()
                .into_iter()
        }
    }

    impl std::ops::Deref for TestStore {
        type Target = AttributeStore;

        fn deref(&self) -> &AttributeStore {
            &self.store
        }
    }

    impl std::ops::DerefMut for TestStore {
        fn deref_mut(&mut self) -> &mut AttributeStore {
            &mut self.store
        }
    }

    fn store_with(entries: &[(u64, &[(&str, Value)])]) -> TestStore {
        let mut store = TestStore::new();
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
        let mut store = TestStore::new();
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
        assert!(store.data.block(0).unwrap().heap.is_empty());
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
        assert_eq!(store.data.block(0).unwrap().heap.len(), 2);
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
        let block = store.data.block(0).unwrap();
        assert_eq!(block.heap.len(), 1);
        assert!(block.heap_free.is_empty());
    }

    #[test]
    fn compaction_reclaims_abandoned_spans() {
        let mut store = TestStore::new();
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
        let block = store.data.block(0).unwrap();
        // Compaction must have run: dead entries bounded by live ones.
        assert!(
            block.dead as usize * 2 <= block.arena.len() || block.arena.len() <= COMPACT_MIN_ARENA
        );
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
        let block_before = store.data.block_arc(0).unwrap().clone();

        // Entity 2 shares block 0 but has no attributes; removing it must not
        // trigger a COW copy of the block.
        let mut keys = RoaringTreemap::new();
        keys.insert(2);
        // Also an id in a block that doesn't exist at all.
        keys.insert(BLOCK_CAP as u64 * 10);
        store.remove_all(&keys);

        assert!(Arc::ptr_eq(&block_before, store.data.block_arc(0).unwrap()));
    }

    #[test]
    fn batch_get_with_default() {
        let store = store_with(&[(1, &[("a", Value::Int(1))]), (2, &[("b", Value::Int(2))])]);
        let idx = store.get_attr_id(&name("a")).unwrap() as u16;
        let mut out = Vec::new();
        store.get_attrs_by_idx_batch_into(&[1, 2, 3], idx, &Value::Null, &mut out);
        assert_eq!(out, vec![Value::Int(1), Value::Null, Value::Null]);
    }

    /// First id of block `block_idx` under the default grain.
    fn id_in_block(block_idx: u64) -> u64 {
        block_idx * BLOCK_CAP as u64
    }

    #[test]
    fn spans_directory_page_boundary() {
        // One directory page covers DIR_FANOUT blocks, so block DIR_FANOUT is the
        // first block of root[1] — the level the flat directory never had.
        let last_of_page0 = id_in_block(DIR_FANOUT as u64 - 1);
        let first_of_page1 = id_in_block(DIR_FANOUT as u64);
        let far_page = id_in_block(DIR_FANOUT as u64 * 5 + 2);

        let store = store_with(&[
            (last_of_page0, &[("a", Value::Int(1))]),
            (first_of_page1, &[("a", Value::Int(2))]),
            (far_page, &[("a", Value::Int(3))]),
        ]);

        // The ids really do land in distinct directory pages.
        assert!(store.data.root.len() > 5, "spans several directory pages");
        assert!(store.data.root[0].is_some() && store.data.root[1].is_some());

        assert_eq!(
            store.get_attr(last_of_page0, &name("a")),
            Some(Value::Int(1))
        );
        assert_eq!(
            store.get_attr(first_of_page1, &name("a")),
            Some(Value::Int(2))
        );
        assert_eq!(store.get_attr(far_page, &name("a")), Some(Value::Int(3)));
        // Neighbours across the boundary are independent.
        assert_eq!(store.get_attr(last_of_page0 + 1, &name("a")), None);
        assert_eq!(store.get_attr(first_of_page1 + 1, &name("a")), None);
        // A never-populated directory page reads as absent, not as a panic.
        assert_eq!(
            store.get_attr(id_in_block(DIR_FANOUT as u64 * 3), &name("a")),
            None
        );
    }

    #[test]
    fn write_shares_untouched_directory_pages() {
        let in_page0 = 1u64;
        let in_page1 = id_in_block(DIR_FANOUT as u64);
        let v1 = store_with(&[
            (in_page0, &[("a", Value::Int(1))]),
            (in_page1, &[("a", Value::Int(2))]),
        ]);
        let page1_before = Arc::clone(v1.data.root[1].as_ref().expect("page 1 present"));

        // Write only into directory page 0.
        let mut v2 = v1.new_version();
        let a = v2.get_or_create_attr_id(&name("a"));
        let mut attrs = FxHashMap::default();
        attrs.insert(in_page0, vec![(a, Value::Int(99))]);
        v2.insert_attrs(&attrs).unwrap();

        // Page 1 was not on the copied path: still the very same allocation.
        assert!(
            Arc::ptr_eq(
                &page1_before,
                v2.data.root[1].as_ref().expect("page 1 present")
            ),
            "untouched directory page must stay shared, not be copied"
        );
        // And the write is invisible to the older snapshot.
        assert_eq!(v2.get_attr(in_page0, &name("a")), Some(Value::Int(99)));
        assert_eq!(v1.get_attr(in_page0, &name("a")), Some(Value::Int(1)));
        assert_eq!(v2.get_attr(in_page1, &name("a")), Some(Value::Int(2)));
    }

    #[test]
    fn cross_block_ids() {
        let far_id = BLOCK_CAP as u64 * 3 + 7;
        let store = store_with(&[(far_id, &[("a", Value::Int(42))])]);
        assert_eq!(store.get_attr(far_id, &name("a")), Some(Value::Int(42)));
        assert_eq!(store.get_attr(far_id - 1, &name("a")), None);
    }

    // ── RDB id bounds ──
    //
    // The store no longer owns the attribute dictionary, so the id bound is a
    // parameter. That seam is easy to leave open: an earlier revision satisfied
    // `Decode::decode_with_count` by delegating with `usize::MAX`, which silently
    // disabled the check on the multi-key load path — every graph large enough to be
    // split across virtual keys. These pin both halves of the current behaviour.

    /// Records the calls a [`Writer`] receives, so a stream can be replayed into a
    /// [`Reader`] without this test knowing how `Value` encodes itself.
    #[derive(Default)]
    struct Recorder {
        ops: Vec<Op>,
    }

    #[derive(Clone)]
    enum Op {
        Unsigned(u64),
        Signed(i64),
        Double(f64),
        Buffer(Vec<u8>),
    }

    impl Writer for Recorder {
        fn write_unsigned(
            &mut self,
            val: u64,
        ) {
            self.ops.push(Op::Unsigned(val));
        }
        fn write_signed(
            &mut self,
            val: i64,
        ) {
            self.ops.push(Op::Signed(val));
        }
        fn write_double(
            &mut self,
            val: f64,
        ) {
            self.ops.push(Op::Double(val));
        }
        fn write_buffer(
            &mut self,
            data: &[u8],
        ) {
            self.ops.push(Op::Buffer(data.to_vec()));
        }
    }

    struct Replay {
        ops: std::collections::VecDeque<Op>,
    }

    impl Replay {
        fn next(&mut self) -> Result<Op, String> {
            self.ops
                .pop_front()
                .ok_or_else(|| "replay: end".to_string())
        }
    }

    impl Reader for Replay {
        fn read_unsigned(&mut self) -> Result<u64, String> {
            match self.next()? {
                Op::Unsigned(v) => Ok(v),
                _ => Err("replay: expected unsigned".to_string()),
            }
        }
        fn read_signed(&mut self) -> Result<i64, String> {
            match self.next()? {
                Op::Signed(v) => Ok(v),
                _ => Err("replay: expected signed".to_string()),
            }
        }
        fn read_double(&mut self) -> Result<f64, String> {
            match self.next()? {
                Op::Double(v) => Ok(v),
                _ => Err("replay: expected double".to_string()),
            }
        }
        fn read_buffer(&mut self) -> Result<Vec<u8>, String> {
            match self.next()? {
                Op::Buffer(v) => Ok(v),
                _ => Err("replay: expected buffer".to_string()),
            }
        }
    }

    /// One entity span carrying `(attr_id, value)` pairs, in the layout
    /// `decode_entities` expects.
    fn span_stream(
        entity_id: u64,
        attrs: &[(u16, Value)],
    ) -> Replay {
        let mut rec = Recorder::default();
        rec.write_unsigned(entity_id);
        rec.write_unsigned(attrs.len() as u64);
        for (attr_id, value) in attrs {
            rec.write_unsigned(u64::from(*attr_id));
            value.encode(&mut rec);
        }
        Replay {
            ops: rec.ops.into(),
        }
    }

    #[test]
    fn decode_entities_drops_ids_beyond_the_dictionary() {
        let mut store = AttributeStore::default();
        let mut r = span_stream(7, &[(0, Value::Int(10)), (5, Value::Int(20))]);

        // A dictionary holding a single name: id 0 is real, id 5 resolves to nothing.
        let mut dict = AttrNameMap::default();
        dict.insert(Arc::new("a".to_string()));
        store.decode_entities(&mut r, 1, dict.len()).unwrap();

        assert_eq!(
            store.get_attr_by_idx(7, 0),
            Some(Value::Int(10)),
            "an id inside the dictionary must be kept"
        );
        assert_eq!(
            dict.get(0).cloned(),
            Some(Arc::new("a".to_string())),
            "and that id must be the one the dictionary resolves to a name"
        );
        assert_eq!(
            store.get_attr_by_idx(7, 5),
            None,
            "an id past the end of the dictionary must be dropped, not stored: it \
             resolves to no name and would read back as a silently absent attribute"
        );
        assert_eq!(
            store.get_all_attrs_by_id(7).count(),
            1,
            "only the in-range attribute should have been stored"
        );
    }
}
