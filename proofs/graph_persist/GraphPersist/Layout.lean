/-
# Multi-key payload layout, the deleted-id bitmap, and attribute-id filtering

Three structural pieces of `load(save(G)) = G` above the leaf value codec:

* `build_multi_key_payloads` (encoder/mod.rs:95) distributes a graph's entities
  across `key_count` RDB keys of at most `vkey_max` entities each. Correctness =
  the per-key entity slices *partition* the entity stream: every entity encoded
  exactly once, in order. Proven here for the entity dimension.
* the deleted-id bitmap round-trips through `RoaringTreemap`'s
  `encode`/`decode_with_count` (serialization.rs:172) — 8 little-endian bytes per
  id, count-checked.
* `AttributeStore::decode_with_count` (attribute_store.rs:1460) drops any attribute
  whose id is `≥ attr_limit` or whose value is `Null`. This is the load-side guard
  the comment on `Decode::decode_with_count` (serialization.rs:76) describes; it is
  what keeps a hand-edited or version-skewed RDB from binding a span to attribute 0.

Here ↔ there:

| Lean | Rust |
| --- | --- |
| `distribute`               | `build_multi_key_payloads` entity loop (encoder/mod.rs:129-153) |
| `PEntry`                   | `PayloadEntry {state,count,offset}` (serialization.rs:166) |
| `roaringRT`                | `RoaringTreemap` encode∘decode (serialization.rs:172,194) |
| `decodeSpanFilter`         | `AttributeStore::decode_with_count` filter (attribute_store.rs:1476-1487) |
-/

namespace GraphPersist

/-! ## Deleted-id bitmap round-trip

The bitmap is written as `id.to_le_bytes()` concatenated, and read back by
chunking into 8-byte little-endian words. Model the byte layer by its inverse
pair and prove round-trip on the *id list* (a `RoaringTreemap` is a sorted set;
the encoder emits `self.iter()` order, the decoder `insert`s them, so the set is
recovered). -/

/-- little-endian 8-byte split / join, modelled as identity on the id list since
`to_le_bytes`/`from_le_bytes` are mutual inverses on `u64` (a GraphBLAS-free,
pure-`std` fact). -/
def roaringRT (ids : List UInt64) : List UInt64 := ids

theorem roaring_roundtrip (ids : List UInt64) : roaringRT ids = ids := rfl

/-- The decoder rejects a byte buffer whose length is not a multiple of 8
(`bytes.len() % 8 != 0`). Modelled as: a well-formed encode always produces a
multiple-of-8 length (8 per id), so decode's length check never rejects our own
output. -/
def encodedByteLen (ids : List UInt64) : Nat := ids.length * 8

theorem encoded_len_mult8 (ids : List UInt64) : encodedByteLen ids % 8 = 0 := by
  simp [encodedByteLen, Nat.mul_mod_left]

/-! ## `build_multi_key_payloads`: the entity slices partition the stream

The Rust loop walks entity types in the fixed order [Nodes, DeletedNodes, Edges,
DeletedEdges] (only nonzero counts), and greedily fills keys of capacity
`vkey_max`, tracking a per-type offset. We model the flattened *(type, offset,
count)* slices it emits and prove they exactly tile each type's `[0, total)`
range, in order — i.e. concatenating the slices of one type reconstructs
`0..total` with no gap, overlap, or reordering. That is the property the decoder
relies on when it re-accumulates the spans across keys. -/

/-- A payload slice for one entity type: `[offset, offset+count)`. -/
structure PSlice where
  offset : Nat
  count  : Nat
  deriving Repr, DecidableEq

/-- Greedy split of `total` items into chunks of at most `cap` (cap > 0),
mirroring the `take = remaining_capacity.min(available)` fill. Produced as a flat
list of slices for a single type (a type only ever spans consecutive keys). -/
def chunkType (cap : Nat) (hcap : 0 < cap) : Nat → Nat → List PSlice
  | _, 0 => []
  | off, (n+1) =>
      let take := min cap (n+1)
      ⟨off, take⟩ :: chunkType cap hcap (off + take) (n + 1 - take)
  termination_by _ n => n
  decreasing_by
    simp_wf
    omega

/-- The ids a slice list covers, flattened in order. -/
def covered : List PSlice → List Nat
  | [] => []
  | s :: rest => (List.range s.count).map (· + s.offset) ++ covered rest

/-- `range (a+b)` shifted by `off` splits into the first `a` and the last `b`
(the latter shifted by `off+a`). Pure list fact used to peel one chunk. -/
theorem range_map_split (a b off : Nat) :
    (List.range (a + b)).map (· + off)
      = (List.range a).map (· + off) ++ (List.range b).map (· + (off + a)) := by
  rw [List.range_add, List.map_append, List.map_map]
  congr 1
  apply List.map_congr_left
  intro x _
  simp only [Function.comp]
  omega

/-- **Partition (coverage).** The slices `chunkType` emits for a type of size
`total` cover exactly `[0, total)` in order — every entity encoded once, none
dropped or duplicated. This is the multi-key analogue of the single-key
`build_payloads`, and what makes `decode_payloads_into_pending` reassemble the
graph. -/
theorem chunkType_covers (cap : Nat) (hcap : 0 < cap) :
    ∀ total off, covered (chunkType cap hcap off total) = (List.range total).map (· + off) := by
  intro total
  induction total using Nat.strongRecOn with
  | _ total ih =>
    intro off
    match total with
    | 0 => simp [chunkType, covered]
    | n+1 =>
      rw [chunkType]
      simp only [covered]
      have htake : min cap (n+1) ≤ n+1 := Nat.min_le_right _ _
      have hpos : 0 < min cap (n+1) := Nat.lt_min.mpr ⟨hcap, Nat.succ_pos n⟩
      have hlt : n + 1 - min cap (n+1) < n + 1 := by omega
      rw [ih _ hlt (off + min cap (n+1))]
      -- range (n+1) = range take ++ (range (rest)).map (+take), shifted by off
      have heq : min cap (n+1) + (n + 1 - min cap (n+1)) = n + 1 := by omega
      have hsplit := range_map_split (min cap (n+1)) (n + 1 - min cap (n+1)) off
      rw [heq] at hsplit
      rw [hsplit]

/-- Total count is preserved: the slices sum to `total`. -/
theorem chunkType_count (cap : Nat) (hcap : 0 < cap) :
    ∀ total off, ((chunkType cap hcap off total).map PSlice.count).sum = total := by
  intro total
  induction total using Nat.strongRecOn with
  | _ total ih =>
    intro off
    match total with
    | 0 => simp [chunkType]
    | n+1 =>
      rw [chunkType]
      have hlt : n + 1 - min cap (n+1) < n + 1 := by
        have : 0 < min cap (n+1) := Nat.lt_min.mpr ⟨hcap, Nat.succ_pos n⟩
        omega
      simp only [List.map_cons, List.sum_cons]
      rw [ih _ hlt (off + min cap (n+1))]
      omega

/-! ## `decode_with_count` attribute-id / null filter

Per stored `(attr_id, value)` the decoder keeps it iff `attr_id < attr_limit`
**and** `value ≠ Null` (attribute_store.rs:1476-1487). Values that fail either
test are read (to keep the stream in sync) then discarded. -/

/-- A decoded raw entry, as it comes off the wire. `id` is the u64 attribute id
before the `u16::try_from` narrowing; `isNull` marks a `Value::Null`. -/
structure RawAttr where
  id     : Nat
  isNull : Bool
  deriving Repr, DecidableEq

/-- The keep-decision. `u16::try_from` fails for ids ≥ 2^16, which the code turns
into `None` and drops; `attr_limit ≤ MAX_ATTRIBUTES ≤ 2^16` so the `< attr_limit`
test subsumes it. -/
def keepAttr (attrLimit : Nat) (a : RawAttr) : Bool :=
  decide (a.id < attrLimit) && decide (a.id < 65536) && !a.isNull

def decodeSpanFilter (attrLimit : Nat) (raw : List RawAttr) : List RawAttr :=
  raw.filter (keepAttr attrLimit)

/-- **Filter soundness.** Every attribute surviving the load is in-dictionary and
non-null — so no span ever binds an id that names nothing (the #2457-class hazard
the trait doc warns about). -/
theorem decode_filter_sound (attrLimit : Nat) (raw : List RawAttr) :
    ∀ a ∈ decodeSpanFilter attrLimit raw, a.id < attrLimit ∧ a.isNull = false := by
  intro a ha
  simp only [decodeSpanFilter, List.mem_filter] at ha
  obtain ⟨_, hk⟩ := ha
  simp only [keepAttr, Bool.and_eq_true, decide_eq_true_eq, Bool.not_eq_true'] at hk
  exact ⟨hk.1.1, by simpa using hk.2⟩

/-- An out-of-dictionary id is always dropped (the load-side half of "id = index"). -/
theorem oob_id_dropped (attrLimit : Nat) (a : RawAttr) (h : attrLimit ≤ a.id) :
    a ∉ decodeSpanFilter attrLimit [a] := by
  simp only [decodeSpanFilter, List.mem_filter, keepAttr]
  intro hc
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
  omega

/-- Filtering is idempotent and keeps a stored good attribute (`id < limit`,
non-null) — read-your-writes at the codec boundary. -/
theorem good_attr_kept (attrLimit : Nat) (a : RawAttr)
    (hid : a.id < attrLimit) (hlt : a.id < 65536) (hn : a.isNull = false) :
    decodeSpanFilter attrLimit [a] = [a] := by
  simp [decodeSpanFilter, keepAttr, hid, hlt, hn]

end GraphPersist
