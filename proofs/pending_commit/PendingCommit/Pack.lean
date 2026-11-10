import Std.Tactic.BVDecide
/-
# Packed entries and the block heap: `pack_value` / `unpack` / free list

attribute_store.rs:256-300 (`Tag`, `PackedAttr`, `heap_index`) and the
`Block` methods :430-532. A stored value is a 12-byte `(id u16, tag u8,
[u8; 8])`; scalars are their little-endian bytes, everything else (`String`,
`List`, `VecF32`, `Map`, …: `V.other`) goes to the block's `heap` and the
payload's first four bytes are its `u32` index. Removed heap values leave a
`Null` hole whose index is pushed on `heap_free` and reused first.

Floats are modelled by their IEEE bits: `f.to_le_bytes()` / `from_le_bytes`
are bit casts, so NaN payloads and `-0.0` round-trip exactly as in Rust.
`free` is the `heap_free` vector with its *end* (where `push`/`pop` act) at the
head of the list.

* `le64_roundtrip`, `le32_roundtrip` — `from_le_bytes ∘ to_le_bytes = id`.
* `pack_unpack` — `unpack` of what `pack_value` wrote is the value, for every
  variant, and no heap value another live entry references is touched.
* `HInv` (refs/free partition the heap, holes are `Null`) is kept by
  `pack_value` (`pack_hinv`) and `release_heap_value` (`release_hinv`);
  `release_span_values` = releasing each live entry (`releaseSpan_hinv`).
* `store_packed_value` writes exactly one arena cell (`store_get`).
* `compactHeap_unpack` — `compact`'s heap remap (:798-822) keeps every live
  entry's value, given live entries reference distinct heap slots (which is
  `HInv.refs_nodup`; with a shared index `mem::replace` would hand the second
  entry a `Null`).
* `heapBytes_eq` — `SpanRef::heap_bytes` (:883).
-/
namespace PendingCommit.Pack

abbrev Byte := BitVec 8

inductive V where
  | null
  | bool (b : Bool)
  | int (x : BitVec 64)
  | float (bits : BitVec 64)
  | point (lat lon : BitVec 32)
  | datetime (x : BitVec 64)
  | date (x : BitVec 64)
  | time (x : BitVec 64)
  | duration (x : BitVec 64)
  | other (o : Nat)
  deriving DecidableEq, Repr

inductive Tag where
  | null | bool | int | float | point | datetime | date | time | duration | heap
  deriving DecidableEq, Repr

structure PA where
  id : Nat
  tag : Tag
  payload : List Byte

def le64 (x : BitVec 64) : List Byte :=
  [x.extractLsb' 0 8, x.extractLsb' 8 8, x.extractLsb' 16 8, x.extractLsb' 24 8,
   x.extractLsb' 32 8, x.extractLsb' 40 8, x.extractLsb' 48 8, x.extractLsb' 56 8]
def fromLe64 (p : List Byte) : BitVec 64 :=
  p.getD 7 0 ++ p.getD 6 0 ++ p.getD 5 0 ++ p.getD 4 0 ++ p.getD 3 0 ++ p.getD 2 0 ++
    p.getD 1 0 ++ p.getD 0 0
def le32 (x : BitVec 32) : List Byte :=
  [x.extractLsb' 0 8, x.extractLsb' 8 8, x.extractLsb' 16 8, x.extractLsb' 24 8]
def fromLe32 (p : List Byte) : BitVec 32 := p.getD 3 0 ++ p.getD 2 0 ++ p.getD 1 0 ++ p.getD 0 0

theorem le64_roundtrip (x : BitVec 64) : fromLe64 (le64 x) = x := by
  simp only [fromLe64, le64, List.getD_cons_succ, List.getD_cons_zero]; bv_decide

theorem le32_roundtrip (x : BitVec 32) : fromLe32 (le32 x) = x := by
  simp only [fromLe32, le32, List.getD_cons_succ, List.getD_cons_zero]; bv_decide

def zeros (n : Nat) : List Byte := List.replicate n 0

/-- `PackedAttr::heap_index` (:282): `u32::from_le_bytes(payload[..4]) as usize`. -/
def heapIndex (p : List Byte) : Nat := (fromLe32 (p.take 4)).toNat

structure Heap where
  heap : List V
  free : List Nat

/-- `pack_value` (:430): `(tag, payload)` and the heap after the side effect. -/
def pack (H : Heap) : V → Tag × List Byte × Heap
  | .null => (.null, zeros 8, H)
  | .bool b => (.bool, (if b then 1 else 0) :: zeros 7, H)
  | .int x => (.int, le64 x, H)
  | .float f => (.float, le64 f, H)
  | .point a o => (.point, le32 a ++ le32 o, H)
  | .datetime t => (.datetime, le64 t, H)
  | .date t => (.date, le64 t, H)
  | .time t => (.time, le64 t, H)
  | .duration t => (.duration, le64 t, H)
  | .other o =>
    match H.free with
    | i :: rest => (.heap, le32 (BitVec.ofNat 32 i) ++ zeros 4, ⟨H.heap.set i (.other o), rest⟩)
    | [] => (.heap, le32 (BitVec.ofNat 32 H.heap.length) ++ zeros 4, ⟨H.heap ++ [.other o], []⟩)

/-- `unpack` (:486). -/
def unpack (H : Heap) (t : Tag) (p : List Byte) : V :=
  match t with
  | .null => .null
  | .bool => .bool (p.getD 0 0 != 0)
  | .int => .int (fromLe64 p)
  | .float => .float (fromLe64 p)
  | .point => .point (fromLe32 (p.take 4)) (fromLe32 (p.drop 4))
  | .datetime => .datetime (fromLe64 p)
  | .date => .date (fromLe64 p)
  | .time => .time (fromLe64 p)
  | .duration => .duration (fromLe64 p)
  | .heap => H.heap.getD (heapIndex p) .null

/-- Heap well-formedness: the indices live entries reference (`refs`) and the
free list partition `0..heap.len()`, and every hole is `Null`. -/
structure HInv (H : Heap) (refs : List Nat) : Prop where
  refs_nodup : refs.Nodup
  free_nodup : H.free.Nodup
  disj : ∀ i ∈ refs, i ∉ H.free
  cover : ∀ i, i < H.heap.length ↔ (i ∈ refs ∨ i ∈ H.free)
  holes : ∀ i ∈ H.free, H.heap.getD i .null = .null

theorem heapIndex_le32 (i : Nat) (h : i < 2 ^ 32) : heapIndex (le32 (BitVec.ofNat 32 i) ++ zeros 4) = i := by
  simp only [heapIndex, zeros]
  have : (le32 (BitVec.ofNat 32 i) ++ List.replicate 4 0).take 4 = le32 (BitVec.ofNat 32 i) := by
    simp [le32]
  rw [this, le32_roundtrip, BitVec.toNat_ofNat]; omega

theorem getD_set_eq {l : List V} {i : Nat} (h : i < l.length) (v : V) : (l.set i v).getD i .null = v := by
  simp [List.getD_eq_getElem?_getD, h]

theorem getD_set_ne (l : List V) {i j : Nat} (h : j ≠ i) (v : V) : (l.set i v).getD j .null = l.getD j .null := by
  simp [List.getD_eq_getElem?_getD, List.getElem?_set_ne (Ne.symm h)]

theorem getD_append_lt {l : List V} {j : Nat} (h : j < l.length) (v : V) : (l ++ [v]).getD j .null = l.getD j .null := by
  simp [List.getD_eq_getElem?_getD, List.getElem?_append_left h]

/-- The heap index `pack_value` hands an out-of-line value. -/
def newIdx (H : Heap) : Nat := match H.free with | i :: _ => i | [] => H.heap.length

/-- **`pack_unpack`**: unpacking what `pack_value` produced gives the value back,
and heap cells referenced by other live entries are untouched. Needs the heap
to stay within `u32` indices (`(heap.len() - 1) as u32`, :458). -/
theorem pack_unpack (H : Heap) (refs : List Nat) (hI : HInv H refs) (hcap : H.heap.length < 2 ^ 32)
    (v : V) :
    let r := pack H v
    unpack r.2.2 r.1 r.2.1 = v ∧ ∀ j ∈ refs, r.2.2.heap.getD j .null = H.heap.getD j .null := by
  cases v with
  | null => exact ⟨rfl, fun _ _ => rfl⟩
  | bool b => cases b <;> exact ⟨by simp [pack, unpack, zeros], fun _ _ => rfl⟩
  | int x => exact ⟨by simp [pack, unpack, le64_roundtrip], fun _ _ => rfl⟩
  | float x => exact ⟨by simp [pack, unpack, le64_roundtrip], fun _ _ => rfl⟩
  | datetime x => exact ⟨by simp [pack, unpack, le64_roundtrip], fun _ _ => rfl⟩
  | date x => exact ⟨by simp [pack, unpack, le64_roundtrip], fun _ _ => rfl⟩
  | time x => exact ⟨by simp [pack, unpack, le64_roundtrip], fun _ _ => rfl⟩
  | duration x => exact ⟨by simp [pack, unpack, le64_roundtrip], fun _ _ => rfl⟩
  | point a o =>
    refine ⟨?_, fun _ _ => rfl⟩
    simp only [pack, unpack]
    have h1 : (le32 a ++ le32 o).take 4 = le32 a := by simp [le32]
    have h2 : (le32 a ++ le32 o).drop 4 = le32 o := by simp [le32]
    rw [h1, h2, le32_roundtrip, le32_roundtrip]
  | other o =>
    simp only [pack]
    cases hf : H.free with
    | cons i rest =>
      have hi : i < H.heap.length := (hI.cover i).2 (.inr (by rw [hf]; simp))
      simp only [unpack]
      rw [heapIndex_le32 i (by omega), getD_set_eq hi]
      refine ⟨rfl, fun j hj => getD_set_ne _ ?_ _⟩
      intro e; subst e; exact hI.disj _ hj (by rw [hf]; simp)
    | nil =>
      simp only [unpack]
      rw [heapIndex_le32 _ hcap]
      refine ⟨by simp [List.getD_eq_getElem?_getD], fun j hj => getD_append_lt ?_ _⟩
      exact (hI.cover j).2 (.inl hj)

/-- `pack_value` of an out-of-line value keeps `HInv`, with the new index
added to the live references. -/
theorem pack_hinv (H : Heap) (refs : List Nat) (hI : HInv H refs) (o : Nat) :
    newIdx H ∉ refs ∧ HInv (pack H (.other o)).2.2 (newIdx H :: refs) := by
  simp only [pack, newIdx]
  cases hf : H.free with
  | cons i rest =>
    have hfr : H.free = i :: rest := hf
    have hnd := hI.free_nodup; rw [hfr] at hnd
    have hni : i ∉ refs := fun h => hI.disj i h (by rw [hfr]; simp)
    refine ⟨hni, ⟨List.nodup_cons.2 ⟨hni, hI.refs_nodup⟩, (List.nodup_cons.1 hnd).2, ?_, ?_, ?_⟩⟩
    · intro j hj hjr
      simp at hj; rcases hj with rfl | hj
      · exact (List.nodup_cons.1 hnd).1 hjr
      · exact hI.disj j hj (by rw [hfr]; simp [hjr])
    · intro j; simp only [List.length_set, List.mem_cons]
      rw [hI.cover j, hfr]; simp only [List.mem_cons]
      constructor
      · rintro (h | h | h)
        · exact .inl (.inr h)
        · exact .inl (.inl h)
        · exact .inr h
      · rintro ((h | h) | h)
        · exact .inr (.inl h)
        · exact .inl h
        · exact .inr (.inr h)
    · intro j hj
      have hji : j ≠ i := fun e => (List.nodup_cons.1 hnd).1 (e ▸ hj)
      rw [getD_set_ne _ hji]; exact hI.holes j (by rw [hfr]; simp [hj])
  | nil =>
    have hfr : H.free = [] := hf
    have hni : H.heap.length ∉ refs := fun h => by have := (hI.cover _).2 (.inl h); omega
    refine ⟨hni, ⟨List.nodup_cons.2 ⟨hni, hI.refs_nodup⟩, List.nodup_nil, fun _ _ h => by simp at h, ?_, fun _ h => by simp at h⟩⟩
    intro j; simp only [List.length_append, List.length_singleton, List.mem_cons, List.not_mem_nil, or_false]
    have := hI.cover j; rw [hfr] at this; simp only [List.not_mem_nil, or_false] at this
    constructor
    · intro hj
      rcases Nat.lt_or_ge j H.heap.length with h1 | h1
      · exact .inr (this.1 h1)
      · exact .inl (by omega)
    · rintro (rfl | h)
      · omega
      · have := this.2 h; omega

/-- `release_heap_value` (:510) of a heap entry with index `i`: the hole gets
`Null` and `i` goes on the free list. -/
def release (H : Heap) (i : Nat) : Heap := ⟨H.heap.set i .null, i :: H.free⟩

theorem release_hinv (H : Heap) (refs : List Nat) (hI : HInv H refs) (i : Nat) (hi : i ∈ refs) :
    HInv (release H i) (refs.erase i) ∧
      ∀ j ∈ refs.erase i, (release H i).heap.getD j .null = H.heap.getD j .null := by
  have hnd := hI.refs_nodup
  have hlt : i < H.heap.length := (hI.cover i).2 (.inl hi)
  have hmemE : ∀ j, j ∈ refs.erase i ↔ (j ≠ i ∧ j ∈ refs) := fun j => by
    rw [List.Nodup.mem_erase_iff hnd]
  refine ⟨⟨hnd.erase i, List.nodup_cons.2 ⟨hI.disj i hi, hI.free_nodup⟩, ?_, ?_, ?_⟩, ?_⟩
  · intro j hj hjf
    rw [hmemE] at hj
    simp [release] at hjf; rcases hjf with rfl | hjf
    · exact hj.1 rfl
    · exact hI.disj j hj.2 hjf
  · intro j; simp only [release, List.length_set, List.mem_cons, hmemE]
    rw [hI.cover j]
    by_cases hj : j = i
    · subst hj; simp [hi]
    · simp [hj]
  · intro j hj; simp [release] at hj
    by_cases hji : j = i
    · subst hji; exact getD_set_eq hlt _
    · simp only [release]; rw [getD_set_ne _ hji]
      rcases hj with rfl | hj
      · exact absurd rfl hji
      · exact hI.holes j hj
  · intro j hj; rw [hmemE] at hj; exact getD_set_ne _ hj.1 _

/-- `release_span_values` (:522): release every live heap entry of the span. -/
def releaseSpan (H : Heap) : List Nat → Heap
  | [] => H
  | i :: is => releaseSpan (release H i) is

theorem erase_sublist_refs {refs : List Nat} (hnd : refs.Nodup) {i : Nat} {xs : List Nat}
    (h : ∀ x ∈ i :: xs, x ∈ refs) (hx : (i :: xs).Nodup) : ∀ x ∈ xs, x ∈ refs.erase i := by
  intro x hxm
  rw [List.Nodup.mem_erase_iff hnd]
  exact ⟨fun e => (List.nodup_cons.1 hx).1 (e ▸ hxm), h x (by simp [hxm])⟩

/-- **`releaseSpan_hinv`**: releasing a span's (distinct) heap entries keeps
`HInv` with those references dropped and the rest untouched. -/
theorem releaseSpan_hinv : ∀ (span : List Nat) (H : Heap) (refs : List Nat), HInv H refs →
    (∀ x ∈ span, x ∈ refs) → span.Nodup →
    HInv (releaseSpan H span) (refs.filter (· ∉ span)) ∧
      ∀ j ∈ refs, j ∉ span → (releaseSpan H span).heap.getD j .null = H.heap.getD j .null
  | [], H, refs, hI, _, _ => by
    refine ⟨?_, fun _ _ _ => rfl⟩
    simp only [releaseSpan]; rw [List.filter_eq_self.2 (by simp)]; exact hI
  | i :: is, H, refs, hI, hs, hn => by
    have hi : i ∈ refs := hs i (by simp)
    obtain ⟨h1, h2⟩ := release_hinv H refs hI i hi
    obtain ⟨h3, h4⟩ := releaseSpan_hinv is (release H i) (refs.erase i) h1
      (erase_sublist_refs hI.refs_nodup hs hn) (List.nodup_cons.1 hn).2
    have hf : (refs.erase i).filter (· ∉ is) = refs.filter (· ∉ i :: is) := by
      rw [List.Nodup.erase_eq_filter hI.refs_nodup, List.filter_filter]
      congr 1; funext x; simp; by_cases hx : x = i <;> simp [hx]
    refine ⟨by simp only [releaseSpan]; rw [← hf]; exact h3, ?_⟩
    intro j hj hjn
    simp only [List.mem_cons, not_or] at hjn
    have hje : j ∈ refs.erase i := by rw [List.Nodup.mem_erase_iff hI.refs_nodup]; exact ⟨hjn.1, hj⟩
    simp only [releaseSpan]; rw [h4 j hje hjn.2, h2 j hje]

/-- `store_packed_value` (:471): write at `index`, appending when it is `len`. -/
def store (arena : List PA) (index : Nat) (e : PA) : List PA :=
  if index = arena.length then arena ++ [e] else arena.set index e

theorem store_get (arena : List PA) (index : Nat) (e : PA) (h : index ≤ arena.length) (j : Nat) :
    (store arena index e)[j]? = (if j = index then some e else arena[j]?) ∧
      (store arena index e).length = max arena.length (index + 1) := by
  unfold store
  split
  · rename_i he; subst he
    refine ⟨?_, by simp <;> omega⟩
    by_cases hj : j = arena.length
    · subst hj; simp
    · simp only [hj, ite_false]
      rw [List.getElem?_append]
      split
      · rfl
      · rename_i hge
        have : j - arena.length ≠ 0 := by omega
        simp [List.getElem?_singleton, this, List.getElem?_eq_none (by omega : arena.length ≤ j)]
  · rename_i hne
    refine ⟨?_, by simp <;> omega⟩
    by_cases hj : j = index
    · subst hj; simp [List.getElem?_set, show j < arena.length by omega]
    · simp [hj, List.getElem?_set_ne (Ne.symm hj)]

/-! ## `compact`'s heap remap (:806-816) -/

/-- Entries are `(tag, value index)` pairs after unpacking the payload. Walk
the live entries in order; a heap entry takes `heap[idx]` (leaving `Null`
behind, `mem::replace`) and is renumbered to the next slot of the new heap. -/
def remap : List V → List V → List (Tag × Nat) → List V × List V × List (Tag × Nat)
  | old, nh, [] => (old, nh, [])
  | old, nh, (t, i) :: es =>
    if t = .heap then
      let r := remap (old.set i .null) (nh ++ [old.getD i .null]) es
      (r.1, r.2.1, (t, nh.length) :: r.2.2)
    else
      let r := remap old nh es
      (r.1, r.2.1, (t, i) :: r.2.2)

def heapRefs (es : List (Tag × Nat)) : List Nat := (es.filter (·.1 = .heap)).map (·.2)

theorem getD_append_left' {l : List V} {j : Nat} (h : j < l.length) (l' : List V) :
    (l ++ l').getD j .null = l.getD j .null := by
  simp [List.getD_eq_getElem?_getD, List.getElem?_append_left h]

theorem remap_prefix : ∀ (es : List (Tag × Nat)) (old nh : List V) (j : Nat), j < nh.length →
    (remap old nh es).2.1.getD j .null = nh.getD j .null
  | [], _, _, _, _ => rfl
  | (t, i) :: es, old, nh, j, hj => by
    simp only [remap]
    split
    · rw [remap_prefix es _ _ j (by simp; omega), getD_append_left' hj]
    · exact remap_prefix es _ _ j hj

def All2 {α β : Type} (P : α → β → Prop) : List α → List β → Prop
  | [], [] => True
  | a :: as, b :: bs => P a b ∧ All2 P as bs
  | _, _ => False

/-- **`compactHeap_unpack`**: after the remap, each live heap entry's new
index points at the value its old index held (`f`, the heap before compaction),
provided live entries reference distinct heap slots; scalar entries keep their
payload. -/
theorem compactHeap_unpack (f : Nat → V) : ∀ (es : List (Tag × Nat)) (old nh : List V),
    (heapRefs es).Nodup → (∀ a ∈ es, a.1 = .heap → old.getD a.2 .null = f a.2) →
    All2 (fun (a b : Tag × Nat) => a.1 = b.1 ∧
        (a.1 = .heap → (remap old nh es).2.1.getD b.2 .null = f a.2) ∧
        (a.1 ≠ .heap → b.2 = a.2))
      es (remap old nh es).2.2
  | [], _, _, _, _ => by simp [remap, All2]
  | (t, i) :: es, old, nh, hnd, hf => by
    by_cases ht : t = .heap
    · subst ht
      have hnd' : (i :: heapRefs es).Nodup := by simpa [heapRefs] using hnd
      have hf' : ∀ a ∈ es, a.1 = .heap → (old.set i .null).getD a.2 .null = f a.2 := by
        intro a ha hh
        have hai : a.2 ≠ i := by
          intro e
          apply (List.nodup_cons.1 hnd').1
          rw [← e]; simp only [heapRefs, List.mem_map, List.mem_filter]
          exact ⟨a, ⟨ha, by simp [hh]⟩, rfl⟩
        rw [getD_set_ne _ hai]; exact hf a (by simp [ha]) hh
      have hrest := compactHeap_unpack f es (old.set i .null) (nh ++ [old.getD i .null])
        (List.nodup_cons.1 hnd').2 hf'
      simp only [remap, ite_true]
      refine ⟨⟨rfl, fun _ => ?_, fun h => absurd rfl h⟩, hrest⟩
      rw [remap_prefix es _ _ nh.length (by simp)]
      simp [List.getD_eq_getElem?_getD]
      exact hf (.heap, i) (by simp) rfl
    · have hnd' : (heapRefs es).Nodup := by simpa [heapRefs, ht] using hnd
      have hrest := compactHeap_unpack f es old nh hnd' (fun a ha hh => hf a (by simp [ha]) hh)
      simp only [remap, ht, ite_false]
      exact ⟨⟨rfl, fun h => absurd h ht, fun _ => rfl⟩, hrest⟩

/-- `SpanRef::heap_bytes` (:883): `len * 12` plus, per heap entry,
`size_of::<Value>() + amortized_heap_size()` (the latter an opaque function). -/
def heapBytes (sizeV : Nat) (amort : V → Nat) (H : Heap) (es : List (Tag × Nat)) : Nat :=
  es.foldl (fun b e => if e.1 = .heap then b + sizeV + amort (H.heap.getD e.2 .null) else b) (es.length * 12)

def heapCost (sizeV : Nat) (amort : V → Nat) (H : Heap) (es : List (Tag × Nat)) : Nat :=
  ((es.filter (·.1 = .heap)).map (fun e => sizeV + amort (H.heap.getD e.2 .null))).sum

theorem foldl_cost (sizeV : Nat) (amort : V → Nat) (H : Heap) : ∀ (es : List (Tag × Nat)) (b : Nat),
    es.foldl (fun b e => if e.1 = .heap then b + sizeV + amort (H.heap.getD e.2 .null) else b) b =
      b + heapCost sizeV amort H es
  | [], b => by simp [heapCost]
  | e :: es, b => by
    simp only [List.foldl_cons]
    rw [foldl_cost sizeV amort H es]
    by_cases h : e.1 = .heap
    · simp [heapCost, h, List.filter_cons]; omega
    · simp [heapCost, h, List.filter_cons]

/-- **`heapBytes_eq`**: an entity's bytes are its 12-byte entries plus, for each
heap entry, one `Value` slot and the value's amortized payload. -/
theorem heapBytes_eq (sizeV : Nat) (amort : V → Nat) (H : Heap) (es : List (Tag × Nat)) :
    heapBytes sizeV amort H es = es.length * 12 + heapCost sizeV amort H es :=
  foldl_cost sizeV amort H es _

end PendingCommit.Pack
