import PendingCommit.StoreApi
import PendingCommit.Pack
/-
# RDB codec, `trim`, memory accounting (attribute_store.rs:1038-1145, :1398-1497)

* `encode_spec` — `encode_with_range` (:1409) writes, for the entities
  `0..=max_id` not in `deleted`, skipping the first `offset`, the next
  `max(count, 1)` (the `break` follows the write; every caller passes
  `count > 0`, src/serializers/encoder/mod.rs:139-196), each as `(id, len,
  entries)`.
* `decode_spec` — `decode_with_count` (:1474) stores, per entity, the entries
  whose wire id fits `u16` (`try_from`, not `as`) and the dictionary, minus
  nulls, sorted by id; nothing for an entity left empty.
* **`encode_decode_roundtrip`** — decoding what was encoded into an empty
  store restores exactly the encoded entities' spans (sorted null-free spans,
  ids below the dictionary size).
* #3643 checks (`decode_with_count`, :1474): `decLoop` refuses an entity id at
  or past `id_bound` (`with_id_bound`, :1106) and more than 65535 attributes
  (`decLoop_refuses_id`, `decLoop_refuses_attrs`); `decLoop_eq` (success iff
  every entity passes, then = `decode`), `decLoop_spec`, `decLoop_roundtrip`
  (an encoded window below the bound always passes).
* `decode_unimplemented` — `Decode::decode` (:1450) is `unimplemented!()`.
* `dtrim_ok` — `DataBlock::trim` (:1038) only compacts exclusively-owned
  blocks and preserves every read and `SInv`.
* `structural_partition` — `structural_memory_usage` (:1102, :1398) for a
  block plus its entities' `heap_bytes` is the block's whole allocation, with
  no `usize` underflow: `live_arena = Σ len` (`Arena.live_eq_sum_len`) and
  `live_heap = |refs|` (`heap_len_partition`).
-/
namespace PendingCommit.Content
open Arena (Slot)

/-! ## Encode -/

def window (deleted : Nat → Bool) (maxId count offset : Nat) : List Nat :=
  (((List.range (maxId + 1)).filter (fun id => !deleted id)).drop offset).take (max count 1)

/-- The loop of `encode_with_range`, literally: skip, write, count, break. -/
def encLoop (d : D) (deleted : Nat → Bool) (count : Nat) :
    List Nat → Nat → Nat → List (Nat × List Entry)
  | [], _, _ => []
  | id :: ids, skipped, encoded =>
    if deleted id then encLoop d deleted count ids skipped encoded
    else if skipped > 0 then encLoop d deleted count ids (skipped - 1) encoded
    else
      let e := (id, dread d id)
      if encoded + 1 ≥ count then [e] else e :: encLoop d deleted count ids 0 (encoded + 1)

def encode (d : D) (deleted : Nat → Bool) (maxId count offset : Nat) : List (Nat × List Entry) :=
  encLoop d deleted count (List.range (maxId + 1)) offset 0

theorem encLoop_spec (d : D) (deleted : Nat → Bool) (count : Nat) : ∀ (ids : List Nat) (sk enc : Nat),
    encLoop d deleted count ids sk enc =
      (((ids.filter (fun id => !deleted id)).drop sk).take (max (count - enc) 1)).map (fun id => (id, dread d id))
  | [], _, _ => by simp [encLoop]
  | id :: ids, sk, enc => by
    simp only [encLoop]
    by_cases hd : deleted id
    · simp only [hd, ite_true, List.filter_cons, Bool.not_true, Bool.false_eq_true, ite_false]
      exact encLoop_spec d deleted count ids sk enc
    · simp only [hd, Bool.false_eq_true, ite_false, List.filter_cons, Bool.not_false, ite_true]
      by_cases hs : sk > 0
      · rw [ite_eq_left hs, encLoop_spec d deleted count ids (sk - 1) enc]
        obtain ⟨k, rfl⟩ : ∃ k, sk = k + 1 := ⟨sk - 1, by omega⟩
        simp
      · rw [ite_eq_right hs]
        have : sk = 0 := by omega
        subst this
        by_cases hc : enc + 1 ≥ count
        · rw [ite_eq_left hc]
          have : max (count - enc) 1 = 1 := by omega
          simp [this]
        · rw [ite_eq_right hc, encLoop_spec d deleted count ids 0 (enc + 1)]
          have : max (count - enc) 1 = max (count - (enc + 1)) 1 + 1 := by omega
          simp [this]

/-- **`encode_spec`**. -/
theorem encode_spec (d : D) (deleted : Nat → Bool) (maxId count offset : Nat) :
    encode d deleted maxId count offset = (window deleted maxId count offset).map (fun id => (id, dread d id)) := by
  simp [encode, encLoop_spec, window]

/-! ## Decode -/

/-- The entries `decode_with_count` keeps (`u16::try_from`, `< attr_limit`, not null). -/
def keep (limit : Nat) (e : Entry) : Bool := decide (e.1 < 65536) && decide (e.1 < limit) && !e.2.isNull

def decode (d : D) (limit : Nat) (ents : List (Nat × List Entry)) : D :=
  let items := (ents.map (fun p => (p.1, sortById (p.2.filter (keep limit))))).filter (fun p => !p.2.isEmpty)
  (runOps (fun d id ps => (dsetSpan d id ps, 0, ps.length)) d items).1

/-- `Decode::decode` for the store: `unimplemented!()` — always a panic. -/
def decodePlain : Option D := none
theorem decode_unimplemented : decodePlain = none := rfl

theorem sortById_len (ps : List Entry) : (sortById ps).length = ps.length := (List.mergeSort_perm _ _).length_eq

theorem sortById_nil (ps : List Entry) : sortById ps = [] ↔ ps = [] := by
  constructor
  · intro e; have := sortById_len ps; rw [e] at this; exact List.eq_nil_of_length_eq_zero this.symm
  · intro e; subst e; exact List.eq_nil_of_length_eq_zero (sortById_len [])

/-- **`decode_spec`**. -/
theorem decode_spec (d : D) (h : SInv d) (limit : Nat) (m : List (Nat × List Entry)) (hnd : (m.map (·.1)).Nodup)
    (hfit : ∀ p ∈ m, p.2.length ≤ 65535) :
    SInv (decode d limit m) ∧ ∀ id, dread (decode d limit m) id = match lk m id with
      | some ps => (if ps.filter (keep limit) = [] then dread d id else sortById (ps.filter (keep limit)))
      | none => dread d id := by
  have hnd' := nodup_filter_map m (fun ps => sortById (ps.filter (keep limit))) hnd
  obtain ⟨r1, r2, -⟩ := runOps_spec _ _ _ _ set_spec _ d h hnd' (by
    intro p hp
    simp only [List.mem_filter, List.mem_map] at hp
    obtain ⟨⟨q, hq, rfl⟩, -⟩ := hp
    simp only [sortById_len]; exact Nat.le_trans (List.length_filter_le _ _) (hfit q hq))
  refine ⟨r1, fun id => ?_⟩
  simp only [decode]; rw [r2 id, lk_filter_map m (fun ps => sortById (ps.filter (keep limit))) id hnd]
  cases lk m id with
  | none => simp [after]
  | some ps =>
    by_cases hf : ps.filter (keep limit) = []
    · have h0 := (sortById_nil _).2 hf
      show after _ _ (if sortById (ps.filter (keep limit)) = [] then none else some _) =
        if ps.filter (keep limit) = [] then dread d id else _
      rw [ite_eq_left h0, ite_eq_left hf]; rfl
    · have h0 : ¬ sortById (ps.filter (keep limit)) = [] := fun e => hf ((sortById_nil _).1 e)
      simp [hf, h0, after]

theorem lk_map_ids (d : D) (ids : List Nat) (hnd : ids.Nodup) (id : Nat) :
    lk (ids.map (fun i => (i, dread d i))) id = if id ∈ ids then some (dread d id) else none := by
  induction ids with
  | nil => rfl
  | cons i is ih =>
    simp only [List.nodup_cons] at hnd
    simp only [List.map_cons, lk, ih hnd.2, List.mem_cons]
    by_cases h : id = i
    · subst h; simp
    · simp [h]

theorem window_nodup (deleted : Nat → Bool) (maxId count offset : Nat) : (window deleted maxId count offset).Nodup :=
  ((List.nodup_range.filter _).sublist (List.drop_sublist _ _)).sublist (List.take_sublist _ _)

/-- **`encode_decode_roundtrip`**: decoding the encoded window into an empty
store gives back every encoded entity's span and nothing else. -/
theorem encode_decode_roundtrip (d : D) (h : SInv d) (hs : SOK d) (limit : Nat)
    (hlim : ∀ id, ∀ e ∈ dread d id, e.1 < limit ∧ e.1 < 65536) (hfit : ∀ id, (dread d id).length ≤ 65535)
    (deleted : Nat → Bool) (maxId count offset : Nat) :
    ∀ id, dread (decode emptyD limit (encode d deleted maxId count offset)) id =
      if id ∈ window deleted maxId count offset then dread d id else [] := by
  intro id
  have hw := window_nodup deleted maxId count offset
  have hnd : ((encode d deleted maxId count offset).map (·.1)).Nodup := by
    rw [encode_spec, List.map_map]; simpa [Function.comp_def] using hw
  have := (decode_spec emptyD new_spec.1 limit _ hnd (by
    intro p hp; rw [encode_spec] at hp; simp only [List.mem_map] at hp
    obtain ⟨i, _, rfl⟩ := hp; exact hfit i)).2 id
  rw [this, encode_spec, lk_map_ids d _ hw]
  have he : dread emptyD id = [] := new_spec.2 id
  by_cases hi : id ∈ window deleted maxId count offset
  · simp only [hi, ite_true]
    have hk : (dread d id).filter (keep limit) = dread d id := by
      apply List.filter_eq_self.2
      intro e he'
      have := hlim id e he'; have hn := (hs id).2 e he'
      simp [keep, this.1, this.2, hn]
    have hsorted : sortById (dread d id) = dread d id := by
      apply List.mergeSort_of_pairwise
      exact (hs id).1.imp (fun h => by simp; omega)
    rw [hk, hsorted]
    by_cases hz : dread d id = []
    · simp [hz, he]
    · simp [hz]
  · simp [hi, he]

/-! ## The checked loop (#3643)

`decode_with_count` (attribute_store.rs:1474) as the PR has it: before an entity
is read, `check_count_fits` refuses a count the bytes left cannot hold (each
entity is at least two 9-byte words, so it refuses only what the loop would run
out of anyway; the model's entity list is what was read). Per entity, an id at
or past `id_bound` (`AttributeStore::with_id_bound`, :1106; `None` for
`AttributeStore::new`) and an attribute count above `MAX_ATTRIBUTES` (65535: a
span length is a `u16`) are an `Err`. Otherwise the entries are filtered,
sorted and stored as before. `none` is the `Err`: the restore is abandoned. -/

/-- One entity passes the two new checks. -/
def entOk (bound : Option Nat) (p : Nat × List Entry) : Bool :=
  !(bound.any fun b => decide (b ≤ p.1)) && decide (p.2.length ≤ 65535)

/-- The loop of `decode_with_count`, literally: check the id, check the count,
then store the filtered sorted span when it is nonempty. -/
def decLoop (bound : Option Nat) (limit : Nat) : D → List (Nat × List Entry) → Option D
  | d, [] => some d
  | d, (id, es) :: L =>
    if bound.any (fun b => decide (b ≤ id)) then none
    else if es.length > 65535 then none
    else
      let ps := sortById (es.filter (keep limit))
      decLoop bound limit (if ps.isEmpty then d else dsetSpan d id ps) L

theorem decode_cons (d : D) (limit id : Nat) (es : List Entry) (L : List (Nat × List Entry)) :
    decode d limit ((id, es) :: L) =
      decode (if (sortById (es.filter (keep limit))).isEmpty then d
        else dsetSpan d id (sortById (es.filter (keep limit)))) limit L := by
  by_cases h : (sortById (es.filter (keep limit))).isEmpty
  · simp [decode, h]
  · simp [decode, h, runOps]

/-- **`decLoop_eq`**: the checked loop succeeds iff every entity passes both
checks, and then stores exactly what the unchecked `decode` stores. -/
theorem decLoop_eq (bound : Option Nat) (limit : Nat) :
    ∀ (L : List (Nat × List Entry)) (d : D),
      decLoop bound limit d L = if L.all (entOk bound) then some (decode d limit L) else none
  | [], d => by simp [decLoop, decode, runOps]
  | (id, es) :: L, d => by
    rw [decode_cons]
    simp only [decLoop, List.all_cons, entOk]
    by_cases hb : bound.any (fun b => decide (b ≤ id)) = true
    · simp [hb]
    · by_cases hl : es.length > 65535
      · have : ¬ es.length ≤ 65535 := by omega
        simp [hb, hl, this]
      · have : es.length ≤ 65535 := by omega
        simp only [hb, hl, this, ite_false, Bool.false_eq_true, Bool.not_false, decide_true,
          Bool.and_self, Bool.true_and]
        exact decLoop_eq bound limit L _

/-- **`decLoop_spec`**: `decode_spec` for the PR's loop — on success the
store holds, per entity, its kept entries sorted by id; the `≤ 65535`
hypothesis of `decode_spec` is now checked by the code, not assumed. -/
theorem decLoop_spec (d : D) (h : SInv d) (bound : Option Nat) (limit : Nat)
    (m : List (Nat × List Entry)) (hnd : (m.map (·.1)).Nodup) (d' : D)
    (hr : decLoop bound limit d m = some d') :
    SInv d' ∧ ∀ id, dread d' id = match lk m id with
      | some ps => (if ps.filter (keep limit) = [] then dread d id else sortById (ps.filter (keep limit)))
      | none => dread d id := by
  rw [decLoop_eq] at hr
  by_cases ha : m.all (entOk bound) = true
  · simp only [ha, ite_true, Option.some.injEq] at hr
    subst hr
    exact decode_spec d h limit m hnd (fun p hp => by
      have := List.all_eq_true.1 ha p hp
      simp only [entOk, Bool.and_eq_true, decide_eq_true_eq] at this
      exact this.2)
  · simp [ha] at hr

/-- **`decLoop_refuses_id`**: an entity id at or past the bound is an `Err`,
whatever else the payload holds. -/
theorem decLoop_refuses_id (b limit : Nat) (d : D) (m : List (Nat × List Entry))
    (p : Nat × List Entry) (hp : p ∈ m) (hid : b ≤ p.1) : decLoop (some b) limit d m = none := by
  rw [decLoop_eq]
  have : m.all (entOk (some b)) = false := by
    rw [List.all_eq_false]; exact ⟨p, hp, by simp [entOk, Option.any, hid]⟩
  simp [this]

/-- **`decLoop_refuses_attrs`**: an entity with more than `MAX_ATTRIBUTES`
entries is an `Err` (before #3643 its `u16` span length wrapped and the
entity's attributes were silently lost). -/
theorem decLoop_refuses_attrs (bound : Option Nat) (limit : Nat) (d : D) (m : List (Nat × List Entry))
    (p : Nat × List Entry) (hp : p ∈ m) (hlen : 65535 < p.2.length) : decLoop bound limit d m = none := by
  rw [decLoop_eq]
  have : m.all (entOk bound) = false := by
    rw [List.all_eq_false]; exact ⟨p, hp, by simp [entOk]; omega⟩
  simp [this]

/-- **`decLoop_roundtrip`**: the encoded window passes both checks whenever the
bound exceeds `max_id` (the header's live + deleted count does, see
graph_persist), so decoding it into a fresh bounded store
(`AttributeStore::with_id_bound`) succeeds and restores exactly what
`encode_decode_roundtrip` says. -/
theorem decLoop_roundtrip (d : D) (h : SInv d) (hs : SOK d) (limit : Nat)
    (hlim : ∀ id, ∀ e ∈ dread d id, e.1 < limit ∧ e.1 < 65536) (hfit : ∀ id, (dread d id).length ≤ 65535)
    (deleted : Nat → Bool) (maxId count offset : Nat) (bound : Option Nat)
    (hb : ∀ b, bound = some b → maxId < b) :
    ∃ d', decLoop bound limit emptyD (encode d deleted maxId count offset) = some d' ∧
      ∀ id, dread d' id = if id ∈ window deleted maxId count offset then dread d id else [] := by
  refine ⟨decode emptyD limit (encode d deleted maxId count offset), ?_,
    encode_decode_roundtrip d h hs limit hlim hfit deleted maxId count offset⟩
  rw [decLoop_eq]
  have : (encode d deleted maxId count offset).all (entOk bound) = true := by
    rw [List.all_eq_true]
    intro p hp
    rw [encode_spec] at hp
    obtain ⟨i, hi, rfl⟩ := List.mem_map.1 hp
    have hir : i < maxId + 1 := by
      have h1 := (List.take_sublist _ _).subset hi
      have h2 := (List.drop_sublist _ _).subset h1
      exact List.mem_range.1 (List.mem_filter.1 h2).1
    cases hbd : bound with
    | none => simp [entOk, Option.any, hfit i]
    | some b =>
      have := hb b hbd
      have : ¬ b ≤ i := by omega
      simp [entOk, Option.any, this, hfit i]
  simp [this]

end PendingCommit.Content
