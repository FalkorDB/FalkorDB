import PendingCommit.StoreApi
import PendingCommit.Pack
/-
# RDB codec, `trim`, memory accounting (attribute_store.rs)

* `encode_spec` — `encode_with_range` (:1409) writes, for the entities
  `0..=max_id` not in `deleted`, skipping the first `offset`, the next
  `max(count, 1)` (the `break` follows the write; every caller passes
  `count > 0`, src/serializers/encoder/mod.rs), each as `(id, len,
  entries)`.
* `decode_spec` — `decode_with_count` (:1474) stores, per entity, the entries
  whose wire id fits `u16` (`try_from`, not `as`) and the dictionary, minus
  nulls, sorted by id; nothing for an entity left empty.
* `decodeChecked_spec`, `encode_decode_roundtrip_checked`, `decodeChecked_refuses_u16_wrap` —
  #3643's refusals (entity id past `id_bound`, more than 65,535 attributes): exact, passed by
  every encoded window, and closing the `set_span` wrap on load.
* **`encode_decode_roundtrip`** — decoding what was encoded into an empty
  store restores exactly the encoded entities' spans (sorted null-free spans,
  ids below the dictionary size).
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

/-! ## The refusals of `decode_with_count` (#3643)

Before decoding an entity, `decode_with_count` refuses (`Err`, `none` here) an entity id at or past
the store's `id_bound` (set by `AttributeStore::with_id_bound` on load) and an attribute count past
`MAX_ATTRIBUTES` (65,535), which `set_span`'s `u16` length could not hold. (`check_count_fits` only
refuses a `count` the remaining bytes cannot hold, which the reads would run out on anyway.) Otherwise
the result is `decode`. -/

def MAX_ATTRIBUTES : Nat := 65535

def admits (bound : Option Nat) (p : Nat × List Entry) : Bool :=
  (match bound with | some b => decide (p.1 < b) | none => true) && decide (p.2.length ≤ MAX_ATTRIBUTES)

def decodeChecked (d : D) (bound : Option Nat) (limit : Nat) (ents : List (Nat × List Entry)) : Option D :=
  if ents.all (admits bound) then some (decode d limit ents) else none

/-- **`decodeChecked_spec`**: an accepted stream decodes exactly as `decode_spec` says; the
refusal is exact. -/
theorem decodeChecked_spec (d : D) (h : SInv d) (bound : Option Nat) (limit : Nat)
    (m : List (Nat × List Entry)) (hnd : (m.map (·.1)).Nodup) :
    (decodeChecked d bound limit m = none ↔ ∃ p ∈ m, admits bound p = false) ∧
    ∀ d', decodeChecked d bound limit m = some d' →
      SInv d' ∧ ∀ id, dread d' id = match lk m id with
        | some ps => (if ps.filter (keep limit) = [] then dread d id else sortById (ps.filter (keep limit)))
        | none => dread d id := by
  refine ⟨?_, fun d' hd => ?_⟩
  · unfold decodeChecked
    by_cases ha : m.all (admits bound) = true
    · simp only [ha, ite_true, reduceCtorEq, false_iff, not_exists, not_and]
      intro p hp hf
      have := List.all_eq_true.1 ha p hp
      simp_all
    · simp only [ha, Bool.false_eq_true, ite_false, true_iff]
      simpa using ha
  · unfold decodeChecked at hd
    split at hd
    · rename_i ha
      cases hd
      exact decode_spec d h limit m hnd (by
        intro p hp
        have := List.all_eq_true.1 ha p hp
        simp [admits, MAX_ATTRIBUTES] at this; exact of_decide_eq_true this.2)
    · cases hd

/-- **`encode_decode_roundtrip_checked`**: with the bound the loader sets (`live + deleted`, here
any `bound > maxId`), what `encode_with_range` wrote passes both checks, so the round trip of
`encode_decode_roundtrip` is what the checked decoder returns. -/
theorem encode_decode_roundtrip_checked (d : D) (h : SInv d) (hs : SOK d) (limit : Nat)
    (hlim : ∀ id, ∀ e ∈ dread d id, e.1 < limit ∧ e.1 < 65536) (hfit : ∀ id, (dread d id).length ≤ 65535)
    (deleted : Nat → Bool) (maxId count offset bound : Nat) (hb : maxId < bound) :
    ∃ d', decodeChecked emptyD (some bound) limit (encode d deleted maxId count offset) = some d' ∧
      ∀ id, dread d' id = if id ∈ window deleted maxId count offset then dread d id else [] := by
  have hall : (encode d deleted maxId count offset).all (admits (some bound)) = true := by
    apply List.all_eq_true.2
    intro p hp
    rw [encode_spec] at hp
    obtain ⟨i, hi, rfl⟩ := List.mem_map.1 hp
    have hi' : i < maxId + 1 := by
      simp only [window] at hi
      have := List.mem_of_mem_take hi
      have := List.mem_of_mem_drop this
      simpa using (List.mem_filter.1 this).1
    simp only [admits, MAX_ATTRIBUTES, Bool.and_eq_true, decide_eq_true_eq]
    exact ⟨by omega, decide_eq_true (hfit i)⟩
  refine ⟨_, ?_, encode_decode_roundtrip d h hs limit hlim hfit deleted maxId count offset⟩
  simp only [decodeChecked, hall, ite_true]

/-- The `set_span` wrap is no longer reachable from a load: 65,536 entries for one entity are
refused. -/
theorem decodeChecked_refuses_u16_wrap (d : D) (bound : Option Nat) (limit id : Nat)
    (ps : List Entry) (hn : ps.length = 65536) : decodeChecked d bound limit [(id, ps)] = none := by
  simp [decodeChecked, admits, MAX_ATTRIBUTES, hn]

end PendingCommit.Content
