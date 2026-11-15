import PendingCommit.ContentCompact
/-
# `DataBlock`: the two-level radix directory (attribute_store.rs:896-1145)

`root[bi / 64] → DirPage.blocks[bi % 64] → Block`, `bi = id >> 6`, slot
`id & 63`. Copy-on-write (`Arc::make_mut`) gives *value* semantics: a writer's
`block_mut` never changes what another version reads. The model is a pure
value, so that is built in; what is proved is the addressing and the frame.

* `locate_eq` — `locate` (:933) is `(id / 64, id % 64)` and inverts.
* `blockAt_set` — after `block_mut` (:962) writes block `bi`, block `bi` is
  the new one and every other block reads as before (pages stay 64 wide).
* `dsetSpan_ok`, `dremove_ok`, `dmergeSpan_ok` — `DataBlock::set_span`
  (:991), `remove` (:1072), `merge_span` (:1008): the entity reads the new
  span, every other entity reads what it did, `SInv` (all blocks `CInv`) holds;
  `merge_span`'s early return (no span, all nulls) is a no-op with counts
  `(0, 0)`, the same as running the merge.
* `dget_eq` — `DataBlock::get` (:976) is `some span` iff the span is nonempty.
-/
namespace PendingCommit.Content
open Arena (Slot)

abbrev Page := List (Option CB)
structure D where
  root : List (Option Page)

def emptyB : CB := ⟨[], [], 0, 0⟩

/-- `DataBlock::locate` (:933): shift / mask with `BLOCK_SHIFT = 6`. -/
def locate (id : Nat) : Nat × Nat := (id >>> 6, id &&& 63)

theorem locate_eq (id : Nat) : locate id = (id / 64, id % 64) ∧ id = 64 * (id / 64) + id % 64 := by
  refine ⟨?_, by omega⟩
  simp only [locate, Nat.shiftRight_eq_div_pow]
  have : id &&& 63 = id % 64 := Nat.and_two_pow_sub_one_eq_mod id 6
  rw [this]

/-- `DataBlock::block` (:941). -/
def blockAt (d : D) (bi : Nat) : Option CB :=
  match d.root.getD (bi / 64) none with
  | some pg => pg.getD (bi % 64) none
  | none => none

/-- `block_mut` (:962) followed by a write of the whole block: grow `root`,
create the page / block on demand, store `B`. -/
def blockSet (d : D) (bi : Nat) (B : CB) : D :=
  let di := bi / 64
  let root := d.root ++ List.replicate (di + 1 - d.root.length) none
  let pg := (root.getD di none).getD (List.replicate 64 none)
  ⟨root.set di (some (pg.set (bi % 64) (some B)))⟩

/-- What `block_mut` hands out: the block, or a fresh default one. -/
def blockOf (d : D) (bi : Nat) : CB := (blockAt d bi).getD emptyB

def DInv (d : D) : Prop := ∀ pg, some pg ∈ d.root → pg.length = 64

theorem getD_pad {α} (l : List (Option α)) (n j : Nat) :
    (l ++ List.replicate n none).getD j none = l.getD j none := by
  simp only [List.getD_eq_getElem?_getD]
  rw [List.getElem?_append]
  split
  · rfl
  · rw [List.getElem?_eq_none (by omega : l.length ≤ j)]; simp [List.getElem?_replicate]; split <;> rfl

theorem rep_none {α} (n k : Nat) : ((List.replicate n (none : Option α))[k]?).getD none = none := by
  rw [List.getElem?_replicate]; split <;> rfl

theorem pageOf_eq (d : D) (di : Nat) :
    ((d.root ++ List.replicate (di + 1 - d.root.length) none).getD di none).getD (List.replicate 64 none) =
      (d.root.getD di none).getD (List.replicate 64 none) := by rw [getD_pad]

theorem blockAt_set (d : D) (h : DInv d) (bi bj : Nat) (B : CB) :
    DInv (blockSet d bi B) ∧ blockAt (blockSet d bi B) bj = if bj = bi then some B else blockAt d bj := by
  have hdi : bi / 64 < (d.root ++ List.replicate (bi / 64 + 1 - d.root.length) none).length := by simp; omega
  have hpg : ((d.root.getD (bi / 64) none).getD (List.replicate 64 none)).length = 64 := by
    cases e : d.root.getD (bi / 64) none with
    | none => simp
    | some pg =>
      simp only [Option.getD_some]; apply h
      have : bi / 64 < d.root.length := by
        rcases Nat.lt_or_ge (bi / 64) d.root.length with h1 | h1
        · exact h1
        · simp [List.getD_eq_getElem?_getD, List.getElem?_eq_none h1] at e
      rw [← e]; simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem this]
  simp only [blockSet, pageOf_eq]
  generalize hP : (d.root.getD (bi / 64) none).getD (List.replicate 64 none) = P at hpg
  generalize hR : d.root ++ List.replicate (bi / 64 + 1 - d.root.length) none = R at hdi
  have hRget : ∀ j, R.getD j none = d.root.getD j none := fun j => by rw [← hR, getD_pad]
  constructor
  · intro pg hm
    rcases List.mem_or_eq_of_mem_set hm with hm | hm
    · rw [← hR] at hm
      rcases List.mem_append.1 hm with hm | hm
      · exact h pg hm
      · simp [List.mem_replicate] at hm
    · cases hm; simp only [List.length_set]; exact hpg
  · simp only [blockAt]
    by_cases hd : bj / 64 = bi / 64
    · rw [hd]
      simp only [List.getD_eq_getElem?_getD, List.getElem?_set_self hdi, Option.getD_some]
      by_cases hs : bj = bi
      · subst hs; simp [List.getElem?_set_self (by omega : bj % 64 < P.length)]
      · have hm : bj % 64 ≠ bi % 64 := by omega
        rw [if_neg hs, List.getElem?_set_ne (Ne.symm hm)]
        rw [← hP]
        simp only [List.getD_eq_getElem?_getD] at hRget ⊢
        rw [← hd]
        cases e : d.root[bj / 64]? with
        | none => simp only [Option.getD_none]; exact rep_none _ _
        | some x => cases x with
          | none => simp only [Option.getD_some, Option.getD_none]; exact rep_none _ _
          | some pg => rfl
    · have hs : bj ≠ bi := fun e => hd (by rw [e])
      rw [if_neg hs]
      simp only [List.getD_eq_getElem?_getD, List.getElem?_set_ne (Ne.symm hd)]
      have := hRget (bj / 64)
      simp only [List.getD_eq_getElem?_getD] at this
      rw [this]

/-! ## Entity-level operations -/

def dslot (d : D) (id : Nat) : Slot := slotAt (blockOf d (id / 64)) (id % 64)
def dread (d : D) (id : Nat) : List Entry := span (blockOf d (id / 64)) (id % 64)

/-- `DataBlock::get` (:976): `None` for a missing block, a missing slot, or `cap == 0`. -/
def dget (d : D) (id : Nat) : Option (List Entry) :=
  if (dslot d id).cap ≠ 0 then some (dread d id) else none

/-- `DataBlock::remove` (:1072): occupancy check first (no COW for an empty slot). -/
def dremove (d : D) (id : Nat) : D :=
  if (dslot d id).cap ≠ 0 then
    blockSet d (id / 64) (maybeCompact (freeSpanC (blockOf d (id / 64)) (id % 64)))
  else d

/-- `DataBlock::set_span` (:991). -/
def dsetSpan (d : D) (id : Nat) (pairs : List Entry) : D :=
  if pairs = [] then dremove d id
  else blockSet d (id / 64) (maybeCompact (setSpanC (blockOf d (id / 64)) (id % 64) pairs))

/-- `DataBlock::merge_span` (:1008). -/
def dmerge (d : D) (id : Nat) (pairs : List Entry) : D × Nat × Nat :=
  if (dslot d id).cap = 0 ∧ allNull pairs = true then (d, 0, 0)
  else
    let r := mergeSpanC (blockOf d (id / 64)) (id % 64) pairs
    (blockSet d (id / 64) (maybeCompact r.1), r.2)

def SInv (d : D) : Prop := DInv d ∧ ∀ bi b, blockAt d bi = some b → CInv b

theorem emptyB_inv : CInv emptyB :=
  ⟨⟨by simp [toBlk, emptyB, Arena.sumBy], by simp [toBlk, emptyB, Arena.sumBy], by simp [toBlk, emptyB],
    by simp [toBlk, emptyB]⟩,
   fun i j _ => by left; simp [slotAt, emptyB, Slot.empty]⟩

theorem blockOf_inv (d : D) (h : SInv d) (bi : Nat) : CInv (blockOf d bi) := by
  unfold blockOf
  cases e : blockAt d bi with
  | none => exact emptyB_inv
  | some b => exact h.2 bi b e

theorem dget_eq (d : D) (h : SInv d) (id : Nat) : dget d id = if dread d id = [] then none else some (dread d id) := by
  have := nonempty_iff_cap _ (blockOf_inv d h (id / 64)) (id % 64)
  unfold dget; simp only [dslot, dread] at *
  by_cases hc : (slotAt (blockOf d (id / 64)) (id % 64)).cap = 0
  · have : span (blockOf d (id / 64)) (id % 64) = [] := Classical.byContradiction fun hn => (this.1 hn) hc
    simp [hc, this]
  · have : span (blockOf d (id / 64)) (id % 64) ≠ [] := this.2 hc
    simp [hc, this]

/-- Writing block `id / 64` with a block that changes only slot `id % 64`. -/
theorem write_frame (d : D) (h : SInv d) (id : Nat) (B : CB) (hB : CInv B)
    (hother : ∀ j, j ≠ id % 64 → span B j = span (blockOf d (id / 64)) j) :
    SInv (blockSet d (id / 64) B) ∧ dread (blockSet d (id / 64) B) id = span B (id % 64) ∧
      ∀ id', id' ≠ id → dread (blockSet d (id / 64) B) id' = dread d id' := by
  have hs := blockAt_set d h.1 (id / 64)
  refine ⟨⟨(hs 0 B).1, fun bj b e => ?_⟩, ?_, ?_⟩
  · rw [(hs bj B).2] at e
    split at e
    · cases e; exact hB
    · exact h.2 bj b e
  · simp [dread, blockOf, (hs (id / 64) B).2]
  · intro id' hne
    simp only [dread, blockOf, (hs (id' / 64) B).2]
    by_cases hb : id' / 64 = id / 64
    · rw [if_pos hb]; simp only [Option.getD_some]
      rw [hb]; apply hother; omega
    · rw [if_neg hb]

/-- **`dremove_ok`**. -/
theorem dremove_ok (d : D) (h : SInv d) (id : Nat) :
    SInv (dremove d id) ∧ dread (dremove d id) id = [] ∧ ∀ id', id' ≠ id → dread (dremove d id) id' = dread d id' := by
  have hb := blockOf_inv d h (id / 64)
  unfold dremove
  split
  · rename_i hc
    have hi : id % 64 < (blockOf d (id / 64)).slots.length := by
      unfold dslot at hc
      rcases Nat.lt_or_ge (id % 64) (blockOf d (id / 64)).slots.length with h1 | h1
      · exact h1
      · simp [slotAt, List.getD_eq_getElem?_getD, List.getElem?_eq_none h1, Slot.empty] at hc
    obtain ⟨f1, f2, f3, -⟩ := freeSpanC_ok _ hb (id % 64) hi
    obtain ⟨m1, m2⟩ := maybeCompact_ok _ f1
    obtain ⟨w1, w2, w3⟩ := write_frame d h id _ m1 (fun j hj => by rw [m2 j, f3 j hj])
    exact ⟨w1, by rw [w2, m2, f2], w3⟩
  · rename_i hc
    refine ⟨h, ?_, fun _ _ => rfl⟩
    have := nonempty_iff_cap _ hb (id % 64)
    unfold dslot at hc
    exact Classical.byContradiction fun hn => hc (this.1 hn)

/-- **`dsetSpan_ok`**: after `DataBlock::set_span`, `id` reads `pairs` and
nothing else changes. -/
theorem dsetSpan_ok (d : D) (h : SInv d) (id : Nat) (pairs : List Entry) (hn : pairs.length ≤ 65535) :
    SInv (dsetSpan d id pairs) ∧ dread (dsetSpan d id pairs) id = pairs ∧
      ∀ id', id' ≠ id → dread (dsetSpan d id pairs) id' = dread d id' := by
  unfold dsetSpan
  split
  · rename_i hp; subst hp; exact dremove_ok d h id
  · have hb := blockOf_inv d h (id / 64)
    obtain ⟨s1, s2, s3⟩ := setSpanC_ok _ hb (id % 64) pairs hn
    obtain ⟨m1, m2⟩ := maybeCompact_ok _ s1
    obtain ⟨w1, w2, w3⟩ := write_frame d h id _ m1 (fun j hj => by rw [m2 j, s3 j hj])
    exact ⟨w1, by rw [w2, m2, s2], w3⟩

theorem mergeGen_allNull : ∀ (ps : List Entry), allNull ps = true → mergeGen [] ps = ([], 0, 0)
  | [], _ => by simp [mergeGen]
  | p :: ps, h => by
    simp only [allNull, List.all_cons, Bool.and_eq_true] at h
    simp only [mergeGen, h.1, ite_true]; exact mergeGen_allNull ps h.2

/-- **`dmergeSpan_ok`**: `DataBlock::merge_span` makes `id` read
`Span.mergeSpan (old span) pairs`, returns its counts, and changes nothing else
(including the early return). -/
theorem dmergeSpan_ok (d : D) (h : SInv d) (id : Nat) (pairs : List Entry)
    (hn : (mergeSpan (dread d id) pairs).1.length ≤ 65535) :
    let r := dmerge d id pairs
    SInv r.1 ∧ dread r.1 id = (mergeSpan (dread d id) pairs).1 ∧ r.2 = (mergeSpan (dread d id) pairs).2 ∧
      ∀ id', id' ≠ id → dread r.1 id' = dread d id' := by
  have hb := blockOf_inv d h (id / 64)
  unfold dmerge
  split
  · rename_i hc
    have hnil : dread d id = [] := by
      have := nonempty_iff_cap _ hb (id % 64)
      exact Classical.byContradiction fun hn => (this.1 hn) hc.1
    have hm : mergeSpan [] pairs = ([], 0, 0) := by
      simp [mergeSpan, mergeGen_allNull pairs hc.2]
    rw [hnil, hm]
    exact ⟨h, hnil, rfl, fun _ _ => rfl⟩
  · obtain ⟨s1, s2, s3, s4⟩ := mergeSpanC_ok _ hb (id % 64) pairs hn
    obtain ⟨m1, m2⟩ := maybeCompact_ok _ s1
    obtain ⟨w1, w2, w3⟩ := write_frame d h id _ m1 (fun j hj => by rw [m2 j, s4 j hj])
    exact ⟨w1, by rw [w2, m2, s2]; rfl, s3, w3⟩

end PendingCommit.Content
