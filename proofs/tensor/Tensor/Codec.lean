/-
# `encode` / `decode` (RDB serialisation, version 19)

`encode` writes the *effective* state in the C-compatible layout: one UINT64
forward matrix whose value is the inline edge id for a single-edge pair and
`(edge_count | MSB)` for a multi-edge pair, followed by empty delta layers,
followed by the tensor section — one BOOL vector *indexed by edge id* per
multi-edge pair.  `decode` reads it back, rebuilding `MULTI` sentinels and `me`.

Proved here:

* `msb_or` / `msb_and_eq_zero` / `msb_and_ne_zero` — the MSB tag is exactly the
  arithmetic `+ 2^63` / `< 2^63` test the model uses (the Rust writes
  `count | MSB` and tests `value & MSB == 0`);
* `edgesAt_decode_encode` — **round-trip**: `decode (encode t)` denotes the same
  multi-graph as `t`, pair by pair;
* `invCore_decode_encode` — the decoded tensor satisfies every invariant except
  the backward-matrix one, which `decode`'s caller restores with
  `rebuild_backward` (`Flush.inv_rebuildBackward` then gives full `Inv`);
* `edgeCount_encode` — the `total` field written to disk is the edge count.
-/
import Mathlib.Data.Nat.Bitwise
import Tensor.Flush
import Tensor.Iter

namespace FalkorDB
namespace Tensor

variable {t : Tensor}

/-! ## The MSB tag -/

/-- `MSB_MASK` of `tensor.rs`. -/
def MSB : Nat := 2 ^ 63

/-- `count | MSB` is `count + MSB` (the Rust's tagging). -/
theorem msb_or {c : Nat} (h : c < MSB) : c ||| MSB = c + MSB := by
  have h1 := Nat.shiftLeft_add_eq_or_of_lt (i := 63) (b := c) (by simpa [MSB] using h) 1
  simp only [MSB, Nat.shiftLeft_eq, Nat.one_mul] at h1 ⊢
  rw [Nat.lor_comm]
  omega

/-- `value & MSB == 0` is exactly `value < 2^63`: an untagged value. -/
theorem msb_and_eq_zero {v : Nat} (h : v < MSB) : v &&& MSB = 0 := by
  rw [MSB, Nat.and_two_pow, Nat.testBit_lt_two_pow (by rw [← MSB]; exact h)]
  simp

/-- …and a tagged one (a `u64`) fails that test. -/
theorem msb_and_ne_zero {v : Nat} (h1 : MSB ≤ v) (h2 : v < 2 ^ 64) : v &&& MSB ≠ 0 := by
  have hb : v.testBit 63 = true :=
    Nat.testBit_of_two_pow_le_and_two_pow_add_one_gt (by simpa [MSB] using h1) (by
      simpa using h2)
  rw [MSB, Nat.and_two_pow, hb]
  simp

/-- A real edge id is never tagged. -/
theorem validId_lt_msb {i : Nat} (h : ValidId i) : i < MSB := by
  have h1 : (2:Nat) ^ 60 < 2 ^ 63 := Nat.pow_lt_pow_right (by omega) (by omega)
  simp only [ValidId, GrBIndexMax] at h
  simp only [MSB]
  omega

/-! ## The serialised form -/

/-- The on-disk image of a tensor (RDB version 19): the three forward layers, the
edge total, and the two tensor groups of per-pair id lists. -/
structure Encoded where
  /-- Forward base: inline id, or `count ||| MSB` for a multi-edge pair. -/
  fwdM : Layer Nat
  /-- Forward delta-plus (always empty: `encode` folds it into the base). -/
  fwdDp : Layer Nat
  /-- Forward delta-minus (always empty). -/
  fwdDm : Finset Pair
  /-- `edge_count`. -/
  total : Nat
  /-- Base tensor group: `(pair, ids)` for every multi-edge pair. -/
  groupBase : List (Pair × Finset Nat)
  /-- Delta-plus tensor group (always empty). -/
  groupDelta : List (Pair × Finset Nat)
  /-- Matrix dimensions. -/
  nrows : Nat
  /-- Matrix dimensions. -/
  ncols : Nat

/-- `Tensor::encode`.  The forward matrix is built from `fwd_iter`, i.e. from the
effective state, so the deltas go out empty. -/
noncomputable def encode (t : Tensor) : Encoded where
  fwdM :=
    { dom := t.effDom
      val := fun p =>
        if t.effGet p = some MULTI then (t.meRow (key p)).card ||| MSB
        else (t.effGet p).getD 0 }
  fwdDp := { dom := ∅, val := fun _ => 0 }
  fwdDm := ∅
  total := edgeCount t
  groupBase := t.multiPairs.toList.map (fun p => (p, t.meRow (key p)))
  groupDelta := []
  nrows := t.nrows
  ncols := t.ncols

/-- Absorb one tensor group into `me` (`me.set(key, edge_id, true)` per id). -/
def loadGroup (me : Finset (Addr × Nat)) (g : List (Pair × Finset Nat)) : Finset (Addr × Nat) :=
  g.foldl (fun s e => s ∪ e.2.image (fun i => (key e.1, i))) me

/-- `Tensor::decode`.  Entries whose `fwd_dm` bit is set are skipped, `fwd_dp`
overrides `fwd_m`, and a tagged value becomes a `MULTI` sentinel whose ids come
from the tensor groups. -/
def decode (e : Encoded) : Tensor where
  m :=
    { dom := (e.fwdM.dom \ e.fwdDm) ∪ e.fwdDp.dom
      val := fun p =>
        let v := if p ∈ e.fwdDp.dom then e.fwdDp.val p else e.fwdM.val p
        if v &&& MSB = 0 then v else MULTI }
  dp := { dom := ∅, val := fun _ => 0 }
  dm := ∅
  mt := ∅
  me := loadGroup (loadGroup ∅ e.groupBase) e.groupDelta
  nrows := e.nrows
  ncols := e.ncols

/-! ## Round-trip -/

/-- A pair's id list is far shorter than the MSB tag, so the tag is unambiguous:
every id is a GraphBLAS index, so a row has at most `2^60` of them. -/
theorem meRow_card_lt_msb (h : Inv t) {q : Pair} (hq : t.effGet q = some MULTI) :
    (t.meRow (key q)).card < MSB := by
  have hsub : t.meRow (key q) ⊆ Finset.range (2 ^ 60) := by
    intro i hi
    have hvi : ValidId i := h.valid_ids q i (by rw [edgesAt_of_multi hq]; exact hi)
    simp only [ValidId, GrBIndexMax] at hvi
    simp only [Finset.mem_range]
    omega
  have hc := Finset.card_le_card hsub
  rw [Finset.card_range] at hc
  have h63 : (2:Nat) ^ 60 < 2 ^ 63 := Nat.pow_lt_pow_right (by omega) (by omega)
  simp only [MSB]
  omega

/-! ### Loading the tensor groups -/

theorem mem_loadGroup {me : Finset (Addr × Nat)} {g : List (Pair × Finset Nat)}
    {x : Addr × Nat} :
    x ∈ loadGroup me g ↔ x ∈ me ∨ ∃ e ∈ g, ∃ i ∈ e.2, x = (key e.1, i) := by
  induction g generalizing me with
  | nil => simp [loadGroup]
  | cons e g ih =>
    rw [show loadGroup me (e :: g) = loadGroup (me ∪ e.2.image (fun i => (key e.1, i))) g from rfl,
      ih]
    simp only [Finset.mem_union, Finset.mem_image, List.mem_cons]
    constructor
    · rintro ((hx | ⟨i, hi, rfl⟩) | ⟨e', he', hi⟩)
      · exact Or.inl hx
      · exact Or.inr ⟨e, Or.inl rfl, i, hi, rfl⟩
      · exact Or.inr ⟨e', Or.inr he', hi⟩
    · rintro (hx | ⟨e', he' | he', i, hi, rfl⟩)
      · exact Or.inl (Or.inl hx)
      · subst he'; exact Or.inl (Or.inr ⟨i, hi, rfl⟩)
      · exact Or.inr ⟨e', he', i, hi, rfl⟩

section RoundTrip

variable {q : Pair}

/-- The `me` set the decoder rebuilds holds exactly the ids of the multi-edge
pairs that were written. -/
theorem mem_me_decode_encode {x : Addr × Nat} :
    x ∈ (decode (encode t)).me ↔ ∃ p ∈ t.multiPairs, x = (key p, x.2) ∧ x.2 ∈ t.meRow (key p) := by
  show x ∈ loadGroup (loadGroup ∅ (encode t).groupBase) (encode t).groupDelta ↔ _
  rw [show (encode t).groupDelta = [] from rfl, show loadGroup (loadGroup ∅ (encode t).groupBase) []
    = loadGroup ∅ (encode t).groupBase from rfl, mem_loadGroup]
  simp only [Finset.notMem_empty, false_or, encode, List.mem_map, Finset.mem_toList]
  constructor
  · rintro ⟨e, ⟨p, hp, rfl⟩, i, hi, rfl⟩
    exact ⟨p, hp, rfl, hi⟩
  · rintro ⟨p, hp, hx, hi⟩
    exact ⟨(p, t.meRow (key p)), ⟨p, hp, rfl⟩, x.2, hi, hx⟩

/-- The decoded `me` row of a multi-edge pair is the original row. -/
theorem meRow_decode_encode_multi (h : Inv t) (hq : q ∈ t.multiPairs) :
    (decode (encode t)).meRow (key q) = t.meRow (key q) := by
  ext i
  rw [mem_meRow, mem_me_decode_encode]
  constructor
  · rintro ⟨p, hp, hx, hi⟩
    have hkey : key p = key q := by
      have := congrArg (fun (y : Addr × Nat) => y.1) hx
      simpa using this.symm
    have hpq : p = q :=
      key_inj hkey
    subst hpq
    exact hi
  · intro hi
    exact ⟨q, hq, rfl, hi⟩

/-- …and a single-edge (or absent) pair gets no row at all. -/
theorem meRow_decode_encode_not_multi (h : Inv t) (hq : q ∉ t.multiPairs) :
    (decode (encode t)).meRow (key q) = ∅ := by
  apply Finset.eq_empty_of_forall_notMem
  intro i hi
  obtain ⟨p, hp, hx, _⟩ := mem_me_decode_encode |>.mp (mem_meRow.mp hi)
  have hkey : key p = key q := by
    have := congrArg (fun (y : Addr × Nat) => y.1) hx
    simpa using this.symm
  have hpq : p = q :=
    key_inj hkey
  exact hq (hpq ▸ hp)

/-- The decoded forward value: single ids survive, sentinels are restored from
the MSB tag. -/
theorem effGet_decode_encode (h : Inv t) :
    (decode (encode t)).effGet q = t.effGet q := by
  have hdp : (decode (encode t)).dp.get q = none := by simp [decode, Layer.get]
  have hdm : q ∉ (decode (encode t)).dm := by simp [decode]
  rw [effGet_of_m hdp hdm]
  by_cases hq : q ∈ t.effDom
  · obtain ⟨v, hv⟩ := Option.isSome_iff_exists.mp (mem_effDom_iff_isSome.mp hq)
    by_cases hM : v = MULTI
    · subst hM
      have hcard := meRow_card_lt_msb h hv
      have htag : (t.meRow (key q)).card ||| MSB = (t.meRow (key q)).card + MSB :=
        msb_or hcard
      have hlt : (t.meRow (key q)).card + MSB < 2 ^ 64 := by
        have h64 : (2:Nat) ^ 64 = 2 ^ 63 * 2 := by rw [← Nat.pow_succ]
        simp only [MSB] at hcard ⊢
        omega
      have hne : ((t.meRow (key q)).card ||| MSB) &&& MSB ≠ 0 := by
        rw [htag]
        exact msb_and_ne_zero (Nat.le_add_left _ _) hlt
      simp [decode, Layer.get, encode, hq, hv, hne]
    · have hvalid : ValidId v := h.valid_ids q v (by rw [edgesAt_of_single hv hM]; simp)
      have hz : v &&& MSB = 0 := msb_and_eq_zero (validId_lt_msb hvalid)
      simp [decode, Layer.get, encode, hq, hv, hM, hz]
  · have h1 : (decode (encode t)).m.get q = none := by
      apply Layer.get_eq_none.mpr
      simp only [decode, encode, Finset.mem_union, Finset.mem_sdiff, Finset.notMem_empty,
        or_false, not_and]
      exact fun hc => absurd hc hq
    rw [h1, effGet_eq_none_iff.mpr hq]

/-- **The RDB round-trip preserves the graph**: every pair keeps exactly its
edges. -/
theorem edgesAt_decode_encode (h : Inv t) : (decode (encode t)).edgesAt q = t.edgesAt q := by
  have hget := effGet_decode_encode (t := t) (q := q) h
  cases hv : t.effGet q with
  | none => rw [edgesAt_of_none (by rw [hget, hv]), edgesAt_of_none hv]
  | some v =>
    by_cases hM : v = MULTI
    · subst hM
      have hq : q ∈ t.multiPairs :=
        mem_multiPairs.mpr ⟨mem_effDom_iff_isSome.mpr (by rw [hv]; rfl), hv⟩
      rw [edgesAt_of_multi (by rw [hget, hv]), edgesAt_of_multi hv,
        meRow_decode_encode_multi h hq]
    · rw [edgesAt_of_single (by rw [hget, hv]) hM, edgesAt_of_single hv hM]

/-- **`decode` restores every invariant except the backward matrix**, which it
deliberately leaves empty for `rebuild_backward`. -/
theorem invCore_decode_encode (h : Inv t) : InvCore (decode (encode t)) := by
  have hdom : (decode (encode t)).effDom = t.effDom := by
    ext r
    rw [mem_effDom_iff_isSome, mem_effDom_iff_isSome, effGet_decode_encode h]
  refine { dm_sub_m := ?_, dp_disj_dm := ?_, cancel_clean := ?_, multi_iff := ?_,
           row_empty := ?_, me_keyed := ?_, in_range := ?_,
           valid_ids := ?_ }
  · simp [decode]
  · simp [decode]
  · intro r hr; simp [decode] at hr
  · intro r hr
    rw [effGet_decode_encode h] at hr
    have hrm : r ∈ t.multiPairs :=
      mem_multiPairs.mpr ⟨mem_effDom_iff_isSome.mpr (by rw [hr]; rfl), hr⟩
    rw [meRow_decode_encode_multi h hrm]
    exact h.multi_iff r hr
  · intro r hr
    rw [effGet_decode_encode h] at hr
    exact meRow_decode_encode_not_multi h (fun hc => hr (mem_multiPairs.mp hc).2)
  · intro x hx
    obtain ⟨p, hp, hxk, hi⟩ := mem_me_decode_encode |>.mp hx
    refine ⟨p, ?_, ?_⟩
    · rw [hdom]; exact (mem_multiPairs.mp hp).1
    · have := congrArg (fun (y : Addr × Nat) => y.1) hxk
      simpa using this
  · intro r hr
    have hr' : r ∈ t.effDom := by
      simp only [decode, encode, Finset.mem_union, Finset.mem_sdiff, Finset.notMem_empty,
        or_false, not_false_eq_true, and_true] at hr
      exact hr
    refine h.in_range r ?_
    rcases Finset.mem_union.mp hr' with h1 | h1
    · exact Finset.mem_union_left _ (Finset.mem_sdiff.mp h1).1
    · exact Finset.mem_union_right _ h1
  · intro r i hi
    rw [edgesAt_decode_encode h] at hi
    exact h.valid_ids r i hi

/-- **End to end**, the way the caller uses it: `decode` then `rebuild_backward`
gives back a fully valid tensor denoting the same graph, with the same
`edge_count`. -/
theorem inv_decode_encode (h : Inv t) : Inv (rebuildBackward (decode (encode t))) :=
  inv_rebuildBackward (invCore_decode_encode h)

theorem edgesAt_roundTrip (h : Inv t) :
    (rebuildBackward (decode (encode t))).edgesAt q = t.edgesAt q := by
  rw [edgesAt_rebuildBackward]
  exact edgesAt_decode_encode h

theorem edgeCount_roundTrip (h : Inv t) :
    edgeCount (rebuildBackward (decode (encode t))) = edgeCount t := by
  rw [edgeCount_eq_sum (inv_decode_encode h), edgeCount_eq_sum h, totalEdges, totalEdges]
  have hdom : (rebuildBackward (decode (encode t))).effDom = t.effDom := by
    ext r
    show r ∈ (decode (encode t)).effDom ↔ r ∈ t.effDom
    rw [mem_effDom_iff_isSome, mem_effDom_iff_isSome, effGet_decode_encode h]
  rw [hdom]
  exact Finset.sum_congr rfl (fun r _ => congrArg Finset.card (edgesAt_roundTrip h))

/-- The `total` field on disk is the edge count. -/
theorem total_encode (h : Inv t) : (encode t).total = totalEdges t := edgeCount_eq_sum h

end RoundTrip

end Tensor
end FalkorDB
