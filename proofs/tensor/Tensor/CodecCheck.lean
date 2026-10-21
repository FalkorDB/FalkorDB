/-
# Rejecting a malformed blob

`Codec.lean` proves the round trip: `decode (encode t)` denotes what `t` denotes,
and satisfies every invariant but the backward one. That says nothing about a
blob `encode` did not write.

It matters because a corrupt blob is one of only two ways a proved invariant can
still break in a running process (the other is a memory error). Everything else
in this development is closed under the operations: start valid, stay valid.
`decode` is the one entry point that manufactures a tensor out of bytes, and
`decode` as modelled is *total* — hand it a blob whose forward matrix tags a pair
as multi-edge while the tensor section carries no ids for it, and it produces a
tensor violating Invariant promotion-completeness, from which the iterator's
`getIds` would index an empty row.

This file closes that. [`WellFormed`] is the predicate a decoder must check, and
[`wellFormed_iff_invCore`] says it is exactly right: **the check accepts a blob
if and only if the tensor it decodes to satisfies the invariants.** Not merely
sound — no valid blob is rejected either, so a conforming writer cannot be locked
out by an over-strict reader.

## What each clause is checking

Every clause ranges over the blob's own tables, so each is a finite check a
decoder can actually run:

* `bounded`, `in_range` — the coordinates fit `u32` and lie inside the declared
  dimensions;
* `multi_rows` — every MSB-tagged cell has at least two ids in the tensor
  section. This is the one that stops the fabricated-sentinel blob above;
* `keyed` — and conversely every tensor-section row belongs to a pair that is
  present *and* tagged, so no row is orphaned and none shadows an inline cell;
* `ids_valid`, `inline_valid` — stored ids are GraphBLAS indices, hence never the
  sentinel.

`keyed` carries the weight twice: with `key_inj` it also gives
Invariant `row_empty`, since a row keyed to a bounded tagged pair cannot also be
the row of some other bounded pair.
-/
import Tensor.Codec

namespace FalkorDB
namespace Tensor

variable {e : Encoded}

/-! ## The blob, read as a tensor

Three abbreviations for what `decode` computes, so the checker can be stated over
the blob rather than over the tensor it produces. -/

/-- The positions `decode` will keep. -/
def decDom (e : Encoded) : Finset Pair := (e.fwdM.dom \ e.fwdDm) ∪ e.fwdDp.dom

/-- The value `decode` will store at a position: the delta overrides the base, and
an MSB-tagged value becomes the sentinel. -/
def decVal (e : Encoded) (p : Pair) : Nat :=
  let v := if p ∈ e.fwdDp.dom then e.fwdDp.val p else e.fwdM.val p
  if v &&& MSB = 0 then v else MULTI

/-- The overflow set `decode` will build from the two tensor groups. -/
def decMe (e : Encoded) : Finset (Addr × Nat) :=
  loadGroup (loadGroup ∅ e.groupBase) e.groupDelta

@[simp] theorem decode_m_dom : (decode e).m.dom = decDom e := rfl
@[simp] theorem decode_m_val (p : Pair) : (decode e).m.val p = decVal e p := rfl
@[simp] theorem decode_me_eq : (decode e).me = decMe e := rfl

theorem decode_effDom : (decode e).effDom = decDom e := by
  simp [decode, Tensor.effDom, decDom]

theorem decode_effGet (p : Pair) :
    (decode e).effGet p = if p ∈ decDom e then some (decVal e p) else none := by
  have hdp : (decode e).dp.get p = none := by simp [decode, Layer.get]
  have hdm : p ∉ (decode e).dm := by simp [decode]
  rw [effGet_of_m hdp hdm]
  simp [decode, Layer.get, decDom, decVal]

theorem decode_meRow (k : Addr) : (decode e).meRow k = meRowOf (decMe e) k := rfl

/-! ## The check -/

/-- What a decoder must verify before trusting a blob. Every clause ranges over
the blob's own tables. -/
structure WellFormed (e : Encoded) : Prop where
  /-- Coordinates lie inside the declared dimensions.

  There is no companion clause about node-id *width*. Before #2579 a decoder had
  to check that each coordinate fitted the compound key's `u32` halves, because a
  blob naming a larger id produced key aliasing or a dropped write. The key now
  accepts every pair, so that obligation is gone — one fewer way for a blob to be
  malformed, ruled out by construction rather than by the decoder. -/
  in_range : ∀ p ∈ decDom e, p.1 < e.nrows ∧ p.2 < e.ncols
  /-- A tagged cell has at least two ids in the tensor section. Without this a
  blob can fabricate a sentinel over an empty row. -/
  multi_rows : ∀ p ∈ decDom e, decVal e p = MULTI → 2 ≤ (meRowOf (decMe e) (key p)).card
  /-- Every tensor-section entry belongs to a present, tagged pair. -/
  keyed : ∀ x ∈ decMe e, ∃ p ∈ decDom e, x.1 = key p ∧ decVal e p = MULTI
  /-- Stored ids are GraphBLAS indices. -/
  ids_valid : ∀ x ∈ decMe e, ValidId x.2
  /-- …as are inline ids. -/
  inline_valid : ∀ p ∈ decDom e, decVal e p ≠ MULTI → ValidId (decVal e p)

/-- A decoder that rejects rather than trusting. The model states the check as a
predicate; each clause is a finite scan of the blob's tables, which is what makes
it implementable. -/
noncomputable def decodeChecked (e : Encoded) : Option Tensor :=
  open Classical in
  if WellFormed e then some (decode e) else none

/-! ## The check is exactly right

Soundness is the half that matters — an accepted blob decodes to a tensor
satisfying the invariants. Completeness matters too, though: a check that
rejected valid blobs would be a compatibility bug rather than a safety one, and
this one does not. -/

theorem invCore_of_wellFormed (hw : WellFormed e) : InvCore (decode e) := by
  refine { dm_sub_m := ?_, dp_disj_dm := ?_, cancel_clean := ?_, multi_iff := ?_,
           row_empty := ?_, me_keyed := ?_, in_range := ?_,
           valid_ids := ?_ }
  · simp [decode]
  · simp [decode]
  · intro q hq; simp [decode] at hq
  · intro q hq
    by_cases hmem : q ∈ decDom e
    · rw [decode_effGet, if_pos hmem] at hq
      rw [decode_meRow]
      exact hw.multi_rows q hmem (Option.some_inj.mp hq)
    · rw [decode_effGet, if_neg hmem] at hq
      exact absurd hq (by simp)
  · intro q hq
    rw [decode_meRow]
    by_contra hne
    obtain ⟨i, hi⟩ := Finset.nonempty_iff_ne_empty.mpr hne
    obtain ⟨p, hp, hkey, hval⟩ := hw.keyed (key q, i) (mem_meRowOf.mp hi)
    have : p = q := key_inj hkey.symm
    subst this
    exact hq (by rw [decode_effGet, if_pos hp, hval])
  · intro x hx
    obtain ⟨p, hp, hkey, _⟩ := hw.keyed x (by simpa using hx)
    exact ⟨p, by rw [decode_effDom]; exact hp, hkey⟩
  · intro q hq
    have hq' : q ∈ decDom e := by
      rcases Finset.mem_union.mp hq with h | h
      · exact h
      · exact absurd h (by simp [decode])
    exact hw.in_range q hq'
  · intro q i hi
    by_cases hq : q ∈ decDom e
    · have hget : (decode e).effGet q = some (decVal e q) := by rw [decode_effGet, if_pos hq]
      by_cases hM : decVal e q = MULTI
      · simp only [Tensor.edgesAt, hget, hM, if_pos, decode_meRow] at hi
        exact hw.ids_valid (key q, i) (mem_meRowOf.mp hi)
      · simp only [Tensor.edgesAt, hget, if_neg hM, Finset.mem_singleton] at hi
        rw [hi]; exact hw.inline_valid q hq hM
    · have hget : (decode e).effGet q = none := by rw [decode_effGet, if_neg hq]
      simp only [Tensor.edgesAt, hget] at hi
      exact absurd hi (by simp)

theorem wellFormed_of_invCore (hi : InvCore (decode e)) : WellFormed e := by
  refine { in_range := ?_, multi_rows := ?_, keyed := ?_, ids_valid := ?_,
           inline_valid := ?_ }
  · intro p hp
    exact hi.in_range p (Finset.mem_union_left _ (by simpa using hp))
  · intro p hp hval
    have := hi.multi_iff p (by rw [decode_effGet, if_pos hp, hval])
    rwa [decode_meRow] at this
  · intro x hx
    obtain ⟨p, hpd, hkey⟩ := hi.me_keyed x (by simpa using hx)
    refine ⟨p, by rw [← decode_effDom]; exact hpd, hkey, ?_⟩
    by_contra hval
    have hrow : (decode e).meRow (key p) = ∅ := by
      refine hi.row_empty p ?_
      rw [decode_effGet, if_pos (by rw [← decode_effDom]; exact hpd)]
      exact fun hc => hval (Option.some_inj.mp hc)
    rw [decode_meRow] at hrow
    have : x.2 ∈ meRowOf (decMe e) (key p) := mem_meRowOf.mpr (by rw [← hkey]; exact hx)
    rw [hrow] at this
    exact absurd this (by simp)
  · intro x hx
    obtain ⟨p, hpd, hkey⟩ := hi.me_keyed x (by simpa using hx)
    refine hi.valid_ids p x.2 ?_
    have hmulti : (decode e).effGet p = some MULTI := by
      by_contra hne
      have hrow := hi.row_empty p hne
      rw [decode_meRow] at hrow
      have hmem2 : x.2 ∈ meRowOf (decMe e) (key p) := mem_meRowOf.mpr (by rw [← hkey]; exact hx)
      rw [hrow] at hmem2
      exact absurd hmem2 (by simp)
    simp only [Tensor.edgesAt, hmulti, if_pos, decode_meRow]
    exact mem_meRowOf.mpr (by rw [← hkey]; exact hx)
  · intro p hp hval
    have hget : (decode e).effGet p = some (decVal e p) := by rw [decode_effGet, if_pos hp]
    refine hi.valid_ids p (decVal e p) ?_
    simp only [Tensor.edgesAt, hget, if_neg hval]
    simp

/-- **The check accepts a blob exactly when the blob decodes to a valid tensor.** -/
theorem wellFormed_iff_invCore : WellFormed e ↔ InvCore (decode e) :=
  ⟨invCore_of_wellFormed, wellFormed_of_invCore⟩

/-- **Soundness**: whatever `decodeChecked` returns satisfies the invariants —
every one but the backward-matrix clause, which `decode`'s caller restores with
`rebuild_backward` (`inv_rebuildBackward`), exactly as on the round-trip path. -/
theorem invCore_decodeChecked {t : Tensor} (h : decodeChecked e = some t) : InvCore t := by
  classical
  unfold decodeChecked at h
  split at h
  · rename_i hw
    rw [← Option.some_inj.mp h]
    exact invCore_of_wellFormed hw
  · exact absurd h (by simp)

/-- **Completeness**: a blob this codec wrote is never rejected. -/
theorem decodeChecked_encode {t : Tensor} (h : Inv t) :
    decodeChecked (encode t) = some (decode (encode t)) := by
  classical
  unfold decodeChecked
  rw [if_pos (wellFormed_of_invCore (invCore_decode_encode h))]

/-- And so the round trip survives the check: a tensor encoded and decoded through
the *checking* decoder denotes what it started as. -/
theorem edgesAt_decodeChecked_encode {t : Tensor} (h : Inv t) {t' : Tensor}
    (hd : decodeChecked (encode t) = some t') (q : Pair) : t'.edgesAt q = t.edgesAt q := by
  rw [decodeChecked_encode h] at hd
  rw [← Option.some_inj.mp hd]
  exact edgesAt_decode_encode (q := q) h

end Tensor
end FalkorDB
