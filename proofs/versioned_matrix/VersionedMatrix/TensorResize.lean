import VersionedMatrix.TensorOps
/-
# `Tensor::resize`, `encode`'s inline values, `EdgeIds::size_hint`, `Iter::new`/`seek`

| here | there (`tensor.rs`) |
| --- | --- |
| `resizeT`  | `resize` (:809): shrink (:814-828) / grow (:829-867) |
| `encVal`/`decVal` | value `encode` (:1451) puts in the C forward matrix / C's reading |
| `sizeHint` | `EdgeIds::size_hint` (:1650) |
| `baseStream`/`newTIt`/`seekTIt`/`outT` | `Iter::new` (:1678), `Iter::seek` (:1710), remaining output |

**Bug (confirmed, API level):** `resize`'s shrink branch resizes `m`/`dp`/
`dm`/`mt` but not `me`. `shrink_orphans_me` exhibits a pair satisfying
every invariant whose shrink leaves `Inv.single_me` false: the pair reads
as edgeless while its `me` row still holds two ids (repro:
`graph/tests/lean_versioned_matrix.rs::tensor_shrink_keeps_orphan_me_rows`:
after shrink `iter_edges` = [(0,0,3),(5,5,1),(5,5,2)], `edge_count` = 2,
expected 1 / [(0,0,3)]). `shrink_safe` proves the branch correct when no
multi-edge pair lies outside the new bounds, which is what the only caller
(`rebuild_derived_matrices`, `node_cap` ≥ every node id) guarantees.
-/
namespace VMTensorOps
open VMTensor

def inB (r c : Nat) (p : Pair) : Bool := decide (p.1 < r) && decide (p.2 < c)

/-- What `GrB_Matrix_resize` on `m`/`dp`/`dm` and `mt` does to one pair: drop
it from every forward layer and from `mt`; `me` is not touched. -/
def dropFwd (s : PairSt) : PairSt := { s with m := none, dp := none, dm := false, mt := false }

def resizeT (t : TT) (r c : Nat) : TT :=
  if r < t.nrows ∨ c < t.ncols then
    let t := flushT t
    { t with st := fun p => if inB r c p then t.st p else dropFwd (t.st p), nrows := r, ncols := c }
  else { t with nrows := r, ncols := c }

theorem resizeT_grow (t : TT) {r c : Nat} (hr : t.nrows ≤ r) (hc : t.ncols ≤ c) :
    (resizeT t r c).st = t.st ∧ (resizeT t r c).me = t.me := by
  unfold resizeT; rw [if_neg (by omega)]; exact ⟨rfl, rfl⟩

theorem effV_dropFwd (s : PairSt) : effV (dropFwd s) = none := rfl

/-- The shrink is correct when no multi-edge pair is dropped. -/
theorem shrink_safe {t : TT} (hi : AllInv t) {r c : Nat}
    (hno : ∀ p, inB r c p = false → (t.st p).me = []) :
    AllInv (resizeT t r c) ∧ ∀ p, edges ((resizeT t r c).st p) =
      if inB r c p ∨ ¬ (r < t.nrows ∨ c < t.ncols) then edges (t.st p) else [] := by
  have hf := flushT_correct hi
  have hme := (flushT_me t).2
  unfold resizeT; split
  · next hs =>
    refine ⟨fun p => ?_, fun p => ?_⟩
    · simp only; split
      · exact hf.1 p
      · next hb =>
        have h0 : ((flushT t).st p).me = [] := by rw [hme]; exact hno p (by simpa using hb)
        have hp := hf.1 p
        exact ⟨by simp [dropFwd], by simp [dropFwd], by simp [dropFwd],
          by intro h; simp [dropFwd, effV] at h, fun _ => by simp [dropFwd, h0],
          by simp [dropFwd, h0], by simp [dropFwd, effV]⟩
    · simp only; split
      · next hb => simp [hb, hf.2 p]
      · next hb => simp [hb, hs, edges, effV_dropFwd]
  · next hs => exact ⟨hi, fun p => by simp [hs]⟩

/-- **The bug.** A valid multi-edge pair outside the new bounds keeps its `me`
row after the shrink, breaking promotion completeness. -/
theorem shrink_orphans_me :
    ∃ t : TT, AllInv t ∧ BlkInv t ∧ hasMultiEdge t = true ∧
      ¬ Inv ((resizeT t 4 4).st (5, 5)) ∧ edges ((resizeT t 4 4).st (5, 5)) = [] ∧
      hasMultiEdge (resizeT t 4 4) = true := by
  let s5 : PairSt := ⟨none, some .multi, false, true, [1, 2]⟩
  let t : TT := ⟨[(5, 5)], fun p => if p = (5, 5) then s5 else emptyP, ⟨1, 0, false, true⟩,
    ⟨0, 0, false, true⟩, true, true, false, [⟨(0, 0), ME_NARROW_NCOLS, true⟩], 8, 8⟩
  have hi : AllInv t := by
    intro p; simp only [t]; split
    · refine ⟨by simp [s5], by simp [s5], by simp [s5], fun _ => by simp [s5], ?_, by simp [s5], by simp [s5, effV]⟩
      intro h; simp [s5, effV] at h
    · refine ⟨by simp [emptyP], by simp [emptyP], by simp [emptyP], ?_, fun _ => rfl, by simp [emptyP], rfl⟩
      intro h; simp [emptyP, effV] at h
  have hblk5 : blkOf (5, 5) = (0, 0) := by decide
  have hb : BlkInv t := by
    refine ⟨⟨_, _, _, rfl⟩, by simp [names, t], ?_, Or.inl rfl, ?_, ?_⟩
    · intro e he; simp [t] at he; subst he; rfl
    · intro p hp; simp only [t] at hp ⊢; split at hp
      · next h => subst h; simp [names, hblk5]
      · exact absurd rfl hp
    · intro p i hi'; simp only [t] at hi' ⊢; split at hi'
      · simp [s5] at hi'; simp [ME_NARROW_NCOLS]; omega
      · simp [emptyP] at hi'
  have hm : ∀ u : TT, u.ps = [(5, 5)] → u.me = t.me → (u.st (5, 5)).me = [1, 2] →
      hasMultiEdge u = true := by
    intro u h1 h2 h3
    simp [hasMultiEdge, blkIds, h1, h2, t, hblk5, h3]
  refine ⟨t, hi, hb, hm t rfl rfl (by simp [t, s5]), ?_, ?_, ?_⟩
  · intro h; have := h.single_me (by simp [resizeT, t, inB, effV, dropFwd, flushT])
    simp [resizeT, t, inB, dropFwd, flushT, s5] at this
  · simp [resizeT, t, inB, dropFwd, flushT, edges, effV]
  · apply hm
    · simp [resizeT, t, flushT]
    · simp [resizeT, t, flushT]
    · simp [resizeT, t, inB, dropFwd, flushT, s5]

/-! ## `encode`: the inline value written for C -/

def MSB : Nat := 2 ^ 63

/-- `f_vals` entry (:1469-1478): the id, or `ids.len() | MSB`. -/
def encVal (s : PairSt) : Option Nat :=
  match effV s with
  | some (.id i) => some i
  | some .multi => some (s.me.length + MSB)
  | none => none

/-- C's reading of that value: MSB set = multi-edge with that many ids. -/
def decVal (v : Nat) : Nat ⊕ Nat := if MSB ≤ v then .inr (v - MSB) else .inl v

/-- The encoding is unambiguous: real ids (`≤ GrB_INDEX_MAX < 2^63`) read back
as single edges, multi pairs as their id count; a pair absent from the forward
matrix writes nothing. -/
theorem encVal_roundtrip (s : PairSt) (h : Inv s) (hid : ∀ i, effV s = some (.id i) → i < MSB) :
    (∀ i, effV s = some (.id i) → (encVal s).map decVal = some (.inl i)) ∧
    (effV s = some .multi → (encVal s).map decVal = some (.inr (edges s).length) ∧ 2 ≤ (edges s).length) ∧
    (effV s = none → encVal s = none) := by
  refine ⟨fun i hi => ?_, fun hm => ?_, fun hn => by simp [encVal, hn]⟩
  · have := hid i hi; simp [encVal, hi, decVal]; omega
  · have h2 := h.multi_me hm
    refine ⟨?_, by simp [edges, hm]; exact h2⟩
    simp [encVal, hm, decVal, edges]

/-- The MSB mask (`1u64 << 63`) and `|` agree with `+` on counts below it. -/
theorem msb_or (n : Nat) (h : n < MSB) : n ||| MSB = n + MSB := by
  unfold MSB at *
  have := Nat.two_pow_add_eq_or_of_lt h 1
  simp at this; rw [Nat.or_comm, ← this]; omega

/-! ## `EdgeIds::size_hint` -/

inductive EdgeIds where
  | inline (o : Option Nat)
  | multi (rest : List Nat)

def remaining : EdgeIds → Nat
  | .inline o => o.toList.length
  | .multi l => l.length

/-- `Option::IntoIter` is exact; `versioned_matrix::Iter` has the default `(0, None)`. -/
def sizeHint : EdgeIds → Nat × Option Nat
  | .inline o => (o.toList.length, some o.toList.length)
  | .multi _ => (0, none)

theorem sizeHint_sound (e : EdgeIds) :
    (sizeHint e).1 ≤ remaining e ∧ ∀ u, (sizeHint e).2 = some u → remaining e ≤ u := by
  cases e <;> simp [sizeHint, remaining]

/-! ## `Tensor::Iter::new` / `seek` -/

/-- The base stream: forward (`fwd_iter`) or backward (`mt.iter`, value by `eff_get`). -/
def baseStream (t : TT) (transpose : Bool) (lo hi : Nat) : List (Pair × Option Val) :=
  if transpose then (t.ps.filter (fun p => inRow lo hi p.2 && (t.st p).mt)).map (fun p => (p, effV (t.st p)))
  else (fwdIter t lo hi).map (fun pv => (pv.1, some pv.2))

structure TIt where
  transpose : Bool
  base : List (Pair × Option Val)
  buf : List Nat
  pos : Nat
  src : Nat
  dst : Nat

def newTIt (t : TT) (lo hi : Nat) (tr : Bool) : TIt := ⟨tr, baseStream t tr lo hi, [], 0, 0, 0⟩
/-- `seek`: the base iterator re-seeks (`VMIterSeek.seek_abs`: = a fresh one), buffer dropped. -/
def seekTIt (t : TT) (s : TIt) (lo hi : Nat) : TIt :=
  { s with base := baseStream t s.transpose lo hi, buf := [], pos := 0 }

/-- What the iterator still has to yield, as `(src, dst, id)`. -/
def outT (t : TT) (s : TIt) : List (Nat × Nat × Option Nat) :=
  (s.buf.drop s.pos).map (fun i => (s.src, s.dst, some i)) ++
  s.base.flatMap (fun pv => match pv.2 with
    | some .multi => ((t.st pv.1).me).map (fun i => (pv.1.1, pv.1.2, some i))
    | some (.id i) => [(pv.1.1, pv.1.2, some i)]
    | none => [(pv.1.1, pv.1.2, none)])

theorem seek_eq_new (t : TT) (s : TIt) (lo hi : Nat) :
    outT t (seekTIt t s lo hi) = outT t (newTIt t lo hi s.transpose) := by
  simp [outT, seekTIt, newTIt]

/-- Backward iteration never hits the `unreachable!`: every `mt` pair has a
forward value (`iterBwd_eff_get_isSome`). -/
theorem iterBwd_eff_get_isSome {t : TT} (hi : AllInv t) (lo hi' : Nat) :
    ∀ pv ∈ baseStream t true lo hi', pv.2.isSome = true := by
  intro pv h
  simp only [baseStream, if_pos] at h
  obtain ⟨p, hp, hpv⟩ := List.mem_map.1 h
  subst hpv
  have := (List.mem_filter.1 hp).2
  simp only [Bool.and_eq_true] at this
  exact mt_has_forward _ (hi p) this.2

end VMTensorOps
