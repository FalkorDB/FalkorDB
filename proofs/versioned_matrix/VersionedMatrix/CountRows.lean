import VersionedMatrix.VMOps
import VersionedMatrix.TensorResize
/-
# `VersionedMatrix::count_in_rows` and `Tensor::count_pairs_into` (#3022)

| here | there |
| --- | --- |
| `LayerCount` | `Matrix::count_in_rows` (`matrix.rs:774`): hypothesis here, proved from `GrB_mxv` + `GrB_Vector_reduce` in `proofs/graphblas_wrappers` (`countInRows_spec`) |
| `vmCountInRows` | `VersionedMatrix::count_in_rows` (`versioned_matrix.rs:818-834`) |
| `countPairsInto` | `Tensor::count_pairs_into` (`tensor.rs:1315-1320`) = `self.mt.count_in_rows(dsts)` |

`count_in_rows` counts per layer and combines `|m| + |dp| − |dm|`; the doc says
this is exact on a bool matrix because `dm ⊆ m` and `dp ∩ m = ∅`.
`vmCountInRows_spec` proves it: the result is the number of *effective* entries
`(m ∖ dm) ∪ dp` in the selected rows, and the `u64` subtraction cannot underflow.
It needs only `dm ⊆ m` and duplicate-free layers (`dp ∩ m = ∅` is what makes the
effective list a set, not what makes the count right).
-/
namespace VMCount
open VMMat VMOps

/-- A coordinate whose row is selected. -/
def inRows (S : List Nat) (p : Pair) : Bool := decide (p.1 ∈ S)

/-- **Hypothesis** — `Matrix::count_in_rows` counts a layer's stored entries in
the selected rows (`proofs/graphblas_wrappers` `countInRows_spec`). -/
structure LayerCount where
  cr : Mat Unit → List Nat → Nat
  spec : ∀ l S, cr l S = (keys l).countP (inRows S)

/-- The `layer` closure: an empty layer costs nothing (`nvals() == 0 → 0`). -/
def layer (C : LayerCount) (l : Mat Unit) (S : List Nat) : Nat := if nvals l = 0 then 0 else C.cr l S

theorem layer_eq (C : LayerCount) (l : Mat Unit) (S : List Nat) : layer C l S = (keys l).countP (inRows S) := by
  unfold layer; split
  · rename_i h
    have : keys l = [] := by unfold keys; unfold nvals at h; exact List.map_eq_nil_iff.mpr (List.length_eq_zero_iff.mp h)
    rw [this]; rfl
  · exact C.spec l S

/-- `count_in_rows` (:818): `self.wait(); self.m.wait();` then `layer(m) + layer(dp) − layer(dm)`. -/
def vmCountInRows (C : LayerCount) (v : BVM) (S : List Nat) : Nat :=
  let w := wait v
  let m := VMMat.wait w.m
  layer C m S + layer C w.dp.layer S - layer C w.dm.layer S

/-- The effective keys `(m ∖ dm) ∪ dp`. -/
def effKeys (v : BVM) : List Pair := (keys v.m).filter (fun p => !decide (p ∈ keys v.dm.layer)) ++ keys v.dp.layer

theorem perm_of_nodup_sub {l d : List Pair} (hl : l.Nodup) (hd : d.Nodup)
    (hsub : ∀ x ∈ d, x ∈ l) : (l.filter (fun x => decide (x ∈ d))).Perm d := by
  rw [List.perm_iff_count]
  intro a
  have hf : (l.filter (fun x => decide (x ∈ d))).Nodup := hl.sublist List.filter_sublist
  rw [hf.count, hd.count]
  by_cases ha : a ∈ d
  · simp [ha, hsub a ha]
  · simp [ha]

theorem countP_split (p : Pair → Bool) {l d : List Pair} (hl : l.Nodup) (hd : d.Nodup)
    (hsub : ∀ x ∈ d, x ∈ l) :
    l.countP p = (l.filter (fun x => !decide (x ∈ d))).countP p + d.countP p := by
  have h1 := (List.filter_append_perm (fun x => decide (x ∈ d)) l).countP_eq p
  rw [← h1, List.countP_append, (perm_of_nodup_sub hl hd hsub).countP_eq p, Nat.add_comm]

/-- **`count_in_rows` is exact**: it returns the number of effective entries in
the selected rows, and `layer(dm) ≤ layer(m)` (no `u64` underflow), given
`dm ⊆ m` and duplicate-free `m`/`dm` keys. -/
theorem vmCountInRows_spec (C : LayerCount) (v : BVM) (S : List Nat)
    (hm : (keys v.m).Nodup) (hdm : (keys v.dm.layer).Nodup)
    (hsub : ∀ p ∈ keys v.dm.layer, p ∈ keys v.m) :
    vmCountInRows C v S = (effKeys v).countP (inRows S) ∧
    layer C (wait v).dm.layer S ≤ layer C (VMMat.wait (wait v).m) S := by
  obtain ⟨k1, k2, k3⟩ := wait_keys v
  have km : keys (VMMat.wait (wait v).m) = keys v.m := by rw [← k1]; rfl
  have hs := countP_split (inRows S) hm hdm hsub
  unfold vmCountInRows
  simp only [layer_eq, km, k2, k3]
  refine ⟨?_, by omega⟩
  unfold effKeys; rw [List.countP_append]
  omega

/-! ## `Tensor::count_pairs_into` -/

open VMTensor VMTensorOps

/-- The backward matrix `mt`'s logical keys: `(dst, src)` for every pair whose `mt` bit is set. -/
def mtKeys (t : TT) : List VMMat.Pair := (t.ps.filter (fun p => (t.st p).mt)).map (fun p => (p.2, p.1))

/-- `count_pairs_into` (tensor.rs:1315): `self.mt.count_in_rows(dsts)`, with
`count_in_rows` exact on `mt` (`vmCountInRows_spec`). -/
def countPairsInto (vmCount : List VMMat.Pair → List Nat → Nat) (t : TT) (dsts : List Nat) : Nat :=
  vmCount (mtKeys t) dsts

/-- **Pairs, not edges, ending at one of `dsts`**; with the tensor invariant (`mt`
mirrors the effective forward structure), positive iff some live pair ends there. -/
theorem countPairsInto_spec (vmCount : List VMMat.Pair → List Nat → Nat)
    (hc : ∀ ks S, vmCount ks S = ks.countP (inRows S)) (t : TT) (dsts : List Nat) :
    countPairsInto vmCount t dsts = (t.ps.filter (fun p => (t.st p).mt && decide (p.2 ∈ dsts))).length ∧
    (AllInv t → (0 < countPairsInto vmCount t dsts ↔
      ∃ p ∈ t.ps, (effV (t.st p)).isSome = true ∧ p.2 ∈ dsts)) := by
  have heq : countPairsInto vmCount t dsts = (t.ps.filter (fun p => (t.st p).mt && decide (p.2 ∈ dsts))).length := by
    unfold countPairsInto mtKeys
    rw [hc, List.countP_map, List.countP_filter, List.countP_eq_length_filter]
    congr 1; apply List.filter_congr; intro p _; simp [inRows, Function.comp, Bool.and_comm]
  refine ⟨heq, fun hi => ?_⟩
  rw [heq, List.length_pos_iff]
  constructor
  · intro hne
    obtain ⟨p, hp⟩ := List.exists_mem_of_ne_nil _ hne
    rw [List.mem_filter, Bool.and_eq_true, decide_eq_true_eq] at hp
    exact ⟨p, hp.1, (hi p).mt_eff ▸ hp.2.1, hp.2.2⟩
  · rintro ⟨p, hp, he, hd⟩ hnil
    have : p ∈ t.ps.filter (fun p => (t.st p).mt && decide (p.2 ∈ dsts)) := by
      rw [List.mem_filter, Bool.and_eq_true, decide_eq_true_eq]
      exact ⟨hp, (hi p).mt_eff ▸ he, hd⟩
    rw [hnil] at this; cases this

end VMCount
