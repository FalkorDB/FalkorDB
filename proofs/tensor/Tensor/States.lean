/-
# The per-pair state diagram

`tensor.rs` documents every reachable layer state of one `(src, dst)` pair and
the transitions between them:

```text
  Uncommitted (pair absent from m):
    A  ·                          empty
    B  dp=1                       pending single edge
    C  dp=M          me={1,2}     pending multi pair

  Committed single (m=1):
    D  m=1                        clean
    G  m=1 dm=x                   deleted
    H  m=1 dp=2                   value replaced   (dp shadows m)
    F  m=1 dp=M      me={1,2}     promoted         (dp shadows m)

  Committed multi (m=M):
    E  m=M           me={1,2}     clean
    I  m=M dp=1      me={}        demoted          (dp shadows m)
    J  m=M dm=x      me={}        deleted
```

Each theorem below is one arrow of that diagram, stated on the *raw layers*
(what `m`, `dp`, `dm`, `me` and `multiCount` hold afterwards) rather than on the
denotation — that is what the diagram claims, and it is where the delicate
"cancel to clean" and "`dp` shadows `m`" behaviour shows up.

These are the *representative* arrows, not all of them: seven distinct arrows
plus the three cancel-to-clean edges, chosen so that every shape the
representation can take is stated somewhere. The rest of the diagram's arrows
are covered denotationally by the general theorems below, which is a weaker
statement about the layers and a sufficient one about behaviour.

The invariants and the denotational effect of each arrow are already covered by
the general theorems (`inv_addEdge`, `edgesAt_addEdge_self`, `removeOne_spec`);
these theorems check that the *representation* is the documented one.
-/
import Tensor.Remove

-- Each theorem states the *whole* documented source state, including parts a
-- particular proof happens not to need (e.g. `dm` being clear in state `I`).
set_option linter.unusedVariables false

namespace FalkorDB
namespace Tensor

variable {t : Tensor} {p : Pair} {i1 i2 i3 : Nat}

/-! ## Adding: `A → B → C`, `D → F`, `G → H`, `I → E` -/

/-- `A --add 1--> B`: the first edge of an absent pair lands inline in `dp`. -/
theorem trans_A_add (hm : t.m.get p = none) (hdm : p ∉ t.dm) (hdp : t.dp.get p = none) :
    (addEdge t p i1).m.get p = none ∧ (addEdge t p i1).dp.get p = some i1 ∧
      p ∉ (addEdge t p i1).dm ∧ (addEdge t p i1).me = t.me ∧
      (addEdge t p i1).multiCount = t.multiCount := by
  have hv : t.effGet p = none := by simp [effGet, hdp, hdm, hm]
  rw [addEdge_first_def hv, if_neg hdm]
  exact ⟨by simp [hm], by simp, by simp [hdm], by simp, by simp⟩

/-- `B --add 2--> C`: the second edge of a pending pair promotes it in place —
the inline slot becomes the sentinel and both ids move to `me`. -/
theorem trans_B_add (hm : t.m.get p = none) (hdm : p ∉ t.dm) (hdp : t.dp.get p = some i1)
    (hrow : t.meRow (key p) = ∅) (hM : i1 ≠ MULTI) :
    (addEdge t p i2).m.get p = none ∧ (addEdge t p i2).dp.get p = some MULTI ∧
      p ∉ (addEdge t p i2).dm ∧ (addEdge t p i2).meRow (key p) = {i1, i2} ∧
      (addEdge t p i2).multiCount = t.multiCount + 1 := by
  have hv : t.effGet p = some i1 := effGet_of_dp hdp
  rw [addEdge_promote_def hv hM, if_pos (by simp [hdp]), hm]
  refine ⟨by simp [hm], by simp, by simp [hdm], ?_, by simp⟩
  rw [meRow, writeInline_me]
  simp only [meRowOf_insert_self]
  simp only [meRow] at hrow
  rw [hrow]
  simp

/-- `D --add 2--> F`: a committed single edge is promoted; `dp` *shadows* the
committed entry with the sentinel while `m` keeps the old id. -/
theorem trans_D_add (hm : t.m.get p = some i1) (hdm : p ∉ t.dm) (hdp : t.dp.get p = none)
    (hrow : t.meRow (key p) = ∅) (hne : i2 ≠ i1) (hM : i1 ≠ MULTI) :
    (addEdge t p i2).m.get p = some i1 ∧ (addEdge t p i2).dp.get p = some MULTI ∧
      p ∉ (addEdge t p i2).dm ∧ (addEdge t p i2).meRow (key p) = {i1, i2} ∧
      (addEdge t p i2).multiCount = t.multiCount + 1 := by
  have hv : t.effGet p = some i1 := by rw [effGet_of_m hdp hdm]; exact hm
  rw [addEdge_promote_def hv hM, if_neg (by simp [hdp])]
  refine ⟨by simp [hm], by simp, by simp [hdm], ?_, by simp⟩
  rw [meRow, writeInline_me]
  simp only [meRowOf_insert_self]
  simp only [meRow] at hrow
  rw [hrow]
  simp

/-- `G --add 1--> D` — **cancel to clean**: re-adding the committed id un-masks
`dm` and leaves *no* delta behind. -/
theorem trans_G_add_cancel (hm : t.m.get p = some i1) (hdm : p ∈ t.dm) (hdp : t.dp.get p = none) :
    (addEdge t p i1).m.get p = some i1 ∧ (addEdge t p i1).dp.get p = none ∧
      p ∉ (addEdge t p i1).dm ∧ (addEdge t p i1).me = t.me ∧
      (addEdge t p i1).multiCount = t.multiCount := by
  have hv : t.effGet p = none := by simp [effGet, hdp, hdm]
  rw [addEdge_first_def hv, if_pos hdm, hm]
  exact ⟨by simp [hm], by simp, by simp, by simp, by simp⟩

/-- `G --add 2--> H`: adding a *different* id to a deleted committed pair un-masks
`dm` and shadows `m` with the new id. -/
theorem trans_G_add_other (hm : t.m.get p = some i1) (hdm : p ∈ t.dm) (hdp : t.dp.get p = none)
    (hne : i2 ≠ i1) :
    (addEdge t p i2).m.get p = some i1 ∧ (addEdge t p i2).dp.get p = some i2 ∧
      p ∉ (addEdge t p i2).dm ∧ (addEdge t p i2).me = t.me ∧
      (addEdge t p i2).multiCount = t.multiCount := by
  have hv : t.effGet p = none := by simp [effGet, hdp, hdm]
  rw [addEdge_first_def hv, if_pos hdm, hm]
  exact ⟨by simp [hm], writeInline_shadow_dp (Ne.symm hne), by simp, by simp, by simp⟩

/-- `I --add 3--> E` — **cancel to clean, re-promotion**: a pair demoted earlier in
the same transaction is promoted again, and because the committed value is already
the sentinel the `dp` shadow is *dropped* rather than rewritten (`dp = M` must
never shadow `m = M`). -/
theorem trans_I_add_cancel (hm : t.m.get p = some MULTI) (hdm : p ∉ t.dm)
    (hdp : t.dp.get p = some i1) (hrow : t.meRow (key p) = ∅) (hM : i1 ≠ MULTI) :
    (addEdge t p i3).m.get p = some MULTI ∧ (addEdge t p i3).dp.get p = none ∧
      p ∉ (addEdge t p i3).dm ∧ (addEdge t p i3).meRow (key p) = {i1, i3} ∧
      (addEdge t p i3).multiCount = t.multiCount + 1 := by
  have hv : t.effGet p = some i1 := effGet_of_dp hdp
  rw [addEdge_promote_def hv hM, if_pos (by simp [hdp]), hm]
  refine ⟨by simp [hm], by simp, by simp, ?_, by simp⟩
  rw [meRow, writeInline_me]
  simp only [meRowOf_insert_self]
  simp only [meRow] at hrow
  rw [hrow]
  simp

/-! ## Removing: `D → G`, `E → I`, `F → D` -/

/-- `D --del 1--> G`: deleting the only edge of a clean committed pair masks the
committed entry. -/
theorem trans_D_del (hm : t.m.get p = some i1) (hdm : p ∉ t.dm) (hdp : t.dp.get p = none)
    (hM : i1 ≠ MULTI) :
    (removeOne t i1 p).1.m.get p = some i1 ∧ (removeOne t i1 p).1.dp.get p = none ∧
      p ∈ (removeOne t i1 p).1.dm ∧ (removeOne t i1 p).1.me = t.me ∧
      (removeOne t i1 p).2 = some p := by
  have hv : t.effGet p = some i1 := by rw [effGet_of_m hdp hdm]; exact hm
  have hmdom : p ∈ t.m.dom := Layer.get_isSome.mp (by rw [hm]; rfl)
  rw [removeOne_single hv hM]
  exact ⟨hm, by simp [deletePair], by simp [deletePair, hmdom], rfl, rfl⟩

/-- `E --del 2--> I` — **demotion with a shadow**: the surviving id returns to the
inline slot in `dp` (the committed value is the sentinel, so the delta stays). -/
theorem trans_E_del (hm : t.m.get p = some MULTI) (hdm : p ∉ t.dm) (hdp : t.dp.get p = none)
    (hrow : t.meRow (key p) = {i1, i2}) (hne : i1 ≠ i2) (hM : i1 ≠ MULTI) :
    (removeOne t i2 p).1.m.get p = some MULTI ∧ (removeOne t i2 p).1.dp.get p = some i1 ∧
      p ∉ (removeOne t i2 p).1.dm ∧ (removeOne t i2 p).1.meRow (key p) = ∅ ∧
      (removeOne t i2 p).1.multiCount = t.multiCount - 1 ∧ (removeOne t i2 p).2 = none := by
  have hv : t.effGet p = some MULTI := by rw [effGet_of_m hdp hdm]; exact hm
  have hafter : rowAfterErase t p i2 = {i1} := by
    rw [rowAfterErase_eq, hrow, Finset.pair_comm, Finset.erase_insert (by simpa using Ne.symm hne)]
  have hcard : ¬ 2 ≤ (rowAfterErase t p i2).card := by rw [hafter]; simp
  have hmin : (rowAfterErase t p i2).min = some i1 := by rw [hafter]; rfl
  have hmv : t.m.get p ≠ some i1 := by rw [hm]; simpa using Ne.symm hM
  rw [removeOne_demote_shadow hv hcard hmin hmv]
  refine ⟨hm, by simp, hdm, ?_, rfl, rfl⟩
  show meRowOf ((t.me.erase (key p, i2)).erase (key p, i1)) (key p) = ∅
  rw [meRowOf_erase_self, meRowOf_erase_self, ← meRow, hrow, Finset.pair_comm,
    Finset.erase_insert (by simpa using Ne.symm hne)]
  simp

/-- `F --del 2--> D` — **demotion that cancels**: the survivor *is* the committed
value, so the `dp` shadow is dropped and the pair returns to the clean state. -/
theorem trans_F_del_cancel (hm : t.m.get p = some i1) (hdm : p ∉ t.dm)
    (hdp : t.dp.get p = some MULTI) (hrow : t.meRow (key p) = {i1, i2}) (hne : i1 ≠ i2) :
    (removeOne t i2 p).1.m.get p = some i1 ∧ (removeOne t i2 p).1.dp.get p = none ∧
      p ∉ (removeOne t i2 p).1.dm ∧ (removeOne t i2 p).1.meRow (key p) = ∅ ∧
      (removeOne t i2 p).1.multiCount = t.multiCount - 1 ∧ (removeOne t i2 p).2 = none := by
  have hv : t.effGet p = some MULTI := effGet_of_dp hdp
  have hafter : rowAfterErase t p i2 = {i1} := by
    rw [rowAfterErase_eq, hrow, Finset.pair_comm, Finset.erase_insert (by simpa using Ne.symm hne)]
  have hcard : ¬ 2 ≤ (rowAfterErase t p i2).card := by rw [hafter]; simp
  have hmin : (rowAfterErase t p i2).min = some i1 := by rw [hafter]; rfl
  rw [removeOne_demote_cancel hv hcard hmin hm]
  refine ⟨hm, by simp, hdm, ?_, rfl, rfl⟩
  show meRowOf ((t.me.erase (key p, i2)).erase (key p, i1)) (key p) = ∅
  rw [meRowOf_erase_self, meRowOf_erase_self, ← meRow, hrow, Finset.pair_comm,
    Finset.erase_insert (by simpa using Ne.symm hne)]
  simp

end Tensor
end FalkorDB
