import VersionedMatrix.TensorSave
import VersionedMatrix.Delta
/-
# `Tensor::clean_fwd` / `Tensor::clean_bwd`: lend the committed base when no delta is pending

| here | there (`tensor.rs`) |
| --- | --- |
| `cleanFwd` | `Tensor::clean_fwd`: `wait_fwd`, then `m` iff `dp.nvals() == 0 && dm.nvals() == 0` |
| `cleanBwd` | `Tensor::clean_bwd`: `mt.wait()`, then `mt.m()` iff `mt.dp().nvals() == 0 && mt.dm().nvals() == 0` |

The forward base is read as a content map (`fastBase`'s shape: a pair's stored
`m` value, the edge id or the multi-edge marker); the backward `mt` is a
`VersionedMatrix<bool>` (`_root_.VM.VM`), whose `wait` moves no entry (`VMOps.wait_eff`).
-/
namespace VMTensorOps
open VMTensor TensorSave

/-- `Tensor::clean_fwd`. -/
def cleanFwd (t : TT) : Option (Pair → Option Val) :=
  let t' := waitFwd t
  if dpCount t' == 0 && dmCount t' == 0 then
    some (fun q => if q ∈ t'.ps then (t'.st q).m else none)
  else none

/-- `clean_fwd` lends a base exactly when neither forward delta holds an entry. -/
theorem cleanFwd_isSome_iff (t : TT) :
    (cleanFwd t).isSome = true ↔ dpCount t = 0 ∧ dmCount t = 0 := by
  have hc : dpCount (waitFwd t) = dpCount t ∧ dmCount (waitFwd t) = dmCount t := by
    unfold dpCount dmCount; rw [waitFwd_ps, (waitFwd_st t).1]; exact ⟨rfl, rfl⟩
  unfold cleanFwd; simp only [hc.1, hc.2]
  split <;> simp_all

/-- **The lent base is the effective forward structure**: its stored pairs are
exactly `extract`'s, with the pair's effective value. -/
theorem cleanFwd_eq_extract (t : TT) (b : Pair → Option Val) (h : cleanFwd t = some b) :
    ∀ q, (b q).isSome = true ↔ q ∈ extract t := by
  have hs := (waitFwd_st t).1
  have hps := waitFwd_ps t
  unfold cleanFwd at h; simp only [hs, hps] at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, beq_iff_eq] at hc
    have hdp := dpCount_zero (waitFwd t) hc.1
    have hdm := dmCount_zero (waitFwd t) hc.2
    rw [hs, hps] at hdp hdm
    injection h with h; subst h
    intro q
    rw [mem_extract]
    by_cases hq : q ∈ t.ps
    · simp [hq, effV, hdp q hq, hdm q hq]
    · simp [hq]
  · cases h

theorem cleanFwd_value (t : TT) (b : Pair → Option Val) (h : cleanFwd t = some b) :
    ∀ q ∈ t.ps, b q = effV (t.st q) := by
  have hs := (waitFwd_st t).1
  have hps := waitFwd_ps t
  unfold cleanFwd at h; simp only [hs, hps] at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, beq_iff_eq] at hc
    have hdp := dpCount_zero (waitFwd t) hc.1
    have hdm := dmCount_zero (waitFwd t) hc.2
    rw [hs, hps] at hdp hdm
    injection h with h; subst h
    intro q hq
    simp [hq, effV, hdp q hq, hdm q hq]
  · cases h

/-- `Tensor::clean_bwd` on the backward `VersionedMatrix` `mt` (after its `wait`). -/
def cleanBwd (mt : _root_.VM.VM) : Option _root_.VM.Layer :=
  if mt.dp.length == 0 && mt.dm.length == 0 then some mt.m else none

theorem cleanBwd_isSome_iff (mt : _root_.VM.VM) :
    (cleanBwd mt).isSome = true ↔ mt.dp = [] ∧ mt.dm = [] := by
  unfold cleanBwd; split <;> simp_all [List.length_eq_zero_iff]

/-- **The lent backward base is the effective backward structure** `(m ∖ dm) ∪ dp`. -/
theorem cleanBwd_eq_eff (mt : _root_.VM.VM) (b : _root_.VM.Layer) (h : cleanBwd mt = some b) :
    ∀ p, p ∈ b ↔ _root_.VM.eff mt p := by
  unfold cleanBwd at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, beq_iff_eq, List.length_eq_zero_iff] at hc
    injection h with h; subst h
    intro p; simp [_root_.VM.eff, hc.1, hc.2]
  · cases h

end VMTensorOps
