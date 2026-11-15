import Pr3437Review.Bridge
import Columnar.BatchMisc
import Columnar.Row

/-!
# PR #3437: the chunk batch of `list_loop_columnar` (eval.rs:1560-1564)

`chunkBatch`: `Batch::new(0)`, then `set_column(var.id, loop_var_column(chunk))` and
`set_column(id, broadcast_column(value, len))` for every outer binding of the plan, in order —
over proofs/columnar's `Batch` model (`setColumn` = `set_column`, batch.rs:1225).
**`chunkBatch_wf`**: well formed, `len` rows, no selection (so its active rows are `0..len`);
**`chunkBatch_loop`** / **`chunkBatch_outer`**: the batch row `BatchRow::value_at` reads the
chunk element at the loop variable and the outer value at every broadcast variable (outer ids
are distinct and differ from the loop variable: `vector_plan` dedups and filters them).
-/

namespace Pr3437.ChunkBatch
open FalkorCore (V)
open Columnar (Batch Column viewAt)
open Columnar.Batch (normCol)
open Pr3437.Bridge (loopVarColumn broadcastColumn loopVar_get broadcast_get)

variable {F : Type} [Columnar.FloatModel F]

set_option linter.unusedSectionVars false

/-- eval.rs:1560-1564. -/
def chunkBatch (vid : Nat) (outer : List (Nat × V F)) (chunk : List (V F)) : Batch F :=
  outer.foldl (fun b p => b.setColumn p.1 (broadcastColumn p.2 chunk.length))
    ((Batch.new0 0).setColumn vid (loopVarColumn chunk))

theorem normCol_len (c : Column F) : (normCol c).len = c.len := by
  cases c <;> simp only [normCol] <;> first | exact Columnar.classifyStored_len _ | rfl

theorem normCol_get? (c : Column F) (r : Nat) : (normCol c).get? r = c.get? r := by
  cases c <;> simp only [normCol] <;> first | exact Columnar.classifyStored_get? _ r | rfl

theorem setColumn_cols_length (b : Batch F) (id : Nat) (c : Column F) :
    (b.setColumn id c).cols.length = b.cols.length + (id + 1 - b.cols.length) := by
  simp [Batch.setColumn]

/-- The batch invariant while the columns go in: no selection / origins, `n` rows, every
bound column `n` long. -/
structure Inv (n : Nat) (b : Batch F) : Prop where
  sel : b.sel = none
  origins : b.origins = none
  len : b.len = n
  cols : ∀ c ∈ b.cols, c.isUnbound = false → c.len = n

theorem Inv.wf {n : Nat} {b : Batch F} (h : Inv n b) : b.WF where
  cols_len := fun c hc hu => by rw [h.len]; exact h.cols c hc hu
  sel_lt := fun s hs => by rw [h.sel] at hs; cases hs
  sel_sorted := fun s hs => by rw [h.sel] at hs; cases hs
  origins_len := fun o ho => by rw [h.origins] at ho; cases ho

theorem mem_set_or {α : Type} {l : List α} {i : Nat} {a x : α} (h : x ∈ l.set i a) : x = a ∨ x ∈ l := by
  rw [List.mem_iff_getElem] at h
  obtain ⟨j, hj, rfl⟩ := h
  rw [List.getElem_set]
  split
  · exact .inl rfl
  · exact .inr (List.getElem_mem _)

theorem inv_setColumn {n : Nat} {b : Batch F} (h : Inv n b) (id : Nat) (c : Column F)
    (hc : c.isUnbound = false) (hl : c.len = n) : Inv n (b.setColumn id c) where
  sel := h.sel
  origins := h.origins
  len := by simp only [Batch.setColumn, h.len, normCol_len, hl]; split <;> rfl
  cols := fun c' hc' hu => by
    simp only [Batch.setColumn] at hc'
    rcases mem_set_or hc' with rfl | hm
    · rw [normCol_len, hl]
    · rcases List.mem_append.1 hm with hm | hm
      · exact h.cols c' hm hu
      · simp only [List.mem_replicate] at hm; rw [hm.2] at hu; cases hu

theorem inv_first (vid : Nat) (chunk : List (V F)) :
    Inv chunk.length ((Batch.new0 0 : Batch F).setColumn vid (loopVarColumn chunk)) where
  sel := rfl
  origins := rfl
  len := by simp [Batch.setColumn, Batch.new0, normCol_len, (loopVar_get chunk).2.1]
  cols := fun c' hc' hu => by
    simp only [Batch.setColumn, Batch.new0] at hc'
    rcases mem_set_or hc' with rfl | hm
    · rw [normCol_len, (loopVar_get chunk).2.1]
    · simp only [List.replicate_zero, List.nil_append, List.mem_replicate] at hm; rw [hm.2] at hu; cases hu

theorem inv_fold (n : Nat) : ∀ (outer : List (Nat × V F)) (b : Batch F), Inv n b →
    Inv n (outer.foldl (fun b p => b.setColumn p.1 (broadcastColumn p.2 n)) b)
  | [], b, h => h
  | p :: ps, b, h => inv_fold n ps _ (inv_setColumn h p.1 _ (broadcast_get p.2 n).1 (broadcast_get p.2 n).2.1)

theorem chunkBatch_inv (vid : Nat) (outer : List (Nat × V F)) (chunk : List (V F)) :
    Inv chunk.length (chunkBatch vid outer chunk) :=
  inv_fold _ outer _ (inv_first vid chunk)

/-- **The chunk batch is well formed, `len` rows, all of them active.** -/
theorem chunkBatch_wf (vid : Nat) (outer : List (Nat × V F)) (chunk : List (V F)) :
    (chunkBatch vid outer chunk).WF ∧ (chunkBatch vid outer chunk).len = chunk.length ∧
      (chunkBatch vid outer chunk).active = List.range chunk.length := by
  have h := chunkBatch_inv vid outer chunk
  exact ⟨h.wf, h.len, by simp [Batch.active, h.sel, h.len]⟩

/-! ## Columns and the batch row -/

theorem fold_column (n : Nat) : ∀ (outer : List (Nat × V F)) (b : Batch F) (j : Nat),
    (outer.foldl (fun b p => b.setColumn p.1 (broadcastColumn p.2 n)) b).column j =
      match (outer.reverse.find? fun p => p.1 == j) with
      | some p => normCol (broadcastColumn p.2 n)
      | none => b.column j
  | [], b, j => rfl
  | p :: ps, b, j => by
    rw [List.foldl_cons, fold_column n ps _ j, List.reverse_cons, List.find?_append]
    cases hf : ps.reverse.find? (fun p => p.1 == j) with
    | some q => rfl
    | none =>
      simp only [Option.none_or, List.find?_singleton]
      rw [Batch.column_setColumn]
      by_cases hj : j = p.1
      · subst hj; simp
      · have : (p.1 == j) = false := by simp only [beq_eq_false_iff_ne, ne_eq]; exact fun h => hj h.symm
        simp [this, hj]

theorem fold_cols_length (n : Nat) : ∀ (outer : List (Nat × V F)) (b : Batch F),
    b.cols.length ≤ (outer.foldl (fun b p => b.setColumn p.1 (broadcastColumn p.2 n)) b).cols.length ∧
    ∀ p ∈ outer, p.1 < (outer.foldl (fun b p => b.setColumn p.1 (broadcastColumn p.2 n)) b).cols.length
  | [], b => ⟨Nat.le_refl _, fun _ h => by simp at h⟩
  | p :: ps, b => by
    obtain ⟨h1, h2⟩ := fold_cols_length n ps (b.setColumn p.1 (broadcastColumn p.2 n))
    have hs := setColumn_cols_length b p.1 (broadcastColumn p.2 n)
    refine ⟨by simp only [List.foldl_cons]; omega, fun q hq => ?_⟩
    simp only [List.foldl_cons]
    rcases List.mem_cons.1 hq with rfl | hq
    · omega
    · exact h2 q hq

theorem find_rev_none {outer : List (Nat × V F)} {j : Nat} (h : j ∉ outer.map Prod.fst) :
    outer.reverse.find? (fun p => p.1 == j) = none := by
  rw [List.find?_eq_none]
  intro q hq hqj
  simp only [beq_iff_eq] at hqj
  exact h (List.mem_map.2 ⟨q, List.mem_reverse.1 hq, hqj⟩)

theorem nodup_fst : ∀ {outer : List (Nat × V F)}, (outer.map Prod.fst).Nodup →
    ∀ {j : Nat} {v w : V F}, (j, v) ∈ outer → (j, w) ∈ outer → (j, v) = (j, w)
  | [], _, _, _, _, h, _ => by simp at h
  | p :: ps, hnd, j, v, w, h1, h2 => by
    simp only [List.map_cons, List.nodup_cons] at hnd
    rcases List.mem_cons.1 h1 with rfl | h1' <;> rcases List.mem_cons.1 h2 with he | h2'
    · exact he.symm ▸ rfl
    · exact absurd (List.mem_map.2 ⟨(j, w), h2', rfl⟩ : j ∈ ps.map Prod.fst) hnd.1
    · subst he; exact absurd (List.mem_map.2 ⟨(j, v), h1', rfl⟩ : j ∈ ps.map Prod.fst) hnd.1
    · exact nodup_fst hnd.2 h1' h2'

theorem find_rev_some {outer : List (Nat × V F)} (hnd : (outer.map Prod.fst).Nodup) {j : Nat} {v : V F}
    (hm : (j, v) ∈ outer) : outer.reverse.find? (fun p => p.1 == j) = some (j, v) := by
  have hex : ∃ q, outer.reverse.find? (fun p => p.1 == j) = some q := by
    cases h : outer.reverse.find? (fun p => p.1 == j) with
    | some q => exact ⟨q, rfl⟩
    | none =>
      rw [List.find?_eq_none] at h
      exact absurd (by simp) (h (j, v) (List.mem_reverse.2 hm))
  obtain ⟨q, hq⟩ := hex
  have hqm := List.mem_reverse.1 (List.mem_of_find?_eq_some hq)
  have hqj : q.1 = j := by simpa using List.find?_some hq
  rw [hq]
  congr 1
  obtain ⟨q1, q2⟩ := q
  simp only at hqj; subst hqj
  exact nodup_fst hnd hqm hm

theorem viewAt_of_col (b : Batch F) (j r : Nat) (c : Column F) (hc : b.column j = normCol c)
    (hu : c.isUnbound = false) (v : V F) (hv : c.get? r = some v) : viewAt b r j = some v := by
  have hu' : (b.column j).isUnbound = false := by rw [hc, Columnar.Batch.normCol_isUnbound]; exact hu
  simp [viewAt, Batch.valueAt, hc, normCol_get?, hv, Columnar.Batch.normCol_isUnbound, hu]

/-- **The batch row reads the chunk element at the loop variable.** -/
theorem chunkBatch_loop (vid : Nat) (outer : List (Nat × V F)) (chunk : List (V F))
    (hv : vid ∉ outer.map Prod.fst) (r : Nat) (hr : r < chunk.length) :
    viewAt (chunkBatch vid outer chunk) r vid = some chunk[r] := by
  apply viewAt_of_col _ vid r (loopVarColumn chunk) _ (loopVar_get chunk).1 _ ((loopVar_get chunk).2.2 r hr)
  unfold chunkBatch
  rw [fold_column, find_rev_none hv, Batch.column_setColumn]
  simp

/-- **…and the outer value at every broadcast variable.** -/
theorem chunkBatch_outer (vid : Nat) (outer : List (Nat × V F)) (chunk : List (V F))
    (hnd : (outer.map Prod.fst).Nodup) (j : Nat) (v : V F) (hm : (j, v) ∈ outer) (r : Nat) (hr : r < chunk.length) :
    viewAt (chunkBatch vid outer chunk) r j = some v := by
  apply viewAt_of_col _ j r (broadcastColumn v chunk.length) _ (broadcast_get v _).1 _ ((broadcast_get v _).2.2 r hr)
  unfold chunkBatch
  rw [fold_column, find_rev_some hnd hm]

/-- Every variable the batch binds is in its column space. -/
theorem chunkBatch_space (vid : Nat) (outer : List (Nat × V F)) (chunk : List (V F)) :
    vid < (chunkBatch vid outer chunk).cols.length ∧
      ∀ p ∈ outer, p.1 < (chunkBatch vid outer chunk).cols.length := by
  have h0 := setColumn_cols_length (Batch.new0 0 : Batch F) vid (loopVarColumn chunk)
  obtain ⟨h1, h2⟩ := fold_cols_length chunk.length outer ((Batch.new0 0 : Batch F).setColumn vid (loopVarColumn chunk))
  have h00 : (Batch.new0 0 : Batch F).cols.length = 0 := rfl
  rw [h00] at h0
  refine ⟨?_, h2⟩
  unfold chunkBatch; omega

end Pr3437.ChunkBatch
