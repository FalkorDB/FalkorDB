import FalkorOpsApply.Correlated
/-
# The per-origin `ValueHashJoin` (#3035) is separable on the batches Apply builds

`Correlated.Separable` quantifies over every argument list, so it cannot hold of a join that
compares origins (two args with the same origin would join). The batches Apply/Optional/Merge hand a
sub-plan are always `tag xs 0` (`clone_active_rows_seq_origin`: distinct sequential origins), so
the property the batched idiom needs is `SepT`, separability on those batches only. `applyBatched_eq_ref_T`
shows `SepT` suffices; `vhjPO_sepT` shows the #3035 join has it (`vhj_not_separable` is the pre-#3035
join, `Correlated.vhj`, which compared no origins).
-/
namespace VhjSep
open Correlated

variable {α β γ a b k : Type}

/-- Separable on sequentially tagged batches. -/
def SepT (f : SubPlan α β) : Prop :=
  ∀ xs : List α, f (tag xs 0) = (tag xs 0).flatMap (fun p => (single f p.2).map (fun y => (p.1, y)))

theorem sepT_of_separable (f : SubPlan α β) (h : Separable f) : SepT f := fun xs => h _

/-- `applyBatched_eq_ref` needs only `SepT` (its proof only ever instantiates `Separable` at `tag xs 0`). -/
theorem applyBatched_eq_ref_T (merge : α → β → γ) (f : SubPlan α β) (hf : SepT f) (xs : List α) :
    applyBatched merge f xs = applyRef merge f xs := by
  unfold applyBatched applyRef
  rw [hf]
  have := lookup_tagged merge (single f) [] xs
  simpa using this

/-- The per-origin join: a left row joins only right rows carrying the same origin (`origin_row`),
the tables being one per origin (`CartVhj.buildPO`, `fillMatchesPO`). -/
def vhjPO [BEq k] (lhs : α → List a) (rhs : α → List b) (lk : a → k) (rk : b → k) : SubPlan α (a × b) :=
  fun args =>
    let right := args.flatMap (fun p => (rhs p.2).map (fun y => (p.1, y)))
    (args.flatMap (fun p => (lhs p.2).map (fun y => (p.1, y)))).flatMap
      (fun l => (right.filter (fun r => r.1 == l.1 && lk l.2 == rk r.2)).map
        (fun r => (l.1, (l.2, r.2))))

/-- One row's join: left rows against the right rows of the same row. -/
def rowJoin [BEq k] (lhs : α → List a) (rhs : α → List b) (lk : a → k) (rk : b → k) (x : α) : List (a × b) :=
  (lhs x).flatMap (fun l => ((rhs x).filter (fun r => lk l == rk r)).map (fun r => (l, r)))

theorem vhjPO_tag [BEq k] (lhs : α → List a) (rhs : α → List b) (lk : a → k) (rk : b → k) :
    ∀ (xs : List α) (m : Nat),
      vhjPO lhs rhs lk rk (tag xs m) =
        (tag xs m).flatMap (fun p => (rowJoin lhs rhs lk rk p.2).map (fun y => (p.1, y)))
  | [], m => by simp [vhjPO, tag]
  | x :: xs, m => by
    have ih := vhjPO_tag lhs rhs lk rk xs (m + 1)
    have hR : ∀ r ∈ (tag xs (m+1)).flatMap (fun p => (rhs p.2).map (fun y => (p.1, y))), m + 1 ≤ r.1 := by
      intro r hr
      simp only [List.mem_flatMap, List.mem_map] at hr
      obtain ⟨q, hq, y, hy, rfl⟩ := hr
      exact (mem_tag (x := q.2) (by simpa using hq)).1
    have hL : ∀ l ∈ (tag xs (m+1)).flatMap (fun p => (lhs p.2).map (fun y => (p.1, y))), m + 1 ≤ l.1 := by
      intro r hr
      simp only [List.mem_flatMap, List.mem_map] at hr
      obtain ⟨q, hq, y, hy, rfl⟩ := hr
      exact (mem_tag (x := q.2) (by simpa using hq)).1
    unfold vhjPO at ih ⊢
    simp only [tag, List.flatMap_cons, List.map_cons] at ih ⊢
    rw [← ih]
    simp only [List.map_append, List.flatMap_append, List.flatMap_map]
    congr 1
    · -- the row's own left rows
      unfold rowJoin
      rw [List.map_flatMap]
      simp only [List.flatMap_def]
      congr 1
      apply List.map_congr_left
      intro l _
      have hnil : List.filter (fun r => r.1 == m && lk l == rk r.2)
          ((tag xs (m + 1)).flatMap (fun p => (rhs p.2).map (fun y => (p.1, y)))) = [] := by
        rw [List.filter_eq_nil_iff]
        intro r hr
        have := hR r hr
        simp only [Bool.and_eq_true, beq_iff_eq, not_and]
        intro h; omega
      simp only [List.flatMap_def] at hnil
      rw [List.filter_append, hnil, List.append_nil, List.filter_map]
      simp [List.map_map, Function.comp_def]
    · simp only [List.flatMap_def]
      congr 1
      apply List.map_congr_left
      intro l hl
      have := hL l hl
      simp only [List.filter_append, List.map_append]
      have hnil : List.filter (fun r => r.1 == l.1 && lk l.2 == rk r.2)
          ((rhs x).map (fun y => (m, y))) = [] := by
        rw [List.filter_eq_nil_iff]
        intro r hr
        simp only [List.mem_map] at hr
        obtain ⟨y, _, rfl⟩ := hr
        simp only [Bool.and_eq_true, beq_iff_eq, not_and]
        intro h; omega
      simp [hnil]

/-- Running the per-origin join on one row alone is that row's join. -/
theorem vhjPO_single [BEq k] (lhs : α → List a) (rhs : α → List b) (lk : a → k) (rk : b → k) (x : α) :
    single (vhjPO lhs rhs lk rk) x = rowJoin lhs rhs lk rk x := by
  have := vhjPO_tag lhs rhs lk rk [x] 0
  simp only [tag] at this
  simp [single, this, List.map_map, Function.comp_def]

/-- **PROVEN** (#3035, `ValueHashJoinOp` with `per_origin`): the per-origin join is separable on the
batches Apply/Optional/Merge hand a sub-plan (`tag xs 0`): its result is the concatenation, in
argument order, of each row's own join re-tagged with that row's origin. The known counterexample
`Correlated.vhj_batched_16_vs_8` (16 rows against 8) is the join WITHOUT the origin test. -/
theorem vhjPO_sepT [BEq k] (lhs : α → List a) (rhs : α → List b) (lk : a → k) (rk : b → k) :
    SepT (vhjPO lhs rhs lk rk) := by
  intro xs
  rw [vhjPO_tag]
  simp only [vhjPO_single]

/-- Hence batched Apply over a per-origin ValueHashJoin equals the per-row reference. -/
theorem applyBatched_vhjPO [BEq k] (merge : α → (a × b) → γ) (lhs : α → List a) (rhs : α → List b)
    (lk : a → k) (rk : b → k) (xs : List α) :
    applyBatched merge (vhjPO lhs rhs lk rk) xs = applyRef merge (vhjPO lhs rhs lk rk) xs :=
  applyBatched_eq_ref_T merge _ (vhjPO_sepT lhs rhs lk rk) xs

/-- The #3012 repro under the per-origin join: batched = per-row = 8 rows. -/
theorem vhjPO_16_vs_8_fixed :
    (applyBatched (fun x y => (x, y)) (vhjPO scanA scanB (fun _ => (1 : Nat)) (fun _ => (1 : Nat))) [1, 2]).length = 8 := by
  rw [applyBatched_vhjPO]; decide

end VhjSep
