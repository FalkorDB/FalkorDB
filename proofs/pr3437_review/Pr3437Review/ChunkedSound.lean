import Pr3437Review.Chunked

/-!
# PR #3437: `list_loop_columnar` is sound (split out of `Chunked.lean`)

`passedIdx_map`: the passed row indices pick out the passed elements; `tested_spec`: the
tested column is main's predicate values read through each lane; **`columnar_sound`**: a chunk
the columnar path finishes gives exactly what main's loop gives.
-/

namespace Pr3437.ListLoop
open FalkorCore FalkorStrList

variable {F : Type}

/-! ## `passed` as indices -/

theorem filterIdx_map {α β : Type} (P : β → Bool) (d : α) (dq : β) :
    ∀ (chunk : List α) (qs : List β), chunk.length = qs.length →
      ((List.range chunk.length).filter (fun i => P (qs.getD i dq))).map (chunk.getD · d) =
        ((chunk.zip qs).filter (fun q => P q.2)).map (·.1)
  | [], _, _ => rfl
  | _ :: _, [], h => by simp at h
  | x :: xs, q :: qs, h => by
    simp only [List.length_cons, Nat.add_right_cancel_iff] at h
    have ih := filterIdx_map P d dq xs qs h
    rw [List.length_cons, List.range_succ_eq_map, List.filter_cons, List.filter_map, List.zip_cons_cons,
      List.filter_cons]
    have hc : ((fun i => P ((q :: qs).getD i dq)) ∘ Nat.succ) = fun i => P (qs.getD i dq) := by
      funext i; simp
    rw [hc]
    by_cases hp : P q
    · simp only [List.getD_cons_zero, hp, ite_true, List.map_cons, List.map_map]
      congr 1
    · simp only [List.getD_cons_zero, hp, Bool.false_eq_true, ite_false, List.map_map]
      rw [← ih]; rfl

theorem range_map_getD {α : Type} (d : α) (chunk : List α) :
    (List.range chunk.length).map (chunk.getD · d) = chunk := by
  apply List.ext_getElem
  · simp
  · intro i h1 h2; simp at h1; simp [List.getElem?_eq_getElem h1]

/-- The passed indices pick out exactly the passed elements. -/
theorem passedIdx_map (chunk : List (V F)) (tested : Option (EC F))
    (hlen : ∀ col, tested = some col → col.vals'.length = chunk.length) :
    (passedIdx chunk.length tested).map (chunk.getD · .null) = passedOf chunk tested := by
  cases tested with
  | none => exact range_map_getD _ chunk
  | some col =>
    have hl := hlen col rfl
    cases col with
    | bools cs =>
      simp only [EC.vals', List.length_map] at hl
      simp only [passedIdx, passedOf]
      rw [← filterIdx_map (fun q : Bool × Bool => q.1 && !q.2) .null (false, true) chunk cs hl.symm]
      congr 1
      apply List.filter_congr
      intro i _
      simp only [List.getD_eq_getElem?_getD]; cases cs[i]? <;> rfl
    | vals vs =>
      simp only [EC.vals'] at hl
      simp only [passedIdx, passedOf, filterPass]
      exact filterIdx_map isTrue .null .null chunk vs hl.symm

theorem passedIdx_lt (len : Nat) (tested : Option (EC F)) : ∀ r ∈ passedIdx len tested, r < len := by
  intro r hr
  cases tested with
  | none => exact List.mem_range.1 hr
  | some col => cases col <;> simp only [passedIdx, List.mem_filter, List.mem_range] at hr <;> exact hr.1

/-! ## The columnar half is sound -/

section
variable (b : Body F) (c : Col F)

/-- Whatever the tested column is, it is main's predicate values `ps`, read through each
lane's own fast path. -/
theorem tested_spec (hd : b.Drops) (hc : c.Sound b.pe b.pj) (chunk : List (V F)) (tested : Option (EC F))
    (h : (if b.hasPred then (c.colP chunk).map some else pure none) = .ok tested) :
    ∃ ps, chunk.mapM b.pe = .ok ps ∧ passedOf chunk tested = filterPass chunk ps ∧
      countLocal chunk.length tested = ps.foldlM cstep (0, 0, 0) ∧
      tallyNoProj chunk.length tested = (filterPass chunk ps).length ∧
      (∀ col, tested = some col → col.vals'.length = chunk.length) := by
  cases hp : b.hasPred
  · simp only [hp, Bool.false_eq_true, ite_false, pure, Except.pure, Except.ok.injEq] at h; subst h
    refine ⟨chunk.map fun _ => .bool true, ?_, ?_, ?_, ?_, by simp⟩
    · have : ∀ xs : List (V F), xs.mapM b.pe = .ok (xs.map fun _ => .bool true) := by
        intro xs; induction xs with
        | nil => rfl
        | cons x xs ih => simp [List.mapM_cons, hd.predTrue hp x, ih, bind, Except.bind, pure, Except.pure]
      exact this chunk
    · simp [passedOf, filterPass_trues]
    · simp [countLocal, cstep_trues]
    · simp [tallyNoProj, filterPass_trues]
  · simp only [hp, ite_true] at h
    cases hcol : c.colP chunk with
    | error e => simp [hcol, Except.map] at h
    | ok col =>
      simp only [hcol, Except.map, Except.ok.injEq] at h; subst h
      have hm := hc.pred chunk col hcol
      have hl := mapM_length hm
      refine ⟨col.vals', hm, ?_, ?_, ?_, fun c' hc' => by cases hc'; exact hl⟩
      · cases col with
        | bools cs => simp only [passedOf, EC.vals']; exact passed_bools chunk cs
        | vals vs => rfl
      · cases col with
        | bools cs => simp only [countLocal, EC.vals']; exact count_bools cs 0 0 0
        | vals vs => rfl
      · cases col with
        | bools cs =>
          simp only [EC.vals', List.length_map] at hl
          simp only [tallyNoProj, EC.vals']; exact tally_bools chunk cs hl.symm
        | vals vs =>
          simp only [EC.vals'] at hl
          simp only [tallyNoProj, EC.vals']; rw [filterPass_length chunk vs hl.symm]

/-- **A chunk the columnar path finishes gives exactly what main's loop gives.** -/
theorem columnar_sound (hd : b.Drops) (hc : c.Sound b.pe b.pj) (s : Sink F) (chunk : List (V F))
    (s' : Sink F) (h : columnar b c s chunk = .ok s') : ref b.pe b.pj s chunk = .ok s' := by
  unfold columnar at h
  cases ht : (if b.hasPred then (c.colP chunk).map some else pure none) with
  | error e => rw [ht] at h; simp [bind, Except.bind] at h
  | ok tested =>
    rw [ht] at h; simp only [bind, Except.bind] at h
    obtain ⟨ps, hm, hpass, hcount, htally, htl⟩ := tested_spec b c hd hc chunk tested ht
    have hpi : (passedIdx chunk.length tested).map (chunk.getD · .null) = filterPass chunk ps := by
      rw [passedIdx_map chunk tested htl, hpass]
    have hplen : (passedIdx chunk.length tested).length = (filterPass chunk ps).length := by
      rw [← hpi, List.length_map]
    have hemp : (passedIdx chunk.length tested).isEmpty = (filterPass chunk ps).isEmpty := by
      rw [← hpi]; cases passedIdx chunk.length tested <;> rfl
    have hlt := passedIdx_lt chunk.length tested
    cases s with
    | count t f n =>
      simp only [hcount] at h
      rw [ref_count_pre _ _ chunk ps t f n hm, cstep_fold_shift]
      cases hps : ps.foldlM cstep (0, 0, 0) with
      | error e => simp [hps, Except.map] at h
      | ok r => simp only [hps, Except.map, Except.ok.injEq] at h ⊢; exact h
    | tally k =>
      rw [ref_tally_pre _ _ chunk ps k hm]
      cases hj : b.hasProj
      · simp only [hj, Bool.not_false, ite_true, pure, Except.pure, Except.ok.injEq] at h
        rw [mapM_id (hd.projId hj)]; simp [Except.map, ← h, htally]
      · simp only [hj, Bool.not_true, Bool.false_eq_true, ite_false, hemp, hplen] at h
        by_cases he : (filterPass chunk ps).isEmpty = true
        · simp only [he, ite_true, pure, Except.pure, Except.ok.injEq] at h
          rw [List.isEmpty_iff.1 he] at h ⊢; simp [← h, pure, Except.pure, Except.map]
        · simp only [he, Bool.false_eq_true, ite_false] at h
          cases hJ : c.colJ chunk (passedIdx chunk.length tested) with
          | error e => simp [hJ, Except.map, bind, Except.bind] at h
          | ok col =>
            simp only [hJ, Except.map, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
            obtain ⟨ys, hys⟩ := hc.proj _ _ col hJ hlt
            rw [hpi] at hys
            simp [hys, Except.map, ← h]
    | collect acc =>
      rw [ref_collect_pre _ _ chunk ps acc hm]
      cases hj : b.hasProj
      · simp only [hj, Bool.not_false, ite_true, hplen, hpi] at h
        rw [mapM_id (hd.projId hj)]
        by_cases hlen : (filterPass chunk ps).length = chunk.length
        · simp only [hlen, ite_true, pure, Except.pure, Except.ok.injEq] at h
          rw [filterPass_full chunk ps hlen]; simp [Except.map, ← h]
        · simp only [hlen, ite_false, pure, Except.pure, Except.ok.injEq] at h
          simp [Except.map, ← h]
      · simp only [hj, Bool.not_true, Bool.false_eq_true, ite_false, hemp] at h
        by_cases he : (filterPass chunk ps).isEmpty = true
        · simp only [he, ite_true, pure, Except.pure, Except.ok.injEq] at h
          rw [List.isEmpty_iff.1 he]; simp [← h, pure, Except.pure, Except.map]
        · simp only [he, Bool.false_eq_true, ite_false] at h
          cases hJ : c.colJv chunk (passedIdx chunk.length tested) with
          | error e => simp [hJ, Except.map] at h
          | ok ys =>
            simp only [hJ, Except.map, Except.ok.injEq] at h
            have := hc.projV _ _ ys hJ hlt
            rw [hpi] at this
            simp [this, Except.map, ← h]

end

end Pr3437.ListLoop
