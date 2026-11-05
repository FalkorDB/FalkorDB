import OpsTraverse.ShortestPaths
/-
allShortestPaths, part 1: what one BFS expansion (`(step g b cur).foldl relax`) does, for the
non-cycle case (`src ≠ dst`, so `is_cycle = false`) with `min_hops = 1` (the planner rejects any
other minimum: "allShortestPaths(...) does not support a minimal length different from 1",
same as C).
-/
namespace OpsTraverse.ASP

open OpsTraverse.VarLen

theorem relax_nc (src dst maxH cur cd : Nat) (st : St) (p : Nat × Nat) :
    relax src dst 1 maxH false cur cd st p =
      match st.dist p.2 with
      | some ed => if cd + 1 == ed then { st with preds := st.preds ++ [(p.2, cur, p.1)] } else st
      | none =>
        { dist := upd st.dist p.2 (cd + 1)
          preds := st.preds ++ [(p.2, cur, p.1)]
          sd := if p.2 == dst && 1 ≤ cd + 1 then some (cd + 1) else st.sd
          queue := if cd + 1 < maxH then st.queue ++ [p.2] else st.queue } := by
  unfold relax; cases h : st.dist p.2 <;> simp [h]

/-- How an intermediate state `X` of the fold relates to the state `G` before it, after the
neighbour list prefix `ps` has been relaxed. -/
structure Rel (g : Graph) (b : Bool) (dst maxH cur dcur : Nat) (G X : St) (ps : List (Nat × Nat)) : Prop where
  dist_keep : ∀ v d, G.dist v = some d → X.dist v = some d
  dist_new : ∀ v d, X.dist v = some d → G.dist v = none → d = dcur + 1 ∧ ∃ e, (e, v) ∈ step g b cur
  preds_ext : ∃ nw, X.preds = G.preds ++ nw ∧ (nw.map fun x => (x.2.2, x.1)).Sublist ps ∧
    ∀ x ∈ nw, x.2.1 = cur ∧ (x.2.2, x.1) ∈ step g b cur ∧ X.dist x.1 = some (dcur + 1)
  queue_ext : ∃ news, X.queue = G.queue ++ news ∧ news.Nodup ∧
    (∀ v ∈ news, G.dist v = none ∧ X.dist v = some (dcur + 1) ∧ dcur + 1 < maxH) ∧
    (∀ v, G.dist v = none → X.dist v ≠ none → dcur + 1 < maxH → v ∈ news)
  sd_rel : G.sd = G.dist dst → X.sd = X.dist dst
  done : ∀ e n, (e, n) ∈ ps → X.dist n ≠ none ∧ (X.dist n = some (dcur + 1) → (n, cur, e) ∈ X.preds)

theorem rel_refl (g : Graph) (b : Bool) (dst maxH cur dcur : Nat) (G : St) : Rel g b dst maxH cur dcur G G [] where
  dist_keep _ _ h := h
  dist_new _ _ h h' := by rw [h'] at h; cases h
  preds_ext := ⟨[], by simp, by simp, by simp⟩
  queue_ext := ⟨[], by simp, List.nodup_nil, by simp, fun v h1 h2 _ => absurd h1 h2⟩
  sd_rel h := h
  done _ _ h := by simp at h

theorem rel_step (g : Graph) (b : Bool) (src dst maxH cur dcur : Nat) (G X : St) (ps : List (Nat × Nat))
    (p : Nat × Nat) (hp : p ∈ step g b cur) (hq : ∀ v ∈ G.queue, G.dist v ≠ none)
    (hr : Rel g b dst maxH cur dcur G X ps) :
    Rel g b dst maxH cur dcur G (relax src dst 1 maxH false cur dcur X p) (ps ++ [p]) := by
  obtain ⟨e, n⟩ := p
  rw [relax_nc]
  simp only
  obtain ⟨nw, hnw, hsub, hnw'⟩ := hr.preds_ext
  obtain ⟨news, hq1, hq2, hq3, hq4⟩ := hr.queue_ext
  cases hxn : X.dist n with
  | some ed =>
    simp only
    by_cases he : (dcur + 1 == ed) = true
    · simp only [he, ite_true]
      have hed : ed = dcur + 1 := (by simpa using he : dcur + 1 = ed).symm
      subst hed
      refine ⟨hr.dist_keep, hr.dist_new, ⟨nw ++ [(n, cur, e)], by simp [hnw],
        by simpa using hsub.append (List.Sublist.refl [(e, n)]), ?_⟩,
        ⟨news, hq1, hq2, hq3, hq4⟩, hr.sd_rel, ?_⟩
      · intro x hx
        simp only [List.mem_append, List.mem_singleton] at hx
        rcases hx with hx | rfl
        · exact hnw' x hx
        · exact ⟨rfl, hp, hxn⟩
      · intro e' n' hm
        simp only [List.mem_append, List.mem_singleton, Prod.mk.injEq] at hm
        rcases hm with hm | ⟨rfl, rfl⟩
        · obtain ⟨h1, h2⟩ := hr.done e' n' hm
          exact ⟨h1, fun h => List.mem_append_left _ (h2 h)⟩
        · exact ⟨by simp [hxn], fun _ => by simp⟩
    · simp only [he, Bool.false_eq_true, ite_false]
      refine ⟨hr.dist_keep, hr.dist_new, ⟨nw, hnw, hsub.trans (List.sublist_append_left _ _), hnw'⟩,
        ⟨news, hq1, hq2, hq3, hq4⟩, hr.sd_rel, ?_⟩
      intro e' n' hm
      simp only [List.mem_append, List.mem_singleton, Prod.mk.injEq] at hm
      rcases hm with hm | ⟨rfl, rfl⟩
      · exact hr.done e' n' hm
      · refine ⟨by simp [hxn], fun h => ?_⟩
        rw [hxn] at h; simp only [Option.some.injEq] at h; subst h; simp at he
  | none =>
    have hGn : G.dist n = none := by
      cases h : G.dist n with
      | none => rfl
      | some d => rw [hr.dist_keep n d h] at hxn; cases hxn
    have hupd : ∀ v, upd X.dist n (dcur + 1) v = if v = n then some (dcur + 1) else X.dist v := fun v => rfl
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
    · intro v d hv
      simp only [hupd]
      have := hr.dist_keep v d hv
      by_cases hvn : v = n
      · subst hvn; rw [hGn] at hv; cases hv
      · simp [hvn, this]
    · intro v d hv hGv
      simp only [hupd] at hv
      by_cases hvn : v = n
      · subst hvn; simp at hv; exact ⟨hv.symm, e, hp⟩
      · simp only [hvn, ite_false] at hv; exact hr.dist_new v d hv hGv
    · refine ⟨nw ++ [(n, cur, e)], by simp [hnw], by simpa using hsub.append (List.Sublist.refl [(e, n)]), ?_⟩
      intro x hx
      simp only [List.mem_append, List.mem_singleton] at hx
      rcases hx with hx | rfl
      · obtain ⟨h1, h2, h3⟩ := hnw' x hx
        refine ⟨h1, h2, ?_⟩
        simp only [hupd]; split
        · rfl
        · exact h3
      · exact ⟨rfl, hp, by simp [hupd]⟩
    · by_cases hm : dcur + 1 < maxH
      · rw [if_pos hm]
        refine ⟨news ++ [n], by simp [hq1], ?_, ?_, ?_⟩
        · rw [List.nodup_append]
          refine ⟨hq2, by simp, ?_⟩
          intro a ha c hc; simp at hc; subst hc; intro hac; subst hac
          have := (hq3 a ha).2.1; rw [hxn] at this; cases this
        · intro v hv
          simp only [List.mem_append, List.mem_singleton] at hv
          rcases hv with hv | rfl
          · obtain ⟨h1, h2, h3⟩ := hq3 v hv
            refine ⟨h1, ?_, h3⟩
            simp only [hupd]; split
            · rfl
            · exact h2
          · exact ⟨hGn, by simp [hupd], hm⟩
        · intro v hGv hXv hlt
          simp only [hupd] at hXv
          by_cases hvn : v = n
          · subst hvn; simp
          · simp only [hvn, ite_false] at hXv; exact List.mem_append_left _ (hq4 v hGv hXv hlt)
      · rw [if_neg hm]
        refine ⟨news, hq1, hq2, ?_, fun v _ _ hlt => absurd hlt hm⟩
        intro v hv
        obtain ⟨h1, h2, h3⟩ := hq3 v hv
        refine ⟨h1, ?_, h3⟩
        simp only [hupd]; split
        · rfl
        · exact h2
    · intro hsd
      have hX := hr.sd_rel hsd
      simp only [hupd]
      by_cases hd : n = dst
      · subst hd; simp
      · have : (n == dst) = false := by simpa using hd
        simp only [this, Bool.false_and, Bool.false_eq_true, ite_false]
        rw [hX]; simp [Ne.symm hd]
    · intro e' n' hm'
      simp only [List.mem_append, List.mem_singleton, Prod.mk.injEq] at hm'
      rcases hm' with hm' | ⟨rfl, rfl⟩
      · obtain ⟨h1, h2⟩ := hr.done e' n' hm'
        refine ⟨?_, fun h => ?_⟩
        · simp only [hupd]; split
          · simp
          · exact h1
        · simp only [hupd] at h
          by_cases hn' : n' = n
          · subst hn'; exact List.mem_append_left _ (by
              rw [hxn] at h1; exact absurd rfl h1)
          · simp only [hn', ite_false] at h; exact List.mem_append_left _ (h2 h)
      · exact ⟨by simp [hupd], fun _ => by simp⟩

theorem rel_fold (g : Graph) (b : Bool) (src dst maxH cur dcur : Nat) (G : St)
    (hq : ∀ v ∈ G.queue, G.dist v ≠ none) :
    ∀ (ps : List (Nat × Nat)) (pre : List (Nat × Nat)) (X : St), (∀ p ∈ ps, p ∈ step g b cur) →
      Rel g b dst maxH cur dcur G X pre →
      Rel g b dst maxH cur dcur G (ps.foldl (relax src dst 1 maxH false cur dcur) X) (pre ++ ps)
  | [], pre, X, _, h => by simpa using h
  | p :: ps, pre, X, hps, h => by
      simp only [List.foldl_cons]
      have := rel_fold g b src dst maxH cur dcur G hq ps (pre ++ [p]) _
        (fun q hq' => hps q (List.mem_cons_of_mem _ hq'))
        (rel_step g b src dst maxH cur dcur G X pre p (hps p (List.mem_cons_self ..)) hq h)
      simpa using this

/-- One full expansion of `cur` relates the popped state to the result over all neighbours. -/
theorem expand_rel (g : Graph) (b : Bool) (src dst maxH cur dcur : Nat) (G : St)
    (hq : ∀ v ∈ G.queue, G.dist v ≠ none) :
    Rel g b dst maxH cur dcur G ((step g b cur).foldl (relax src dst 1 maxH false cur dcur) G) (step g b cur) := by
  have := rel_fold g b src dst maxH cur dcur G hq (step g b cur) [] G (fun _ h => h) (rel_refl g b dst maxH cur dcur G)
  simpa using this

end OpsTraverse.ASP
