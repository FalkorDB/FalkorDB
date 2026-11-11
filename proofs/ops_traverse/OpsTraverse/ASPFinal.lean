import OpsTraverse.ASPStep
/-
allShortestPaths, part 4: the result. For `src ≠ dst` (no cycle) and `min_hops = 1`, once the
BFS has drained its queue:
* **minimality** — every returned path is a walk `src → dst` no longer than ANY walk between
  them (`asp_minimal`);
* **completeness** — every shortest walk of at most `max_hops` edges is returned
  (`asp_complete`);
* **exactly once** — the returned list has no duplicates (`asp_nodup`).
-/
namespace OpsTraverse.ASP

open OpsTraverse.VarLen

theorem inv_bstep (g : Graph) (hwf : g.WF) (b : Bool) (src dst maxH : Nat) (st : St)
    (hi : Inv g b src dst maxH st) : Inv g b src dst maxH (bstep g b src dst maxH st) := by
  unfold bstep
  cases hq : st.queue with
  | nil => simpa [hq] using hi
  | cons cur q =>
    simp only
    obtain ⟨dcur, hc, hcm⟩ := hi.q_dist cur (by rw [hq]; simp)
    simp only [hc, Option.getD_some]
    split
    · rename_i hcond
      apply inv_skip g b src dst maxH st hi cur q hq dcur hc
      simp only [Bool.or_eq_true, decide_eq_true_eq] at hcond
      rcases hcond with h | h
      · cases hs : st.sd with
        | none => rw [hs] at h; simp at h
        | some s => rw [hs] at h; simp at h; exact ⟨s, rfl, h⟩
      · omega
    · rename_i hcond
      apply inv_expand g hwf b src dst maxH st hi cur q hq dcur hc
      intro s hs
      simp only [Bool.or_eq_true, decide_eq_true_eq, not_or] at hcond
      rw [hs] at hcond
      simp at hcond; exact hcond.1

theorem inv_bfs (g : Graph) (hwf : g.WF) (b : Bool) (src dst maxH : Nat) :
    ∀ (fuel : Nat) (st : St), Inv g b src dst maxH st → Inv g b src dst maxH (bfs g b src dst 1 maxH false fuel st)
  | 0, _, h => h
  | fuel + 1, st, h => by
      cases hq : st.queue with
      | nil => simp only [bfs, hq]; exact h
      | cons cur q =>
        rw [bfs_unfold g b src dst maxH fuel st cur q hq]
        exact inv_bfs g hwf b src dst maxH fuel _ (inv_bstep g hwf b src dst maxH st h)

/-! ## Walk plumbing -/

theorem walk_append {g : Graph} {b : Bool} : ∀ {s m t : Nat} {es1 es2 : List Nat},
    Walk g b s es1 m → Walk g b m es2 t → Walk g b s (es1 ++ es2) t
  | _, _, _, [], _, .nil _, h => h
  | _, _, _, _ :: _, _, .cons hp hw, h => .cons hp (walk_append hw h)

theorem walk_split {g : Graph} {b : Bool} : ∀ {s t : Nat} {es : List Nat}, Walk g b s es t →
    ∀ k, ∃ m, Walk g b s (es.take k) m ∧ Walk g b m (es.drop k) t
  | _, _, [], .nil s, k => ⟨s, by simp; exact .nil _, by simp; exact .nil _⟩
  | _, _, _ :: _, .cons hp hw, 0 => ⟨_, by simp; exact .nil _, by simp; exact .cons hp hw⟩
  | _, _, _ :: _, .cons hp hw, k + 1 => by
      obtain ⟨m, h1, h2⟩ := walk_split hw k
      exact ⟨m, by simp; exact .cons hp h1, by simpa using h2⟩

/-- Prefixes of a shortest walk are shortest. -/
theorem prefix_isDist (g : Graph) (b : Bool) (src dst : Nat) (es : List Nat) (hw : Walk g b src es dst)
    (hmin : ∀ es', Walk g b src es' dst → es.length ≤ es'.length) (k : Nat) (hk : k ≤ es.length) (m : Nat)
    (h1 : Walk g b src (es.take k) m) (h2 : Walk g b m (es.drop k) dst) : IsDist g b src m k := by
  refine ⟨⟨es.take k, by simp; omega, h1⟩, fun j hj hr => ?_⟩
  obtain ⟨es1, hl, hw1⟩ := hr
  have := hmin _ (walk_append hw1 h2)
  simp at this; omega

/-! ## Predecessor chains and the backtrack -/

/-- A walk every edge of which is a recorded predecessor entry. -/
inductive PredWalk (st : St) (src : Nat) : List Nat → Nat → Prop
  | nil : PredWalk st src [] src
  | snoc {es : List Nat} {w m e : Nat} : PredWalk st src es w → (m, w, e) ∈ st.preds → PredWalk st src (es ++ [e]) m

theorem predWalk_dist {g : Graph} {b : Bool} {src dst maxH : Nat} {st : St} (hi : Inv g b src dst maxH st) :
    ∀ {es m}, PredWalk st src es m → st.dist m = some es.length
  | _, _, .nil => hi.src_d
  | _, _, .snoc hp hm => by
      have ih := predWalk_dist hi hp
      obtain ⟨_, du, h1, h2⟩ := hi.preds_ok _ _ _ hm
      rw [ih] at h1; cases h1; simpa using h2

theorem mem_predsOf {st : St} {v u e : Nat} (h : (v, u, e) ∈ st.preds) : (u, e) ∈ predsOf st v := by
  simp only [predsOf, List.mem_map, List.mem_filter, beq_iff_eq]
  exact ⟨(v, u, e), ⟨h, rfl⟩, rfl⟩

theorem back_has {g : Graph} {b : Bool} {src dst maxH : Nat} {st : St} (hi : Inv g b src dst maxH st) :
    ∀ {es m}, PredWalk st src es m → ∀ (acc : List Nat) (fuel : Nat), es.length + 1 ≤ fuel → (acc ≠ [] ∨ es ≠ []) →
      acc ++ es.reverse ∈ back st src fuel m acc
  | _, _, .nil, acc, fuel, hf, hne => by
      obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by omega⟩
      have hacc : acc ≠ [] := by simpa using hne
      unfold back
      have : (src == src && !acc.isEmpty) = true := by cases acc <;> simp_all
      rw [if_pos this]; simp
  | _, _, @PredWalk.snoc _ _ es w m e hp hm, acc, fuel, hf, _ => by
      obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by simp at hf; omega⟩
      unfold back
      have hdm := predWalk_dist hi (PredWalk.snoc hp hm)
      have hms : m ≠ src := by
        intro h; subst h; rw [hi.src_d] at hdm; simp at hdm
      have : (m == src && !acc.isEmpty) = false := by simp [hms]
      simp only [this, Bool.false_eq_true, ite_false, List.mem_flatMap]
      refine ⟨(w, e), mem_predsOf hm, ?_⟩
      have := back_has hi hp (acc ++ [e]) f (by simp at hf; omega) (Or.inl (by simp))
      simpa using this

theorem back_len {g : Graph} {b : Bool} {src dst maxH : Nat} {st : St} (hi : Inv g b src dst maxH st) :
    ∀ (fuel node : Nat) (acc res : List Nat) (dn : Nat), st.dist node = some dn →
      res ∈ back st src fuel node acc → res.length = acc.length + dn
  | 0, _, _, _, _, _, h => by simp [back] at h
  | fuel + 1, node, acc, res, dn, hd, h => by
      unfold back at h
      by_cases hc : (node == src && !acc.isEmpty) = true
      · rw [if_pos hc] at h
        simp only [List.mem_singleton] at h; subst h
        simp only [Bool.and_eq_true, beq_iff_eq] at hc
        rw [hc.1, hi.src_d] at hd; cases hd; simp
      · rw [if_neg hc] at h
        simp only [List.mem_flatMap] at h
        obtain ⟨⟨u, e⟩, hp, hr⟩ := h
        simp only [predsOf, List.mem_map, List.mem_filter, beq_iff_eq] at hp
        obtain ⟨⟨v', u', e'⟩, ⟨hm, hv⟩, heq⟩ := hp
        simp only [Prod.mk.injEq] at heq; obtain ⟨rfl, rfl⟩ := heq
        simp only at hv; subst hv
        obtain ⟨_, du, h1, h2⟩ := hi.preds_ok _ _ _ hm
        rw [hd] at h2; cases h2
        have := back_len hi fuel u' (acc ++ [e']) res du h1 hr
        simp at this; omega

end OpsTraverse.ASP
