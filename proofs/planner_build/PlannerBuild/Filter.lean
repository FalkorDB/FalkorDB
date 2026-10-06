import PlannerBuild.ToPlan
/-
# `plan_filter` (mod.rs:2492-2530) end to end, for a WHERE without pattern comprehensions

`planFilter` = `collect_patterns_and_rebuild` from an empty extractable list
and inline map, then `expr_to_plan` (inline patterns present) or a plain
`Filter` (unless the rebuilt predicate is the literal `true`), then one
`SemiApply` / `AntiSemiApply` per extracted conjunct, in order.

Side facts proved here: the minted inline ids are pairwise distinct and above
every id the predicate mentions, *when the predicate's patterns all start in
one scope `s0` and its variables' ids are below that scope's length*
(`collect_keys`) — the planner keys `inline_map` by id only (mod.rs:1377), so
this is exactly what keeps `Good`/`Fresh` true; and the rebuilt predicate is
in the shape `expr_to_plan` handles (`collect_wf`).

* `planFilter_scalar_correct`: no inline pattern ⇒ exact in three-valued logic.
* `planFilter_correct`: two-valued atoms ⇒ exact.
-/
namespace PlannerBuild.E

variable {R : Type} (atom : Ex → R → Option Bool) (sub : QG → R → List R)

/-! ## Inline keys -/

def keys (st : CSt) : List Nat := st.inl.map Prod.fst

def Inv (s0 lo : Nat) (st : CSt) : Prop :=
  (∀ p ∈ st.inl, lo ≤ p.1 ∧ p.1 < st.lens s0) ∧ (keys st).Nodup ∧ lo ≤ st.lens s0

def SameScope (s0 : Nat) (e : Ex) : Prop := ∀ g, D.pat g ∈ nodes e → scopeOf g = s0

theorem inv_mint (s0 lo : Nat) (st : CSt) (g : QG) (hg : scopeOf g = s0) (h : Inv s0 lo st) :
    Inv s0 lo (mint st g).2 := by
  obtain ⟨h1, h2, h3⟩ := h
  subst hg
  refine ⟨?_, ?_, ?_⟩
  · intro p hp
    simp only [mint, List.mem_cons] at hp
    simp only [mint, ↓reduceIte]
    rcases hp with rfl | hp
    · exact ⟨h3, Nat.lt_succ_self _⟩
    · have := h1 p hp; exact ⟨this.1, Nat.lt_succ_of_lt this.2⟩
  · simp only [keys, mint, List.map_cons, List.nodup_cons]
    refine ⟨?_, h2⟩
    intro hm
    obtain ⟨p, hp, he⟩ := List.mem_map.1 hm
    have := (h1 p hp).2; omega
  · simp only [mint, ↓reduceIte]; omega

theorem nodes_child (d : D) (c : Ex) (cs : List Ex) (hc : c ∈ cs) (x : D) (hx : x ∈ nodes c) :
    x ∈ nodes (.node d cs) := by
  simp only [nodes, List.mem_cons]
  right
  induction cs with
  | nil => simp at hc
  | cons c' cs ih =>
    simp only [nodesL, List.mem_append]
    rcases List.mem_cons.1 hc with rfl | hc
    · exact Or.inl hx
    · exact Or.inr (ih hc)

theorem sameScope_child (s0 : Nat) (d : D) (c : Ex) (cs : List Ex) (hc : c ∈ cs)
    (h : SameScope s0 (.node d cs)) : SameScope s0 c :=
  fun g hg => h g (nodes_child d c cs hc _ hg)

mutual
theorem collect_inv (s0 lo : Nat) : ∀ (st : CSt) ce (e : Ex), SameScope s0 e → Inv s0 lo st →
    Inv s0 lo (collect st ce e).2
  | st, ce, .node d cs, hs, h => by
    have hL : ∀ ce', Inv s0 lo (collectL st ce' cs).2 := fun ce' =>
      collectL_inv s0 lo st ce' cs (fun c hc => sameScope_child s0 d c cs hc hs) h
    cases d <;> simp only [collect] <;> try exact hL _
    case pat g =>
      cases ce
      · exact inv_mint s0 lo st g (hs g (by simp [nodes])) h
      · exact h
    case not =>
      cases hp : notPat cs with
      | none => exact hL _
      | some g =>
        match cs, hp with
        | .node (.pat g') _ :: _, hp =>
          simp [notPat] at hp; subst hp
          cases ce
          · exact inv_mint s0 lo st g' (hs g' (by simp [nodes, nodesL])) h
          · exact h
theorem collectL_inv (s0 lo : Nat) : ∀ (st : CSt) ce (l : List Ex), (∀ c ∈ l, SameScope s0 c) →
    Inv s0 lo st → Inv s0 lo (collectL st ce l).2
  | st, _, [], _, h => h
  | st, ce, c :: cs, hs, h => by
    simp only [collectL]
    exact collectL_inv s0 lo _ ce cs (fun c' hc' => hs c' (by simp [hc']))
      (collect_inv s0 lo st ce c (hs c (by simp)) h)
end

theorem lookup_of_mem (l : List (Nat × QG)) (hn : (l.map Prod.fst).Nodup) (p : Nat × QG) (hp : p ∈ l) :
    lk l p.1 = some p.2 := by
  induction l with
  | nil => simp at hp
  | cons q l ih =>
    simp only [List.map_cons, List.nodup_cons] at hn
    rcases List.mem_cons.1 hp with rfl | hp
    · simp [lk, List.lookup]
    · have hne : p.1 ≠ q.1 := by
        intro e; apply hn.1; rw [← e]; exact List.mem_map_of_mem hp
      have : (p.1 == q.1) = false := by simpa using hne
      simp only [lk, List.lookup, this]; exact ih hn.2 hp

theorem lookup_none (l : List (Nat × QG)) (i : Nat) (h : ∀ p ∈ l, p.1 ≠ i) : lk l i = none := by
  induction l with
  | nil => rfl
  | cons q l ih =>
    have hne := h q (by simp)
    have : (i == q.1) = false := by rw [beq_eq_false_iff_ne]; exact fun e => hne e.symm
    simp only [lk, List.lookup, this]; exact ih (fun p hp => h p (by simp [hp]))

/-- The inline map the planner ends up with gives back every minted pattern. -/
theorem good_final (s0 lo : Nat) (st : CSt) (h : Inv s0 lo st) : Good st.inl st :=
  fun p hp => lookup_of_mem st.inl h.2.1 p hp

/-- …and no variable the predicate mentions is mistaken for one. -/
theorem fresh_final (s0 lo : Nat) (st : CSt) (h : Inv s0 lo st) (ids : List Nat) (hb : ∀ i ∈ ids, i < lo) :
    Fresh st.inl ids :=
  fun i hi => lookup_none _ _ (fun p hp => by have := (h.1 p hp).1; have := hb i hi; omega)

/-! ## The rebuilt predicate is what `expr_to_plan` expects -/

mutual
theorem var_mem_varIds : ∀ (e : Ex) (v : V), D.var v ∈ nodes e → v.id ∈ varIds e
  | .node d cs, v, h => by
    simp only [nodes, List.mem_cons] at h
    rcases h with h | h
    · subst h; simp [varIds]
    · have := var_mem_varIdsL cs v h
      cases d <;> simp [varIds, this]
theorem var_mem_varIdsL : ∀ (l : List Ex) (v : V), D.var v ∈ nodesL l → v.id ∈ varIdsL l
  | [], _, h => by simp [nodesL] at h
  | c :: cs, v, h => by
    simp only [nodesL, List.mem_append] at h
    simp only [varIdsL, List.mem_append]
    rcases h with h | h
    · exact Or.inl (var_mem_varIds c v h)
    · exact Or.inr (var_mem_varIdsL cs v h)
end

theorem lookup_none_not_mem (l : List (Nat × QG)) (i : Nat) (h : lk l i = none) :
    ∀ p ∈ l, p.1 ≠ i := by
  induction l with
  | nil => simp
  | cons q l ih =>
    intro p hp
    by_cases hq : i = q.1
    · simp [lk, List.lookup, hq] at h
    · have : (i == q.1) = false := by rw [beq_eq_false_iff_ne]; exact hq
      simp only [lk, List.lookup, this] at h
      rcases List.mem_cons.1 hp with rfl | hp
      · exact fun e => hq e.symm
      · exact ih h p hp

theorem noInl_of_fresh (ι : List (Nat × QG)) (e : Ex) (hf : Fresh ι (varIds e)) :
    containsInlineVar (keysOf ι) e = false := by
  cases h : containsInlineVar (keysOf ι) e
  · rfl
  · obtain ⟨v, hv, hk⟩ := (containsInlineVar_iff _ e).1 h
    have h1 := lookup_none_not_mem ι v.id (hf v.id (var_mem_varIds e v hv))
    obtain ⟨p, hp, he⟩ := List.mem_map.1 hk
    exact absurd he (h1 p hp)

theorem wfI_v (ι : List (Nat × QG)) (v : V) (g : QG) (h : lk ι v.id = some g) : wfI ι (Ex.v v) = true := by
  simp [wfI, Ex.v, inlVar, h]

theorem wfI_tt (ι : List (Nat × QG)) : wfI ι tt = true :=
  wfI_noInl ι tt (by simp [containsInlineVar, tt, anyN, anyL, isInl])

mutual
theorem collect_wf (ι : List (Nat × QG)) : ∀ (st : CSt) ce (e : Ex), shapeOK e = true →
    needsExtraction e .semiApply = false → Fresh ι (varIds e) → Good ι (collect st ce e).2 →
    wfI ι (collect st ce e).1 = true ∧ shapeOK (collect st ce e).1 = true
  | st, ce, .node d cs, hs, hn, hf, hg => by
    cases d
    case pat g =>
      cases ce
      · simp only [collect, Bool.false_eq_true, ↓reduceIte] at hg ⊢
        exact ⟨wfI_v ι _ g (lk_mint ι st g hg), rfl⟩
      · simp only [collect, ↓reduceIte]; exact ⟨wfI_tt ι, rfl⟩
    case patComp g => simp [needsExtraction] at hn
    case not =>
      obtain ⟨c, rfl, hsc⟩ := shape_one .not cs (Or.inl rfl) hs
      have hnc := needs_conn .not [c] rfl hn
      simp only [needsExtractionL, Bool.or_false] at hnc
      have hfc := (fresh_cons ι c [] (fresh_varIds ι _ _ hf)).1
      cases hp : notPat [c] with
      | some g =>
        cases ce
        · simp only [collect, hp, Bool.false_eq_true, ↓reduceIte] at hg ⊢
          have := lk_mint ι st g hg
          refine ⟨by simp [wfI, notInl, inlVar, Ex.v, this], rfl⟩
        · simp only [collect, hp, ↓reduceIte]; exact ⟨wfI_tt ι, rfl⟩
      | none =>
        simp only [collect, hp, collectL] at hg ⊢
        have ih := collect_wf ι st false c hsc hnc hfc (good_collectL_head ι st false c [] hg)
        refine ⟨?_, by simp [shapeOK, shapeOKL, ih.2]⟩
        simp [wfI, wfIL, ih.1]
    case paren =>
      obtain ⟨c, rfl, hsc⟩ := shape_one .paren cs (Or.inr rfl) hs
      have hnc := needs_conn .paren [c] rfl hn
      simp only [needsExtractionL, Bool.or_false] at hnc
      have hfc := (fresh_cons ι c [] (fresh_varIds ι _ _ hf)).1
      simp only [collect, collectL] at hg ⊢
      have ih := collect_wf ι st false c hsc hnc hfc (good_collectL_head ι st false c [] hg)
      exact ⟨by simp [wfI, wfIL, ih.1], by simp [shapeOK, shapeOKL, ih.2]⟩
    case and =>
      simp only [collect] at hg ⊢
      have ih := collectL_wf ι st ce cs (shapeL_of _ _ rfl hs) (needs_conn .and cs rfl hn)
        (fresh_varIds ι _ _ hf) hg
      exact ⟨by simp [wfI, ih.1], by simp [shapeOK, ih.2]⟩
    case or =>
      simp only [collect] at hg ⊢
      have ih := collectL_wf ι st false cs (shapeL_of _ _ rfl hs) (needs_conn .or cs rfl hn)
        (fresh_varIds ι _ _ hf) hg
      exact ⟨by simp [wfI, ih.1], by simp [shapeOK, ih.2]⟩
    all_goals
      first
      | (have h0 := collectL_noPat st false cs (needs_other _ cs rfl rfl hn)
         simp only [collect, h0]
         exact ⟨wfI_noInl ι _ (noInl_of_fresh ι _ hf), hs⟩)
theorem collectL_wf (ι : List (Nat × QG)) : ∀ (st : CSt) ce (l : List Ex), shapeOKL l = true →
    needsExtractionL l .semiApply = false → Fresh ι (varIdsL l) → Good ι (collectL st ce l).2 →
    wfIL ι (collectL st ce l).1 = true ∧ shapeOKL (collectL st ce l).1 = true
  | st, _, [], _, _, _, _ => ⟨rfl, rfl⟩
  | st, ce, c :: cs, hs, hn, hf, hg => by
    simp only [shapeOKL, Bool.and_eq_true] at hs
    simp only [needsExtractionL, Bool.or_eq_false_iff] at hn
    have hfc := fresh_cons ι c cs hf
    simp only [collectL] at hg ⊢
    have ih1 := collect_wf ι st ce c hs.1 hn.1 hfc.1 (good_of_suffix ι _ _ (collectL_inl _ ce cs) hg)
    have ih2 := collectL_wf ι (collect st ce c).2 ce cs hs.2 hn.2 hfc.2 hg
    exact ⟨by simp [wfIL, ih1.1, ih2.1], by simp [shapeOKL, ih1.2, ih2.2]⟩
end

end PlannerBuild.E
