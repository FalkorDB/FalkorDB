import OpsTraverse.ASPFinal
/-
allShortestPaths, part 5: minimality, completeness, exactly-once.
-/
namespace OpsTraverse.ASP

open OpsTraverse.VarLen

theorem predWalk_walk {g : Graph} {b : Bool} {src dst maxH : Nat} {st : St} (hi : Inv g b src dst maxH st) :
    ∀ {es m}, PredWalk st src es m → Walk g b src es m
  | _, _, .nil => .nil _
  | _, _, .snoc hp hm => Walk.snoc (predWalk_walk hi hp) (hi.preds_ok _ _ _ hm).1

theorem shortest_isDist (g : Graph) (b : Bool) (src dst : Nat) (es : List Nat) (hw : Walk g b src es dst)
    (hmin : ∀ es', Walk g b src es' dst → es.length ≤ es'.length) : IsDist g b src dst es.length := by
  refine ⟨⟨es, rfl, hw⟩, fun k hk hr => ?_⟩
  obtain ⟨es1, hl, hw1⟩ := hr
  have := hmin es1 hw1; omega

/-- A shortest walk of at most `max_hops` edges is a predecessor chain in the drained state. -/
theorem shortest_predWalk (g : Graph) (b : Bool) (src dst maxH : Nat) (st : St) (hi : Inv g b src dst maxH st)
    (hq : st.queue = []) (es : List Nat) (hw : Walk g b src es dst)
    (hmin : ∀ es', Walk g b src es' dst → es.length ≤ es'.length) (hmax : es.length ≤ maxH) :
    ∀ k, k ≤ es.length → ∃ m, PredWalk st src (es.take k) m ∧ Walk g b m (es.drop k) dst := by
  have hDdst := shortest_isDist g b src dst es hw hmin
  intro k
  induction k with
  | zero => intro _; exact ⟨src, by simpa using PredWalk.nil, by simpa using hw⟩
  | succ k ih =>
    intro hk
    obtain ⟨m, hp, hrest⟩ := ih (by omega)
    have hdm := predWalk_dist hi hp
    have hkl : (es.take k).length = k := by simp; omega
    rw [hkl] at hdm
    have hdrop : es.drop k = es[k]'(by omega) :: es.drop (k + 1) := List.drop_eq_getElem_cons _
    rw [hdrop] at hrest
    cases hrest with
    | cons hstep hw' =>
      rename_i m'
      have hD : IsDist g b src m' (k + 1) := by
        refine ⟨reach_snoc g b src m m' k _ ⟨es.take k, hkl, predWalk_walk hi hp⟩ hstep, fun j hj hr => ?_⟩
        obtain ⟨es1, hl1, hw1⟩ := hr
        have := hmin _ (walk_append hw1 hw')
        simp at this; omega
      have hsd : ∀ s, st.sd = some s → k < s := by
        intro s hs
        rw [hi.sd_ok] at hs
        have := isDist_unique g b src dst s es.length (hi.dist_ok dst s hs) hDdst
        omega
      obtain ⟨dn, hdn, _, hpred⟩ := hi.closed m k hdm (by rw [hq]; simp) (by omega) hsd _ m' hstep
      have : dn = k + 1 := isDist_unique g b src m' dn (k + 1) (hi.dist_ok m' dn hdn) hD
      refine ⟨m', ?_, hw'⟩
      have htake : es.take (k + 1) = es.take k ++ [es[k]'(by omega)] := by
        rw [List.take_add_one, List.getElem?_eq_getElem (by omega)]; rfl
      rw [htake]
      exact PredWalk.snoc hp (hpred this)

/-- **Completeness**: every shortest walk from `src` to `dst` with at most `max_hops` edges is
returned by allShortestPaths (non-cycle, forward orientation), once the BFS has drained its
queue within the fuel and the backtrack has fuel for the path length. -/
theorem asp_complete (g : Graph) (hwf : g.WF) (b : Bool) (src dst maxH fuel : Nat) (hsd : src ≠ dst)
    (hm : 1 ≤ maxH) (hq : (bfs g b src dst 1 maxH false fuel (st0 src)).queue = [])
    (es : List Nat) (hw : Walk g b src es dst)
    (hmin : ∀ es', Walk g b src es' dst → es.length ≤ es'.length) (hmax : es.length ≤ maxH)
    (hfuel : es.length + 1 ≤ fuel) :
    es ∈ asp g b src dst 1 maxH false fuel := by
  have hi := inv_bfs g hwf b src dst maxH fuel _ (inv_st0 g b src dst maxH hsd hm)
  obtain ⟨m, hp, hrest⟩ := shortest_predWalk g b src dst maxH _ hi hq es hw hmin hmax es.length (Nat.le_refl _)
  simp only [List.take_length, List.drop_length] at hp hrest
  cases hrest
  have hne : es ≠ [] := by intro h; subst h; cases hw; exact hsd rfl
  have hb := back_has hi hp [] fuel hfuel (Or.inr hne)
  simp only [List.nil_append] at hb
  unfold asp
  have hcyc : (src == dst) = false := by simpa using hsd
  simp only [hcyc, Bool.false_eq_true, ite_false]
  have hpne : (predsOf (bfs g b src dst 1 maxH false fuel (st0 src)) dst).isEmpty = false := by
    cases hp with
    | nil => exact absurd rfl hsd
    | snoc _ hmem =>
      cases h : predsOf (bfs g b src dst 1 maxH false fuel (st0 src)) dst with
      | nil => have := mem_predsOf hmem; rw [h] at this; simp at this
      | cons _ _ => rfl
  rw [if_neg (by simp [hpne])]
  simp only [List.mem_map]
  exact ⟨es.reverse, hb, by simp⟩

/-- **Minimality**: every returned path is a walk from `src` to `dst` no longer than any other
walk between them (the BFS distance is the true shortest distance). -/
theorem asp_minimal (g : Graph) (hwf : g.WF) (b : Bool) (src dst maxH fuel : Nat) (hsd : src ≠ dst)
    (hm : 1 ≤ maxH) (es : List Nat) (h : es ∈ asp g b src dst 1 maxH false fuel) :
    Walk g b src es dst ∧ ∀ es', Walk g b src es' dst → es.length ≤ es'.length := by
  have hi := inv_bfs g hwf b src dst maxH fuel _ (inv_st0 g b src dst maxH hsd hm)
  refine ⟨asp_sound hsd h, fun es' hw' => ?_⟩
  unfold asp at h
  have hcyc : (src == dst) = false := by simpa using hsd
  simp only [hcyc, Bool.false_eq_true, ite_false] at h
  split at h
  · simp at h
  · simp only [List.mem_map] at h
    obtain ⟨res, hres, rfl⟩ := h
    -- `dst` was reached, so it has a distance `d`, and every result has exactly `d` edges
    have hdst : ∃ d, (bfs g b src dst 1 maxH false fuel (st0 src)).dist dst = some d := by
      rename_i hne
      obtain ⟨p, hp⟩ := List.exists_mem_of_ne_nil _ (by simpa using hne)
      simp only [predsOf, List.mem_map, List.mem_filter, beq_iff_eq] at hp
      obtain ⟨⟨v, u, e⟩, ⟨hm', hv⟩, _⟩ := hp
      simp only at hv; subst hv
      obtain ⟨_, du, _, h2⟩ := hi.preds_ok _ _ _ hm'
      exact ⟨_, h2⟩
    obtain ⟨d, hd⟩ := hdst
    have hl := back_len hi fuel dst [] res d hd hres
    simp at hl
    have := isDist_le g b src dst d es'.length (hi.dist_ok dst d hd) ⟨es', rfl, hw'⟩
    simp; omega

/-! ## Exactly once -/

theorem wf_eq_of_id (g : Graph) (hwf : g.WF) : ∀ {x y : Edge}, x ∈ g → y ∈ g → x.id = y.id → x = y := by
  unfold Graph.WF at hwf
  induction g with
  | nil => intro x y hx; simp at hx
  | cons z zs ih =>
    simp only [List.map_cons, List.nodup_cons] at hwf
    intro x y hx hy he
    rcases List.mem_cons.mp hx with rfl | hx' <;> rcases List.mem_cons.mp hy with rfl | hy'
    · rfl
    · exact absurd (he ▸ List.mem_map_of_mem (f := Edge.id) hy') hwf.1
    · exact absurd (he.symm ▸ List.mem_map_of_mem (f := Edge.id) hx') hwf.1
    · exact ih hwf.2 hx' hy' he

/-- An edge id and its far endpoint determine the predecessor. -/
theorem pred_unique (g : Graph) (hwf : g.WF) (b : Bool) (src dst maxH : Nat) (st : St) (hi : Inv g b src dst maxH st)
    (v u u' e : Nat) (h1 : (v, u, e) ∈ st.preds) (h2 : (v, u', e) ∈ st.preds) : u = u' := by
  obtain ⟨s1, du, hu, hv⟩ := hi.preds_ok _ _ _ h1
  obtain ⟨s2, du', hu', hv'⟩ := hi.preds_ok _ _ _ h2
  obtain ⟨x, hx, hxe, hx1⟩ := mem_step.mp s1
  obtain ⟨y, hy, hye, hy1⟩ := mem_step.mp s2
  have hxy := wf_eq_of_id g hwf hx hy (by rw [hxe, hye])
  subst hxy
  rcases hx1 with ⟨a1, a2⟩ | ⟨_, a1, a2, a3⟩ <;> rcases hy1 with ⟨b1, b2⟩ | ⟨_, b1, b2, b3⟩
  · rw [← a1, ← b1]
  · -- u = x.src = v: a self-loop predecessor, impossible (dist v = dist u + 1)
    have : u = v := by rw [← a1, b3]
    subst this; rw [hu] at hv; cases hv
  · have : u' = v := by rw [← b1, a3]
    subst this; rw [hu'] at hv'; cases hv'
  · rw [← a2, ← b2]

theorem predsOf_nodup (g : Graph) (hwf : g.WF) (b : Bool) (src dst maxH : Nat) (st : St) (hi : Inv g b src dst maxH st)
    (v : Nat) : (predsOf st v).Nodup := by
  unfold predsOf
  have hf : (st.preds.filter fun x => x.1 == v).Nodup := hi.preds_nodup.filter _
  generalize hl : st.preds.filter (fun x => x.1 == v) = l at hf
  have hmem : ∀ x ∈ l, x.1 = v ∧ x ∈ st.preds := by
    intro x hx; rw [← hl] at hx; simp only [List.mem_filter, beq_iff_eq] at hx; exact ⟨hx.2, hx.1⟩
  clear hl
  induction l with
  | nil => simp
  | cons x xs ih =>
    simp only [List.map_cons, List.nodup_cons]
    refine ⟨fun hm => ?_, ih (List.nodup_cons.mp hf).2 (fun y hy => hmem y (List.mem_cons_of_mem _ hy))⟩
    obtain ⟨y, hy, hxy⟩ := List.mem_map.mp hm
    have h1 := hmem x (List.mem_cons_self ..)
    have h2 := hmem y (List.mem_cons_of_mem _ hy)
    have : x = y := by
      obtain ⟨xv, xu, xe⟩ := x; obtain ⟨yv, yu, ye⟩ := y
      simp only [Prod.mk.injEq] at hxy h1 h2 ⊢
      obtain ⟨rfl, rfl⟩ := hxy
      exact ⟨h1.1.trans h2.1.symm, rfl, rfl⟩
    subst this; exact (List.nodup_cons.mp hf).1 hy

theorem back_prefix (st : St) (src : Nat) : ∀ (fuel node : Nat) (acc res : List Nat),
    res ∈ back st src fuel node acc → ∃ w, res = acc ++ w
  | 0, _, _, _, h => by simp [back] at h
  | fuel + 1, node, acc, res, h => by
      unfold back at h
      split at h
      · simp at h; exact ⟨[], by simp [h]⟩
      · simp only [List.mem_flatMap] at h
        obtain ⟨p, _, hr⟩ := h
        obtain ⟨w, rfl⟩ := back_prefix st src fuel p.1 (acc ++ [p.2]) res hr
        exact ⟨p.2 :: w, by simp⟩

theorem nodup_flatMap {α β : Type} (f : α → List β) : ∀ (l : List α), l.Nodup → (∀ x ∈ l, (f x).Nodup) →
    (∀ x ∈ l, ∀ y ∈ l, x ≠ y → ∀ r, r ∈ f x → r ∉ f y) → (l.flatMap f).Nodup
  | [], _, _, _ => by simp
  | x :: xs, hn, hf, hd => by
      simp only [List.flatMap_cons]
      rw [List.nodup_append]
      refine ⟨hf x (List.mem_cons_self ..), nodup_flatMap f xs (List.nodup_cons.mp hn).2
        (fun y hy => hf y (List.mem_cons_of_mem _ hy))
        (fun y hy z hz hyz => hd y (List.mem_cons_of_mem _ hy) z (List.mem_cons_of_mem _ hz) hyz), ?_⟩
      intro a ha c hc hac
      subst hac
      obtain ⟨y, hy, hcy⟩ := List.mem_flatMap.mp hc
      have hxy : x ≠ y := fun e => (List.nodup_cons.mp hn).1 (e ▸ hy)
      exact hd x (List.mem_cons_self ..) y (List.mem_cons_of_mem _ hy) hxy a ha hcy

theorem back_nodup (g : Graph) (hwf : g.WF) (b : Bool) (src dst maxH : Nat) (st : St) (hi : Inv g b src dst maxH st) :
    ∀ (fuel node : Nat) (acc : List Nat), (back st src fuel node acc).Nodup
  | 0, _, _ => by simp [back]
  | fuel + 1, node, acc => by
      unfold back
      split
      · simp
      · apply nodup_flatMap _ _ (predsOf_nodup g hwf b src dst maxH st hi node)
          (fun p _ => back_nodup g hwf b src dst maxH st hi fuel p.1 (acc ++ [p.2]))
        intro x hx y hy hxy r hrx hry
        obtain ⟨w1, rfl⟩ := back_prefix st src fuel x.1 _ r hrx
        obtain ⟨w2, hw⟩ := back_prefix st src fuel y.1 _ _ hry
        have he : x.2 = y.2 := by
          have := congrArg (fun l => l[acc.length]?) hw
          simp at this; exact this
        apply hxy
        simp only [predsOf, List.mem_map, List.mem_filter, beq_iff_eq] at hx hy
        obtain ⟨⟨v1, u1, e1⟩, ⟨h1, hv1⟩, rfl⟩ := hx
        obtain ⟨⟨v2, u2, e2⟩, ⟨h2, hv2⟩, rfl⟩ := hy
        simp only at hv1 hv2 he ⊢; subst hv1; subst hv2; subst he
        rw [pred_unique g hwf b src dst maxH st hi _ _ _ _ h1 h2]

/-- **Exactly once**: allShortestPaths returns no path twice. -/
theorem asp_nodup (g : Graph) (hwf : g.WF) (b : Bool) (src dst maxH fuel : Nat) (hsd : src ≠ dst)
    (hm : 1 ≤ maxH) (rev : Bool) : (asp g b src dst 1 maxH rev fuel).Nodup := by
  have hi := inv_bfs g hwf b src dst maxH fuel _ (inv_st0 g b src dst maxH hsd hm)
  unfold asp
  have hcyc : (src == dst) = false := by simpa using hsd
  simp only [hcyc, Bool.false_eq_true, ite_false]
  split
  · simp
  · have hb := back_nodup g hwf b src dst maxH _ hi fuel dst []
    -- `reverse` (twice when `reversed`) is injective
    have hinj : ∀ x y : List Nat, (if rev = true then x.reverse.reverse else x.reverse) =
        (if rev = true then y.reverse.reverse else y.reverse) → x = y := by
      intro x y h; cases rev <;> simpa using h
    generalize back (bfs g b src dst 1 maxH false fuel (st0 src)) src fuel dst [] = L at hb ⊢
    induction L with
    | nil => simp
    | cons x xs ih =>
      simp only [List.map_cons, List.nodup_cons] at hb ⊢
      refine ⟨fun hm' => ?_, ih hb.2⟩
      obtain ⟨y, hy, hxy⟩ := List.mem_map.mp hm'
      exact hb.1 ((hinj y x hxy) ▸ hy)

end OpsTraverse.ASP
