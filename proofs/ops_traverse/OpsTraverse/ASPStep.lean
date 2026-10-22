import OpsTraverse.ASPInv
/-
allShortestPaths, part 3: every BFS iteration preserves the invariant.
-/
namespace OpsTraverse.ASP

open OpsTraverse.VarLen

theorem layered_tail (a : Nat) (l : List Nat) (h : Layered (a :: l)) : Layered l :=
  ⟨(List.pairwise_cons.mp h.1).2, fun x hx y hy => h.2 x (List.mem_cons_of_mem _ hx) y (List.mem_cons_of_mem _ hy)⟩

/-- Skip branch: `shortest_dist ≤ current_dist` — the node is dropped. -/
theorem inv_skip (g : Graph) (b : Bool) (src dst maxH : Nat) (st : St) (hi : Inv g b src dst maxH st)
    (cur : Nat) (q : List Nat) (hq : st.queue = cur :: q) (dcur : Nat) (hc : st.dist cur = some dcur)
    (hs : ∃ s, st.sd = some s ∧ s ≤ dcur) :
    Inv g b src dst maxH { st with queue := q } where
  dist_ok := hi.dist_ok
  dist_le := hi.dist_le
  src_d := hi.src_d
  q_dist v hv := hi.q_dist v (by rw [hq]; exact List.mem_cons_of_mem _ hv)
  q_nodup := by have := hi.q_nodup; rw [hq] at this; exact (List.nodup_cons.mp this).2
  q_layer := by
    have := hi.q_layer; rw [hq, List.map_cons] at this
    exact layered_tail (dOf st cur) _ this
  preds_ok := hi.preds_ok
  preds_done v u e h := fun hm => hi.preds_done v u e h (by rw [hq]; exact List.mem_cons_of_mem _ hm)
  preds_nodup := hi.preds_nodup
  sd_ok := hi.sd_ok
  closed u du hu huq hdu hsd := by
    by_cases huc : u = cur
    · subst huc; rw [hc] at hu; cases hu
      obtain ⟨s, h1, h2⟩ := hs
      have := hsd s h1; omega
    · exact hi.closed u du hu (by rw [hq]; simp [huc, huq]) hdu hsd

/-- Expand branch: all neighbours of `cur` are relaxed. -/
theorem inv_expand (g : Graph) (hwf : g.WF) (b : Bool) (src dst maxH : Nat) (st : St) (hi : Inv g b src dst maxH st)
    (cur : Nat) (q : List Nat) (hq : st.queue = cur :: q) (dcur : Nat) (hc : st.dist cur = some dcur)
    (hs : ∀ s, st.sd = some s → dcur < s) :
    Inv g b src dst maxH ((step g b cur).foldl (relax src dst 1 maxH false cur dcur) { st with queue := q }) := by
  have hGq : ∀ v ∈ ({ st with queue := q } : St).queue, ({ st with queue := q } : St).dist v ≠ none := by
    intro v hv; obtain ⟨d, hd, _⟩ := hi.q_dist v (by rw [hq]; exact List.mem_cons_of_mem _ hv)
    simp [hd]
  have R := expand_rel g b src dst maxH cur dcur { st with queue := q } hGq
  generalize hX : List.foldl (relax src dst 1 maxH false cur dcur) { st with queue := q } (step g b cur) = X at R ⊢
  have keep : ∀ v d, st.dist v = some d → X.dist v = some d := fun v d h => R.dist_keep v d h
  have dnew : ∀ v d, X.dist v = some d → st.dist v = none → d = dcur + 1 ∧ ∃ e, (e, v) ∈ step g b cur :=
    fun v d h h' => R.dist_new v d h h'
  obtain ⟨nw, hnw, hsub, hnw'⟩ := R.preds_ext
  obtain ⟨news, hXq, hnd, hnews, hall⟩ := R.queue_ext
  simp only at hnw hXq
  have hsdX : X.sd = X.dist dst := R.sd_rel hi.sd_ok
  have hfront := frontier g b src dst maxH st hi cur q hq dcur hc hs
  have hcm : dcur < maxH := by
    obtain ⟨d, hd, hdm⟩ := hi.q_dist cur (by rw [hq]; simp); rw [hc] at hd; cases hd; exact hdm
  have hnod := hi.q_nodup; rw [hq] at hnod
  have hcq : cur ∉ q := (List.nodup_cons.mp hnod).1
  have hXcur : X.dist cur = some dcur := keep cur dcur hc
  -- the new distances are true shortest distances
  have dist_ok' : ∀ v d, X.dist v = some d → IsDist g b src v d := by
    intro v d hv
    cases hGv : st.dist v with
    | some d' =>
      have := keep v d' hGv; rw [this] at hv; cases hv; exact hi.dist_ok v _ hGv
    | none =>
      obtain ⟨rfl, e, he⟩ := dnew v d hv hGv
      refine ⟨reach_snoc g b src cur v dcur e (hi.dist_ok cur dcur hc).1 he, fun k hk hr => ?_⟩
      exact hfront k (by omega) v hr hGv
  have news_none : ∀ v ∈ news, st.dist v = none := fun v hv => (hnews v hv).1
  have q_has : ∀ v ∈ q, ∃ d, st.dist v = some d := fun v hv => by
    obtain ⟨d, hd, _⟩ := hi.q_dist v (by rw [hq]; exact List.mem_cons_of_mem _ hv); exact ⟨d, hd⟩
  refine ⟨dist_ok', ?_, keep src 0 hi.src_d, ?_, ?_, ?_, ?_, ?_, ?_, hsdX, ?_⟩
  · -- dist_le
    intro v d hv
    cases hGv : st.dist v with
    | some d' => have := keep v d' hGv; rw [this] at hv; cases hv; exact hi.dist_le v _ hGv
    | none => obtain ⟨rfl, _⟩ := dnew v d hv hGv; omega
  · -- q_dist
    intro v hv
    rw [hXq] at hv
    rcases List.mem_append.mp hv with hv | hv
    · obtain ⟨d, hd, hdm⟩ := hi.q_dist v (by rw [hq]; exact List.mem_cons_of_mem _ hv)
      exact ⟨d, keep v d hd, hdm⟩
    · obtain ⟨_, h2, h3⟩ := hnews v hv; exact ⟨_, h2, h3⟩
  · -- q_nodup
    rw [hXq, List.nodup_append]
    refine ⟨(List.nodup_cons.mp hnod).2, hnd, fun a ha c hc' hac => ?_⟩
    subst hac
    obtain ⟨d, hd⟩ := q_has a ha
    have := news_none a hc'; rw [hd] at this; cases this
  · -- q_layer
    have hl := hi.q_layer
    rw [hq, List.map_cons] at hl
    have hval : ∀ v ∈ q, dOf st v ≥ dcur ∧ dOf st v ≤ dcur + 1 := by
      intro v hv
      have h1 := (List.pairwise_cons.mp hl.1).1 (dOf st v) (List.mem_map_of_mem hv)
      have h2 := hl.2 (dOf st cur) (by simp) (dOf st v) (List.mem_cons_of_mem _ (List.mem_map_of_mem hv))
      simp only [dOf, hc, Option.getD_some] at h1 h2
      exact ⟨h1, h2⟩
    have hmapq : q.map (dOf X) = q.map (dOf st) := by
      apply List.map_congr_left; intro v hv
      obtain ⟨d, hd⟩ := q_has v hv
      simp [dOf, hd, keep v d hd]
    have hmapn : news.map (dOf X) = news.map (fun _ => dcur + 1) := by
      apply List.map_congr_left; intro v hv; simp [dOf, (hnews v hv).2.1]
    rw [hXq, List.map_append, hmapq, hmapn]
    refine ⟨?_, ?_⟩
    · rw [List.pairwise_append]
      refine ⟨(List.pairwise_cons.mp hl.1).2, ?_, ?_⟩
      · clear hXq hnd hnews hall news_none hmapn
        induction news with
        | nil => simp
        | cons x xs ih => simp only [List.map_cons, List.pairwise_cons]; exact ⟨fun _ h => by simp at h; omega, ih⟩
      · intro a ha c hc'
        obtain ⟨v, hv, rfl⟩ := List.mem_map.mp ha
        obtain ⟨_, _, rfl⟩ := List.mem_map.mp hc'
        exact (hval v hv).2
    · intro a ha c hc'
      have hb : ∀ x ∈ q.map (dOf st) ++ news.map (fun _ => dcur + 1), dcur ≤ x ∧ x ≤ dcur + 1 := by
        intro x hx
        rcases List.mem_append.mp hx with hx | hx
        · obtain ⟨v, hv, rfl⟩ := List.mem_map.mp hx; exact hval v hv
        · obtain ⟨_, _, rfl⟩ := List.mem_map.mp hx; omega
      have := hb a ha; have := hb c hc'; omega
  · -- preds_ok
    intro v u e hm
    rw [hnw] at hm
    rcases List.mem_append.mp hm with hm | hm
    · obtain ⟨h1, du, h2, h3⟩ := hi.preds_ok v u e hm
      exact ⟨h1, du, keep u du h2, keep v (du + 1) h3⟩
    · obtain ⟨h1, h2, h3⟩ := hnw' (v, u, e) hm
      simp only at h1 h2 h3; subst h1
      exact ⟨h2, dcur, hXcur, h3⟩
  · -- preds_done
    intro v u e hm
    rw [hnw] at hm; rw [hXq]
    intro hu
    rcases List.mem_append.mp hm with hm | hm
    · have hnot := hi.preds_done v u e hm
      rw [hq] at hnot
      rcases List.mem_append.mp hu with hu | hu
      · exact hnot (List.mem_cons_of_mem _ hu)
      · obtain ⟨_, du, h2, _⟩ := hi.preds_ok v u e hm
        have := news_none u hu; rw [h2] at this; cases this
    · obtain ⟨h1, _, _⟩ := hnw' (v, u, e) hm
      simp only at h1; subst h1
      rcases List.mem_append.mp hu with hu | hu
      · exact hcq hu
      · have := news_none u hu; rw [hc] at this; cases this
  · -- preds_nodup
    rw [hnw, List.nodup_append]
    refine ⟨hi.preds_nodup, ?_, ?_⟩
    · exact nodup_of_map _ ((step_nodup g hwf b cur).sublist hsub)
    · intro a ha c hc' hac
      subst hac
      obtain ⟨h1, _, _⟩ := hnw' a hc'
      have := hi.preds_done a.1 a.2.1 a.2.2 ha
      rw [hq, h1] at this; exact this (List.mem_cons_self ..)
  · -- closed
    intro u du hu huq hdu hsd e n hen
    by_cases huc : u = cur
    · subst huc
      rw [hXcur] at hu; cases hu
      obtain ⟨h1, h2⟩ := R.done e n hen
      obtain ⟨dn, hdn⟩ := Option.ne_none_iff_exists'.mp h1
      refine ⟨dn, hdn, ?_, fun h => h2 (by rw [hdn, h])⟩
      exact isDist_le g b src n dn (dcur + 1) (dist_ok' n dn hdn) (reach_snoc g b src u n dcur e (hi.dist_ok u dcur hc).1 hen)
    · cases hGu : st.dist u with
      | none =>
        have hnew := dnew u du hu hGu
        exact absurd (hall u hGu (by rw [hu]; simp) (by omega)) (fun hm => huq (by rw [hXq]; exact List.mem_append_right _ hm))
      | some du' =>
        have := keep u du' hGu; rw [this] at hu; cases hu
        have huq' : u ∉ st.queue := by
          rw [hq]; intro hm
          rcases List.mem_cons.mp hm with hm | hm
          · exact huc hm
          · exact huq (by rw [hXq]; exact List.mem_append_left _ hm)
        have hsd' : ∀ s, st.sd = some s → du < s := by
          intro s hs'
          have hd : st.dist dst = some s := by rw [← hi.sd_ok]; exact hs'
          have : X.sd = some s := by rw [hsdX]; exact keep dst s hd
          exact hsd s this
        obtain ⟨dn, hdn, hle, hp⟩ := hi.closed u du hGu huq' hdu hsd' e n hen
        exact ⟨dn, keep n dn hdn, hle, fun h => by rw [hnw]; exact List.mem_append_left _ (hp h)⟩

end OpsTraverse.ASP
