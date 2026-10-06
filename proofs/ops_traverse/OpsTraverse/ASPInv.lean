import OpsTraverse.ASPFold
/-
allShortestPaths, part 2: the BFS invariant. `dist` holds TRUE shortest distances, the queue is
FIFO-layered (two consecutive levels), every node taken off the queue below the target level
has all its neighbours discovered with the right predecessor entries, and `shortest_dist`
mirrors `dist[dst]`.
-/
namespace OpsTraverse.ASP

open OpsTraverse.VarLen

/-- A walk of exactly `k` edges from `src` to `v`. -/
def Reach (g : Graph) (b : Bool) (src v k : Nat) : Prop := ∃ es : List Nat, es.length = k ∧ Walk g b src es v

/-- `d` is the shortest distance from `src` to `v`. -/
def IsDist (g : Graph) (b : Bool) (src v d : Nat) : Prop := Reach g b src v d ∧ ∀ k < d, ¬ Reach g b src v k

def Layered (l : List Nat) : Prop := l.Pairwise (· ≤ ·) ∧ ∀ a ∈ l, ∀ c ∈ l, c ≤ a + 1

def dOf (st : St) (v : Nat) : Nat := (st.dist v).getD 0

structure Inv (g : Graph) (b : Bool) (src dst maxH : Nat) (st : St) : Prop where
  dist_ok : ∀ v d, st.dist v = some d → IsDist g b src v d
  dist_le : ∀ v d, st.dist v = some d → d ≤ maxH
  src_d : st.dist src = some 0
  q_dist : ∀ v ∈ st.queue, ∃ d, st.dist v = some d ∧ d < maxH
  q_nodup : st.queue.Nodup
  q_layer : Layered (st.queue.map (dOf st))
  preds_ok : ∀ v u e, (v, u, e) ∈ st.preds → (e, v) ∈ step g b u ∧ ∃ du, st.dist u = some du ∧ st.dist v = some (du + 1)
  preds_done : ∀ v u e, (v, u, e) ∈ st.preds → u ∉ st.queue
  preds_nodup : st.preds.Nodup
  sd_ok : st.sd = st.dist dst
  closed : ∀ u du, st.dist u = some du → u ∉ st.queue → du < maxH → (∀ s, st.sd = some s → du < s) →
    ∀ e n, (e, n) ∈ step g b u → ∃ dn, st.dist n = some dn ∧ dn ≤ du + 1 ∧ (dn = du + 1 → (n, u, e) ∈ st.preds)

theorem reach_zero (g : Graph) (b : Bool) (src v : Nat) : Reach g b src v 0 ↔ v = src := by
  constructor
  · rintro ⟨es, hl, hw⟩
    cases es with
    | nil => cases hw; rfl
    | cons _ _ => simp at hl
  · rintro rfl; exact ⟨[], rfl, .nil _⟩

theorem walk_unsnoc {g : Graph} {b : Bool} : ∀ {s v : Nat} {es : List Nat} {e : Nat},
    Walk g b s (es ++ [e]) v → ∃ w, Walk g b s es w ∧ (e, v) ∈ step g b w
  | _, _, [], _, .cons h (.nil _) => ⟨_, .nil _, h⟩
  | _, _, _ :: _, _, .cons h hw => by
      obtain ⟨w, h1, h2⟩ := walk_unsnoc hw
      exact ⟨w, .cons h h1, h2⟩

theorem reach_succ (g : Graph) (b : Bool) (src v k : Nat) (h : Reach g b src v (k + 1)) :
    ∃ w e, Reach g b src w k ∧ (e, v) ∈ step g b w := by
  obtain ⟨es, hl, hw⟩ := h
  obtain ⟨es', e, rfl⟩ : ∃ es' e, es = es' ++ [e] := by
    have hne : es ≠ [] := by intro h; subst h; simp at hl
    exact ⟨es.dropLast, es.getLast hne, (List.dropLast_concat_getLast hne).symm⟩
  obtain ⟨w, h1, h2⟩ := walk_unsnoc hw
  exact ⟨w, e, ⟨es', by simpa using hl, h1⟩, h2⟩

theorem reach_snoc (g : Graph) (b : Bool) (src w v k e : Nat) (h : Reach g b src w k) (he : (e, v) ∈ step g b w) :
    Reach g b src v (k + 1) := by
  obtain ⟨es, hl, hw⟩ := h
  exact ⟨es ++ [e], by simp [hl], Walk.snoc hw he⟩

theorem isDist_le (g : Graph) (b : Bool) (src v d k : Nat) (hd : IsDist g b src v d) (hk : Reach g b src v k) : d ≤ k := by
  apply Nat.le_of_not_lt; intro h; exact hd.2 k h hk

theorem isDist_unique (g : Graph) (b : Bool) (src v d d' : Nat) (h : IsDist g b src v d) (h' : IsDist g b src v d') :
    d = d' := Nat.le_antisymm (isDist_le g b src v d d' h h'.1) (isDist_le g b src v d' d h' h.1)

/-- Initial state: `distances = {src: 0}`, `queue = [src]`. -/
theorem inv_st0 (g : Graph) (b : Bool) (src dst maxH : Nat) (hsd : src ≠ dst) (hm : 1 ≤ maxH) :
    Inv g b src dst maxH (st0 src) where
  dist_ok v d h := by
    simp only [st0, upd] at h
    split at h
    · rename_i hv; subst hv; cases h
      exact ⟨(reach_zero g b v v).mpr rfl, fun k hk => by omega⟩
    · cases h
  dist_le v d h := by
    simp only [st0, upd] at h; split at h
    · cases h; omega
    · cases h
  src_d := by simp [st0, upd]
  q_dist v hv := by simp [st0] at hv; subst hv; exact ⟨0, by simp [st0, upd], by omega⟩
  q_nodup := by simp [st0]
  q_layer := by simp [st0, Layered, dOf]
  preds_ok := by simp [st0]
  preds_done := by simp [st0]
  preds_nodup := by simp [st0]
  sd_ok := by simp [st0, upd, Ne.symm hsd]
  closed u du h hq := by
    simp only [st0, upd] at h hq
    split at h
    · rename_i hu; subst hu; simp at hq
    · cases h

/-- Every node within `dcur` of `src` is already discovered when `cur` (level `dcur`, the queue
head) is expanded — the BFS frontier argument. -/
theorem frontier (g : Graph) (b : Bool) (src dst maxH : Nat) (st : St) (hi : Inv g b src dst maxH st)
    (cur : Nat) (q : List Nat) (hq : st.queue = cur :: q) (dcur : Nat) (hc : st.dist cur = some dcur)
    (hsd : ∀ s, st.sd = some s → dcur < s) :
    ∀ k, k ≤ dcur → ∀ v, Reach g b src v k → st.dist v ≠ none := by
  intro k
  induction k with
  | zero => intro _ v hv; rw [(reach_zero g b src v).mp hv, hi.src_d]; simp
  | succ k ih =>
    intro hk v hv
    obtain ⟨w, e, hw, he⟩ := reach_succ g b src v k hv
    have hwd := ih (by omega) w hw
    obtain ⟨dw, hdw⟩ := Option.ne_none_iff_exists'.mp hwd
    have hle : dw ≤ k := isDist_le g b src w dw k (hi.dist_ok w dw hdw) hw
    -- `w` is not queued: every queued node sits at level ≥ dcur
    have hwq : w ∉ st.queue := by
      intro hm
      have hl := hi.q_layer.1
      rw [hq, List.map_cons, List.pairwise_cons] at hl
      rw [hq] at hm
      rcases List.mem_cons.mp hm with rfl | hm'
      · rw [hc] at hdw; cases hdw; omega
      · have := hl.1 (dOf st w) (List.mem_map_of_mem hm')
        simp [dOf, hc, hdw] at this; omega
    have hcl := hi.closed w dw hdw hwq (by
      obtain ⟨d, hd, hdm⟩ := hi.q_dist cur (by rw [hq]; simp)
      rw [hc] at hd; cases hd; omega) (fun s hs => by have := hsd s hs; omega) e v he
    obtain ⟨dn, hdn, _, _⟩ := hcl
    rw [hdn]; simp

end OpsTraverse.ASP

namespace OpsTraverse.ASP
open OpsTraverse.VarLen

/-- One iteration of the `while let Some(current) = queue.pop_front()` loop (non-cycle,
`min_hops = 1`); `bfs (fuel+1) = bfs fuel ∘ bstep` on a non-empty queue. -/
def bstep (g : Graph) (b : Bool) (src dst maxH : Nat) (st : St) : St :=
  match st.queue with
  | [] => st
  | cur :: q =>
    let st1 := { st with queue := q }
    let cd := (st1.dist cur).getD 0
    if st1.sd.any (fun sd => sd ≤ cd) || maxH ≤ cd then st1
    else (step g b cur).foldl (relax src dst 1 maxH false cur cd) st1

theorem bfs_unfold (g : Graph) (b : Bool) (src dst maxH fuel : Nat) (st : St) (cur : Nat) (q : List Nat)
    (hq : st.queue = cur :: q) :
    bfs g b src dst 1 maxH false (fuel + 1) st = bfs g b src dst 1 maxH false fuel (bstep g b src dst maxH st) := by
  simp only [bfs, bstep, hq]
  split <;> rfl

theorem filterMap_ids_sublist (f : Edge → Option (Nat × Nat)) (hf : ∀ x y, f x = some y → y.1 = x.id) :
    ∀ g : Graph, ((g.filterMap f).map (·.1)).Sublist (g.map Edge.id)
  | [] => by simp
  | x :: xs => by
      have ih := filterMap_ids_sublist f hf xs
      simp only [List.filterMap_cons, List.map_cons]
      cases h : f x with
      | none => exact ih.cons _
      | some y => simp only [List.map_cons, hf x y h]; exact ih.cons_cons _

theorem step_ids_nodup (g : Graph) (hwf : g.WF) (b : Bool) (cur : Nat) : ((step g b cur).map (·.1)).Nodup := by
  apply List.Nodup.sublist _ hwf
  apply filterMap_ids_sublist
  intro x y h
  split at h
  · cases h; rfl
  · split at h
    · cases h; rfl
    · cases h

theorem nodup_of_map {α β : Type} (f : α → β) : ∀ {l : List α}, (l.map f).Nodup → l.Nodup
  | [], _ => List.nodup_nil
  | a :: l, h => by
      simp only [List.map_cons, List.nodup_cons] at h
      exact List.nodup_cons.mpr ⟨fun ha => h.1 (List.mem_map_of_mem ha), nodup_of_map f h.2⟩

theorem step_nodup (g : Graph) (hwf : g.WF) (b : Bool) (cur : Nat) : (step g b cur).Nodup :=
  nodup_of_map _ (step_ids_nodup g hwf b cur)

end OpsTraverse.ASP
