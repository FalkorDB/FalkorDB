import PendingCommit.PendingState
/-
# Created relationships, label scans and degrees in `Pending`
(pending.rs:710-1096). See `PendingState.lean`'s header for the statements.
-/
namespace PendingCommit.PS
open PA

/-- Edges of type `t` created in this batch. -/
def grp (p : P) (t : Nat) : List (Nat × Nat × Nat) := (fget p.relsByType t).getD []

/-- `created_rel_types` is the index of `created_rels_by_type`; ids unique per group. -/
def RelInv (p : P) : Prop :=
  (∀ r t, fget p.relTypes r = some t ↔ r ∈ (grp p t).map (·.1)) ∧ (∀ t, ((grp p t).map (·.1)).Nodup)

/-- `created_relationship` (:770). -/
def createdRel (p : P) (rid f to t : Nat) : P :=
  { p with relsByType := fset p.relsByType t (grp p t ++ [(rid, f, to)]),
           relTypes := fset p.relTypes rid t, taken := sins p.taken rid }

/-- `get_created_relationship_endpoints` (:958). -/
def createdEndpoints (p : P) (rid : Nat) : Option (Nat × Nat) :=
  (fget p.relTypes rid).bind fun t => (fget p.relsByType t).bind fun es =>
    (es.find? (·.1 == rid)).map (fun e => e.2)

theorem grp_created (p : P) (rid f to t t' : Nat) :
    grp (createdRel p rid f to t) t' = if t' = t then grp p t ++ [(rid, f, to)] else grp p t' := by
  simp only [grp, createdRel, fget_fset]; split <;> simp

/-- **`createdRel_spec`**: a fresh id keeps `RelInv` and is then reported
created, with its type and endpoints. -/
theorem createdRel_spec (p : P) (h : RelInv p) (rid f to t : Nat) (hf : fget p.relTypes rid = none) :
    RelInv (createdRel p rid f to t) ∧ getRelType (createdRel p rid f to t) rid = some t ∧
      createdEndpoints (createdRel p rid f to t) rid = some (f, to) ∧ rid ∈ (createdRel p rid f to t).taken := by
  have hfresh : ∀ t', rid ∉ (grp p t').map (·.1) := fun t' hm => by
    have := (h.1 rid t').2 hm; rw [hf] at this; cases this
  refine ⟨⟨fun r t' => ?_, fun t' => ?_⟩, by simp [getRelType, createdRel], ?_, (mem_sins _ _ _).2 (.inl rfl)⟩
  · rw [grp_created]
    have e1 : fget (createdRel p rid f to t).relTypes r = if r = rid then some t else fget p.relTypes r := by
      simp [createdRel]
    rw [e1]
    by_cases hr : r = rid
    · subst hr; rw [if_pos rfl]
      by_cases ht : t' = t
      · subst ht; simp
      · rw [if_neg ht]; constructor
        · intro e; cases e; exact absurd rfl ht
        · intro hm; exact absurd hm (hfresh t')
    · rw [if_neg hr, h.1 r t']
      by_cases ht : t' = t
      · subst ht; simp [hr]
      · simp [ht]
  · rw [grp_created]
    split
    · rename_i ht; subst ht
      simp only [List.map_append, List.map_cons, List.map_nil]
      exact List.nodup_append.2 ⟨h.2 t', by simp, fun a ha b hb e => by
        simp at hb; subst hb; subst e; exact hfresh t' ha⟩
    · exact h.2 t'
  · simp only [createdEndpoints, createdRel, fget_fset, ite_true, Option.bind_some]
    have : ((grp p t ++ [(rid, f, to)]).find? (·.1 == rid)) = some (rid, f, to) := by
      rw [List.find?_append]
      have : (grp p t).find? (·.1 == rid) = none := by
        rw [List.find?_eq_none]; intro x hx e
        simp at e; exact hfresh t (e ▸ List.mem_map_of_mem hx)
      simp [this]
    simp [this]

theorem inj_of_nodup_map {α β : Type} {f : α → β} : ∀ {l : List α}, (l.map f).Nodup →
    ∀ {x y : α}, x ∈ l → y ∈ l → f x = f y → x = y
  | [], _, _, _, hx, _, _ => by simp at hx
  | a :: l, hnd, x, y, hx, hy, e => by
    simp only [List.map_cons, List.nodup_cons, List.mem_map] at hnd
    simp only [List.mem_cons] at hx hy
    rcases hx with rfl | hx <;> rcases hy with rfl | hy
    · rfl
    · exact absurd ⟨y, hy, e.symm⟩ hnd.1
    · exact absurd ⟨x, hx, e⟩ hnd.1
    · exact inj_of_nodup_map hnd.2 hx hy e

/-- **`createdEndpoints_spec`**: under `RelInv` the lookup finds the entry. -/
theorem createdEndpoints_spec (p : P) (h : RelInv p) (rid f to t : Nat) (hm : (rid, f, to) ∈ grp p t) :
    createdEndpoints p rid = some (f, to) := by
  have ht : fget p.relTypes rid = some t := (h.1 rid t).2 (List.mem_map_of_mem hm)
  unfold createdEndpoints; rw [ht]; simp only [Option.bind_some]
  have hg : fget p.relsByType t = some (grp p t) := by
    unfold grp; cases e : fget p.relsByType t with
    | none => simp [grp, e] at hm
    | some es => rfl
  rw [hg]; simp only [Option.bind_some]
  -- the first entry with this id is the one: ids are unique in the group
  have hnd := h.2 t
  obtain ⟨x, hx, hxe⟩ : ∃ x, (grp p t).find? (·.1 == rid) = some x ∧ x = (rid, f, to) := by
    cases e : (grp p t).find? (·.1 == rid) with
    | none => rw [List.find?_eq_none] at e; exact absurd (by simp) (e _ hm)
    | some x =>
      refine ⟨x, rfl, ?_⟩
      have hxm := List.mem_of_find?_eq_some e
      have hx1 : x.1 = rid := by simpa using List.find?_some e
      exact inj_of_nodup_map hnd hxm hm (by simp [hx1])
  rw [hx, hxe]; rfl

/-! ## `remove_pending_relationships_for_node` -/

def incident (id : Nat) (e : Nat × Nat × Nat) : Bool := e.2.1 == id || e.2.2 == id

/-- The edges collected by the first loop, with their types. -/
def collect (p : P) (id : Nat) : List ((Nat × Nat × Nat) × Nat) :=
  p.relsByType.flatMap (fun g => (g.2.filter (incident id)).map (·, g.1))

/-- One iteration of the second loop. -/
def rmStep (p : P) (x : (Nat × Nat × Nat) × Nat) : P :=
  let (e, t) := x
  let rb := match fget p.relsByType t with
    | some es => let es' := es.filter (·.1 != e.1)
                 if es'.isEmpty then frem p.relsByType t else fset p.relsByType t es'
    | none => p.relsByType
  { p with relTypes := frem p.relTypes e.1, relsByType := rb, newR := frem p.newR e.1,
           deletedR := p.deletedR.filter (· != e.1), taken := p.taken.filter (· != e.1),
           cancelledR := p.cancelledR ++ [(e.1, t, e.2.1, e.2.2)] }

def removeRels (p : P) (id : Nat) : P := (collect p id).foldl rmStep p

theorem grp_rmStep (p : P) (e : Nat × Nat × Nat) (t t' : Nat) :
    grp (rmStep p (e, t)) t' = if t' = t then (grp p t).filter (·.1 != e.1) else grp p t' := by
  simp only [grp, rmStep]
  cases h : fget p.relsByType t with
  | none => split <;> simp_all
  | some es =>
    simp only
    split
    · rename_i he; simp only [fget_frem]; split
      · simp [List.isEmpty_iff.1 he]
      · rfl
    · simp only [fget_fset]; split <;> rfl

def Rids (L : List ((Nat × Nat × Nat) × Nat)) : List Nat := L.map (·.1.1)

/-- **`removeRels_grp`**: every group loses exactly the collected ids. -/
theorem removeRels_fold : ∀ (L : List ((Nat × Nat × Nat) × Nat)) (p : P),
    (∀ x ∈ L, ∀ t, x.1.1 ∈ (grp p t).map (·.1) → t = x.2) →
    (∀ t, grp (L.foldl rmStep p) t = (grp p t).filter (fun e => e.1 ∉ Rids L)) ∧
    (∀ r, fget (L.foldl rmStep p).relTypes r = if r ∈ Rids L then none else fget p.relTypes r) ∧
    (∀ r, fget (L.foldl rmStep p).newR r = if r ∈ Rids L then none else fget p.newR r) ∧
    (L.foldl rmStep p).deletedR = p.deletedR.filter (· ∉ Rids L) ∧
    (L.foldl rmStep p).cancelledR = p.cancelledR ++ L.map (fun x => (x.1.1, x.2, x.1.2.1, x.1.2.2)) ∧
    (L.foldl rmStep p).taken = p.taken.filter (· ∉ Rids L)
  | [], p, _ => by
    refine ⟨fun t => ?_, fun r => by simp [Rids], fun r => by simp [Rids], ?_, by simp, ?_⟩
    · simp only [List.foldl_nil]; exact (List.filter_eq_self.2 (by simp [Rids])).symm
    · simp only [List.foldl_nil]; exact (List.filter_eq_self.2 (by simp [Rids])).symm
    · simp only [List.foldl_nil]; exact (List.filter_eq_self.2 (by simp [Rids])).symm
  | (e, t) :: L, p, hu => by
    have hu' : ∀ x ∈ L, ∀ t', x.1.1 ∈ (grp (rmStep p (e, t)) t').map (·.1) → t' = x.2 := by
      intro x hx t' hm
      rw [grp_rmStep] at hm
      apply hu x (by simp [hx]) t'
      split at hm
      · rename_i ht; subst ht
        exact (List.map_subset _ List.filter_sublist.subset) hm
      · exact hm
    obtain ⟨a1, a2, a3, a4, a5, a6⟩ := removeRels_fold L (rmStep p (e, t)) hu'
    simp only [List.foldl_cons]
    refine ⟨fun t' => ?_, fun r => ?_, fun r => ?_, ?_, ?_, ?_⟩
    · rw [a1, grp_rmStep]
      split
      · rename_i ht; subst ht
        rw [List.filter_filter]; congr 1; funext x
        by_cases hx : x.1 = e.1 <;> by_cases h2 : x.1 ∈ Rids L <;> simp [Rids] at h2 <;> simp [Rids, hx, h2]
      · rename_i ht
        apply List.filter_congr
        intro x hm
        simp only [Rids, List.map_cons, List.mem_cons]
        by_cases hx : x.1 = e.1
        · -- an id of type `t` cannot sit in another group
          exact absurd (hu (e, t) (by simp) t' (hx ▸ List.mem_map_of_mem hm)) ht
        · by_cases h2 : x.1 ∈ List.map (fun x => x.1.1) L <;> simp [hx, h2]
    · rw [a2]; clear a1 a3 a4 a5 a6 hu hu'; simp only [rmStep, fget_frem]
      by_cases h1 : r = e.1 <;> by_cases h2 : r ∈ Rids L <;> simp_all [Rids]
    · rw [a3]; clear a1 a2 a4 a5 a6 hu hu'; simp only [rmStep, fget_frem]
      by_cases h1 : r = e.1 <;> by_cases h2 : r ∈ Rids L <;> simp_all [Rids]
    · rw [a4]; simp only [rmStep, List.filter_filter]
      congr 1; funext x; by_cases hx : x = e.1 <;> by_cases h2 : x ∈ Rids L <;> simp [Rids] at h2 <;> simp [Rids, hx, h2]
    · rw [a5]; simp [rmStep]
    · rw [a6]; simp only [rmStep, List.filter_filter]
      congr 1; funext x; by_cases hx : x = e.1 <;> by_cases h2 : x ∈ Rids L <;> simp [Rids] at h2 <;> simp [Rids, hx, h2]

theorem fget_of_mem {V : Type} : ∀ (m : FMap V), (m.map (·.1)).Nodup → ∀ t es, (t, es) ∈ m → fget m t = some es
  | [], _, _, _, h => by simp at h
  | (a, b) :: m, hnd, t, es, h => by
    simp only [List.map_cons, List.nodup_cons] at hnd
    simp only [List.mem_cons, Prod.mk.injEq] at h
    simp only [fget]
    rcases h with ⟨rfl, rfl⟩ | h
    · simp
    · have : a ≠ t := fun e => hnd.1 (e ▸ List.mem_map_of_mem (f := (·.1)) h)
      rw [if_neg this]; exact fget_of_mem m hnd.2 t es h

theorem mem_of_fget {V : Type} : ∀ (m : FMap V) t es, fget m t = some es → (t, es) ∈ m
  | [], _, _, h => by simp [fget] at h
  | (a, b) :: m, t, es, h => by
    simp only [fget] at h
    split at h
    · rename_i ha; cases h; subst ha; simp
    · exact List.mem_cons_of_mem _ (mem_of_fget m t es h)

theorem mem_collect (p : P) (hk : (p.relsByType.map (·.1)).Nodup) (id : Nat) (x : (Nat × Nat × Nat) × Nat) :
    x ∈ collect p id ↔ x.1 ∈ grp p x.2 ∧ incident id x.1 = true := by
  obtain ⟨e, t⟩ := x
  simp only [collect, List.mem_flatMap, List.mem_map, List.mem_filter, Prod.mk.injEq]
  constructor
  · rintro ⟨⟨t', es⟩, hm, e', ⟨he', hi⟩, rfl, rfl⟩
    refine ⟨?_, hi⟩
    simp only [grp, fget_of_mem _ hk _ _ hm, Option.getD_some]; exact he'
  · rintro ⟨hm, hi⟩
    unfold grp at hm
    cases hf : fget p.relsByType t with
    | none => rw [hf] at hm; simp at hm
    | some es =>
      rw [hf] at hm
      exact ⟨(t, es), mem_of_fget _ _ _ hf, e, ⟨hm, hi⟩, rfl, rfl⟩

/-- **`removeRels_spec`**: `remove_pending_relationships_for_node` removes
exactly the edges incident on the node — from `created_rels_by_type`,
`created_rel_types`, `new_relationships_attrs` and `deleted_relationships` —
records each in `cancelled_relationships`, and keeps `RelInv`. -/
theorem removeRels_spec (p : P) (h : RelInv p) (hk : (p.relsByType.map (·.1)).Nodup) (id : Nat) :
    let p' := removeRels p id
    let R := Rids (collect p id)
    RelInv p' ∧ (∀ t, grp p' t = (grp p t).filter (fun e => !incident id e)) ∧
      (∀ r, fget p'.relTypes r = if r ∈ R then none else fget p.relTypes r) ∧
      (∀ r, fget p'.newR r = if r ∈ R then none else fget p.newR r) ∧
      p'.deletedR = p.deletedR.filter (· ∉ R) ∧
      p'.cancelledR = p.cancelledR ++ (collect p id).map (fun x => (x.1.1, x.2, x.1.2.1, x.1.2.2)) ∧
      p'.taken = p.taken.filter (· ∉ R) := by
  have hu : ∀ x ∈ collect p id, ∀ t, x.1.1 ∈ (grp p t).map (·.1) → t = x.2 := by
    intro x hx t hm
    have h1 := (h.1 x.1.1 t).2 hm
    have h2 := (h.1 x.1.1 x.2).2 (List.mem_map_of_mem ((mem_collect p hk id x).1 hx).1)
    rw [h1] at h2; exact Option.some.inj h2
  obtain ⟨a1, a2, a3, a4, a5, a6⟩ := removeRels_fold (collect p id) p hu
  -- an entry's id is collected iff the entry is incident
  have key : ∀ t e, e ∈ grp p t → (e.1 ∈ Rids (collect p id) ↔ incident id e = true) := by
    intro t e he
    constructor
    · intro hr
      simp only [Rids, List.mem_map] at hr
      obtain ⟨⟨e2, t2⟩, hx, he2⟩ := hr
      obtain ⟨hm2, hi2⟩ := (mem_collect p hk id _).1 hx
      have ht : t = t2 := hu _ hx t (he2 ▸ List.mem_map_of_mem he)
      subst ht
      have : e2 = e := inj_of_nodup_map (h.2 t) hm2 he he2
      subst this; exact hi2
    · intro hi
      simp only [Rids, List.mem_map]
      exact ⟨(e, t), (mem_collect p hk id _).2 ⟨he, hi⟩, rfl⟩
  have hg : ∀ t, grp (removeRels p id) t = (grp p t).filter (fun e => !incident id e) := by
    intro t; unfold removeRels; rw [a1]
    apply List.filter_congr; intro e he
    have := key t e he
    by_cases hi : incident id e = true
    · simp [hi, this.2 hi]
    · have : e.1 ∉ Rids (collect p id) := fun hr => hi ((key t e he).1 hr)
      simp [hi, this]
  refine ⟨⟨fun r t => ?_, fun t => ?_⟩, hg, a2, a3, a4, a5, a6⟩
  · unfold removeRels; rw [a2, ← removeRels, hg]
    by_cases hr : r ∈ Rids (collect p id)
    · simp only [hr, ite_true, reduceCtorEq, false_iff]
      intro hm
      simp only [List.mem_map, List.mem_filter] at hm
      obtain ⟨e, ⟨he, hi⟩, rfl⟩ := hm
      simp at hi; exact absurd ((key t e he).1 hr) (by simp [hi])
    · simp only [hr, ite_false]
      rw [h.1 r t]
      constructor
      · intro hm
        obtain ⟨e, he, rfl⟩ := List.mem_map.1 hm
        have : incident id e = false := by
          cases hi : incident id e
          · rfl
          · exact absurd ((key t e he).2 hi) hr
        exact List.mem_map_of_mem (List.mem_filter.2 ⟨he, by simp [this]⟩)
      · intro hm
        obtain ⟨e, he, rfl⟩ := List.mem_map.1 hm
        exact List.mem_map_of_mem (List.mem_filter.1 he).1
  · rw [hg]
    exact (h.2 t).sublist (List.filter_sublist.map _)

/-- Nothing incident survives. -/
theorem removeRels_none_incident (p : P) (h : RelInv p) (hk : (p.relsByType.map (·.1)).Nodup) (id t : Nat)
    (e : Nat × Nat × Nat) (he : e ∈ grp (removeRels p id) t) : incident id e = false := by
  rw [(removeRels_spec p h hk id).2.1 t] at he
  simpa using (List.mem_filter.1 he).2

end PendingCommit.PS
