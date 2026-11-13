import CchTopologyReview.Fold

/-!
# A lineage's lifetime: writers and folds, in any order

`Lin t t'`: `t'` is reached from `t` by writer forks and steps, and by fitted folds (a fold forked at an
earlier state `tF` of the lineage and published onto a copy reached from `tF` by writers only, 03 §6). S6
and S7 start a new lineage (03 §2: "only S6 and S7 rebuild the pool"; "S6 and S7 start a new lineage with
dense ids"), so they are not steps of `Lin`.

`lin_all` carries, along any `Lin`: the invariant, every arc with its id, `arc_capacity` (never
decreasing), "new arcs get ids at or above the old `arc_capacity`", every holder, and "the pool only
shrinks". The headline theorems follow: `no_reuse`, `ids_only_appended`, `rank_never_returns`,
`rank_of_stable`.
-/

namespace CchTopo

variable {α : Type}

/-- A lineage, from a state to a later one. -/
inductive Lin (inf : α) : Cch α → Cch α → Prop
  | refl (t : Cch α) : Lin inf t t
  | fork {t t' : Cch α} : Lin inf t t' → Lin inf t (fork t')
  | step {t t' t'' : Cch α} : Lin inf t t' → Step inf t' t'' → Lin inf t t''
  | fold {t tF t' : Cch α} {nu nd : Rows} : Lin inf t tF → WLin inf tF t' →
      AddSpec tF.upB tF.upD nu → AddSpec tF.downB tF.downD nd → Lin inf t (fold tF.ver nu nd t')

/-- What survives from `t` to `t'`. -/
structure Carry (t t' : Cch α) : Prop where
  inv : Inv t'
  arcs : ∀ e, upE t e → upE t' e
  cap : t.cap ≤ t'.cap
  fresh : ∀ e, upE t' e → ¬ upE t e → t.cap ≤ e.id
  holders : ∀ r x, t.nodeOf r = some x → t'.nodeOf r = some x
  pool : ∀ r, r ∈ t'.free → r ∈ t.free

theorem Carry.refl (t : Cch α) (h : Inv t) : Carry t t :=
  ⟨h, fun _ h => h, Nat.le_refl _, fun _ h h' => absurd h h', fun _ _ h => h, fun _ h => h⟩

theorem Carry.trans {t₁ t₂ t₃ : Cch α} (a : Carry t₁ t₂) (b : Carry t₂ t₃) : Carry t₁ t₃ := by
  refine ⟨b.inv, fun e h => b.arcs e (a.arcs e h), Nat.le_trans a.cap b.cap, fun e h3 h1 => ?_,
    fun r x h => b.holders r x (a.holders r x h), fun r h => a.pool r (b.pool r h)⟩
  rcases Classical.em (upE t₂ e) with h2 | h2
  · exact a.fresh e h2 h1
  · exact Nat.le_trans a.cap (b.fresh e h3 h2)

theorem carry_fork (t : Cch α) (h : Inv t) : Carry t (fork t) :=
  ⟨fork_inv t h, fun _ h => h, Nat.le_refl _, fun _ h h' => absurd h h', fun _ _ h => h, fun _ h => h⟩

theorem carry_step (inf : α) {t t' : Cch α} (hs : Step inf t t') (h : Inv t) : Carry t t' := by
  have hi := step_inv inf hs h
  cases hs with
  | alloc E =>
    have ha := alloc_arcs t E
    have hu : ∀ e, upE (alloc t E) e ↔ upE t e := fun e => by rw [upE_arcs, upE_arcs, ha]
    exact ⟨hi, fun e he => (hu e).2 he, by rw [(alloc_fields t E).2.2.2.2.2.2]; exact Nat.le_refl _, fun e h1 h2 => absurd ((hu e).1 h1) h2,
      fun r x hx => alloc_keeps_holder E t h r x hx, fun r hr => alloc_free_sub E t r hr⟩
  | s2 N hN =>
    have hu := upE_s2 inf t N h.2.1 hN
    refine ⟨hi, fun e he => (hu e).2 (Or.inl he), Nat.le_add_right _ _, fun e h1 h2 => ?_,
      fun _ _ hx => hx, fun _ hr => hr⟩
    rcases (hu e).1 h1 with h3 | h3
    · exact absurd h3 h2
    · exact (mem_upEnts _ _ e h3).2.1
  | ndel deg => exact ⟨hi, fun _ h => h, Nat.le_refl _, fun _ h h' => absurd h h', fun _ _ h => h, fun _ h => h⟩
  | reuse => exact Carry.refl t h
  | mset i x =>
    have hu : ∀ e, upE (mset t i x) e ↔ upE t e := fun e => Iff.rfl
    exact ⟨hi, fun e he => (hu e).2 he, Nat.le_refl _, fun e h1 h2 => absurd ((hu e).1 h1) h2,
      fun _ _ hx => hx, fun _ hr => hr⟩

theorem carry_wlin (inf : α) {tF t : Cch α} (hw : WLin inf tF t) (h : Inv tF) : Carry tF t := by
  induction hw with
  | refl => exact Carry.refl _ h
  | fork _ ih => exact ih.trans (carry_fork _ ih.inv)
  | step _ _ hs ih => exact ih.trans (carry_step inf hs ih.inv)

theorem carry_fold (inf : α) {tF t : Cch α} {nu nd : Rows} (hI : Inv tF) (hw : WLin inf tF t)
    (hU : AddSpec tF.upB tF.upD nu) (hD : AddSpec tF.downB tF.downD nd) : Carry t (fold tF.ver nu nd t) := by
  obtain ⟨hup, _, _, _, hI', _⟩ := fold_facts inf hI hw hU hD
  exact ⟨hI', fun e he => (hup e).2 he, Nat.le_refl _, fun e h1 h2 => absurd ((hup e).1 h1) h2,
    fun _ _ hx => hx, fun _ hr => hr⟩

/-- **Everything `Carry` lists holds along any lineage.** -/
theorem lin_all (inf : α) {t t' : Cch α} (hl : Lin inf t t') (h : Inv t) : Carry t t' := by
  induction hl with
  | refl => exact Carry.refl _ h
  | fork _ ih => exact ih.trans (carry_fork _ ih.inv)
  | step _ hs ih => exact ih.trans (carry_step inf hs ih.inv)
  | fold _ hw hU hD ih =>
    have c1 := carry_wlin inf hw ih.inv
    exact ih.trans (c1.trans (carry_fold inf ih.inv hw hU hD))

/-- **No id is ever reused within a lineage**: an id held by an arc at any state names the same arc at
    every later state. -/
theorem no_reuse (inf : α) {t t' : Cch α} (hl : Lin inf t t') (h : Inv t) (e e' : Ent) (he : upE t e)
    (he' : upE t' e') (hid : e.id = e'.id) : e = e' := by
  have c := lin_all inf hl h
  exact ids_unique c.inv e e' (c.arcs e he) he' hid

/-- **Ids are only appended**: an arc that is new since `t` has an id at or above `t`'s `arc_capacity`, and
    every arc of `t` is still there with its id. -/
theorem ids_only_appended (inf : α) {t t' : Cch α} (hl : Lin inf t t') (h : Inv t) :
    t.cap ≤ t'.cap ∧ (∀ e, upE t e → upE t' e) ∧ (∀ e, upE t' e → ¬ upE t e → t.cap ≤ e.id ∧ e.id < t'.cap) := by
  have c := lin_all inf hl h
  exact ⟨c.cap, c.arcs, fun e h1 h2 => ⟨c.fresh e h1 h2, (ids_lt_cap c.inv e h1).1⟩⟩

/-- **A rank never returns to the pool within a lineage**: a rank's holder stays its holder (deleted nodes
    and reused ids keep their rank), so it is never free again; and the pool only shrinks. -/
theorem rank_never_returns (inf : α) {t t' : Cch α} (hl : Lin inf t t') (h : Inv t) (r x : Nat)
    (hx : t.nodeOf r = some x) : t'.nodeOf r = some x ∧ r ∉ t'.free := by
  have c := lin_all inf hl h
  have h1 := c.holders r x hx
  refine ⟨h1, fun hf => ?_⟩
  have := ((c.inv.1.free_iff r).1 hf).2
  simp only [Cch.ranks_nodeOf] at this
  rw [h1] at this; cases this

theorem pool_only_shrinks (inf : α) {t t' : Cch α} (hl : Lin inf t t') (h : Inv t) (r : Nat) (hr : r ∈ t'.free) :
    r ∈ t.free := (lin_all inf hl h).pool r hr

/-- A node keeps its rank for the whole lineage. -/
theorem rank_of_stable (inf : α) {t t' : Cch α} (hl : Lin inf t t') (h : Inv t) (x r : Nat)
    (hx : t.rankOf x = some r) : t'.rankOf x = some r := by
  have c := lin_all inf hl h
  exact (c.inv.1.rank_node x r).2 (c.holders r x ((h.1.rank_node x r).1 hx))

end CchTopo
