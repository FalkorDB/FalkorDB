import CchTopologyReview.S2

/-!
# The writer's steps on a Built lineage, and what they leave alone

`Step` is every change 03 §6 lets a writer's flush make to a Built entry's hierarchy: rank allocation,
S2's append (its new arcs given by `S2Ok`), node delete, a created or reused id, and a metric write by id
(S1, the recompute, the full-seed scatter). `fork` is the writer's private copy at `version + 1`.

`WLin tF t`: `t` is reached from the published state `tF` by writers only, each step on a private copy
whose version is above `tF`'s (a writer forks at the latest published data version and publishes at
`+1`; `mvcc_graph.rs:155-160, 301` @08c). This is what lies between a fold's fork `F` and its publish
when the fold is fitted (generation and ticket unchanged, 03 §6 "Publish under the lock"); the fit itself
is proved in `proofs/cch_redesign_review` (`stale_job_gone`, `job_publish_fitted`).
-/

namespace CchTopo

variable {α : Type}

theorem fork_inv (t : Cch α) (h : Inv t) : Inv (fork t) := by
  obtain ⟨hr, ha, hh⟩ := h
  refine ⟨hr, ?_, hh⟩
  exact { ha with added_seq := fun p hp => Nat.le_trans (ha.added_seq p hp) (Nat.le_succ _) }

theorem ndel_inv (t : Cch α) (deg : Nat) (h : Inv t) : Inv (ndel t deg) := h
theorem reuse_inv (t : Cch α) (h : Inv t) : Inv (reuse t) := h

theorem mset_arcs (t : Cch α) (i : Nat) (x : α) : (mset t i x).arcs = t.arcs := by
  simp [Cch.arcs, mset]

theorem mset_inv (t : Cch α) (i : Nat) (x : α) (h : Inv t) : Inv (mset t i x) := by
  unfold Inv; rw [mset_arcs]; exact h

/-- A writer's change to a Built hierarchy (03 §6). -/
inductive Step (inf : α) : Cch α → Cch α → Prop
  | alloc (t : Cch α) (E : List (Nat × Nat)) : Step inf t (alloc t E)
  | s2 (t : Cch α) (N : List (Nat × Nat)) (h : S2Ok t N) : Step inf t (s2 inf t N)
  | ndel (t : Cch α) (deg : Nat) : Step inf t (ndel t deg)
  | reuse (t : Cch α) : Step inf t (reuse t)
  | mset (t : Cch α) (i : Nat) (x : α) : Step inf t (mset t i x)

theorem step_inv (inf : α) {t t' : Cch α} (hs : Step inf t t') (h : Inv t) : Inv t' := by
  cases hs with
  | alloc E => exact alloc_inv E t h
  | s2 N hN => exact s2_inv inf t N h hN
  | ndel deg => exact ndel_inv t deg h
  | reuse => exact reuse_inv t h
  | mset i x => exact mset_inv t i x h

/-- Writers only, from the published state `tF`. -/
inductive WLin (inf : α) (tF : Cch α) : Cch α → Prop
  | refl : WLin inf tF tF
  | fork {t : Cch α} : WLin inf tF t → WLin inf tF (fork t)
  | step {t t' : Cch α} : WLin inf tF t → tF.ver < t.ver → Step inf t t' → WLin inf tF t'

theorem alloc_fields (t : Cch α) (E : List (Nat × Nat)) :
    (alloc t E).upB = t.upB ∧ (alloc t E).downB = t.downB ∧ (alloc t E).upD = t.upD ∧
    (alloc t E).downD = t.downD ∧ (alloc t E).added = t.added ∧ (alloc t E).ver = t.ver ∧ (alloc t E).cap = t.cap := by
  have := alloc_arcs t E
  exact ⟨congrArg Arcs.upB this, congrArg Arcs.downB this, congrArg Arcs.upD this, congrArg Arcs.downD this,
    congrArg Arcs.added this, congrArg Arcs.ver this, congrArg Arcs.cap this⟩

/-- What a step leaves alone, and how it extends `added_since_fold`. -/
theorem step_shape (inf : α) {t t' : Cch α} (hs : Step inf t t') :
    t'.upB = t.upB ∧ t'.downB = t.downB ∧ t'.ver = t.ver ∧ t.cap ≤ t'.cap ∧
    (t'.added = t.added ∨ ∃ U, t'.added = t.added ++ [(t.ver, U)]) := by
  cases hs with
  | alloc E =>
    obtain ⟨h1, h2, _, _, h5, h6, h7⟩ := alloc_fields t E
    exact ⟨h1, h2, h6, h7 ▸ Nat.le_refl _, Or.inl h5⟩
  | s2 N hN => exact ⟨rfl, rfl, rfl, Nat.le_add_right _ _, Or.inr ⟨_, rfl⟩⟩
  | ndel deg => exact ⟨rfl, rfl, rfl, Nat.le_refl _, Or.inl rfl⟩
  | reuse => exact ⟨rfl, rfl, rfl, Nat.le_refl _, Or.inl rfl⟩
  | mset i x => exact ⟨rfl, rfl, rfl, Nat.le_refl _, Or.inl rfl⟩

/-- **Between a fold's fork and its publish**: the bases are the fork's, `added_since_fold` is the fork's
    plus passes with `seq > F`, and the invariant holds. -/
theorem wlin_shape (inf : α) {tF t : Cch α} (hw : WLin inf tF t) (hI : Inv tF) :
    t.upB = tF.upB ∧ t.downB = tF.downB ∧ tF.ver ≤ t.ver ∧ tF.cap ≤ t.cap ∧
    (∃ ext, t.added = tF.added ++ ext ∧ ∀ p ∈ ext, tF.ver < p.1) ∧ Inv t := by
  induction hw with
  | refl => exact ⟨rfl, rfl, Nat.le_refl _, Nat.le_refl _, ⟨[], by simp, by simp⟩, hI⟩
  | fork _ ih =>
    obtain ⟨h1, h2, h3, h4, h5, h6⟩ := ih
    exact ⟨h1, h2, Nat.le_trans h3 (Nat.le_succ _), h4, h5, fork_inv _ h6⟩
  | step _ hv hs ih =>
    obtain ⟨h1, h2, h3, h4, ⟨ext, he, hx⟩, h6⟩ := ih
    obtain ⟨g1, g2, g3, g4, g5⟩ := step_shape inf hs
    refine ⟨g1.trans h1, g2.trans h2, g3 ▸ h3, Nat.le_trans h4 g4, ?_, step_inv inf hs h6⟩
    rcases g5 with g5 | ⟨U, g5⟩
    · exact ⟨ext, g5.trans he, hx⟩
    · refine ⟨ext ++ [(_, U)], by rw [g5, he, List.append_assoc], fun p hp => ?_⟩
      rcases List.mem_append.1 hp with hp | hp
      · exact hx p hp
      · simp at hp; rw [hp]; exact hv

end CchTopo
