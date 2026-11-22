import FalkorNativeIndex.CchPaths
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-!
# The elimination-tree query's search (`cch/query.rs`)
-/
namespace NI

/-- `Chain::new`'s ranks (`query.rs`): the start, then repeatedly the first entry of the row. -/
def chain {up wt A N} (h : Ctx up wt A N) (r : Nat) : List Nat :=
  match hh : (up r).head? with
  | none => [r]
  | some p => r :: chain h p
termination_by N - r
decreasing_by
  have h1 := h.gt r p (List.mem_of_mem_head? hh)
  have h2 := h.lt r p (List.mem_of_mem_head? hh)
  omega

theorem chain_none {up wt A N} (h : Ctx up wt A N) (r : Nat) (hh : (up r).head? = none) : chain h r = [r] := by
  rw [chain]; split <;> simp_all

theorem chain_some {up wt A N} (h : Ctx up wt A N) (r p : Nat) (hh : (up r).head? = some p) :
    chain h r = r :: chain h p := by
  rw [chain]; split <;> simp_all

theorem chain_head {up wt A N} (h : Ctx up wt A N) (r : Nat) : (chain h r).head? = some r := by
  cases hh : (up r).head? with
  | none => rw [chain_none h r hh]; rfl
  | some p => rw [chain_some h r p hh]; rfl

theorem chain_mem_self {up wt A N} (h : Ctx up wt A N) (r : Nat) : r ∈ chain h r := by
  have := chain_head h r
  exact List.mem_of_mem_head? this

/-- A chain ascends. -/
theorem chain_sorted {up wt A N} (h : Ctx up wt A N) :
    ∀ (k r : Nat), N - r ≤ k → (chain h r).Pairwise (· < ·) := by
  intro k
  induction k with
  | zero => intro r hr; cases hh : (up r).head? with
    | none => rw [chain_none h r hh]; simp
    | some p => exfalso; have := h.lt r p (List.mem_of_mem_head? hh); have := h.gt r p (List.mem_of_mem_head? hh); omega
  | succ k ih =>
    intro r hr
    cases hh : (up r).head? with
    | none => rw [chain_none h r hh]; simp
    | some p =>
      have hp := List.mem_of_mem_head? hh
      have h1 := h.gt r p hp
      have h2 := h.lt r p hp
      rw [chain_some h r p hh]
      have ihp := ih p (by omega)
      refine List.pairwise_cons.2 ⟨?_, ihp⟩
      intro a ha
      -- every element of `chain p` is `≥ p > r`
      have hge : ∀ x ∈ chain h p, p ≤ x := by
        intro x hx
        have := chain_head h p
        cases hc : chain h p with
        | nil => rw [hc] at this; simp at this
        | cons y ys =>
          rw [hc] at this ihp hx
          simp at this; subst this
          rcases List.mem_cons.1 hx with rfl | hx
          · exact Nat.le_refl _
          · exact Nat.le_of_lt (List.rel_of_pairwise_cons ihp hx)
      have := hge a ha
      omega

theorem chain_sorted' {up wt A N} (h : Ctx up wt A N) (r : Nat) : (chain h r).Pairwise (· < ·) :=
  chain_sorted h N r (by omega)

/-- A row's first entry is its least. -/
theorem head_le_row {up wt A N} (h : Ctx up wt A N) (r p w : Nat) (hp : (up r).head? = some p) (hw : w ∈ up r) :
    p ≤ w := by
  have hs := h.sorted r
  cases hl : up r with
  | nil => rw [hl] at hw; simp at hw
  | cons a t =>
    rw [hl] at hp hs hw
    simp at hp; subst hp
    rcases List.mem_cons.1 hw with rfl | hw
    · exact Nat.le_refl _
    · exact Nat.le_of_lt (List.rel_of_pairwise_cons hs hw)

/-- **Every upper neighbour of a chain rank is on the chain** (the claim `relax_from`'s `debug_assert` rests on:
    an upper neighbour is an ancestor). -/
theorem chain_closed {up wt A N} (h : Ctx up wt A N) :
    ∀ (k r : Nat), N - r ≤ k → ∀ a ∈ chain h r, ∀ w ∈ up a, w ∈ chain h r := by
  intro k
  induction k with
  | zero =>
    intro r hr a ha w hw
    cases hh : (up r).head? with
    | none =>
      rw [chain_none h r hh] at ha; simp at ha; subst ha
      cases hl : up a with
      | nil => rw [hl] at hw; simp at hw
      | cons y ys => rw [hl] at hh; simp at hh
    | some p => exfalso; have := h.lt r p (List.mem_of_mem_head? hh); have := h.gt r p (List.mem_of_mem_head? hh); omega
  | succ k ih =>
    intro r hr a ha w hw
    cases hh : (up r).head? with
    | none =>
      rw [chain_none h r hh] at ha; simp at ha; subst ha
      cases hl : up a with
      | nil => rw [hl] at hw; simp at hw
      | cons y ys => rw [hl] at hh; simp at hh
    | some p =>
      have hp := List.mem_of_mem_head? hh
      have h1 := h.gt r p hp
      have h2 := h.lt r p hp
      rw [chain_some h r p hh] at ha ⊢
      rcases List.mem_cons.1 ha with rfl | ha
      · -- `a = r`: `w` is `p` or above it, then in `up p`
        rcases Nat.lt_or_ge p w with hlt | hge
        · have hwp : w ∈ up p := h.closure a p w hp hw hlt
          exact List.mem_cons_of_mem _ (ih p (by omega) p (chain_mem_self h p) w hwp)
        · have : w = p := Nat.le_antisymm hge (head_le_row h a p w hh hw)
          subst this; exact List.mem_cons_of_mem _ (chain_mem_self h w)
      · exact List.mem_cons_of_mem _ (ih p (by omega) a ha w hw)

/-- The ranks of an ascending path from a rank on `s`'s chain lie on `s`'s chain. -/
theorem upPath_closed {up wt A N} (h : Ctx up wt A N) {wf} {s a z e : Nat} (ha : a ∈ chain h s)
    (p : UpPath up wf a z e) : z ∈ chain h s := by
  obtain ⟨k, p⟩ := p
  induction p with
  | nil => exact ha
  | @cons a m z k w e hm hw rest ih => exact ih (chain_closed h N s (by omega) a ha m hm)

theorem upPath_mem_chain {up wt A N} (h : Ctx up wt A N) {wf} {s a e : Nat} (p : UpPath up wf s a e) :
    a ∈ chain h s := upPath_closed h (chain_mem_self h s) p

/-- Any member of a chain starts a suffix that is that member's own chain. -/
theorem chain_suffix {up wt A N} (h : Ctx up wt A N) :
    ∀ (k r : Nat), N - r ≤ k → ∀ a ∈ chain h r, ∃ P, chain h r = P ++ chain h a := by
  intro k
  induction k with
  | zero =>
    intro r hr a ha
    cases hh : (up r).head? with
    | none => rw [chain_none h r hh] at ha; simp at ha; subst ha; exact ⟨[], by simp [chain_none h _ hh]⟩
    | some p => exfalso; have := h.lt r p (List.mem_of_mem_head? hh); have := h.gt r p (List.mem_of_mem_head? hh); omega
  | succ k ih =>
    intro r hr a ha
    cases hh : (up r).head? with
    | none => rw [chain_none h r hh] at ha; simp at ha; subst ha; exact ⟨[], by simp [chain_none h _ hh]⟩
    | some p =>
      have hp := List.mem_of_mem_head? hh
      have h1 := h.gt r p hp
      have h2 := h.lt r p hp
      rw [chain_some h r p hh] at ha ⊢
      rcases List.mem_cons.1 ha with rfl | ha
      · exact ⟨[], by simp [chain_some h a p hh]⟩
      · obtain ⟨P, hP⟩ := ih p (by omega) a ha
        exact ⟨r :: P, by rw [hP]; rfl⟩

/-! ## One direction of the search -/

/-- `x ≤ y` on `[0, +inf]`. -/
def wle (x y : Wt) : Prop := x = y ∨ wlt x y = true

theorem wle_refl (x : Wt) : wle x x := Or.inl rfl

theorem wle_trans {x y z : Wt} (h1 : wle x y) (h2 : wle y z) : wle x z := by
  rcases h1 with rfl | h1
  · exact h2
  · rcases h2 with rfl | h2
    · exact Or.inr h1
    · right
      cases x <;> cases y <;> cases z <;> simp_all [wlt] <;> omega

theorem wlt_of_wle_wlt {x y z : Wt} (h1 : wle x y) (h2 : wlt y z = true) : wlt x z = true := by
  rcases h1 with rfl | h1
  · exact h2
  · cases x <;> cases y <;> cases z <;> simp_all [wlt] <;> omega

theorem wlt_of_wlt_wle {x y z : Wt} (h1 : wlt x y = true) (h2 : wle y z) : wlt x z = true := by
  rcases h2 with rfl | h2
  · exact h1
  · cases x <;> cases y <;> cases z <;> simp_all [wlt] <;> omega

theorem wle_some {a b : Nat} (h : a ≤ b) : wle (some a) (some b) := by
  rcases Nat.lt_or_ge a b with h' | h'
  · exact Or.inr (by simp [wlt, h'])
  · have : a = b := by omega
    exact Or.inl (by rw [this])

theorem wle_none (x : Wt) : wle x none := by
  cases x with
  | none => exact Or.inl rfl
  | some a => exact Or.inr rfl

theorem wlt_some_le {a b : Nat} {z : Wt} (h : a ≤ b) (hz : wlt (some b) z = true) : wlt (some a) z = true :=
  wlt_of_wle_wlt (wle_some h) hz

theorem wle_of_not_wlt {x y : Wt} (h : ¬ wlt x y = true) : wle y x := by
  cases x <;> cases y <;> simp_all [wlt, wle] <;> omega

/-- The search state of one direction: the distance and predecessor of every rank (`Reached`). -/
structure DS where
  d : Nat → Wt
  pr : Nat → Option Nat

/-- `Chain::new`'s initial state: distance `0` at the start. -/
def DS.init (s : Nat) : DS := ⟨fun a => if a = s then some 0 else none, fun _ => none⟩

/-- `Chain::relax_from` (`query.rs`) for the rank `r`: offer every upward arc the distance through `r`;
    take the offer on a strict improvement and record `r` as the predecessor. An unreached `r` (distance `+inf`)
    offers `+inf`, which never wins, so the early return changes nothing. -/
def relaxFrom (up : Nat → List Nat) (wf : Nat → Nat → Wt) (r : Nat) (ds : DS) : DS :=
  ⟨fun a => if a ∈ up r ∧ wlt (wadd (ds.d r) (wf r a)) (ds.d a) = true then wadd (ds.d r) (wf r a) else ds.d a,
   fun a => if a ∈ up r ∧ wlt (wadd (ds.d r) (wf r a)) (ds.d a) = true then some r else ds.pr a⟩

/-- A step gated by the current best: `if dist < best { relax_from }`. -/
def gstep (up : Nat → List Nat) (wf : Nat → Nat → Wt) (best : Wt) (o : Nat) (ds : DS) : DS :=
  if wlt (ds.d o) best = true then relaxFrom up wf o ds else ds

theorem relaxFrom_eq_gstep (up wf o ds) : relaxFrom up wf o ds = gstep up wf none o ds := by
  unfold gstep
  cases hd : ds.d o with
  | none =>
    simp only [wlt, Bool.false_eq_true, ite_false]
    cases ds with | mk d pr =>
      simp only at hd
      simp [relaxFrom, hd, wadd, wlt]
  | some x => simp [wlt]

theorem wle_some_inv {x : Wt} {e : Nat} (h : wle x (some e)) : ∃ e', x = some e' ∧ e' ≤ e := by
  rcases h with rfl | h
  · exact ⟨e, rfl, Nat.le_refl _⟩
  · cases x with
    | none => simp [wlt] at h
    | some a => simp [wlt] at h; exact ⟨a, rfl, by omega⟩

theorem rf_d_notmem (up wf r ds x) (hx : x ∉ up r) : (relaxFrom up wf r ds).d x = ds.d x := by
  simp [relaxFrom, hx]

theorem rf_pr_notmem (up wf r ds x) (hx : x ∉ up r) : (relaxFrom up wf r ds).pr x = ds.pr x := by
  simp [relaxFrom, hx]

theorem rf_d_le (up wf r ds x) : wle ((relaxFrom up wf r ds).d x) (ds.d x) := by
  unfold relaxFrom
  simp only
  by_cases hc : x ∈ up r ∧ wlt (wadd (ds.d r) (wf r x)) (ds.d x) = true
  · rw [ifP hc]; exact Or.inr hc.2
  · rw [ifN hc]; exact wle_refl _

theorem rf_d_off (up wf r ds x) (hx : x ∈ up r) : wle ((relaxFrom up wf r ds).d x) (wadd (ds.d r) (wf r x)) := by
  unfold relaxFrom
  simp only
  by_cases hc : wlt (wadd (ds.d r) (wf r x)) (ds.d x) = true
  · rw [ifP ⟨hx, hc⟩]; exact wle_refl _
  · rw [ifN (fun h => hc h.2)]; exact wle_of_not_wlt hc

structure DPInv {up wt A N} (h : Ctx up wt A N) (wf : Nat → Nat → Wt) (s : Nat) (done : List Nat) (best : Wt)
    (ds : DS) : Prop where
  z : ds.d s = some 0
  sound : ∀ a e, ds.d a = some e → UpPath up wf s a e
  off : ∀ a ∈ done, wlt (ds.d a) best = true → ∀ x ∈ up a, wle (ds.d x) (wadd (ds.d a) (wf a x))
  exact : ∀ a, (a = s ∨ a ∈ done) → ∀ e, UpPath up wf s a e → wlt (some e) best = true →
    ∃ e', ds.d a = some e' ∧ e' ≤ e
  pred : ∀ a e p, ds.d a = some e → ds.pr a = some p →
    p ∈ done ∧ a ∈ up p ∧ ∃ ep w, ds.d p = some ep ∧ wf p a = some w ∧ e = ep + w
  reached : ∀ a e, ds.d a = some e → a ≠ s → ∃ p, ds.pr a = some p
  prs : ds.pr s = none

theorem dp_init {up wt A N} (h : Ctx up wt A N) (wf s) (best : Wt) : DPInv h wf s [] best (DS.init s) := by
  refine ⟨by simp [DS.init], ?_, ?_, ?_, ?_, ?_, rfl⟩
  · intro a e he
    simp only [DS.init] at he
    by_cases hs : a = s
    · subst hs; simp at he; subst he; exact UpPath.refl _ _ _
    · simp [hs] at he
  · intro a ha; simp at ha
  · intro a ha e _ _
    rcases ha with rfl | ha
    · exact ⟨0, by simp [DS.init], Nat.zero_le _⟩
    · simp at ha
  · intro a e p _ hp; simp [DS.init] at hp
  · intro a e he hs
    simp only [DS.init] at he
    simp [hs] at he

/-- **Exactness of the next rank**: if every rank of the chain below `o` is processed, and the processed ranks
    offered their arcs whenever their distance beat `best`, then a path of weight `e < best'` to `o` is matched. -/
theorem dp_exact_next {up wt A N} (h : Ctx up wt A N) (wf s done best best' o ds)
    (inv : DPInv h wf s done best ds) (hle : wle best' best)
    (hdone : ∀ a, a ∈ chain h s → a < o → a ∈ done) :
    ∀ e, UpPath up wf s o e → wlt (some e) best' = true → ∃ e', ds.d o = some e' ∧ e' ≤ e := by
  intro e p hlt
  rcases p.snoc with ⟨rfl, rfl⟩ | ⟨a, e0, w, hp, hmem, hw, he⟩
  · exact ⟨0, inv.z, Nat.le_refl _⟩
  · have ha : a ∈ done := hdone a (upPath_mem_chain h hp) (h.gt a o hmem)
    have hb : wlt (some e) best = true := wlt_of_wlt_wle hlt hle
    have hb0 : wlt (some e0) best = true := wlt_some_le (by omega) hb
    obtain ⟨e1, hd1, hle1⟩ := inv.exact a (Or.inr ha) e0 hp hb0
    have hb1 : wlt (ds.d a) best = true := by rw [hd1]; exact wlt_some_le hle1 hb0
    have := inv.off a ha hb1 o hmem
    rw [hd1, hw] at this
    simp only [wadd] at this
    obtain ⟨e2, he2, hle2⟩ := wle_some_inv (wle_trans this (wle_some (by omega : e1 + w ≤ e)))
    exact ⟨e2, he2, hle2⟩

/-- **One gated step keeps the invariant**: processing the next chain rank `o` (everything on the chain below it
    already processed), with the best possibly lowered to `best'`. -/
theorem dp_step {up wt A N} (h : Ctx up wt A N) (wf s done best best' o ds)
    (inv : DPInv h wf s done best ds) (hle : wle best' best)
    (hlt : ∀ a ∈ done, a < o) (hdone : ∀ a, a ∈ chain h s → a < o → a ∈ done) :
    DPInv h wf s (done ++ [o]) best' (gstep up wf best' o ds) := by
  have hex := dp_exact_next h wf s done best best' o ds inv hle hdone
  have hnot : ∀ a ∈ done, a ∉ up o := fun a ha hm => by have := h.gt o a hm; have := hlt a ha; omega
  have hoo : o ∉ up o := fun hm => by have := h.gt o o hm; omega
  have hmono : ∀ a ∈ done, ∀ x, wle (ds.d x) (ds.d x) := fun _ _ _ => wle_refl _
  unfold gstep
  by_cases hg : wlt (ds.d o) best' = true
  · rw [ifP hg]
    have hdo : (relaxFrom up wf o ds).d o = ds.d o := rf_d_notmem up wf o ds o hoo
    have hda : ∀ a ∈ done, (relaxFrom up wf o ds).d a = ds.d a := fun a ha => rf_d_notmem up wf o ds a (hnot a ha)
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · -- z
      have := inv.z
      have h1 := rf_d_le up wf o ds s
      rw [this] at h1
      obtain ⟨e', he', hle'⟩ := wle_some_inv h1
      have : e' = 0 := by omega
      rw [he', this]
    · -- sound
      intro a e he
      unfold relaxFrom at he
      simp only at he
      by_cases hc : a ∈ up o ∧ wlt (wadd (ds.d o) (wf o a)) (ds.d a) = true
      · rw [ifP hc] at he
        cases hd : ds.d o with
        | none => rw [hd] at he; simp [wadd] at he
        | some e0 =>
          cases hw : wf o a with
          | none => rw [hd, hw] at he; simp [wadd] at he
          | some w =>
            rw [hd, hw] at he
            simp only [wadd, Option.some.injEq] at he
            subst he
            exact (inv.sound o e0 hd).append (UpPath.step hc.1 hw)
      · rw [ifN hc] at he; exact inv.sound a e he
    · -- off
      intro a ha hlta x hx
      rcases List.mem_append.1 ha with ha | ha
      · rw [hda a ha] at hlta ⊢
        have hb : wlt (ds.d a) best = true := wlt_of_wlt_wle hlta hle
        exact wle_trans (rf_d_le up wf o ds x) (inv.off a ha hb x hx)
      · have : a = o := by simpa using ha
        subst this
        rw [hdo]
        exact rf_d_off up wf a ds x hx
    · -- exact
      intro a ha e p hlt'
      rcases ha with rfl | ha
      · obtain ⟨e', he', hle'⟩ := inv.exact a (Or.inl rfl) e p (wlt_of_wlt_wle hlt' hle)
        have := rf_d_le up wf o ds a
        rw [he'] at this
        obtain ⟨e'', he'', hle''⟩ := wle_some_inv this
        exact ⟨e'', he'', by omega⟩
      · rcases List.mem_append.1 ha with ha | ha
        · obtain ⟨e', he', hle'⟩ := inv.exact a (Or.inr ha) e p (wlt_of_wlt_wle hlt' hle)
          rw [hda a ha]; exact ⟨e', he', hle'⟩
        · have : a = o := by simpa using ha
          subst this
          obtain ⟨e', he', hle'⟩ := hex e p hlt'
          rw [hdo]; exact ⟨e', he', hle'⟩
    · -- pred
      intro a e p hd hp
      unfold relaxFrom at hd hp
      simp only at hd hp
      by_cases hc : a ∈ up o ∧ wlt (wadd (ds.d o) (wf o a)) (ds.d a) = true
      · rw [ifP hc] at hd hp
        cases hp
        cases hdo' : ds.d o with
        | none => rw [hdo'] at hd; simp [wadd] at hd
        | some ep =>
          cases hw : wf o a with
          | none => rw [hdo', hw] at hd; simp [wadd] at hd
          | some w =>
            rw [hdo', hw] at hd
            simp only [wadd, Option.some.injEq] at hd
            have h1 : (relaxFrom up wf o ds).d o = some ep := by rw [hdo, hdo']
            refine ⟨by simp, hc.1, ep, w, h1, rfl, ?_⟩
            omega
      · rw [ifN hc] at hd hp
        obtain ⟨hpd, hpm, ep, w, hep, hw, he⟩ := inv.pred a e p hd hp
        exact ⟨List.mem_append_left _ hpd, hpm, ep, w, by rw [hda p hpd]; exact hep, hw, he⟩
    · -- reached
      intro a e hd hne
      unfold relaxFrom at hd ⊢
      simp only at hd ⊢
      by_cases hc : a ∈ up o ∧ wlt (wadd (ds.d o) (wf o a)) (ds.d a) = true
      · rw [ifP hc]; exact ⟨o, rfl⟩
      · rw [ifN hc] at hd ⊢; exact inv.reached a e hd hne
    · -- prs
      unfold relaxFrom
      simp only
      rw [ifN]
      · exact inv.prs
      · rintro ⟨hm, hl⟩
        have := inv.z
        rw [this] at hl
        cases hh : wadd (ds.d o) (wf o s) with
        | none => rw [hh] at hl; simp [wlt] at hl
        | some x => rw [hh] at hl; simp [wlt] at hl
  · rw [ifN hg]
    refine ⟨inv.z, inv.sound, ?_, ?_, ?_, inv.reached, inv.prs⟩
    · intro a ha hlta x hx
      rcases List.mem_append.1 ha with ha | ha
      · exact inv.off a ha (wlt_of_wlt_wle hlta hle) x hx
      · have : a = o := by simpa using ha
        subst this; exact absurd hlta hg
    · intro a ha e p hlt'
      rcases ha with rfl | ha
      · exact inv.exact a (Or.inl rfl) e p (wlt_of_wlt_wle hlt' hle)
      · rcases List.mem_append.1 ha with ha | ha
        · exact inv.exact a (Or.inr ha) e p (wlt_of_wlt_wle hlt' hle)
        · have : a = o := by simpa using ha
          subst this; exact hex e p hlt'
    · intro a e p hd hp
      obtain ⟨hpd, hpm, r⟩ := inv.pred a e p hd hp
      exact ⟨List.mem_append_left _ hpd, hpm, r⟩

theorem gstep_d_notmem (up wf best o ds x) (hx : x ∉ up o) : (gstep up wf best o ds).d x = ds.d x := by
  unfold gstep; split
  · exact rf_d_notmem up wf o ds x hx
  · rfl

/-- **Phase one** (`query.rs`, "below the lowest common ancestor the halves cannot meet: relax all of it"): with
    `best = +inf` every step is gated only by the rank being reached, which `relax_from` checks itself. A prefix
    of the chain is processed in order. -/
theorem dp_fold_const {up wt A N} (h : Ctx up wt A N) (wf s ch best) (hch : ch = chain h s) :
    ∀ (rest done tl : List Nat) (ds : DS), ch = done ++ (rest ++ tl) → DPInv h wf s done best ds →
      DPInv h wf s (done ++ rest) best (rest.foldl (fun d o => gstep up wf best o d) ds) := by
  intro rest
  induction rest with
  | nil => intro done tl ds _ inv; simpa using inv
  | cons o rest ih =>
    intro done tl ds hc inv
    have hs : ch.Pairwise (· < ·) := by rw [hch]; exact chain_sorted' h s
    rw [hc] at hs
    have hlt : ∀ a ∈ done, a < o := fun a ha =>
      (List.pairwise_append.1 hs).2.2 a ha o (by simp)
    have hdone : ∀ a, a ∈ chain h s → a < o → a ∈ done := by
      intro a ha hao
      rw [← hch, hc] at ha
      rcases List.mem_append.1 ha with ha | ha
      · exact ha
      · exfalso
        rcases List.mem_append.1 ha with ha | ha
        · rcases List.mem_cons.1 ha with rfl | ha
          · omega
          · have := List.rel_of_pairwise_cons (List.pairwise_append.1 hs).2.1 (List.mem_append_left _ ha)
            omega
        · have := List.rel_of_pairwise_cons (List.pairwise_append.1 hs).2.1 (List.mem_append_right _ ha)
          omega
    have inv' := dp_step h wf s done best best o ds inv (wle_refl _) hlt hdone
    have := ih (done ++ [o]) tl (gstep up wf best o ds) (by rw [hc]; simp) inv'
    simpa using this

/-! ## The meeting climb (`query.rs`) -/

/-- The climb's state: both halves, the best total so far, and the rank it met at. -/
structure LS where
  F : DS
  B : DS
  best : Wt
  meet : Option Nat

/-- One shared rank `o`: take it as the meeting if strictly better, then let each half offer its arcs only while
    its distance is below the (updated) best. `A` is the customized arc weight: forward reads `A r a`, backward
    `A a r` (`downward_weight`). -/
def stepL (up : Nat → List Nat) (A : Nat → Nat → Wt) (o : Nat) (ls : LS) : LS :=
  let cand := wadd (ls.F.d o) (ls.B.d o)
  let best1 := if wlt cand ls.best = true then cand else ls.best
  let meet1 := if wlt cand ls.best = true then some o else ls.meet
  ⟨gstep up (fun x y => A x y) best1 o ls.F, gstep up (fun x y => A y x) best1 o ls.B, best1, meet1⟩

theorem stepL_pos (up A o ls) (hc : wlt (wadd (ls.F.d o) (ls.B.d o)) ls.best = true) :
    (stepL up A o ls).best = wadd (ls.F.d o) (ls.B.d o) ∧ (stepL up A o ls).meet = some o := by
  unfold stepL; simp only; rw [ifP hc, ifP hc]; exact ⟨rfl, rfl⟩

theorem stepL_neg (up A o ls) (hc : ¬ wlt (wadd (ls.F.d o) (ls.B.d o)) ls.best = true) :
    (stepL up A o ls).best = ls.best ∧ (stepL up A o ls).meet = ls.meet := by
  unfold stepL; simp only; rw [ifN hc, ifN hc]; exact ⟨rfl, rfl⟩

structure LoopInv {up wt A N} (h : Ctx up wt A N) (s t : Nat) (pre1 pre2 P : List Nat) (ls : LS) : Prop where
  F : DPInv h (fun x y => A x y) s (pre1 ++ P) ls.best ls.F
  B : DPInv h (fun x y => A y x) t (pre2 ++ P) ls.best ls.B
  bi : (ls.best = none ∧ ls.meet = none) ∨
    ∃ m e1 e2, ls.best = some (e1 + e2) ∧ ls.meet = some m ∧ m ∈ P ∧ ls.F.d m = some e1 ∧ ls.B.d m = some e2
  k : ∀ o ∈ P, wle ls.best (wadd (ls.F.d o) (ls.B.d o))

theorem loop_inv {up wt A N} (h : Ctx up wt A N) (s t : Nat) (pre1 pre2 Sh : List Nat)
    (hc1 : chain h s = pre1 ++ Sh) (hc2 : chain h t = pre2 ++ Sh) :
    ∀ (R P : List Nat) (ls : LS), Sh = P ++ R → LoopInv h s t pre1 pre2 P ls →
      LoopInv h s t pre1 pre2 (P ++ R) (R.foldl (fun l o => stepL up A o l) ls) := by
  intro R
  induction R with
  | nil => intro P ls _ inv; simpa using inv
  | cons o R ih =>
    intro P ls hSh inv
    have s1 : (pre1 ++ (P ++ o :: R)).Pairwise (· < ·) := by rw [← hSh, ← hc1]; exact chain_sorted' h s
    have s2 : (pre2 ++ (P ++ o :: R)).Pairwise (· < ·) := by rw [← hSh, ← hc2]; exact chain_sorted' h t
    have key : ∀ (pre : List Nat) (c : Nat), chain h c = pre ++ Sh →
        (pre ++ (P ++ o :: R)).Pairwise (· < ·) →
        (∀ a ∈ pre ++ P, a < o) ∧ (∀ a, a ∈ chain h c → a < o → a ∈ pre ++ P) := by
      intro pre c hc hs
      have hs' : ((pre ++ P) ++ o :: R).Pairwise (· < ·) := by simpa [List.append_assoc] using hs
      refine ⟨fun a ha => (List.pairwise_append.1 hs').2.2 a ha o (by simp), ?_⟩
      intro a ha hao
      rw [hc, hSh] at ha
      have ha' : a ∈ (pre ++ P) ++ o :: R := by simpa [List.append_assoc] using ha
      rcases List.mem_append.1 ha' with ha' | ha'
      · exact ha'
      · exfalso
        rcases List.mem_cons.1 ha' with rfl | ha'
        · omega
        · have := List.rel_of_pairwise_cons (List.pairwise_append.1 hs').2.1 ha'
          omega
    obtain ⟨hlt1, hd1⟩ := key pre1 s hc1 s1
    obtain ⟨hlt2, hd2⟩ := key pre2 t hc2 s2
    have hoo : o ∉ up o := fun hm => by have := h.gt o o hm; omega
    -- the climb's decision
    have hlebest : wle (if wlt (wadd (ls.F.d o) (ls.B.d o)) ls.best = true then wadd (ls.F.d o) (ls.B.d o) else ls.best)
        ls.best := by
      by_cases hc : wlt (wadd (ls.F.d o) (ls.B.d o)) ls.best = true
      · rw [ifP hc]; exact Or.inr hc
      · rw [ifN hc]; exact wle_refl _
    have invF := dp_step h (fun x y => A x y) s (pre1 ++ P) ls.best _ o ls.F inv.F hlebest hlt1 hd1
    have invB := dp_step h (fun x y => A y x) t (pre2 ++ P) ls.best _ o ls.B inv.B hlebest hlt2 hd2
    have hdF : (stepL up A o ls).F.d o = ls.F.d o := by
      show (gstep up _ _ o ls.F).d o = _
      exact gstep_d_notmem _ _ _ o _ o hoo
    have hdB : (stepL up A o ls).B.d o = ls.B.d o := by
      show (gstep up _ _ o ls.B).d o = _
      exact gstep_d_notmem _ _ _ o _ o hoo
    have hold : ∀ m ∈ P, m ∉ up o := fun m hm hmem => by
      have := h.gt o m hmem; have := hlt1 m (List.mem_append_right _ hm); omega
    have hdFo : ∀ m ∈ P, (stepL up A o ls).F.d m = ls.F.d m := fun m hm =>
      gstep_d_notmem _ _ _ o _ m (hold m hm)
    have hdBo : ∀ m ∈ P, (stepL up A o ls).B.d m = ls.B.d m := fun m hm =>
      gstep_d_notmem _ _ _ o _ m (hold m hm)
    have inv' : LoopInv h s t pre1 pre2 (P ++ [o]) (stepL up A o ls) := by
      refine ⟨?_, ?_, ?_, ?_⟩
      · simpa [List.append_assoc, stepL] using invF
      · simpa [List.append_assoc, stepL] using invB
      · by_cases hc : wlt (wadd (ls.F.d o) (ls.B.d o)) ls.best = true
        · right
          cases hf : ls.F.d o with
          | none => rw [hf] at hc; simp [wadd, wlt] at hc
          | some e1 =>
            cases hb : ls.B.d o with
            | none => rw [hf, hb] at hc; simp [wadd, wlt] at hc
            | some e2 =>
              refine ⟨o, e1, e2, ?_, ?_, by simp, ?_, ?_⟩
              · rw [(stepL_pos up A o ls hc).1, hf, hb]; rfl
              · exact (stepL_pos up A o ls hc).2
              · rw [hdF, hf]
              · rw [hdB, hb]
        · rcases inv.bi with ⟨hb, hm⟩ | ⟨m, e1, e2, hb, hm, hmP, hf, hbm⟩
          · left; rw [(stepL_neg up A o ls hc).1, (stepL_neg up A o ls hc).2]; exact ⟨hb, hm⟩
          · right
            refine ⟨m, e1, e2, ?_, ?_, List.mem_append_left _ hmP, ?_, ?_⟩
            · rw [(stepL_neg up A o ls hc).1]; exact hb
            · rw [(stepL_neg up A o ls hc).2]; exact hm
            · rw [hdFo m hmP, hf]
            · rw [hdBo m hmP, hbm]
      · intro o' ho'
        rcases List.mem_append.1 ho' with ho' | ho'
        · rw [hdFo o' ho', hdBo o' ho']
          exact wle_trans hlebest (inv.k o' ho')
        · have : o' = o := by simpa using ho'
          subst this
          rw [hdF, hdB]
          by_cases hc : wlt (wadd (ls.F.d o') (ls.B.d o')) ls.best = true
          · rw [(stepL_pos up A _ ls hc).1]; exact wle_refl _
          · rw [(stepL_neg up A _ ls hc).1]
            exact wle_of_not_wlt hc
    have := ih (P ++ [o]) (stepL up A o ls) (by rw [hSh]; simp) inv'
    simpa [List.append_assoc] using this

theorem wle_wlt_false {x y : Wt} (h1 : wle x y) (h2 : wlt y x = true) : False := by
  rcases h1 with rfl | h1
  · cases x <;> simp [wlt] at h2
  · cases x <;> cases y <;> simp_all [wlt] <;> omega

theorem upPath_lt {up wt A N} (h : Ctx up wt A N) {wf a z e} (p : UpPath up wf a z e) (ha : a < N) : z < N := by
  obtain ⟨k, p⟩ := p
  induction p with
  | nil => exact ha
  | cons hm _ _ ih => exact ih (h.lt _ _ hm)

/-- **The search finds the true distance** (the heart of `Cch::query`, over `Nat` weights). The two chains meet in
    a shared tail `Sh` (`chain s = pre1 ++ Sh`, `chain t = pre2 ++ Sh`) holding every common ancestor; the lower
    halves are processed in full, then the shared ranks are climbed with the pruning of `query.rs`. The final `best`
    is exactly the cheapest walk `s -> t` in the original graph (`IsMin`, with `none` = no walk), and the meeting
    rank is recorded iff a walk exists. -/
theorem query_core {up wt A N} (h : Ctx up wt A N) (s t : Nat) (hst : s ≠ t) (hs : s < N) (ht : t < N)
    (pre1 pre2 Sh : List Nat) (hc1 : chain h s = pre1 ++ Sh) (hc2 : chain h t = pre2 ++ Sh)
    (hcomm : ∀ m, m ∈ chain h s → m ∈ chain h t → m ∈ Sh) :
    let ls0 : LS := ⟨pre1.foldl (fun d o => gstep up (fun x y => A x y) none o d) (DS.init s),
      pre2.foldl (fun d o => gstep up (fun x y => A y x) none o d) (DS.init t), none, none⟩
    let ls := Sh.foldl (fun l o => stepL up A o l) ls0
    IsMin wt N s t ls.best ∧ (ls.meet = none ↔ ls.best = none) ∧
      LoopInv h s t pre1 pre2 Sh ls := by
  intro ls0 ls
  have p1 := dp_fold_const h (fun x y => A x y) s (chain h s) none rfl pre1 [] Sh (DS.init s)
    (by simpa using hc1) (dp_init h _ s none)
  have p2 := dp_fold_const h (fun x y => A y x) t (chain h t) none rfl pre2 [] Sh (DS.init t)
    (by simpa using hc2) (dp_init h _ t none)
  have inv0 : LoopInv h s t pre1 pre2 [] ls0 := by
    refine ⟨by simpa using p1, by simpa using p2, Or.inl ⟨rfl, rfl⟩, by simp⟩
  have inv := loop_inv h s t pre1 pre2 Sh hc1 hc2 Sh [] ls0 (by simp) inv0
  simp only [List.nil_append] at inv
  refine ⟨⟨?_, ?_⟩, ?_, inv⟩
  · -- the lower bound: nothing is cheaper
    intro c hc
    obtain ⟨m, e1, e2, f1, f2, hle⟩ := updown_complete h N s t c hc
    have hm1 := upPath_mem_chain h f1
    have hm2 := upPath_mem_chain h f2
    have hmSh := hcomm m hm1 hm2
    by_cases hlt : wlt (some (e1 + e2)) ls.best = true
    · exfalso
      obtain ⟨f, hf, hfle⟩ := inv.F.exact m (Or.inr (List.mem_append_right _ hmSh)) e1 f1
        (wlt_some_le (by omega) hlt)
      obtain ⟨b, hb, hble⟩ := inv.B.exact m (Or.inr (List.mem_append_right _ hmSh)) e2 f2
        (wlt_some_le (by omega) hlt)
      have := inv.k m hmSh
      rw [hf, hb] at this
      exact wle_wlt_false (wle_trans this (wle_some (by omega : f + b ≤ e1 + e2))) hlt
    · obtain ⟨e, he, hle2⟩ := wle_some_inv (wle_of_not_wlt hlt)
      exact ⟨e, he, by omega⟩
  · -- soundness: a finite best is a real walk
    intro e he
    rcases inv.bi with ⟨hb, _⟩ | ⟨m, e1, e2, hb, hm, hmP, hf, hbm⟩
    · rw [hb] at he; cases he
    · rw [hb] at he
      cases he
      have u1 := inv.F.sound m e1 hf
      have u2 := inv.B.sound m e2 hbm
      have hmN := upPath_lt h u1 hs
      rcases updown_sound h hmN u1 u2 with ⟨rfl, _⟩ | w
      · exact absurd rfl hst
      · exact w
  · rcases inv.bi with ⟨hb, hm⟩ | ⟨m, e1, e2, hb, hm, _⟩
    · exact ⟨fun _ => hb, fun _ => hm⟩
    · exact ⟨fun h => (by rw [hm] at h; cases h), fun h => (by rw [hb] at h; cases h)⟩

/-! ## The shared tail (`query.rs`) -/

/-- `forward.ranks.iter().rev().zip(backward.ranks.iter().rev()).take_while(equal).count()`. -/
def commonCount (f b : List Nat) : Nat :=
  ((f.reverse.zip b.reverse).takeWhile (fun p => p.1 == p.2)).length

theorem tw_len_le : ∀ (l1 l2 : List Nat),
    ((l1.zip l2).takeWhile (fun p => p.1 == p.2)).length ≤ l1.length ∧
    ((l1.zip l2).takeWhile (fun p => p.1 == p.2)).length ≤ l2.length ∧
    l1.take ((l1.zip l2).takeWhile (fun p => p.1 == p.2)).length =
      l2.take ((l1.zip l2).takeWhile (fun p => p.1 == p.2)).length
  | [], _ => by simp
  | _ :: _, [] => by simp
  | a :: l1, b :: l2 => by
    have ih := tw_len_le l1 l2
    simp only [List.zip_cons_cons, List.takeWhile_cons]
    by_cases hab : (a == b) = true
    · have : a = b := by simpa using hab
      subst this
      simp only [hab, if_true, List.length_cons, List.take_succ_cons]
      refine ⟨by omega, by omega, ?_⟩
      rw [ih.2.2]
    · simp [hab]

theorem tw_self : ∀ x : List Nat, (x.zip x).takeWhile (fun p => p.1 == p.2) = x.zip x
  | [] => rfl
  | a :: l => by simp [List.zip_cons_cons, List.takeWhile_cons, tw_self l]

theorem tw_ge (x P1 P2 : List Nat) :
    x.length ≤ (((x ++ P1).zip (x ++ P2)).takeWhile (fun p => p.1 == p.2)).length := by
  rw [List.zip_append rfl, List.takeWhile_append]
  have hall : ((x.zip x).takeWhile (fun p => p.1 == p.2)).length = (x.zip x).length := by
    rw [tw_self]
  rw [ifP hall]
  simp

/-- **The shared tail**: the chains split as `pre ++ Sh`, the same `Sh` for both (`query.rs`'s
    `forward_first_shared = len - shared`), and every common ancestor is in `Sh`. -/
theorem shared_spec {up wt A N} (h : Ctx up wt A N) (s t : Nat) :
    let f := chain h s
    let b := chain h t
    let c := commonCount f b
    f = f.take (f.length - c) ++ f.drop (f.length - c) ∧
    b = b.take (b.length - c) ++ f.drop (f.length - c) ∧
    ∀ m, m ∈ f → m ∈ b → m ∈ f.drop (f.length - c) := by
  intro f b c
  have ⟨hl1, hl2, heq⟩ := tw_len_le f.reverse b.reverse
  have hlf : c ≤ f.length := by have := hl1; simp [c, commonCount, f] at this ⊢; exact this
  have hlb : c ≤ b.length := by have := hl2; simp [c, commonCount, b] at this ⊢; exact this
  have hsh : b.drop (b.length - c) = f.drop (f.length - c) := by
    have e1 : f.reverse.take c = (f.drop (f.length - c)).reverse := by
      rw [List.take_reverse]
    have e2 : b.reverse.take c = (b.drop (b.length - c)).reverse := by
      rw [List.take_reverse]
    have : (b.drop (b.length - c)).reverse = (f.drop (f.length - c)).reverse := by
      rw [← e1, ← e2]; exact heq.symm
    simpa using congrArg List.reverse this
  refine ⟨(List.take_append_drop _ _).symm, ?_, ?_⟩
  · rw [← hsh]; exact (List.take_append_drop _ _).symm
  · intro m hmf hmb
    obtain ⟨P1, hP1⟩ := chain_suffix h N s (by omega) m hmf
    obtain ⟨P2, hP2⟩ := chain_suffix h N t (by omega) m hmb
    have hge : (chain h m).length ≤ c := by
      have := tw_ge (chain h m).reverse P1.reverse P2.reverse
      have e1 : f.reverse = (chain h m).reverse ++ P1.reverse := by simp [f, hP1]
      have e2 : b.reverse = (chain h m).reverse ++ P2.reverse := by simp [b, hP2]
      simp only [c, commonCount, e1, e2]
      simpa using this
    have hin : m ∈ f.drop (f.length - (chain h m).length) := by
      have : f.drop (f.length - (chain h m).length) = chain h m := by
        have hf : f = P1 ++ chain h m := hP1
        rw [hf]; simp
      rw [this]; exact chain_mem_self h m
    have hdd : f.drop (f.length - c) = (f.drop (f.length - c)) := rfl
    -- `drop (len - c)` holds `drop (len - |chain m|)`, as `c ≥ |chain m|`
    have : f.drop (f.length - (chain h m).length) =
        (f.drop (f.length - c)).drop ((f.length - (chain h m).length) - (f.length - c)) := by
      rw [List.drop_drop]; congr 1; omega
    rw [this] at hin
    exact List.mem_of_mem_drop hin

end NI
