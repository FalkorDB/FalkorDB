import CchRedesignReview.Writer
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false

/-!
# Writer rebase (03 §5 table) and retention at publish (03 §7)

`priv_rebase`: every reachable row of the rebase table keeps `Priv`, so the writer's own pairs survive the
rebase. Row 4 (`Built` fork, `Building` published, same generation) is excluded by the hypothesis `hbb`, which
`row4_unreachable` (`Main.lean`) discharges for every reachable state.

`priv_retain`: retention turns `Priv` into `PInv` for the published version.
-/
namespace CchReview

section
variable {B c : Nat} {w : List (Nat × Pair)} {O : List Pair} {n0 nx : Nat}

/-- Row 1: relink the fork's own nodes onto the published log, then the bound. -/
theorem priv_relink {bt : Nat} {e1 ce : Entry} {ph : Phase} (hst : ce.st = .building ph)
    (h : Priv B c w O n0 e1 nx) (hng : e1.gen < n0) (hcur : PInv B w ⟨c, bt, ce⟩) (hcf : Fresh n0 ce) :
    Priv B c w O n0 (checkIf D.checkAll B (c + 1) (relinkE (own (c + 1) e1.log) ce)) nx := by
  refine priv_checkIf (ph := ph) (e := relinkE (own (c + 1) e1.log) ce) hst ?_
  have hok := h.ownK hng
  have hos : ∀ n ∈ own (c + 1) e1.log, n.seq = c + 1 := fun n hn => own_seq hn
  have hcl : ∀ n ∈ ce.log, n.seq < c + 1 := fun n hn => Nat.lt_succ_of_le (hcur.seq_le n hn)
  have hown : own (c + 1) (own (c + 1) e1.log ++ ce.log) = own (c + 1) e1.log := own_seam _ _ _ hos hcl
  have hb : ce.isBuilt = false := isBuilt_building ce ph hst
  simp only [relinkE]
  refine ⟨fun p => Nat.le_trans (hcur.res_le p) (Nat.le_succ c), ?_, ⟨_, ce.log, rfl, hos, hcur.seq_le⟩, ?_, ?_, ?_,
    ?_, ?_, ?_, ?_, ?_, ⟨h.fresh.1, Nat.lt_of_lt_of_le hcf.gen h.fresh.1,
      Nat.lt_of_lt_of_le hcf.lineage h.fresh.1, by simp [Entry.upkeep, hst]⟩⟩
  · rw [sumPairs_append, show ce.pairs = sumPairs ce.log from hcur.cnt]
  · intro n hn p hp
    rcases List.mem_append.1 hn with hn | hn
    · exact h.sound n (own_sub hn) p hp
    · exact Or.inl (hcur.sound n hn p hp)
  · intro s p hsp hs
    rcases hcur.nlw s p hsp hs with h1 | h1
    · exact Or.inl (Nat.le_trans h1 (Nat.le_refl _))
    · exact Or.inr (covered_append_right _ h1)
  · intro p hp; exact Or.inr (covered_append_left _ (hok p hp))
  · intro _ p hp; rw [hown]; exact hok p hp
  · intro h'; simp [Entry.isBuilt, hst] at h'
  · intro h'; simp [Entry.isBuilt, hst] at h'
  · intro h'; simp [Entry.isBuilt, hst] at h'
  · intro u hu; simp [hst] at hu

/-- Rows 2 and 3 (S6), within budget: the §6 flush over the own nodes onto the published `Built` copy. -/
theorem priv_flushOwn {bt : Nat} {e1 ce : Entry} {up : Option Upkeep} (hst : ce.st = .built up)
    (hov : hasOv (own (c + 1) e1.log) = false)
    (h : Priv B c w O n0 e1 nx) (hng : e1.gen < n0) (hcur : PInv B w ⟨c, bt, ce⟩) (hcf : Fresh n0 ce) :
    Priv B c w O n0 (flushOwn (c + 1) (own (c + 1) e1.log) ce) nx := by
  have hok := h.ownK hng
  have hbt : ce.isBuilt = true := isBuilt_built ce up hst
  have hos : ∀ n ∈ own (c + 1) e1.log, n.seq = c + 1 := fun n hn => own_seq hn
  have hcl : ∀ n ∈ ce.log, n.seq < c + 1 := fun n hn => Nat.lt_succ_of_le (hcur.seq_le n hn)
  have hown : own (c + 1) (own (c + 1) e1.log ++ ce.log) = own (c + 1) e1.log := own_seam _ _ _ hos hcl
  have rle : ∀ p, ce.res p ≤ c + 1 := fun p => Nat.le_trans (hcur.res_le p) (Nat.le_succ c)
  have rold : ∀ s p, (s, p) ∈ w → s ≤ c → s ≤ resolve ce.res (allPairs (own (c + 1) e1.log)) (c + 1) p :=
    fun s p hsp hs => Nat.le_trans (hcur.built hbt s p hsp hs) (resolve_ge p (rle p))
  have rown : ∀ p ∈ O, c + 1 ≤ resolve ce.res (allPairs (own (c + 1) e1.log)) (c + 1) p := by
    intro p hp; rw [resolve_mem (allPairs_of_covered (hok p hp) hov)]; exact Nat.le_refl _
  simp only [flushOwn]
  refine ⟨⟨fun p => resolve_le p (rle p), ?_, ⟨_, ce.log, rfl, hos, hcur.seq_le⟩, ?_,
    fun s p hsp hs => Or.inl (rold s p hsp hs), fun p hp => Or.inl (rown p hp), ?_, fun _ => rold, fun _ => rown,
    ?_, ?_, ⟨h.fresh.1, Nat.lt_of_lt_of_le hcf.gen h.fresh.1, Nat.lt_of_lt_of_le hcf.lineage h.fresh.1, ?_⟩⟩, ?_⟩
  · rw [sumPairs_append, show ce.pairs = sumPairs ce.log from hcur.cnt]
  · intro n hn p hp
    rcases List.mem_append.1 hn with hn | hn
    · exact h.sound n (own_sub hn) p hp
    · exact Or.inl (hcur.sound n hn p hp)
  · intro _ p hp; rw [hown]; exact hok p hp
  · intro _; simp [hasOv_append, hov, hcur.noOv hbt]
  · intro u hu
    have hu' : ce.st = .built (some u) := hu
    obtain ⟨a, _, cc⟩ := hcur.upSome u hu'
    refine ⟨by simp at a ⊢; omega, ?_⟩
    rintro s p (⟨hsp, hs⟩ | ⟨rfl, hp⟩) h1
    · obtain ⟨n, hn, hh⟩ := cc s p hsp h1 hs
      exact ⟨n, List.mem_append_right _ hn, hh⟩
    · obtain ⟨n, hn, hh⟩ := covered_noOv (hok p hp) hov
      exact ⟨n, List.mem_append_left _ hn, hh⟩
  · intro u hu; exact Nat.lt_of_lt_of_le (hcf.ticket u hu) h.fresh.1
  · intro h'; simp [Entry.isBuilt, hst] at h'

/-- Rows 2 and 3 (S6), over budget (or an own Overflow): `Absorbed::Restart` under a new generation, base the
    published hierarchy, log the own nodes. -/
theorem priv_restart {bt : Nat} {e1 ce : Entry} (hbt : ce.isBuilt = true)
    (h : Priv B c w O n0 e1 nx) (hng : e1.gen < n0) (hcur : PInv B w ⟨c, bt, ce⟩) (hcf : Fresh n0 ce) :
    Priv B c w O n0 (restart D B (c + 1) (own (c + 1) e1.log) ce nx) (nx + 1) := by
  have hok := h.ownK hng
  have hos : ∀ n ∈ own (c + 1) e1.log, n.seq = c + 1 := fun n hn => own_seq hn
  refine priv_checkIf (ph := .repair) (e := restartE (own (c + 1) e1.log) ce nx) rfl ?_
  simp only [restartE]
  refine ⟨fun p => Nat.le_trans (hcur.res_le p) (Nat.le_succ c), rfl, ⟨_, [], by simp, hos, by simp⟩, ?_,
    fun s p hsp hs => Or.inl (hcur.built hbt s p hsp hs), fun p hp => Or.inr (hok p hp), ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro n hn p hp; exact h.sound n (own_sub hn) p hp
  · intro hg; have := h.fresh.1; simp at hg; omega
  · intro h'; simp [Entry.isBuilt] at h'
  · intro h'; simp [Entry.isBuilt] at h'
  · intro h'; simp [Entry.isBuilt] at h'
  · intro u hu; simp at hu
  · exact ⟨by have := h.fresh.1; omega, by simp, by have := hcf.lineage; have := h.fresh.1; simp; omega,
      by simp [Entry.upkeep]⟩

/-- Row 3, fold: the fork's copy, re-folded (reading R3). -/
theorem priv_refold {e1 ce : Entry} {up : Option Upkeep} (hst : e1.st = .built up) (h : Priv B c w O n0 e1 nx) :
    Priv B c w O n0 (refoldE (own (c + 1) e1.log) e1 ce) nx := by
  have hbt : e1.isBuilt = true := isBuilt_built e1 up hst
  have hos : ∀ n ∈ own (c + 1) e1.log, n.seq = c + 1 := fun n hn => own_seq hn
  simp only [refoldE]
  refine ⟨⟨h.res_le, rfl, ⟨_, [], by simp, hos, by simp⟩, ?_, fun s p hsp hs => Or.inl (h.builtOld hbt s p hsp hs),
    fun p hp => Or.inl (h.builtOwn hbt p hp), ?_, fun _ => h.builtOld hbt, fun _ => h.builtOwn hbt, ?_, ?_,
    ⟨h.fresh.1, h.fresh.2.gen, h.fresh.2.lineage, by simp [Entry.upkeep]⟩⟩, ?_⟩
  · intro n hn p hp; exact h.sound n (own_sub hn) p hp
  · intro hg p hp; rw [own_own]; exact h.ownK hg p hp
  · intro _; exact hasOv_sub (fun n hn => own_sub hn) (h.noOv hbt)
  · intro u hu; simp at hu
  · intro h'; simp [Entry.isBuilt] at h'

/-- **Every reachable row of the rebase table keeps the writer's own pairs** (03 §5: "every row above has
    them"). `hbb` excludes row 4. -/
theorem priv_rebase {bt : Nat} {e1 ce : Entry} (h : Priv B c w O n0 e1 nx) (hng : e1.gen < n0)
    (hcur : PInv B w ⟨c, bt, ce⟩) (hcf : Fresh n0 ce) (hbb : e1.isBuilt = true → ce.isBuilt = true) (b : Bool) :
    Priv B c w O n0 (rebase D B (c + 1) b e1 ce nx).1 (rebase D B (c + 1) b e1 ce nx).2 := by
  cases h1 : e1.st with
  | building ph1 =>
    cases h2 : ce.st with
    | building ph2 =>
      simp only [rebase, h1, h2, D, if_true]
      exact priv_relink h2 h hng hcur hcf
    | built up =>
      have hbt : ce.isBuilt = true := isBuilt_built ce up h2
      simp only [rebase, h1, h2]
      split
      · rename_i hc
        have hov : hasOv (own (c + 1) e1.log) = false := by simp at hc; exact hc.2
        exact priv_flushOwn h2 hov h hng hcur hcf
      · exact priv_restart hbt h hng hcur hcf
  | built up1 =>
    have hb1 : e1.isBuilt = true := isBuilt_built e1 up1 h1
    cases h2 : ce.st with
    | building ph2 =>
      have := hbb hb1; simp [Entry.isBuilt, h2] at this
    | built up =>
      have hbt : ce.isBuilt = true := isBuilt_built ce up h2
      have hov : hasOv (own (c + 1) e1.log) = false := hasOv_sub (fun n hn => own_sub hn) (h.noOv hb1)
      simp only [rebase, h1, h2]
      split
      · exact priv_refold h1 h
      · split
        · exact priv_flushOwn h2 hov h hng hcur hcf
        · exact priv_restart hbt h hng hcur hcf

/-- **Retention (03 §7) turns the private copy into a published version that keeps `PInv`.** -/
theorem priv_retain {bt : Nat} {e : Entry} (h : Priv B c w O n0 e nx) (hw : ∀ x ∈ w, x.1 ≤ c) :
    PInv B (w ++ O.map (fun p => (c + 1, p))) ⟨c + 1, bt, retain e⟩ ∧ Fresh nx (retain e) := by
  obtain ⟨M, Rr, hl, hM, hR⟩ := h.shape
  have hseq : ∀ n ∈ e.log, n.seq ≤ c + 1 := by
    intro n hn; rw [hl] at hn; rcases List.mem_append.1 hn with hn | hn
    · exact Nat.le_of_eq (hM n hn)
    · exact Nat.le_succ_of_le (hR n hn)
  have hsnd : ∀ n ∈ e.log, ∀ p ∈ n.ps, (n.seq, p) ∈ w ++ O.map (fun p => (c + 1, p)) := by
    intro n hn p hp
    rcases h.sound n hn p hp with h1 | ⟨h1, h2⟩
    · exact List.mem_append_left _ h1
    · rw [h1]; exact List.mem_append_right _ (List.mem_map.2 ⟨p, h2, rfl⟩)
  -- a write is an older one (`seq ≤ c`) or an own one
  have hsplit : ∀ s p, (s, p) ∈ w ++ O.map (fun p => (c + 1, p)) → ((s, p) ∈ w ∧ s ≤ c) ∨ (s = c + 1 ∧ p ∈ O) := by
    intro s p hsp
    rcases List.mem_append.1 hsp with h1 | h1
    · exact Or.inl ⟨h1, hw _ h1⟩
    · obtain ⟨q, hq, hqe⟩ := List.mem_map.1 h1; simp at hqe; obtain ⟨rfl, rfl⟩ := hqe; exact Or.inr ⟨rfl, hq⟩
  have hbuilt : e.isBuilt = true → ∀ s p, (s, p) ∈ w ++ O.map (fun p => (c + 1, p)) → s ≤ c + 1 → s ≤ e.res p := by
    intro hb s p hsp _
    rcases hsplit s p hsp with ⟨h1, h2⟩ | ⟨rfl, h2⟩
    · exact h.builtOld hb s p h1 h2
    · exact h.builtOwn hb p h2
  cases hst : e.st with
  | building ph =>
    have hb : e.isBuilt = false := isBuilt_building e ph hst
    simp only [retain, hst]
    refine ⟨⟨h.res_le, hseq, h.cnt, hsnd, ?_, fun h' => by simp [hb] at h', fun h' => by simp [hb] at h',
      fun h' => by simp [hst] at h', fun u hu => by simp [hst] at hu, fun _ => h.bound hb⟩, h.fresh.2⟩
    intro s p hsp _
    rcases hsplit s p hsp with ⟨h1, h2⟩ | ⟨rfl, h2⟩
    · exact h.old s p h1 h2
    · exact h.ownN p h2
  | built up =>
    have hb : e.isBuilt = true := isBuilt_built e up hst
    cases up with
    | none =>
      simp only [retain, hst]
      refine ⟨⟨h.res_le, by simp, by simp, by simp, fun s p hsp hs => Or.inl (hbuilt hb s p hsp hs),
        fun _ => hbuilt hb, fun _ => by simp, fun _ => rfl, fun u hu => by simp [hst] at hu, fun h' => ?_⟩,
        ⟨h.fresh.2.gen, h.fresh.2.lineage, fun u' hu' => by simp [Entry.upkeep] at hu'⟩⟩
      simp [Entry.isBuilt, hst] at h'
    | some u =>
      simp only [retain, hst]
      obtain ⟨a, cov⟩ := h.up u hst
      refine ⟨⟨h.res_le, fun n hn => hseq n (List.mem_filter.1 hn).1, rfl,
        fun n hn => hsnd n (List.mem_filter.1 hn).1, fun s p hsp hs => Or.inl (hbuilt hb s p hsp hs),
        fun _ => hbuilt hb, fun _ => hasOv_sub (fun n hn => (List.mem_filter.1 hn).1) (h.noOv hb),
        fun h' => by simp [hst] at h', ?_, fun h' => ?_⟩,
        ⟨h.fresh.2.gen, h.fresh.2.lineage, fun u' hu' => h.fresh.2.ticket u' (by simp [Entry.upkeep, hst] at hu' ⊢; exact hu')⟩⟩
      · intro u' hu'
        simp [hst] at hu'; subst hu'
        refine ⟨a, fun n hn => by simpa using (List.mem_filter.1 hn).2, ?_⟩
        intro s p hsp h1 h2
        obtain ⟨n, hn, hs, hp⟩ := cov s p (hsplit s p hsp) h1
        exact ⟨n, List.mem_filter.2 ⟨hn, by simp [hs]; omega⟩, hs, hp⟩
      · simp [Entry.isBuilt, hst] at h'

end

end CchReview
