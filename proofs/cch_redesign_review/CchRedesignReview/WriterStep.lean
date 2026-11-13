import CchRedesignReview.Rebase
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false

/-!
# The writer step keeps the invariant

`Track B c n0 ce e`: the writer's copy `e` against the published entry `ce`: either a new generation
(hand-off, S7, Restart; then `Building`), or the published generation, count and state, with the log only
extended by own nodes (`Ext`) while `Building`, and `upkeep` the published one, or a fresh ticket.
-/
namespace CchReview

def Track (B c n0 : Nat) (ce e : Entry) : Prop :=
  (n0 ≤ e.gen ∧ e.isBuilt = false) ∨
  (e.gen = ce.gen ∧ e.commits = ce.commits ∧ e.isBuilt = ce.isBuilt ∧ (e.isBuilt = false → Ext B c ce e) ∧
    ∀ u, e.upkeep = some u → ce.upkeep = some u ∨ n0 ≤ u.ticket)

theorem track_refl (B c n0 : Nat) (ce : Entry) : Track B c n0 ce ce :=
  Or.inr ⟨rfl, rfl, rfl, fun _ => Ext.refl B c ce, fun u hu => Or.inl hu⟩

theorem checkB_fields (B v : Nat) (e : Entry) :
    (checkB B v e).st = e.st ∧ (checkB B v e).res = e.res ∧ (checkB B v e).gen = e.gen ∧
    (checkB B v e).commits = e.commits ∧ (checkB B v e).lineage = e.lineage := by
  unfold checkB; split <;> simp

theorem isBuilt_of_st {a b : Entry} (h : a.st = b.st) : a.isBuilt = b.isBuilt := by simp [Entry.isBuilt, h]
theorem upkeep_of_st {a b : Entry} (h : a.st = b.st) : a.upkeep = b.upkeep := by simp [Entry.upkeep, h]

/-- The writer's bound check after its own nodes went on top is an `Ext` (03 §5). -/
theorem ext_checkB_top (B c : Nat) (e : Entry) (o : List Node) (ho : ∀ n ∈ o, n.seq = c + 1)
    (t : Entry) (hl : t.log = o ++ e.log) (hp : t.pairs = sumPairs t.log)
    (hst : t.st = e.st) (hres : t.res = e.res) (hg : t.gen = e.gen) (hc : t.commits = e.commits)
    (hlin : t.lineage = e.lineage) : Ext B c e (checkB B (c + 1) t) := by
  unfold checkB
  split
  · rename_i hlt
    refine ⟨hst, hres, hg, hc, hlin, [⟨c + 1, .overflow⟩], by simp, Or.inr ⟨rfl, by simp⟩⟩
  · rename_i hle
    refine ⟨hst, hres, hg, hc, hlin, o, fun n hn => by rw [ho n hn]; omega, Or.inl ⟨hl, Or.inr ?_⟩⟩
    rw [← hp]; omega

theorem flushB_fields (R : Rules) (B v : Nat) (ps : List Pair) (up : Option Upkeep) (e : Entry) :
    (flushB R B v ps up e).gen = e.gen ∧ (flushB R B v ps up e).commits = e.commits ∧
    (flushB R B v ps up e).isBuilt = true ∧
    ((flushB R B v ps up e).st = .built none ∨ (flushB R B v ps up e).st = .built up) := by
  unfold flushB; cases up with
  | none => simp [Entry.isBuilt]
  | some u => simp only; split <;> simp [Entry.isBuilt]

/-- One segment keeps `Track`. -/
theorem track_seg {B c w O n0 nx} {ce e : Entry} (ht : Track B c n0 ce e) (hp : Priv B c w O n0 e nx)
    (ps : List Pair) (o : Out) : Track B c n0 ce (seg D B (c + 1) ps o e nx).1 := by
  have hnx := hp.fresh.1
  cases hst : e.st with
  | building ph =>
    simp only [seg, hst]
    obtain ⟨f1, _, f3, f4, _⟩ := checkB_fields B (c + 1) (appendE (c + 1) ps e)
    rcases ht with ⟨hg, hb⟩ | ⟨hg, hc, hb, hx, _⟩
    · left; refine ⟨by rw [f3]; simpa [appendE] using hg, ?_⟩
      rw [isBuilt_of_st f1, isBuilt_of_st (show (appendE (c + 1) ps e).st = e.st from rfl)]; exact hb
    · have hb' : e.isBuilt = false := isBuilt_building e ph hst
      have hext : Ext B c e (checkB B (c + 1) (appendE (c + 1) ps e)) :=
        ext_checkB_top B c e [⟨c + 1, .pairs ps⟩] (by simp) _ (by simp [appendE])
          (by simp [appendE, hp.cnt]; omega) rfl rfl rfl rfl rfl
      right
      refine ⟨by rw [f3]; simpa [appendE] using hg, by rw [f4]; simpa [appendE] using hc,
        by rw [isBuilt_of_st f1, isBuilt_of_st (show (appendE (c + 1) ps e).st = e.st from rfl)]; exact hb,
        fun _ => (hx hb').trans hext, ?_⟩
      intro u hu; rw [upkeep_of_st f1] at hu; simp [appendE, Entry.upkeep, hst] at hu
  | built up =>
    have hbt : e.isBuilt = true := isBuilt_built e up hst
    rcases ht with ⟨_, hb⟩ | ⟨hg, hc, hb, _, hu⟩
    · simp [hbt] at hb
    cases o with
    | fits =>
      simp only [seg, hst]
      obtain ⟨g1, c1, b1, s1⟩ := flushB_fields D B (c + 1) ps up e
      right
      refine ⟨g1.trans hg, c1.trans hc, b1.trans (hbt.symm.trans hb), fun h' => by simp [b1] at h', ?_⟩
      intro u hu'
      rcases s1 with s1 | s1
      · simp [Entry.upkeep, s1] at hu'
      · have : e.upkeep = some u := by simp [Entry.upkeep, s1] at hu'; simp [Entry.upkeep, hst, hu']
        exact hu _ this
    | setUp k =>
      simp only [seg, hst]
      obtain ⟨g1, c1, b1, s1⟩ := flushB_fields D B (c + 1) ps (some ⟨k, nx, c + 1⟩) e
      right
      refine ⟨g1.trans hg, c1.trans hc, b1.trans (hbt.symm.trans hb), fun h' => by simp [b1] at h', ?_⟩
      intro u hu'
      rcases s1 with s1 | s1
      · simp [Entry.upkeep, s1] at hu'
      · simp [Entry.upkeep, s1] at hu'; subst hu'; exact Or.inr hnx
    | abort =>
      simp only [seg, hst, D, checkIf, ite_true]
      obtain ⟨f1, _, f3, _, _⟩ := checkB_fields B (c + 1) (handoffE (c + 1) ps e nx)
      left; exact ⟨by rw [f3]; simpa [handoffE] using hnx, by rw [isBuilt_of_st f1]; simp [handoffE, Entry.isBuilt]⟩
    | s7 =>
      simp only [seg, hst, D, checkIf, ite_true]
      obtain ⟨f1, _, f3, _, _⟩ := checkB_fields B (c + 1) (s7E (c + 1) ps nx)
      left; exact ⟨by rw [f3]; simpa [s7E] using hnx, by rw [isBuilt_of_st f1]; simp [s7E, Entry.isBuilt]⟩

theorem track_segs {B c w n0} {ce : Entry} (l : List (List Pair × Out)) :
    ∀ {O nx} {e : Entry}, Track B c n0 ce e → Priv B c w O n0 e nx →
      Track B c n0 ce (segs D B (c + 1) l e nx).1 := by
  induction l with
  | nil => intro O nx e ht _; simpa [segs] using ht
  | cons x rest ih =>
    intro O nx e ht hp
    obtain ⟨ps, o⟩ := x
    simp only [segs]
    exact ih (track_seg ht hp ps o) (priv_seg hp ps o)

/-- The rebase result tracks the published entry. -/
theorem track_rebase {B c w O n0 nx bt} {e1 ce : Entry} (hp : Priv B c w O n0 e1 nx) (hg : e1.gen = ce.gen)
    (hcur : PInv B w ⟨c, bt, ce⟩) (hbb : e1.isBuilt = true → ce.isBuilt = true) (b : Bool) :
    Track B c n0 ce (rebase D B (c + 1) b e1 ce nx).1 := by
  have hnx := hp.fresh.1
  have hos : ∀ n ∈ own (c + 1) e1.log, n.seq = c + 1 := fun n hn => own_seq hn
  have restart_t : Track B c n0 ce (restart D B (c + 1) (own (c + 1) e1.log) ce nx) := by
    simp only [restart, D, checkIf, ite_true]
    obtain ⟨f1, _, f3, _, _⟩ := checkB_fields B (c + 1) (restartE (own (c + 1) e1.log) ce nx)
    left; exact ⟨by rw [f3]; simpa [restartE] using hnx, by rw [isBuilt_of_st f1]; simp [restartE, Entry.isBuilt]⟩
  have flush_t : ∀ up, ce.st = .built up → Track B c n0 ce (flushOwn (c + 1) (own (c + 1) e1.log) ce) := by
    intro up hst
    refine Or.inr ⟨rfl, rfl, rfl, fun h' => ?_, fun u hu => Or.inl hu⟩
    simp [flushOwn, Entry.isBuilt, hst] at h'
  cases h1 : e1.st with
  | building ph1 =>
    cases h2 : ce.st with
    | building ph2 =>
      simp only [rebase, h1, h2, D, ite_true, checkIf]
      obtain ⟨f1, _, f3, f4, _⟩ := checkB_fields B (c + 1) (relinkE (own (c + 1) e1.log) ce)
      have hext : Ext B c ce (checkB B (c + 1) (relinkE (own (c + 1) e1.log) ce)) :=
        ext_checkB_top B c ce _ hos _ rfl (by
          simp only [relinkE]; rw [sumPairs_append, show ce.pairs = sumPairs ce.log from hcur.cnt]) rfl rfl rfl rfl rfl
      refine Or.inr ⟨by rw [f3]; rfl, by rw [f4]; rfl, by rw [isBuilt_of_st f1]; rfl, fun _ => hext, ?_⟩
      intro u hu; rw [upkeep_of_st f1] at hu; simp [relinkE, Entry.upkeep, h2] at hu
    | built up =>
      simp only [rebase, h1, h2]
      split
      · exact flush_t up h2
      · exact restart_t
  | built up1 =>
    have hb1 : e1.isBuilt = true := isBuilt_built e1 up1 h1
    cases h2 : ce.st with
    | building ph2 => have := hbb hb1; simp [Entry.isBuilt, h2] at this
    | built up =>
      simp only [rebase, h1, h2]
      split
      · refine Or.inr ⟨by simp [refoldE, hg], by simp [refoldE], by simp [refoldE, Entry.isBuilt, h2],
          fun h' => by simp [refoldE, Entry.isBuilt] at h', fun u hu => by simp [refoldE, Entry.upkeep] at hu⟩
      · split
        · exact flush_t up h2
        · exact restart_t

theorem track_retain {B c n0} {ce e : Entry} (ht : Track B c n0 ce e) : Track B c n0 ce (retain e) := by
  cases hst : e.st with
  | building ph => simpa [retain, hst] using ht
  | built up =>
    have hs : (retain e).st = e.st := by cases up <;> simp [retain, hst]
    have hg : (retain e).gen = e.gen := by cases up <;> simp [retain, hst]
    have hc : (retain e).commits = e.commits := by cases up <;> simp [retain, hst]
    rcases ht with ⟨h1, h2⟩ | ⟨h1, h2, h3, h4, h5⟩
    · exact Or.inl ⟨by rw [hg]; exact h1, by rw [isBuilt_of_st hs]; exact h2⟩
    · refine Or.inr ⟨by rw [hg]; exact h1, by rw [hc]; exact h2, by rw [isBuilt_of_st hs]; exact h3, ?_, ?_⟩
      · intro h'; rw [isBuilt_of_st hs] at h'; simp [Entry.isBuilt, hst] at h'
      · intro u hu; rw [upkeep_of_st hs] at hu; exact h5 u hu

theorem segPairs_append (a b : List (List Pair × Out)) : segPairs (a ++ b) = segPairs a ++ segPairs b := by
  induction a with
  | nil => rfl
  | cons x rest ih => obtain ⟨ps, o⟩ := x; simp [segPairs, ih, List.append_assoc]

/-- What a writer's fork may assume of the published entry (it is not stale: same data version). -/
theorem fork_facts {B : Nat} {σ : Sys} (hI : Inv B σ) {fork : Pub} (hf : fork ∈ σ.cur :: σ.hist)
    (hv : fork.ver = σ.cur.ver) :
    fork.e.gen = σ.cur.e.gen ∧ (fork.e.commits = σ.cur.e.commits → fork.e = σ.cur.e) ∧
    (σ.cur.batches = fork.batches → fork.e = σ.cur.e) ∧ (fork.e.isBuilt = true → σ.cur.e.isBuilt = true) := by
  rcases List.mem_cons.1 hf with rfl | hh
  · exact ⟨rfl, fun _ => rfl, fun _ => rfl, id⟩
  · have hx := hI.rel fork hh
    obtain ⟨hg, he⟩ := hx.same_ver hv
    refine ⟨hg, he, ?_, fun hb => (hx.same_gen hg).2.1 hb⟩
    intro hb
    rcases hx.ver_le with h1 | ⟨_, h1⟩ <;> omega

/-- **The writer's published entry**: `PInv` at its version, fresh ids, and `Track` against the published entry. -/
theorem writer_core {B : Nat} {σ : Sys} (hI : Inv B σ) {fork : Pub} (hf : fork ∈ σ.cur :: σ.hist)
    (hv : fork.ver = σ.cur.ver) (pre post : List (List Pair × Out)) (b : Bool) :
    PInv B (σ.writes ++ (segPairs (pre ++ post)).map (fun p => (σ.cur.ver + 1, p)))
        ⟨σ.cur.ver + 1, σ.cur.batches, (writerE D B σ.cur fork pre post b σ.next).1⟩ ∧
      Fresh (writerE D B σ.cur fork pre post b σ.next).2 (writerE D B σ.cur fork pre post b σ.next).1 ∧
      σ.next ≤ (writerE D B σ.cur fork pre post b σ.next).2 ∧
      Track B σ.cur.ver σ.next σ.cur.e (writerE D B σ.cur fork pre post b σ.next).1 := by
  obtain ⟨hPf, hFf⟩ := hI.pub fork hf
  obtain ⟨hPc, hFc⟩ := hI.pub σ.cur (List.mem_cons_self ..)
  obtain ⟨fg, fc, fb, fbb⟩ := fork_facts hI hf hv
  have hPc' : PInv B σ.writes ⟨σ.cur.ver, σ.cur.batches, σ.cur.e⟩ := hPc
  have p0 : Priv B σ.cur.ver σ.writes [] σ.next fork.e σ.next := priv_init hPf hv hFf
  have p1 := priv_segs (B := B) pre p0
  have t1 := track_segs (B := B) pre (track_refl B σ.cur.ver σ.next fork.e) p0
  -- the writer's copy after the gate (rebased, or its own), with `Priv` and `Track` against `cur`
  have step2 : ∀ {r2 : Entry × Nat}, r2 = (if NI.overtakenAny (σ.cur.batches != fork.batches)
        [overtaken D (segs D B (σ.cur.ver + 1) pre fork.e σ.next).1 σ.cur.e]
        then rebase D B (σ.cur.ver + 1) b (segs D B (σ.cur.ver + 1) pre fork.e σ.next).1 σ.cur.e
          (segs D B (σ.cur.ver + 1) pre fork.e σ.next).2
        else segs D B (σ.cur.ver + 1) pre fork.e σ.next) →
      Priv B σ.cur.ver σ.writes ([] ++ segPairs pre) σ.next r2.1 r2.2 ∧
        Track B σ.cur.ver σ.next σ.cur.e r2.1 := by
    intro r2 hr2
    split at hr2
    · rename_i hgate
      subst hr2
      have ho : overtaken D (segs D B (σ.cur.ver + 1) pre fork.e σ.next).1 σ.cur.e = true := by
        have := (NI.overtakenAny_iff _ _).1 hgate; simpa using this.2
      have hge : (segs D B (σ.cur.ver + 1) pre fork.e σ.next).1.gen = σ.cur.e.gen := by
        simp [overtaken, D] at ho; exact ho.1
      have hbb : (segs D B (σ.cur.ver + 1) pre fork.e σ.next).1.isBuilt = true → σ.cur.e.isBuilt = true := by
        intro hb
        rcases t1 with ⟨_, hb'⟩ | ⟨_, _, hb', _, _⟩
        · simp [hb] at hb'
        · exact fbb (hb'.symm.trans hb)
      have hng : (segs D B (σ.cur.ver + 1) pre fork.e σ.next).1.gen < σ.next := by rw [hge]; exact hFc.gen
      exact ⟨priv_rebase p1 hng hPc' hFc hbb b, track_rebase p1 hge hPc' hbb b⟩
    · rename_i hgate
      subst hr2
      refine ⟨p1, ?_⟩
      rcases t1 with h | ⟨hg1, hc1, hb1, hx1, hu1⟩
      · exact Or.inl h
      · have hfe : fork.e = σ.cur.e := by
          by_cases hbt : σ.cur.batches = fork.batches
          · exact fb hbt
          · apply fc
            have hno : overtaken D (segs D B (σ.cur.ver + 1) pre fork.e σ.next).1 σ.cur.e = false := by
              cases hov : overtaken D (segs D B (σ.cur.ver + 1) pre fork.e σ.next).1 σ.cur.e
              · rfl
              · exfalso; apply hgate; rw [NI.overtakenAny_iff]; exact ⟨by simpa using hbt, _, by simp, hov⟩
            have hgen : (segs D B (σ.cur.ver + 1) pre fork.e σ.next).1.gen = σ.cur.e.gen := hg1.trans fg
            have hcm : (segs D B (σ.cur.ver + 1) pre fork.e σ.next).1.commits = σ.cur.e.commits := by
              by_cases hne : (segs D B (σ.cur.ver + 1) pre fork.e σ.next).1.commits = σ.cur.e.commits
              · exact hne
              exfalso
              have : overtaken D (segs D B (σ.cur.ver + 1) pre fork.e σ.next).1 σ.cur.e = true := by
                unfold overtaken; simp only [show D.d6 = true from rfl, ite_true]; simp [hgen, hne]
              rw [this] at hno; exact absurd hno (by decide)
            rw [← hc1]; exact hcm
        rw [← hfe]; exact Or.inr ⟨hg1, hc1, hb1, hx1, hu1⟩
  obtain ⟨p2, t2⟩ := step2 rfl
  have p3 := priv_segs (B := B) post p2
  have t3 := track_segs (B := B) post t2 p2
  obtain ⟨hP, hF⟩ := priv_retain (bt := σ.cur.batches) p3 hI.wver
  refine ⟨?_, hF, Nat.le_trans p3.fresh.1 (Nat.le_refl _), track_retain t3⟩
  simpa [writerE, segPairs_append, List.append_assoc] using hP

theorem Fresh.mono {nx nx' : Nat} {e : Entry} (h : Fresh nx e) (hle : nx ≤ nx') : Fresh nx' e :=
  ⟨Nat.lt_of_lt_of_le h.gen hle, Nat.lt_of_lt_of_le h.lineage hle, fun u hu => Nat.lt_of_lt_of_le (h.ticket u hu) hle⟩

end CchReview
