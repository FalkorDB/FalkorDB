import CchRedesignReview.Inv
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false

/-!
# A writer's private copy (03 §5 "The log", §6 flush and hand-off, §5 rebase table, §7 retention)

`Priv B c w O n0 e nx`: the writer committing version `c + 1` (its fork's data version is `c`), with `w` the
writes committed before it and `O` its own pairs so far, holds the private copy `e`, with `n0` the counter it
started from and `nx` the next fresh id. Its parts: nothing in `e` is from the future; its own nodes are the top
of the log; every older write is resolved or logged, as is every own write; while `e` keeps the fork's
generation, every own pair is in an own node (or under an own Overflow), whatever the entry state (03 §5: "A
writer always keeps its own nodes in its private version until it publishes, whatever the entry state").
-/
namespace CchReview

structure PrivNB (B c : Nat) (w : List (Nat × Pair)) (O : List Pair) (n0 : Nat) (e : Entry) (nx : Nat) : Prop where
  res_le : ∀ p, e.res p ≤ c + 1
  cnt : e.pairs = sumPairs e.log
  shape : ∃ M Rr, e.log = M ++ Rr ∧ (∀ n ∈ M, n.seq = c + 1) ∧ (∀ n ∈ Rr, n.seq ≤ c)
  sound : ∀ n ∈ e.log, ∀ p ∈ n.ps, (n.seq, p) ∈ w ∨ (n.seq = c + 1 ∧ p ∈ O)
  old : ∀ s p, (s, p) ∈ w → s ≤ c → s ≤ e.res p ∨ covered e.log s p
  ownN : ∀ p ∈ O, c + 1 ≤ e.res p ∨ covered e.log (c + 1) p
  ownK : e.gen < n0 → ∀ p ∈ O, covered (own (c + 1) e.log) (c + 1) p
  builtOld : e.isBuilt = true → ∀ s p, (s, p) ∈ w → s ≤ c → s ≤ e.res p
  builtOwn : e.isBuilt = true → ∀ p ∈ O, c + 1 ≤ e.res p
  noOv : e.isBuilt = true → hasOv e.log = false
  up : ∀ u, e.st = .built (some u) → u.since ≤ c + 1 ∧
    ∀ s p, ((s, p) ∈ w ∧ s ≤ c ∨ s = c + 1 ∧ p ∈ O) → u.since < s → ∃ n ∈ e.log, n.seq = s ∧ p ∈ n.ps
  fresh : n0 ≤ nx ∧ Fresh nx e

structure Priv (B c : Nat) (w : List (Nat × Pair)) (O : List Pair) (n0 : Nat) (e : Entry) (nx : Nat) : Prop
    extends PrivNB B c w O n0 e nx where
  bound : e.isBuilt = false → e.pairs ≤ B ∨ hasOv e.log = true

/-! ## Small lemmas -/

theorem mem_takeWhile_true {P : Node → Bool} {l : List Node} {n : Node} (h : n ∈ l.takeWhile P) : P n = true := by
  induction l with
  | nil => simp at h
  | cons m l ih =>
    simp only [List.takeWhile] at h
    cases hm : P m <;> simp [hm] at h
    rcases h with rfl | h
    · exact hm
    · exact ih h

theorem own_seq {v : Nat} {l : List Node} {n : Node} (h : n ∈ own v l) : n.seq = v := by
  have := mem_takeWhile_true h; simpa using this

theorem own_sub {v : Nat} {l : List Node} {n : Node} (h : n ∈ own v l) : n ∈ l := mem_of_mem_takeWhile h

theorem own_cons_self (v : Nat) (k : NKind) (l : List Node) : own v (⟨v, k⟩ :: l) = ⟨v, k⟩ :: own v l := by
  simp [own, List.takeWhile]

theorem own_own (v : Nat) (l : List Node) : own v (own v l) = own v l :=
  own_all v _ (fun n hn => own_seq hn)

theorem hasOv_sub {a b : List Node} (h : ∀ n ∈ a, n ∈ b) (hb : hasOv b = false) : hasOv a = false :=
  (hasOv_false_iff a).2 (fun n hn => (hasOv_false_iff b).1 hb n (h n hn))

theorem resolve_ge {r : Pair → Nat} {ps : List Pair} {v : Nat} (p : Pair) (h : r p ≤ v) :
    r p ≤ resolve r ps v p := by
  unfold resolve; split <;> omega

theorem resolve_le {r : Pair → Nat} {ps : List Pair} {v : Nat} (p : Pair) (h : r p ≤ v) :
    resolve r ps v p ≤ v := by
  unfold resolve; split <;> omega

theorem resolve_mem {r : Pair → Nat} {ps : List Pair} {v : Nat} {p : Pair} (h : p ∈ ps) : resolve r ps v p = v := by
  simp [resolve, h]

/-- A pair under an own node of a log with no Overflow is among the own nodes' pairs. -/
theorem allPairs_of_covered {o : List Node} {v p} (h : covered o v p) (hov : hasOv o = false) : p ∈ allPairs o := by
  obtain ⟨n, hn, _, hp⟩ := covered_noOv h hov
  exact (mem_allPairs o p).2 ⟨n, hn, hp⟩

/-! ## The bound check -/

/-- A writer's bound check keeps `Priv` (03 §5 "Bound and Overflow"): either nothing changes and the counter is
    under the bound, or the log becomes `[Overflow{c+1}]`, which covers every write so far. -/
theorem priv_checkB {B c w O n0 nx} {e : Entry} {ph : Phase} (hst : e.st = .building ph)
    (h : PrivNB B c w O n0 e nx) : Priv B c w O n0 (checkB B (c + 1) e) nx := by
  have hb : e.isBuilt = false := isBuilt_building e ph hst
  unfold checkB
  split
  · refine ⟨⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, ?_⟩
    · exact h.res_le
    · simp
    · exact ⟨[⟨c + 1, .overflow⟩], [], by simp, by simp, by simp⟩
    · intro n hn p hp; simp at hn; subst hn; simp at hp
    · intro s p _ hs; exact Or.inr (covered_ov p (List.mem_singleton_self _) (by omega))
    · intro p _; exact Or.inr (covered_ov p (List.mem_singleton_self _) (Nat.le_refl _))
    · intro _ p _
      rw [own_all (c + 1) _ (by simp)]
      exact covered_ov p (List.mem_singleton_self _) (Nat.le_refl _)
    · intro h'; simp [Entry.isBuilt, hst] at h'
    · intro h'; simp [Entry.isBuilt, hst] at h'
    · intro h'; simp [Entry.isBuilt, hst] at h'
    · intro u hu; simp [hst] at hu
    · exact ⟨h.fresh.1, h.fresh.2.gen, h.fresh.2.lineage, by simp [Entry.upkeep, hst]⟩
    · intro _; right; simp
  · rename_i hle
    exact ⟨h, fun _ => Or.inl (by omega)⟩

theorem priv_checkIf {B c w O n0 nx} {e : Entry} {ph : Phase} (hst : e.st = .building ph)
    (h : PrivNB B c w O n0 e nx) : Priv B c w O n0 (checkIf D.checkAll B (c + 1) e) nx := by
  simpa [checkIf, D] using priv_checkB hst h

/-- The writer's fork, as its private copy before any segment. -/
theorem priv_init {B c : Nat} {w : List (Nat × Pair)} {S : Pub} {n0 : Nat} (hS : PInv B w S) (hv : S.ver = c)
    (hf : Fresh n0 S.e) : Priv B c w [] n0 S.e n0 := by
  subst hv
  refine ⟨⟨?_, hS.cnt, ⟨[], S.e.log, by simp, by simp, hS.seq_le⟩, ?_, hS.nlw, by simp, by simp, hS.built,
    by simp, hS.noOv, ?_, ⟨Nat.le_refl _, hf⟩⟩, hS.bound⟩
  · intro p; have := hS.res_le p; omega
  · intro n hn p hp; exact Or.inl (hS.sound n hn p hp)
  · intro u hu
    obtain ⟨a, _, c⟩ := hS.upSome u hu
    refine ⟨by omega, ?_⟩
    rintro s p (⟨hsp, hs⟩ | ⟨_, hp⟩) h1
    · exact c s p hsp h1 hs
    · simp at hp

/-! ## One segment -/

/-- A Building copy logs the segment (03 §5), then checks the bound. -/
theorem priv_seg_building {B c w O n0 nx} {e : Entry} {ph : Phase} (hst : e.st = .building ph)
    (h : Priv B c w O n0 e nx) (ps : List Pair) :
    Priv B c w (O ++ ps) n0 (checkB B (c + 1) (appendE (c + 1) ps e)) nx := by
  refine priv_checkB (ph := ph) (e := appendE (c + 1) ps e) hst ?_
  simp only [appendE]
  obtain ⟨M, Rr, hl, hM, hR⟩ := h.shape
  have hb : e.isBuilt = false := isBuilt_building e ph hst
  refine ⟨h.res_le, ?_, ⟨⟨c + 1, .pairs ps⟩ :: M, Rr, by simp [hl], ?_, hR⟩, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    ⟨h.fresh.1, h.fresh.2.gen, h.fresh.2.lineage, by simp [Entry.upkeep, hst]⟩⟩
  · simp [h.cnt]; omega
  · intro n hn; simp at hn; rcases hn with rfl | hn
    · rfl
    · exact hM n hn
  · intro n hn p hp; simp at hn; rcases hn with rfl | hn
    · right; simp at hp; exact ⟨rfl, List.mem_append_right O hp⟩
    · rcases h.sound n hn p hp with h1 | ⟨h1, h2⟩
      · exact Or.inl h1
      · exact Or.inr ⟨h1, List.mem_append_left ps h2⟩
  · intro s p hsp hs
    rcases h.old s p hsp hs with h1 | h1
    · exact Or.inl h1
    · exact Or.inr (covered_cons _ h1)
  · intro p hp; rcases List.mem_append.1 hp with hp | hp
    · rcases h.ownN p hp with h1 | h1
      · exact Or.inl h1
      · exact Or.inr (covered_cons _ h1)
    · exact Or.inr ⟨_, List.mem_cons_self .., Or.inl ⟨rfl, by simpa using hp⟩⟩
  · intro hg p hp
    simp only [own_cons_self]
    rcases List.mem_append.1 hp with hp | hp
    · exact covered_cons _ (h.ownK hg p hp)
    · exact ⟨_, List.mem_cons_self .., Or.inl ⟨rfl, by simpa using hp⟩⟩
  · intro h'; simp [Entry.isBuilt, hst] at h'
  · intro h'; simp [Entry.isBuilt, hst] at h'
  · intro h'; simp [Entry.isBuilt, hst] at h'
  · intro u hu; simp [hst] at hu

/-- A Built copy flushes the segment (03 §6): fits, or crosses a fold/S6 trigger. -/
theorem priv_flushB {B c w O n0 nx nx'} {e : Entry} {up0 : Option Upkeep} (hst : e.st = .built up0)
    (h : Priv B c w O n0 e nx) (ps : List Pair) (up : Option Upkeep) (hnx : nx ≤ nx')
    (hup : ∀ u, up = some u → u.ticket < nx' ∧ u.since ≤ c + 1 ∧
      ∀ s p, ((s, p) ∈ w ∧ s ≤ c ∨ s = c + 1 ∧ p ∈ O) → u.since < s → ∃ n ∈ e.log, n.seq = s ∧ p ∈ n.ps) :
    Priv B c w (O ++ ps) n0 (flushB D B (c + 1) ps up e) nx' := by
  have hbt : e.isBuilt = true := isBuilt_built e up0 hst
  obtain ⟨M, Rr, hl, hM, hR⟩ := h.shape
  have hres : ∀ p, c + 1 ≤ resolve e.res ps (c + 1) p ∨ True := fun _ => Or.inr trivial
  -- common facts of the flushed copy
  have rle : ∀ p, resolve e.res ps (c + 1) p ≤ c + 1 := fun p => resolve_le p (h.res_le p)
  have rold : ∀ s p, (s, p) ∈ w → s ≤ c → s ≤ resolve e.res ps (c + 1) p :=
    fun s p hsp hs => Nat.le_trans (h.builtOld hbt s p hsp hs) (resolve_ge p (h.res_le p))
  have rown : ∀ p ∈ O ++ ps, c + 1 ≤ resolve e.res ps (c + 1) p := by
    intro p hp; rcases List.mem_append.1 hp with hp | hp
    · exact Nat.le_trans (h.builtOwn hbt p hp) (resolve_ge p (h.res_le p))
    · rw [resolve_mem hp]; exact Nat.le_refl _
  have nov : hasOv (⟨c + 1, .pairs ps⟩ :: e.log) = false := by simp [h.noOv hbt]
  have snd : ∀ n ∈ (⟨c + 1, .pairs ps⟩ :: e.log : List Node), ∀ p ∈ n.ps,
      (n.seq, p) ∈ w ∨ (n.seq = c + 1 ∧ p ∈ O ++ ps) := by
    intro n hn p hp; simp at hn; rcases hn with rfl | hn
    · right; simp at hp; exact ⟨rfl, List.mem_append_right O hp⟩
    · rcases h.sound n hn p hp with h1 | ⟨h1, h2⟩
      · exact Or.inl h1
      · exact Or.inr ⟨h1, List.mem_append_left ps h2⟩
  have okd : ∀ p ∈ O ++ ps, covered (own (c + 1) (⟨c + 1, .pairs ps⟩ :: e.log)) (c + 1) p ∨ e.gen ≥ n0 := by
    intro p hp
    by_cases hg : e.gen < n0
    · left; simp only [own_cons_self]
      rcases List.mem_append.1 hp with hp | hp
      · exact covered_cons _ (h.ownK hg p hp)
      · exact ⟨_, List.mem_cons_self .., Or.inl ⟨rfl, by simpa using hp⟩⟩
    · right; omega
  have shp : (⟨c + 1, .pairs ps⟩ :: e.log : List Node) = (⟨c + 1, .pairs ps⟩ :: M) ++ Rr := by simp [hl]
  have hM' : ∀ n ∈ (⟨c + 1, .pairs ps⟩ :: M : List Node), n.seq = c + 1 := by
    intro n hn; simp at hn; rcases hn with rfl | hn
    · rfl
    · exact hM n hn
  have hRlt : ∀ n ∈ Rr, n.seq < c + 1 := fun n hn => Nat.lt_succ_of_le (hR n hn)
  have hown : own (c + 1) (⟨c + 1, .pairs ps⟩ :: e.log) = ⟨c + 1, .pairs ps⟩ :: M := by
    rw [shp]; exact own_seam _ _ _ hM' hRlt
  -- the three shapes of the result
  unfold flushB
  cases up with
  | none =>
    refine ⟨⟨rle, ?_, ⟨_, Rr, shp, hM', hR⟩, snd, fun s p hsp hs => Or.inl (rold s p hsp hs),
      fun p hp => Or.inl (rown p hp), ?_, fun _ => rold, fun _ => rown, fun _ => nov, ?_, ?_⟩, ?_⟩
    · simp [h.cnt]; omega
    · intro hg p hp
      rcases okd p hp with h1 | h1
      · exact h1
      · simp at hg; omega
    · intro u hu; simp at hu
    · exact ⟨Nat.le_trans h.fresh.1 hnx, Nat.lt_of_lt_of_le h.fresh.2.gen hnx,
        Nat.lt_of_lt_of_le h.fresh.2.lineage hnx, by simp [Entry.upkeep]⟩
    · intro h'; simp [Entry.isBuilt] at h'
  | some u =>
    obtain ⟨ht, hs1, hcov⟩ := hup u rfl
    dsimp only
    split
    · -- past the bound: upkeep and the retained nodes cleared, own nodes kept (reading R2)
      simp only [D, if_true]
      rw [hown]
      refine ⟨⟨rle, rfl, ⟨_, [], by simp, hM', by simp⟩, ?_, fun s p hsp hs => Or.inl (rold s p hsp hs),
        fun p hp => Or.inl (rown p hp), ?_, fun _ => rold, fun _ => rown, ?_, ?_, ?_⟩, ?_⟩
      · intro n hn p hp
        exact snd n (by rw [shp]; exact List.mem_append_left _ hn) p hp
      · intro hg p hp
        rw [own_all _ _ hM']
        rcases okd p hp with h1 | h1
        · rwa [hown] at h1
        · simp at hg; omega
      · intro _; exact hasOv_sub (fun n hn => by rw [shp]; exact List.mem_append_left _ hn) nov
      · intro u' hu'; simp at hu'
      · exact ⟨Nat.le_trans h.fresh.1 hnx, Nat.lt_of_lt_of_le h.fresh.2.gen hnx,
          Nat.lt_of_lt_of_le h.fresh.2.lineage hnx, by simp [Entry.upkeep]⟩
      · intro h'; simp [Entry.isBuilt] at h'
    · refine ⟨⟨rle, ?_, ⟨_, Rr, shp, hM', hR⟩, snd, fun s p hsp hs => Or.inl (rold s p hsp hs),
        fun p hp => Or.inl (rown p hp), ?_, fun _ => rold, fun _ => rown, fun _ => nov, ?_, ?_⟩, ?_⟩
      · simp [h.cnt]; omega
      · intro hg p hp
        rcases okd p hp with h1 | h1
        · exact h1
        · simp at hg; omega
      · intro u' hu'
        simp at hu'; subst hu'
        refine ⟨hs1, ?_⟩
        rintro s p (⟨hsp, hs⟩ | ⟨rfl, hp⟩) h1
        · obtain ⟨n, hn, hh⟩ := hcov s p (Or.inl ⟨hsp, hs⟩) h1
          exact ⟨n, List.mem_cons_of_mem _ hn, hh⟩
        · rcases List.mem_append.1 hp with hp | hp
          · obtain ⟨n, hn, hh⟩ := hcov _ p (Or.inr ⟨rfl, hp⟩) h1
            exact ⟨n, List.mem_cons_of_mem _ hn, hh⟩
          · exact ⟨_, List.mem_cons_self .., rfl, by simpa using hp⟩
      · exact ⟨Nat.le_trans h.fresh.1 hnx, Nat.lt_of_lt_of_le h.fresh.2.gen hnx,
          Nat.lt_of_lt_of_le h.fresh.2.lineage hnx, by intro u' hu'; simp [Entry.upkeep] at hu'; subst hu'; exact ht⟩
      · intro h'; simp [Entry.isBuilt] at h'

/-- 03 §6 Hand-off: `Building{Repair, base = the private hierarchy at the abort, log: [this segment]}` under a new
    generation. The earlier segments of the query are in the base. -/
theorem priv_handoff {B c w O n0 nx} {e : Entry} {up0 : Option Upkeep} (hst : e.st = .built up0)
    (h : Priv B c w O n0 e nx) (ps : List Pair) :
    Priv B c w (O ++ ps) n0 (checkIf D.checkAll B (c + 1) (handoffE (c + 1) ps e nx)) (nx + 1) := by
  have hbt : e.isBuilt = true := isBuilt_built e up0 hst
  refine priv_checkIf (ph := .repair) rfl ?_
  simp only [handoffE]
  refine ⟨h.res_le, by simp, ⟨[⟨c + 1, .pairs ps⟩], [], by simp, by simp, by simp⟩,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro n hn p hp; simp at hn; subst hn; right; simp at hp; exact ⟨rfl, List.mem_append_right O hp⟩
  · intro s p hsp hs; exact Or.inl (h.builtOld hbt s p hsp hs)
  · intro p hp; rcases List.mem_append.1 hp with hp | hp
    · exact Or.inl (h.builtOwn hbt p hp)
    · exact Or.inr ⟨_, List.mem_singleton_self _, Or.inl ⟨rfl, by simpa using hp⟩⟩
  · intro hg; have := h.fresh.1; simp at hg; omega
  · intro h'; simp [Entry.isBuilt] at h'
  · intro h'; simp [Entry.isBuilt] at h'
  · intro h'; simp [Entry.isBuilt] at h'
  · intro u hu; simp at hu
  · exact ⟨by have := h.fresh.1; omega, by simp, by have := h.fresh.2.lineage; simp; omega, by simp [Entry.upkeep]⟩

/-- 03 §6 Decay, S7: a new generation through the batched build, this segment as its log; the skeleton is read at
    or after this version, so the build reflects every write up to it. -/
theorem priv_s7 {B c w O n0 nx} {e : Entry} {up0 : Option Upkeep} (hst : e.st = .built up0)
    (h : Priv B c w O n0 e nx) (ps : List Pair) :
    Priv B c w (O ++ ps) n0 (checkIf D.checkAll B (c + 1) (s7E (c + 1) ps nx)) (nx + 1) := by
  refine priv_checkIf (ph := .readSkeleton) rfl ?_
  simp only [s7E]
  refine ⟨fun _ => Nat.le_refl _, by simp,
    ⟨[⟨c + 1, .pairs ps⟩], [], by simp, by simp, by simp⟩, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro n hn p hp; simp at hn; subst hn; right; simp at hp; exact ⟨rfl, List.mem_append_right O hp⟩
  · intro s p _ hs; left; show s ≤ c + 1; omega
  · intro p _; left; exact Nat.le_refl _
  · intro hg; have := h.fresh.1; simp at hg; omega
  · intro h'; simp [Entry.isBuilt] at h'
  · intro h'; simp [Entry.isBuilt] at h'
  · intro h'; simp [Entry.isBuilt] at h'
  · intro u hu; simp at hu
  · exact ⟨by have := h.fresh.1; omega, by simp, by simp, by simp [Entry.upkeep]⟩

/-- **One segment keeps `Priv`**, whatever the entry state and the budget outcome. -/
theorem priv_seg {B c w O n0 nx} {e : Entry} (h : Priv B c w O n0 e nx) (ps : List Pair) (o : Out) :
    Priv B c w (O ++ ps) n0 (seg D B (c + 1) ps o e nx).1 (seg D B (c + 1) ps o e nx).2 := by
  cases hst : e.st with
  | building ph => simp only [seg, hst]; exact priv_seg_building hst h ps
  | built up =>
    cases o with
    | fits =>
      simp only [seg, hst]
      refine priv_flushB hst h ps up (Nat.le_refl _) ?_
      intro u hu; subst hu
      obtain ⟨a, b⟩ := h.up u hst
      exact ⟨h.fresh.2.ticket u (by simp [Entry.upkeep, hst]), a, b⟩
    | setUp k =>
      simp only [seg, hst]
      refine priv_flushB hst h ps _ (Nat.le_succ _) ?_
      intro u hu; simp at hu; subst hu
      refine ⟨Nat.lt_succ_self _, Nat.le_refl _, ?_⟩
      rintro s p (⟨_, hs⟩ | ⟨rfl, _⟩) h1 <;> simp at h1 <;> omega
    | abort => simp only [seg, hst]; exact priv_handoff hst h ps
    | s7 => simp only [seg, hst]; exact priv_s7 hst h ps

theorem priv_segs {B c w n0} (l : List (List Pair × Out)) :
    ∀ {O nx} {e : Entry}, Priv B c w O n0 e nx →
      Priv B c w (O ++ segPairs l) n0 (segs D B (c + 1) l e nx).1 (segs D B (c + 1) l e nx).2 := by
  induction l with
  | nil => intro O nx e h; simpa [segs, segPairs] using h
  | cons x rest ih =>
    intro O nx e h
    obtain ⟨ps, o⟩ := x
    have := ih (priv_seg h ps o)
    simpa [segs, segPairs, List.append_assoc] using this

end CchReview
