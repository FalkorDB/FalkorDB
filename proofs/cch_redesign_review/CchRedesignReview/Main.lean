import CchRedesignReview.Step
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false

/-!
# The design's claims, for every reachable state (every interleaving, every bound `B`)

`σ.cur :: σ.hist` is every version published so far; a writer or job may hold any of them as its fork, so each
claim is stated for all of them.
-/
namespace CchReview

/-! ## 1. No lost write (03 §5, §6, §7; Appendix B "Does the write log hold now") -/

/-- **No lost write.** A pair a writer logged at seq `s` is, in every published version `S` at or after `s`,
    resolved in `S`'s structure at a version `≥ s`, or still in `S`'s log (its own node, or under an Overflow at
    or after `s`). Building and Built alike. -/
theorem no_lost_write {B : Nat} {σ : Sys} (h : Reach D B σ) {S : Pub} (hS : S ∈ σ.cur :: σ.hist)
    (s : Nat) (p : Pair) (hw : (s, p) ∈ σ.writes) (hs : s ≤ S.ver) :
    s ≤ S.e.res p ∨ covered S.e.log s p :=
  ((reach_inv h).pub S hS).1.nlw s p hw hs

/-- A write's pair is resolved only by a version that read the graph at or after it: no structure reads the
    future (`res ≤ ver`), and no log holds a node from the future (`seq ≤ ver`). -/
theorem nothing_from_the_future {B : Nat} {σ : Sys} (h : Reach D B σ) {S : Pub} (hS : S ∈ σ.cur :: σ.hist) :
    (∀ p, S.e.res p ≤ S.ver) ∧ ∀ n ∈ S.e.log, n.seq ≤ S.ver :=
  ⟨((reach_inv h).pub S hS).1.res_le, ((reach_inv h).pub S hS).1.seq_le⟩

/-! ## 2. A Building entry never serves a hierarchy (03 §5 "Readers while Building", §6b MVCC) -/

theorem building_serves_fallback (S : Pub) : serve S = none ↔ S.e.isBuilt = false := by
  unfold serve Entry.isBuilt; cases S.e.st <;> simp

/-- **What a reader is served is current**: every write committed by its version is resolved in it. -/
theorem served_is_current {B : Nat} {σ : Sys} (h : Reach D B σ) {S : Pub} (hS : S ∈ σ.cur :: σ.hist)
    {r : Pair → Nat} (hr : serve S = some r) (s : Nat) (p : Pair) (hw : (s, p) ∈ σ.writes) (hs : s ≤ S.ver) :
    s ≤ r p := by
  have hb : S.e.isBuilt = true := by
    cases hst : S.e.st <;> simp [serve, hst] at hr ⊢; simp [Entry.isBuilt, hst]
  have hr' : r = S.e.res := by cases hst : S.e.st <;> simp [serve, hst] at hr; exact hr.symm
  subst hr'; exact ((reach_inv h).pub S hS).1.built hb s p hw hs

/-! ## 3. Built retention (03 §6 "Retention", §7 "Retention") -/

theorem built_no_upkeep_no_log {B : Nat} {σ : Sys} (h : Reach D B σ) {S : Pub} (hS : S ∈ σ.cur :: σ.hist)
    (hst : S.e.st = .built none) : S.e.log = [] :=
  ((reach_inv h).pub S hS).1.upNone hst

/-- **With `upkeep = Some{since}` the log holds exactly the writes after `since`**: every node is after `since`
    and is a committed writer's node, every write after `since` is in a node of its seq, and there is no
    Overflow (03 §5: "An Overflow is never created on a Built entry"). -/
theorem built_upkeep_log_exact {B : Nat} {σ : Sys} (h : Reach D B σ) {S : Pub} (hS : S ∈ σ.cur :: σ.hist)
    {u : Upkeep} (hst : S.e.st = .built (some u)) :
    u.since ≤ S.ver ∧ (∀ n ∈ S.e.log, u.since < n.seq ∧ ∀ p ∈ n.ps, (n.seq, p) ∈ σ.writes) ∧
    (∀ s p, (s, p) ∈ σ.writes → u.since < s → s ≤ S.ver → ∃ n ∈ S.e.log, n.seq = s ∧ p ∈ n.ps) ∧
    hasOv S.e.log = false := by
  have hP := ((reach_inv h).pub S hS).1
  obtain ⟨a, b, c⟩ := hP.upSome u hst
  exact ⟨a, fun n hn => ⟨b n hn, hP.sound n hn⟩, c, hP.noOv (isBuilt_built _ _ hst)⟩

/-! ## 4. A stale job never publishes (03 §6 "Publish under the lock", §7 Framework changes, D6) -/

/-- **A job whose generation or ticket does not match the published entry is `Gone`**: nothing is published,
    and the system is unchanged. -/
theorem stale_job_gone (b : Bool) (S0 : Pub) (σ : Sys)
    (hstale : S0.e.gen ≠ σ.cur.e.gen ∨ ∀ u0 u, S0.e.upkeep = some u0 → σ.cur.e.upkeep = some u → u0.ticket ≠ u.ticket) :
    jobResult b S0 σ.cur σ.next = none ∧ jobSys b S0 σ = σ := by
  have hn : jobResult b S0 σ.cur σ.next = none := by
    unfold jobResult
    split
    · rename_i u0 u h1 h2
      have hnf : ¬ (S0.e.gen = σ.cur.e.gen ∧ u0.ticket = u.ticket) := by
        rintro ⟨hg, ht⟩
        rcases hstale with hs | hs
        · exact hs hg
        · exact hs u0 u (by simp [Entry.upkeep, h1]) (by simp [Entry.upkeep, h2]) ht
      simp [hnf]
    · rfl
  exact ⟨hn, by simp [jobSys, hn]⟩

/-- A job that publishes was fitted: its fork's generation and ticket are the published entry's. -/
theorem job_publish_fitted {b : Bool} {S0 cur : Pub} {nx : Nat} {r : Entry × Nat}
    (h : jobResult b S0 cur nx = some r) :
    S0.e.gen = cur.e.gen ∧ ∃ u0 u, S0.e.st = .built (some u0) ∧ cur.e.st = .built (some u) ∧ u0.ticket = u.ticket := by
  unfold jobResult at h
  split at h
  · rename_i u0 u h1 h2
    split at h
    · rename_i hf; exact ⟨hf.1, u0, u, h1, h2, hf.2⟩
    · simp at h
  · simp at h

/-- **A fitted job saw only writers since its fork** (03 §6 "Publish under the lock", §7 D6). When a fold or S6
    job forked at `S0` publishes onto the current version, the fork's entry and the published one have the same
    generation, the same `indexer_commits` and the same `upkeep`. The generation never falls (`step_gen_mono`),
    every batch or job publish of this index moves `indexer_commits` by one (`builder_publish_bumps_commits`) and
    the count never falls within a generation (`commits_monotone`), so between the fork and the publish only
    writers (and other indexes' batches) ran. Every version published in that window keeps the job's generation,
    count and `upkeep` (`job_window_frozen`), and a writer that escalates on such a version is never overtaken, so
    none of them rebased (`upkeep_writer_not_overtaken`). -/
theorem fitted_job_only_writers_between {B : Nat} {σ : Sys} (h : Reach D B σ) {S0 : Pub}
    (h0 : S0 ∈ σ.cur :: σ.hist) {b : Bool} {r : Entry × Nat} (he : jobResult b S0 σ.cur σ.next = some r) :
    S0.e.gen = σ.cur.e.gen ∧ S0.e.commits = σ.cur.e.commits ∧ S0.ver ≤ σ.cur.ver ∧
      ∃ u, S0.e.st = .built (some u) ∧ σ.cur.e.st = .built (some u) := by
  obtain ⟨hg, u0, u, h1, h2, ht⟩ := job_publish_fitted he
  rcases List.mem_cons.1 h0 with rfl | hh
  · rw [h1] at h2; simp at h2; subst h2; exact ⟨rfl, rfl, Nat.le_refl _, u0, h1, h1⟩
  · have hx := (reach_inv h).rel S0 hh
    obtain ⟨hu, _, hc⟩ := hx.ticket u0 u h1 h2 ht
    subst hu
    refine ⟨hg, hc, ?_, u0, h1, h2⟩
    rcases hx.ver_le with h' | ⟨h', _⟩ <;> omega

/-- **The generation never falls** from one published version to the next: a writer keeps it or takes a fresh one
    (hand-off, S7, `Restart`), a batch or job keeps it, another index's batch leaves the entry alone. -/
theorem step_gen_mono {B : Nat} {σ σ' : Sys} (h : Reach D B σ) (hs : Step D B σ σ') : σ.cur.e.gen ≤ σ'.cur.e.gen := by
  have hI := reach_inv h
  cases hs with
  | writer fork hf hv pre post b =>
    obtain ⟨_, _, _, hT⟩ := writer_core hI hf hv pre post b
    have hg := (hI.pub σ.cur (List.mem_cons_self ..)).2.gen
    show σ.cur.e.gen ≤ (writerE D B σ.cur fork pre post b σ.next).1.gen
    rcases hT with ⟨h1, _⟩ | ⟨h1, _⟩ <;> omega
  | batch S0 h0 b adv =>
    unfold batchSys
    split
    · rename_i e he
      obtain ⟨_, _, hg, _⟩ := batch_core hI h0 b adv he
      show σ.cur.e.gen ≤ e.gen; omega
    · exact Nat.le_refl _
  | job S0 h0 b =>
    unfold jobSys
    split
    · rename_i e n he
      obtain ⟨_, _, _, hg, _⟩ := job_core hI h0 b he
      show σ.cur.e.gen ≤ e.gen; omega
    · exact Nat.le_refl _
  | other => exact Nat.le_refl _

/-- **A fitted job's window is frozen.** When a fold or S6 job forked at `S0` publishes onto the current version,
    every version published at or after the fork's data version has the job's generation, `indexer_commits` and
    `upkeep`: in the window, only writers that kept the job's `upkeep` ran (`Inv.upw`). -/
theorem job_window_frozen {B : Nat} {σ : Sys} (h : Reach D B σ) {S0 : Pub} (h0 : S0 ∈ σ.cur :: σ.hist)
    {b : Bool} {r : Entry × Nat} (he : jobResult b S0 σ.cur σ.next = some r) :
    ∀ S ∈ σ.hist, S0.ver ≤ S.ver →
      S.e.gen = σ.cur.e.gen ∧ S.e.commits = σ.cur.e.commits ∧ S.e.st = σ.cur.e.st := by
  have hI := reach_inv h
  obtain ⟨_, _, _, u, h1, h2⟩ := fitted_job_only_writers_between h h0 he
  have hsince : u.since ≤ S0.ver := ((hI.pub S0 h0).1.upSome u h1).1
  intro S hS hle
  obtain ⟨a, c, st⟩ := hI.upw u h2 S hS (Nat.le_trans hsince hle)
  exact ⟨a, c, st.trans h2.symm⟩

/-- **A writer that escalates while `upkeep` is published is not overtaken** (03 §5 rebase table; §6 Built-entry
    jobs): every fork it can hold at the current data version holds the published entry, so the gate's `overtaken`
    is false whatever its segments before escalation did, and it does not rebase. With `job_window_frozen`: no
    writer in a fitted job's window rebased. -/
theorem upkeep_writer_not_overtaken {B : Nat} {σ : Sys} (h : Reach D B σ) {u : Upkeep}
    (hu : σ.cur.e.st = .built (some u)) {fork : Pub} (hf : fork ∈ σ.cur :: σ.hist) (hv : fork.ver = σ.cur.ver)
    (pre : List (List Pair × Out)) :
    fork.e = σ.cur.e ∧ overtaken D (segs D B (σ.cur.ver + 1) pre fork.e σ.next).1 σ.cur.e = false := by
  have hI := reach_inv h
  obtain ⟨hPc, hFc⟩ := hI.pub σ.cur (List.mem_cons_self ..)
  have hfe : fork.e = σ.cur.e := by
    rcases List.mem_cons.1 hf with rfl | hh
    · rfl
    · have hs : u.since ≤ fork.ver := by rw [hv]; exact (hPc.upSome u hu).1
      obtain ⟨_, hc, _⟩ := hI.upw u hu fork hh hs
      exact ((hI.rel fork hh).same_ver hv).2 hc
  refine ⟨hfe, ?_⟩
  rw [hfe]
  have p0 : Priv B σ.cur.ver σ.writes [] σ.next σ.cur.e σ.next := priv_init hPc rfl hFc
  have t1 := track_segs (B := B) pre (track_refl B σ.cur.ver σ.next σ.cur.e) p0
  unfold overtaken; simp only [show D.d6 = true from rfl, ite_true]
  rcases t1 with ⟨hg, _⟩ | ⟨_, hc, _⟩
  · have : (segs D B (σ.cur.ver + 1) pre σ.cur.e σ.next).1.gen ≠ σ.cur.e.gen := by
      have := hFc.gen; omega
    simp [this]
  · simp [hc]

/-- Every step that keeps the data version either leaves the index's entry as it was (another index's batch, a
    discarded batch, a `Gone` job) or is a batch or job publish of this index, which keeps the generation and moves
    `indexer_commits` by one. Only a writer moves the data version. -/
theorem builder_publish_bumps_commits {B : Nat} {σ σ' : Sys} (h : Reach D B σ) (hs : Step D B σ σ')
    (hv : σ'.cur.ver = σ.cur.ver) :
    σ'.cur.e = σ.cur.e ∨ (σ'.cur.e.gen = σ.cur.e.gen ∧ σ'.cur.e.commits = σ.cur.e.commits + 1) := by
  have hI := reach_inv h
  cases hs with
  | writer fork hf hv' pre post b => simp [writerSys] at hv
  | batch S0 h0 b adv =>
    unfold batchSys at hv ⊢
    split
    · rename_i e he
      obtain ⟨_, _, hg, hc, _⟩ := batch_core hI h0 b adv he
      exact Or.inr ⟨hg, hc⟩
    · exact Or.inl rfl
  | job S0 h0 b =>
    unfold jobSys at hv ⊢
    split
    · rename_i e n he
      obtain ⟨_, _, _, hg, hc, _⟩ := job_core hI h0 b he
      exact Or.inr ⟨hg, hc⟩
    · exact Or.inl rfl
  | other => exact Or.inl rfl

/-- **A job that publishes in a reachable state has every write resolved** (S6 replays the retained nodes after
    its fork; a fold keeps the published hierarchy), clears `upkeep` and the log, and counts the publish. -/
theorem job_publish_sound {B : Nat} {σ : Sys} (h : Reach D B σ) {S0 : Pub} (h0 : S0 ∈ σ.cur :: σ.hist)
    (b : Bool) {e : Entry} {n : Nat} (he : jobResult b S0 σ.cur σ.next = some (e, n)) :
    (∀ s p, (s, p) ∈ σ.writes → s ≤ σ.cur.ver → s ≤ e.res p) ∧ e.st = .built none ∧ e.log = [] ∧
      e.commits = σ.cur.e.commits + 1 := by
  obtain ⟨hP, _, _, _, hc, hst, _⟩ := job_core (reach_inv h) h0 b he
  exact ⟨hP.built (isBuilt_built _ _ hst), hst, hP.upNone hst, hc⟩

/-! ## 5. The bound (03 §5 "Bound and Overflow") -/

/-- **While Building, the log is under the bound or holds an Overflow.** -/
theorem building_bound {B : Nat} {σ : Sys} (h : Reach D B σ) {S : Pub} (hS : S ∈ σ.cur :: σ.hist)
    (hb : S.e.isBuilt = false) : sumPairs S.e.log ≤ B ∨ ∃ n ∈ S.e.log, n.isOv = true := by
  have hP := ((reach_inv h).pub S hS).1
  rcases hP.bound hb with h1 | h1
  · left; rw [← hP.cnt]; exact h1
  · right; exact (hasOv_iff _).1 h1

/-- The `pairs` counter is exact: it is the number of pairs the log holds (03 §2, §7 Accounting). -/
theorem pairs_counter_exact {B : Nat} {σ : Sys} (h : Reach D B σ) {S : Pub} (hS : S ∈ σ.cur :: σ.hist) :
    S.e.pairs = sumPairs S.e.log :=
  ((reach_inv h).pub S hS).1.cnt

/-- An Overflow at `o` stands for every write up to `o` (03 §5): it covers them in `no_lost_write`. -/
theorem overflow_covers (l : List Node) (o s : Nat) (p : Pair) (h : ⟨o, .overflow⟩ ∈ l) (hs : s ≤ o) :
    covered l s p := covered_ov p h hs

/-- The consumer of an Overflow rescans: every pair is resolved at its fork (03 §5 "The phase that consumes an
    Overflow rescans"). -/
theorem consume_overflow_rescans (F : Nat) (e : Entry) (h : hasOv e.log = true) (p : Pair) :
    (consume F e).res p = F := by simp [consume, h]

/-- **A consumed Overflow does not come back** (03 §5: "its own copy then holds no Overflow, so a later CatchUp
    does not bring it back"): after a consuming batch publishes, every node in the log, Overflow or segment, is
    newer than its fork. -/
theorem consumed_does_not_come_back {B : Nat} {σ : Sys} (h : Reach D B σ) {S0 : Pub} (h0 : S0 ∈ σ.cur :: σ.hist)
    {ph : Phase} (hst : S0.e.st = .building ph) (hc : consumes ph = true) (b adv : Bool) {e : Entry}
    (he : batchResult b adv S0 σ.cur = some e) : ∀ n ∈ e.log, S0.ver < n.seq :=
  (batch_core (reach_inv h) h0 b adv he).2.2.2.2.2.2 ph hst hc

/-! ## The writer-rebase table (03 §5 "Writer rebase, by published state") -/

/-- Row 1 (`Building`, `Building`): the fork's own nodes relinked onto the published log, then the bound. -/
theorem rebase_row1 (B v : Nat) (b : Bool) (e1 ce : Entry) (nx : Nat) (ph1 ph2 : Phase)
    (h1 : e1.st = .building ph1) (h2 : ce.st = .building ph2) :
    rebase D B v b e1 ce nx = (checkB B v { ce with log := own v e1.log ++ ce.log, pairs := sumPairs (own v e1.log) + ce.pairs }, nx) := by
  simp [rebase, h1, h2, D, checkIf, relinkE]

/-- Row 2 (`Building`, `Built`): within budget and no own Overflow, the §6 flush over the own nodes onto the
    published hierarchy; otherwise `Restart` under a new generation (`nx`), base the published hierarchy, log the
    own nodes. -/
theorem rebase_row2 (B v : Nat) (b : Bool) (e1 ce : Entry) (nx : Nat) (ph1 : Phase) (up : Option Upkeep)
    (h1 : e1.st = .building ph1) (h2 : ce.st = .built up) :
    rebase D B v b e1 ce nx =
      if b && !hasOv (own v e1.log) then (flushOwn v (own v e1.log) ce, nx)
      else (checkB B v { st := .building .repair, res := ce.res, lineage := ce.lineage, log := own v e1.log,
                         pairs := sumPairs (own v e1.log), gen := nx, commits := ce.commits }, nx + 1) := by
  simp [rebase, h1, h2, restart, restartE, D, checkIf]

/-- Row 3 (`Built`, `Built`, a job published): the same lineage (a fold) re-applies the fold to the fork's copy;
    a new lineage (S6) replays the own nodes within budget, else `Restart`. -/
theorem rebase_row3 (B v : Nat) (b : Bool) (e1 ce : Entry) (nx : Nat) (up1 up : Option Upkeep)
    (h1 : e1.st = .built up1) (h2 : ce.st = .built up) :
    rebase D B v b e1 ce nx =
      if e1.lineage = ce.lineage then (refoldE (own v e1.log) e1 ce, nx)
      else if b then (flushOwn v (own v e1.log) ce, nx)
      else (restart D B v (own v e1.log) ce nx, nx + 1) := by
  simp [rebase, h1, h2]

/-- **Row 4 is unreachable**: a `Built` published version is followed, within its generation, only by `Built`
    ones; so a writer's `Built` fork never meets a `Building` published entry of its generation. -/
theorem row4_unreachable {B : Nat} {σ : Sys} (h : Reach D B σ) {S0 : Pub} (h0 : S0 ∈ σ.cur :: σ.hist)
    (hb : S0.e.isBuilt = true) (hg : S0.e.gen = σ.cur.e.gen) : σ.cur.e.isBuilt = true := by
  rcases List.mem_cons.1 h0 with rfl | hh
  · exact hb
  · exact ((reach_inv h).rel S0 hh |>.same_gen hg).2.1 hb

/-! ## Other claims of 03 about these rules -/

/-- Only a writer moves the data version, and within one data version the generation is constant
    (03 §5: hand-off, S7 and writer Restart take a new generation; adoption Restart keeps it). -/
theorem gen_const_within_version {B : Nat} {σ : Sys} (h : Reach D B σ) {S0 : Pub} (h0 : S0 ∈ σ.hist)
    (hv : S0.ver = σ.cur.ver) : S0.e.gen = σ.cur.e.gen :=
  (((reach_inv h).rel S0 h0).same_ver hv).1

/-- **No publish is reverted within a generation** (03 §6 D6, Appendix B V1.5/V2.4): `indexer_commits` never
    decreases, so a writer whose fork predates a batch, fold or S6 publish never puts the older copy back. -/
theorem commits_monotone {B : Nat} {σ : Sys} (h : Reach D B σ) {S0 : Pub} (h0 : S0 ∈ σ.hist)
    (hg : S0.e.gen = σ.cur.e.gen) : S0.e.commits ≤ σ.cur.e.commits :=
  (((reach_inv h).rel S0 h0).same_gen hg).1

/-- **CatchUp's relink is exactly the newer nodes** (03 §5: "walked from the head, stopping at the first
    `seq ≤ fork`"): for a batch whose fit passes, `relinked` is the published log's nodes after the fork, and
    the rest of the published log is the fork's own log, or nothing when an Overflow replaced it. -/
theorem relink_exact {B : Nat} {σ : Sys} (h : Reach D B σ) {S0 : Pub} (h0 : S0 ∈ σ.hist)
    (hb0 : S0.e.isBuilt = false) (hb : σ.cur.e.isBuilt = false) (hg : S0.e.gen = σ.cur.e.gen)
    (hc : S0.e.commits = σ.cur.e.commits) :
    (∀ n ∈ relinked S0.ver σ.cur.e.log, S0.ver < n.seq) ∧
    (σ.cur.e.log = relinked S0.ver σ.cur.e.log ++ S0.e.log ∨
      (σ.cur.e.log = relinked S0.ver σ.cur.e.log ∧ hasOv σ.cur.e.log = true)) := by
  have hI := reach_inv h
  obtain ⟨_, _, _, _, _, N, hN, hl⟩ := (hI.rel S0 h0).chain hb0 hb hg hc
  have hS0 := (hI.pub S0 (List.mem_cons_of_mem _ h0)).1
  rcases hl with ⟨hl, _⟩ | ⟨hl, ho⟩
  · have : relinked S0.ver σ.cur.e.log = N := by rw [hl]; exact relinked_seam _ _ _ hN hS0.seq_le
    rw [this]; exact ⟨hN, Or.inl hl⟩
  · have : relinked S0.ver σ.cur.e.log = N := by rw [hl]; exact relinked_all _ _ hN
    rw [this]; exact ⟨hN, Or.inr ⟨hl, by rw [hl]; exact ho⟩⟩

/-- The fit's discard rule is native_index's `discarded_against` on the `(generation, commits)` projection. -/
theorem fit_discard_is_ni (S0 cur : Pub) (e1 : Entry) :
    IsDiscarded (fit S0 cur e1) = (NI.discardedAgainst e1.toSE (some cur.e.toSE)).isSome := by
  unfold fit
  cases NI.discardedAgainst e1.toSE (some cur.e.toSE) with
  | some why => rfl
  | none => simp only [Option.isSome_none]; split <;> (try split) <;> rfl

/-- With `d6 = false` the rule is native_index's model of today's `Graph::overtaken` (`NI.overtakenSE`), which
    requires a `Building` fork; with D6 it fires on any entry of the same generation whose count moved. -/
theorem overtaken_pre_d6 (ours theirs : Entry) :
    overtaken { d6 := false } ours theirs = NI.overtakenSE ours.toSE theirs.toSE := rfl

theorem overtaken_d6 (ours theirs : Entry) :
    overtaken D ours theirs = (ours.gen == theirs.gen && ours.commits != theirs.commits) := rfl

end CchReview
