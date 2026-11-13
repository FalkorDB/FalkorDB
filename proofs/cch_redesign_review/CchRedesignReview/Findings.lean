import CchRedesignReview.Main
set_option linter.unusedVariables false
set_option maxRecDepth 20000

/-!
# Findings and break checks: concrete interleavings

Each trace below is a closed term built from the step functions; its prefix is `Reach D B` (the design as read
here), and the last step either runs under `D` too (a design finding) or swaps in one rule (a break check:
the pre-03 rule, or an alternative reading of 03's words). `decide` evaluates the final state.

* `F1` (alternative reading of §6 Retention): "past the bound the writer clears upkeep and the log", read
  literally (own nodes included), loses a write through the S6 row of the rebase table.
* `F2` (design, minor): a segment that hands off before the writer escalates gives its copy a new generation,
  so `overtaken` is false and the hand-off's base, the pre-S6 hierarchy, replaces a published S6 lineage. No
  write is lost (`no_lost_write`); the S6 is.
* `F3` (alternative reading of §5 CatchUp's counter): "`pairs` = the batch's remainder plus theirs" with
  "theirs" the published counter double-counts the nodes before the fork.
* `F4` (alternative reading of §5 Bound): a check only on append lets a hand-off leave a `Building` log over the
  bound with no Overflow.
* `F5` = `B4` (missing from 03 §7's D6 list): a fold or S6 publish that does not move `batches_published`
  leaves `overtaken_by_indexer_batches`'s gate shut, and a writer reverts the job.
* `B1` (Appendix B V2.3, the @08c hook): a rebase that takes the published copy without the fork's own nodes
  loses the write. `B2` (V1.5): `overtaken` without D6 lets a writer revert an S6 publish. `B3` (A R1.1 / B
  V1.1): a CatchUp that does not relink loses the write. `B5` (reading R4): an adoption that lets an Overflow pass
  as absorbed loses the writes under it. Each passes under `D`.
-/
namespace CchReview
namespace Findings

/-! ## Traces -/

/-- One batch on the published version, reaching its phase's end (`Same`: nothing committed in between). -/
def adv1 (σ : Sys) : Sys := batchSys true true σ.cur σ

theorem reach_adv1 {B : Nat} {σ : Sys} (h : Reach D B σ) : Reach D B (adv1 σ) :=
  Reach.step h (Step.batch σ σ.cur (List.mem_cons_self ..) true true)

/-- `CREATE INDEX`, then ReadSkeleton → Order → Supergraph → Seed → Relax → Seal: adopted, `Built`, version 0. -/
def built0 : Sys := adv1 (adv1 (adv1 (adv1 (adv1 (adv1 init)))))

theorem reach_built0 (B : Nat) : Reach D B built0 :=
  reach_adv1 (reach_adv1 (reach_adv1 (reach_adv1 (reach_adv1 (reach_adv1 Reach.init)))))

theorem built0_state : built0.cur.e.st = .built none ∧ built0.cur.ver = 0 ∧ built0.cur.e.commits = 6 := by decide

/-- A writer that forks on the published version and flushes its segments after escalating. -/
def wr (R : Rules) (B : Nat) (pre : List (List Pair × Out)) (σ : Sys) : Sys := writerSys R B σ.cur pre [] true σ

theorem reach_wr {B : Nat} {σ : Sys} (h : Reach D B σ) (pre : List (List Pair × Out)) : Reach D B (wr D B pre σ) :=
  Reach.step h (Step.writer σ σ.cur (List.mem_cons_self ..) rfl pre [] true)

/-- W1 at version 1 crosses an S6 trigger (`upkeep = Some{S6, ticket 1, since 1}`); W2 at version 2 writes pair 8,
    retained. Bound `B = 1`. -/
def s7a : Sys := wr D 1 [([7], .setUp .s6)] built0
def s8a : Sys := wr D 1 [([8], .fits)] s7a
/-- The S6 job, forked at version 2, publishes: a new lineage (2), `upkeep` cleared, commits 7. -/
def s9a : Sys := jobSys true s8a.cur s8a

theorem reach_s9a : Reach D 1 s9a :=
  Reach.step (reach_wr (reach_wr (reach_built0 1) _) _) (Step.job s8a s8a.cur (List.mem_cons_self ..) true)

theorem s9a_state : s9a.cur.e.lineage = 2 ∧ s9a.cur.e.commits = 7 ∧ s9a.cur.e.st = .built none ∧
    s9a.cur.ver = 2 ∧ s8a.cur.e.st = .built (some ⟨.s6, 1, 1⟩) := by decide

theorem s8a_in_hist : s8a.cur ∈ s9a.cur :: s9a.hist := by
  have : s9a.hist = s8a.cur :: s8a.hist := rfl
  rw [this]; exact List.mem_cons_of_mem _ (List.mem_cons_self ..)

/-- W3 forked at version 2 *before* the S6 publish (`s8a.cur`) and commits after it, at version 3. -/
def w3 (R : Rules) (pre post : List (List Pair × Out)) : Sys := writerSys R 1 s8a.cur pre post true s9a

theorem step_w3 (R : Rules) (pre post : List (List Pair × Out)) : Step R 1 s9a (w3 R pre post) :=
  Step.writer s9a s8a.cur s8a_in_hist rfl pre post true

/-! ## F1: the literal "clears upkeep and the log" loses a write (design text, reading R2) -/

/-- Under the literal reading, W3's flush of pair 9 (before escalating) passes the bound with the retained pair
    8, clears the log *with its own node*, and the S6 rebase row then has no own node to replay: the published
    version 3 has pair 9 neither resolved (`res 9 = 2 < 3`) nor logged. -/
theorem F1_literal_clear_loses_write :
    Reach D 1 s9a ∧ Step { keepOwn := false } 1 s9a (w3 { keepOwn := false } [([9], .fits)] []) ∧
    let σ' := w3 { keepOwn := false } [([9], .fits)] []
    (3, 9) ∈ σ'.writes ∧ 3 ≤ σ'.cur.ver ∧ σ'.cur.e.res 9 < 3 ∧ ¬ covered σ'.cur.e.log 3 9 := by
  refine ⟨reach_s9a, step_w3 _ _ _, ?_⟩
  decide

/-- Under reading R2 (own nodes kept), the same interleaving resolves pair 9 at version 3. -/
theorem F1_reading_R2_keeps_it : (w3 D [([9], .fits)] []).cur.e.res 9 = 3 := by decide

/-! ## F2: a hand-off before escalation reverts a published S6 (design, minor) -/

/-- W3's first segment aborts before it escalates (03 §6 Hand-off: new generation, base = its private hierarchy).
    Its copy then has another generation, so it is not `overtaken`, and it replaces the published S6 entry: the
    published lineage goes from 2 back to 0, under a `Building{Repair}` whose base is the pre-S6 hierarchy. -/
theorem F2_preescalation_handoff_reverts_s6 :
    Reach D 1 (w3 D [([9], .abort)] []) ∧
    s9a.cur.e.lineage = 2 ∧ (w3 D [([9], .abort)] []).cur.e.lineage = 0 ∧
    (w3 D [([9], .abort)] []).cur.e.st = .building .repair := by
  refine ⟨Reach.step reach_s9a (step_w3 _ _ _), ?_⟩
  decide

/-- The same abort after escalating takes the S6 lineage as its base: the rebase runs before the flush. -/
theorem F2_after_escalation_keeps_s6 :
    (w3 D [] [([9], .abort)]).cur.e.lineage = 2 ∧ (w3 D [] [([9], .abort)]).cur.e.st = .building .repair := by
  decide

/-! ## F3: the CatchUp counter, "the batch's remainder plus theirs" -/

def t1 : Sys := wr D 10 [([5], .fits)] init
def t2 : Sys := wr D 10 [([6], .fits)] t1

theorem reach_t2 : Reach D 10 t2 := reach_wr (reach_wr Reach.init _) _

theorem t1_in_hist : t1.cur ∈ t2.cur :: t2.hist := List.mem_cons_of_mem _ (List.mem_cons_self ..)

/-- A ReadSkeleton batch read at version 1 catches up at version 2: the log is `[2:{6}, 1:{5}]`, 2 pairs. Read
    as "the batch's remainder (1) plus the published counter (2)", the counter would say 3. -/
theorem F3_counter_reading :
    Reach D 10 t2 ∧ t1.cur ∈ t2.cur :: t2.hist ∧
    (batchResult true false t1.cur t2.cur).map (fun e => (e.pairs, sumPairs e.log)) = some (2, 2) ∧
    (batchCopy t1.cur.ver .readSkeleton false t1.cur.e).pairs + t2.cur.e.pairs = 3 := by
  refine ⟨reach_t2, t1_in_hist, ?_⟩
  decide

/-! ## F4: a bound checked only on append -/

/-- A hand-off of a two-pair segment with `B = 1`: checked only on append, the new `Building` log is over the
    bound with no Overflow; checked on every writer change (reading R5), it becomes an Overflow. -/
theorem F4_bound_only_on_append :
    let bad := wr { checkAll := false } 1 [([5, 6], .abort)] built0
    let good := wr D 1 [([5, 6], .abort)] built0
    Step { checkAll := false } 1 built0 bad ∧
    bad.cur.e.isBuilt = false ∧ 1 < sumPairs bad.cur.e.log ∧ hasOv bad.cur.e.log = false ∧
    hasOv good.cur.e.log = true := by
  refine ⟨Step.writer built0 built0.cur (List.mem_cons_self ..) rfl _ [] true, ?_⟩
  decide

/-! ## Break checks: the invariant can see the Appendix A/B bugs -/

/-- B1 (Appendix B V2.3, `versioned.rs:786, 805` @08c): the rebase takes the published copy and carries only
    `by_this_version`, which is empty for CCH. A writer forked before a batch publish loses its pair. -/
def r1 : Sys := adv1 init

theorem B1_no_relink_loses_write :
    let σ' := writerSys { relink := false } 1 init.cur [([5], .fits)] [] true r1
    Reach D 1 r1 ∧ Step { relink := false } 1 r1 σ' ∧
    (1, 5) ∈ σ'.writes ∧ σ'.cur.e.res 5 < 1 ∧ ¬ covered σ'.cur.e.log 1 5 ∧
    covered (writerSys D 1 init.cur [([5], .fits)] [] true r1).cur.e.log 1 5 := by
  have hm : init.cur ∈ r1.cur :: r1.hist := by
    have : r1.hist = [init.cur] := rfl
    rw [this]; exact List.mem_cons_of_mem _ (List.mem_cons_self ..)
  refine ⟨reach_adv1 Reach.init, Step.writer r1 init.cur hm rfl _ [] true, ?_⟩
  decide

/-- B2 (Appendix B V1.5): with today's `overtaken` (a `Building` fork required), W3's `Built` fork is not
    overtaken, and its copy replaces the S6 publish: the lineage goes back to 0 and the count from 7 to 6 within
    the generation, which `commits_monotone` rules out under D6. -/
theorem B2_pre_d6_reverts_job :
    Step { d6 := false } 1 s9a (w3 { d6 := false } [([9], .fits)] []) ∧
    (w3 { d6 := false } [([9], .fits)] []).cur.e.gen = s9a.cur.e.gen ∧
    (w3 { d6 := false } [([9], .fits)] []).cur.e.commits < s9a.cur.e.commits ∧
    (w3 { d6 := false } [([9], .fits)] []).cur.e.lineage = 0 ∧
    (w3 D [([9], .fits)] []).cur.e.commits = 7 ∧ (w3 D [([9], .fits)] []).cur.e.lineage = 2 := by
  refine ⟨step_w3 _ _ _, ?_⟩
  decide

/-- B3 (Appendix A R1.1, B V1.1): `batchResult` with CatchUp treated as `Same` (no relink). -/
def batchResultNoRelink (b adv : Bool) (S0 cur : Pub) : Option Entry :=
  match S0.e.st with
  | .built _ => none
  | .building ph =>
    let e1 := batchCopy S0.ver ph adv S0.e
    if IsDiscarded (fit S0 cur e1) then none
    else if adopts ph then some { e1 with st := .built none, log := [], pairs := 0 }
    else some e1

def q1 : Sys := wr D 1 [([5], .fits)] init

/-- A ReadSkeleton batch read at version 0 publishes after a writer committed pair 5 at version 1: without the
    relink, the published version 1 has pair 5 neither resolved nor logged; with it, it is logged. -/
theorem B3_catchup_without_relink_loses_write :
    Reach D 1 q1 ∧ init.cur ∈ q1.cur :: q1.hist ∧
    (batchResultNoRelink true false init.cur q1.cur).map (fun e => decide (covered e.log 1 5) || decide (1 ≤ e.res 5))
      = some false ∧
    (batchResult true false init.cur q1.cur).map (fun e => decide (covered e.log 1 5)) = some true := by
  refine ⟨reach_wr Reach.init _, List.mem_cons_of_mem _ (List.mem_cons_self ..), ?_⟩
  decide

/-- B4 (a handoff risk, 03 §7 silent on it): the writer's rebase runs only when `overtaken_by_indexer_batches`'s
    gate sees `batch_published_since` (`versioned.rs:729-747` @08c, `NI.overtakenAny`). A fold or S6 publish
    that does not move `batches_published` (`mvcc_graph.rs:310` @08c moves it only through `prepare(_, _, false)`)
    leaves the gate shut, and the writer reverts the job even with D6's `overtaken`. -/
def jobSysNoBump (b : Bool) (S0 : Pub) (σ : Sys) : Sys :=
  match jobResult b S0 σ.cur σ.next with
  | some (e, n) => { σ with cur := ⟨σ.cur.ver, σ.cur.batches, e⟩, hist := σ.cur :: σ.hist, next := n }
  | none => σ

def s9n : Sys := jobSysNoBump true s8a.cur s8a

theorem B4_job_publish_must_bump_batches :
    let σ' := writerSys D 1 s8a.cur [([9], .fits)] [] true s9n
    s9n.cur.e.lineage = 2 ∧ s9n.cur.e.commits = 7 ∧
    σ'.cur.e.lineage = 0 ∧ σ'.cur.e.commits = 6 ∧ σ'.cur.e.gen = s9n.cur.e.gen := by
  decide

/-- B5 (reading R4): adoption that absorbs the relinked nodes' pairs but lets an Overflow pass as "fits" (03 §5
    adoption row is silent; §6b puts Overflow rescans off the lock only). -/
def batchResultAbsorbOv (b adv : Bool) (S0 cur : Pub) : Option Entry :=
  match S0.e.st with
  | .built _ => none
  | .building ph =>
    let e1 := batchCopy S0.ver ph adv S0.e
    let f := fit S0 cur e1
    if IsDiscarded f then none
    else
      let new := if f = .catchUp .refresh then relinked S0.ver cur.e.log else []
      if adopts ph then some { e1 with st := .built none, res := resolve e1.res (allPairs new) cur.ver, log := [], pairs := 0 }
      else some { e1 with log := new ++ e1.log, pairs := sumPairs new + e1.pairs }

/-- Five batches reach the Seal phase (`B = 1`). -/
def seal5 : Sys := adv1 (adv1 (adv1 (adv1 (adv1 init))))
/-- A writer then commits two pairs: past the bound, its copy's log is `[Overflow{1}]`. -/
def ov1 : Sys := wr D 1 [([5, 6], .fits)] seal5

/-- The Seal read at version 0 adopts at version 1 over the Overflow: absorbed as if it fit, pairs 5 and 6 are
    neither resolved nor logged in the `Built` result; under reading R4 it is `Restart`ed with the Overflow. -/
theorem B5_adoption_must_not_absorb_overflow :
    Reach D 1 ov1 ∧ seal5.cur ∈ ov1.cur :: ov1.hist ∧ seal5.cur.e.st = .building .seal ∧
    ov1.cur.e.log = [⟨1, .overflow⟩] ∧
    (batchResultAbsorbOv true false seal5.cur ov1.cur).map
      (fun e => (e.isBuilt, decide (1 ≤ e.res 5), decide (covered e.log 1 5))) = some (true, false, false) ∧
    (batchResult true false seal5.cur ov1.cur).map (fun e => (e.isBuilt, decide (covered e.log 1 5))) =
      some (false, true) := by
  refine ⟨reach_wr (reach_adv1 (reach_adv1 (reach_adv1 (reach_adv1 (reach_adv1 Reach.init))))) _,
    List.mem_cons_of_mem _ (List.mem_cons_self ..), ?_⟩
  decide

end Findings
end CchReview
