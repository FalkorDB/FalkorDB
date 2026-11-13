import CchRedesignReview.Findings
set_option linter.unusedVariables false
set_option maxRecDepth 20000

/-!
# Witnesses: the claimed transitions happen in reachable states

Each theorem gives a reachable state (`Reach D B`, the design as read here) and evaluates it with `decide`, so
the transitions the invariant covers are not vacuous: a fold publish, rebase row 3 (fold), rebase row 2 within
and over budget, a CatchUp with relink, and an adoption `Restart`.
-/
namespace CchReview
namespace Witness
open Findings

/-! ## A fold publish, then rebase row 3 (fold) -/

/-- Version 1 crosses a fold trigger (`upkeep = Some{fold, ticket 1, since 1}`); version 2 writes pair 8,
    retained. Bound `B = 5`. -/
def f7 : Sys := wr D 5 [([7], .setUp .fold)] built0
def f8 : Sys := wr D 5 [([8], .fits)] f7
/-- The fold job, forked at version 2, publishes. -/
def fj : Sys := jobSys true f8.cur f8

theorem reach_fj : Reach D 5 fj :=
  Reach.step (reach_wr (reach_wr (reach_built0 5) _) _) (Step.job f8 f8.cur (List.mem_cons_self ..) true)

/-- **Fold publish**: the count moves from 6 to 7, `upkeep` and the retained log are cleared, the lineage and the
    hierarchy are kept. -/
theorem fold_publish :
    Reach D 5 fj ∧ f8.cur.e.st = .built (some ⟨.fold, 1, 1⟩) ∧ f8.cur.e.commits = 6 ∧
    fj.cur.e.commits = 7 ∧ fj.cur.e.st = .built none ∧ fj.cur.e.log = [] ∧
    fj.cur.e.lineage = f8.cur.e.lineage ∧ fj.cur.e.res 8 = 2 := by
  refine ⟨reach_fj, ?_⟩
  decide

theorem f8_in_hist : f8.cur ∈ fj.cur :: fj.hist := by
  have : fj.hist = f8.cur :: f8.hist := rfl
  rw [this]; exact List.mem_cons_of_mem _ (List.mem_cons_self ..)

/-- A writer forked at version 2, before the fold published, commits after it at version 3. -/
def fw : Sys := writerSys D 5 f8.cur [([9], .fits)] [] true fj

/-- **Rebase row 3, fold**: the writer is overtaken (the count moved), the lineage is the fold's, so it re-applies
    the fold to its copy: the published count stays 7 (no revert) and pair 9 is resolved at 3. -/
theorem row3_fold :
    Reach D 5 fw ∧ fw.cur.e.commits = 7 ∧ fw.cur.e.res 9 = 3 ∧ fw.cur.e.st = .built none ∧
    fw.cur.e.lineage = fj.cur.e.lineage ∧ fw.cur.ver = 3 := by
  refine ⟨Reach.step reach_fj (Step.writer fj f8.cur f8_in_hist rfl _ [] true), ?_⟩
  decide

/-! ## Rebase row 2: a writer forked at the Seal version commits after the adoption -/

theorem built0_eq : built0 = adv1 seal5 := rfl

theorem seal5_in_hist : seal5.cur ∈ built0.cur :: built0.hist := by
  have : built0.hist = seal5.cur :: seal5.hist := rfl
  rw [this]; exact List.mem_cons_of_mem _ (List.mem_cons_self ..)

/-- The writer forked at the Seal's version (Building, phase Seal) and logged pair 4; the Seal adopted. -/
def r2 (b : Bool) : Sys := writerSys D 5 seal5.cur [([4], .fits)] [] b built0

/-- **Row 2 within budget**: the §6 flush over the own nodes onto the adopted hierarchy: `Built`, pair 4 resolved
    at 1. -/
theorem row2_fits :
    Reach D 5 (r2 true) ∧ seal5.cur.e.st = .building .seal ∧ built0.cur.e.st = .built none ∧
    (r2 true).cur.e.st = .built none ∧ (r2 true).cur.e.res 4 = 1 ∧ (r2 true).cur.e.gen = built0.cur.e.gen := by
  refine ⟨Reach.step (reach_built0 5) (Step.writer built0 seal5.cur seal5_in_hist rfl _ [] true), ?_⟩
  decide

/-- **Row 2 over budget**: `Absorbed::Restart` under a new generation (`next_generation()` = 1), base the adopted
    hierarchy, the own node as its log: pair 4 is covered. -/
theorem row2_restart :
    Reach D 5 (r2 false) ∧ (r2 false).cur.e.st = .building .repair ∧ (r2 false).cur.e.gen = 1 ∧
    built0.cur.e.gen = 0 ∧ (r2 false).cur.e.log = [⟨1, .pairs [4]⟩] ∧ covered (r2 false).cur.e.log 1 4 := by
  refine ⟨Reach.step (reach_built0 5) (Step.writer built0 seal5.cur seal5_in_hist rfl _ [] false), ?_⟩
  decide

/-! ## CatchUp with relink (F3's state) and adoption `Restart` (B5's state) -/

def cu : Sys := batchSys true false t1.cur t2

/-- **CatchUp**: a ReadSkeleton batch read at version 1 publishes at version 2: the writer's node at 2 relinked on
    top of the batch's copy, the counter 2, the count 1. -/
theorem catchup_reachable :
    Reach D 10 cu ∧ cu.cur.e.log = [⟨2, .pairs [6]⟩, ⟨1, .pairs [5]⟩] ∧ cu.cur.e.pairs = 2 ∧
    cu.cur.e.commits = 1 ∧ cu.cur.e.st = .building .readSkeleton ∧ cu.cur.ver = 2 := by
  refine ⟨Reach.step reach_t2 (Step.batch t2 t1.cur t1_in_hist true false), ?_⟩
  decide

def ar : Sys := batchSys true false seal5.cur ov1

/-- **Adoption `Restart`**: the Seal read at version 0 adopts at version 1 over a writer's Overflow: it keeps the
    generation and the job (`Building{Repair}`, generation 0), the Overflow as its log. -/
theorem adoption_restart_reachable :
    Reach D 1 ar ∧ ar.cur.e.st = .building .repair ∧ ar.cur.e.gen = 0 ∧ ar.cur.e.log = [⟨1, .overflow⟩] ∧
    ar.cur.e.commits = 6 ∧ covered ar.cur.e.log 1 5 ∧ covered ar.cur.e.log 1 6 := by
  refine ⟨Reach.step (reach_wr (reach_adv1 (reach_adv1 (reach_adv1 (reach_adv1 (reach_adv1 Reach.init))))) _)
    (Step.batch ov1 seal5.cur (List.mem_cons_of_mem _ (List.mem_cons_self ..)) true false), ?_⟩
  decide

end Witness
end CchReview
