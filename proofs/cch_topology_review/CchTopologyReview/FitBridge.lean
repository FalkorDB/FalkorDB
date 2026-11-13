import CchTopologyReview.Lifetime
import CchRedesignReview

/-!
# FoldFit's first half, from the write-log model (`proofs/cch_redesign_review`)

The fold theorems (Fold.lean) take `WLin tF t`: only writers ran between the fold's fork `F` and its publish,
each on a private copy above `F`. This file proves the write-log side of that, in `cch_redesign_review`'s own
model (`CchReview.Sys`, `CchReview.Step D B`), over `Trace`, a run of that model, from its exported theorems
(its Main.lean):

* `fitted_job_only_writers_between`: a fitted job's fork and the published entry have the same generation,
  the same `indexer_commits` and the same `upkeep`;
* `builder_publish_bumps_commits`: a step that keeps the data version leaves the entry as it was, or is a
  batch or job publish of this index, which moves `indexer_commits` by one in the same generation;
* `commits_monotone`: `indexer_commits` never falls within a generation;
* `step_gen_mono`: no step lowers the generation;
* `job_window_frozen`: every version published in the window has the published entry's generation, count
  and state;
* `upkeep_writer_not_overtaken`: a writer that forks at a Built entry with `upkeep` set starts from the
  published entry and is not `overtaken`.

The theorems, for any run from the state `σ0` the job forked at to the state `σ1` it publishes onto, fitted:

* `fitted_job_no_builder_publish`: every step in the window that keeps the data version leaves this index's
  entry unchanged: no batch or job of this index publishes.
* `fitted_job_only_writer_steps`: every step in the window is a writer step (the data version moves by one)
  or leaves the entry unchanged.
* `fitted_window_writer_no_rebase`: every writer forking in the window starts from the published entry and
  does not rebase: its result is its own segments, flushed on the published entry, then retained.

**What stays argued**: only the cross-model step. That these write-log writer steps are this project's `fork`
followed by `Step`s, i.e. `WLin`: the two models have different state types (the write-log model abstracts
the topology to `res`), and they are not composed.
-/

namespace CchTopo.Fit

open CchReview

/-- The write-log model's steps from `σ0` to `σ1`. -/
inductive Trace (B : Nat) : Sys → Sys → Prop
  | refl (σ : Sys) : Trace B σ σ
  | step {σ0 σ1 σ2 : Sys} : Trace B σ0 σ1 → CchReview.Step D B σ1 σ2 → Trace B σ0 σ2

theorem trace_reach {B : Nat} {σ0 σ1 : Sys} (h : CchReview.Reach D B σ0) (ht : Trace B σ0 σ1) :
    CchReview.Reach D B σ1 := by
  induction ht with
  | refl => exact h
  | step _ hs ih => exact CchReview.Reach.step ih hs

/-- A step in front of a run is a run. -/
theorem trace_cons {B : Nat} {σi σj σ1 : Sys} (hs : CchReview.Step D B σi σj) (ht : Trace B σj σ1) :
    Trace B σi σ1 := by
  induction ht with
  | refl => exact Trace.step (Trace.refl σi) hs
  | step _ hs' ih => exact Trace.step ih hs'

/-- **No run lowers the generation** (`step_gen_mono`, step by step). -/
theorem trace_gen_mono {B : Nat} {σ0 σ1 : Sys} (h : CchReview.Reach D B σ0) (ht : Trace B σ0 σ1) :
    σ0.cur.e.gen ≤ σ1.cur.e.gen := by
  induction ht with
  | refl => exact Nat.le_refl _
  | step ht' hs ih => exact Nat.le_trans ih (step_gen_mono (trace_reach h ht') hs)

/-- A step forgets no published version. -/
theorem step_keeps_published {B : Nat} {σ σ' : Sys} (hs : CchReview.Step D B σ σ') :
    ∀ x ∈ σ.cur :: σ.hist, x ∈ σ'.cur :: σ'.hist := by
  intro x hx
  cases hs with
  | writer fork hf hv pre post b => exact List.mem_cons_of_mem _ hx
  | batch S0 h0 b adv =>
    unfold batchSys
    split
    · exact List.mem_cons_of_mem _ hx
    · exact hx
  | job S0 h0 b =>
    unfold jobSys
    split
    · exact List.mem_cons_of_mem _ hx
    · exact hx
  | other => exact List.mem_cons_of_mem _ hx

theorem trace_keeps_published {B : Nat} {σ0 σ1 : Sys} (ht : Trace B σ0 σ1) :
    ∀ x ∈ σ0.cur :: σ0.hist, x ∈ σ1.cur :: σ1.hist := by
  induction ht with
  | refl => exact fun _ h => h
  | step _ hs ih => exact fun x hx => step_keeps_published hs x (ih x hx)

/-- Only a writer moves the data version, by one. -/
theorem step_ver {B : Nat} {σ σ' : Sys} (hs : CchReview.Step D B σ σ') :
    σ'.cur.ver = σ.cur.ver ∨ σ'.cur.ver = σ.cur.ver + 1 := by
  cases hs with
  | writer fork hf hv pre post b => exact Or.inr rfl
  | batch S0 h0 b adv =>
    left; unfold batchSys; split <;> rfl
  | job S0 h0 b =>
    left; unfold jobSys; split <;> rfl
  | other => exact Or.inl rfl

theorem trace_ver_mono {B : Nat} {σ0 σ1 : Sys} (ht : Trace B σ0 σ1) : σ0.cur.ver ≤ σ1.cur.ver := by
  induction ht with
  | refl => exact Nat.le_refl _
  | step _ hs ih => rcases step_ver hs with e | e <;> omega

/-- Inside the window, the entry is in the fork's generation. -/
theorem window_gen {B : Nat} {σ0 σ1 : Sys} (h : CchReview.Reach D B σ0) (ht : Trace B σ0 σ1)
    {b : Bool} {r : CchReview.Entry × Nat} (he : jobResult b σ0.cur σ1.cur σ1.next = some r)
    {σi : Sys} (h1 : Trace B σ0 σi) (h2 : Trace B σi σ1) : σi.cur.e.gen = σ0.cur.e.gen := by
  have hS0 : σ0.cur ∈ σ1.cur :: σ1.hist := trace_keeps_published ht _ (List.mem_cons_self ..)
  obtain ⟨hgF, _, _, _⟩ := fitted_job_only_writers_between (trace_reach h ht) hS0 he
  have a := trace_gen_mono h h1
  have c := trace_gen_mono (trace_reach h h1) h2
  omega

/-- **No batch or job of this index publishes between a fitted job's fork and its publish.** `σ0` is the
    state whose published version the job forked at; `σ1` is the state it publishes onto, fitted
    (`jobResult … = some r`). Every step `σi → σj` on the run from `σ0` to `σ1` that keeps the data version (so
    not a writer) leaves the entry unchanged. -/
theorem fitted_job_no_builder_publish {B : Nat} {σ0 σ1 : Sys} (h : CchReview.Reach D B σ0) (ht : Trace B σ0 σ1)
    {b : Bool} {r : CchReview.Entry × Nat} (he : jobResult b σ0.cur σ1.cur σ1.next = some r)
    {σi σj : Sys} (h1 : Trace B σ0 σi) (hs : CchReview.Step D B σi σj) (h2 : Trace B σj σ1)
    (hv : σj.cur.ver = σi.cur.ver) : σj.cur.e = σi.cur.e := by
  have hg : σi.cur.e.gen = σ0.cur.e.gen := window_gen h ht he h1 (trace_cons hs h2)
  have hRi := trace_reach h h1
  have hR1 := trace_reach h ht
  have hS0 : σ0.cur ∈ σ1.cur :: σ1.hist := trace_keeps_published ht _ (List.mem_cons_self ..)
  obtain ⟨hgF, hcF, _, _⟩ := fitted_job_only_writers_between hR1 hS0 he
  rcases builder_publish_bumps_commits hRi hs hv with e | ⟨hg', hc'⟩
  · exact e
  · exfalso
    have hS0i : σ0.cur ∈ σi.cur :: σi.hist := trace_keeps_published h1 _ (List.mem_cons_self ..)
    have hle : σ0.cur.e.commits ≤ σi.cur.e.commits := by
      rcases List.mem_cons.1 hS0i with e | e
      · rw [e]; exact Nat.le_refl _
      · exact commits_monotone hRi e hg.symm
    have hj : σj.cur ∈ σ1.cur :: σ1.hist := trace_keeps_published h2 _ (List.mem_cons_self ..)
    have hle2 : σj.cur.e.commits ≤ σ1.cur.e.commits := by
      rcases List.mem_cons.1 hj with e | e
      · rw [e]; exact Nat.le_refl _
      · exact commits_monotone hR1 e (by rw [hg', hg, hgF])
    omega

/-- **Every step from a fitted job's fork to its publish is a writer step or leaves this index's entry
    unchanged.** -/
theorem fitted_job_only_writer_steps {B : Nat} {σ0 σ1 : Sys} (h : CchReview.Reach D B σ0) (ht : Trace B σ0 σ1)
    {b : Bool} {r : CchReview.Entry × Nat} (he : jobResult b σ0.cur σ1.cur σ1.next = some r)
    {σi σj : Sys} (h1 : Trace B σ0 σi) (hs : CchReview.Step D B σi σj) (h2 : Trace B σj σ1) :
    σj.cur.ver = σi.cur.ver + 1 ∨ σj.cur.e = σi.cur.e := by
  rcases step_ver hs with hv | hv
  · exact Or.inr (fitted_job_no_builder_publish h ht he h1 hs h2 hv)
  · exact Or.inl hv

/-- **Every writer in the window starts from the published entry and does not rebase**: a writer that forks
    (not stale) at a state `σi` of the window gets, from `writerE`, its own pre-escalation segments flushed on
    the published entry, then the rest, then retention. No `overtaken` rebase happens. -/
theorem fitted_window_writer_no_rebase {B : Nat} {σ0 σ1 : Sys} (h : CchReview.Reach D B σ0)
    (ht : Trace B σ0 σ1) {b : Bool} {r : CchReview.Entry × Nat} (he : jobResult b σ0.cur σ1.cur σ1.next = some r)
    {σi : Sys} (h1 : Trace B σ0 σi) (h2 : Trace B σi σ1)
    {fork : Pub} (hf : fork ∈ σi.cur :: σi.hist) (hv : fork.ver = σi.cur.ver)
    (pre post : List (List Pair × Out)) (bw : Bool) :
    fork.e = σi.cur.e ∧
    writerE D B σi.cur fork pre post bw σi.next =
      (retain (segs D B (σi.cur.ver + 1) post (segs D B (σi.cur.ver + 1) pre σi.cur.e σi.next).1
        (segs D B (σi.cur.ver + 1) pre σi.cur.e σi.next).2).1,
       (segs D B (σi.cur.ver + 1) post (segs D B (σi.cur.ver + 1) pre σi.cur.e σi.next).1
        (segs D B (σi.cur.ver + 1) pre σi.cur.e σi.next).2).2) := by
  have hR1 := trace_reach h ht
  have hRi := trace_reach h h1
  have hS0 : σ0.cur ∈ σ1.cur :: σ1.hist := trace_keeps_published ht _ (List.mem_cons_self ..)
  obtain ⟨_, _, _, u, _, hu1⟩ := fitted_job_only_writers_between hR1 hS0 he
  -- the window's entry carries the job's `upkeep`
  have hui : σi.cur.e.st = .built (some u) := by
    rcases List.mem_cons.1 (trace_keeps_published h2 _ (List.mem_cons_self ..)) with e | e
    · rw [e]; exact hu1
    · have hver : σ0.cur.ver ≤ σi.cur.ver := trace_ver_mono h1
      rw [(job_window_frozen hR1 hS0 he σi.cur e hver).2.2]; exact hu1
  obtain ⟨hfe, hov⟩ := upkeep_writer_not_overtaken hRi hui hf hv pre
  rw [hfe] at hov
  refine ⟨hfe, ?_⟩
  unfold writerE
  simp only [hfe, hov, NI.overtakenAny, List.any_cons, List.any_nil, Bool.or_false, id, Bool.and_false,
    Bool.false_eq_true, ↓reduceIte]

end CchTopo.Fit
