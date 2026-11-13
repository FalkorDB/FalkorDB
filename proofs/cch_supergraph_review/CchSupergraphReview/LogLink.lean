import CchSupergraphReview.Skeleton
import CchRedesignReview.Main
set_option linter.unusedVariables false

/-!
# Discharging `LogComplete` with cch_redesign_review's `no_lost_write` (03 §4 step 2 "Why the drop is valid")

03 §4 step 2 justifies the drop in one sentence: "Every pair written after its slab was read is in the log, and
is resolved here." That is two premises, named here.

* **`Records`** (03 §5 "The log": "`record()` already receives `(id, u, v)` for creates, deletes (including the
  node-delete cascade) and weight SETs"; "Nothing is dropped silently"). A writer whose commit changes the
  liveness of a directed pair logs the pair, at its own version. It is about the writer, the one actor that moves
  the data version. Its Rust is on the stack (`maint.rs:238-253, 297-314` @08c), where `set_structural` and the
  `PAIR_CAP` overflow still drop pairs (`maint.rs:121-138` @08c, 03 §5); the design removes the drop. Not proved
  anywhere: it is a premise of this theorem and of `no_lost_write`'s meaning.
* **The log keeps every logged write** from the build's birth to `V_sg`: cch_redesign_review's `no_lost_write`
  (`CchReview.no_lost_write`), for every reachable state of its model of 03 §5-§7.

`logComplete_of_reach` combines them. Its remaining hypotheses are about the build's own versions:

* `hres`: the Building entry's structure is credited no later than `lo` for any pair (cch_redesign_review's
  reading R1: a fresh build's `res` is its birth version; `s7E` and the initial state set it, and only a
  consumer, the Supergraph phase itself, raises it);
* `hrv`: every slab was read at a version in `[lo, V_sg]` (03 §5 "Phases": ReadSkeleton runs after the birth and
  before Supergraph forks; one builder thread runs one batch at a time).

`hasOv`: an Overflow in the log makes Supergraph rescan the whole skeleton at its fork (03 §5), which is
`rescan_exact`; `supergraph_input_exact` covers both branches.
-/
namespace CchSG

open VMTensor CchReview

variable {H : Hist} {T : List Nat}

section
variable (H T)
/-- **Premise `Records`** (03 §5 "The log"). The writer that publishes data version `s + 1` logs, as
    `enc u w` at seq `s + 1`, every pair whose liveness its commit changes. -/
def Records (enc : Nat → Nat → Pair) (writes : List (Nat × Pair)) : Prop :=
  ∀ s u w, u ≠ w → liveDir T (H s) u w ≠ liveDir T (H (s + 1)) u w → (s + 1, enc u w) ∈ writes
end

/-- `enc` is the id of `(min(u,v), max(u,v))` (cch_redesign_review's `Pair`): symmetric. -/
def EncSym (enc : Nat → Nat → Pair) : Prop := ∀ u w, enc u w = enc w u

/-- The pairs resolved from `log` (every node with `seq ≤ V_sg`, which is every node: `nothing_from_the_future`). -/
def inLog (enc : Nat → Nat → Pair) (log : List Node) (u w : Nat) : Bool := decide (enc u w ∈ allPairs log)

theorem inLog_symm {enc : Nat → Nat → Pair} (he : EncSym enc) (log : List Node) : SymL (inLog enc log) :=
  fun u w => by simp [inLog, he u w]

/-- A Boolean that differs at `a ≤ b` changes at some single step in between. -/
theorem step_change (f : Nat → Bool) (a : Nat) : ∀ b, a ≤ b → f a ≠ f b → ∃ s, a ≤ s ∧ s < b ∧ f s ≠ f (s + 1)
  | 0, hab, hne => by
    have : a = 0 := by omega
    subst this; exact absurd rfl hne
  | b + 1, hab, hne => by
    by_cases ha : a = b + 1
    · subst ha; exact absurd rfl hne
    · by_cases hb : f a = f b
      · exact ⟨b, by omega, by omega, by rw [← hb]; exact hne⟩
      · obtain ⟨s, h1, h2, h3⟩ := step_change f a b (by omega) hb
        exact ⟨s, h1, by omega, h3⟩

/-- **`LogComplete` from the writes**: if writers record (`Records`) and the log holds every write after `lo` up
    to `V`, then every directed pair whose liveness changed after its row's read is resolved. -/
theorem logComplete_of_writes {enc : Nat → Nat → Pair} {writes : List (Nat × Pair)}
    (hrec : Records H T enc writes) (log : List Node) {lo V : Nat} {rv : Nat → Nat}
    (hrv : ∀ u, lo ≤ rv u ∧ rv u ≤ V)
    (hcov : ∀ s p, (s, p) ∈ writes → lo < s → s ≤ V → p ∈ allPairs log) :
    LogComplete H T rv V (inLog enc log) := by
  intro u w huw hch
  obtain ⟨s, h1, h2, h3⟩ := step_change (fun v => liveDir T (H v) u w) (rv u) V (hrv u).2 hch
  have hw := hrec s u w huw h3
  simp only [inLog, decide_eq_true_eq]
  exact hcov (s + 1) (enc u w) hw (by have := (hrv u).1; omega) (by omega)

/-- **`LogComplete` from cch_redesign_review's `no_lost_write`**, at any published version `S` of any reachable
    state of its model (the Supergraph batch's fork, `S.ver = V_sg`) whose log holds no Overflow. -/
theorem logComplete_of_reach {B : Nat} {σ : Sys} (h : Reach D B σ) {S : Pub} (hS : S ∈ σ.cur :: σ.hist)
    (hov : hasOv S.e.log = false) {lo : Nat} (hres : ∀ p, S.e.res p ≤ lo) {rv : Nat → Nat}
    (hrv : ∀ u, lo ≤ rv u ∧ rv u ≤ S.ver) {enc : Nat → Nat → Pair} (hrec : Records H T enc σ.writes) :
    LogComplete H T rv S.ver (inLog enc S.e.log) := by
  refine logComplete_of_writes hrec S.e.log hrv (fun s p hw hlo hs => ?_)
  rcases no_lost_write h hS s p hw hs with hr | ⟨n, hn, ⟨_, hp⟩ | ⟨hov', _⟩⟩
  · have := hres p; omega
  · exact (mem_allPairs S.e.log p).2 ⟨n, hn, hp⟩
  · have := (hasOv_false_iff S.e.log).1 hov n hn
    rw [this] at hov'; cases hov'

/-- **Theorem 2, end to end.** At the Supergraph fork `S` of any reachable state, step 3's input
    `U0 = triu(B′, 1)` is exactly the live pattern at `V_sg = S.ver`. Without an Overflow, `B′` is built from the
    slabs and the resolved log; with one, Supergraph rescans every row at `V_sg` (03 §5). Premises: the tensor
    invariants (versioned_matrix), `Records`, `EncSym`, and the build's version bounds `hres`, `hrv`. -/
theorem supergraph_input_exact (hinv : ∀ v, GraphInv (H v)) {B : Nat} {σ : Sys} (h : Reach D B σ) {S : Pub}
    (hS : S ∈ σ.cur :: σ.hist) {lo : Nat} (hres : ∀ p, S.e.res p ≤ lo) {rv : Nat → Nat}
    (hrv : ∀ u, lo ≤ rv u ∧ rv u ≤ S.ver) {enc : Nat → Nat → Pair} (he : EncSym enc)
    (hrec : Records H T enc σ.writes) (u w : Nat) :
    let rv' := if hasOv S.e.log then (fun _ => S.ver) else rv
    let inL' := if hasOv S.e.log then (fun _ _ => false) else inLog enc S.e.log
    (decide (u < w) && Bp H T rv' S.ver inL' u w) = (decide (u < w) && liveU T (H S.ver) u w) := by
  intro rv' inL'
  cases hov : hasOv S.e.log
  · have e1 : rv' = rv := by simp [rv', hov]
    have e2 : inL' = inLog enc S.e.log := by simp [inL', hov]
    rw [e1, e2]
    exact triu_bprime_live hinv (logComplete_of_reach h hS hov hres hrv hrec) (inLog_symm he _) u w
  · have e1 : rv' = fun _ => S.ver := by simp [rv', hov]
    have e2 : inL' = fun _ _ => false := by simp [inL', hov]
    rw [e1, e2]
    exact triu_bprime_live hinv (rescan_logComplete S.ver) (fun _ _ => rfl) u w

/-! ## Theorem 2 meets theorem 3: the supergraph is the game of the live pattern -/

/-- `U0 = triu(B′(π, π), 1)` in rank space, through `node_of` (`none` for a pool rank). -/
def rankU0 (nodeOf : Nat → Option Nat) (rel : Nat → Nat → Bool) (a b : Nat) : Bool :=
  match nodeOf a, nodeOf b with
  | some u, some w => decide (a < b) && rel u w
  | _, _ => false

/-- **The supergraph is the elimination game's chordal completion of the live pattern at `V_sg`** (03 §4 steps
    2-3, Revision 3.1): with an injective `node_of`, the game over step 3's input `U0` has the same rows, parents
    and arcs as the game over the live pattern at `V_sg`, because the two inputs are equal. Premises as in
    `supergraph_input_exact`. -/
theorem supergraph_is_live_game (hinv : ∀ v, GraphInv (H v)) {B : Nat} {σ : Sys} (h : Reach D B σ) {S : Pub}
    (hS : S ∈ σ.cur :: σ.hist) {lo : Nat} (hres : ∀ p, S.e.res p ≤ lo) {rv : Nat → Nat}
    (hrv : ∀ u, lo ≤ rv u ∧ rv u ≤ S.ver) {enc : Nat → Nat → Pair} (he : EncSym enc)
    (hrec : Records H T enc σ.writes) (nodeOf : Nat → Option Nat)
    (hinj : ∀ a b u, nodeOf a = some u → nodeOf b = some u → a = b) (n : Nat) :
    let rv' := if hasOv S.e.log then (fun _ => S.ver) else rv
    let inL' := if hasOv S.e.log then (fun _ _ => false) else inLog enc S.e.log
    rankU0 nodeOf (Bp H T rv' S.ver inL') = rankU0 nodeOf (liveU T (H S.ver)) ∧
    ∀ r, gameRow n (rankU0 nodeOf (Bp H T rv' S.ver inL')) r = gameRow n (rankU0 nodeOf (liveU T (H S.ver))) r := by
  intro rv' inL'
  have hc : LogComplete H T rv' S.ver inL' := by
    cases hov : hasOv S.e.log
    · have e1 : rv' = rv := by simp [rv', hov]
      have e2 : inL' = inLog enc S.e.log := by simp [inL', hov]
      rw [e1, e2]; exact logComplete_of_reach h hS hov hres hrv hrec
    · have e1 : rv' = fun _ => S.ver := by simp [rv', hov]
      have e2 : inL' = fun _ _ => false := by simp [inL', hov]
      rw [e1, e2]; exact rescan_logComplete S.ver
  have hsym : SymL inL' := by
    cases hov : hasOv S.e.log
    · have e2 : inL' = inLog enc S.e.log := by simp [inL', hov]
      rw [e2]; exact inLog_symm he _
    · have e2 : inL' = fun _ _ => false := by simp [inL', hov]
      rw [e2]; exact fun _ _ => rfl
  have heq : rankU0 nodeOf (Bp H T rv' S.ver inL') = rankU0 nodeOf (liveU T (H S.ver)) := by
    funext a b
    unfold rankU0
    cases ha : nodeOf a with
    | none => rfl
    | some u =>
      cases hb : nodeOf b with
      | none => rfl
      | some w =>
        by_cases hab : a < b
        · have huw : u ≠ w := fun e => by subst e; have := hinj a b u ha hb; omega
          simp only [hab, decide_true, Bool.true_and]
          exact bprime_offdiag hinv hc hsym huw
        · simp [hab]
  exact ⟨heq, fun r => by rw [heq]⟩

/-! ## Why `hrv`'s upper bound matters: a slab read after `V_sg` (design-level, no code) -/

/-- The empty graph. -/
def lateG0 : Graph := fun _ _ _ => ⟨none, none, false, false, []⟩
/-- One pending edge `0 → 1` of type 0, id 7. -/
def lateG1 : Graph := fun t u w =>
  if t = 0 ∧ u = 0 ∧ w = 1 then ⟨none, some (.id 7), false, true, []⟩ else lateG0 t u w
/-- The edge is created by the writer of version 1. -/
def lateH : Hist := fun v => if v = 0 then lateG0 else lateG1
/-- Any symmetric pair id. -/
def lateEnc : Nat → Nat → Pair := fun u w => u + w
/-- The writer of version 1 logs `{0, 1}`. -/
def lateWrites : List (Nat × Pair) := [(1, lateEnc 0 1)]

theorem lateDir (s u w : Nat) :
    liveDir [0] (lateH s) u w = (decide (s ≠ 0) && decide (u = 0) && decide (w = 1)) := by
  by_cases hs : s = 0 <;> by_cases hu : u = 0 <;> by_cases hw : w = 1 <;>
    simp [lateH, lateG0, lateG1, liveDir, edges, effV, hs, hu, hw]

/-- **`hrv`'s upper bound is needed** (design-level, no code to reproduce). Against `logComplete_of_writes`:
    writers record (`Records`, with `writes = [(1, enc 0 1)]`), `enc` is symmetric, the log at `V_sg = 0` holds
    every write in `(lo, V_sg]` (`hcov`, vacuous: there is none), and every row is read at or after `lo = 0`. Only
    `rv u ≤ V_sg` fails: row 0 is read at version 1. Then `LogComplete` fails, and `B′` holds `{0, 1}`, which is
    not live at `V_sg`. -/
theorem late_slab_breaks :
    Records lateH [0] lateEnc lateWrites ∧ EncSym lateEnc ∧
    (∀ s p, (s, p) ∈ lateWrites → 0 < s → s ≤ 0 → p ∈ allPairs ([] : List Node)) ∧
    (∀ u, 0 ≤ (fun _ => 1 : Nat → Nat) u) ∧ ¬ (∀ u, (fun _ => 1 : Nat → Nat) u ≤ 0) ∧
    ¬ LogComplete lateH [0] (fun _ => 1) 0 (inLog lateEnc []) ∧
    Bp lateH [0] (fun _ => 1) 0 (inLog lateEnc []) 0 1 = true ∧ liveU [0] (lateH 0) 0 1 = false := by
  refine ⟨fun s u w huw hch => ?_, fun u w => by simp [lateEnc, Nat.add_comm], fun s p _ h1 h2 => by omega,
    fun _ => Nat.zero_le _, fun h => by have := h 0; simp at this, fun hc => ?_, by decide, by decide⟩
  · rw [lateDir, lateDir] at hch
    by_cases hs : s = 0
    · subst hs
      by_cases hu : u = 0 <;> by_cases hw : w = 1 <;> simp_all [lateWrites]
    · simp [hs] at hch
  · have := hc 0 1 (by decide) (by rw [lateDir, lateDir]; decide)
    simp [inLog] at this

/-! ## The premises are satisfiable -/

/-- A history that never changes needs no write: `Records` holds with none. -/
theorem records_const (g : Graph) (T : List Nat) (enc : Nat → Nat → Pair) :
    Records (fun _ => g) T enc [] := fun _ _ _ _ h => absurd rfl h

example : Records lateH [0] lateEnc lateWrites := late_slab_breaks.1

/-- `hres` at the build's two births in cch_redesign_review's model: `CREATE INDEX` (`init`, `res = 0` at version
    0) and S7 (`s7E`, `res = v`). -/
theorem hres_init : ∀ p, CchReview.init.cur.e.res p ≤ CchReview.init.cur.ver := fun _ => Nat.le_refl _

theorem hres_s7E (v : Nat) (ps : List Pair) (nx : Nat) : ∀ p, (CchReview.s7E v ps nx).res p ≤ v :=
  fun _ => Nat.le_refl _

/-- Before Supergraph, a batch keeps `res` (ReadSkeleton and Order consume nothing), and a writer's segment on a
    Building copy only logs. -/
theorem batchCopy_keeps_res (F : Nat) (ph : CchReview.Phase) (adv : Bool) (e : CchReview.Entry)
    (h : CchReview.consumes ph = false) : (CchReview.batchCopy F ph adv e).res = e.res := by
  simp [CchReview.batchCopy, h]

theorem appendE_keeps_res (v : Nat) (ps : List Pair) (e : CchReview.Entry) :
    (CchReview.appendE v ps e).res = e.res := rfl

end CchSG
