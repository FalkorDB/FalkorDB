import CchSupergraphReview.Masked
import CchSupergraphReview.Game
set_option linter.unusedVariables false

/-!
# B′ is the live pattern at V_sg (03 §3 "Input skeleton", step 0; §4 steps 1-2; §5 "Bound and Overflow")

`H v` is the graph at data version `v` (only a writer moves it, `mvcc_graph.rs:247-248, 293-304` @08c). `T` is the
index's relationship types. Row `u` of the skeleton is read in its slab, at the slab's version `rv u` (03 §5
"Phases": ReadSkeleton reads node-id slabs at any snapshot). Everything is in node space; `bprime_rank` moves it to
rank space through any injective `π`.

* `liveDir g u w`: some indexed type has a live `u → w` edge in `g`. `liveU` is either direction.
* `P`: the union across types of the slab's masked reads (03 §3 first two bullets). `skel`:
  `select(OFFDIAG, eWiseAdd(ONEB, P, Pᵀ))` (third bullet; `input.rs:112-113` @05).
* `inL u w`: the pair `{u, w}` was resolved from the log (any representation; symmetric).
* `Aset`, `Dset`, `Bp`: 03 §4 step 2, `B′ = (B ∖ D) ∪ A`, with `A` the resolved pairs with a live indexed edge at
  `V` (`V_sg`) in any type and either direction, `D` the resolved pairs with none, both orientations.
* `LogComplete`: **the premise**. Every directed pair whose liveness changed between its row's read and `V` was
  resolved. `logComplete_of_writes` and `logComplete_of_reach` derive it from cch_redesign_review's `no_lost_write`
  plus `Records` (writers log every pair whose liveness they change, 03 §5 "The log").
-/
namespace CchSG

open VMTensor

/-- A data history: the graph at each data version. -/
abbrev Hist := Nat → Graph

section
variable (H : Hist) (T : List Nat)

/-- Some indexed type has a live `u → w` edge in `g`. -/
def liveDir (g : Graph) (u w : Nat) : Bool := T.any (fun t => !(edges (g t u w)).isEmpty)

/-- A live indexed edge between `u` and `w`, either direction. -/
def liveU (g : Graph) (u w : Nat) : Bool := liveDir T g u w || liveDir T g w u

/-- Row `u`'s masked reads, unioned across types, at its slab's version `rv u`. -/
def P (rv : Nat → Nat) (u w : Nat) : Bool := T.any (fun t => maskedRead (H (rv u) t u w))

/-- 03 §3: `S = select(OFFDIAG, eWiseAdd(ONEB, P, Pᵀ))`. -/
def skel (rv : Nat → Nat) (u w : Nat) : Bool := (u != w) && (P H T rv u w || P H T rv w u)

/-- 03 §4 step 2's `A`: resolved pairs with a live indexed edge at `V`. -/
def Aset (V : Nat) (inL : Nat → Nat → Bool) (u w : Nat) : Bool := inL u w && liveU T (H V) u w

/-- 03 §4 step 2's `D`: resolved pairs with none. -/
def Dset (V : Nat) (inL : Nat → Nat → Bool) (u w : Nat) : Bool := inL u w && !liveU T (H V) u w

/-- 03 §4 step 2: `B′ = (B ∖ D) ∪ A`. -/
def Bp (rv : Nat → Nat) (V : Nat) (inL : Nat → Nat → Bool) (u w : Nat) : Bool :=
  (skel H T rv u w && !Dset H T V inL u w) || Aset H T V inL u w

/-- **The premise of theorem 2**: every directed pair whose liveness changed between its row's read and `V` is a
    resolved log pair. -/
def LogComplete (rv : Nat → Nat) (V : Nat) (inL : Nat → Nat → Bool) : Prop :=
  ∀ u w, u ≠ w → liveDir T (H (rv u)) u w ≠ liveDir T (H V) u w → inL u w = true

/-- The log never holds a self-loop (03 §5 "The log": dropped, as at `maint.rs:131` @08c). -/
def NoSelfLoop (inL : Nat → Nat → Bool) : Prop := ∀ u, inL u u = false

/-- `inL` is a set of unordered pairs. -/
def SymL (inL : Nat → Nat → Bool) : Prop := ∀ u w, inL u w = inL w u
end

variable {H : Hist} {T : List Nat}

theorem liveDir_iff (g : Graph) (u w : Nat) : liveDir T g u w = true ↔ ∃ t ∈ T, edges (g t u w) ≠ [] := by
  simp [liveDir]

/-- **Theorem 1, across types**: row `u`'s read is the live pattern at its slab's version. -/
theorem P_live (hinv : ∀ v, GraphInv (H v)) (rv : Nat → Nat) (u w : Nat) :
    P H T rv u w = liveDir T (H (rv u)) u w := by
  apply Bool.eq_iff_iff.2
  simp only [P, List.any_eq_true, liveDir_iff]
  exact ⟨fun ⟨t, ht, h⟩ => ⟨t, ht, (maskedRead_live _ (hinv _ t u w)).1 h⟩,
         fun ⟨t, ht, h⟩ => ⟨t, ht, (maskedRead_live _ (hinv _ t u w)).2 h⟩⟩

theorem skel_symm (rv : Nat → Nat) (u w : Nat) : skel H T rv u w = skel H T rv w u := by
  simp only [skel, bne_comm, Bool.or_comm]

theorem skel_offdiag (rv : Nat → Nat) (u : Nat) : skel H T rv u u = false := by simp [skel]

theorem liveU_symm (g : Graph) (u w : Nat) : liveU T g u w = liveU T g w u := by
  simp only [liveU, Bool.or_comm]

/-- **Theorem 2 (03 §4 step 2, Revision 3.1).** Given `LogComplete`, `B′` is exactly the live pattern at `V`, off
    the diagonal: a pair is in `B′` iff its ends differ and it has a live indexed edge at `V`. -/
theorem bprime_live (hinv : ∀ v, GraphInv (H v)) {rv : Nat → Nat} {V : Nat} {inL : Nat → Nat → Bool}
    (hc : LogComplete H T rv V inL) (hsym : SymL inL) (hloop : NoSelfLoop inL) (u w : Nat) :
    Bp H T rv V inL u w = ((u != w) && liveU T (H V) u w) := by
  by_cases huw : u = w
  · subst huw
    simp [Bp, skel_offdiag, Aset, hloop u]
  · have hne : (u != w) = true := by simpa using huw
    unfold Bp Aset Dset
    rw [hne, Bool.true_and]
    cases hl : inL u w
    · -- not logged: both directions are unchanged since their reads
      have h1 : liveDir T (H (rv u)) u w = liveDir T (H V) u w :=
        Classical.byContradiction fun h => by rw [hc u w huw h] at hl; cases hl
      have h2 : liveDir T (H (rv w)) w u = liveDir T (H V) w u :=
        Classical.byContradiction fun h => by
          have h' := hc w u (Ne.symm huw) h; rw [hsym w u, hl] at h'; cases h'
      simp only [skel, hne, Bool.true_and, P_live hinv, h1, h2, liveU, Bool.false_and, Bool.not_false,
        Bool.and_true, Bool.or_false]
    · cases liveU T (H V) u w <;> simp

/-- Off the diagonal no self-loop premise is needed: `B′` is the live pattern at `V` wherever `u ≠ w`. -/
theorem bprime_offdiag (hinv : ∀ v, GraphInv (H v)) {rv : Nat → Nat} {V : Nat} {inL : Nat → Nat → Bool}
    (hc : LogComplete H T rv V inL) (hsym : SymL inL) {u w : Nat} (hne : u ≠ w) :
    Bp H T rv V inL u w = liveU T (H V) u w := by
  -- replace `inL` by its off-diagonal part, which satisfies every premise
  let inL' : Nat → Nat → Bool := fun a b => (a != b) && inL a b
  have hc' : LogComplete H T rv V inL' := fun a b hab hch => by simp [inL', hab, hc a b hab hch]
  have hsym' : SymL inL' := fun a b => by simp only [inL', bne_comm, hsym a b]
  have hloop' : NoSelfLoop inL' := fun a => by simp [inL']
  have e : Bp H T rv V inL u w = Bp H T rv V inL' u w := by simp [Bp, Aset, Dset, inL', bne_iff_ne.2 hne]
  rw [e, bprime_live hinv hc' hsym' hloop']
  simp [bne_iff_ne.2 hne]

/-- Step 3 reads `U0 = triu(B′, 1)`: there the diagonal is gone, so no self-loop premise is needed. -/
theorem triu_bprime_live (hinv : ∀ v, GraphInv (H v)) {rv : Nat → Nat} {V : Nat} {inL : Nat → Nat → Bool}
    (hc : LogComplete H T rv V inL) (hsym : SymL inL) (u w : Nat) :
    (decide (u < w) && Bp H T rv V inL u w) = (decide (u < w) && liveU T (H V) u w) := by
  by_cases huw : u < w
  · rw [bprime_offdiag hinv hc hsym (Nat.ne_of_lt huw)]
  · simp [huw]

/-- `B′` is symmetric: the skeleton symmetrizes and `A`, `D` hold both orientations. -/
theorem bprime_symm {rv : Nat → Nat} {V : Nat} {inL : Nat → Nat → Bool} (hsym : SymL inL) (u w : Nat) :
    Bp H T rv V inL u w = Bp H T rv V inL w u := by
  simp only [Bp, Aset, Dset, skel_symm rv u w, hsym u w, liveU_symm (H V) u w]

/-! ## The Overflow rescan (03 §5 "Bound and Overflow": "at Supergraph: it re-reads the full skeleton at its fork") -/

/-- **A rescan needs no log**: every row read at `V` makes `LogComplete` hold for the empty log. -/
theorem rescan_logComplete (V : Nat) : LogComplete H T (fun _ => V) V (fun _ _ => false) :=
  fun _ _ _ h => absurd rfl h

theorem rescan_exact (hinv : ∀ v, GraphInv (H v)) (V : Nat) (u w : Nat) :
    Bp H T (fun _ => V) V (fun _ _ => false) u w = ((u != w) && liveU T (H V) u w) :=
  bprime_live hinv (rescan_logComplete V) (fun _ _ => rfl) (fun _ => rfl) u w

/-! ## Rank allocation (03 §3 step 0, §4 step 1): every live node at `V` gets a rank (I1′ at V_sg) -/

section
variable (H T)
/-- `u` has a live indexed edge at `V` to another node. -/
def liveNode (V : Nat) (u : Nat) : Prop := ∃ x, x ≠ u ∧ liveU T (H V) u x = true

/-- Who holds a rank after step 1: a node of positive skeleton degree (step 0), or a resolved pair's endpoint
    with a live indexed edge at `V` (§4 step 1). -/
def Ranked (rv : Nat → Nat) (V : Nat) (inL : Nat → Nat → Bool) (u : Nat) : Prop :=
  (∃ w, skel H T rv u w = true) ∨ ((∃ w, inL u w = true) ∧ liveNode H T V u)
end

/-- **Every node with a live indexed edge at `V` has a rank** (03 §4 Contract: "every node with an indexed edge
    at V_sg has a rank"). -/
theorem live_node_ranked (hinv : ∀ v, GraphInv (H v)) {rv : Nat → Nat} {V : Nat} {inL : Nat → Nat → Bool}
    (hc : LogComplete H T rv V inL) (hsym : SymL inL) {u w : Nat} (huw : u ≠ w)
    (hl : liveU T (H V) u w = true) : Ranked H T rv V inL u := by
  cases hlog : inL u w
  · left; refine ⟨w, ?_⟩
    have h1 : liveDir T (H (rv u)) u w = liveDir T (H V) u w :=
      Classical.byContradiction fun h => by rw [hc u w huw h] at hlog; cases hlog
    have h2 : liveDir T (H (rv w)) w u = liveDir T (H V) w u :=
      Classical.byContradiction fun h => by
        have h' := hc w u (Ne.symm huw) h; rw [hsym w u, hlog] at h'; cases h'
    simp only [skel, P_live hinv, h1, h2]
    simp only [liveU] at hl
    simp [huw, hl]
  · exact Or.inr ⟨⟨w, hlog⟩, w, Ne.symm huw, hl⟩

/-- **A ranked node that loses all its edges keeps its rank and gets no arcs** (03 §4 step 2). -/
theorem dead_node_no_entry (hinv : ∀ v, GraphInv (H v)) {rv : Nat → Nat} {V : Nat} {inL : Nat → Nat → Bool}
    (hc : LogComplete H T rv V inL) (hsym : SymL inL) (hloop : NoSelfLoop inL) {u : Nat}
    (hd : ¬ liveNode H T V u) (w : Nat) : Bp H T rv V inL u w = false := by
  rw [bprime_live hinv hc hsym hloop]
  by_cases huw : u = w
  · simp [huw]
  · cases hl : liveU T (H V) u w
    · simp
    · exact absurd ⟨w, Ne.symm huw, hl⟩ hd

/-! ## Rank space -/

/-- `B′` in rank space through `π` (03 §4 step 2: `(π(u), π(v))`). -/
def BpRank (π : Nat → Option Nat) (rv : Nat → Nat) (V : Nat) (inL : Nat → Nat → Bool) (a b : Nat) : Prop :=
  ∃ u w, π u = some a ∧ π w = some b ∧ Bp H T rv V inL u w = true

/-- The live pattern at `V` in rank space through `π`. -/
def LiveRank (π : Nat → Option Nat) (V : Nat) (a b : Nat) : Prop :=
  ∃ u w, π u = some a ∧ π w = some b ∧ u ≠ w ∧ liveU T (H V) u w = true

/-- **B′ in rank space is the permuted live pattern at `V`**, and every live pair has both ends ranked (with an
    allocation that ranks exactly what 03 §3 step 0 and §4 step 1 rank). -/
theorem bprime_rank (hinv : ∀ v, GraphInv (H v)) {rv : Nat → Nat} {V : Nat} {inL : Nat → Nat → Bool}
    (hc : LogComplete H T rv V inL) (hsym : SymL inL) (hloop : NoSelfLoop inL) (π : Nat → Option Nat)
    (halloc : ∀ u, Ranked H T rv V inL u → (π u).isSome = true) :
    (∀ a b, BpRank (H := H) (T := T) π rv V inL a b ↔ LiveRank (H := H) (T := T) π V a b) ∧
    (∀ u w, u ≠ w → liveU T (H V) u w = true → ∃ a b, π u = some a ∧ π w = some b ∧
      BpRank (H := H) (T := T) π rv V inL a b) := by
  refine ⟨fun a b => ?_, fun u w huw hl => ?_⟩
  · simp only [BpRank, LiveRank, bprime_live hinv hc hsym hloop, Bool.and_eq_true, bne_iff_ne]
  · obtain ⟨a, ha⟩ := Option.isSome_iff_exists.1 (halloc u (live_node_ranked hinv hc hsym huw hl))
    obtain ⟨b, hb⟩ := Option.isSome_iff_exists.1
      (halloc w (live_node_ranked hinv hc hsym (Ne.symm huw) (by rw [liveU_symm]; exact hl)))
    refine ⟨a, b, ha, hb, u, w, ha, hb, ?_⟩
    rw [bprime_live hinv hc hsym hloop]; simp [huw, hl]

end CchSG
