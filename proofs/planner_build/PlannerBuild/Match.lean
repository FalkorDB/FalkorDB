import PlannerBuild.Clauses
/-
# Whole queries, and the MATCH-specific findings

* `query_correct`: if every clause plan implements its clause (the theorems of
  Clauses.lean), the plan `plan_query` stitches evaluates to the openCypher
  clause-sequence semantics `Fₙ ∘ … ∘ F₂ ∘ ⟦c₁⟧`.
* `selfloop_ignores_chain` (bug 4): `ExpandInto(scan, res)` (mod.rs:2144-2152)
  never runs `res`.
* `cp_stitch_loses_correlation` (bug 2) / `cp_stitch_sound`: a MATCH plan
  prepended to the next MATCH's CartesianProduct (mod.rs:2704-2708) is
  correct only when no other component reads its variables.
* `where_scan_indistinguishable` (bug 3): `MATCH (a) WHERE φ MATCH (a)-->(b)`
  stitches into the exact tree shape `is_planner_scan_subtree`
  (optimizer/select_scan_node.rs:239) treats as prunable.
-/
namespace PlannerBuild

variable {V G : Type} (S : Sem V G)

/-! ## Composition -/

/-- A clause plan together with its meaning at the slot of walk `w`. -/
def Chain (w : Plan → Path) : List (Plan × (Rec V → Res V G → Res V G)) → Prop
  | [] => True
  | c :: cs => Implements S w (fun _ => True) c.1 c.2 ∧ Chain w cs

/-- The composed meaning, last clause outermost (the list is in reverse
clause order, as `nestLoop` takes it). -/
def compose (first : Rec V → G → Res V G) :
    List (Rec V → Res V G → Res V G) → Rec V → G → Res V G
  | [], r, g => first r g
  | F :: Fs, r, g => F r (compose first Fs r g)

theorem nestLoop_correct (ps : List (Plan × (Rec V → Res V G → Res V G))) (p0 : Plan)
    (h : Chain S walkLoop ps) (r : Rec V) (g : G) :
    ev S (nestLoop (ps.map Prod.fst ++ [p0])) r g = compose (ev S p0) (ps.map Prod.snd) r g := by
  induction ps generalizing r g with
  | nil => rfl
  | cons c cs ih =>
    obtain ⟨hc, hcs⟩ := h
    cases cs with
    | nil =>
      show ev S (refill c.1 (slotWith walkLoop c.1) p0 (nestLoop [p0])) r g = _
      rw [hc p0 _ r g trivial]; rfl
    | cons d ds =>
      show ev S (refill c.1 (slotWith walkLoop c.1) d.1 (nestLoop (d.1 :: (ds.map Prod.fst ++ [p0])))) r g = _
      rw [hc d.1 _ r g trivial]
      show c.2 r (ev S (nestLoop ((d :: ds).map Prod.fst ++ [p0])) r g) = _
      rw [ih hcs]; rfl

/-- **Query correctness.** Clause plans `p₁ … pₙ` (clause order): if the last
implements `Fₙ` at the first-walk slot and every middle one implements its
`Fᵢ` at the loop-walk slot, the stitched plan (`plan_query`) evaluates to
`Fₙ ∘ … ∘ F₂ ∘ ⟦p₁⟧`. -/
theorem query_correct (p1 : Plan) (mid : List (Plan × (Rec V → Res V G → Res V G)))
    (last : Plan) (Fl : Rec V → Res V G → Res V G)
    (hl : Implements S walkFirst (fun _ => True) last Fl) (hm : Chain S walkLoop mid)
    (r : Rec V) (g : G) :
    ev S (rustStitch (p1 :: (mid.map Prod.fst).reverse ++ [last])) r g =
      Fl r (compose (ev S p1) (mid.map Prod.snd) r g) := by
  rw [rustStitch_eq_nestStitch]
  unfold nestStitch
  have hrev : (p1 :: (mid.map Prod.fst).reverse ++ [last]).reverse =
      last :: (mid.map Prod.fst ++ [p1]) := by simp
  rw [hrev]
  cases hmid : mid.map Prod.fst ++ [p1] with
  | nil => simp at hmid
  | cons q rest =>
    show ev S (refill last (slotWith walkFirst last) q (nestLoop (q :: rest))) r g = _
    rw [hl q _ r g trivial, ← hmid, nestLoop_correct S mid p1 hm]

/-! ## Bug 4: self-loop hop after other hops -/

/-- `ExpandInto(scan, res)`: the runtime builds child 0 only. Whatever `res`
is — every hop of the component planned so far — the result is the same. -/
theorem selfloop_ignores_chain (s : Nat) (scan res res' : Plan) (r : Rec V) (g : G) :
    ev S (.node (.hop .expandInto s) [scan, res]) r g =
      ev S (.node (.hop .expandInto s) [scan, res']) r g := rfl

/-! ## Bug 2: a MATCH stitched as a CartesianProduct branch -/

/-- `insertStep` at a CartesianProduct with a plan that does not need wrapping
(its root is a scan / traversal / Filter over one, mod.rs:2769-2789) makes it
the first branch. -/
theorem cp_stitch_shape (cs : List Plan) (n : Plan) (h : needsApplyWrapping n = false) :
    fillAt (.node .cartesian cs) [] n = .node .cartesian (n :: cs) := by
  unfold fillAt
  rw [insertStep_CP _ [] n cs rfl, h]; rfl

/-- The branches are evaluated on the operator's argument row, not on the
rows of the new first branch: -/
theorem cp_eval (n c : Plan) (r : Rec V) (g : G) :
    ev S (.node .cartesian [n, c]) r g =
      (cpRows (ev S n r g).1 [(ev S c r (ev S n r g).2).1], (ev S c r (ev S n r g).2).2) := rfl

/-- Sound case: if the other component reads nothing the first one binds
(read-only and independent of the incoming record), the product is the
correlated join the clause sequence means. -/
theorem cp_stitch_sound (n c : Plan) (k : List (Rec V))
    (hc : ∀ row g, ev S c row g = (k, g)) (hn : ∀ r g, (ev S n r g).2 = g) (r : Rec V) (g : G) :
    ev S (.node .cartesian [n, c]) r g =
      ((ev S n r g).1.flatMap (fun l => k.map l.merge), g) := by
  rw [cp_eval, hc, hn]; rfl

/-- Counterexample: records map variable 0 (`a`) to a node id; the first
clause binds `a := 1` (the node with `v = 0` in the Rust repro); the other
component `(a)-[:R]->(b)` was planned as a traversal from `a` "already bound",
but under the product it sees only the argument row and re-binds `a` from
every node with an `R` edge (here node 2, which binds `b := 3`). -/
def cexS : Sem Nat Unit where
  scan := fun _ _ _ => [[(0, 1)]]
  hop := fun _ row _ => match row.lookup 0 with
    | some 1 => []                              -- node 1 has no outgoing R
    | _ => [[(1, 3), (0, 2)]]                   -- unbound `a`: every R edge
  test := fun _ _ _ => true
  proj := fun _ r _ => r
  agg := fun _ _ rows => rows
  sort := fun _ rows => rows
  dedup := id
  items := fun _ _ _ => []
  path := fun _ r => r
  proc := fun _ _ _ => []
  create := fun _ r g => (r, g)
  write := fun _ _ g => g
  pad := fun _ r => r

/-- The correlated meaning: extend each record of clause 1 by its matches. -/
def correlated (n c : Plan) (r : Rec Nat) : List (Rec Nat) :=
  (ev cexS n r ()).1.flatMap fun row => (ev cexS c row ()).1

theorem cp_stitch_loses_correlation :
    let n := Plan.node (.scan 0) []
    let c := Plan.node (.hop .condTraverse 0) []
    correlated n c [] = [] ∧ (ev cexS (.node .cartesian [n, c]) [] ()).1 ≠ [] := by
  decide

/-! ## Bug 3: the WHERE of the previous MATCH looks like a planner scan -/

/-- `is_planner_scan_subtree` (select_scan_node.rs:239-252). -/
def isPlannerScanSubtree : Plan → Bool
  | .node (.scan _) _ => true
  | .node (.filter _) [c] => isPlannerScanSubtree c
  | _ => false

/-- Stitching `Filter(φ, Scan a)` (the plan of `MATCH (a) WHERE φ`) below the
next MATCH's leaf traversal gives a traversal whose child is a planner scan
subtree — the optimizer prunes it as its own (select_scan_node.rs:1186) and
the WHERE disappears. -/
theorem where_scan_indistinguishable (φ s t : Nat) :
    let prev := Plan.node (.filter φ) [.node (.scan s) []]
    let next := Plan.node (.hop .condTraverse t) []
    fillAt next (slotLoop next) prev = .node (.hop .condTraverse t) [prev] ∧
      isPlannerScanSubtree prev = true := by
  refine ⟨?_, rfl⟩
  have : slotLoop (Plan.node (.hop .condTraverse t) []) = [] := by
    simp [slotLoop, slotWith, walkLoop, Plan.get, projDescend, descendClause_nil, descendOne]
    exact descendClause_nil _ rfl
  rw [this, fillAt, insertStep_nonCP _ [] _ _ [] rfl (by simp)]; rfl

end PlannerBuild
