import PlannerBuild.Clauses
/-
# Whole queries, and the MATCH-specific findings

* `query_correct`: if every clause plan implements its clause (the theorems of
  Clauses.lean), the plan `plan_query` stitches evaluates to the openCypher
  clause-sequence semantics `Fₙ ∘ … ∘ F₂ ∘ ⟦c₁⟧`.
* `selfloop_ignores_chain`: `ExpandInto(scan, res)` never runs `res` — the shape of an earlier
  planner, kept as a historical counterexample. The planner builds
  `ExpandInto(Scan(res))` (mod.rs:2527-2561), `selfloop_scan_keeps_chain`.
* `cp_stitch_loses_correlation` / `cp_stitch_sound`: a MATCH plan
  prepended to the next MATCH's CartesianProduct (mod.rs:3112-3120) is
  correct only when no other component reads its variables. The MATCH arm
  Apply-wraps a multi-component MATCH that reads a bound variable
  (`PathBind.planMatchArm_joined_wrapped`), so it is not stitched this way.
* A MATCH after a MATCH that binds none of the bound variables is wrapped as a
  one-branch CartesianProduct (`join_as_cartesian_branch`) and the earlier clause becomes
  its other branch: `disjoint_match_cp`, `disjoint_match_where_cp`, `disjoint_match_sound`
  (`disjoint_match_unwrapped` is the unwrapped shape, kept as a counterexample).
* `where_scan_indistinguishable`: `MATCH (a) WHERE φ MATCH (a)-->(b)`
  stitches into the exact tree shape `is_planner_scan_subtree`
  (optimizer/select_scan_node.rs:377) treats as prunable. The rebuild carries the spine's
  Filters along (`filters_of`, select_scan_node.rs:312, re-wrapped by `make_scan_subtree` :214) —
  `rebuild_keeps_where`.
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

/-! ## Self-loop hop after other hops -/

/-- `ExpandInto(scan, res)`: the runtime builds child 0 only. Whatever `res`
is — every hop of the component planned so far — the result is the same. -/
theorem selfloop_ignores_chain (s : Nat) (scan res res' : Plan) (r : Rec V) (g : G) :
    ev S (.node (.hop .expandInto s) [scan, res]) r g =
      ev S (.node (.hop .expandInto s) [scan, res']) r g := rfl

/-- **#2999 fixed by #3102 (61284ea4f)**: the planner now builds `ExpandInto(Scan(res))`
(PlanMatch2 `chainRel_selfloop_unbound`): the chain planned so far is the scan's input, and the
rows of that evaluation are fed to the scan and then the expansion operators (what `S.step` does
with them is abstract here: the theorem is the structural identity, not that `S.step` keeps them). -/
theorem selfloop_scan_keeps_chain (s sc : Nat) (res : Plan) (r : Rec V) (g : G) :
    ev S (.node (.hop .expandInto s) [.node (.scan sc) [res]]) r g =
      forRows (S.step (.hop .expandInto s)) (forRows (S.step (.scan sc)) (ev S res r g).1 (ev S res r g).2).1
        (forRows (S.step (.scan sc)) (ev S res r g).1 (ev S res r g).2).2 := by
  simp [ev, Op.isBag]

/-- In particular an empty chain now yields nothing (before, `res` was ignored). -/
theorem selfloop_scan_empty_chain (s sc : Nat) (res : Plan) (r : Rec V) (g : G)
    (h : ev S res r g = ([], g)) : ev S (.node (.hop .expandInto s) [.node (.scan sc) [res]]) r g = ([], g) := by
  rw [selfloop_scan_keeps_chain, h]; rfl

/-! ## A MATCH stitched as a CartesianProduct branch -/

/-- `insertStep` at a CartesianProduct with a plan that does not need wrapping
(its root is a scan / traversal / Filter over one, mod.rs:3233-3266) makes it
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

/-! ## A disjoint MATCH becomes a CartesianProduct branch -/

section disjoint
set_option linter.unusedSimpArgs false

/-- `MATCH (a) MATCH (b)` before #2981: the second scan is stitched *over* the first
(`Scan_b(Scan_a)`), a shape the scan selection could later re-select and drop. -/
theorem disjoint_match_unwrapped (s t : Nat) :
    rustStitch [.node (.scan s) [], .node (.scan t) []] = .node (.scan t) [.node (.scan s) []] := by
  simp [rustStitch, stitchLoop, slotFirst, slotLoop, slotWith, walkFirst, walkLoop, Plan.get, projDescend,
    descendClause, descendOne, Plan.depth, insertStep, insertAt, Plan.modify, Plan.push0, firstPass,
    isApply, clauseMin, applyChain]

/-- Since #2981 the second clause is wrapped (`join_as_cartesian_branch`) and the first one
becomes the other branch: `CartesianProduct(Scan_a, Scan_b)`. -/
theorem disjoint_match_cp (rb : Plan → Bool) (s t : Nat) (hrb : rb (.node (.scan t) []) = false) :
    rustStitch (wrapDisjoint rb [(.node (.scan s) [], false), (.node (.scan t) [], true)]) =
      .node .cartesian [.node (.scan s) [], .node (.scan t) []] := by
  simp [wrapDisjoint, joinAsCartesianBranch, branchWalk, hrb, rustStitch, stitchLoop, slotFirst, slotLoop,
    slotWith, walkFirst, walkLoop, Plan.get, projDescend, descendClause, descendOne, Plan.depth, insertStep,
    insertAt, Plan.modify, Plan.push0, firstPass, isApply, clauseMin, applyChain, needsApplyWrapping]

/-- With a WHERE on the second clause the product goes below it, so the WHERE reads both:
`MATCH (a) MATCH (b) WHERE φ` → `Filter_φ(CartesianProduct(Scan_a, Scan_b))`. -/
theorem disjoint_match_where_cp (rb : Plan → Bool) (s t f : Nat) (hrb : rb (.node (.scan t) []) = false) :
    rustStitch (wrapDisjoint rb [(.node (.scan s) [], false), (.node (.filter f) [.node (.scan t) []], true)]) =
      .node (.filter f) [.node .cartesian [.node (.scan s) [], .node (.scan t) []]] := by
  simp [wrapDisjoint, joinAsCartesianBranch, branchWalk, hrb, rustStitch, stitchLoop, slotFirst, slotLoop,
    slotWith, walkFirst, walkLoop, Plan.get, projDescend, descendClause, descendOne, Plan.depth, insertStep,
    insertAt, Plan.modify, Plan.push0, firstPass, isApply, clauseMin, applyChain, needsApplyWrapping]

/-- And the wrapped plan means the product of the two clauses: each row of the first,
merged with each row of the second (`cp_stitch_sound`; the second reads none of the first). -/
theorem disjoint_match_sound (s t : Nat) (k : List (Rec V))
    (ht : ∀ row g, ev S (.node (.scan t) []) row g = (k, g)) (r : Rec V) (g : G) :
    ev S (.node .cartesian [.node (.scan s) [], .node (.scan t) []]) r g =
      ((ev S (.node (.scan s) []) r g).1.flatMap (fun l => k.map l.merge), g) :=
  cp_stitch_sound S _ _ k ht (fun _ _ => by simp [ev, Op.isBag, Sem.step, forRows_single]) r g

/-- A subtree that reads a bound variable is left as it is. -/
theorem joinAsCartesianBranch_reads (rb : Plan → Bool) (p q : Plan) (hq : p.get (branchWalk p) = some q)
    (h : rb q = true) : joinAsCartesianBranch rb p = p := by
  simp [joinAsCartesianBranch, hq, h]

end disjoint

/-! ## The WHERE of the previous MATCH looks like a planner scan -/

/-- `is_planner_scan_subtree` (select_scan_node.rs:377-391). -/
def isPlannerScanSubtree : Plan → Bool
  | .node (.scan _) _ => true
  | .node (.filter _) [c] => isPlannerScanSubtree c
  | _ => false

/-- Stitching `Filter(φ, Scan a)` (the plan of `MATCH (a) WHERE φ`) below the
next MATCH's leaf traversal gives a traversal whose child is a planner scan
subtree — the optimizer prunes it as its own (select_scan_node.rs:1162) and
the WHERE disappears. -/
theorem where_scan_indistinguishable (φ s t : Nat) :
    let prev := Plan.node (.filter φ) [.node (.scan s) []]
    let next := Plan.node (.hop .condTraverse t) []
    fillAt next (slotLoop next) prev = .node (.hop .condTraverse t) [prev] ∧
      isPlannerScanSubtree prev = true := by
  refine ⟨?_, rfl⟩
  have : slotLoop (Plan.node (.hop .condTraverse t) []) = [] := by
    simp [slotLoop, slotWith, walkLoop, Plan.get, projDescend]
    exact descendClause_nil _ rfl
  rw [this, fillAt, insertStep_nonCP _ [] _ _ [] rfl (by simp)]; rfl

/-! ### The rebuilt scan keeps the spine's Filters -/

/-- `filters_of` (select_scan_node.rs:312-327): the `Filter`s on a scan subtree's single-child
spine, outermost first. -/
def spineFilters : Plan → List Nat
  | .node (.filter φ) [c] => φ :: spineFilters c
  | .node (.filter φ) _ => [φ]
  | .node _ [c] => spineFilters c
  | .node _ _ => []

/-- `make_scan_subtree`'s tail (select_scan_node.rs:276-299): wrap the new scan in the salvaged
filters, innermost first, so the original nesting is preserved. -/
def wrapFilters (fs : List Nat) (scan : Plan) : Plan :=
  fs.foldr (fun φ p => .node (.filter φ) [p]) scan

/-- A `Filter` with no child or several is still collected (`filters_of` pushes it before the
single-child test), then the walk stops. -/
theorem spineFilters_filter_branch (φ : Nat) (a b : Plan) (cs : List Plan) :
    spineFilters (.node (.filter φ) (a :: b :: cs)) = [φ] := rfl

theorem spineFilters_wrap (fs : List Nat) (s : Nat) :
    spineFilters (wrapFilters fs (.node (.scan s) [])) = fs := by
  induction fs with
  | nil => rfl
  | cons φ fs ih => simp only [wrapFilters, List.foldr_cons] at ih ⊢; simp [spineFilters, ih]

/-- **Bug 3 fixed**: pruning `Filter(φ, Scan a)` and rebuilding the scan from another endpoint
keeps `φ` — every Filter of the old spine is on the new one, in order. -/
theorem rebuild_keeps_where (φ s s' : Nat) :
    let prev := Plan.node (.filter φ) [.node (.scan s) []]
    spineFilters (wrapFilters (spineFilters prev) (.node (.scan s') [])) = spineFilters prev ∧
      spineFilters prev = [φ] := by
  exact ⟨spineFilters_wrap _ _, rfl⟩

end PlannerBuild
