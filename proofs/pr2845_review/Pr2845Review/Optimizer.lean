import Pr2845Review.Row
/-
# The boundary is invisible to `push_filters_down` (claim 2 REFUTED in general)

Model of the "Case 2" inheritance in
graph/src/planner/optimizer/push_filters_down.rs:207-249 (merged tree, and
unchanged by the PR): from a Filter, walk the ancestors; at the first
`Apply` whose left child does not contain the Filter, every variable id of
that left subtree (`collect_subtree_variables`, ids only, `HashSet<u32>`)
is "inherited"; the walk stops at `Merge`, and at nothing else. Inherited
ids are added to every child subtree that contains an `Argument` and is not
env-resetting (:252-262); a conjunct is pushed into a child whose set covers
its variables.

The PR's entry `Project` sits at the bottom of the body, below the Filter's
ancestors, and the Filter's `Argument` belongs to an `Optional` inside the
body — whose rows are that `Optional`'s left input (the entry projection's
output), not the Apply's left rows. The walk ignores both, so ids of the
OUTER left branch are believed available at that `Argument`, while the
runtime row there has the slot unbound (`body_own_unbound`).
-/
namespace Pr2845

inductive K
  | apply (leftVars : List Nat) (filterInLeft : Bool)
  /-- `Optional` with the variables its left input provides. -/
  | optional (leftOut : List Nat) (filterInLeft : Bool)
  | merge
  | other
  deriving DecidableEq, Repr

/-- push_filters_down.rs:213-249, ancestors listed from the Filter's parent up. -/
def inherited : List K → List Nat
  | [] => []
  | .apply lv inLeft :: _ => if inLeft then [] else lv
  | .merge :: _ => []
  | _ :: rest => inherited rest

/-- Proposed fix: an `Optional` whose right branch holds the Filter installs
its own argument batch (its left input), so it ends the walk and supplies
that input's variables — the same treatment `Merge` already gets. -/
def inheritedFixed : List K → List Nat
  | [] => []
  | .apply lv inLeft :: _ => if inLeft then [] else lv
  | .optional lo inLeft :: rest => if inLeft then inheritedFixed rest else lo
  | .merge :: _ => []
  | _ :: rest => inheritedFixed rest

/-- A conjunct is pushable into a child containing an `Argument` when its
variables are covered by the child's own variables plus the inherited ones. -/
def pushable (conjVars childVars inh : List Nat) : Bool :=
  conjVars.all (fun v => childVars.contains v || inh.contains v)

/-- `MATCH (x:A:B) CALL { OPTIONAL MATCH (n:A:B {id:3}) RETURN n.id AS nid }`:
`x` is id 0 of the outer scope, the body's `n` is id 0 of the body scope.
EXPLAIN (merged): Apply(Scan x, Project(Optional(Project[](Argument),
Scan n(Filter(Argument))))). Before the push the Filter sat between the
Optional and the scan; its ancestors are the Optional (right branch, left
input = the empty entry projection), the RETURN Project, the Apply. The
child (the scan subtree) contains an `Argument`, so `n.id = 3` is pushed
under the scan, onto the Optional's `Argument`. -/
def callOptAncestors : List K := [.optional [] false, .other, .apply [0] false]

theorem filter_pushed_through_boundary :
    inherited callOptAncestors = [0] ∧ pushable [0] [] (inherited callOptAncestors) = true := by
  decide

/-- At that `Argument` the row is the Optional's left input, i.e. the entry
projection's output: slot 0 is unbound for every outer row, so the pushed
`n.id` read fails ("Variable n not found"; C returns `[[1,3],[2,3],[3,3]]`). -/
theorem pushed_read_unbound {V : Type} (r : RowF V) :
    bodyInputNew (importProj [] 1 []).2 r 0 = none := rfl

/-- The fix keeps the conjunct above (`n` is not available there) … -/
theorem fixed_keeps_filter :
    inheritedFixed callOptAncestors = [] ∧ pushable [0] [] (inheritedFixed callOptAncestors) = false := by
  decide

/-- … and changes nothing on a plan with no `Optional` right branch between
the Filter and its Apply/Merge. -/
def NoOpt : List K → Prop
  | [] => True
  | .apply _ _ :: _ | .merge :: _ => True
  | .optional _ inLeft :: rest => inLeft = true ∧ NoOpt rest
  | .other :: rest => NoOpt rest

theorem fixed_conservative : ∀ (as : List K), NoOpt as → inheritedFixed as = inherited as
  | [], _ => rfl
  | .apply _ _ :: _, _ => rfl
  | .merge :: _, _ => rfl
  | .optional _ inLeft :: rest, h => by
    obtain ⟨hl, hr⟩ := h
    simp [inheritedFixed, inherited, hl, fixed_conservative rest hr]
  | .other :: rest, h => fixed_conservative rest h

/-- With the fix, what an Optional's right branch inherits is exactly what
its left input binds; under the PR's entry projection that is the import
targets, whose slots `body_import_value` shows carry the outer values. -/
theorem fixed_optional_inherits_left (lo : List Nat) (rest : List K) :
    inheritedFixed (.optional lo false :: rest) = lo := rfl

end Pr2845
