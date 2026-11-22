/-
# The boundary is invisible to `push_filters_down` (claim 2 REFUTED in general)

Model of the "Case 2" inheritance in
graph/src/planner/optimizer/push_filters_down.rs as of 343e077f2
(`inheritedPre3281`; the current walk, push_filters_down.rs, is `inherited`): from a Filter, walk the ancestors; at the first
`Apply` whose left child does not contain the Filter, every variable id of
that left subtree (`collect_subtree_variables`, ids only, `HashSet<u32>`)
is "inherited"; the walk stops at the first `Apply` (inheriting its left variables, or
nothing when the Filter sits in its left child) and at `Merge` (inheriting nothing), and at
nothing else. Inherited
ids are added to every child subtree that contains an `Argument` and is not
env-resetting (:252-262); a conjunct is pushed into a child whose set covers
its variables.

The PR's entry `Project` sits at the bottom of the body, below the Filter's
ancestors, and the Filter's `Argument` belongs to an `Optional` inside the
body — whose rows are that `Optional`'s left input (the entry projection's
output), not the Apply's left rows. The walk ignores both, so ids of the
OUTER left branch are believed available at that `Argument`, while the
runtime row there has the slot unbound (proofs/binder
`FalkorBinder.Pr2845.body_own_unbound`, `pushed_read_unbound`). W6-opt-1.
Folded in from the retired proofs/pr2845_review.
-/
namespace Falkor.Opt.Pr2845

inductive K
  | apply (leftVars : List Nat) (filterInLeft : Bool)
  /-- `Optional` with the variables its left input provides. -/
  | optional (leftOut : List Nat) (filterInLeft : Bool)
  | merge
  | other
  deriving DecidableEq, Repr

/-- The walk before #3281 (343e077f2, push_filters_down.rs), ancestors
listed from the Filter's parent up: an `Optional` is walked through. -/
def inheritedPre3281 : List K → List Nat
  | [] => []
  | .apply lv inLeft :: _ => if inLeft then [] else lv
  | .merge :: _ => []
  | _ :: rest => inheritedPre3281 rest

/-- The walk since #3281 (push_filters_down.rs): a two-child `Optional`
(one with a left input, `.optional`) ends it like an `Apply`, inheriting its left
branch's `branch_visible_variables` (`lv`/`leftOut`), or nothing when the Filter sits in
its left child. A single-child `Optional` is `.other` (walked through). -/
def inherited : List K → List Nat
  | [] => []
  | .apply lv inLeft :: _ => if inLeft then [] else lv
  | .optional lo inLeft :: _ => if inLeft then [] else lo
  | .merge :: _ => []
  | _ :: rest => inherited rest

/-- Proposed fix: an `Optional` whose right branch holds the Filter installs
its own argument batch (its left input), so the walk ends at the `Optional` and supplies
its left input's variables. (`Merge` also ends the walk, but supplies none: `inherited` returns
`[]` there.) -/
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

/-- Historical (W6-opt-1, fixed by #3281): the pre-PR walk passes the Optional
and pushes `n.id = 3` onto the Optional's `Argument`. -/
theorem filter_pushed_through_boundary :
    inheritedPre3281 callOptAncestors = [0] ∧
      pushable [0] [] (inheritedPre3281 callOptAncestors) = true := by
  decide

/-- #3281: the walk ends at the Optional, whose left input (the empty entry
projection) exposes nothing; the conjunct stays above. -/
theorem keeps_filter_main :
    inherited callOptAncestors = [] ∧ pushable [0] [] (inherited callOptAncestors) = false := by
  decide

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

theorem fixed_conservative : ∀ (as : List K), NoOpt as → inheritedFixed as = inheritedPre3281 as
  | [], _ => rfl
  | .apply _ _ :: _, _ => rfl
  | .merge :: _, _ => rfl
  | .optional _ inLeft :: rest, h => by
    obtain ⟨hl, hr⟩ := h
    simp [inheritedFixed, inheritedPre3281, hl, fixed_conservative rest hr]
  | .other :: rest, h => fixed_conservative rest h

/-- With the fix, what an Optional's right branch inherits is exactly what
its left input binds. True by definition (`rfl`): it only restates `inheritedFixed`. The
further claim that, under the PR's entry projection, those are the import targets whose
slots carry the outer values is `body_import_value` in proofs/binder, which this project
does not import, so that link is unchecked here. -/
theorem fixed_optional_inherits_left (lo : List Nat) (rest : List K) :
    inheritedFixed (.optional lo false :: rest) = lo := rfl

/-- #3281 implements the proposed fix, except that a Filter in an Optional's
*left* child inherits nothing (Rust breaks) where the fix walks on: #3281 never
inherits more than the fix, and agrees with it on every walk with no such step. -/
theorem main_sub_fixed : ∀ (as : List K) (v : Nat), v ∈ inherited as → v ∈ inheritedFixed as
  | [], _, h => h
  | .apply _ _ :: _, _, h => h
  | .merge :: _, _, h => h
  | .optional _ inLeft :: _, _, h => by
    cases inLeft <;> simp_all [inherited, inheritedFixed]
  | .other :: rest, v, h => main_sub_fixed rest v h

def NoOptLeft : List K → Prop
  | [] => True
  | .apply _ _ :: _ | .merge :: _ => True
  | .optional _ inLeft :: _ => inLeft = false
  | .other :: rest => NoOptLeft rest

theorem main_eq_fixed : ∀ (as : List K), NoOptLeft as → inherited as = inheritedFixed as
  | [], _ => rfl
  | .apply _ _ :: _, _ => rfl
  | .merge :: _, _ => rfl
  | .optional _ inLeft :: _, h => by simp_all [NoOptLeft, inherited, inheritedFixed]
  | .other :: rest, h => main_eq_fixed rest h

end Falkor.Opt.Pr2845
