/-
# Core: records, bags, the IR tree and its reference semantics

`Plan` is `DynTree<IR>` (planner/mod.rs:65-328) with the operator payloads
replaced by spec ids; a `Sem` gives each spec id its meaning (scans,
traversals, predicates, projections, writes are the other proof projects'
business — here they are parameters). `ev` is the reference semantics of the
IR tree: what the batched runtime (`Runtime::build_batch_op`,
runtime/runtime.rs:752-1180) computes, with batches flattened to lists and
evaluation eager per operator.

Faithful structural points (each is load-bearing for a theorem):
* every operator reads its input from child 0, and uses one empty argument row
  (`pop_or_once` / `pop_or_argument`, runtime.rs:758-765) when it has none;
* `Apply`, `SemiApply`, `AntiSemiApply`, `Optional`, `Merge`, `ForEach` have
  their input at child 0 only when they have two children
  (`children_to_recurse`, runtime.rs:654-672), else the sub-plan is child 0;
* every other operator — `PathBuilder` and `ExpandInto` included — builds
  **only child 0**; further children are never run (the `_` arm of
  `children_to_recurse`, runtime.rs:670);
* `CartesianProduct` streams child 0 and materialises every other child once,
  on the operator's own argument row, merging columns right-biased
  (cartesian_product.rs:1-40, `MergePlan`).
-/
namespace PlannerBuild

abbrev Var := Nat

/-- A record (one row of the driving table). Newest binding first; lookup is
first match, so `merge r s` lets `s` win, like the runtime's `MergePlan`. -/
abbrev Rec (V : Type) := List (Var × V)

def Rec.set {V : Type} (r : Rec V) (x : Var) (v : V) : Rec V := (x, v) :: r
def Rec.merge {V : Type} (r s : Rec V) : Rec V := s ++ r

/-- Rows plus the graph state after producing them. -/
abbrev Res (V G : Type) := List (Rec V) × G

/-- Run a per-row step over a bag, threading state, concatenating outputs in
input order. This is the driving-table "for each record" of openCypher. -/
def forRows {V G : Type} (f : Rec V → G → Res V G) : List (Rec V) → G → Res V G
  | [], g => ([], g)
  | r :: rs, g =>
    let a := f r g
    let b := forRows f rs a.2
    (a.1 ++ b.1, b.2)

/-- Fold a state change over a list (FOREACH's loop over its list). -/
def forItems {A G : Type} (f : A → G → G) : List A → G → G
  | [], g => g
  | a :: as, g => forItems f as (f a g)

section forRowsLemmas
variable {V G : Type}

@[simp] theorem forRows_nil (f : Rec V → G → Res V G) (g : G) : forRows f [] g = ([], g) := rfl

theorem forRows_cons (f : Rec V → G → Res V G) (r : Rec V) (rs : List (Rec V)) (g : G) :
    forRows f (r :: rs) g = ((f r g).1 ++ (forRows f rs (f r g).2).1, (forRows f rs (f r g).2).2) := rfl

@[simp] theorem forRows_single (f : Rec V → G → Res V G) (r : Rec V) (g : G) :
    forRows f [r] g = f r g := by
  simp [forRows]

theorem forRows_append (f : Rec V → G → Res V G) (xs ys : List (Rec V)) (g : G) :
    forRows f (xs ++ ys) g =
      ((forRows f xs g).1 ++ (forRows f ys (forRows f xs g).2).1,
       (forRows f ys (forRows f xs g).2).2) := by
  induction xs generalizing g with
  | nil => simp
  | cons x xs ih => simp [forRows, ih, List.append_assoc]

/-- Driving-table associativity for a read-only first step: feeding its
output into a second step = one per-row step doing both. (With a stateful first
step this is false — eager clause-at-a-time evaluation is not per-row
interleaving — so the stitching theorems below are all stated eagerly.) -/
theorem forRows_bind_pure (f : Rec V → G → Res V G) (k : Rec V → List (Rec V))
    (rows : List (Rec V)) (g : G) :
    forRows f (rows.flatMap k) g = forRows (fun r g => forRows f (k r) g) rows g := by
  induction rows generalizing g with
  | nil => simp
  | cons x xs ih => simp [forRows_cons, forRows_append, ih]

theorem forRows_congr (f h : Rec V → G → Res V G) (hf : ∀ r g, f r g = h r g)
    (rows : List (Rec V)) (g : G) : forRows f rows g = forRows h rows g := by
  induction rows generalizing g with
  | nil => rfl
  | cons x xs ih => simp [forRows_cons, hf, ih]

/-- A pure per-row step is a `flatMap`. -/
theorem forRows_pure (k : Rec V → List (Rec V)) (rows : List (Rec V)) (g : G) :
    forRows (fun r g => (k r, g)) rows g = (rows.flatMap k, g) := by
  induction rows generalizing g with
  | nil => rfl
  | cons x xs ih => simp [forRows_cons, ih]

/-- A read-only per-row step (it returns the state it was given) is a `flatMap`
at the incoming state. -/
theorem forRows_ro (f : Rec V → G → Res V G) (k : Rec V → G → List (Rec V))
    (hf : ∀ r g, f r g = (k r g, g)) (rows : List (Rec V)) (g : G) :
    forRows f rows g = (rows.flatMap fun r => k r g, g) := by
  induction rows with
  | nil => rfl
  | cons x xs ih => simp [forRows_cons, hf, ih]

end forRowsLemmas

/-- Which traversal operator a hop was planned as (the stitching walk cares). -/
inductive HopKind
  | condTraverse | varLen | shortest | expandInto
  deriving DecidableEq, Repr

/-- The IR operators (planner/mod.rs:65-328), payloads as spec ids. -/
inductive Op
  | argument
  | filter (p : Nat)
  | project (p : Nat)
  | aggregate (p : Nat)
  | distinct
  | sort (p : Nat)
  | skip (n : Nat)
  | limit (n : Nat)
  | unwind (e : Nat) (x : Var)
  | scan (s : Nat)
  | hop (k : HopKind) (s : Nat)
  | cartesian
  | apply
  | semiApply
  | antiSemiApply
  | orMux
  | optional (vars : List Var)
  | pathBuilder (p : Nat)
  | create (p : Nat)
  | merge (p : Nat)
  | set (p : Nat)
  | remove (p : Nat)
  | delete (p : Nat)
  | commit
  | forEach (e : Nat) (x : Var)
  | union
  | procCall (p : Nat)
  deriving DecidableEq, Repr

/-- `DynTree<IR>`. -/
inductive Plan
  | node (op : Op) (cs : List Plan)
  deriving Repr

def Plan.op : Plan → Op | .node o _ => o
def Plan.cs : Plan → List Plan | .node _ c => c

/-- Meaning of the spec ids. -/
structure Sem (V G : Type) where
  scan : Nat → Rec V → G → List (Rec V)
  hop : Nat → Rec V → G → List (Rec V)
  test : Nat → Rec V → G → Bool
  proj : Nat → Rec V → G → Rec V
  agg : Nat → Rec V → List (Rec V) → List (Rec V)
  sort : Nat → List (Rec V) → List (Rec V)
  dedup : List (Rec V) → List (Rec V)
  items : Nat → Rec V → G → List V
  path : Nat → Rec V → Rec V
  proc : Nat → Rec V → G → List (Rec V)
  create : Nat → Rec V → G → Rec V × G
  write : Nat → Rec V → G → G
  pad : List Var → Rec V → Rec V

variable {V G : Type}

/-- Per-row operators: the runtime op maps each input row independently. -/
def Sem.step (S : Sem V G) : Op → Rec V → G → Res V G
  | .filter p, r, g => (if S.test p r g then [r] else [], g)
  | .project p, r, g => ([S.proj p r g], g)
  | .unwind e x, r, g => ((S.items e r g).map (r.set x), g)
  | .scan s, r, g => (S.scan s r g, g)
  | .hop _ s, r, g => (S.hop s r g, g)
  | .pathBuilder p, r, g => ([S.path p r], g)
  | .create p, r, g => ([(S.create p r g).1], (S.create p r g).2)
  | .set p, r, g => ([r], S.write p r g)
  | .remove p, r, g => ([r], S.write p r g)
  | .delete p, r, g => ([r], S.write p r g)
  | .procCall p, r, g => (S.proc p r g, g)
  | _, r, g => ([r], g)

/-- Whole-bag operators. -/
def Sem.bag (S : Sem V G) : Op → Rec V → List (Rec V) → G → Res V G
  | .aggregate p, r, rows, g => (S.agg p r rows, g)
  | .distinct, _, rows, g => (S.dedup rows, g)
  | .sort p, _, rows, g => (S.sort p rows, g)
  | .skip n, _, rows, g => (rows.drop n, g)
  | .limit n, _, rows, g => (rows.take n, g)
  | _, _, rows, g => (rows, g)

def Op.isBag : Op → Bool
  | .aggregate _ | .distinct | .sort _ | .skip _ | .limit _ => true
  | _ => false

/-- Operators with a sub-plan child besides the input (`children_to_recurse`,
runtime.rs:666; Apply / Semi / Anti read child 1 in their ops). -/
def Op.isCorr : Op → Bool
  | .apply | .semiApply | .antiSemiApply | .optional _ | .merge _ | .forEach _ _ => true
  | _ => false

/-- Right-biased product of the left rows with materialised right branches. -/
def cpRows : List (Rec V) → List (List (Rec V)) → List (Rec V)
  | ls, [] => ls
  | ls, rs :: rss => cpRows (ls.flatMap fun l => rs.map fun r => l.merge r) rss

mutual
/-- Reference semantics of an IR tree on argument row `r` and graph state `g`. -/
def ev (S : Sem V G) : Plan → Rec V → G → Res V G
  | .node .argument _, r, g => ([r], g)
  | .node .union cs, r, g => evUnion S cs r g
  | .node .cartesian [], r, g => ([r], g)
  | .node .cartesian (c :: rs), r, g =>
    let a := ev S c r g
    let b := evRights S rs r a.2
    (cpRows a.1 b.1, b.2)
  | .node .orMux [], r, g => ([r], g)
  | .node .orMux (c :: bs), r, g =>
    let a := ev S c r g
    forRows (fun row g => (if evAny S bs row g then [row] else [], g)) a.1 a.2
  -- correlated operators: one child = sub-plan only, input is the argument row.
  -- The sub-plan's rows already carry the argument row's bindings (its
  -- leaves are `Argument` taps), so Apply's `merge_over_input` is the identity
  -- on them and is not modelled.
  | .node .apply [s], r, g => ev S s r g
  | .node .apply (x :: s :: _), r, g =>
    let a := ev S x r g
    forRows (fun row g => ev S s row g) a.1 a.2
  | .node .semiApply [s], r, g =>
    let b := ev S s r g
    (if b.1.isEmpty then [] else [r], b.2)
  | .node .semiApply (x :: s :: _), r, g =>
    let a := ev S x r g
    forRows (fun row g => let b := ev S s row g; (if b.1.isEmpty then [] else [row], b.2)) a.1 a.2
  | .node .antiSemiApply [s], r, g =>
    let b := ev S s r g
    (if b.1.isEmpty then [r] else [], b.2)
  | .node .antiSemiApply (x :: s :: _), r, g =>
    let a := ev S x r g
    forRows (fun row g => let b := ev S s row g; (if b.1.isEmpty then [row] else [], b.2)) a.1 a.2
  | .node (.optional vs) [m], r, g =>
    let b := ev S m r g
    (if b.1.isEmpty then [S.pad vs r] else b.1, b.2)
  | .node (.optional vs) (x :: m :: _), r, g =>
    let a := ev S x r g
    forRows (fun row g => let b := ev S m row g;
      (if b.1.isEmpty then [S.pad vs row] else b.1, b.2)) a.1 a.2
  | .node (.merge p) [m], r, g =>
    let b := ev S m r g
    (if b.1.isEmpty then [(S.create p r b.2).1] else b.1,
     if b.1.isEmpty then (S.create p r b.2).2 else b.2)
  | .node (.merge p) (x :: m :: _), r, g =>
    let a := ev S x r g
    forRows (fun row g => let b := ev S m row g;
      (if b.1.isEmpty then [(S.create p row b.2).1] else b.1,
       if b.1.isEmpty then (S.create p row b.2).2 else b.2)) a.1 a.2
  | .node (.forEach e x) [body], r, g =>
    ([r], forItems (fun v g => (ev S body (r.set x v) g).2) (S.items e r g) g)
  | .node (.forEach e x) (inp :: body :: _), r, g =>
    let a := ev S inp r g
    forRows (fun row g =>
      ([row], forItems (fun v g => (ev S body (row.set x v) g).2) (S.items e row g) g)) a.1 a.2
  -- every other operator: input = child 0 (or the argument row); other children ignored
  | .node op [], r, g => if op.isBag then S.bag op r [r] g else S.step op r g
  | .node op (c :: _), r, g =>
    let a := ev S c r g
    if op.isBag then S.bag op r a.1 a.2 else forRows (S.step op) a.1 a.2

/-- `Union`: every branch runs on the argument row, outputs concatenated. -/
def evUnion (S : Sem V G) : List Plan → Rec V → G → Res V G
  | [], _, g => ([], g)
  | c :: cs, r, g =>
    let a := ev S c r g
    let b := evUnion S cs r a.2
    (a.1 ++ b.1, b.2)

/-- CartesianProduct's right branches, each materialised once on the argument row. -/
def evRights (S : Sem V G) : List Plan → Rec V → G → List (List (Rec V)) × G
  | [], _, g => ([], g)
  | c :: cs, r, g =>
    let a := ev S c r g
    let b := evRights S cs r a.2
    (a.1 :: b.1, b.2)

/-- OrApplyMultiplexer's branches (read-only in every plan the planner builds). -/
def evAny (S : Sem V G) : List Plan → Rec V → G → Bool
  | [], _, _ => false
  | c :: cs, r, g => !(ev S c r g).1.isEmpty || evAny S cs r g
end

end PlannerBuild
