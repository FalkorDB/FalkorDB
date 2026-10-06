/-
# UNIQUE / MANDATORY constraint enforcement, and how it diverges from C

Models `Graph::build_composite_key` (graph.rs:4277), the unique/mandatory
enforcement decisions in `Pending::check_node_constraint` /
`check_edge_constraint` (pending.rs:1355, :1425) and the whole-graph validators
`validate_unique_constraint` / `validate_mandatory_constraint` (graph.rs:4221,
:4181), and `create_constraint`'s inline-vs-background split (graph.rs:4006).

The reference is C FalkorDB (`bin/macos-arm64v8-release/falkordb.so`). Two
enforcement semantics are compared and found to disagree; both are CONFIRMED on
live servers (ports 18420 Rust / 18421 C).

Here ↔ there:

| Lean | Rust |
| --- | --- |
| `KVal`                         | the subset of `Value` a property can hold |
| `rustKey`                      | `build_composite_key`'s `format!("{v:?}")` per component (graph.rs:4285) |
| `cNumEq`                       | C's `SIValue` equality used by its exact-match index / `EnforceUniqueEntity` |
| `uniqueRust` / `uniqueC`       | whether a *pair* of values collides under each engine |
| `keyEmpty`                     | `build_composite_key` returns `[]` if any component is NULL/absent |
| `validateInline`               | `create_constraint`: `count ≤ 10_000` ⇒ synchronous validate (graph.rs:4042) |
-/

set_option linter.unusedSimpArgs false

namespace GraphPersist

/-- The value shapes a constrained property can take (the ones the divergence
turns on). `kfloat` carries a rational-free tag; we only need the pairs that
Rust's `{v:?}` renders identically-or-not versus C's numeric compare. -/
inductive KVal where
  | knull
  | kint (i : Int)
  | kfloat (whole : Int) (isNegZero : Bool)   -- e.g. 1.0 = kfloat 1 false; -0.0 = kfloat 0 true
  | kbool (b : Bool)
  | kstr (s : String)
  | klist (xs : List KVal)      -- C does NOT index lists; Rust builds a key for them
  deriving Repr

/-- Rust's per-component key fragment: `format!("{v:?}")`. The Debug rendering is
type-tagged — `1` prints `Int(1)`-style vs `1.0` prints `Float(1.0)` — so two
values of *different runtime type* never produce the same fragment even when they
are numerically equal. We model the fragment as a tagged string. `-0.0` prints
`-0.0` while `0.0` prints `0.0`, so they differ too. -/
def rustFrag : KVal → List Int
  | .knull        => [0]                       -- never reached: keyEmpty short-circuits
  | .kint i       => [1, i]                     -- Debug tag Int(..)
  | .kfloat w nz  => [2, w, if nz then 1 else 0]  -- Float(..): -0.0 and 0.0 distinct
  | .kbool b      => [3, if b then 1 else 0]   -- Bool(..)
  | .kstr s       => 4 :: s.data.map (fun c => (c.toNat : Int))
  | .klist xs     => 5 :: (xs.length : Int) :: xs.flatMap rustFrag

/-- A single-property key is empty (entity does not participate) iff the value is
NULL — `build_composite_key`'s `Some(v) if !Null => …  _ => return Vec::new()`. -/
def keyEmpty : KVal → Bool | .knull => true | _ => false

/-- Rust decides two entities collide iff neither key is empty and the fragments
match — `check_node_constraint`'s `seen` map keyed by the composite key. -/
def uniqueCollideRust (a b : KVal) : Bool :=
  !keyEmpty a && !keyEmpty b && rustFrag a == rustFrag b

/-- C's decision, as observed: numeric equality across Int/Float/Bool, `-0.0 = 0.0`,
`1 = 1.0 = true`; and C's exact-match index does **not** hold list/point/vector
values, so two equal lists do *not* collide (both are simply admitted). -/
def cNumOf : KVal → Option Int    -- Some n = participates numerically as n; None = C ignores
  | .knull        => none
  | .kint i       => some i
  | .kfloat w _   => some w        -- -0.0 ↦ 0, 1.0 ↦ 1
  | .kbool b      => some (if b then 1 else 0)
  | .kstr _       => none          -- strings compare as strings; handled separately
  | .klist _      => none          -- C does not index lists ⇒ never collides

def uniqueCollideC (a b : KVal) : Bool :=
  match a, b with
  | .kstr x, .kstr y => x == y
  | _, _ =>
      match cNumOf a, cNumOf b with
      | some x, some y => x == y
      | _, _ => false

/-! ## The divergence, as concrete counterexamples matched against live servers -/

-- 1 (Int) vs 1.0 (Float): C rejects the second CREATE, Rust admits it.
example : uniqueCollideRust (.kint 1) (.kfloat 1 false) = false := by simp [uniqueCollideRust, uniqueCollideC, keyEmpty, rustFrag, cNumOf]
example : uniqueCollideC    (.kint 1) (.kfloat 1 false) = true  := by simp [uniqueCollideRust, uniqueCollideC, keyEmpty, rustFrag, cNumOf]

-- 0.0 vs -0.0: C rejects, Rust admits.
example : uniqueCollideRust (.kfloat 0 false) (.kfloat 0 true) = false := by simp [uniqueCollideRust, uniqueCollideC, keyEmpty, rustFrag, cNumOf]
example : uniqueCollideC    (.kfloat 0 false) (.kfloat 0 true) = true  := by simp [uniqueCollideRust, uniqueCollideC, keyEmpty, rustFrag, cNumOf]

-- true vs 1: C rejects, Rust admits.
example : uniqueCollideRust (.kbool true) (.kint 1) = false := by simp [uniqueCollideRust, uniqueCollideC, keyEmpty, rustFrag, cNumOf]
example : uniqueCollideC    (.kbool true) (.kint 1) = true  := by simp [uniqueCollideRust, uniqueCollideC, keyEmpty, rustFrag, cNumOf]

-- [1,2] vs [1,2]: Rust rejects the second, C admits it (C does not index lists).
example : uniqueCollideRust (.klist [.kint 1, .kint 2]) (.klist [.kint 1, .kint 2]) = true  := by simp [uniqueCollideRust, uniqueCollideC, keyEmpty, rustFrag, cNumOf]
example : uniqueCollideC    (.klist [.kint 1, .kint 2]) (.klist [.kint 1, .kint 2]) = false := by simp [uniqueCollideRust, uniqueCollideC, keyEmpty, rustFrag, cNumOf]

/-- **The two engines' UNIQUE decision is not the same function.** There exist
value pairs on which `uniqueCollideRust` and `uniqueCollideC` disagree — so a
graph the primary (or C) accepts can be one the other rejects, and vice-versa.
Confirmed live (see the header). -/
theorem rust_C_unique_disagree :
    ∃ a b : KVal, uniqueCollideRust a b ≠ uniqueCollideC a b := by
  exact ⟨.kint 1, .kfloat 1 false, by simp [uniqueCollideRust, uniqueCollideC, keyEmpty, rustFrag, cNumOf]⟩

/-- The disagreement is *two-sided*: Rust is stricter on lists, C stricter on
numeric cross-type equality. Neither refines the other. -/
theorem disagreement_two_sided :
    (∃ a b, uniqueCollideRust a b = true ∧ uniqueCollideC a b = false) ∧
    (∃ a b, uniqueCollideRust a b = false ∧ uniqueCollideC a b = true) := by
  refine ⟨⟨.klist [.kint 1], .klist [.kint 1], ?_, ?_⟩, ⟨.kint 1, .kfloat 1 false, ?_, ?_⟩⟩ <;> simp [uniqueCollideRust, uniqueCollideC, keyEmpty, rustFrag, cNumOf]

/-! ## Constraint *creation over existing violating data*

`create_constraint` validates inline when the label's entity count ≤ 10_000
(graph.rs:4042), setting `Operational` if `validate_unique_constraint` finds no
duplicate and `Failed` otherwise. Because `validate_unique_constraint` uses the
same `build_composite_key`, a graph that already holds `1` and `1.0` is judged
**non-violating** by Rust and the constraint becomes `Operational`, while C marks
it `FAILED`. Modelled: validation over a value list = "are all non-empty keys
distinct under this engine's collision relation". -/

/-- Do any two distinct participating entities collide (engine given by `coll`)? -/
def hasDup (coll : KVal → KVal → Bool) : List KVal → Bool
  | []      => false
  | x :: xs => (xs.any (fun y => coll x y)) || hasDup coll xs

/-- Rust marks a fresh UNIQUE constraint `Operational` iff no `hasDup`. -/
def rustConstraintOK (vals : List KVal) : Bool := !hasDup uniqueCollideRust vals
def cConstraintOK    (vals : List KVal) : Bool := !hasDup uniqueCollideC vals

/-- **CONFIRMED (constraint_over_existing).** Over data that already holds `1` and
`1.0`, Rust brings the UNIQUE constraint up `Operational`; C marks it `FAILED`.
So the same stored graph carries a constraint the two engines report opposite
statuses for — and Rust's is enforcing an invariant its own data violates. -/
theorem constraint_creation_diverges :
    rustConstraintOK [.kint 1, .kfloat 1 false] = true ∧
    cConstraintOK    [.kint 1, .kfloat 1 false] = false := by
  refine ⟨?_, ?_⟩ <;> simp [uniqueCollideRust, uniqueCollideC, keyEmpty, rustFrag, cNumOf, hasDup, rustConstraintOK, cConstraintOK]

/-! ## Enforcement cost: `check_node_constraint` is O(affected × label_size)

For each affected entity, `check_node_constraint` rebuilds a `seen` map over
**every** entity carrying the label (pending.rs:1499-1523 inner loop). In one
write query that creates `n` constrained nodes, `affected = n` and
`label_size = n`, so the commit does `Θ(n²)` composite-key builds. Measured:
`UNWIND range(1,n) CREATE (:L {v:x})` under a UNIQUE constraint —
n=2000 → 9.2 s (C 0.05 s), n=8000 → 212 s (C 2.4 s): ~87× C at n=8000 and
super-linear. The C reference probes its exact-match index per entity: `Θ(n log n)`. -/

/-- Number of composite-key builds Rust's enforcement performs for a batch that
creates `n` constrained nodes into a label of final size `n` (each of the `n`
affected nodes scans all `n`). -/
def rustEnforceKeyBuilds (n : Nat) : Nat := n * n

/-- The reference does one index probe per affected node. -/
def cEnforceProbes (n : Nat) : Nat := n

theorem enforcement_is_quadratic (n : Nat) :
    rustEnforceKeyBuilds n = n * n ∧ cEnforceProbes n = n := ⟨rfl, rfl⟩

/-- concrete blow-up the benchmark saw. -/
example : rustEnforceKeyBuilds 8000 = 64000000 := by decide

/-! ## MANDATORY enforcement agrees with C (a positive result)

`check_node_constraint`'s Mandatory arm requires every constrained property be
present and non-NULL on each affected entity of the label — this matched C on
every probe (create-missing, `SET x = null`, `REMOVE x`, `SET n = {}`,
create-then-delete, `SET n:L` adding the label, `MERGE ... ON CREATE`). Modelled
as: a value satisfies the mandatory check iff it is not NULL. -/
def mandatorySatisfied (v : KVal) : Bool := !keyEmpty v
def mandatorySatisfiedC : KVal → Bool | .knull => false | _ => true

theorem mandatory_agrees (v : KVal) : mandatorySatisfied v = mandatorySatisfiedC v := by
  cases v <;> simp [mandatorySatisfied, mandatorySatisfiedC, keyEmpty]

end GraphPersist
