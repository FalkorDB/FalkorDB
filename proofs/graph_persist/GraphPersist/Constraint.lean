/-
# UNIQUE / MANDATORY constraint enforcement, and how it diverges from C

Models `Graph::build_composite_key` (graph.rs:4866), the unique/mandatory
enforcement decisions in `Pending::check_node_constraint` /
`check_edge_constraint`, the whole-graph validators
`validate_unique_constraint` / `validate_mandatory_constraint` (graph.rs:4796,
:4757), and `create_constraint`'s inline-vs-background split (graph.rs:4582, the
`count <= 10_000` test at :4618).

The reference is C FalkorDB. Before #2991 the two UNIQUE semantics disagreed
(confirmed live; kept as `pre2991_*`: the key was `format!("{v:?}")`, type-tagged).
Since #2991 the key is C's exact-match equality: `rust_C_unique_agree`,
`composite_key_unambiguous`, `constraint_creation_agrees`.

Here ↔ there:

| Lean | Rust |
| --- | --- |
| `KVal`                         | the subset of `Value` a property can hold (`knan`: a NaN float) |
| `keyFrag`, `rustKey`           | `build_composite_key` (graph.rs:4866): `'s'`+len+bytes, or `'n'`+`f64` bits (`+ 0.0`), `[]` if vacuous |
| `uniqueCollideRust`            | equal non-empty keys (`seen` in `validate_unique_constraint`, `keys` in `unique_violated`) |
| `cNumOfF`, `uniqueCollideCF`   | C's `EnforceUniqueEntity` exact match (`toF` = `Int → double`) |
| `rustConstraintOK`             | `validate_unique_constraint` (graph.rs:4796) |
| `pre2991_rustFrag`, `pre2991_uniqueCollideRust` | the key before #2991 (`format!("{v:?}")`) |
| `validateInline`               | `create_constraint`: `count ≤ 10_000` ⇒ synchronous validate (graph.rs:4618) |
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
  | knan                        -- a NaN float (any payload)
  deriving Repr

/-- Rust's per-component key fragment: `format!("{v:?}")`. The Debug rendering is
type-tagged — `1` prints `Int(1)`-style vs `1.0` prints `Float(1.0)` — so two
values of *different runtime type* never produce the same fragment even when they
are numerically equal. We model the fragment as a tagged string. `-0.0` prints
`-0.0` while `0.0` prints `0.0`, so they differ too. -/
def pre2991_rustFrag : KVal → List Int
  | .knull        => [0]                       -- never reached: keyEmpty short-circuits
  | .kint i       => [1, i]                     -- Debug tag Int(..)
  | .kfloat w nz  => [2, w, if nz then 1 else 0]  -- Float(..): -0.0 and 0.0 distinct
  | .kbool b      => [3, if b then 1 else 0]   -- Bool(..)
  | .kstr s       => 4 :: s.toList.map (fun c => (c.toNat : Int))
  | .klist xs     => 5 :: (xs.length : Int) :: xs.flatMap pre2991_rustFrag
  | .knan         => [2, 0, 2]                  -- Float(NaN)

/-- A single-property key is empty (entity does not participate) iff the value is
NULL — `build_composite_key`'s `Some(v) if !Null => …  _ => return Vec::new()`. -/
def keyEmpty : KVal → Bool | .knull => true | _ => false

/-- Rust decides two entities collide iff neither key is empty and the fragments
match — `check_node_constraint`'s `seen` map keyed by the composite key. -/
def pre2991_uniqueCollideRust (a b : KVal) : Bool :=
  !keyEmpty a && !keyEmpty b && pre2991_rustFrag a == pre2991_rustFrag b

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
  | .knan         => none          -- NaN equals nothing

def uniqueCollideC (a b : KVal) : Bool :=
  match a, b with
  | .kstr x, .kstr y => x == y
  | _, _ =>
      match cNumOf a, cNumOf b with
      | some x, some y => x == y
      | _, _ => false

/-! ## Before #2991: the divergence, as concrete counterexamples matched against live servers

Historical: the key was `format!("{v:?}")`. Fixed by #2991 (`rust_C_unique_agree` below). -/

-- 1 (Int) vs 1.0 (Float): C rejects the second CREATE, Rust admits it.
example : pre2991_uniqueCollideRust (.kint 1) (.kfloat 1 false) = false := by simp [pre2991_uniqueCollideRust, uniqueCollideC, keyEmpty, pre2991_rustFrag, cNumOf]
example : uniqueCollideC    (.kint 1) (.kfloat 1 false) = true  := by simp [pre2991_uniqueCollideRust, uniqueCollideC, keyEmpty, pre2991_rustFrag, cNumOf]

-- 0.0 vs -0.0: C rejects, Rust admits.
example : pre2991_uniqueCollideRust (.kfloat 0 false) (.kfloat 0 true) = false := by simp [pre2991_uniqueCollideRust, uniqueCollideC, keyEmpty, pre2991_rustFrag, cNumOf]
example : uniqueCollideC    (.kfloat 0 false) (.kfloat 0 true) = true  := by simp [pre2991_uniqueCollideRust, uniqueCollideC, keyEmpty, pre2991_rustFrag, cNumOf]

-- true vs 1: C rejects, Rust admits.
example : pre2991_uniqueCollideRust (.kbool true) (.kint 1) = false := by simp [pre2991_uniqueCollideRust, uniqueCollideC, keyEmpty, pre2991_rustFrag, cNumOf]
example : uniqueCollideC    (.kbool true) (.kint 1) = true  := by simp [pre2991_uniqueCollideRust, uniqueCollideC, keyEmpty, pre2991_rustFrag, cNumOf]

-- [1,2] vs [1,2]: Rust rejects the second, C admits it (C does not index lists).
example : pre2991_uniqueCollideRust (.klist [.kint 1, .kint 2]) (.klist [.kint 1, .kint 2]) = true  := by simp [pre2991_uniqueCollideRust, uniqueCollideC, keyEmpty, pre2991_rustFrag, cNumOf]
example : uniqueCollideC    (.klist [.kint 1, .kint 2]) (.klist [.kint 1, .kint 2]) = false := by simp [pre2991_uniqueCollideRust, uniqueCollideC, keyEmpty, pre2991_rustFrag, cNumOf]

/-- **The two engines' UNIQUE decision is not the same function.** There exist
value pairs on which `pre2991_uniqueCollideRust` and `uniqueCollideC` disagree — so a
graph the primary (or C) accepts can be one the other rejects, and vice-versa.
Confirmed live (see the header). -/
theorem pre2991_rust_C_unique_disagree :
    ∃ a b : KVal, pre2991_uniqueCollideRust a b ≠ uniqueCollideC a b := by
  exact ⟨.kint 1, .kfloat 1 false, by simp [pre2991_uniqueCollideRust, uniqueCollideC, keyEmpty, pre2991_rustFrag, cNumOf]⟩

/-- The disagreement is *two-sided*: Rust is stricter on lists, C stricter on
numeric cross-type equality. Neither refines the other. -/
theorem pre2991_disagreement_two_sided :
    (∃ a b, pre2991_uniqueCollideRust a b = true ∧ uniqueCollideC a b = false) ∧
    (∃ a b, pre2991_uniqueCollideRust a b = false ∧ uniqueCollideC a b = true) := by
  refine ⟨⟨.klist [.kint 1], .klist [.kint 1], ?_, ?_⟩, ⟨.kint 1, .kfloat 1 false, ?_, ?_⟩⟩ <;> simp [pre2991_uniqueCollideRust, uniqueCollideC, keyEmpty, pre2991_rustFrag, cNumOf]

/-! ## Constraint *creation over existing violating data*

`create_constraint` validates inline when the label's entity count ≤ 10_000
(graph.rs:4537), setting `Operational` if `validate_unique_constraint` finds no
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
def pre2991_rustConstraintOK (vals : List KVal) : Bool := !hasDup pre2991_uniqueCollideRust vals
def cConstraintOK    (vals : List KVal) : Bool := !hasDup uniqueCollideC vals

/-- **CONFIRMED (constraint_over_existing).** Over data that already holds `1` and
`1.0`, Rust brings the UNIQUE constraint up `Operational`; C marks it `FAILED`.
So the same stored graph carries a constraint the two engines report opposite
statuses for — and Rust's is enforcing an invariant its own data violates. -/
theorem pre2991_constraint_creation_diverges :
    pre2991_rustConstraintOK [.kint 1, .kfloat 1 false] = true ∧
    cConstraintOK    [.kint 1, .kfloat 1 false] = false := by
  refine ⟨?_, ?_⟩ <;> simp [pre2991_uniqueCollideRust, uniqueCollideC, keyEmpty, pre2991_rustFrag, cNumOf, hasDup, pre2991_rustConstraintOK, cConstraintOK]

/-! ## Since #2991: the UNIQUE key is C's exact-match equality

`build_composite_key` (graph.rs:4866) now emits, per constrained property, either
`'s'` + the string's length (`u64` LE) + its bytes, or `'n'` + the `f64` bits of the
value folded into one numeric domain (`Bool` as `0.0`/`1.0`, `Int` via `i as f64`,
`Float` itself; `+ 0.0` folds `-0.0` into `0.0`), and returns the empty key for NULL,
an absent property, NaN or any other type. Here a string's bytes are its characters
and a non-NaN `f64` (after the fold) is the `Int` it denotes; the bit pattern of a
folded non-NaN `f64` determines it, so only equality matters. `toF` is `i as f64`
(rounding to nearest beyond 2^53), left abstract: C compares the same `double`, so the
results hold for every `toF`. -/

/-- C's numeric domain with `Int → double` conversion `toF` (`cNumOf` is `toF = id`). -/
def cNumOfF (toF : Int → Int) : KVal → Option Int
  | .kint i => some (toF i)
  | v       => cNumOf v

def uniqueCollideCF (toF : Int → Int) (a b : KVal) : Bool :=
  match a, b with
  | .kstr x, .kstr y => x == y
  | _, _ =>
      match cNumOfF toF a, cNumOfF toF b with
      | some x, some y => x == y
      | _, _ => false

theorem uniqueCollideCF_id (a b : KVal) : uniqueCollideCF id a b = uniqueCollideC a b := by
  cases a <;> cases b <;> simp [uniqueCollideCF, uniqueCollideC, cNumOfF, cNumOf]

/-- One component of `build_composite_key` (graph.rs:4872-4888); `none` is the
`_ => return Vec::new()` arm. 115 = `b's'`, 110 = `b'n'`. -/
def keyFrag (toF : Int → Int) : KVal → Option (List Int)
  | .kstr s     => some (115 :: (s.toList.length : Int) :: s.toList.map (fun c => (c.toNat : Int)))
  | .kbool b    => some [110, if b then 1 else 0]
  | .kint i     => some [110, toF i]
  | .kfloat w _ => some [110, w]
  | _           => none

/-- `build_composite_key`: the concatenation, or `[]` when a component is vacuous. -/
def rustKey (toF : Int → Int) (vs : List KVal) : List Int :=
  match vs.mapM (keyFrag toF) with
  | some fs => fs.flatten
  | none    => []

/-- Two single-property entities collide in Rust iff both keys are non-empty and equal
(`validate_unique_constraint`'s `seen` set, `unique_violated`'s `keys`). -/
def uniqueCollideRust (toF : Int → Int) (a b : KVal) : Bool :=
  !(rustKey toF [a]).isEmpty && !(rustKey toF [b]).isEmpty && rustKey toF [a] == rustKey toF [b]

theorem char_map_inj (l l' : List Char) :
    l.map (fun c => (c.toNat : Int)) = l'.map (fun c => (c.toNat : Int)) ↔ l = l' :=
  List.map_inj_right (fun a b h => Char.toNat_inj.mp (by omega))

/-- A component's fragment is C's equality: equal fragments iff C calls the values
duplicates. -/
theorem keyFrag_eq_iff (toF : Int → Int) (a b : KVal) (fa fb : List Int)
    (ha : keyFrag toF a = some fa) (hb : keyFrag toF b = some fb) :
    fa = fb ↔ uniqueCollideCF toF a b = true := by
  cases a <;> cases b <;> simp [keyFrag] at ha hb <;> subst ha <;> subst hb <;>
    simp [uniqueCollideCF, cNumOfF, cNumOf, char_map_inj, String.toList_inj] <;> try omega
  case kstr.kstr x y => rintro rfl; rfl

/-- The vacuous values are exactly the ones C never collides. -/
theorem keyFrag_none_iff (toF : Int → Int) (a : KVal) :
    keyFrag toF a = none ↔ ∀ b, uniqueCollideCF toF a b = false := by
  cases a <;> simp [keyFrag]
  all_goals first
    | (intro b; cases b <;> simp [uniqueCollideCF, cNumOfF, cNumOf])
    | skip
  case kstr s => exact ⟨.kstr s, by simp [uniqueCollideCF]⟩
  case kbool b => exact ⟨.kbool b, by simp [uniqueCollideCF, cNumOfF, cNumOf]⟩
  case kint i => exact ⟨.kint i, by simp [uniqueCollideCF, cNumOfF, cNumOf]⟩
  case kfloat w z => exact ⟨.kfloat w z, by simp [uniqueCollideCF, cNumOfF, cNumOf]⟩

/-- **`rust_C_unique_agree`** (#2991): the Rust UNIQUE decision on a single property is
C's, for every value pair (it replaces `pre2991_rust_C_unique_disagree`). -/
theorem rust_C_unique_agree (toF : Int → Int) (a b : KVal) :
    uniqueCollideRust toF a b = uniqueCollideCF toF a b := by
  cases ha : keyFrag toF a with
  | none =>
    have := (keyFrag_none_iff toF a).1 ha b
    simp [uniqueCollideRust, rustKey, ha, this]
  | some fa =>
    cases hb : keyFrag toF b with
    | none =>
      have h := (keyFrag_none_iff toF b).1 hb a
      have : uniqueCollideCF toF a b = false := by
        cases hc : uniqueCollideCF toF a b
        · rfl
        · obtain ⟨fa', hfa'⟩ : ∃ f, keyFrag toF a = some f := ⟨fa, ha⟩
          exfalso
          cases a <;> cases b <;> simp_all [uniqueCollideCF, cNumOfF, cNumOf, keyFrag]
      simp [uniqueCollideRust, rustKey, hb, this]
    | some fb =>
      have hfa : fa ≠ [] := by cases a <;> simp [keyFrag] at ha <;> subst ha <;> simp
      have hfb : fb ≠ [] := by cases b <;> simp [keyFrag] at hb <;> subst hb <;> simp
      have e := keyFrag_eq_iff toF a b fa fb ha hb
      simp only [uniqueCollideRust, rustKey, List.mapM_cons, List.mapM_nil, ha, hb]
      simp [hfa, hfb]
      cases h : uniqueCollideCF toF a b <;> simp_all

/-- No fragment is a proper prefix of another: the concatenation splits uniquely. -/
theorem keyFrag_append (toF : Int → Int) (a b : KVal) (fa fb r1 r2 : List Int)
    (ha : keyFrag toF a = some fa) (hb : keyFrag toF b = some fb) :
    fa ++ r1 = fb ++ r2 ↔ fa = fb ∧ r1 = r2 := by
  constructor
  · intro h
    cases a <;> cases b <;> simp [keyFrag] at ha hb <;> subst ha <;> subst hb <;>
      simp at h ⊢
    all_goals first
      | exact h
      | (obtain ⟨h1, h2⟩ := h
         have hl := List.append_inj h2 (by simp only [List.length_map]; omega)
         exact ⟨⟨h1, hl.1⟩, hl.2⟩)
  · rintro ⟨rfl, rfl⟩; rfl

/-- **`composite_key_unambiguous`** (#2991): two composite keys over the same properties
are equal iff the entities' values are C-duplicates property by property. -/
theorem composite_key_unambiguous (toF : Int → Int) :
    ∀ (vs ws : List KVal) (fs gs : List (List Int)),
      vs.mapM (keyFrag toF) = some fs → ws.mapM (keyFrag toF) = some gs →
      vs.length = ws.length →
      (fs.flatten = gs.flatten ↔ ((vs.zip ws).all fun p => uniqueCollideCF toF p.1 p.2) = true)
  | [], [], fs, gs, h1, h2, _ => by
    simp [List.mapM_nil] at h1 h2; subst h1; subst h2; simp
  | [], _ :: _, _, _, _, _, hl => by simp at hl
  | _ :: _, [], _, _, _, _, hl => by simp at hl
  | a :: vs, b :: ws, fs, gs, h1, h2, hl => by
    rw [List.mapM_cons] at h1 h2
    cases ha : keyFrag toF a with
    | none => rw [ha] at h1; simp at h1
    | some fa =>
    cases hb : keyFrag toF b with
    | none => rw [hb] at h2; simp at h2
    | some fb =>
    cases hvs : vs.mapM (keyFrag toF) with
    | none => rw [ha, hvs] at h1; simp at h1
    | some fs' =>
    cases hws : ws.mapM (keyFrag toF) with
    | none => rw [hb, hws] at h2; simp at h2
    | some gs' =>
    rw [ha, hvs] at h1; rw [hb, hws] at h2
    simp at h1 h2; subst h1; subst h2
    simp only [List.flatten_cons, List.zip_cons_cons, List.all_cons, Bool.and_eq_true]
    rw [keyFrag_append toF a b fa fb _ _ ha hb, keyFrag_eq_iff toF a b fa fb ha hb,
      composite_key_unambiguous toF vs ws fs' gs' hvs hws (by simpa using hl)]

/-- `validate_unique_constraint` since #2991: the label holds no duplicate. -/
def rustConstraintOK (toF : Int → Int) (vals : List KVal) : Bool :=
  !hasDup (uniqueCollideRust toF) vals

theorem hasDup_congr (f g : KVal → KVal → Bool) (h : ∀ a b, f a b = g a b) :
    ∀ l, hasDup f l = hasDup g l
  | [] => rfl
  | x :: xs => by simp [hasDup, h, hasDup_congr f g h xs]

/-- **`constraint_creation_agrees`** (#2991): creating a UNIQUE constraint over existing
data brings it up `Operational` in Rust iff C does (replaces
`pre2991_constraint_creation_diverges`). -/
theorem constraint_creation_agrees (toF : Int → Int) (vals : List KVal) :
    rustConstraintOK toF vals = !hasDup (uniqueCollideCF toF) vals := by
  simp [rustConstraintOK, hasDup_congr _ _ (rust_C_unique_agree toF)]

theorem constraint_creation_agrees_id (vals : List KVal) :
    rustConstraintOK id vals = cConstraintOK vals := by
  rw [constraint_creation_agrees, hasDup_congr _ _ uniqueCollideCF_id]; rfl

/-- The pre-#2991 divergences are gone: `1`/`1.0`, `0.0`/`-0.0`, `true`/`1` collide, and
duplicate lists or NaNs do not. -/
example (toF : Int → Int) (h : toF 1 = 1) :
    uniqueCollideRust toF (.kint 1) (.kfloat 1 false) = true ∧
    uniqueCollideRust toF (.kfloat 0 false) (.kfloat 0 true) = true ∧
    uniqueCollideRust toF (.kbool true) (.kint 1) = true ∧
    uniqueCollideRust toF (.klist [.kint 1]) (.klist [.kint 1]) = false ∧
    uniqueCollideRust toF .knan .knan = false ∧
    rustConstraintOK toF [.kint 1, .kfloat 1 false] = false := by
  simp [uniqueCollideRust, rustConstraintOK, hasDup, rustKey, keyFrag, h]

/-! ## Enforcement cost: HISTORICAL model of `check_node_constraint` before #3334

**Fixed by #3334 (4ace60ec7, closes #3330).** `check_node_constraint` (pending.rs:1580)
now calls `unique_violated` (pending.rs:217): the affected members' keys go into one set,
and each member is answered by an index lookup (`query_for`/`lookup`), or the label is
scanned once for the whole set when the index cannot answer (the closure over
`get_label_matrix(label)` at pending.rs:1634-1642). That is O(affected) lookups or
O(affected + label_size), not the product; it is modelled in `pending_commit`
(`uniqueViolated_spec`). The definitions below keep the behaviour before #3334, which
rebuilt a `seen` map over **every** entity carrying the label for each affected entity. In one
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
