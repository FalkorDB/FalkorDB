import FalkorValueOrder.Glue
import FalkorValueOrder.Json
/-
# Value order: what `compare_value` promises, and where it breaks

A model of FalkorDB-rs's value comparison (`Value::compare_value`), equality
(`PartialEq for Value`), hashing (`Hash for Value` fed to `FxHasher`) and the
comparators built on them (ORDER BY, DISTINCT, grouping, `=`/`<>`), with
machine-checked proofs of what holds and `decide`-checked counterexamples of
what does not.

## What is modelled, and where it lives in the tree

| here | there |
| --- | --- |
| `V`                      | `enum Value` (`graph/src/runtime/value.rs:180`) — Null, Bool, Int, Float, String, List, Map, Node |
| `F`                      | `f64`, restricted to NaN and integral values (see "gaps") |
| `Tag`, `order`           | `impl OrderedEnum for Value` (`value.rs:1177`) — all 16 variants |
| `roundF`                 | `i as f64` for `i: i64` (round-to-nearest, ties-to-even at 53 bits) |
| `cmpF`                   | `compare_floats` (`value.rs:1801`) — `partial_cmp`, NaN ↦ `(Less, NaN)` |
| `cmpV`                   | `impl CompareValue for Value::compare_value` (`value.rs:1224`) |
| `St`, `step`, `listLoop`, `listFinish` | `Value::compare_list` (`value.rs:1399`) — the four counters, line by line |
| `keysCmp`, `mapVals`     | `Value::compare_map` (`value.rs:1472`), maps held with keys already sorted |
| `eqV`                    | `impl PartialEq for Value` (`value.rs:1215`) — `compare_value(..).0 == Equal` |
| `sortCmp`                | ORDER BY: `OrderedKey::cmp_key` (`ops/sort.rs:76`), `Column::compare_at` (`batch.rs:408`) |
| `floatLane`              | ORDER BY float lane (`ops/sort.rs:347`): `partial_cmp(..).unwrap_or(Less)` |
| `evalEq`, `evalNeq`      | `compare_values` `CmpOp::Eq`/`Neq` (`runtime/vectorized.rs:239`) |
| `hwords`, `floatWord`    | `impl Hash for Value` (`value.rs:822`) as the word stream it feeds a `Hasher` |
| `fxAdd`, `fxFinish`, `fx`| `rustc_hash::FxHasher` 2.1.3 (`add_to_hash`, `finish` = `rotate_left(26)`) |
| `rowHash`                | `DistinctOp::next` (`ops/distinct.rs:79`), `ValuesDeduper::is_seen` (`value.rs:1869`) |
| `minFold`                | `min_batch` (`functions/aggregation.rs:487`) |

## Findings (each one reproduced against the real engine, see REPORT.md)

1. `nan_breaks_antisymmetry` — NaN is `Less` than 1.0 *and* 1.0 is `Less` than
   NaN. `slice::sort_by` (Rust ≥ 1.81) detects this and panics; the panic kills
   `redis-server`. Twenty-one floats with a NaN among them in `ORDER BY` do it.
2. `mixed_numeric_not_transitive` — `2^53 = 2^53.0 = 2^53+1` but `2^53 < 2^53+1`.
   Same panic from `ORDER BY` over a mixed int/float column near 2^53.
3. `map_order_not_transitive` — `compare_map` answers `Equal` at the first value
   pair that is disjoint or null, without looking further. Same panic.
4. `map_neq_both_false` — hence `{a:1} = {a:'x'}` and `{a:1} <> {a:'x'}` are
   both `false` (C and openCypher: `<>` is `true`).
5. `hash_not_consistent_with_eq` — `2^53+1 == 2^53.0` but their hashes differ.
6. `nan_eq_irreflexive` — `NaN != NaN` under `PartialEq`, but NaNs hash
   alike; grouping (`HashMap<GroupKey,_>`, `aggregate.rs:59`) needs a
   reflexive `Eq`, so every NaN row becomes its own group.
7. `fx_row_collision`, `fx_collision_general` — DISTINCT, `count(DISTINCT)` and
   the MERGE pattern cache keep *only* the 64-bit Fx hash. Fx is linear, so a
   collision is one subtraction away: rows `(a, b)` and `(a+1, b - K²)` always
   collide, and the second one is silently dropped (or, for MERGE, never
   created).
8. `min_nan_picks_larger` — `min([1.0, NaN, 3.0])` is `3.0`.

## What is proved to hold

* `order_injective`, `no_tag_between_int_and_float` — the type order is a
  strict order on tags and nothing sits between Int and Float, which is what
  lets numeric cross-type comparison coexist with the tag order.
* `cmp_sane_eq_key` — on the *sane* scalar fragment (no NaN, every Int exactly
  representable as f64) `compare_value` is a lexicographic compare of a key;
  hence `sane_refl`, `sane_swap`, `sane_le_trans`: a total preorder.
* `roundF_exact` — every |i| < 2^53 is exactly representable, so the sane
  fragment contains every Int a user normally meets.
* `sane_hash_consistent` — on that fragment `a == b ⇒ hash a = hash b`.
* `list_fst_eq_lex` — the ORDER-BY half of `compare_list` *is* plain
  lexicographic order (first unequal element, then length), whatever the four
  counters do; `lex_swap`, `lex_le_trans` then lift any total preorder on the
  elements to lists — so lists are not where ORDER BY goes wrong; maps are.
* `floatLane_eq_cmpV` — the float fast lane of ORDER BY is the same comparator,
  so fixing `compare_floats` fixes both.

## Gaps (what the model does NOT cover)

* Floats are NaN or integral. Fractional floats cannot create new
  int/float collisions (an Int never rounds to a fraction), but their hash path
  (`to_bits`) is modelled only for integral values.
* `-0.0` is identified with `0.0` (they compare Equal and hash alike in Rust,
  checked by the Rust test instead).
* Strings are `Nat`s compared numerically (Rust: byte order; any total order
  gives the same laws). Their hash words are abstracted.
* Path, Relationship, Point, VecF32 and the temporals appear only as `Tag`s for
  the type order; `Path` shares `compare_list`, temporals share Int's `cmp`.
* Maps are held with sorted keys and one value per key (`compare_map` sorts
  first).

## Checking it

    cd proofs/value_order && lake build

## Wave 5: value.rs glue and JSON (`FalkorValueOrder/Glue.lean`, `Json.lean`)

| here | there |
| --- | --- |
| `Glue.W` (16 variants), `W.clone`, `W.order`, `W.getType`, `W.valueOfType`, `W.name` | value.rs:180, :221, :1177, :1328, :1282, :1352 |
| `DeletedNode.new`, `DeletedRelationship.new` | value.rs:69, :87 |
| classes `OrderedEnum`, `CompareValue`, `ValueTypeOf`, `ValueGetType`, `Contains` | trait decls value.rs:1174, :1209, :1275, :1324, :1756 |
| `withCapacity`, `checkAndInsert` | value.rs:1859, :1887 |
| `Json.fmtS`/`fmtJson`, `writeJsonString`, `writeNodeJson`, `relJson`, `toJsonString` | value.rs:1582, :1704, :1716, :1622, :1374/:1384 |

f64/f32 Display, `escape_str` and the runtime accessors are fields of the `JsonEnv`
structure (no axioms). Theorems: `clone_eq`, `name_eq_iff_order_eq`, `valueOfType_self`,
`getType_concrete`, `partialCmpVia_none_iff`, `containsImpl_false`, `withCapacity_seen_iff`,
`DeletedRelationship.new_injective`, `fmt_float_nonfinite`, `fmt_list`,
`writeNodeJson_includeType`, `fmtS_entityFree`, `writeNodeJson_depth`.
-/

namespace ValueOrder

/-! ## Integers and orderings -/

/-- `Ord::cmp` on `i64`/`usize`/ids — written out so proofs can `split`. -/
def cmpI (a b : Int) : Ordering :=
  if a < b then .lt else if a = b then .eq else .gt

def cmpN (a b : Nat) : Ordering := cmpI a b

def cmpB (a b : Bool) : Ordering := cmpN (if a then 1 else 0) (if b then 1 else 0)

theorem cmpI_swap (a b : Int) : cmpI b a = (cmpI a b).swap := by
  unfold cmpI
  by_cases h1 : a < b
  · have : ¬ b < a := by omega
    have : ¬ b = a := by omega
    simp_all [Ordering.swap]
  · by_cases h2 : a = b
    · subst h2; simp
    · have h3 : b < a := by omega
      simp [h3, h1, h2, Ordering.swap]

theorem cmpI_le_trans (a b c : Int) :
    cmpI a b ≠ .gt → cmpI b c ≠ .gt → cmpI a c ≠ .gt := by
  unfold cmpI
  intro h1 h2
  by_cases x1 : a < b <;> by_cases x2 : a = b <;> by_cases y1 : b < c <;>
    by_cases y2 : b = c <;> by_cases z1 : a < c <;> by_cases z2 : a = c <;>
    simp_all <;> omega

@[simp] theorem cmpI_refl (a : Int) : cmpI a a = .eq := by simp [cmpI]

@[simp] theorem cmpN_refl (a : Nat) : cmpN a a = .eq := by simp [cmpN, cmpI]

/-! ## Floats: NaN, or an integral value -/

inductive F where
  | nan
  | num (n : Int)
deriving DecidableEq, Repr

/-- `compare_floats` (`value.rs:1801`): `partial_cmp`, and NaN answers
`(Less, NaN)` whichever side it is on. -/
inductive Flag where
  | none | comparedNull | disjoint | nan
deriving DecidableEq, Repr

def cmpF : F → F → Ordering × Flag
  | .num a, .num b => (cmpI a b, .none)
  | _, _ => (.lt, .nan)

/-! ### `i as f64` -/

def bitlenAux : Nat → Nat → Nat
  | 0, _ => 0
  | fuel + 1, n => if n = 0 then 0 else 1 + bitlenAux fuel (n / 2)

/-- Number of significant bits (`0` for `0`); fuel 70 covers every `i64`. -/
def bitlen (n : Nat) : Nat := bitlenAux 70 n

/-- Round a natural to 53 significant bits, ties to even. -/
def roundNat (m : Nat) : Nat :=
  let L := bitlen m
  if L ≤ 53 then m else
    let e := L - 53
    let q := m / 2 ^ e
    let r := m % 2 ^ e
    let half := 2 ^ (e - 1)
    let q' := if r > half ∨ (r = half ∧ q % 2 = 1) then q + 1 else q
    q' * 2 ^ e

/-- `i as f64` for an `i64` (IEEE round-to-nearest-even). -/
def roundF (i : Int) : Int :=
  if 0 ≤ i then (roundNat i.toNat : Int) else -(roundNat (-i).toNat : Int)

def P53 : Int := 9007199254740992

example : roundF (P53 + 1) = P53 := by decide
example : roundF (P53 + 3) = P53 + 4 := by decide
example : roundF (-(P53 + 1)) = -P53 := by decide
example : roundF 9223372036854775807 = 9223372036854775808 := by decide

/-! ## Values -/

/-- Every `Value` variant, for the type order. -/
inductive Tag where
  | null | bool | int | float | string | list | map | node | rel | path
  | point | datetime | date | time | duration | vecf32
deriving DecidableEq, Repr

/-- `OrderedEnum::order` (`value.rs:1177`). -/
def orderTag : Tag → Nat
  | .null => 2 ^ 15 | .bool => 2 ^ 12 | .int => 2 ^ 13 | .float => 2 ^ 14
  | .string => 2 ^ 11 | .list => 2 ^ 3 | .map => 2 ^ 0 | .node => 2 ^ 1
  | .rel => 2 ^ 2 | .path => 2 ^ 4 | .point => 2 ^ 5 | .datetime => 2 ^ 6
  | .date => 2 ^ 7 | .time => 2 ^ 8 | .duration => 2 ^ 10 | .vecf32 => 2 ^ 18

def allTags : List Tag :=
  [.null, .bool, .int, .float, .string, .list, .map, .node, .rel, .path,
   .point, .datetime, .date, .time, .duration, .vecf32]

theorem allTags_complete (t : Tag) : t ∈ allTags := by cases t <;> decide

/-- Different variants never tie on the type order. -/
theorem order_injective :
    ∀ a ∈ allTags, ∀ b ∈ allTags, orderTag a = orderTag b → a = b := by decide

/-- Nothing sits strictly between Int and Float in the type order. -/
theorem no_tag_between_int_and_float :
    ∀ t ∈ allTags, ¬ (orderTag .int < orderTag t ∧ orderTag t < orderTag .float) := by
  decide

/-- Null is *not* last: vectors sort after it (same as C's `T_VECTOR_F32`). -/
theorem null_not_greatest : orderTag .null < orderTag .vecf32 := by decide

inductive V where
  | null
  | bool (b : Bool)
  | int (i : Int)
  | flt (f : F)
  | str (s : Nat)
  | list (xs : List V)
  /-- keys already sorted (`compare_map` sorts them), one value per key -/
  | map (keys : List Nat) (vals : List V)
  | node (n : Nat)
deriving Repr

def V.tag : V → Tag
  | .null => .null | .bool _ => .bool | .int _ => .int | .flt _ => .float
  | .str _ => .string | .list _ => .list | .map .. => .map | .node _ => .node

def order (v : V) : Nat := orderTag v.tag

/-! ## `compare_list`'s state -/

structure St where
  first : Ordering
  nullc : Nat
  nec : Nat
  incc : Nat
deriving Repr

def St.init : St := ⟨.eq, 0, 0, 0⟩

/-- One iteration of the `for (a_value, b_value)` loop in `compare_list`. -/
def step (st : St) (r : Ordering × Flag) : St :=
  let first' := if st.first = .eq then r.1 else st.first
  if r.2 ≠ .none then
    { first := first'
      nullc := st.nullc + (if r.2 = .comparedNull then 1 else 0)
      incc := st.incc + (if r.2 = .comparedNull then 0 else 1)
      nec := st.nec + 1 }
  else if r.1 ≠ .eq then
    { st with nec := st.nec + 1, first := first' }
  else st

/-- The five returns after the loop, in source order. -/
def listFinish (la lb : Nat) (st : St) : Ordering × Flag :=
  if st.nec = min la lb ∧ st.nullc < st.nec ∧ st.first ≠ .eq then (st.first, .none)
  else if st.nullc > 0 ∧ la = lb then (st.first, .comparedNull)
  else if st.incc > 0 ∧ st.first = .eq ∧ la = lb then (.eq, .disjoint)
  else if st.first ≠ .eq then (st.first, .none)
  else (cmpN la lb, .none)

/-- First differing key (maps have equal key counts when this runs). -/
def keysCmp : List Nat → List Nat → Ordering
  | a :: as, b :: bs => if a ≠ b then cmpN a b else keysCmp as bs
  | _, _ => .eq

def isNull : V → Bool
  | .null => true
  | _ => false

mutual
/-- `compare_value` (`value.rs:1224`), arm by arm. -/
def cmpV : V → V → Ordering × Flag
  | .bool a, .bool b => (cmpB a b, .none)
  | .flt a, .flt b => cmpF a b
  | .str a, .str b => (cmpN a b, .none)
  | .list xs, .list ys =>
      if xs.length = 0 ∧ ys.length = 0 then (.eq, .none)
      else listFinish xs.length ys.length (listLoop xs ys St.init)
  | .map ka va, .map kb vb =>
      if ka.length ≠ kb.length then (cmpN ka.length kb.length, .none)
      else if keysCmp ka kb ≠ .eq then (keysCmp ka kb, .none)
      else mapVals va vb
  | .node a, .node b => (cmpN a b, .none)
  | .int a, .int b => (cmpI a b, .none)
  | .int i, .flt f => cmpF (.num (roundF i)) f
  | .flt f, .int i => cmpF f (.num (roundF i))
  | a, b =>
      if isNull a || isNull b then (cmpN (order a) (order b), .comparedNull)
      else (cmpN (order a) (order b), .disjoint)

def listLoop : List V → List V → St → St
  | x :: xs, y :: ys, st => listLoop xs ys (step st (cmpV x y))
  | _, _, st => st

/-- `compare_map`'s value loop: the first disjoint-or-null pair answers
`Equal` and stops. -/
def mapVals : List V → List V → Ordering × Flag
  | x :: xs, y :: ys =>
      let r := cmpV x y
      if r.2 = .comparedNull ∨ r.2 = .disjoint then (.eq, r.2)
      else if r.1 ≠ .eq then r
      else mapVals xs ys
  | _, _ => (.eq, .none)
end

/-- `PartialEq for Value` (`value.rs:1215`). -/
def eqV (a b : V) : Bool := (cmpV a b).1 = .eq

/-- The ORDER BY comparator: `compare_value(..).0` (`ops/sort.rs:80`). -/
def sortCmp (a b : V) : Ordering := (cmpV a b).1

/-- ORDER BY's `Column::Floats` lane (`ops/sort.rs:347`). -/
def floatLane (a b : F) : Ordering :=
  match a, b with
  | .num x, .num y => cmpI x y
  | _, _ => .lt

theorem floatLane_eq_cmpV (a b : F) : floatLane a b = sortCmp (.flt a) (.flt b) := by
  cases a <;> cases b <;> rfl

/-- `compare_values`, `CmpOp::Eq` (`vectorized.rs:239`): `none` is Cypher null. -/
def evalEq (a b : V) : Option Bool :=
  match cmpV a b with
  | (_, .comparedNull) => none
  | (_, .nan) | (_, .disjoint) => some false
  | (o, .none) => some (o = .eq)

/-- `compare_values`, `CmpOp::Neq`. -/
def evalNeq (a b : V) : Option Bool :=
  match cmpV a b with
  | (_, .comparedNull) => none
  | (o, _) => some (o ≠ .eq)

/-! ## Hashing -/

/-- Two's-complement `i64 → u64`. -/
def w (i : Int) : UInt64 := UInt64.ofNat (i % (2 ^ 64 : Int)).toNat

def I64MIN : Int := -9223372036854775808
def I64MAX : Int := 9223372036854775807

/-- Rust's saturating `f as i64`. -/
def satI64 (n : Int) : Int := if n < I64MIN then I64MIN else if n > I64MAX then I64MAX else n

/-- IEEE bits of an integral, representable double. -/
def ieeeBits (n : Int) : UInt64 :=
  let m := n.natAbs
  if m = 0 then (if n < 0 then 0x8000000000000000 else 0) else
    let L := bitlen m
    let mant := m / 2 ^ (L - 53) * 2 ^ (53 - L)
    let exp := L - 1 + 1023
    let s : Nat := if n < 0 then 2 ^ 63 else 0
    UInt64.ofNat (s + exp * 2 ^ 52 + (mant - 2 ^ 52))

/-- The quiet NaN `0.0/0.0` produces. -/
def NAN_BITS : UInt64 := 0x7ff8000000000000

/-- `Self::Float(x)` arm of `Hash for Value` (`value.rs:836`): hash as the
integer if `x - (x as i64) as f64 == 0`, else the bits. -/
def floatWord : F → UInt64
  | .nan => NAN_BITS
  | .num n =>
      let casted := satI64 n
      if roundF casted = n then w casted else ieeeBits n

mutual
/-- The words `Hash for Value` writes, in order (each `write_*` is one Fx
`add_to_hash`). -/
def hwords : V → List UInt64
  | .null => [0]
  | .bool b => [1, if b then 1 else 0]
  | .int i => [2, w i]
  | .flt f => [2, floatWord f]
  | .str s => [3, UInt64.ofNat s, 0xff]
  | .list xs => [4, UInt64.ofNat xs.length] ++ hwordsList xs
  | .map ks vs => [5, UInt64.ofNat ks.length] ++ hwordsMap ks vs
  | .node n => [6, UInt64.ofNat n]

def hwordsList : List V → List UInt64
  | [] => []
  | x :: xs => hwords x ++ hwordsList xs

def hwordsMap : List Nat → List V → List UInt64
  | k :: ks, v :: vs => [UInt64.ofNat k, 0xff] ++ hwords v ++ hwordsMap ks vs
  | _, _ => []
end

def K : UInt64 := 0xf1357aea2e62a9c5

/-- `FxHasher::add_to_hash`. -/
def fxAdd (h i : UInt64) : UInt64 := (h + i) * K

/-- `FxHasher::finish`: `rotate_left(26)`. -/
def fxFinish (h : UInt64) : UInt64 := (h <<< 26) ||| (h >>> 38)

def fx (ws : List UInt64) : UInt64 := fxFinish (ws.foldl fxAdd 0)

/-- `DistinctOp::next`: one hasher, every projected column in turn. -/
def rowHash (row : List V) : UInt64 := fx (row.flatMap hwords)

/-- `hash(&[Value])` (`ValuesDeduper::is_seen`, `count(DISTINCT x)`). -/
def sliceHash (xs : List V) : UInt64 := fx ([UInt64.ofNat xs.length] ++ hwordsList xs)

/-! ## Counterexamples -/

/-- **Finding 1.** NaN is less than 1.0, and 1.0 is less than NaN. -/
theorem nan_breaks_antisymmetry :
    sortCmp (.flt .nan) (.flt (.num 1)) = .lt ∧
    sortCmp (.flt (.num 1)) (.flt .nan) = .lt ∧
    sortCmp (.flt .nan) (.flt .nan) = .lt := by decide

/-- ...so the ORDER BY comparator is not antisymmetric (`cmp b a ≠ (cmp a b).swap`). -/
theorem sortCmp_not_antisymmetric :
    ¬ ∀ a b : V, sortCmp b a = (sortCmp a b).swap := fun h => by
  have := h (.flt .nan) (.flt (.num 1)); revert this; decide

/-- **Finding 2.** Equality is not transitive across Int/Float near 2^53. -/
theorem mixed_numeric_not_transitive :
    sortCmp (.int P53) (.flt (.num P53)) = .eq ∧
    sortCmp (.flt (.num P53)) (.int (P53 + 1)) = .eq ∧
    sortCmp (.int P53) (.int (P53 + 1)) = .lt := by decide

def m1 : V := .map [0, 1] [.int 1, .int 1]      -- {a:1,   b:1}
def m2 : V := .map [0, 1] [.str 7, .int 3]      -- {a:'s', b:3}
def m3 : V := .map [0, 1] [.int 1, .int 2]      -- {a:1,   b:2}

/-- **Finding 3.** Map order is not transitive. -/
theorem map_order_not_transitive :
    sortCmp m1 m2 = .eq ∧ sortCmp m2 m3 = .eq ∧ sortCmp m1 m3 = .lt := by decide

/-- Same with a null instead of a disjoint type. -/
theorem map_order_not_transitive_null :
    sortCmp (.map [0, 1] [.int 1, .int 1]) (.map [0, 1] [.null, .int 3]) = .eq ∧
    sortCmp (.map [0, 1] [.null, .int 3]) (.map [0, 1] [.int 1, .int 2]) = .eq ∧
    sortCmp (.map [0, 1] [.int 1, .int 1]) (.map [0, 1] [.int 1, .int 2]) = .lt := by
  decide

/-- No total preorder can agree with `sortCmp`: that is what makes
`slice::sort_by` panic. -/
theorem sortCmp_not_transitive :
    ¬ ∀ a b c : V, sortCmp a b = .eq → sortCmp b c = .eq → sortCmp a c = .eq :=
  fun h => by
    have := h m1 m2 m3 (by decide) (by decide); revert this; decide

/-- **Finding 4.** `{a:1} = {a:'x'}` and `{a:1} <> {a:'x'}` are both false. -/
theorem map_neq_both_false :
    evalEq (.map [0] [.int 1]) (.map [0] [.str 7]) = some false ∧
    evalNeq (.map [0] [.int 1]) (.map [0] [.str 7]) = some false := by decide

/-- **Finding 5.** `Hash` is not consistent with `PartialEq`. -/
theorem hash_not_consistent_with_eq :
    eqV (.int (P53 + 1)) (.flt (.num P53)) = true ∧
    hwords (.int (P53 + 1)) ≠ hwords (.flt (.num P53)) ∧
    rowHash [.int (P53 + 1)] ≠ rowHash [.flt (.num P53)] := by decide

/-- **Finding 6.** `PartialEq` is not reflexive on NaN, yet NaNs hash alike. -/
theorem nan_eq_irreflexive :
    eqV (.flt .nan) (.flt .nan) = false ∧
    hwords (.flt .nan) = hwords (.flt .nan) := by decide

/-- `-K²` as an `i64`. -/
def B : Int := -1452335207727870361

example : w B = 0 - K * K := by decide

/-- **Finding 7.** Two different rows, one DISTINCT hash. -/
theorem fx_row_collision :
    rowHash [.int 0, .int 0] = rowHash [.int 1, .int B] ∧
    sliceHash [.list [.int 0, .int 0]] = sliceHash [.list [.int 1, .int B]] ∧
    eqV (.int 0) (.int 1) = false := by decide


/-- **Finding 7, in general.** Fx is affine in each word, so from *any* hasher
state `s`, bumping one word by 1 and the word two writes later by `-K²`
lands on the same state. Every two-Int row (and every two-element list) has a
partner it collides with. -/
theorem fx_collision_general (s a b : UInt64) :
    fxAdd (fxAdd (fxAdd s (a + 1)) 2) (b - K * K) = fxAdd (fxAdd (fxAdd s a) 2) b := by
  unfold fxAdd
  have h1 : (s + (a + 1)) * K = (s + a) * K + K := by
    rw [← UInt64.add_assoc, UInt64.add_mul, UInt64.one_mul]
  have h2 : ((s + a) * K + K + 2) * K = ((s + a) * K + 2) * K + K * K := by
    rw [UInt64.add_assoc, UInt64.add_comm K 2, ← UInt64.add_assoc, UInt64.add_mul]
  have h3 : ∀ y : UInt64, y + K * K + (b - K * K) = y + b := by
    intro y
    rw [UInt64.add_assoc, UInt64.add_comm (K * K), UInt64.sub_add_cancel]
  rw [h1, h2, h3]

/-- `min_batch` (`functions/aggregation.rs:487`): replace `best` when the new
value compares `Less` (and not against a null). -/
def minFold (xs : List V) : V :=
  xs.foldl (fun best v =>
    match v, best with
    | .null, _ => best
    | _, .null => v
    | _, _ =>
        let r := cmpV v best
        if r.1 = .lt ∧ r.2 ≠ .comparedNull then v else best) .null

/-- **Finding 8.** `min(x)` over `[1.0, NaN, 3.0]` is `3.0`: NaN displaces 1.0
(NaN < 1.0) and then 3.0 displaces NaN (3.0 < NaN). -/
theorem min_nan_picks_larger :
    (match minFold [.flt (.num 1), .flt .nan, .flt (.num 3)] with
      | .flt (.num n) => some n
      | _ => none) = some 3 ∧
    sortCmp (.flt (.num 1)) (.flt (.num 3)) = .lt := by decide

/-! ## What holds -/

/-! ### The sane scalar fragment is totally preordered -/

/-- Scalars with no NaN, and Ints (i64) that `as f64` does not move. -/
def sane : V → Prop
  | .null => True
  | .bool _ => True
  | .int i => roundF i = i ∧ I64MIN ≤ i ∧ i ≤ I64MAX
  | .flt (.num _) => True
  | .str _ => True
  | .node _ => True
  | _ => False

/-- Int and Float share a rank (Int's); `no_tag_between_int_and_float` is why
that is harmless. -/
def key : V → Nat × Int
  | .null => (2 ^ 15, 0)
  | .bool b => (2 ^ 12, if b then 1 else 0)
  | .int i => (2 ^ 13, i)
  | .flt (.num n) => (2 ^ 13, n)
  | .str s => (2 ^ 11, s)
  | .node n => (2 ^ 1, n)
  | _ => (0, 0)

def lexK (p q : Nat × Int) : Ordering :=
  match cmpN p.1 q.1 with
  | .eq => cmpI p.2 q.2
  | o => o

theorem cmp_sane_eq_key (a b : V) (ha : sane a) (hb : sane b) :
    sortCmp a b = lexK (key a) (key b) := by
  cases a with
  | flt f => cases f with
    | nan => simp [sane] at ha
    | num n => cases b with
      | flt g => cases g with
        | nan => simp [sane] at hb
        | num m => simp [sortCmp, cmpV, cmpF, key, lexK]
      | int j =>
          obtain ⟨hj, -⟩ := hb
          simp [sortCmp, cmpV, cmpF, key, lexK, hj]
      | list _ | map _ _ => simp [sane] at hb
      | _ => simp [sortCmp, cmpV, key, lexK, cmpN, cmpI, order, V.tag, orderTag, isNull]
  | int i =>
      obtain ⟨hi, -⟩ := ha
      cases b with
      | flt g => cases g with
        | nan => simp [sane] at hb
        | num m => simp [sortCmp, cmpV, cmpF, key, lexK, hi]
      | int j => simp [sortCmp, cmpV, key, lexK]
      | list _ | map _ _ => simp [sane] at hb
      | _ => simp [sortCmp, cmpV, key, lexK, cmpN, cmpI, order, V.tag, orderTag, isNull]
  | list _ | map _ _ => simp [sane] at ha
  | bool x => cases b with
      | flt g => cases g with
        | nan => simp [sane] at hb
        | num m => simp [sortCmp, cmpV, key, lexK, cmpN, cmpI, order, V.tag, orderTag, isNull]
      | list _ | map _ _ => simp [sane] at hb
      | bool y => cases x <;> cases y <;> decide
      | _ => simp [sortCmp, cmpV, key, lexK, cmpN, cmpI, order, V.tag, orderTag, isNull]
  | str x => cases b with
      | flt g => cases g with
        | nan => simp [sane] at hb
        | num m => simp [sortCmp, cmpV, key, lexK, cmpN, cmpI, order, V.tag, orderTag, isNull]
      | list _ | map _ _ => simp [sane] at hb
      | str y => simp [sortCmp, cmpV, key, lexK, cmpN]
      | _ => simp [sortCmp, cmpV, key, lexK, cmpN, cmpI, order, V.tag, orderTag, isNull]
  | node x => cases b with
      | flt g => cases g with
        | nan => simp [sane] at hb
        | num m => simp [sortCmp, cmpV, key, lexK, cmpN, cmpI, order, V.tag, orderTag, isNull]
      | list _ | map _ _ => simp [sane] at hb
      | node y => simp [sortCmp, cmpV, key, lexK, cmpN]
      | _ => simp [sortCmp, cmpV, key, lexK, cmpN, cmpI, order, V.tag, orderTag, isNull]
  | null => cases b with
      | flt g => cases g with
        | nan => simp [sane] at hb
        | num m => simp [sortCmp, cmpV, key, lexK, cmpN, cmpI, order, V.tag, orderTag, isNull]
      | list _ | map _ _ => simp [sane] at hb
      | _ => simp [sortCmp, cmpV, key, lexK, cmpN, cmpI, order, V.tag, orderTag, isNull]


theorem lexK_le_iff (p q : Nat × Int) :
    lexK p q ≠ .gt ↔ ((p.1 : Int) < q.1 ∨ ((p.1 : Int) = q.1 ∧ p.2 ≤ q.2)) := by
  obtain ⟨a, b⟩ := p
  obtain ⟨c, d⟩ := q
  simp only [lexK, cmpN, cmpI]
  by_cases h1 : (a : Int) < c
  · simp [h1]
  · by_cases h2 : (a : Int) = c
    · by_cases h3 : b < d
      · simp [h1, h2, h3] <;> omega
      · by_cases h4 : b = d
        · simp [h1, h2, h3, h4]
        · simp [h1, h2, h3, h4] <;> omega
    · simp [h1, h2] <;> omega

theorem lexK_eq_iff (p q : Nat × Int) : lexK p q = .eq ↔ p = q := by
  obtain ⟨a, b⟩ := p
  obtain ⟨c, d⟩ := q
  simp only [lexK, cmpN, cmpI]
  by_cases h1 : (a : Int) < c
  · simp [h1] <;> omega
  · by_cases h2 : (a : Int) = c
    · have : a = c := by omega
      subst this
      by_cases h3 : b < d
      · simp [h3] <;> omega
      · by_cases h4 : b = d <;> simp [h3, h4]
    · simp [h1, h2] <;> omega

theorem lexK_swap (p q : Nat × Int) : lexK q p = (lexK p q).swap := by
  obtain ⟨a, b⟩ := p
  obtain ⟨c, d⟩ := q
  simp only [lexK, cmpN]
  rw [cmpI_swap (a : Int) c, cmpI_swap b d]
  cases cmpI (a : Int) c <;> rfl

theorem sane_refl (a : V) (ha : sane a) : sortCmp a a = .eq := by
  rw [cmp_sane_eq_key a a ha ha, lexK_eq_iff]

theorem sane_swap (a b : V) (ha : sane a) (hb : sane b) :
    sortCmp b a = (sortCmp a b).swap := by
  rw [cmp_sane_eq_key a b ha hb, cmp_sane_eq_key b a hb ha, lexK_swap]

/-- **Transitivity** on the sane fragment. -/
theorem sane_le_trans (a b c : V) (ha : sane a) (hb : sane b) (hc : sane c) :
    sortCmp a b ≠ .gt → sortCmp b c ≠ .gt → sortCmp a c ≠ .gt := by
  rw [cmp_sane_eq_key a b ha hb, cmp_sane_eq_key b c hb hc, cmp_sane_eq_key a c ha hc,
    lexK_le_iff, lexK_le_iff, lexK_le_iff]
  omega

/-! ### `|i| < 2^53` is exact -/

theorem bitlenAux_le : ∀ (fuel m k : Nat), m < 2 ^ k → bitlenAux fuel m ≤ k := by
  intro fuel
  induction fuel with
  | zero => intro m k _; simp [bitlenAux]
  | succ f ih =>
    intro m k hm
    simp only [bitlenAux]
    split
    · omega
    · cases k with
      | zero => simp at hm; omega
      | succ k =>
        have : m / 2 < 2 ^ k := by rw [Nat.pow_succ] at hm; omega
        have := ih (m / 2) k this
        omega

theorem roundNat_exact (m : Nat) (h : m < 2 ^ 53) : roundNat m = m := by
  have : bitlen m ≤ 53 := bitlenAux_le 70 m 53 h
  simp [roundNat, this]

/-- Every Int with `|i| < 2^53` survives `as f64`. -/
theorem roundF_exact (i : Int) (h1 : -P53 < i) (h2 : i < P53) : roundF i = i := by
  unfold roundF P53 at *
  split
  · rw [roundNat_exact _ (by omega)]; omega
  · rw [roundNat_exact _ (by omega)]; omega

/-- So every Int in that range is sane. -/
theorem int_sane (i : Int) (h1 : -P53 < i) (h2 : i < P53) : sane (.int i) :=
  ⟨roundF_exact i h1 h2, by unfold I64MIN P53 at *; omega, by unfold I64MAX P53 at *; omega⟩

/-! ### Hash agrees with `==` on the sane fragment -/

theorem satI64_id (n : Int) (lo : I64MIN ≤ n) (hi : n ≤ I64MAX) : satI64 n = n := by
  unfold satI64
  have h1 : ¬ n < I64MIN := by omega
  have h2 : ¬ n > I64MAX := by omega
  simp [h1, h2]

theorem sane_hash_consistent (a b : V) (ha : sane a) (hb : sane b)
    (h : eqV a b = true) : hwords a = hwords b := by
  have hk : key a = key b := by
    simp only [eqV, decide_eq_true_eq] at h
    rw [← lexK_eq_iff, ← cmp_sane_eq_key a b ha hb]; exact h
  cases a with
  | flt f => cases f with
    | nan => simp [sane] at ha
    | num n => cases b with
      | flt g => cases g with
        | nan => simp [sane] at hb
        | num m => simp [key] at hk; subst hk; rfl
      | int j =>
          obtain ⟨hj, lo, hi⟩ := hb
          simp [key] at hk; subst hk
          have hs : satI64 n = n := satI64_id _ lo (by assumption)
          simp [hwords, floatWord, hs, hj]
      | list _ | map _ _ => simp [sane] at hb
      | _ => simp [key] at hk
  | int i =>
      obtain ⟨hi, lo, up⟩ := ha
      cases b with
      | flt g => cases g with
        | nan => simp [sane] at hb
        | num m =>
          simp [key] at hk; subst hk
          have hs : satI64 i = i := satI64_id _ lo (by assumption)
          simp [hwords, floatWord, hs, hi]
      | int j => simp [key] at hk; subst hk; rfl
      | list _ | map _ _ => simp [sane] at hb
      | _ => simp [key] at hk
  | list _ | map _ _ => simp [sane] at ha
  | bool x => cases b with
      | flt g => cases g <;> simp [key] at hk
      | bool y => cases x <;> cases y <;> simp_all [key]
      | _ => simp [key] at hk
  | str x => cases b with
      | flt g => cases g <;> simp [key] at hk
      | str y => simp [key] at hk; have : x = y := by omega
                 subst this; rfl
      | _ => simp [key] at hk
  | node x => cases b with
      | flt g => cases g <;> simp [key] at hk
      | node y => simp [key] at hk; have : x = y := by omega
                  subst this; rfl
      | _ => simp [key] at hk
  | null => cases b with
      | flt g => cases g <;> simp [key] at hk
      | null => rfl
      | _ => simp [key] at hk


/-! ### Lists: the ORDER BY half of `compare_list` is lexicographic -/

def lex (c : V → V → Ordering) : List V → List V → Ordering
  | [], [] => .eq
  | [], _ :: _ => .lt
  | _ :: _, [] => .gt
  | x :: xs, y :: ys =>
      match c x y with
      | .eq => lex c xs ys
      | o => o

/-- First unequal element over the shared prefix. -/
def firstDiff : List V → List V → Ordering
  | x :: xs, y :: ys => if sortCmp x y = .eq then firstDiff xs ys else sortCmp x y
  | _, _ => .eq

theorem step_first (st : St) (r : Ordering × Flag) :
    (step st r).first = if st.first = .eq then r.1 else st.first := by
  unfold step
  by_cases h1 : r.2 ≠ .none
  · simp [h1]
  · by_cases h2 : r.1 ≠ .eq
    · simp [h1, h2]
    · simp only [h1, h2, if_false]
      by_cases h3 : st.first = .eq <;> simp_all

theorem loop_first : ∀ (xs ys : List V) (st : St),
    (listLoop xs ys st).first = if st.first = .eq then firstDiff xs ys else st.first
  | [], ys, st => by cases ys <;> simp [listLoop, firstDiff]
  | x :: xs, [], st => by simp [listLoop, firstDiff]
  | x :: xs, y :: ys, st => by
      rw [listLoop, loop_first xs ys, step_first]
      simp only [firstDiff, sortCmp]
      by_cases h : st.first = .eq <;> by_cases h' : (cmpV x y).1 = .eq <;> simp [h, h']

theorem finish_fst (la lb : Nat) (st : St) :
    (listFinish la lb st).1 = if st.first ≠ .eq then st.first else cmpN la lb := by
  unfold listFinish
  split
  · rename_i h; simp [h.2.2]
  · split
    · rename_i h2; by_cases hf : st.first = .eq <;> simp [hf, h2.2]
    · split
      · rename_i h3; simp [h3.2.1, h3.2.2]
      · split
        · rename_i h4; simp [h4]
        · rename_i h4; simp at h4; simp [h4]

theorem cmpN_succ (a b : Nat) : cmpN (a + 1) (b + 1) = cmpN a b := by
  simp only [cmpN, cmpI]
  by_cases h1 : (a : Int) < b
  · have : ((a + 1 : Nat) : Int) < ((b + 1 : Nat) : Int) := by omega
    simp [h1, this]
  · by_cases h2 : (a : Int) = b
    · have : a = b := by omega
      subst this; simp
    · have h3 : ¬ ((a + 1 : Nat) : Int) < ((b + 1 : Nat) : Int) := by omega
      have h4 : ¬ ((a + 1 : Nat) : Int) = ((b + 1 : Nat) : Int) := by omega
      rw [if_neg h1, if_neg h2, if_neg h3, if_neg h4]

theorem firstDiff_lex : ∀ xs ys : List V,
    (if firstDiff xs ys ≠ .eq then firstDiff xs ys else cmpN xs.length ys.length)
      = lex sortCmp xs ys
  | [], [] => by decide
  | [], y :: ys => by simp [firstDiff, lex, cmpN, cmpI]
  | x :: xs, [] => by
      simp only [firstDiff, lex, cmpN, cmpI, List.length_cons, List.length_nil]
      have h1 : ¬ (((xs.length + 1 : Nat) : Int) < ((0 : Nat) : Int)) := by omega
      have h2 : ¬ (((xs.length + 1 : Nat) : Int) = ((0 : Nat) : Int)) := by omega
      simp only [ne_eq, not_true_eq_false, if_false]
      rw [if_neg h1, if_neg h2]
  | x :: xs, y :: ys => by
      have ih := firstDiff_lex xs ys
      simp only [firstDiff, lex, List.length_cons, cmpN_succ]
      cases h : sortCmp x y <;> simp [ih]

/-- **The ORDER BY half of `compare_list` is lexicographic order**, whatever
the null/inconclusive counters decide about the flag. -/
theorem list_fst_eq_lex (xs ys : List V) :
    sortCmp (.list xs) (.list ys) = lex sortCmp xs ys := by
  rw [← firstDiff_lex]
  unfold sortCmp
  rw [cmpV]
  by_cases h : xs.length = 0 ∧ ys.length = 0
  · obtain ⟨h1, h2⟩ := h
    rw [List.length_eq_zero_iff] at h1 h2
    subst h1; subst h2; decide
  · rw [if_neg h, finish_fst, loop_first]
    simp [St.init]

/-! ### Lexicographic order lifts a total preorder -/

section lexlaws
variable (P : V → Prop) (c : V → V → Ordering)
  (hswap : ∀ a b, P a → P b → c b a = (c a b).swap)
  (htrans : ∀ a b d, P a → P b → P d → c a b ≠ .gt → c b d ≠ .gt → c a d ≠ .gt)
include hswap htrans

theorem lt_le_lt (a b d : V) (ha : P a) (hb : P b) (hd : P d)
    (h1 : c a b = .lt) (h2 : c b d ≠ .gt) : c a d = .lt := by
  have hle : c a d ≠ .gt := htrans a b d ha hb hd (by simp [h1]) h2
  cases had : c a d
  · rfl
  · have hda : c d a = .eq := by rw [hswap a d ha hd, had]; rfl
    have := htrans b d a hb hd ha h2 (by simp [hda])
    rw [hswap a b ha hb, h1] at this; exact absurd rfl this
  · exact absurd had hle

theorem le_lt_lt (a b d : V) (ha : P a) (hb : P b) (hd : P d)
    (h1 : c a b ≠ .gt) (h2 : c b d = .lt) : c a d = .lt := by
  have hle : c a d ≠ .gt := htrans a b d ha hb hd h1 (by simp [h2])
  cases had : c a d
  · rfl
  · have hda : c d a = .eq := by rw [hswap a d ha hd, had]; rfl
    have := htrans d a b hd ha hb (by simp [hda]) h1
    rw [hswap b d hb hd, h2] at this; exact absurd rfl this
  · exact absurd had hle

theorem eq_eq_eq (a b d : V) (ha : P a) (hb : P b) (hd : P d)
    (h1 : c a b = .eq) (h2 : c b d = .eq) : c a d = .eq := by
  have hle : c a d ≠ .gt := htrans a b d ha hb hd (by simp [h1]) (by simp [h2])
  have hba : c b a = .eq := by rw [hswap a b ha hb, h1]; rfl
  have hdb : c d b = .eq := by rw [hswap b d hb hd, h2]; rfl
  have hge : c d a ≠ .gt := htrans d b a hd hb ha (by simp [hdb]) (by simp [hba])
  rw [hswap a d ha hd] at hge
  cases had : c a d
  · rw [had] at hge; simp [Ordering.swap] at hge
  · rfl
  · exact absurd had hle

omit htrans in
theorem lex_swap : ∀ xs ys : List V, (∀ x ∈ xs, P x) → (∀ y ∈ ys, P y) →
    lex c ys xs = (lex c xs ys).swap
  | [], [] , _, _ => rfl
  | [], _ :: _, _, _ => rfl
  | _ :: _, [], _, _ => rfl
  | x :: xs, y :: ys, hx, hy => by
      have e := hswap x y (hx x (by simp)) (hy y (by simp))
      have ih := lex_swap xs ys (fun z hz => hx z (by simp [hz])) (fun z hz => hy z (by simp [hz]))
      simp only [lex, e]
      cases c x y <;> simp [Ordering.swap, ih]

theorem lex_le_trans : ∀ xs ys zs : List V,
    (∀ x ∈ xs, P x) → (∀ y ∈ ys, P y) → (∀ z ∈ zs, P z) →
    lex c xs ys ≠ .gt → lex c ys zs ≠ .gt → lex c xs zs ≠ .gt
  | [], [], _, _, _, _, _, h2 => h2
  | [], _ :: _, [], _, _, _, _, h2 => by simp [lex] at h2
  | [], _ :: _, _ :: _, _, _, _, _, _ => by simp [lex]
  | _ :: _, [], _, _, _, _, h1, _ => by simp [lex] at h1
  | _ :: _, _ :: _, [], _, _, _, _, h2 => by simp [lex] at h2
  | x :: xs, y :: ys, z :: zs, hx, hy, hz, h1, h2 => by
      have px := hx x (by simp); have py := hy y (by simp); have pz := hz z (by simp)
      have ih := lex_le_trans xs ys zs (fun w hw => hx w (by simp [hw]))
        (fun w hw => hy w (by simp [hw])) (fun w hw => hz w (by simp [hw]))
      simp only [lex] at h1 h2 ⊢
      cases e1 : c x y <;> cases e2 : c y z <;> simp only [e1, e2] at h1 h2
      all_goals first
        | (simp at h1; done)
        | (simp at h2; done)
        | (rw [lt_le_lt P c hswap htrans x y z px py pz e1 (by simp [e2])]; simp)
        | (rw [le_lt_lt P c hswap htrans x y z px py pz (by simp [e1]) e2]; simp)
        | (rw [eq_eq_eq P c hswap htrans x y z px py pz e1 e2]; simpa using ih h1 h2)
end lexlaws

/-- **Lists of sane scalars are totally preordered by ORDER BY.** -/
theorem sane_list_le_trans (xs ys zs : List V)
    (hx : ∀ x ∈ xs, sane x) (hy : ∀ y ∈ ys, sane y) (hz : ∀ z ∈ zs, sane z) :
    sortCmp (.list xs) (.list ys) ≠ .gt → sortCmp (.list ys) (.list zs) ≠ .gt →
    sortCmp (.list xs) (.list zs) ≠ .gt := by
  rw [list_fst_eq_lex, list_fst_eq_lex, list_fst_eq_lex]
  exact lex_le_trans sane sortCmp (fun a b ha hb => sane_swap a b ha hb)
    (fun a b d ha hb hd => sane_le_trans a b d ha hb hd) xs ys zs hx hy hz

theorem sane_list_swap (xs ys : List V) (hx : ∀ x ∈ xs, sane x) (hy : ∀ y ∈ ys, sane y) :
    sortCmp (.list ys) (.list xs) = (sortCmp (.list xs) (.list ys)).swap := by
  rw [list_fst_eq_lex, list_fst_eq_lex]
  exact lex_swap sane sortCmp (fun a b ha hb => sane_swap a b ha hb) xs ys hx hy

/-! ## Axiom audit -/

#print axioms fx_row_collision
#print axioms fx_collision_general
#print axioms sortCmp_not_transitive
#print axioms sane_le_trans
#print axioms sane_hash_consistent
#print axioms list_fst_eq_lex
#print axioms sane_list_le_trans
#print axioms roundF_exact
#print axioms sane_list_swap
#print axioms map_neq_both_false
#print axioms hash_not_consistent_with_eq
#print axioms min_nan_picks_larger

end ValueOrder
