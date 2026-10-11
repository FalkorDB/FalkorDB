/-! Copied verbatim from proofs/value_order `FalkorValueOrder/Model.lean` (2026-10-11): the extended float/value model, `compare_value`, and main's `Hash for Value` (`hwords`). -/
/-! Core model of `compare_value`, `PartialEq` and `Hash for Value` (see the report in `FalkorValueOrder.lean`). -/

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

/-! ## Floats

`F` represents every class of IEEE value that ordering or hashing can tell
apart (extended 2026-10-11; it used to be `nan | num`): NaN with its sign bit and
payload, both zeros, integral values, non-integral finite values and both
infinities. The same type models `f64` (Float) and `f32` (VecF32 and Point
components); only the bit layouts (`bits64`/`bits32`) differ. -/

inductive F where
  /-- NaN: sign bit and payload. The mantissa field is `payload % (2^52-1) + 1`
  for f64 and `payload % (2^23-1) + 1` for f32 (any non-zero mantissa is a NaN). -/
  | nan (neg : Bool) (payload : Nat)
  /-- `-0.0` (`num 0` is `+0.0`). -/
  | negZero
  /-- an integral finite value -/
  | num (n : Int)
  /-- a non-integral finite value strictly between `lo` and `lo + 1`; `r` orders
  the ones that share `lo` (the model's stand-in for the fraction) -/
  | frac (lo : Int) (r : Nat)
  | negInf
  | posInf
deriving DecidableEq, Repr

/-- `f64::NAN` (`0x7ff8000000000000`), the NaN the old two-constructor model had. -/
def F.qnan : F := .nan false (2 ^ 51 - 1)

/-- The numeric position of a non-NaN float: `-inf < (… num/frac by value …) < +inf`.
`-0.0` and `+0.0` share one position (IEEE `==`). -/
def F.nkey : F → Option (Nat × Int × Nat)
  | .nan _ _ => none
  | .negInf => some (0, 0, 0)
  | .negZero => some (1, 0, 0)
  | .num n => some (1, n, 0)
  | .frac lo r => some (1, lo, r + 1)
  | .posInf => some (2, 0, 0)

def cmp3 (a b : Nat × Int × Nat) : Ordering :=
  match cmpI a.1 b.1 with
  | .eq => match cmpI a.2.1 b.2.1 with
    | .eq => cmpI a.2.2 b.2.2
    | o => o
  | o => o

/-- `f64::partial_cmp` / `f32::partial_cmp`: `none` iff either side is NaN. -/
def fpcmp (a b : F) : Option Ordering :=
  match a.nkey, b.nkey with
  | some x, some y => some (cmp3 x y)
  | _, _ => none

@[simp] theorem fpcmp_num (a b : Int) : fpcmp (.num a) (.num b) = some (cmpI a b) := by
  simp only [fpcmp, F.nkey, cmp3, cmpI]
  by_cases h1 : a < b
  · simp [h1]
  · by_cases h2 : a = b
    · subst h2; simp
    · simp [h1, h2]

/-- `compare_floats` (`value.rs:1801`): `partial_cmp`, and NaN answers
`(Less, NaN)` whichever side it is on. -/
inductive Flag where
  | none | comparedNull | disjoint | nan
deriving DecidableEq, Repr

def cmpF (a b : F) : Ordering × Flag :=
  match fpcmp a b with
  | some o => (o, .none)
  | none => (.lt, .nan)

@[simp] theorem cmpF_num (a b : Int) : cmpF (.num a) (.num b) = (cmpI a b, .none) := by
  simp [cmpF]

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
  /-- `VecF32(Arc<[f32]>)`: the components (extended 2026-10-11; was a `Tag` only) -/
  | vecf32 (xs : List F)
  /-- `Point { latitude, longitude }` (f32 components) -/
  | point (lat lon : F)
deriving Repr

def V.tag : V → Tag
  | .null => .null | .bool _ => .bool | .int _ => .int | .flt _ => .float
  | .str _ => .string | .list _ => .list | .map .. => .map | .node _ => .node
  | .vecf32 _ => .vecf32 | .point .. => .point

/-- IEEE `x != y` on floats (true when either side is NaN; `-0.0 == 0.0`). -/
def fne (x y : F) : Bool := fpcmp x y != some .eq

/-- IEEE `x > y`. -/
def fgt (x y : F) : Bool := fpcmp x y == some .gt

/-- `first_difference` (`value.rs:1821`): the first pair with `x != y` decides by `x > y`. -/
def firstDifference : List F → List F → Ordering
  | x :: xs, y :: ys => if fne x y then (if fgt x y then .gt else .lt) else firstDifference xs ys
  | _, _ => .eq

/-- `compare_vecf32` (`value.rs:1817`): length, then `first_difference` (the chunked
scan equals the plain one: proofs/value_math `chunked_eq_firstDiff`). -/
def cmpVec (a b : List F) : Ordering :=
  match cmpI a.length b.length with
  | .eq => firstDifference a b
  | o => o

/-- The `(Point, Point)` arm of `compare_value` (`value.rs:1247`): longitude, then
latitude, each by `partial_cmp`; NaN answers `(Less, NaN)`. -/
def cmpPoint (alat alon blat blon : F) : Ordering × Flag :=
  match fpcmp alon blon with
  | some .eq => match fpcmp alat blat with
    | some o => (o, .none)
    | none => (.lt, .nan)
  | some o => (o, .none)
  | none => (.lt, .nan)

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
  | .vecf32 a, .vecf32 b => (cmpVec a b, .none)
  | .point alat alon, .point blat blon => cmpPoint alat alon blat blon
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
def floatLane (a b : F) : Ordering := (fpcmp a b).getD .lt

theorem floatLane_eq_cmpV (a b : F) : floatLane a b = sortCmp (.flt a) (.flt b) := by
  simp only [floatLane, sortCmp, cmpV, cmpF]
  cases fpcmp a b <;> rfl

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

/-- `f64::NAN.to_bits()`. -/
def NAN_BITS : UInt64 := 0x7ff8000000000000

/-- IEEE bits of an integral, representable float (`f32`). -/
def ieeeBits32 (n : Int) : Nat :=
  let m := n.natAbs
  if m = 0 then (if n < 0 then 0x80000000 else 0) else
    let L := bitlen m
    let mant := m / 2 ^ (L - 24) * 2 ^ (24 - L)
    let exp := L - 1 + 127
    let s : Nat := if n < 0 then 2 ^ 31 else 0
    s + exp * 2 ^ 23 + (mant - 2 ^ 23)

/-- `f64::to_bits`. NaN, zeros, integral values and infinities get their IEEE
bits; a non-integral value gets *some* function of `(lo, r)` — the model only
needs `to_bits` to be a function of the value there. -/
def bits64 : F → UInt64
  | .nan neg p => UInt64.ofNat ((if neg then 2 ^ 63 else 0) + 0x7FF0000000000000 + (p % (2 ^ 52 - 1) + 1))
  | .negZero => 0x8000000000000000
  | .num n => ieeeBits n
  | .frac lo r => UInt64.ofNat (lo.natAbs * 4096 + r)
  | .negInf => 0xFFF0000000000000
  | .posInf => 0x7FF0000000000000

/-- `f32::to_bits` (same conventions as `bits64`). -/
def bits32 : F → Nat
  | .nan neg p => (if neg then 2 ^ 31 else 0) + 0x7F800000 + (p % (2 ^ 23 - 1) + 1)
  | .negZero => 0x80000000
  | .num n => ieeeBits32 n
  | .frac lo r => lo.natAbs * 4096 + r
  | .negInf => 0xFF800000
  | .posInf => 0x7F800000

example : bits64 F.qnan = NAN_BITS := by decide
example : bits32 (.nan false (2 ^ 22 - 1)) = 0x7fc00000 := by decide   -- f32::NAN

/-- `x as i64` (saturating; NaN ↦ 0; truncation toward zero). -/
def asI64 : F → Int
  | .nan _ _ => 0
  | .negZero => 0
  | .num n => satI64 n
  | .frac lo _ => satI64 (if 0 ≤ lo then lo else lo + 1)
  | .negInf => I64MIN
  | .posInf => I64MAX

/-- `x - casted as f64 == 0.0`: only for zeros and integral values that
`casted as f64` gives back (NaN, ±inf and fractions leave a non-zero or NaN). -/
def diffZero : F → Bool
  | .negZero => true
  | .num n => roundF (satI64 n) = n
  | _ => false

/-- `Self::Float(x)` arm of `Hash for Value` (`value.rs:836`): hash as the
integer if `x - (x as i64) as f64 == 0`, else `to_bits` — so every NaN sign and
payload hashes apart. -/
def floatWord (f : F) : UInt64 := if diffZero f then w (asI64 f) else bits64 f

example : floatWord F.qnan = NAN_BITS := by decide

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
  | .vecf32 xs => [9] ++ xs.map (fun f => UInt64.ofNat (bits32 f))
  | .point lat lon => [10, UInt64.ofNat (bits32 lat), UInt64.ofNat (bits32 lon)]

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
    sortCmp (.flt F.qnan) (.flt (.num 1)) = .lt ∧
    sortCmp (.flt (.num 1)) (.flt F.qnan) = .lt ∧
    sortCmp (.flt F.qnan) (.flt F.qnan) = .lt := by decide

/-- ...so the ORDER BY comparator is not antisymmetric (`cmp b a ≠ (cmp a b).swap`). -/
theorem sortCmp_not_antisymmetric :
    ¬ ∀ a b : V, sortCmp b a = (sortCmp a b).swap := fun h => by
  have := h (.flt F.qnan) (.flt (.num 1)); revert this; decide

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
    eqV (.flt F.qnan) (.flt F.qnan) = false ∧
    hwords (.flt F.qnan) = hwords (.flt F.qnan) := by decide

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
    (match minFold [.flt (.num 1), .flt F.qnan, .flt (.num 3)] with
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
    | nan _ _ | negZero | frac _ _ | negInf | posInf => simp [sane] at ha
    | num n => cases b with
      | flt g => cases g with
        | nan _ _ | negZero | frac _ _ | negInf | posInf => simp [sane] at hb
        | num m => simp [sortCmp, cmpV, cmpF, key, lexK]
      | int j =>
          obtain ⟨hj, -⟩ := hb
          simp [sortCmp, cmpV, cmpF, key, lexK, hj]
      | list _ | map _ _ | vecf32 _ | point _ _ => simp [sane] at hb
      | _ => simp [sortCmp, cmpV, key, lexK, cmpN, cmpI, order, V.tag, orderTag, isNull]
  | int i =>
      obtain ⟨hi, -⟩ := ha
      cases b with
      | flt g => cases g with
        | nan _ _ | negZero | frac _ _ | negInf | posInf => simp [sane] at hb
        | num m => simp [sortCmp, cmpV, cmpF, key, lexK, hi]
      | int j => simp [sortCmp, cmpV, key, lexK]
      | list _ | map _ _ | vecf32 _ | point _ _ => simp [sane] at hb
      | _ => simp [sortCmp, cmpV, key, lexK, cmpN, cmpI, order, V.tag, orderTag, isNull]
  | list _ | map _ _ | vecf32 _ | point _ _ => simp [sane] at ha
  | bool x => cases b with
      | flt g => cases g with
        | nan _ _ | negZero | frac _ _ | negInf | posInf => simp [sane] at hb
        | num m => simp [sortCmp, cmpV, key, lexK, cmpN, cmpI, order, V.tag, orderTag, isNull]
      | list _ | map _ _ | vecf32 _ | point _ _ => simp [sane] at hb
      | bool y => cases x <;> cases y <;> decide
      | _ => simp [sortCmp, cmpV, key, lexK, cmpN, cmpI, order, V.tag, orderTag, isNull]
  | str x => cases b with
      | flt g => cases g with
        | nan _ _ | negZero | frac _ _ | negInf | posInf => simp [sane] at hb
        | num m => simp [sortCmp, cmpV, key, lexK, cmpN, cmpI, order, V.tag, orderTag, isNull]
      | list _ | map _ _ | vecf32 _ | point _ _ => simp [sane] at hb
      | str y => simp [sortCmp, cmpV, key, lexK, cmpN]
      | _ => simp [sortCmp, cmpV, key, lexK, cmpN, cmpI, order, V.tag, orderTag, isNull]
  | node x => cases b with
      | flt g => cases g with
        | nan _ _ | negZero | frac _ _ | negInf | posInf => simp [sane] at hb
        | num m => simp [sortCmp, cmpV, key, lexK, cmpN, cmpI, order, V.tag, orderTag, isNull]
      | list _ | map _ _ | vecf32 _ | point _ _ => simp [sane] at hb
      | node y => simp [sortCmp, cmpV, key, lexK, cmpN]
      | _ => simp [sortCmp, cmpV, key, lexK, cmpN, cmpI, order, V.tag, orderTag, isNull]
  | null => cases b with
      | flt g => cases g with
        | nan _ _ | negZero | frac _ _ | negInf | posInf => simp [sane] at hb
        | num m => simp [sortCmp, cmpV, key, lexK, cmpN, cmpI, order, V.tag, orderTag, isNull]
      | list _ | map _ _ | vecf32 _ | point _ _ => simp [sane] at hb
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
    | nan _ _ | negZero | frac _ _ | negInf | posInf => simp [sane] at ha
    | num n => cases b with
      | flt g => cases g with
        | nan _ _ | negZero | frac _ _ | negInf | posInf => simp [sane] at hb
        | num m => simp [key] at hk; subst hk; rfl
      | int j =>
          obtain ⟨hj, lo, hi⟩ := hb
          simp [key] at hk; subst hk
          have hs : satI64 n = n := satI64_id _ lo (by assumption)
          simp [hwords, floatWord, diffZero, asI64, hs, hj]
      | list _ | map _ _ | vecf32 _ | point _ _ => simp [sane] at hb
      | _ => simp [key] at hk
  | int i =>
      obtain ⟨hi, lo, up⟩ := ha
      cases b with
      | flt g => cases g with
        | nan _ _ | negZero | frac _ _ | negInf | posInf => simp [sane] at hb
        | num m =>
          simp [key] at hk; subst hk
          have hs : satI64 i = i := satI64_id _ lo (by assumption)
          simp [hwords, floatWord, diffZero, asI64, hs, hi]
      | int j => simp [key] at hk; subst hk; rfl
      | list _ | map _ _ | vecf32 _ | point _ _ => simp [sane] at hb
      | _ => simp [key] at hk
  | list _ | map _ _ | vecf32 _ | point _ _ => simp [sane] at ha
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
