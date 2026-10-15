import FalkorCore.Order
import FalkorCore.Int64
/-!
# The f64/f32 model shared by every project (from proofs/value_order, extended 2026-10-11)

`F` represents every class of IEEE value that ordering or hashing can tell apart: NaN with its
sign bit and payload, both zeros, integral values, non-integral finite values and both
infinities. `fpcmp` is `partial_cmp`, `cmpF` is `compare_floats` (`value.rs:1801`),
`roundF` is `i as f64`, `bits64`/`bits32` are `to_bits`, `asI64` is `x as i64`.
No axiom: non-integral values are abstract (`frac lo r`).
-/
namespace FalkorCore

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

/-! ## Bit layouts and casts -/


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

end FalkorCore
