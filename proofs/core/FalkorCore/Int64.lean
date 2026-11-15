/-!
# `i64` as a mathematical `Int`

Rust's `i64` is an `Int` with the range predicate `InI64`; a release build wraps
(`wrap`, two's complement onto `[-2^63, 2^63)`), so `wrapping_*` are `wrap ∘ (exact op)`
(`wrapping_div` wraps only `MIN / -1`, `wrapping_rem` gives `0` for `MIN % -1`).
`w` is the two's-complement `i64 → u64` reinterpretation (`as u64`, hash words).
Merged from proofs/value_order, proofs/expr_semantics, proofs/value_math and
proofs/functions_str_list (pilot of the shared core).
-/
namespace FalkorCore

def I64MIN : Int := -9223372036854775808
def I64MAX : Int := 9223372036854775807

/-- In `i64` range. The bounds are literals so `unfold InI64; omega` works. -/
def InI64 (x : Int) : Prop := -9223372036854775808 ≤ x ∧ x ≤ 9223372036854775807

instance : DecidablePred InI64 := fun x => by unfold InI64; infer_instance

theorem InI64_iff {x : Int} : InI64 x ↔ I64MIN ≤ x ∧ x ≤ I64MAX := Iff.rfl

/-- Two's-complement wrap of a mathematical integer onto `i64`. -/
def wrap (x : Int) : Int := (x + 9223372036854775808) % 18446744073709551616 - 9223372036854775808

theorem wrap_inRange (x : Int) : InI64 (wrap x) := by
  unfold InI64 wrap; omega

/-- Wrapping is the identity on in-range results. -/
theorem wrap_id {x : Int} (h : InI64 x) : wrap x = x := by
  unfold InI64 at h; unfold wrap; omega

def wAdd (a b : Int) : Int := wrap (a + b)
def wSub (a b : Int) : Int := wrap (a - b)
def wMul (a b : Int) : Int := wrap (a * b)
/-- `i64::wrapping_div` (`b ≠ 0`): truncating division; only `MIN / -1` wraps. -/
def wDiv (a b : Int) : Int := wrap (Int.tdiv a b)
/-- `i64::wrapping_rem` (`b ≠ 0`): truncating remainder (`MIN % -1 = 0`). -/
def wRem (a b : Int) : Int := wrap (Int.tmod a b)

/-- Rust's saturating `f as i64`. -/
def satI64 (n : Int) : Int := if n < I64MIN then I64MIN else if n > I64MAX then I64MAX else n

/-- Two's-complement `i64 → u64`. -/
def w (i : Int) : UInt64 := UInt64.ofNat (i % (2 ^ 64 : Int)).toNat

end FalkorCore
