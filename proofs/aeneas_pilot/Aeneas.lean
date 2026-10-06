module
public import Lean
/-!
# Minimal shim of the Aeneas standard library (NOT the real one)

The real library (`AeneasVerif/aeneas` `backends/lean`, nightly-2026.09.24-557f7a1)
needs Lean v4.31.0 + Mathlib v4.31.0. This shim reproduces, on core Lean 4.34.0,
only what `Extracted/NarrowInt.lean` touches, mirroring the real definitions:

| shim                  | real Aeneas                                                          |
|-----------------------|----------------------------------------------------------------------|
| `Error`               | `Aeneas/Std/Primitives.lean:65` (same constructors)                  |
| `Result` ok/fail/div  | `Primitives.lean:92` — real one is an `ITree`; same `ok` observable  |
| `UScalar ty` / `.bv`  | `Aeneas/Std/Scalar/Core.lean:71` (`bv : BitVec ty.numBits`)           |
| `UScalar.val`         | `Scalar/Core.lean:76` (`x.bv.toNat`)                                 |
| `LE (UScalar ty)`     | `Scalar/Core.lean:897` (`a.val ≤ b.val`)                              |
| `n#u64`, `n#u8`       | `Scalar/Notations.lean:27` (`U64.ofNat n`)                           |

The `set_option linter.*` names the generated header sets are Mathlib/Aeneas
options; they are registered here so the header elaborates.
-/
@[expose] public section

register_option linter.style.whitespace : Bool := { defValue := false }
register_option linter.style.setOption : Bool := { defValue := false }
register_option linter.style.longLine : Bool := { defValue := false }
register_option linter.dupNamespace : Bool := { defValue := false }
register_option linter.hashCommand : Bool := { defValue := false }

namespace Aeneas
namespace Std

inductive Error where
  | assertionFailure | integerOverflow | divisionByZero | arrayOutOfBounds
  | maximumSizeExceeded | panic | undef
deriving Repr, BEq, DecidableEq

inductive Result (α : Type u) where
  | ok (a : α)
  | fail (e : Error)
  | div
deriving Repr, BEq, DecidableEq

instance : Monad Result where
  pure := .ok
  bind x f := match x with
    | .ok a => f a
    | .fail e => .fail e
    | .div => .div

inductive ControlFlow (α β : Type) where
  | cont (a : α)
  | done (b : β)

inductive UScalarTy where
  | Usize | U8 | U16 | U32 | U64 | U128

/-- 64-bit target, as in Aeneas' default. -/
def UScalarTy.numBits : UScalarTy → Nat
  | .Usize => 64 | .U8 => 8 | .U16 => 16 | .U32 => 32 | .U64 => 64 | .U128 => 128

structure UScalar (ty : UScalarTy) where
  bv : BitVec ty.numBits
deriving Repr, BEq, DecidableEq

def UScalar.val {ty} (x : UScalar ty) : Nat := x.bv.toNat

instance {ty} : LE (UScalar ty) where le a b := LE.le a.val b.val
instance {ty} (a b : UScalar ty) : Decidable (a ≤ b) :=
  inferInstanceAs (Decidable (a.val ≤ b.val))

def UScalar.ofNat {ty : UScalarTy} (x : Nat) (_h : x < 2 ^ ty.numBits := by decide) :
    UScalar ty := { bv := BitVec.ofNat _ x }

abbrev U8 := UScalar .U8
abbrev U64 := UScalar .U64
abbrev Usize := UScalar .Usize
def U8.ofNat (x : Nat) (h : x < 2 ^ 8 := by decide) : U8 := UScalar.ofNat x h
def U64.ofNat (x : Nat) (h : x < 2 ^ 64 := by decide) : U64 := UScalar.ofNat x h

macro:max x:term:max noWs "#u8" : term => `(U8.ofNat $x)
macro:max x:term:max noWs "#u64" : term => `(U64.ofNat $x)

end Std
end Aeneas
