import FalkorTemporal.Calendar
/-!
# `pad` / `twoDigits` and the four-digit identity

`fourDigits_aux` is a brute-force kernel check over 100 x 100 cases (~45 s in one piece).
It is cut into ten slices of ten values of `a` (`PadChunk0..9.lean`, one file each, so Lake
runs them in parallel and alongside the era enumeration) and reassembled here.
-/
namespace FalkorTemporal.Fns

/-- `{:02}` / `{:04}` of a non-negative number: left-pad with `0` (no truncation). -/
def pad (w : Nat) (n : Nat) : String :=
  let s := toString n
  String.ofList (List.replicate (w - s.length) '0') ++ s

/-- `write_two_digits` (value.rs:853): `v / 10 % 10` and `v % 10` as ASCII digits. -/
def twoDigits (v : Nat) : String :=
  String.ofList [Char.ofNat (48 + v / 10 % 10), Char.ofNat (48 + v % 10)]

/-- Slice `k` of `fourDigits_aux`: `a = 10 k + j` for `j < 10`. -/
def FourDigitsSlice (k : Nat) : Prop :=
  ∀ j, j < 10 → ∀ b, b < 100 → twoDigits (10 * k + j) ++ twoDigits b = pad 4 ((10 * k + j) * 100 + b)

end FalkorTemporal.Fns
