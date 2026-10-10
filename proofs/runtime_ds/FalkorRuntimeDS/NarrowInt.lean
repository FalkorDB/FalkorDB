/-
# `width_for` (`graph/src/narrow_int.rs:19`) picks the narrowest lossless width

| here | there |
| --- | --- |
| `widthFor`  | `narrow_int::width_for` (narrow_int.rs:19) |
| `narrow`    | the `as u8` / `as u16` / `as u32` / `u64` truncation a writer applies at that width |

`max : u64` is modelled as `Nat` with `max < 2^64`.
-/
namespace FalkorRuntimeDS.NarrowIntModel

/-- narrow_int.rs:19 — the four match arms, in order. -/
def widthFor (max : Nat) : Nat :=
  if max ≤ 0xFF then 1
  else if max ≤ 0xFFFF then 2
  else if max ≤ 0xFFFF_FFFF then 4
  else 8

/-- Truncating `as` cast to `w` bytes. -/
def narrow (w x : Nat) : Nat := x % 2 ^ (8 * w)

theorem widthFor_mem (m : Nat) : widthFor m = 1 ∨ widthFor m = 2 ∨ widthFor m = 4 ∨ widthFor m = 8 := by
  unfold widthFor; split <;> (try split) <;> (try split) <;> simp

/-- The chosen width holds `max` (every `u64` fits, since the last arm is 8 bytes). -/
theorem widthFor_fits (m : Nat) (hm : m < 2 ^ 64) : m < 2 ^ (8 * widthFor m) := by
  unfold widthFor
  split
  · simp; omega
  · split
    · simp; omega
    · split
      · simp; omega
      · simpa using hm

/-- …and it is the narrowest power-of-two width that does. -/
theorem widthFor_minimal (m w : Nat) (hw : w = 1 ∨ w = 2 ∨ w = 4 ∨ w = 8)
    (hlt : w < widthFor m) : 2 ^ (8 * w) ≤ m := by
  unfold widthFor at hlt
  rcases hw with rfl | rfl | rfl | rfl <;>
    (split at hlt <;> (try split at hlt) <;> (try split at hlt) <;> simp_all <;> omega)

/-- Monotone: every value `≤ max` fits in `widthFor max`. -/
theorem widthFor_mono {a b : Nat} (h : a ≤ b) : widthFor a ≤ widthFor b := by
  unfold widthFor
  split <;> split <;> (try split) <;> (try split) <;> (try split) <;> (try split) <;> omega

/-- Narrowing any `x ≤ max` to `widthFor max` bytes is lossless. -/
theorem narrow_lossless (m x : Nat) (hm : m < 2 ^ 64) (hx : x ≤ m) :
    narrow (widthFor m) x = x :=
  Nat.mod_eq_of_lt (Nat.lt_of_le_of_lt hx (widthFor_fits m hm))

/-- Boundary table (same as the Rust unit test `boundaries`). -/
example : widthFor 0 = 1 ∧ widthFor 0xFF = 1 ∧ widthFor 0x100 = 2 ∧ widthFor 0xFFFF = 2 ∧
    widthFor 0x1_0000 = 4 ∧ widthFor 0xFFFF_FFFF = 4 ∧ widthFor 0x1_0000_0000 = 8 ∧
    widthFor (2 ^ 64 - 1) = 8 := by decide

end FalkorRuntimeDS.NarrowIntModel
