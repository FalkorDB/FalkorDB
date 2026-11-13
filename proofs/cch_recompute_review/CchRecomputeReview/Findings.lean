import CchRecomputeReview.Main
import CchRecomputeReview.Weights

/-!
# Findings, as theorems

* **R-F1** (stack code @08c..tip; confirmed live). `recustomize_seeded` tests "changed" with `f64 !=`
  (`partial.rs:221-222` @08c) and writes each value only where `!=` says it differs (`:225-230`). `+0.0 != -0.0`
  is false, so a seed that flips the sign of a zero is neither written nor propagated, while `relax` stores it.
  The result is `==`-equal to the full customization (Theorem 2, agreement form, with `R = (· == ·)`), middles
  included, but not bit-identical, which `partial.rs:19-21, :83` and 03 §6b ("bit-identical to a full
  customization of H′") claim. The 3-rank trace below is the live repro's graph (the scoped pass keeps `0x0`
  where `customize` gives `0x8000000000000000`, on the changed arc and on the fill arc above it). With a
  bitwise test (`to_bits()`), the same run is bit-identical (`R_F1_bitwise_fixes_it`), as
  `scoped_after_seed_change_bitwise` proves in general.
* **R-W1** (03 §6b wording, design-level, no code to reproduce). "Its middle is the first `x` to win a strict
  `<` in increasing `x`" read literally names the first apex that beats the running value. The code
  (`customize.rs:298-301`, `partial.rs:206-210`) records the *last* apex that beats it: the first apex attaining
  the final minimum. `R_W1_middle_is_not_the_first_winner` has apex 0 win first (5 < ∞) and apex 1 win last
  (3 < 5): the middle is 1. `customize.rs:66-71`'s own wording ("among detours that tie, the lowest apex") is
  the right one.
-/

namespace CchRecompute

/-! ## R-F1: the live repro's graph, in rank space

Ranks 0, 1, 2 (node-id order). Edges `1 → 0` (the changed weight, `+0.0` then `-0.0`), `0 → 1 = 5`,
`0 → 2 = -0.0`, `2 → 0 = 5`. Eliminating 0 adds the fill arc `(1, 2)`, whose upward weight is
`dn(0,1) + up(0,2) = w(1→0) + w(0→2)`. -/

def up3 : Nat → List Nat
  | 0 => [1, 2]
  | 1 => [2]
  | _ => []

def T3 : Topo := ⟨3, up3, fun r => (List.range 3).filter (fun x => decide (r ∈ up3 x))⟩

theorem T3_valid : T3.Valid where
  sorted x := by
    match x with
    | 0 => decide
    | 1 => decide
    | _ + 2 => simp [T3, up3]
  above x y h := by
    match x with
    | 0 => simp [T3, up3] at h; omega
    | 1 => simp [T3, up3] at h; omega
    | _ + 2 => simp [T3, up3] at h
  bound x y h := by
    match x with
    | 0 => simp [T3, up3] at h; simp [T3]; omega
    | 1 => simp [T3, up3] at h; simp [T3]; omega
    | _ + 2 => simp [T3, up3] at h

theorem T3_chordal : T3.Chordal := by
  intro x y z hy hz hlt
  match x with
  | 0 =>
    simp [T3, up3] at hy hz
    rcases hy with rfl | rfl <;> rcases hz with rfl | rfl <;> simp_all [T3, up3]
  | 1 => simp [T3, up3] at hy hz; omega
  | _ + 2 => simp [T3, up3] at hy

theorem T3_down : T3.DownOK := fun _ => rfl

/-- Seeds: `(0, 1)` is `(w(0→1), w(1→0)) = (5, ±0.0)`, `(0, 2)` is `(-0.0, 5)`, the rest `+∞`. -/
def S3 (neg : Bool) : Seeds SZ := fun lo hi =>
  if lo = 0 ∧ hi = 1 then (.pos 4, .zero neg)
  else if lo = 0 ∧ hi = 2 then (.zero true, .pos 4)
  else (.inf, .inf)

/-- `relax` for the old (`+0.0`) or new (`-0.0`) weight. -/
def full3 (neg : Bool) : Metric SZ := (relax SZ.ops T3 (seedM (S3 neg))).getD (seedM (S3 neg))

/-- The scoped pass with the code's `!=`, front = the changed arc. -/
def scoped3 : St SZ := run SZ.ops T3 (S3 true) pickMin 4 (init (full3 false) [(0, 1)])

/-- Before: `1 → 0 → 2 = +0.0 + -0.0 = +0.0`. -/
theorem R_F1_before : (full3 false 1 2).up = .zero false := by decide

/-- `relax` for the new weight stores `-0.0` on the changed arc and on the fill arc above it. -/
theorem R_F1_full : (full3 true 0 1).dn = .zero true ∧ (full3 true 1 2).up = .zero true := by decide

/-- The scoped pass keeps `+0.0` on both, and its heap is empty: it did finish. -/
theorem R_F1_scoped : (scoped3.M 0 1).dn = .zero false ∧ (scoped3.M 1 2).up = .zero false ∧
    scoped3.heap = [] := by decide

/-- **R-F1.** Not bit-identical … -/
theorem R_F1_not_bit_identical : scoped3.M 0 1 ≠ full3 true 0 1 ∧ scoped3.M 1 2 ≠ full3 true 1 2 := by
  decide

/-- … but `==`-equal on every arc, middles included: Theorem 2's agreement form, instantiated. -/
theorem R_F1_agrees : ∀ lo hi, hi ∈ T3.up lo →
    ER (fun a b => SZ.beq a b = true) (scoped3.M lo hi) (full3 true lo hi) := by
  have h0 : relax SZ.ops T3 (seedM (S3 false)) = some (full3 false) := by
    unfold full3; cases h : relax SZ.ops T3 (seedM (S3 false)) with
    | none => exact absurd h (by decide)
    | some M => rfl
  have h1 : relax SZ.ops T3 (seedM (S3 true)) = some (full3 true) := by
    unfold full3; cases h : relax SZ.ops T3 (seedM (S3 true)) with
    | none => exact absurd h (by decide)
    | some M => rfl
  apply scoped_after_seed_change T3_valid T3_chordal T3_down SZ.agree
    (relax_isFull _ T3_valid _ h0) (relax_isFull _ T3_valid _ h1)
    (by decide) (by decide) _ pickMin_spec (by decide)
  intro lo hi harc hne
  match lo, hi with
  | 0, 1 => simp
  | 0, 2 => exact absurd rfl hne
  | 1, 2 => exact absurd rfl hne
  | 0, 0 | 0, _ + 3 | 1, 0 | 1, 1 | 1, _ + 3 | _ + 2, _ => simp [T3, up3] at harc

/-- With the change test made bitwise, the same run is bit-identical. -/
theorem R_F1_bitwise_fixes_it :
    let full := fun neg => (relax (bitwise SZ.ops) T3 (seedM (S3 neg))).getD (seedM (S3 neg))
    let st := run (bitwise SZ.ops) T3 (S3 true) pickMin 4 (init (full false) [(0, 1)])
    st.M 0 1 = full true 0 1 ∧ st.M 0 2 = full true 0 2 ∧ st.M 1 2 = full true 1 2 := by
  decide

/-! ## R-W1: the middle is the last strict winner, not the first -/

def up4 : Nat → List Nat
  | 0 => [2, 3]
  | 1 => [2, 3]
  | 2 => [3]
  | _ => []

def T4 : Topo := ⟨4, up4, fun r => (List.range 4).filter (fun x => decide (r ∈ up4 x))⟩

/-- Apex 0 offers `2 + 3 = 5`, apex 1 offers `1 + 2 = 3`; arc `(2, 3)` has no edge. -/
def S4 : Seeds NInf := fun lo hi =>
  if lo = 0 ∧ hi = 2 then (.inf, .fin 2)
  else if lo = 0 ∧ hi = 3 then (.fin 3, .inf)
  else if lo = 1 ∧ hi = 2 then (.inf, .fin 1)
  else if lo = 1 ∧ hi = 3 then (.fin 2, .inf)
  else (.inf, .inf)

/-- **R-W1.** Apex 0 is the first to win a strict `<` (5 < ∞), yet the middle is apex 1. -/
theorem R_W1_middle_is_not_the_first_winner :
    NInf.ops.lt (NInf.ops.add (.fin 2) (.fin 3)) .inf = true ∧
    (recomp NInf.ops T4 S4 (seedM S4) 2 3).up = .fin 3 ∧
    (recomp NInf.ops T4 S4 (seedM S4) 2 3).mup = some 1 := by
  decide

end CchRecompute
