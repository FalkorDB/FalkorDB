/-
# Fold policy (`versioned_matrix.rs:140-200`)

| here | there |
| --- | --- |
| `satMul`            | `u64::saturating_mul` |
| `foldBalance`       | `fold_balance` (:175) |
| `shouldFold`        | `should_fold` (:154), `WRITE_FOLD_K = 20_500_000` |
| `shouldFoldRead`    | `should_fold_read` (:166), `READ_FOLD_K = 82_000` |
| `deltaDominatesBase`| `delta_dominates_base` (:195), `MIN_FOLD_DELTA = 256` |

The policy is performance-only (the layer theorems in `DeltaProofs` hold
for every fold decision); what is proved here is that it means what its
docs say: no fold on a read-only transaction or a tiny delta, the escape
hatch always folds, monotone in the delta size, and the read path is
strictly looser-to-fold than the write path (write ⇒ read).
-/
namespace VMPolicy

def U64MAX : Nat := 2 ^ 64 - 1
def WRITE_FOLD_K : Nat := 20500000
def READ_FOLD_K : Nat := 82000
def MIN_FOLD_DELTA : Nat := 256

/-- `u64::saturating_mul`. -/
def satMul (a b : Nat) : Nat := min (a * b) U64MAX

def foldBalance (d tx base k : Nat) : Bool :=
  decide (tx > 0) && decide (d ≥ MIN_FOLD_DELTA) &&
    (decide (satMul d 2 ≥ base) || decide (satMul d d ≥ satMul k tx))

def shouldFold (d tx base : Nat) : Bool := foldBalance d tx base WRITE_FOLD_K
def shouldFoldRead (d tx base : Nat) : Bool := foldBalance d tx base READ_FOLD_K

def deltaDominatesBase (d base : Nat) : Bool :=
  decide (d ≥ MIN_FOLD_DELTA) && decide (satMul d 2 ≥ base)

theorem satMul_mono {a a' b b' : Nat} (ha : a ≤ a') (hb : b ≤ b') : satMul a b ≤ satMul a' b' := by
  unfold satMul; have := Nat.mul_le_mul ha hb; omega

theorem satMul_exact {a b : Nat} (h : a * b ≤ U64MAX) : satMul a b = a * b := by
  unfold satMul; omega

theorem foldBalance_readOnly (d base k : Nat) : foldBalance d 0 base k = false := by
  simp [foldBalance]

theorem foldBalance_tiny {d : Nat} (h : d < MIN_FOLD_DELTA) (tx base k : Nat) :
    foldBalance d tx base k = false := by
  simp [foldBalance]; intro _ h2; exact absurd h2 (by omega)

/-- The escape hatch: a delta comparable to the base folds whatever `k` says. -/
theorem dominates_folds {d tx base : Nat} (htx : 0 < tx) (k : Nat)
    (h : deltaDominatesBase d base = true) : foldBalance d tx base k = true := by
  simp [deltaDominatesBase] at h; simp [foldBalance]; omega

theorem foldBalance_mono_delta {d d' tx base k : Nat} (hd : d ≤ d')
    (h : foldBalance d tx base k = true) : foldBalance d' tx base k = true := by
  simp [foldBalance] at *
  have h1 := satMul_mono hd (Nat.le_refl 2)
  have h2 := satMul_mono hd hd
  refine ⟨⟨h.1.1, by omega⟩, ?_⟩
  rcases h.2 with h3 | h3 <;> omega

theorem foldBalance_mono_k {d tx base k k' : Nat} (hk : k' ≤ k)
    (h : foldBalance d tx base k = true) : foldBalance d tx base k' = true := by
  simp [foldBalance] at *
  have := satMul_mono hk (Nat.le_refl tx)
  refine ⟨h.1, ?_⟩; rcases h.2 with h3 | h3 <;> omega

theorem foldBalance_mono_base {d tx base base' k : Nat} (hb : base' ≤ base)
    (h : foldBalance d tx base k = true) : foldBalance d tx base' k = true := by
  simp [foldBalance] at *
  refine ⟨h.1, ?_⟩; rcases h.2 with h3 | h3 <;> omega

/-- Whatever the write path folds, the read path folds too. -/
theorem write_implies_read {d tx base : Nat} (h : shouldFold d tx base = true) :
    shouldFoldRead d tx base = true :=
  foldBalance_mono_k (by decide) h

/-- Below saturation the rule is exactly `|delta| ≥ sqrt(k·tx) ∨ 2|delta| ≥ base`. -/
theorem foldBalance_exact {d tx base k : Nat} (hd : d < 2 ^ 32) (hk : k * tx ≤ U64MAX) :
    foldBalance d tx base k = true ↔
      0 < tx ∧ MIN_FOLD_DELTA ≤ d ∧ (base ≤ 2 * d ∨ k * tx ≤ d * d) := by
  have hdd : d * d ≤ U64MAX := by
    have := Nat.mul_lt_mul_of_lt_of_le hd (Nat.le_of_lt hd) (by omega)
    simp [U64MAX]; omega
  have h2 : d * 2 ≤ U64MAX := by simp [U64MAX]; omega
  simp [foldBalance, satMul_exact hdd, satMul_exact h2, satMul_exact hk]
  constructor
  · rintro ⟨⟨a, b⟩, c⟩; exact ⟨a, b, by omega⟩
  · rintro ⟨a, b, c⟩; exact ⟨⟨a, b⟩, by omega⟩

/-! Concrete checks mirroring the unit tests (:1509-1590). -/
example : shouldFoldRead 286 1 (10 ^ 9) = false := by decide
example : shouldFoldRead 287 1 (10 ^ 9) = true := by decide
example : shouldFold 4528 1 (10 ^ 9) = true := by decide
example : shouldFold 4527 1 (10 ^ 9) = false := by decide
example : shouldFold 300 5 600 = true := by decide   -- delta comparable to base
example : shouldFold 255 5 0 = false := by decide    -- below the floor
example : deltaDominatesBase 256 512 = true := by decide

end VMPolicy
