import FalkorTemporal.Calendar
/-! Machine-checked enumeration over one 400-year era (146097 days). Evaluated by the
kernel (`decide +kernel`: no `native_decide`, no `ofReduceBool` axiom). -/
namespace FalkorTemporal

/-- Forward check for era day `n`: the pieces of `civil_from_days` are in range and the
day is a real day of its month. -/
def fwdOk (n : Nat) : Bool :=
  let doe : Int := n
  decide (0 ≤ doyOf doe) && decide (doyOf doe ≤ 365) && decide (0 ≤ mpOf doe) &&
  decide (mpOf doe ≤ 11) && decide (1 ≤ dOf doe) && decide (dOf doe ≤ dimMarch (yoeOf doe) (mpOf doe))

def gridYoe (k : Nat) : Int := (k / 372 : Nat)
def gridMp (k : Nat) : Int := (k / 31 % 12 : Nat)
def gridD (k : Nat) : Int := (k % 31 + 1 : Nat)

/-- Backward check for grid point `k` = (yoe, mp, d): decomposing the era day of a real
date gives the date back. -/
def bwdOk (k : Nat) : Bool :=
  let yoe := gridYoe k; let mp := gridMp k; let d := gridD k
  let x := dfcEra yoe mp d
  if d ≤ dimMarch yoe mp then
    decide (yoeOf x = yoe) && decide (mpOf x = mp) && decide (dOf x = d)
  else true

def allRange (p : Nat → Bool) (lo : Nat) : Nat → Bool
  | 0 => true
  | n+1 => p (lo + n) && allRange p lo n

theorem allRange_spec (p : Nat → Bool) (lo : Nat) : ∀ n, allRange p lo n = true →
    ∀ k, lo ≤ k → k < lo + n → p k = true := by
  intro n; induction n with
  | zero => intro _ k h1 h2; omega
  | succ n ih =>
    intro h k h1 h2
    simp only [allRange, Bool.and_eq_true] at h
    by_cases hk : k = lo + n
    · subst hk; exact h.1
    · exact ih h.2 k h1 (by omega)

end FalkorTemporal
