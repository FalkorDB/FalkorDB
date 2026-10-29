/-!
# Orderings on integers, and generic comparator laws

`cmpI`/`cmpN`/`cmpB` are `Ord::cmp` on `i64`/`usize`/ids/`bool`, written out so proofs can
`split`; `cmp3` is the lexicographic order on `Nat × Int × Nat` (the float position key).
`FalkorCore.Laws` (below, `Laws.lean`) proves the comparator laws (`Lw`, `thn`, `lexA`).
Moved from proofs/value_order `Model.lean` (pilot of the shared core).
-/
namespace FalkorCore

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

def cmp3 (a b : Nat × Int × Nat) : Ordering :=
  match cmpI a.1 b.1 with
  | .eq => match cmpI a.2.1 b.2.1 with
    | .eq => cmpI a.2.2 b.2.2
    | o => o
  | o => o


end FalkorCore
