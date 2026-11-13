import CchRecomputeReview.Basic
import CchRecomputeReview.Scoped

/-!
# Weight types and the heap

* `NInf` (`ℕ ∪ {+∞}`): exact arithmetic. Its change test is equality, so `Agree NInf.ops Eq` and `TopLaws`
  hold: every exact theorem applies to it with no premise left.
* `SZ`: non-negative doubles reduced to what decides bit identity here: `+0.0`, `-0.0`, a positive finite value
  and `+∞`, with IEEE addition on zeros (`-0 + -0 = -0`, otherwise `+0`) and numeric `<`/`==`, under which
  `+0.0 == -0.0`. It satisfies `Agree` for `R = (· == ·)` and `TopLaws`, but its `==` is not sound for equality:
  the premise of the exact theorems fails exactly as it does for `f64`. `Findings.lean` uses it.
* `pickMin`: the heap's pop by least `(lo, hi)`, 03 §6b "Algorithm" 2's key (`id` is determined by `(lo, hi)`,
  I3′); it satisfies `PickSpec`. So does the code's `(lo, slot)` key, which orders by `lo` first.
-/

namespace CchRecompute

/-! ## Exact: `ℕ ∪ {+∞}` -/

inductive NInf where
  | fin (n : Nat)
  | inf
deriving DecidableEq

def NInf.add : NInf → NInf → NInf
  | .fin a, .fin b => .fin (a + b)
  | _, _ => .inf

def NInf.lt : NInf → NInf → Bool
  | .fin a, .fin b => decide (a < b)
  | .fin _, .inf => true
  | .inf, _ => false

def NInf.ops : Ops NInf := ⟨NInf.add, NInf.lt, fun a b => decide (a = b), .inf⟩

theorem NInf.agree : Agree NInf.ops Eq := agree_eq _ (fun a b h => by simpa [NInf.ops] using h)

theorem NInf.top : TopLaws NInf.ops :=
  ⟨fun a => by cases a <;> rfl, fun a => by cases a <;> rfl, fun a => by cases a <;> rfl⟩

/-! ## Signed zeros -/

/-- `zero neg` is `±0.0`, `pos n` is the positive value `n + 1`, `inf` is `+∞`. -/
inductive SZ where
  | zero (neg : Bool)
  | pos (n : Nat)
  | inf
deriving DecidableEq

/-- IEEE `+` under round-to-nearest, on these values: only `-0 + -0` is `-0`. -/
def SZ.add : SZ → SZ → SZ
  | .inf, _ => .inf
  | _, .inf => .inf
  | .zero s, .zero t => .zero (s && t)
  | .zero _, .pos n => .pos n
  | .pos n, .zero _ => .pos n
  | .pos a, .pos b => .pos (a + b + 1)

/-- IEEE `<`: the sign of a zero does not count. -/
def SZ.lt : SZ → SZ → Bool
  | .zero _, .zero _ => false
  | .zero _, _ => true
  | .pos _, .zero _ => false
  | .pos a, .pos b => decide (a < b)
  | .pos _, .inf => true
  | .inf, _ => false

/-- IEEE `==`: `+0.0 == -0.0`. -/
def SZ.beq : SZ → SZ → Bool
  | .zero _, .zero _ => true
  | .pos a, .pos b => decide (a = b)
  | .inf, .inf => true
  | _, _ => false

def SZ.ops : Ops SZ := ⟨SZ.add, SZ.lt, SZ.beq, .inf⟩

/-- **P-f64, on the model**: `==` is an equivalence that `+` and `<` respect. -/
theorem SZ.agree : Agree SZ.ops (fun a b => SZ.beq a b = true) where
  refl a := by cases a <;> simp [SZ.beq]
  symm {a b} h := by cases a <;> cases b <;> simp_all [SZ.beq] <;> omega
  trans {a b c} h1 h2 := by cases a <;> cases b <;> cases c <;> simp_all [SZ.beq]
  beq_sound h := h
  add_congr {a a' b b'} h1 h2 := by
    cases a <;> cases a' <;> cases b <;> cases b' <;> simp_all [SZ.beq, SZ.add, SZ.ops]
  lt_congr {a a' b b'} h1 h2 := by
    cases a <;> cases a' <;> cases b <;> cases b' <;> simp_all [SZ.beq, SZ.lt, SZ.ops]

theorem SZ.top : TopLaws SZ.ops :=
  ⟨fun a => by cases a <;> rfl, fun a => by cases a <;> rfl, fun a => by cases a <;> rfl⟩

/-- … but `==` is not sound for equality: the premise of every `_exact` theorem fails. -/
theorem SZ.beq_not_sound : SZ.ops.beq (.zero false) (.zero true) = true ∧ SZ.zero false ≠ SZ.zero true := by
  decide

/-! ## The heap's pop -/

/-- Least `(lo, hi)`, first on ties. -/
def pickMin : List (Nat × Nat) → Option (Nat × Nat)
  | [] => none
  | p :: t =>
    match pickMin t with
    | none => some p
    | some q => if p.1 < q.1 ∨ (p.1 = q.1 ∧ p.2 ≤ q.2) then some p else some q

theorem pickMin_cons (p : Nat × Nat) (t : List (Nat × Nat)) :
    pickMin (p :: t) = some p ∨ (∃ q, pickMin t = some q ∧ pickMin (p :: t) = some q ∧
      ¬(p.1 < q.1 ∨ (p.1 = q.1 ∧ p.2 ≤ q.2))) := by
  cases hq : pickMin t with
  | none => left; simp [pickMin, hq]
  | some q =>
    by_cases hc : p.1 < q.1 ∨ (p.1 = q.1 ∧ p.2 ≤ q.2)
    · left; simp only [pickMin, hq, hc, ↓reduceIte]
    · right; exact ⟨q, rfl, by simp only [pickMin, hq, hc, ↓reduceIte], hc⟩

theorem pickMin_cons_isSome (p : Nat × Nat) (t : List (Nat × Nat)) : pickMin (p :: t) ≠ none := by
  rcases pickMin_cons p t with h | ⟨q, _, h, _⟩ <;> simp [h]

theorem pickMin_spec : PickSpec pickMin := by
  refine ⟨?_, ?_, ?_⟩
  · intro h hn
    cases h with
    | nil => rfl
    | cons p t => exact absurd hn (pickMin_cons_isSome p t)
  · intro h
    induction h with
    | nil => intro a ha; simp [pickMin] at ha
    | cons p t ih =>
      intro a ha
      rcases pickMin_cons p t with h | ⟨q, hq, h, _⟩
      · rw [h] at ha; cases ha; exact List.mem_cons_self
      · rw [h] at ha; cases ha; exact List.mem_cons_of_mem _ (ih _ hq)
  · intro h
    induction h with
    | nil => intro a ha; simp [pickMin] at ha
    | cons p t ih =>
      intro a ha b hb
      rcases pickMin_cons p t with h | ⟨q, hq, h, hc⟩
      · rw [h] at ha; cases ha
        rcases List.mem_cons.mp hb with rfl | hb
        · exact Nat.le_refl _
        · cases hq : pickMin t with
          | none =>
            cases t with
            | nil => simp at hb
            | cons p' t' => exact absurd hq (pickMin_cons_isSome p' t')
          | some q =>
            have := ih q hq b hb
            have hc : p.1 < q.1 ∨ (p.1 = q.1 ∧ p.2 ≤ q.2) := by
              apply Classical.byContradiction
              intro hc
              have hp : pickMin (p :: t) = some q := by simp only [pickMin, hq, hc, ↓reduceIte]
              rw [h] at hp; cases hp
              exact hc (Or.inr ⟨rfl, Nat.le_refl _⟩)
            omega
      · rw [h] at ha; cases ha
        rcases List.mem_cons.mp hb with rfl | hb
        · omega
        · exact ih _ hq b hb

end CchRecompute
