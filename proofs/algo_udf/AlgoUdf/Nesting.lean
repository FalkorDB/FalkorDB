/-! # Depth-capped recursion with an explicit stack (`graph/src/udf/type_convert.rs`, #3336)

Both UDF conversions (`value_to_js_at` :110 / `js_to_value_at` :262) have the same shape: one
conversion level runs `step (depth >= MAX_UDF_VALUE_NESTING) (convert children at depth + 1) v`.
`Step` abstracts that shape; `Step.run cap f d v` runs it at depth `d` with `f` stack frames left
(`none` = the stack ran out, i.e. the abort #3336 fixes). `cap = some c` is the code after #3336,
`cap = none` the unbounded code before it (no depth check).

`R α := Option (Except String α)`: `none` is a stack overflow (it aborts the process at once, so it
short-circuits like an error but is not one).

Generic results (instantiated for both directions in `MarshalCap.lean`):
* `run_frames`: with the cap, `c + 1` frames always suffice — for *any* input, cyclic ones included.
* `run_eq_uncapped`: on an input whose nesting fits the cap (`within`), the capped and the old code
  return the same thing for every stack size.
* `run_past_cap`: an input that does not fit the cap is an error (never an overflow).
* `run_mono`: once a run finishes, more stack does not change its result.
-/
namespace AlgoUdf.Marshal

abbrev R (α : Type) := Option (Except String α)

@[inline] def R.ok {α} (a : α) : R α := some (.ok a)
@[inline] def R.err {α} (e : String) : R α := some (.error e)
def R.bind {α β} (x : R α) (k : α → R β) : R β :=
  match x with
  | none => none
  | some (.error e) => some (.error e)
  | some (.ok a) => k a

@[simp] theorem R.bind_none {α β} (k : α → R β) : R.bind none k = none := rfl
@[simp] theorem R.bind_err {α β} (e : String) (k : α → R β) : R.bind (some (.error e)) k = some (.error e) := rfl
@[simp] theorem R.bind_ok {α β} (a : α) (k : α → R β) : R.bind (some (.ok a)) k = k a := rfl

/-- The result of a run that finished (a stack overflow aborts; it never yields a value). -/
def Rget {α} : R α → Except String α
  | some r => r
  | none => .error "stack overflow"

/-- `MAX_UDF_VALUE_NESTING` (`type_convert.rs:67`). -/
def MAX : Nat := 1024
/-- `udf_nesting_error()` (`type_convert.rs:87-89`). -/
def nestErr : String := "UDF value nesting exceeds the maximum depth of 1024"

/-- `depth >= MAX_UDF_VALUE_NESTING`; `none` = no check (before #3336). -/
def chk : Option Nat → Nat → Bool
  | some c, d => decide (c ≤ d)
  | none, _ => false

/-- One conversion level and the facts every level satisfies. `needs v`: the level checks the
depth (list/array/map/object/path); `kids v`: the children it converts at `depth + 1`. -/
structure Step (α β : Type) where
  step : Bool → (α → R β) → α → R β
  needs : α → Bool
  kids : α → List α
  leaf_indep : ∀ b b' conv conv' v, needs v = false → step b conv v = step b' conv' v
  leaf_some : ∀ b conv v, needs v = false → step b conv v ≠ none
  chk_err : ∀ conv v, needs v = true → step true conv v = R.err nestErr
  ext : ∀ b conv conv' v, (∀ x ∈ kids v, conv x = conv' x) → step b conv v = step b conv' v
  total : ∀ b conv v, (∀ x, conv x ≠ none) → step b conv v ≠ none
  kid_err : ∀ conv v x e, (∀ y, conv y ≠ none) → x ∈ kids v → conv x = R.err e →
    ∃ e', step false conv v = R.err e'
  mono : ∀ b conv conv' v, step b conv v ≠ none → (∀ x, conv x ≠ none → conv' x = conv x) →
    step b conv' v = step b conv v

namespace Step
variable {α β : Type} (S : Step α β)

/-- The recursion: depth `d`, `f` frames of stack left. -/
def run (cap : Option Nat) : Nat → Nat → α → R β
  | 0, _, _ => none
  | f + 1, d, v => S.step (chk cap d) (run cap f (d + 1)) v

/-- `v` nests at most `r` levels below the current one: a depth-checked value at budget 0 is too deep. -/
def within : Nat → α → Bool
  | 0, v => !S.needs v
  | r + 1, v => (S.kids v).all (within r)

theorem within_succ (r : Nat) (v : α) :
    S.within (r + 1) v = true ↔ ∀ x ∈ S.kids v, S.within r x = true := by
  simp [within, List.all_eq_true]

/-- **Frames**: with the cap `c`, a run started at depth `d ≤ c` needs at most `c + 1 - d` frames. -/
theorem run_frames (c : Nat) : ∀ (f d : Nat) (v : α), c + 1 ≤ f + d → d ≤ c → S.run (some c) f d v ≠ none
  | 0, d, _, h1, h2 => by omega
  | f + 1, d, v, h1, _ => by
    simp only [run]
    by_cases hd : c ≤ d
    · have hb : chk (some c) d = true := by simp [chk, hd]
      rw [hb]
      cases hn : S.needs v
      · exact S.leaf_some _ _ _ hn
      · rw [S.chk_err _ _ hn]; exact fun h => nomatch h
    · exact S.total _ _ _ (fun x => run_frames c f (d + 1) x (by omega) (by omega))

/-- **The cap changes nothing on an input that fits it** (any stack size). -/
theorem run_eq_uncapped (c : Nat) : ∀ (f d : Nat) (v : α), S.within (c - d) v = true →
    S.run none f d v = S.run (some c) f d v
  | 0, _, _, _ => rfl
  | f + 1, d, v, hw => by
    simp only [run]
    by_cases hd : c ≤ d
    · have h0 : c - d = 0 := by omega
      rw [h0] at hw
      simp only [within, Bool.not_eq_true'] at hw
      exact S.leaf_indep _ _ _ _ _ hw
    · have hb : chk (some c) d = false := by simp [chk, hd]
      rw [hb, show chk none d = false from rfl]
      have hs : c - d = (c - (d + 1)) + 1 := by omega
      rw [hs, S.within_succ] at hw
      exact S.ext _ _ _ _ (fun x hx => run_eq_uncapped c f (d + 1) x (hw x hx))

/-- **Past the cap is an error**, given the stack the cap guarantees. -/
theorem run_past_cap (c : Nat) : ∀ (f d : Nat) (v : α), d ≤ c → c + 1 ≤ f + d →
    S.within (c - d) v = false → ∃ e, S.run (some c) f d v = R.err e
  | 0, d, _, h1, h2, _ => by omega
  | f + 1, d, v, h1, h2, hw => by
    simp only [run]
    by_cases hd : c ≤ d
    · have h0 : c - d = 0 := by omega
      rw [h0] at hw
      simp only [within, Bool.not_eq_false'] at hw
      have hb : chk (some c) d = true := by simp [chk, hd]
      rw [hb, S.chk_err _ _ hw]
      exact ⟨_, rfl⟩
    · have hb : chk (some c) d = false := by simp [chk, hd]
      rw [hb]
      have hs : c - d = (c - (d + 1)) + 1 := by omega
      rw [hs] at hw
      have : ¬ ∀ x ∈ S.kids v, S.within (c - (d + 1)) x = true := fun h => by
        rw [← S.within_succ] at h; rw [h] at hw; cases hw
      have ⟨x, hx, hxw⟩ : ∃ x ∈ S.kids v, S.within (c - (d + 1)) x = false := by
        apply Classical.byContradiction; intro hn; apply this; intro x hx
        cases h : S.within (c - (d + 1)) x
        · exact absurd ⟨x, hx, h⟩ hn
        · rfl
      obtain ⟨e, he⟩ := run_past_cap c f (d + 1) x (by omega) (by omega) hxw
      exact S.kid_err _ _ x e (fun y => S.run_frames c f (d + 1) y (by omega) (by omega)) hx he

/-- **More stack does not change a finished run.** -/
theorem run_mono (cap : Option Nat) : ∀ (f d : Nat) (v : α), S.run cap f d v ≠ none →
    S.run cap (f + 1) d v = S.run cap f d v
  | 0, _, _, h => absurd rfl h
  | f + 1, d, v, h => by
    simp only [run] at h ⊢
    exact S.mono _ _ _ _ h (fun x hx => run_mono cap f (d + 1) x hx)

theorem run_mono_add (cap : Option Nat) (f d : Nat) (v : α) (h : S.run cap f d v ≠ none) :
    ∀ k, S.run cap (f + k) d v = S.run cap f d v
  | 0 => rfl
  | k + 1 => by
    have ih := run_mono_add cap f d v h k
    rw [← Nat.add_assoc, S.run_mono cap (f + k) d v (by rw [ih]; exact h)]
    exact ih

end Step
end AlgoUdf.Marshal

namespace AlgoUdf.Marshal
/-- `udf_nesting_error()` is `format!("... {MAX_UDF_VALUE_NESTING}")`. -/
theorem nestErr_eq : nestErr = "UDF value nesting exceeds the maximum depth of " ++ toString MAX := by decide
end AlgoUdf.Marshal
