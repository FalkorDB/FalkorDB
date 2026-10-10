import RedisLayer.QueryArgs

/-!
# `parse_query_flags` agrees with C's `_read_flags`

`rust_c_agree`: for every argument vector, under the configuration invariant that
`GRAPH.CONFIG SET` maintains (`CfgOk`), Rust's `parseQueryFlags` and C's `cReadFlags`
either both fail with corresponding errors (`QErr.toC`) or both succeed with
corresponding flags (`Rel`: same compact flag, same version, same timeout, where C's
`TIMEOUT 0` → default substitution is the one `compute_effective_timeout` performs
for Rust).
-/

namespace RedisLayer

def QErr.toC : QErr → CErr
  | .wrongArity => .wrongArity
  | .timeoutParse => .badTimeout
  | .timeoutMax => .exceedsMax
  | .versionParse => .badVersion

/-- What a Rust flag record means in C terms. -/
def Rel (c : TCfg) (r : QFlags) (g : CFlags) : Prop :=
  r.compact = g.compact ∧ g.timeoutRw = (cInit c).timeoutRw ∧
  g.hash = (match r.version with | none => -1 | some v => (v : Int)) ∧
  g.timeout = (match r.timeout with
    | none => (cInit c).timeout
    | some t => if t = 0 ∧ g.timeoutRw then dflt c else t)

/-- The `GRAPH.CONFIG` invariant (`Config.rustSet_inv`): non-negative values and
`TIMEOUT_DEFAULT ≤ TIMEOUT_MAX` when a max is set. -/
structure CfgOk (c : TCfg) : Prop where
  max_nn : 0 ≤ c.timeoutMax
  def_nn : 0 ≤ c.timeoutDefault
  inv : c.timeoutMax ≠ 0 → c.timeoutDefault ≤ c.timeoutMax

def Agree (c : TCfg) : Except QErr QFlags → Except CErr CFlags → Prop
  | .error e, .error e' => e.toC = e'
  | .ok r, .ok g => Rel c r g
  | _, _ => False

theorem eqIC_excl {a x y : List Nat} (h : eqIC a x = true) (hxy : eqIC x y = false) :
    eqIC a y = false := by
  simp only [eqIC, beq_iff_eq] at h hxy ⊢
  rw [h]; exact hxy

theorem k1 : eqIC (b "--track-memory") (b "--compact") = false := by decide
theorem k2 : eqIC (b "--track-memory") (b "timeout") = false := by decide
theorem k3 : eqIC (b "--track-memory") (b "version") = false := by decide
theorem k4 : eqIC (b "timeout") (b "--compact") = false := by decide
theorem k5 : eqIC (b "timeout") (b "--track-memory") = false := by decide
theorem k6 : eqIC (b "version") (b "--compact") = false := by decide
theorem k7 : eqIC (b "version") (b "--track-memory") = false := by decide
theorem k8 : eqIC (b "--compact") (b "timeout") = false := by decide
theorem k9 : eqIC (b "--compact") (b "version") = false := by decide
theorem k10 : eqIC (b "version") (b "timeout") = false := by decide

theorem dflt_le (c : TCfg) (hc : CfgOk c) (h : c.timeoutMax ≠ 0) : dflt c ≤ c.timeoutMax := by
  unfold dflt; split
  · omega
  · exact hc.inv h

theorem dflt_nn (c : TCfg) (hc : CfgOk c) : 0 ≤ dflt c := by
  unfold dflt; split
  · exact hc.max_nn
  · exact hc.def_nn

theorem scan_agree (c : TCfg) (hc : CfgOk c) : ∀ (l : List Bytes) (r : QFlags) (g : CFlags),
    Rel c r g → (c.timeoutMax ≠ 0 → g.timeout ≤ c.timeoutMax) →
    Agree c (scanFlags c.timeoutMax l r) (cScan c l g)
  | [], r, g, hr, _ => by unfold scanFlags cScan; simp only [Agree]; exact hr
  | a :: rest, r, g, hr, hi => by
    obtain ⟨hcmp, hrw, hh, ht⟩ := hr
    by_cases h1 : eqIC (upToNul a) (b "--compact") = true
    · unfold scanFlags cScan; simp only [h1, ↓reduceIte]
      exact scan_agree c hc rest _ _ ⟨by simp, hrw, hh, ht⟩ hi
    by_cases h2 : eqIC (upToNul a) (b "--track-memory") = true
    · have h3 := eqIC_excl h2 k2
      have h4 := eqIC_excl h2 k3
      unfold scanFlags cScan; simp only [h1, h2, h3, h4, ↓reduceIte, Bool.false_eq_true]
      exact scan_agree c hc rest _ _ ⟨hcmp, hrw, hh, ht⟩ hi
    by_cases h3 : eqIC (upToNul a) (b "timeout") = true
    · unfold scanFlags cScan; simp only [h1, h2, h3, ↓reduceIte, Bool.false_eq_true]
      cases rest with
      | nil => rfl
      | cons v rest' =>
        simp only
        cases hp : string2ll v with
        | none =>
          simp only [Option.getD_none, Option.isNone_none, true_or, ↓reduceIte]
          have : ¬ (c.timeoutMax ≠ 0 ∧ g.timeout > c.timeoutMax) := fun ⟨a1, a2⟩ => by
            have := hi a1; omega
          simp only [this, ↓reduceIte]; rfl
        | some t =>
          simp only [Option.getD_some, Option.isNone_some]
          have hm := hc.max_nn
          by_cases hx : c.timeoutMax ≠ 0 ∧ t > c.timeoutMax
          · have hx' : c.timeoutMax > 0 ∧ t > c.timeoutMax := ⟨by omega, hx.2⟩
            simp [hx.1, hx.2, show c.timeoutMax > 0 by omega, Agree, QErr.toC]
          · have hx' : ¬ (c.timeoutMax > 0 ∧ t > c.timeoutMax) := fun ⟨a1, a2⟩ => hx ⟨by omega, a2⟩
            simp only [hx, hx', ↓reduceIte]
            by_cases hn : t < 0
            · have : ¬ (t = 0 ∧ g.timeoutRw = true) := fun ⟨e, _⟩ => by omega
              simp only [hn, this, ↓reduceIte]; rfl
            · simp only [hn, ↓reduceIte]
              have hd := dflt_nn c hc
              have hnn : ¬ ((if t = 0 ∧ g.timeoutRw = true then dflt c else t) < 0) := by
                split <;> omega
              simp only [hnn]
              refine scan_agree c hc rest' _ _ ⟨hcmp, hrw, hh, ?_⟩ ?_
              · simp
              · intro hz
                dsimp only
                split
                · exact dflt_le c hc hz
                · omega
    by_cases h4 : eqIC (upToNul a) (b "version") = true
    · unfold scanFlags cScan; simp only [h1, h2, h3, h4, ↓reduceIte, Bool.false_eq_true]
      cases rest with
      | nil => rfl
      | cons v rest' =>
        simp only
        cases hp : string2ll v with
        | none => rfl
        | some n =>
          simp only
          by_cases hn : 0 ≤ n ∧ n ≤ (uintMax : Int)
          · have : ¬ (n < 0 ∨ n > (uintMax : Int)) := by omega
            simp only [hn, this, and_self, ↓reduceIte]
            refine scan_agree c hc rest' _ _ ⟨hcmp, hrw, ?_, ht⟩ hi
            simp; omega
          · have : (n < 0 ∨ n > (uintMax : Int)) := by omega
            simp only [hn, this, ↓reduceIte]; rfl
    · unfold scanFlags cScan; simp only [h1, h2, h3, h4, Bool.false_eq_true, ↓reduceIte]
      exact scan_agree c hc rest _ _ ⟨hcmp, hrw, hh, ht⟩ hi

/-- **Rust = C on every argument vector** (`557f18868`, #3010). -/
theorem rust_c_agree (c : TCfg) (hc : CfgOk c) (args : List Bytes) :
    Agree c (parseQueryFlags c.timeoutMax args) (cReadFlags c args) := by
  unfold parseQueryFlags cReadFlags maxArgs
  split
  · rfl
  · apply scan_agree c hc
    · refine ⟨?_, rfl, ?_, ?_⟩ <;> unfold cInit <;> split <;> rfl
    · intro hz
      unfold cInit
      rw [if_pos (Or.inl hz)]
      exact dflt_le c hc hz

/-- Corollary: the two engines accept exactly the same argument vectors. -/
theorem rust_c_same_accept (c : TCfg) (hc : CfgOk c) (args : List Bytes) :
    (∃ r, parseQueryFlags c.timeoutMax args = .ok r) ↔ (∃ g, cReadFlags c args = .ok g) := by
  have := rust_c_agree c hc args
  revert this
  cases parseQueryFlags c.timeoutMax args <;> cases cReadFlags c args <;> simp [Agree]

end RedisLayer
