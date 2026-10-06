import FalkorIdSpace.Record
/-! `verify`: `checked` plus the hole test, which is exactly set equality of
`taken ∩ [entry, ∞)` with `[entry, entry + created)`; and `open_batch`. -/
namespace IdSpaceModel

/-- The interval `[a, b)` as an `IdSet`. -/
def ival (a b : Nat) : IdSet := fun i => decide (a ≤ i) && decide (i < b)

/-- The hole test on a set `T` that lies at or above `eb` (the pre-#2846
`verify`, whose `created` was exactly this set). -/
def holeOn (N eb : Nat) (T : IdSet) : Option Nat :=
  match sMin N T, sMax N T with
  | some lo, some hi =>
    if lo != eb || sLen N T - 1 != hi - eb then some hi else none
  | _, _ => none

theorem mem_max_ge {N : Nat} {s : IdSet} {lo hi : Nat} (hl : sMin N s = some lo)
    (hh : sMax N s = some hi) : lo ≤ hi := by
  obtain ⟨hlN, hsl, _⟩ := sMin_some.mp hl
  obtain ⟨_, _, hmx⟩ := sMax_some.mp hh
  apply Nat.not_lt.mp; intro h; have := hmx lo h hlN; rw [hsl] at this; cases this

/-- The engine's hole test (`select` + `max` over the untrimmed `taken`) is the
hole test on `taken ∩ [entry, ∞)`. -/
theorem holeCheck_eq (N : Nat) (sp : IdSpace) (hb : sp.eb ≤ N) :
    sp.holeCheck N = holeOn N sp.eb (sFrom sp.taken sp.eb) := by
  unfold IdSpace.holeCheck holeOn
  dsimp only
  rw [select_above N sp.taken hb, above_sFrom N sp.taken hb]
  cases hm : sMin N (sFrom sp.taken sp.eb) with
  | none => rfl
  | some lo => rw [sMax_sFrom N sp.taken hb hm]; rfl

/-- The hole check passes iff `T` is exactly `[eb, eb + |T|)` (for `T ⊆ [eb, N)`). -/
theorem noHole_iff (N eb : Nat) (T : IdSet) (hi : ∀ i, i < N → T i = true → eb ≤ i)
    (hle : eb + sLen N T ≤ N) :
    holeOn N eb T = none ↔ (∀ i, i < N → (T i = true ↔ eb ≤ i ∧ i < eb + sLen N T)) := by
  unfold holeOn
  cases hmn : sMin N T with
  | none =>
    have hz := sMin_none.mp hmn
    have hL : sLen N T = 0 := cnt_eq_zero.mpr hz
    simp only [hL, Nat.add_zero, true_iff]
    intro i hi; rw [hz i hi]; simp
  | some lo =>
    cases hmx : sMax N T with
    | none => rw [← sMin_none_iff_sMax_none] at hmx; rw [hmx] at hmn; cases hmn
    | some hi' =>
      simp only
      obtain ⟨hlN, hsl, hlmin⟩ := sMin_some.mp hmn
      obtain ⟨hhN, hsh, hhmax⟩ := sMax_some.mp hmx
      have hlh := mem_max_ge hmn hmx
      have hLpos : sLen N T ≠ 0 := cnt_pos.mpr ⟨lo, hlN, hsl⟩
      have heb : eb ≤ lo := hi lo hlN hsl
      constructor
      · intro h
        split at h; · cases h
        rename_i hc
        simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hc
        obtain ⟨hlo, hlen⟩ := hc
        subst hlo
        have hsub : ∀ i, i < N → T i = true → ival lo (hi' + 1) i = true := by
          intro i hiN hci
          have := hi i hiN hci
          have : i ≤ hi' := by
            apply Nat.not_lt.mp; intro hlt; have := hhmax i hlt hiN; rw [hci] at this; cases this
          simp [ival]; omega
        have hcnt : cnt N (ival lo (hi' + 1)) = hi' + 1 - lo :=
          cnt_interval lo (hi' + 1) N (by omega)
        have hsup := cnt_sub_eq N hsub (by unfold sLen at hlen hLpos; rw [hcnt]; omega)
        refine fun i hiN => ⟨fun hci => ?_, fun hr => ?_⟩
        · have := hsub i hiN hci; simp [ival] at this; unfold sLen at *; omega
        · exact hsup i hiN (by simp [ival]; unfold sLen at *; omega)
      · intro hiv
        have hlo : lo = eb := by
          have := (hiv lo hlN).mp hsl
          apply Nat.le_antisymm _ heb
          apply Nat.not_lt.mp; intro hlt
          have hm := (hiv eb (by omega)).mpr ⟨Nat.le_refl _, by omega⟩
          have := hlmin eb hlt; rw [hm] at this; cases this
        have hhi : hi' = eb + sLen N T - 1 := by
          have := (hiv hi' hhN).mp hsh
          apply Nat.le_antisymm (by omega)
          apply Nat.not_lt.mp; intro hlt
          have hm := (hiv (eb + sLen N T - 1) (by omega)).mpr ⟨by omega, by omega⟩
          have := hhmax _ hlt (by omega); rw [hm] at this; cases this
        subst hlo
        simp; omega

/-- **`verify` accepts exactly a consistent, dense batch**: the count half of
the invariant holds (`checked`), and the ids taken at or above the entry
boundary are precisely `[entry, entry + created)`. -/
theorem verify_ok_iff (N : Nat) (sp : IdSpace) :
    sp.verify N = .ok () ↔
      (sp.checked N = .ok () ∧
        ∀ i, i < N → ((sp.taken i = true ∧ sp.eb ≤ i) ↔
          sp.eb ≤ i ∧ i < sp.eb + above N sp.taken sp.eb)) := by
  unfold IdSpace.verify
  cases hc : sp.checked N with
  | error e => simp
  | ok u =>
    cases u
    have ⟨hfit, _⟩ := (checked_ok_iff N sp).mp hc
    have hb : sp.eb ≤ N := by omega
    have hT : ∀ i, i < N → sFrom sp.taken sp.eb i = true → sp.eb ≤ i := by
      intro i _ h; simp [sFrom] at h; exact h.2
    have hab := above_sFrom N sp.taken hb
    have key := noHole_iff N sp.eb (sFrom sp.taken sp.eb) hT (by rw [← hab]; omega)
    rw [holeCheck_eq N sp hb]
    simp only [true_and]
    rw [hab]
    cases hh : holeOn N sp.eb (sFrom sp.taken sp.eb) with
    | some h =>
      simp only [reduceCtorEq, false_iff]
      intro hall; have := key.mpr (fun i hi => by rw [← hall i hi]; simp [sFrom]); rw [hh] at this
      cases this
    | none =>
      simp only [true_iff]
      intro i hi; rw [← key.mp hh i hi]; simp [sFrom]

/-- A refusal from the hole test names the highest id taken. -/
theorem verify_hole (N : Nat) (sp : IdSpace) (h : Nat) (hc : sp.checked N = .ok ())
    (hh : sp.holeCheck N = some h) :
    sp.verify N = .error (.hole sp.eb h (above N sp.taken sp.eb)) := by
  simp [IdSpace.verify, hc, hh]

/-- `verify`'s arithmetic never wraps: `highest - entry` is only evaluated once
`lowest == entry`, and `lowest ≤ highest`; `created - 1` once `created ≥ 1`. -/
theorem verify_no_underflow (N : Nat) (sp : IdSpace) (lo hi : Nat) (hb : sp.eb ≤ N)
    (hl : sSelect N sp.taken (sLen N sp.taken - above N sp.taken sp.eb) = some lo)
    (hh : sMax N sp.taken = some hi) (he : lo = sp.eb) :
    sp.eb ≤ hi ∧ 1 ≤ above N sp.taken sp.eb := by
  rw [select_above N sp.taken hb] at hl
  rw [sMax_sFrom N sp.taken hb hl] at hh
  obtain ⟨hlN, hsl, _⟩ := sMin_some.mp hl
  rw [above_sFrom N sp.taken hb]
  exact ⟨he ▸ mem_max_ge hl hh, Nat.pos_of_ne_zero (cnt_pos.mpr ⟨lo, hlN, hsl⟩)⟩

/-! ### `open_batch` -/

/-- `open_batch` refuses exactly when `checked` does (and then changes
nothing); otherwise it re-anchors at `bound` with an empty ledger. -/
theorem openBatch_spec (N : Nat) (sp : IdSpace) :
    sp.openBatch N = (match sp.checked N with
      | .error e => .error e
      | .ok () => .ok { sp with eb := sp.bound N, taken := sEmpty }) := rfl

theorem openBatch_ok (N : Nat) (sp : IdSpace) (h : sp.checked N = .ok ()) :
    sp.openBatch N = .ok { sp with eb := sp.bound N, taken := sEmpty } := by
  simp [IdSpace.openBatch, h]

/-- An opened batch is the same space as `new_version` would fork. -/
theorem openBatch_eq_newVersion (N : Nat) (sp : IdSpace) (h : sp.checked N = .ok ()) :
    sp.openBatch N = .ok (sp.newVersion N) := by
  rw [openBatch_ok N sp h]; rfl

end IdSpaceModel
