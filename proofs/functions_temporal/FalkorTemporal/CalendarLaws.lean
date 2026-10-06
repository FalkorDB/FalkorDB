import FalkorTemporal.EraAll
/-!
# Calendar laws: `civil_from_days` and `days_from_civil` are mutually inverse

The two headline theorems (`dfc_civil`, `civil_dfc`) are about the *unbounded*
Int functions; `civilFromDays_trunc` says where the Rust `y as i32` cast
(`value.rs:800`) starts to matter.
-/
namespace FalkorTemporal

theorem era_fwd (doe : Int) (h0 : 0 ≤ doe) (h1 : doe < 146097) :
    0 ≤ doyOf doe ∧ doyOf doe ≤ 365 ∧ 0 ≤ mpOf doe ∧ mpOf doe ≤ 11 ∧
    1 ≤ dOf doe ∧ dOf doe ≤ dimMarch (yoeOf doe) (mpOf doe) := by
  have h := fwdOk_all doe.toNat (by omega)
  have e : ((doe.toNat : Nat) : Int) = doe := Int.toNat_of_nonneg h0
  simp only [fwdOk, e, Bool.and_eq_true, decide_eq_true_eq] at h
  omega

theorem era_bwd (yoe mp d : Int) (hy0 : 0 ≤ yoe) (hy1 : yoe ≤ 399) (hm0 : 0 ≤ mp)
    (hm1 : mp ≤ 11) (hd0 : 1 ≤ d) (hd1 : d ≤ dimMarch yoe mp) :
    yoeOf (dfcEra yoe mp d) = yoe ∧ mpOf (dfcEra yoe mp d) = mp ∧ dOf (dfcEra yoe mp d) = d := by
  have hdim : dimMarch yoe mp ≤ 31 := by
    unfold dimMarch; split <;> (try split) <;> omega
  have h := bwdOk_all (yoe.toNat * 372 + mp.toNat * 31 + (d - 1).toNat) (by omega)
  have gy : gridYoe (yoe.toNat * 372 + mp.toNat * 31 + (d - 1).toNat) = yoe := by
    unfold gridYoe; omega
  have gm : gridMp (yoe.toNat * 372 + mp.toNat * 31 + (d - 1).toNat) = mp := by
    unfold gridMp; omega
  have gd : gridD (yoe.toNat * 372 + mp.toNat * 31 + (d - 1).toNat) = d := by
    unfold gridD; omega
  simp only [bwdOk, gy, gm, gd] at h
  simp only [hd1, ite_true, Bool.and_eq_true, decide_eq_true_eq] at h
  exact ⟨h.1.1, h.1.2, h.2⟩

/-- `dfcEra` never leaves the era for a real date. -/
theorem dfcEra_range (yoe mp d : Int) (hy0 : 0 ≤ yoe) (hy1 : yoe ≤ 399) (hm0 : 0 ≤ mp)
    (hm1 : mp ≤ 11) (hd0 : 1 ≤ d) (hd1 : d ≤ dimMarch yoe mp) :
    0 ≤ dfcEra yoe mp d ∧ dfcEra yoe mp d < 146097 := by
  have hdim : dimMarch yoe mp ≤ (if mp = 11 then 29 else 31) := by
    unfold dimMarch; split <;> (try split) <;> simp_all <;> omega
  unfold dfcEra
  split at hdim <;> omega

/-- The month-length tables agree: the March-based length of `mp` is the civil
`days_in_month` of `m`. -/
theorem dim_link (y m : Int) (hm1 : 1 ≤ m) (hm2 : m ≤ 12) :
    dimMarch ((if m ≤ 2 then y - 1 else y) % 400) (if m > 2 then m - 3 else m + 9) =
    daysInMonth y m := by
  have hl : isLeap ((y - 1) % 400 + 1) = isLeap y := by
    have : (y - 1) % 400 + 1 = y + 400 * (-((y - 1) / 400)) := by omega
    rw [this, isLeap_add400]
  have : m = 1 ∨ m = 2 ∨ m = 3 ∨ m = 4 ∨ m = 5 ∨ m = 6 ∨ m = 7 ∨ m = 8 ∨ m = 9 ∨
      m = 10 ∨ m = 11 ∨ m = 12 := by omega
  rcases this with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp [dimMarch, daysInMonth, hl]

theorem dfc_civil_core (era doe yoe mp d : Int)
    (hm0 : 0 ≤ mp) (hm1 : mp ≤ 11) (hy0 : 0 ≤ yoe) (hy1 : yoe ≤ 399)
    (hrt : yoe * 365 + yoe / 4 - yoe / 100 + ((153 * mp + 2) / 5 + d - 1) = doe) :
    daysFromCivil
      (if (if mp < 10 then mp + 3 else mp - 9) ≤ 2 then yoe + era * 400 + 1 else yoe + era * 400)
      (if mp < 10 then mp + 3 else mp - 9) d = era * 146097 + doe - 719468 := by
  unfold daysFromCivil
  dsimp only
  by_cases hmp : mp < 10
  · have c1 : ¬ (mp + 3 ≤ 2) := by omega
    have c2 : mp + 3 > 2 := by omega
    simp only [hmp, c1, c2, ite_true, ite_false]
    have e1 : (yoe + era * 400) / 400 = era := by omega
    have e2 : (yoe + era * 400) % 400 = yoe := by omega
    have e3 : mp + 3 - 3 = mp := by omega
    rw [e1, e2, e3]
    omega
  · have c1 : mp - 9 ≤ 2 := by omega
    have c2 : ¬ (mp - 9 > 2) := by omega
    simp only [hmp, c1, c2, ite_true, ite_false]
    have e1 : (yoe + era * 400 + 1 - 1) / 400 = era := by omega
    have e2 : (yoe + era * 400 + 1 - 1) % 400 = yoe := by omega
    have e3 : mp - 9 + 9 = mp := by omega
    rw [e1, e2, e3]
    omega

/-- **`days_from_civil ∘ civil_from_days = id`** on every `Int` day number. -/
theorem dfc_civil (z : Int) :
    daysFromCivil (civilFromDaysI z).1 (civilFromDaysI z).2.1 (civilFromDaysI z).2.2 = z := by
  have h0 := Int.emod_nonneg (z + 719468) (by decide : (146097:Int) ≠ 0)
  have h1 := Int.emod_lt_of_pos (z + 719468) (by decide : (0:Int) < 146097)
  have hdiv : 146097 * ((z + 719468) / 146097) + (z + 719468) % 146097 = z + 719468 :=
    Int.mul_ediv_add_emod _ _
  have hf := era_fwd _ h0 h1
  have hy := yoe_bounds _ h0 h1
  have hrt := dfcEra_of ((z + 719468) % 146097)
  unfold dfcEra at hrt
  have := dfc_civil_core ((z + 719468) / 146097) ((z + 719468) % 146097)
    (yoeOf ((z + 719468) % 146097)) (mpOf ((z + 719468) % 146097)) (dOf ((z + 719468) % 146097))
    hf.2.2.1 hf.2.2.2.1 hy.1 hy.2 hrt
  simp only [civilFromDaysI]
  rw [this]
  omega

/-- `civil_from_days` only produces real dates (month 1..12, day within the month,
leap years included). -/
theorem civil_valid (z : Int) :
    ValidDate (civilFromDaysI z).1 (civilFromDaysI z).2.1 (civilFromDaysI z).2.2 := by
  have h0 := Int.emod_nonneg (z + 719468) (by decide : (146097:Int) ≠ 0)
  have h1 := Int.emod_lt_of_pos (z + 719468) (by decide : (0:Int) < 146097)
  have hf := era_fwd _ h0 h1
  have hy := yoe_bounds _ h0 h1
  -- the civil month/year recomputed by `dim_link`
  have key : ∀ (yoe era mp : Int), 0 ≤ yoe → yoe ≤ 399 → 0 ≤ mp → mp ≤ 11 →
      daysInMonth (if (if mp < 10 then mp + 3 else mp - 9) ≤ 2 then yoe + era * 400 + 1
          else yoe + era * 400) (if mp < 10 then mp + 3 else mp - 9) = dimMarch yoe mp := by
    intro yoe era mp a b c d
    have := dim_link (if (if mp < 10 then mp + 3 else mp - 9) ≤ 2 then yoe + era * 400 + 1
          else yoe + era * 400) (if mp < 10 then mp + 3 else mp - 9)
          (by split <;> omega) (by split <;> omega)
    rw [← this]
    by_cases hmp : mp < 10
    · have c1 : ¬ (mp + 3 ≤ 2) := by omega
      have c2 : mp + 3 > 2 := by omega
      simp only [hmp, c1, c2, ite_true, ite_false]
      have e2 : (yoe + era * 400) % 400 = yoe := by omega
      rw [e2]; congr 1; omega
    · have c1 : mp - 9 ≤ 2 := by omega
      have c2 : ¬ (mp - 9 > 2) := by omega
      simp only [hmp, c1, c2, ite_true, ite_false]
      have e2 : (yoe + era * 400 + 1 - 1) % 400 = yoe := by omega
      rw [e2]; congr 1; omega
  have k := key (yoeOf ((z + 719468) % 146097)) ((z + 719468) / 146097)
    (mpOf ((z + 719468) % 146097)) hy.1 hy.2 hf.2.2.1 hf.2.2.2.1
  simp only [civilFromDaysI, ValidDate]
  rw [k]
  refine ⟨by split <;> omega, by split <;> omega, hf.2.2.2.2.1, hf.2.2.2.2.2⟩

/-- **`civil_from_days ∘ days_from_civil = id`** on every real date. -/
theorem civil_dfc (y m d : Int) (h : ValidDate y m d) :
    civilFromDaysI (daysFromCivil y m d) = (y, m, d) := by
  obtain ⟨hm1, hm2, hd1, hd2⟩ := h
  have hl := dim_link y m hm1 hm2
  generalize hy' : (if m ≤ 2 then y - 1 else y) = y' at hl
  generalize hmp : (if m > 2 then m - 3 else m + 9) = mp at hl
  have hmp0 : 0 ≤ mp ∧ mp ≤ 11 := by subst hmp; split <;> omega
  have hyoe : 0 ≤ y' % 400 ∧ y' % 400 ≤ 399 := by omega
  have hd2' : d ≤ dimMarch (y' % 400) mp := by rw [hl]; exact hd2
  have hb := era_bwd (y' % 400) mp d hyoe.1 hyoe.2 hmp0.1 hmp0.2 hd1 hd2'
  have hr := dfcEra_range (y' % 400) mp d hyoe.1 hyoe.2 hmp0.1 hmp0.2 hd1 hd2'
  have hz : daysFromCivil y m d + 719468 = (y' / 400) * 146097 + dfcEra (y' % 400) mp d := by
    unfold daysFromCivil dfcEra; simp only [hy', hmp]; omega
  have hmod : (daysFromCivil y m d + 719468) % 146097 = dfcEra (y' % 400) mp d := by
    rw [hz]; omega
  have hdv : (daysFromCivil y m d + 719468) / 146097 = y' / 400 := by
    rw [hz]; omega
  simp only [civilFromDaysI]
  rw [hmod, hdv, hb.1, hb.2.1, hb.2.2]
  by_cases hm : m ≤ 2
  · have e1 : mp = m + 9 := by subst hmp; simp [show ¬ m > 2 by omega]
    have e2 : y' = y - 1 := by subst hy'; simp [hm]
    have c : ¬ (mp < 10) := by omega
    simp only [c, ite_false]
    have c2 : mp - 9 ≤ 2 := by omega
    simp only [c2, ite_true]
    ext <;> simp <;> omega
  · have e1 : mp = m - 3 := by subst hmp; simp [show m > 2 by omega]
    have e2 : y' = y := by subst hy'; simp [hm]
    have c : mp < 10 := by omega
    simp only [c, ite_true]
    have c2 : ¬ (mp + 3 ≤ 2) := by omega
    simp only [c2, ite_false]
    ext <;> simp <;> omega

/-- The Rust result equals the Int result whenever the year fits an `i32`. -/
theorem civilFromDays_exact (z : Int) (h : I32 (civilFromDaysI z).1) :
    civilFromDays z = civilFromDaysI z := by
  simp only [civilFromDays, toI32_id _ h]

/-- …and it silently wraps when it does not: day `784 353 015 833` is in year
2 147 487 588 (> i32::MAX), which `y as i32` turns into year −2 147 479 708. Reached by
`decompose_duration` on any duration beyond ~5.8 million years of seconds
(`duration({seconds: 9223372036854775807})`). -/
theorem civilFromDays_trunc :
    (civilFromDaysI 784353015833).1 = 2147487588 ∧
    (civilFromDays 784353015833).1 = -2147479708 := by decide

/-- Distinct real dates get distinct day numbers (injectivity, from `civil_dfc`). -/
theorem dfc_injective (y m d y' m' d' : Int) (h : ValidDate y m d) (h' : ValidDate y' m' d')
    (e : daysFromCivil y m d = daysFromCivil y' m' d') : (y, m, d) = (y', m', d') := by
  rw [← civil_dfc y m d h, ← civil_dfc y' m' d' h', e]

end FalkorTemporal
