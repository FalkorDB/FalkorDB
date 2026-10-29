import FalkorIdSpace.Inv
/-! `release` in a reachable state, rolling a batch, and the whole lifecycle. -/
namespace IdSpaceModel

/-- **`live -= freed.len()` never wraps.** In a reachable state, once
`release`'s refusals pass, every freed id is live, so there are at least as
many live ids as freed ones (the `debug_assert!(freed ⊆ requested)` is the
caller contract). -/
theorem release_no_underflow (N : Nat) (sp : IdSpace) (requested freed : IdSet)
    (h : Inv N sp) (hok : ReleaseOk N sp requested)
    (hsub : ∀ i, i < N → freed i = true → requested i = true) :
    sLen N freed ≤ sp.live := by
  rw [live_eq N sp h]
  apply cnt_mono
  intro i hi hf
  have hr := hsub i hi hf
  have h1 : sp.recycled i = false := by
    cases e : sp.recycled i
    · rfl
    · exact absurd ⟨hr, e⟩ (hok.1 i hi)
  simp only [liveSet, h1, Bool.not_false, Bool.true_and, Bool.or_eq_true, decide_eq_true_eq]
  by_cases hlt : i < sp.eb
  · exact .inl hlt
  · exact .inr (hok.2 i hi hr (by omega))

/-- **A release its refusals accept passes `checked` and leaves a reachable
state**: the boundary does not move (`live` down by `|freed|`, the bin up by as
many), and nothing taken changes. -/
theorem release_succeeds (N : Nat) (sp : IdSpace) (requested freed : IdSet)
    (h : Inv N sp) (hok : ReleaseOk N sp requested)
    (hsub : ∀ i, i < N → freed i = true → requested i = true) :
    sp.release N requested freed = (sp.afterRelease N freed, .ok ()) ∧
      Inv N (sp.afterRelease N freed) := by
  have hrel := (release_spec N sp requested freed).1 hok
  have hu := release_no_underflow N sp requested freed h hok hsub
  have ⟨hfit, hc⟩ := (checked_ok_iff N sp).mp h.checked
  have hchk : (sp.afterRelease N freed).checked N = .ok () := by
    apply (checked_ok_iff N _).mpr
    simp only [IdSpace.afterRelease, IdSpace.bound] at hfit hc ⊢
    have hd := cnt_or_disjoint N sp.recycled freed (fun i hi ⟨a, b⟩ =>
      hok.1 i hi ⟨hsub i hi b, a⟩)
    unfold sLen sUnion at *
    omega
  rw [hchk] at hrel
  refine ⟨hrel, hchk, fun i hi hr => ?_, h.noMax, fun i hi hr => ?_⟩
  rotate_left
  · simp only [IdSpace.afterRelease, sUnion, Bool.or_eq_true] at hr
    rcases hr with hr | hr
    · exact h.relOk i hi hr
    · by_cases hlt : i < sp.eb
      · exact .inl hlt
      · exact .inr (hok.2 i hi (hsub i hi hr) (by omega))
  simp only [IdSpace.afterRelease, sUnion, Bool.or_eq_true] at hr
  rcases hr with hr | hr
  · exact h.binOk i hi hr
  · by_cases hlt : i < sp.eb
    · exact .inl hlt
    · exact .inr (hok.2 i hi (hsub i hi hr) (by omega))

/-- A refused release changes nothing. -/
theorem release_refused (N : Nat) (sp : IdSpace) (requested freed : IdSet)
    (hok : ¬ ReleaseOk N sp requested) : (sp.release N requested freed).1 = sp :=
  ((release_spec N sp requested freed).2 hok).1

/-! ### Rolling a batch (`Graph::roll_id_batches` = `verify` then `open_batch`) -/

/-- **A batch that verifies rolls into a reachable one**: `open_batch` succeeds,
equals `new_version` (what the next MVCC version forks with), and satisfies the
invariant — every free id is now below the new boundary. -/
theorem roll_inv (N : Nat) (sp : IdSpace) (h : Inv N sp) (hv : sp.verify N = .ok ()) :
    sp.openBatch N = .ok (sp.newVersion N) ∧ Inv N (sp.newVersion N) := by
  have ⟨hc, hden⟩ := (verify_ok_iff N sp).mp hv
  have ⟨hfit, hcnt⟩ := (checked_ok_iff N sp).mp hc
  refine ⟨openBatch_eq_newVersion N sp hc, ?_⟩
  have hab : above N sEmpty (sp.bound N) = 0 := by
    rw [above_sFrom N _ (by omega)]; exact cnt_eq_zero.mpr (fun _ _ => rfl)
  refine ⟨(checked_ok_iff N _).mpr ?_, fun i hi hr => .inl ?_, rfl,
    fun _ _ hr => absurd hr (by simp [IdSpace.newVersion, IdSpace.restored, sEmpty])⟩
  · simp only [IdSpace.newVersion, IdSpace.restored, IdSpace.bound] at hab ⊢
    rw [hab]; unfold IdSpace.bound at hcnt; omega
  · simp only [IdSpace.newVersion, IdSpace.restored]
    have : i < sp.bound N := by
      rcases h.binOk i hi hr with a | a
      · omega
      · by_cases hlt : i < sp.eb
        · omega
        · have := ((hden i hi).mp ⟨a, by omega⟩).2; omega
    unfold IdSpace.bound at this; exact this

/-- `Graph::roll_id_batches` refuses exactly when `verify` does. -/
theorem roll_refused (N : Nat) (sp : IdSpace) (e : IdSpaceError) (hv : sp.verify N = .error e) :
    (match sp.verify N with | .error e => .error e | .ok () => sp.openBatch N) =
      (.error e : Except IdSpaceError IdSpace) := by
  rw [hv]

/-! ### The ordinary write path, end to end

A fresh batch over an empty bin: reserve `n`, create them, verify — it is
`[bound, bound + n)` and every step succeeds. -/

theorem lifecycle_fresh (N L live n : Nat) (hN : live + n < N) (hL : L < N) (hn : live + n ≤ L) :
    (IdSpace.restored N live sEmpty).reserve N true n sEmpty = .ok (List.range' live n) ∧
    (IdSpace.restored N live sEmpty).create N L (ival live (live + n)) =
      ((IdSpace.restored N live sEmpty).created N (ival live (live + n)), .ok ()) ∧
    ((IdSpace.restored N live sEmpty).created N (ival live (live + n))).verify N = .ok () := by
  have h0 : cnt N sEmpty = 0 := cnt_eq_zero.mpr (fun _ _ => rfl)
  have hfree : cnt N (freeOf sEmpty sEmpty sEmpty) = 0 :=
    cnt_eq_zero.mpr (fun i _ => by simp [freeOf_spec, sEmpty])
  have hasc : asc N (freeOf sEmpty sEmpty sEmpty) = [] := by
    apply List.eq_nil_of_length_eq_zero; rw [asc_length, hfree]
  have hab : above N sEmpty live = 0 := by
    rw [above_sFrom N _ (by omega)]; exact cnt_eq_zero.mpr (fun _ _ => rfl)
  have hinv := inv_restored N live sEmpty (by omega) (by unfold sLen; omega) (fun _ _ h => by cases h)
  have hok : CreateOk N L (IdSpace.restored N live sEmpty) (ival live (live + n)) := by
    refine ⟨fun i _ hi => ?_, fun i _ hn _ => ⟨?_, rfl⟩⟩
    · simp [ival] at hi; omega
    · simp [ival] at hn; simp [IdSpace.restored, sLen, h0]; exact hn.1
  have ⟨hcr, hinv'⟩ := create_succeeds N L _ _ (by omega) hL hinv hok
  refine ⟨?_, hcr, ?_⟩
  · rw [reserve_spec]
    simp only [IdSpace.restored, IdSpace.start, sLen, h0, Nat.add_zero, hab] at hasc hfree ⊢
    simp [hasc, hfree]
  · apply (verify_ok_iff N _).mpr
    refine ⟨hinv'.checked, fun i hi => ?_⟩
    have heb : (IdSpace.created N (IdSpace.restored N live sEmpty) (ival live (live + n))).eb = live := by
      simp [IdSpace.created, IdSpace.restored, sLen, h0]
    have hT : ∀ j, j < N → (IdSpace.created N (IdSpace.restored N live sEmpty) (ival live (live + n))).taken j
        = ival live (live + n) j := fun j _ => by simp [IdSpace.created, IdSpace.restored, sUnion, sEmpty]
    have habv : above N (IdSpace.created N (IdSpace.restored N live sEmpty) (ival live (live + n))).taken
        live = n := by
      rw [above_sFrom N _ (by omega)]; unfold sLen
      have : cnt N (ival live (live + n)) = live + n - live := cnt_interval _ _ _ (by omega)
      have e2 : cnt N (sFrom (IdSpace.created N (IdSpace.restored N live sEmpty)
          (ival live (live + n))).taken live) = cnt N (ival live (live + n)) := by
        apply cnt_congr; intro j hj; simp only [sFrom]; rw [hT j hj]; simp [ival]; omega
      rw [e2, this]; omega
    rw [heb, habv, hT i hi]; simp [ival]; exact fun a _ => a

end IdSpaceModel
