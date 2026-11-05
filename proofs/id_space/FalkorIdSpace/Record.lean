import FalkorIdSpace.Reserve
/-! `checked`, `cancel`, `refuse_recycled`, `create`, `refuse_undeletable`,
`release`: exactly what they refuse, and the state they leave. -/
namespace IdSpaceModel

/-- `checked` passes iff `entry_bound + |taken ≥ entry_bound|` fits in `u64`
and equals `live + recycled.len()`. -/
theorem checked_ok_iff (N : Nat) (sp : IdSpace) :
    sp.checked N = .ok () ↔
      sp.eb + above N sp.taken sp.eb < N ∧ sp.eb + above N sp.taken sp.eb = sp.bound N := by
  unfold IdSpace.checked checkedAdd
  by_cases h : sp.eb + above N sp.taken sp.eb < N
  · simp only [h, ite_true, true_and]
    by_cases h2 : sp.eb + above N sp.taken sp.eb = sp.bound N <;> simp [h2]
  · simp [h]

/-! ### `cancel` -/

theorem cancel_taken (N : Nat) (sp : IdSpace) (id : Nat) (h : sp.taken id = true) :
    sp.cancel N id = (sp, .error (.alreadyTaken id)) := by
  simp [IdSpace.cancel, h]

theorem cancel_fresh (N : Nat) (sp : IdSpace) (id : Nat) (h : sp.taken id = false) :
    sp.cancel N id =
      let sp' := { sp with taken := sIns sp.taken id, recycled := sIns sp.recycled id }
      (sp', sp'.checked N) := by
  simp [IdSpace.cancel, h]

/-! ### `refuse_recycled` -/

theorem refuseRecycled_ok (N : Nat) (sp : IdSpace) (nodes : IdSet) :
    sp.refuseRecycled N nodes = .ok () ↔
      ∀ i, i < N → ¬ (nodes i = true ∧ sp.recycled i = true) := by
  unfold IdSpace.refuseRecycled
  cases h : sMin N (sInter nodes sp.recycled) with
  | none =>
    simp only [true_iff]
    intro i hi ⟨a, b⟩
    have := sMin_none.mp h i hi; simp [sInter, a, b] at this
  | some id =>
    simp only [reduceCtorEq, false_iff, Classical.not_forall]
    obtain ⟨hl, hs, _⟩ := sMin_some.mp h
    simp [sInter] at hs
    exact ⟨id, hl, by simp [hs]⟩

/-- When it refuses, it names the lowest id already in the bin. -/
theorem refuseRecycled_err (N : Nat) (sp : IdSpace) (nodes : IdSet) (e : IdSpaceError)
    (h : sp.refuseRecycled N nodes = .error e) :
    ∃ id, e = .alreadyRecycled id ∧ id < N ∧ nodes id = true ∧ sp.recycled id = true ∧
      ∀ i, i < id → ¬ (nodes i = true ∧ sp.recycled i = true) := by
  unfold IdSpace.refuseRecycled at h
  cases hm : sMin N (sInter nodes sp.recycled) with
  | none => rw [hm] at h; cases h
  | some m =>
    rw [hm] at h; cases h
    obtain ⟨hl, hs, hmin⟩ := sMin_some.mp hm
    simp [sInter] at hs
    refine ⟨m, rfl, hl, hs.1, hs.2, fun i hi ⟨a, b⟩ => ?_⟩
    have := hmin i hi; simp [sInter, a, b] at this

/-! ### `create` -/

/-- `.expect("the sets are not disjoint")` never panics: in that branch the
intersection has a minimum. -/
theorem create_expect_unreachable (N : Nat) (a b : IdSet) (h : sDisjoint N b a = false) :
    (sMin N (sInter a b)).isSome := by
  cases hm : sMin N (sInter a b) with
  | some _ => rfl
  | none =>
    have hd : sDisjoint N b a = true := sDisjoint_iff.mpr (fun i hi ⟨x, y⟩ => by
      have := sMin_none.mp hm i hi; simp [sInter, x, y] at this)
    rw [h] at hd; cases hd

/-- What `create`'s refusals let through: every id below `ID_LIMIT` (`L`), and
every id that is not free is neither below the boundary (live before the batch)
nor already taken by this batch (claimed twice). -/
def CreateOk (N L : Nat) (sp : IdSpace) (nodes : IdSet) : Prop :=
  (∀ i, i < N → nodes i = true → i < L) ∧
  ∀ i, i < N → nodes i = true → sp.recycled i = false → (sp.eb ≤ i ∧ sp.taken i = false)

/-- The state `create` moves to once its refusals pass. -/
def IdSpace.created (N : Nat) (sp : IdSpace) (nodes : IdSet) : IdSpace :=
  { sp with recycled := sDiff sp.recycled nodes, live := sp.live + sLen N nodes,
            taken := sUnion sp.taken nodes }

/-- **The `ID_LIMIT` refusal names the highest id** (#2911): `create` refuses
with `IdOutOfRange(max nodes)` exactly when that max is `≥ L`, and then touches
nothing. -/
theorem create_out_of_range (N L : Nat) (sp : IdSpace) (nodes : IdSet) (m : Nat)
    (hm : sMax N nodes = some m) (hL : L ≤ m) :
    sp.create N L nodes = (sp, .error (.idOutOfRange m)) := by
  unfold IdSpace.create
  have : (sMax N nodes).filter (· ≥ L) = some m := by
    rw [hm]; simp [Option.filter, hL]
  rw [this]

/-- **`create` is all or nothing.** If its refusals pass it moves to
`created` and returns that state's `checked`; otherwise the state is untouched
and the result is an error (`IdOutOfRange` naming an id `≥ L`, or `AlreadyLive`). -/
theorem create_spec (N L : Nat) (sp : IdSpace) (nodes : IdSet) :
    (CreateOk N L sp nodes → sp.create N L nodes = (sp.created N nodes, (sp.created N nodes).checked N)) ∧
    (¬ CreateOk N L sp nodes → (sp.create N L nodes).1 = sp ∧
       ∃ e, (sp.create N L nodes).2 = .error e ∧
         ((∃ id, L ≤ id ∧ e = .idOutOfRange id) ∨ ∃ id, e = .alreadyLive id sp.eb)) := by
  unfold IdSpace.create CreateOk
  cases hmx : (sMax N nodes).filter (· ≥ L) with
  | some m =>
    rw [Option.filter_eq_some_iff] at hmx
    obtain ⟨hm, hge⟩ := hmx
    obtain ⟨hl, hs, _⟩ := sMax_some.mp hm
    refine ⟨fun h => absurd (h.1 m hl hs) (by simp at hge; omega),
      fun _ => ⟨rfl, _, rfl, .inl ⟨m, by simpa using hge, rfl⟩⟩⟩
  | none =>
  have hmx' : ∀ i, i < N → nodes i = true → i < L := by
    intro i hi hn
    cases hm : sMax N nodes with
    | none => have := sMax_none.mp hm i hi; rw [hn] at this; cases this
    | some m =>
      rw [hm] at hmx
      obtain ⟨_, _, hmax⟩ := sMax_some.mp hm
      have hmL : m < L := by
        apply Nat.lt_of_not_le; intro hle; simp [Option.filter, hle] at hmx
      have : i ≤ m := Nat.not_lt.mp (fun hlt => by have := hmax i hlt hi; rw [hn] at this; cases this)
      omega
  dsimp only
  cases hc : (sMin N (sDiff nodes sp.recycled)).filter (· < sp.eb) with
  | some id =>
    refine ⟨fun ⟨_, hall⟩ => ?_, fun _ => ⟨rfl, _, rfl, .inr ⟨id, rfl⟩⟩⟩
    exfalso
    rw [Option.filter_eq_some_iff] at hc
    obtain ⟨hmin, hlt⟩ := hc
    obtain ⟨hl, hs, _⟩ := sMin_some.mp hmin
    simp [sDiff] at hs
    have := (hall id hl hs.1 hs.2).1; simp at hlt; omega
  | none =>
    have hge : ∀ i, i < N → nodes i = true → sp.recycled i = false → sp.eb ≤ i := by
      intro i hi hn hr
      cases hm : sMin N (sDiff nodes sp.recycled) with
      | none => have := sMin_none.mp hm i hi; simp [sDiff, hn, hr] at this
      | some m =>
        rw [hm] at hc
        obtain ⟨hl, hs, hmin⟩ := sMin_some.mp hm
        have hmge : ¬ m < sp.eb := by
          intro hlt; simp [Option.filter, hlt] at hc
        have hmi : m ≤ i := by
          apply Nat.not_lt.mp; intro hlt
          have := hmin i hlt; simp [sDiff, hn, hr] at this
        omega
    by_cases hd : sDisjoint N sp.taken (sDiff nodes sp.recycled) = true
    · simp only [hd, Bool.not_true, Bool.false_eq_true, ite_false]
      have hdis := sDisjoint_iff.mp hd
      refine ⟨fun _ => rfl, fun hn => absurd ⟨hmx', fun i hi hn hr => ⟨hge i hi hn hr, ?_⟩⟩ hn⟩
      cases hcr : sp.taken i with
      | false => rfl
      | true => exact absurd ⟨hcr, by simp [sDiff, hn, hr]⟩ (hdis i hi)
    · have hd' : sDisjoint N sp.taken (sDiff nodes sp.recycled) = false := by simpa using hd
      simp only [hd', Bool.not_false, ite_true]
      refine ⟨fun ⟨_, hall⟩ => ?_, fun _ => ?_⟩
      · exfalso; apply hd
        apply sDisjoint_iff.mpr
        intro i hi ⟨hc, hdf⟩
        simp [sDiff] at hdf
        have := (hall i hi hdf.1 hdf.2).2; rw [hc] at this; cases this
      · split <;> exact ⟨rfl, _, rfl, .inr ⟨_, rfl⟩⟩

/-! ### `refuse_undeletable` -/

theorem refuseUndeletable_ok (N : Nat) (sp : IdSpace) (nodes : IdSet) :
    sp.refuseUndeletable N nodes = .ok () ↔
      ∀ i, i < N → nodes i = true → sp.eb ≤ i → sp.taken i = true := by
  unfold IdSpace.refuseUndeletable
  by_cases hq : (sMax N nodes).all (· < sp.eb) = true
  · simp only [hq, ite_true, true_iff]
    intro i hi hn hge
    cases hm : sMax N nodes with
    | none => have := sMax_none.mp hm i hi; rw [hn] at this; cases this
    | some m =>
      rw [hm] at hq; simp at hq
      obtain ⟨_, _, hmx⟩ := sMax_some.mp hm
      have := hmx i (by omega) hi; rw [hn] at this; cases this
  · simp only [hq, Bool.false_eq_true, ite_false]
    cases hc : (sMax N (sDiff nodes sp.taken)).filter (· ≥ sp.eb) with
    | some id =>
      simp only [reduceCtorEq, false_iff, Classical.not_forall]
      rw [Option.filter_eq_some_iff] at hc
      obtain ⟨hm, hge⟩ := hc
      obtain ⟨hl, hs, _⟩ := sMax_some.mp hm
      simp [sDiff] at hs hge
      exact ⟨id, hl, hs.1, hge, by simp [hs.2]⟩
    | none =>
      simp only [true_iff]
      intro i hi hn hge
      cases hcr : sp.taken i with
      | true => rfl
      | false =>
        exfalso
        cases hm : sMax N (sDiff nodes sp.taken) with
        | none => have := sMax_none.mp hm i hi; simp [sDiff, hn, hcr] at this
        | some m =>
          rw [hm] at hc
          obtain ⟨_, _, hmx⟩ := sMax_some.mp hm
          have hmi : i ≤ m := by
            apply Nat.not_lt.mp; intro hlt
            have := hmx i hlt hi; simp [sDiff, hn, hcr] at this
          simp [Option.filter] at hc; omega

/-- …and when it refuses it names the highest offending id. -/
theorem refuseUndeletable_err (N : Nat) (sp : IdSpace) (nodes : IdSet) (e : IdSpaceError)
    (h : sp.refuseUndeletable N nodes = .error e) :
    ∃ id, e = .neverCreated id ∧ id < N ∧ nodes id = true ∧ sp.eb ≤ id ∧ sp.taken id = false ∧
      ∀ i, id < i → i < N → nodes i = true → sp.taken i = true := by
  unfold IdSpace.refuseUndeletable at h
  split at h; · cases h
  split at h
  · rename_i id' hc
    cases h
    rw [Option.filter_eq_some_iff] at hc
    obtain ⟨hm, hge⟩ := hc
    obtain ⟨hl, hs, hmx⟩ := sMax_some.mp hm
    simp [sDiff] at hs hge
    refine ⟨id', rfl, hl, hs.1, hge, hs.2, fun i hi hiN hn => ?_⟩
    have := hmx i hi hiN
    cases hc : sp.taken i <;> simp_all [sDiff]
  · cases h

/-! ### `release` and `refuse_not_live` -/

/-- What `refuse_not_live` (and so `release`'s refusals) let through: nothing
requested is already free, and everything requested at or above the boundary
was taken by this batch. -/
def ReleaseOk (N : Nat) (sp : IdSpace) (requested : IdSet) : Prop :=
  (∀ i, i < N → ¬ (requested i = true ∧ sp.recycled i = true)) ∧
  (∀ i, i < N → requested i = true → sp.eb ≤ i → sp.taken i = true)

/-- **`refuse_not_live` (#3022) passes exactly on `ReleaseOk`.** -/
theorem refuseNotLive_ok (N : Nat) (sp : IdSpace) (ids : IdSet) :
    sp.refuseNotLive N ids = .ok () ↔ ReleaseOk N sp ids := by
  unfold IdSpace.refuseNotLive ReleaseOk
  cases h1 : sp.refuseRecycled N ids with
  | error e =>
    obtain ⟨id, rfl, hid, hn, hr, _⟩ := refuseRecycled_err N sp ids e h1
    simp only [reduceCtorEq, false_iff]
    exact fun ⟨h, _⟩ => h id hid ⟨hn, hr⟩
  | ok u =>
    cases u
    have h1' := (refuseRecycled_ok N sp ids).mp h1
    simp only
    rw [refuseUndeletable_ok]
    exact ⟨fun h => ⟨h1', h⟩, fun h => h.2⟩

/-- …and when it refuses, it names the lowest free id (`AlreadyRecycled`, checked
first) or else the highest never-allocated one (`NeverCreated`). -/
theorem refuseNotLive_err (N : Nat) (sp : IdSpace) (ids : IdSet) (e : IdSpaceError)
    (h : sp.refuseNotLive N ids = .error e) :
    (∃ id, e = .alreadyRecycled id ∧ id < N ∧ ids id = true ∧ sp.recycled id = true ∧
      ∀ i, i < id → ¬ (ids i = true ∧ sp.recycled i = true)) ∨
    ((∀ i, i < N → ¬ (ids i = true ∧ sp.recycled i = true)) ∧
     ∃ id, e = .neverCreated id ∧ id < N ∧ ids id = true ∧ sp.eb ≤ id ∧ sp.taken id = false ∧
      ∀ i, id < i → i < N → ids i = true → sp.taken i = true) := by
  unfold IdSpace.refuseNotLive at h
  cases h1 : sp.refuseRecycled N ids with
  | error e' =>
    rw [h1] at h; cases h
    exact .inl (refuseRecycled_err N sp ids _ h1)
  | ok u =>
    cases u; rw [h1] at h
    exact .inr ⟨(refuseRecycled_ok N sp ids).mp h1, refuseUndeletable_err N sp ids e h⟩

/-- `refuse_not_live` changes nothing: it is a pure check (`&self`). -/
theorem refuseNotLive_pure (N : Nat) (sp : IdSpace) (ids : IdSet) :
    sp.refuseNotLive N ids = (match sp.refuseRecycled N ids with
      | .error e => .error e | .ok () => sp.refuseUndeletable N ids) := rfl

/-- The state `release` moves to once its refusals pass. -/
def IdSpace.afterRelease (N : Nat) (sp : IdSpace) (freed : IdSet) : IdSpace :=
  { sp with recycled := sUnion sp.recycled freed, live := sp.live - sLen N freed,
            released := sUnion sp.released freed }

/-- **`release` is all or nothing**, judged over everything *requested*: the
refusals pass iff `ReleaseOk`, and then the state frees `freed` (and records it
in `released`); otherwise it is untouched and the error is `AlreadyRecycled`
(checked first) or `NeverCreated`. -/
theorem release_spec (N : Nat) (sp : IdSpace) (requested freed : IdSet) :
    (ReleaseOk N sp requested →
      sp.release N requested freed = (sp.afterRelease N freed, (sp.afterRelease N freed).checked N)) ∧
    (¬ ReleaseOk N sp requested → (sp.release N requested freed).1 = sp ∧
      ∃ e, (sp.release N requested freed).2 = .error e ∧
        ((∃ id, e = .alreadyRecycled id) ∨ ∃ id, e = .neverCreated id)) := by
  unfold IdSpace.release
  cases h : sp.refuseNotLive N requested with
  | error e =>
    have hn : ¬ ReleaseOk N sp requested := fun hk => by
      rw [(refuseNotLive_ok N sp requested).mpr hk] at h; cases h
    refine ⟨fun hk => absurd hk hn, fun _ => ⟨rfl, e, rfl, ?_⟩⟩
    rcases refuseNotLive_err N sp requested e h with ⟨id, rfl, _⟩ | ⟨_, id, rfl, _⟩
    · exact .inl ⟨id, rfl⟩
    · exact .inr ⟨id, rfl⟩
  | ok u =>
    cases u
    have hk := (refuseNotLive_ok N sp requested).mp h
    exact ⟨fun _ => rfl, fun hn => absurd hk hn⟩

end IdSpaceModel
