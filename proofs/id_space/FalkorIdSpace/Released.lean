import FalkorIdSpace.Lifecycle
/-! # `released`, `taken()`, `released()` and `refuse_not_live` (#3022, `18fc277b9`)

`released` is per-batch state: every id the open batch freed, whether or not a
later create reclaimed it. It is cleared by `open_batch` and not carried by
`new_version`, grows only in `release`, and is untouched by `create`, `cancel`
and the checks. `refuse_not_live` is both halves of liveness for a caller that
acts on an entity rather than creating or freeing it.
-/
namespace IdSpaceModel

/-- `taken()` (`:311`) and `released()` (`:317`) return the fields. -/
theorem takenSet_eq (sp : IdSpace) : sp.takenSet = sp.taken := rfl
theorem releasedSet_eq (sp : IdSpace) : sp.releasedSet = sp.released := rfl

/-! ### Where `released` starts empty -/

theorem released_new : ∀ i, (IdSpace.new).released i = false := fun _ => rfl
theorem released_restored (N live : Nat) (rec : IdSet) :
    ∀ i, (IdSpace.restored N live rec).released i = false := fun _ => rfl
/-- `new_version` does not carry the batch's `released` forward. -/
theorem released_newVersion (N : Nat) (sp : IdSpace) :
    ∀ i, (sp.newVersion N).released i = false := fun _ => rfl
/-- `open_batch` clears `released` together with `taken`. -/
theorem openBatch_clears (N : Nat) (sp sp' : IdSpace) (h : sp.openBatch N = .ok sp') :
    (∀ i, sp'.released i = false) ∧ (∀ i, sp'.taken i = false) := by
  unfold IdSpace.openBatch at h
  split at h
  · cases h
  · cases h; exact ⟨fun _ => rfl, fun _ => rfl⟩

/-! ### Who changes it -/

theorem create_keeps_released (N L : Nat) (sp : IdSpace) (nodes : IdSet) :
    (sp.create N L nodes).1.released = sp.released := by
  by_cases hk : CreateOk N L sp nodes
  · rw [(create_spec N L sp nodes).1 hk]; rfl
  · rw [((create_spec N L sp nodes).2 hk).1]

theorem cancel_keeps_released (N : Nat) (sp : IdSpace) (id : Nat) :
    (sp.cancel N id).1.released = sp.released := by
  unfold IdSpace.cancel; split <;> rfl

/-- A release that passes its refusals adds exactly `freed`; a refused one adds nothing. -/
theorem release_records (N : Nat) (sp : IdSpace) (requested freed : IdSet) :
    (ReleaseOk N sp requested →
      (sp.release N requested freed).1.released = sUnion sp.released freed) ∧
    (¬ ReleaseOk N sp requested → (sp.release N requested freed).1.released = sp.released) :=
  ⟨fun hk => by rw [(release_spec N sp requested freed).1 hk]; rfl,
   fun hk => by rw [((release_spec N sp requested freed).2 hk).1]⟩

/-! ### `refuse_not_live` accepts exactly live ids -/

/-- **`refuse_not_live` passes iff every id named is live** (`liveSet`: not free,
and below the entry boundary or taken by this batch) — in any state, reachable or
not. -/
theorem refuseNotLive_live (N : Nat) (sp : IdSpace) (ids : IdSet) :
    sp.refuseNotLive N ids = .ok () ↔ ∀ i, i < N → ids i = true → liveSet sp i = true := by
  rw [refuseNotLive_ok]
  constructor
  · intro ⟨h1, h2⟩ i hi hn
    have hr : sp.recycled i = false := by
      cases e : sp.recycled i
      · rfl
      · exact absurd ⟨hn, e⟩ (h1 i hi)
    simp only [liveSet, hr, Bool.not_false, Bool.true_and, Bool.or_eq_true, decide_eq_true_eq]
    by_cases hlt : i < sp.eb
    · exact .inl hlt
    · exact .inr (h2 i hi hn (by omega))
  · intro h
    refine ⟨fun i hi ⟨hn, hr⟩ => ?_, fun i hi hn hge => ?_⟩
    · have := h i hi hn; simp [liveSet, hr] at this
    · have := h i hi hn; simp [liveSet] at this
      rcases this.2 with a | a
      · omega
      · exact a

/-- In a reachable state `live` counts exactly the ids `refuse_not_live` accepts. -/
theorem live_eq_accepted (N : Nat) (sp : IdSpace) (h : Inv N sp) :
    sp.live = cnt N (liveSet sp) ∧
    ∀ i, i < N → (sp.refuseNotLive N (fun j => decide (j = i)) = .ok () ↔ liveSet sp i = true) := by
  refine ⟨live_eq N sp h, fun i hi => ?_⟩
  rw [refuseNotLive_live]
  constructor
  · intro hh; exact hh i hi (by simp)
  · intro hl j _ hj; simp at hj; subst hj; exact hl

/-! ### The batch exemption the effects path takes

`apply_record`'s `CreateEdge` checks `(src ∪ dst) − released` (`apply.rs:279`).
In a reachable state that admits exactly the endpoints that are live now, or
that this batch released — and a released id was live *in this batch*: below
the entry boundary or taken by it (`Inv.relOk`). Never an id that was never
allocated, nor one freed by an earlier batch. -/
theorem refuseNotLive_minus_released (N : Nat) (sp : IdSpace) (h : Inv N sp) (ids : IdSet)
    (hok : sp.refuseNotLive N (sDiff ids sp.released) = .ok ()) :
    ∀ i, i < N → ids i = true →
      liveSet sp i = true ∨ (sp.released i = true ∧ (i < sp.eb ∨ sp.taken i = true)) := by
  intro i hi hn
  cases hr : sp.released i
  · exact .inl ((refuseNotLive_live N sp _).mp hok i hi (by simp [sDiff, hn, hr]))
  · exact .inr ⟨rfl, h.relOk i hi hr⟩

/-- **An id freed by an earlier batch is refused in the next one.** After the
batch rolls (`open_batch`), `released` is empty, so `ids − released = ids`, and a
free id there is `AlreadyRecycled` — what `liveness_setup`'s `new_version` relies on. -/
theorem freed_earlier_refused (N : Nat) (sp sp' : IdSpace) (hob : sp.openBatch N = .ok sp')
    (i : Nat) (hi : i < N) (hr : sp'.recycled i = true) :
    ∃ e, sp'.refuseNotLive N (sDiff (fun j => decide (j = i)) sp'.released) = .error e := by
  have ⟨hrel, _⟩ := openBatch_clears N sp sp' hob
  cases hq : sp'.refuseNotLive N (sDiff (fun j => decide (j = i)) sp'.released) with
  | error e => exact ⟨e, rfl⟩
  | ok u =>
    cases u
    have := (refuseNotLive_live N sp' _).mp hq i hi (by simp [sDiff, hrel i])
    simp [liveSet, hr] at this

end IdSpaceModel
