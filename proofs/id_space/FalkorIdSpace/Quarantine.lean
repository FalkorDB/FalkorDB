import FalkorIdSpace.Released
/-! # The quarantine (#3164, fixed by #3641 `e0d9c2188`)

A write query is many batches over one MVCC version. A row it still carries may name an
entity an earlier batch deleted (a `CALL {}` body commits once per input row), so reissuing that
id to a node the next invocation creates makes the row's `DELETE` remove the new node.
`quarantined` is every id earlier batches of this version freed (`released`) or took and handed
back (`taken ∩ recycled`, a cancelled creation); `open_batch` accumulates it, `reclaim_ids`
(hence `reserve`) subtracts it, `new_version`/`new`/`restored` start it empty, and
`end_quarantine` clears it.

* `openBatch_quarantines` — what `open_batch` adds, exactly.
* `reserve_not_quarantined` — neither the reclaimed nor the fresh part of a `reserve` is in
  `quarantined` (the fresh part needs `QuarOk`: the quarantine lies below the boundary).
* `roll_quarOk` — a verified roll keeps `QuarOk`.
* **`rolled_not_reissued`** — an id the previous batch freed or cancelled is never handed out
  by the next batch of the same version.
* `pre3641_*` — the model without the quarantine reissues (fixed by #3641, `e0d9c2188`).
-/
namespace IdSpaceModel

theorem endQuarantine_spec (sp : IdSpace) :
    sp.endQuarantine.quarantined = sEmpty ∧ sp.endQuarantine.live = sp.live ∧
    sp.endQuarantine.recycled = sp.recycled ∧ sp.endQuarantine.eb = sp.eb ∧
    sp.endQuarantine.taken = sp.taken ∧ sp.endQuarantine.released = sp.released :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem quarantined_new : ∀ i, (IdSpace.new).quarantined i = false := fun _ => rfl
theorem quarantined_restored (N live : Nat) (rec : IdSet) :
    ∀ i, (IdSpace.restored N live rec).quarantined i = false := fun _ => rfl
/-- `new_version` starts a fresh quarantine: the freed ids are ordinary free ids to the next query. -/
theorem quarantined_newVersion (N : Nat) (sp : IdSpace) :
    ∀ i, (sp.newVersion N).quarantined i = false := fun _ => rfl

/-- `open_batch` (when it passes) adds exactly what the batch released plus what it cancelled
(`taken ∩ recycled`), keeps the free set and count, and clears `taken`/`released`. -/
theorem openBatch_quarantines (N : Nat) (sp sp' : IdSpace) (h : sp.openBatch N = .ok sp') :
    (∀ i, sp'.quarantined i = (sp.quarantined i || sp.released i || (sp.taken i && sp.recycled i))) ∧
    sp'.recycled = sp.recycled ∧ sp'.live = sp.live ∧ sp'.eb = sp.bound N := by
  unfold IdSpace.openBatch at h
  split at h
  · cases h
  · cases h
    refine ⟨fun i => ?_, rfl, rfl, rfl⟩
    simp [IdSpace.rolledQuar, sUnion, sInter, Bool.or_assoc]

/-- The quarantine lies below the entry boundary. -/
def QuarOk (N : Nat) (sp : IdSpace) : Prop :=
  ∀ i, i < N → sp.quarantined i = true → i < sp.eb

theorem quarOk_new (N : Nat) : QuarOk N IdSpace.new := fun _ _ h => by cases h
theorem quarOk_restored (N live : Nat) (rec : IdSet) : QuarOk N (IdSpace.restored N live rec) :=
  fun _ _ h => by cases h
theorem quarOk_endQuarantine (N : Nat) (sp : IdSpace) : QuarOk N sp.endQuarantine :=
  fun _ _ h => by simp [IdSpace.endQuarantine, sEmpty] at h

/-- **A verified roll keeps `QuarOk`**: everything released or cancelled is below the new boundary
(`relOk`, then the dense-batch characterisation of `verify`). -/
theorem roll_quarOk (N : Nat) (sp : IdSpace) (h : Inv N sp) (hq : QuarOk N sp)
    (hv : sp.verify N = .ok ()) : QuarOk N (sp.rolled N) := by
  have ⟨hc, hden⟩ := (verify_ok_iff N sp).mp hv
  have ⟨_, hcnt⟩ := (checked_ok_iff N sp).mp hc
  have hbd : sp.bound N = sp.eb + above N sp.taken sp.eb := hcnt.symm
  have htk : ∀ i, i < N → sp.taken i = true → i < sp.bound N := by
    intro i hi ht
    by_cases hlt : i < sp.eb
    · omega
    · have := ((hden i hi).mp ⟨ht, by omega⟩).2; omega
  intro i hi hqi
  show i < sp.bound N
  simp only [IdSpace.rolled, IdSpace.rolledQuar, sUnion, sInter, Bool.or_eq_true,
    Bool.and_eq_true] at hqi
  rcases hqi with a | a | ⟨a, _⟩
  · have := hq i hi a; omega
  · rcases h.relOk i hi a with b | b
    · omega
    · exact htk i hi b
  · exact htk i hi a

/-- `reserve` never hands out a quarantined id: the reclaimed part is `pool − taken − issued −
quarantined` by construction, and the fresh run starts at or above the boundary, above the quarantine. -/
theorem reserve_not_quarantined (N : Nat) (sp : IdSpace) (hN : 0 < N) (hq : QuarOk N sp)
    (count : Nat) (issued : IdSet) (ids : List Nat)
    (h : sp.reserve N true count issued = .ok ids) : ∀ x ∈ ids, sp.quarantined x = false := by
  rw [reserve_spec] at h
  cases h
  intro x hx
  rcases List.mem_append.mp hx with hx | hx
  · exact (reclaimed_free N sp.recycled sp.taken issued sp.quarantined count x hx).2.2.2.2
  · have hs := List.mem_range'_1.mp hx
    have hge : sp.eb ≤ sp.start N issued := by unfold IdSpace.start; omega
    have hlt : x < N := by
      have := Nat.mod_lt (sp.start N issued + (count - min count
        (cnt N (freeOf sp.recycled sp.taken issued sp.quarantined)))) hN
      omega
    cases hqx : sp.quarantined x
    · rfl
    · have := hq x hlt hqx; omega

/-- **An id the previous batch freed or cancelled is not reissued by the next batch of the same
version** (#3164): roll a verified batch, reserve in the new one. -/
theorem rolled_not_reissued (N : Nat) (sp : IdSpace) (hN : 0 < N) (h : Inv N sp)
    (hq : QuarOk N sp) (hv : sp.verify N = .ok ()) (count : Nat) (issued : IdSet) (ids : List Nat)
    (hr : (sp.rolled N).reserve N true count issued = .ok ids) :
    ∀ x ∈ ids, ¬ (sp.quarantined x = true ∨ sp.released x = true ∨
      (sp.taken x = true ∧ sp.recycled x = true)) := by
  intro x hx hbad
  have hnq := reserve_not_quarantined N (sp.rolled N) hN (roll_quarOk N sp h hq hv) count issued ids hr x hx
  have : (sp.rolled N).quarantined x = true := by
    simp only [IdSpace.rolled, IdSpace.rolledQuar, sUnion, sInter]
    rcases hbad with a | a | ⟨a, b⟩ <;> simp [*]
  rw [this] at hnq; cases hnq

/-! ### Before #3641: the id came back -/

section Pre3641
/-- the root file's test helpers (universe N = 32), local to this module. -/
private def set (l : List Nat) : IdSet := fun i => l.contains i
private def resv (r : Except String (List Nat)) : Option (List Nat) := r.toOption
private def ob (sp : IdSpace) : IdSpace := match sp.openBatch 32 with | .ok s => s | .error _ => sp

/-- Pre-#3641 `open_batch` was `new_version` (no quarantine). `IdSpace::restored(3)`, release id 0
in batch 1, roll: the freed 0 is in the pool. -/
def pre3641_s1 : IdSpace := (((ob (IdSpace.restored 32 3 (set []))).release 32 (set [0]) (set [0])).1)

/-- pre-#3641 (`open_batch` = `new_version`): the next batch of the version reserves the id the
previous one freed — a `CALL {}` row still holding node 0 would `DELETE` the new node. -/
theorem pre3641_reissues :
    resv (pre3641_s1.newVersion 32 |>.reserve 32 true 1 (set [])) = some [0] := by decide

/-- after #3641 (`open_batch` quarantines it): fresh id 3, not the freed 0. -/
theorem post3641_not_reissued :
    resv ((ob pre3641_s1).reserve 32 true 1 (set [])) = some [3] := by decide

/-- the rolled space carries the freed id in its quarantine; `new_version` (what `open_batch` used
to be) does not. -/
theorem pre3641_openBatch_quarantines :
    ((pre3641_s1.openBatch 32).toOption.map (·.quarantined 0)) = some true := by decide

/-- the cancelled-creation variant (the second #3164 test): 0 reserved and cancelled, then rolled. -/
def pre3641_c1 : IdSpace :=
  (((ob (IdSpace.restored 32 0 (set []))).cancel 32 0).1)

theorem pre3641_cancelled_reissues :
    resv (pre3641_c1.newVersion 32 |>.reserve 32 true 1 (set [])) = some [0] := by decide

theorem post3641_cancelled_not_reissued :
    resv ((ob pre3641_c1).reserve 32 true 1 (set [])) = some [1] := by decide

/-- a new version starts clean: the freed id is an ordinary free one. -/
theorem post3641_new_version_reissues :
    resv ((ob pre3641_s1).newVersion 32 |>.reserve 32 true 1 (set [])) = some [0] := by decide

end Pre3641

end IdSpaceModel
