import GraphQueries.IdContract
/-
# Discharging `IdSpaceContract`: the Rust `IdSpace` mutators satisfy it

`IdContract.lean` states what graph.rs needs from `IdSpace` as the hypothesis
`IdSpaceContract O`. Here `rustIdOps` models the five mutators of
`graph/src/graph/id_space.rs` (origin/main `e8f8a3017`) line by line over the
list form of `IdS` (a `RoaringTreemap` is a duplicate-free `List Nat`; `min`/
`max` are `smin`/`smax`), and `rustIdOps_contract` proves the contract for it.
So every theorem that takes `(hC : IdSpaceContract O)` holds for the real
`IdSpace` (instantiate `O := rustIdOps`, `hC := rustIdOps_contract`).

This copies proofs/id_space's model (`IdSpaceModel.IdSpace.create/cancel/
release/openBatch/verify`, `Model.lean`) into `IdS`'s list form. `released`
(#3022, a per-batch field graph.rs never reads) is not part of `IdS`.
-/
namespace GQ

/-- `RoaringTreemap::min`. -/
def smin : List Nat → Option Nat
  | [] => none
  | x :: xs => match smin xs with
    | none => some x
    | some m => some (min x m)

/-- `RoaringTreemap::max`. -/
def smax : List Nat → Option Nat
  | [] => none
  | x :: xs => match smax xs with
    | none => some x
    | some m => some (max x m)

/-- `a & b`. -/
def inter (a b : List Nat) : List Nat := a.filter (fun x => x ∈ b)

theorem smin_le : ∀ (l : List Nat) (n : Nat), n ∈ l → ∃ m, smin l = some m ∧ m ≤ n
  | [], _, h => by simp at h
  | x :: xs, n, h => by
    simp only [smin]
    rcases List.mem_cons.mp h with rfl | h
    · cases smin xs with
      | none => exact ⟨n, rfl, Nat.le_refl _⟩
      | some m => exact ⟨min n m, rfl, Nat.min_le_left _ _⟩
    · obtain ⟨m, hm, hle⟩ := smin_le xs n h
      rw [hm]; exact ⟨min x m, rfl, by omega⟩

theorem smax_ge : ∀ (l : List Nat) (n : Nat), n ∈ l → ∃ m, smax l = some m ∧ n ≤ m
  | [], _, h => by simp at h
  | x :: xs, n, h => by
    simp only [smax]
    rcases List.mem_cons.mp h with rfl | h
    · cases smax xs with
      | none => exact ⟨n, rfl, Nat.le_refl _⟩
      | some m => exact ⟨max n m, rfl, Nat.le_max_left _ _⟩
    · obtain ⟨m, hm, hle⟩ := smax_ge xs n h
      rw [hm]; exact ⟨max x m, rfl, by omega⟩

/-- `above(set, bound)`: how many ids of `set` are `≥ bound`. -/
def above (t : List Nat) (eb : Nat) : Nat := (t.filter (fun x => decide (eb ≤ x))).length

/-- `ID_LIMIT = GrB_INDEX_MAX = 2^60 - 1` (id_space.rs:112). -/
def ID_LIMIT : Nat := 2 ^ 60 - 1

/-- `checked` (id_space.rs:403): `entry_bound.checked_add(taken) == Some(bound)`. -/
def rChecked (s : IdS) : Bool :=
  let e := s.entryBound + above s.taken s.entryBound
  decide (e < 2 ^ 64) && e == s.bound

/-- `create` (id_space.rs:582). -/
def rCreate (s : IdS) (ns : List Nat) : Option IdS :=
  match (smax ns).filter (· ≥ ID_LIMIT) with
  | some _ => none                                   -- IdOutOfRange
  | none =>
    let claimed := diff ns s.recycled
    match (smin claimed).filter (· < s.entryBound) with
    | some _ => none                                 -- AlreadyLive (before the batch)
    | none =>
      if inter claimed s.taken ≠ [] then none        -- AlreadyLive (this batch)
      else
        let s' : IdS := { s with recycled := diff s.recycled ns, live := s.live + ns.length,
                                 taken := union s.taken ns }
        if rChecked s' then some s' else none

/-- `refuse_recycled` (id_space.rs:550): `true` = `Ok(())`. -/
def rRefuseRecycled (s : IdS) (ns : List Nat) : Bool := (smin (inter ns s.recycled)).isNone

/-- `refuse_undeletable` (id_space.rs:638): `true` = `Ok(())`. -/
def rRefuseUndeletable (s : IdS) (ns : List Nat) : Bool :=
  if (smax ns).all (· < s.entryBound) then true
  else ((smax (diff ns s.taken)).filter (· ≥ s.entryBound)).isNone

/-- `release` (id_space.rs:703): `refuse_not_live(requested)?` (= `refuse_recycled` then
`refuse_undeletable`, :679), then `recycled |= freed; live -= freed.len()`. -/
def rRelease (s : IdS) (req fr : List Nat) : Option IdS :=
  if rRefuseRecycled s req && rRefuseUndeletable s req then
    let s' : IdS := { s with recycled := union s.recycled fr, live := s.live - fr.length }
    if rChecked s' then some s' else none
  else none

/-- `cancel` (id_space.rs:529): `taken.insert(id)` (refuse if present), `recycled.insert(id)`,
`checked()`. -/
def rCancel (s : IdS) (id : Nat) : Option IdS :=
  if id ∈ s.taken then none
  else
    let s' : IdS := { s with taken := ins s.taken id, recycled := ins s.recycled id }
    if rChecked s' then some s' else none

/-- `open_batch` (id_space.rs:447): `checked()?`, then `entry_bound = bound()`, clear `taken`. -/
def rOpen (s : IdS) : Option IdS :=
  if rChecked s then some { s with entryBound := s.bound, taken := [] } else none

/-- `verify` (id_space.rs:737): `checked()?`, then the hole test: the lowest id of `taken`
at or above the boundary must be the boundary, and the ids above it contiguous. -/
def rVerify (s : IdS) : Bool :=
  rChecked s &&
    let created := above s.taken s.entryBound
    match smin (s.taken.filter (fun x => decide (s.entryBound ≤ x))), smax s.taken with
    | some lo, some hi => !(lo != s.entryBound || created - 1 != hi - s.entryBound)
    | _, _ => true

/-- The Rust `IdSpace` mutators. -/
def rustIdOps : IdSpaceOps :=
  { create := rCreate, release := rRelease, cancel := rCancel, openBatch := rOpen,
    verify := rVerify }

/-- **The contract holds for the Rust `IdSpace`.** -/
theorem rustIdOps_contract : IdSpaceContract rustIdOps where
  create_ok := by
    intro s ns s' h
    simp only [rustIdOps, rCreate] at h
    split at h
    · simp at h
    · split at h
      · simp at h
      · split at h
        · simp at h
        · split at h
          · simp at h; subst h; exact ⟨rfl, rfl⟩
          · simp at h
  create_refuses_live := by
    intro s ns n hn hr hb
    simp only [rustIdOps, rCreate]
    split
    · rfl
    · have hc : n ∈ diff ns s.recycled := by simp [diff, hn, hr]
      obtain ⟨m, hm, hle⟩ := smin_le _ n hc
      have : (smin (diff ns s.recycled)).filter (· < s.entryBound) = some m := by
        rw [hm]; simp; omega
      simp only [this]
  create_refuses_limit := by
    intro s ns n hn hl
    simp only [rustIdOps, rCreate]
    obtain ⟨m, hm, hle⟩ := smax_ge ns n hn
    have : (smax ns).filter (· ≥ ID_LIMIT) = some m := by
      rw [hm]; simp [ID_LIMIT]; omega
    simp only [this]
  release_ok := by
    intro s req fr s' h
    simp only [rustIdOps, rRelease] at h
    split at h
    · split at h
      · simp at h; subst h; exact ⟨rfl, rfl⟩
      · simp at h
    · simp at h
  release_refuses_recycled := by
    intro s req fr n hn hr
    simp only [rustIdOps, rRelease]
    have hi : n ∈ inter req s.recycled := by simp [inter, hn, hr]
    obtain ⟨m, hm, -⟩ := smin_le _ n hi
    have : rRefuseRecycled s req = false := by simp [rRefuseRecycled, hm]
    simp [this]
  cancel_ok := by
    intro s id s' h
    simp only [rustIdOps, rCancel] at h
    split at h
    · simp at h
    · split at h
      · simp at h; subst h; exact ⟨rfl, rfl⟩
      · simp at h
  open_ok := by
    intro s s' h
    simp only [rustIdOps, rOpen] at h
    split at h
    · simp at h; subst h; exact ⟨rfl, rfl, rfl, rfl⟩
    · simp at h

#print axioms rustIdOps_contract

end GQ
