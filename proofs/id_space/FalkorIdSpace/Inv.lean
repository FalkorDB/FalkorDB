import FalkorIdSpace.Verify
/-! # The invariant every reachable `IdSpace` satisfies, and what it buys

`Inv` = the count half (`checked` passes: `entry + |taken ≥ entry| = live +
|recycled|`, no overflow), plus the two structural facts the Rust relies on
without checking: every free id is below the entry boundary or was taken by the
batch, and the batch never took `u64::MAX`; plus (#3022) every id the batch
*released* is below the entry boundary or was taken by it (`relOk`).

From it: `live` is exactly the number of live ids (`live_eq`), so `release`
never underflows `live -= freed.len()` (`release_no_underflow`); every
successful mutator preserves it; and `open_batch` after a passing `verify` opens
a batch that satisfies it again (`roll_inv`), which is `Graph::roll_id_batches`.
-/
namespace IdSpaceModel

structure Inv (N : Nat) (sp : IdSpace) : Prop where
  checked : sp.checked N = .ok ()
  binOk   : ∀ i, i < N → sp.recycled i = true → i < sp.eb ∨ sp.taken i = true
  noMax   : sp.taken (umax N) = false
  /-- #3022: everything the open batch released was live in it — below the
  entry boundary (live when the batch began) or taken by the batch. -/
  relOk   : ∀ i, i < N → sp.released i = true → i < sp.eb ∨ sp.taken i = true

/-- The ids that are live: below the boundary or taken by the batch, and not free. -/
def liveSet (sp : IdSpace) : IdSet := fun i => !sp.recycled i && (decide (i < sp.eb) || sp.taken i)

theorem cnt_lt (N eb : Nat) (h : eb ≤ N) : cnt N (fun i => decide (i < eb)) = eb := by
  have := cnt_interval 0 eb N h
  rw [Nat.sub_zero] at this
  calc cnt N (fun i => decide (i < eb))
      = cnt N (fun i => decide (0 ≤ i) && decide (i < eb)) := cnt_congr N (fun i _ => by simp)
    _ = eb := this

theorem cnt_ins (N : Nat) (s : IdSet) (id : Nat) (h : id < N) :
    cnt N (sIns s id) = cnt N s + (if s id then 0 else 1) := by
  rw [cnt_split N (sIns s id) s]
  have e1 : cnt N (fun i => sIns s id i && s i) = cnt N s :=
    cnt_congr N (fun i _ => by simp [sIns]; cases s i <;> simp)
  rw [e1]
  congr 1
  by_cases hs : s id = true
  · simp only [hs, ite_true]
    apply cnt_eq_zero.mpr; intro i _
    by_cases e : i = id
    · subst e; simp [hs]
    · simp [sIns, e]
  · simp only [hs, Bool.false_eq_true, ite_false]
    have := cnt_interval id (id + 1) N (by omega)
    rw [Nat.add_sub_cancel_left] at this; rw [← this]
    apply cnt_congr; intro i _
    simp [sIns]; by_cases e : i = id
    · subst e; simp at hs; simp [hs]
    · simp [e]; omega

theorem eb_lt {N : Nat} {sp : IdSpace} (h : Inv N sp) : sp.eb < N := by
  have := ((checked_ok_iff N sp).mp h.checked).1; omega

/-- **`live` counts the live ids.** -/
theorem live_eq (N : Nat) (sp : IdSpace) (h : Inv N sp) : sp.live = cnt N (liveSet sp) := by
  have ⟨_, hc⟩ := (checked_ok_iff N sp).mp h.checked
  have hb : sp.eb ≤ N := Nat.le_of_lt (eb_lt h)
  let U : IdSet := fun i => decide (i < sp.eb) || sp.taken i
  have hU : cnt N U = sp.eb + above N sp.taken sp.eb := by
    have h1 := cnt_or_disjoint N (fun i => decide (i < sp.eb)) (sFrom sp.taken sp.eb)
      (fun i _ ⟨a, b⟩ => by simp [sFrom] at a b; omega)
    rw [cnt_lt N sp.eb hb] at h1
    rw [above_sFrom N sp.taken hb]; unfold sLen
    rw [← h1]
    apply cnt_congr; intro i _
    simp only [U, sFrom]; by_cases e : i < sp.eb
    · simp [e]
    · simp [e]; omega
  have hsplit := cnt_split N U sp.recycled
  have e1 : cnt N (fun i => U i && sp.recycled i) = cnt N sp.recycled := cnt_congr N (fun i hi => by
    cases hr : sp.recycled i
    · simp
    · rcases h.binOk i hi hr with a | a <;> simp [U, a])
  have e2 : cnt N (fun i => U i && !sp.recycled i) = cnt N (liveSet sp) :=
    cnt_congr N (fun i _ => by simp only [U, liveSet]; cases sp.recycled i <;> simp)
  unfold IdSpace.bound sLen at hc
  omega

theorem inv_restored (N live : Nat) (rec : IdSet) (hN : 1 ≤ N)
    (hfit : live + sLen N rec < N) (hrec : ∀ i, i < N → rec i = true → i < live + sLen N rec) :
    Inv N (IdSpace.restored N live rec) := by
  have hab : above N sEmpty (live + sLen N rec) = 0 := by
    rw [above_sFrom N _ (by omega)]; exact cnt_eq_zero.mpr (fun _ _ => rfl)
  refine ⟨(checked_ok_iff N _).mpr ?_, fun i hi hr => .inl (hrec i hi hr), rfl,
    fun _ _ hr => absurd hr (by simp [IdSpace.restored, sEmpty])⟩
  simp only [IdSpace.restored, IdSpace.bound, hab]; omega

theorem inv_new (N : Nat) (hN : 1 ≤ N) : Inv N IdSpace.new := by
  have h0 : cnt N sEmpty = 0 := cnt_eq_zero.mpr (fun _ _ => rfl)
  have := inv_restored N 0 sEmpty hN (by unfold sLen; omega) (fun _ _ h => by cases h)
  have e : IdSpace.restored N 0 sEmpty = IdSpace.new := by
      simp [IdSpace.restored, IdSpace.new, sLen, h0]
  rwa [e] at this

/-- Whatever `taken` holds at or above `eb`, if it avoids `u64::MAX` the count fits. -/
theorem fits_of_noMax (N : Nat) (sp : IdSpace) (hN : 1 ≤ N) (heb : sp.eb < N)
    (hm : sp.taken (umax N) = false) : sp.eb + above N sp.taken sp.eb < N := by
  rw [above_sFrom N _ (by omega)]
  have := cnt_mono (s := sFrom sp.taken sp.eb) (t := ival sp.eb (N - 1)) N (by
    intro i hi h; simp [sFrom] at h; simp [ival, h.2]
    have : i ≠ N - 1 := fun e => by subst e; simp [umax] at hm; simp [hm] at h
    omega)
  have hiv : cnt N (ival sp.eb (N - 1)) = N - 1 - sp.eb := cnt_interval _ _ _ (by omega)
  unfold sLen; omega

/-! ### `create` -/

/-- **A create its refusals accept always passes `checked`** in a reachable
state, and leaves a reachable state. -/
theorem create_succeeds (N L : Nat) (sp : IdSpace) (nodes : IdSet) (hN : 1 ≤ N) (hL : L < N)
    (h : Inv N sp) (hok : CreateOk N L sp nodes) :
    sp.create N L nodes = (sp.created N nodes, .ok ()) ∧ Inv N (sp.created N nodes) := by
  have hcr := (create_spec N L sp nodes).1 hok
  have hmx : nodes (umax N) = false := by
    cases hn : nodes (umax N) with
    | false => rfl
    | true => have := hok.1 (umax N) (by unfold umax; omega) hn; unfold umax at this; omega
  have hb : sp.eb ≤ N := Nat.le_of_lt (eb_lt h)
  have ⟨_, hc⟩ := (checked_ok_iff N sp).mp h.checked
  have hnoMax : (sp.created N nodes).taken (umax N) = false := by
    simp [IdSpace.created, sUnion, h.noMax, hmx]
  have hchk : (sp.created N nodes).checked N = .ok () := by
    apply (checked_ok_iff N _).mpr
    refine ⟨fits_of_noMax N _ hN (by have := eb_lt h; exact this) hnoMax, ?_⟩
    simp only [IdSpace.created, IdSpace.bound]
    rw [above_sFrom N _ hb] at *
    unfold IdSpace.bound at hc
    -- |(taken ∪ nodes) ≥ eb| = |taken ≥ eb| + |nodes ∖ recycled|
    have s1 := cnt_split N (sFrom (sUnion sp.taken nodes) sp.eb) sp.taken
    have e1 : cnt N (fun i => sFrom (sUnion sp.taken nodes) sp.eb i && sp.taken i)
        = cnt N (sFrom sp.taken sp.eb) :=
      cnt_congr N (fun i _ => by simp [sFrom, sUnion]; cases sp.taken i <;> simp)
    have e2 : cnt N (fun i => sFrom (sUnion sp.taken nodes) sp.eb i && !sp.taken i)
        = cnt N (fun i => nodes i && !sp.recycled i) := cnt_congr N (fun i hi => by
      simp only [sFrom, sUnion]
      have f1 := hok.2 i hi; have f2 := h.binOk i hi
      cases hn : nodes i <;> cases ht : sp.taken i <;> cases hr : sp.recycled i <;>
        simp [hn, ht, hr] at f1 f2 ⊢ <;> omega)
    have s2 := cnt_split N nodes sp.recycled
    have s3 := cnt_split N sp.recycled nodes
    have e3 : cnt N (fun i => nodes i && sp.recycled i) = cnt N (fun i => sp.recycled i && nodes i) :=
      cnt_congr N (fun i _ => Bool.and_comm _ _)
    unfold sLen sDiff at *
    omega
  rw [hchk] at hcr
  refine ⟨hcr, hchk, fun i hi hr => ?_, hnoMax, fun i hi hr =>
    (h.relOk i hi hr).elim .inl (fun a => .inr (by simp [IdSpace.created, sUnion, a]))⟩
  simp [IdSpace.created, sDiff] at hr
  rcases h.binOk i hi hr.1 with a | a
  · exact .inl a
  · exact .inr (by simp [IdSpace.created, sUnion, a])

/-- …and a create that *returns* `Ok` (whatever the reason) leaves a reachable state. -/
theorem create_preserves (N L : Nat) (sp : IdSpace) (nodes : IdSet) (hN : 1 ≤ N) (hL : L < N)
    (h : Inv N sp) (hr : (sp.create N L nodes).2 = .ok ()) : Inv N (sp.create N L nodes).1 := by
  by_cases hok : CreateOk N L sp nodes
  · rw [(create_succeeds N L sp nodes hN hL h hok).1]; exact (create_succeeds N L sp nodes hN hL h hok).2
  · obtain ⟨_, e, he, _⟩ := (create_spec N L sp nodes).2 hok
    rw [hr] at he; cases he

/-! ### `cancel` -/

/-- **Cancelling an id the batch has not taken passes `checked` exactly when
it is free iff it is below the boundary** — a reclaimed reservation (free, below)
moves neither half, a fresh one (not free, at or above) moves both. -/
theorem cancel_succeeds (N : Nat) (sp : IdSpace) (id : Nat) (hN : 1 ≤ N) (h : Inv N sp)
    (hid : id < N) (hmax : id ≠ umax N) (ht : sp.taken id = false)
    (hc : sp.recycled id = true ↔ id < sp.eb) :
    (sp.cancel N id).2 = .ok () ∧ Inv N (sp.cancel N id).1 := by
  rw [cancel_fresh N sp id ht]
  dsimp only
  have hb : sp.eb ≤ N := Nat.le_of_lt (eb_lt h)
  have ⟨_, hcnt⟩ := (checked_ok_iff N sp).mp h.checked
  have hnoMax : sIns sp.taken id (umax N) = false := by
    simp [sIns, h.noMax]; omega
  have hchk : IdSpace.checked N { sp with taken := sIns sp.taken id, recycled := sIns sp.recycled id }
      = .ok () := by
    apply (checked_ok_iff N _).mpr
    refine ⟨fits_of_noMax N { sp with taken := sIns sp.taken id, recycled := sIns sp.recycled id }
      hN (show sp.eb < N from eb_lt h) hnoMax, ?_⟩
    simp only [IdSpace.bound]
    rw [above_sFrom N _ hb]; rw [above_sFrom N _ hb] at hcnt
    unfold IdSpace.bound at hcnt
    unfold sLen at *
    rw [cnt_ins N _ id hid]
    by_cases hge : sp.eb ≤ id
    · have e : sFrom (sIns sp.taken id) sp.eb = sIns (sFrom sp.taken sp.eb) id := by
        funext i; simp [sFrom, sIns]; by_cases e : i = id
        · subst e; simp [hge]
        · simp [e]
      rw [e, cnt_ins N _ id hid]
      have : sp.recycled id = false := by
        cases hr : sp.recycled id
        · rfl
        · have := hc.mp hr; omega
      simp [sFrom, ht, this]; omega
    · have e : sFrom (sIns sp.taken id) sp.eb = sFrom sp.taken sp.eb := by
        funext i; simp only [sFrom, sIns]; by_cases e : i = id
        · subst e; simp [hge]
        · simp [e]
      rw [e]
      have : sp.recycled id = true := hc.mpr (by omega)
      simp [this]; omega
  refine ⟨hchk, hchk, fun i hi hr => ?_, hnoMax, fun i hi hr =>
    (h.relOk i hi hr).elim .inl (fun a => .inr (by simp [sIns, a]))⟩
  simp only [sIns] at hr ⊢
  by_cases e : i = id
  · subst e; exact .inr (by simp)
  · simp [e] at hr
    rcases h.binOk i hi hr with a | a
    · exact .inl a
    · exact .inr (by simp [a])

/-- **Every id `reserve` hands out can be cancelled.** -/
theorem reserve_then_cancel (N : Nat) (sp : IdSpace) (count : Nat) (issued : IdSet) (ids : List Nat)
    (hN : 1 ≤ N) (h : Inv N sp) (hr : sp.reserve N true count issued = .ok ids)
    (hdense : ∀ i, (sp.taken i = true ∨ issued i = true) → sp.eb ≤ i → i < sp.start N issued)
    (x : Nat) (hx : x ∈ ids) (hxN : x < N) (hxm : x ≠ umax N) :
    (sp.cancel N x).2 = .ok () ∧ Inv N (sp.cancel N x).1 := by
  obtain ⟨_, hdis, hpos⟩ := reserve_fresh N sp count issued ids hr hdense h.binOk
  have ⟨ht, _⟩ := hdis x hx
  refine cancel_succeeds N sp x hN h hxN hxm ht ⟨fun hrx => ?_, fun hlt => ?_⟩
  · rcases hpos x hx with ⟨_, a⟩ | a
    · exact a
    · rcases h.binOk x hxN hrx with b | b
      · exact b
      · rw [ht] at b; cases b
  · rcases hpos x hx with ⟨a, _⟩ | a
    · exact a
    · unfold IdSpace.start at a; omega

end IdSpaceModel
