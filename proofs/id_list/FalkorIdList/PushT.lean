import FalkorIdList.CostProps
/-!
# `push` with the real collapse decision

`IdListPush.push c` takes the decision `c` as a parameter and its theorems hold
for every `c`. Here the tally is threaded through `push` exactly where the Rust
touches `self.run` (`restart` at `:875`, `:989`, `:1023`, `:1105`; `absorb` at
`:1015`), and `c` *is* `prefers_bitmap()` of the tally after the absorb. So every
`IdListPush` theorem (`push_ok`, `pushAll_ok`, `fromIter_iter`) applies to the
list the Rust actually builds (`pushT_st`, `pushAllT_ok`).
-/
namespace IdListCost
open IdListPush IdListWire

/-- `claim_direction` (`:857`) on the tally: it restarts the run on a flip. -/
def claimT (st : St) (d : Bool) (t : Tally) : Tally :=
  match st.desc with
  | none => t
  | some d' => if d' = d then t else restart

/-- The pre-extension `claim_direction` calls (`:896-908`) on the tally. -/
def claimStepT (st : St) (id : Nat) (t : Tally) : Tally :=
  match st.segs.getLast? with
  | some (.range b 1) => if b + 1 = id then claimT st false t else t
  | some (.rdesc b 1) => if id + 1 = b then claimT st true t else t
  | _ => t

/-- The tally after `:1012-1016`: the superseded last segment is absorbed iff it
is a `Range`/`RangeDescending`. -/
def absorbedT (R : Roaring) (st : St) (t : Tally) : Tally :=
  match st.segs.getLast? with
  | some s => if Seg.RL s then absorb R t s else t
  | none => t

/-- The decision `maybe_collapse_run` (`:1029`) takes: `prefers_bitmap()`. -/
def decision (R : Roaring) (st0 : St) (t0 : Tally) (id : Nat) : Bool :=
  let st := claimStep { st0 with len := st0.len + 1 } id
  prefersBitmap (absorbedT R st (claimStepT { st0 with len := st0.len + 1 } id t0))

/-- The tally `push` leaves behind, branch by branch as in `IdListPush.push`. -/
def tallyOut (R : Roaring) (st0 : St) (t0 : Tally) (id : Nat) : Tally :=
  let st := claimStep { st0 with len := st0.len + 1 } id
  let t := claimStepT { st0 with len := st0.len + 1 } id t0
  match st.segs.getLast? with
  | none => restart                                           -- :1023
  | some last =>
    match extend? last id with
    | some _ => t                                             -- hot path, no tally change
    | none =>
      match last with
      | .range b 1 =>
        if id + 1 = b then claimT { st with segs := st.segs.dropLast ++ [.rdesc b 2] } true t
        else if b = id then restart                           -- :989
        else tail st t last
      | _ => tail st t last
where
  tail (st : St) (t : Tally) (last : Seg) : Tally :=
    let continues : Bool := match st.desc with
      | some false => decide (last.max < id)
      | some true => decide (id < last.min)
      | none => decide (last.max < id ∨ id < last.min)
    if continues then
      let tA := absorbedT R st t
      if prefersBitmap tA then restart else tA                -- :1105 / no collapse
    else restart                                              -- :1023

/-- `IdList::push` with its tally. -/
def pushT (R : Roaring) (st0 : St) (t0 : Tally) (id : Nat) : Option (St × Tally) :=
  (push (decision R st0 t0 id) st0 id).map fun st => (st, tallyOut R st0 t0 id)

/-- The list `pushT` builds is `IdListPush.push` with `c := prefers_bitmap()`. -/
theorem pushT_st (R : Roaring) (st0 : St) (t0 : Tally) (id : Nat) :
    (pushT R st0 t0 id).map Prod.fst = push (decision R st0 t0 id) st0 id := by
  simp only [pushT, Option.map_map]; cases push (decision R st0 t0 id) st0 id <;> rfl

/-- `FromIterator`: push each id with the computed decision. -/
def pushAllT (R : Roaring) : St → Tally → List Nat → Option (St × Tally)
  | st, t, [] => some (st, t)
  | st, t, x :: xs => do
    let (st', t') ← pushT R st t x
    pushAllT R st' t' xs

/-- **Every push sequence is kept, in order, with the real collapse arithmetic**
(and the `assert!` in `maybe_collapse_run` never fires). -/
theorem pushAllT_ok (R : Roaring) : ∀ (st : St) (t : Tally) (xs : List Nat), Inv st → (∀ x ∈ xs, x < W) →
    ∃ st' t', pushAllT R st t xs = some (st', t') ∧ Inv st' ∧ flat st'.segs = flat st.segs ++ xs ∧
      st'.len = st.len + xs.length
  | st, t, [], hi, _ => ⟨st, t, rfl, hi, by simp, rfl⟩
  | st, t, x :: xs, hi, hx => by
    obtain ⟨st1, h1, hi1, hf1, hl1⟩ := push_ok (decision R st t x) st hi x (hx x (by simp))
    have hp : pushT R st t x = some (st1, tallyOut R st t x) := by simp [pushT, h1]
    obtain ⟨st2, t2, h2, hi2, hf2, hl2⟩ :=
      pushAllT_ok R st1 (tallyOut R st t x) xs hi1 (fun y hy => hx y (by simp [hy]))
    refine ⟨st2, t2, ?_, hi2, ?_, ?_⟩
    · simp only [pushAllT, hp]; exact h2
    · rw [hf2, hf1]; simp
    · rw [hl2, hl1]; simp; omega

/-- `IdList::from_iter` with the real decision: ids come back as pushed. -/
theorem fromIterT (R : Roaring) (xs : List Nat) (hx : ∀ x ∈ xs, x < W) :
    ∃ st t, pushAllT R St.empty restart xs = some (st, t) ∧ Inv st ∧ flat st.segs = xs ∧
      st.len = xs.length := by
  obtain ⟨st, t, h1, h2, h3, h4⟩ := pushAllT_ok R St.empty restart xs inv_empty hx
  exact ⟨st, t, h1, h2, by simpa [St.empty, flat] using h3, by simpa [St.empty] using h4⟩

/-- When `push` collapses, it is because the closed segments' range cost is past
the floor and strictly above the bitmap's (`prefersBitmap_iff`); the tally is
then restarted after the collapsed segment. -/
theorem collapse_when (R : Roaring) (st0 : St) (t0 : Tally) (id : Nat)
    (h : decision R st0 t0 id = true) :
    let t := absorbedT R (claimStep { st0 with len := st0.len + 1 } id)
      (claimStepT { st0 with len := st0.len + 1 } id t0)
    32 ≤ t.rangeBytes ∧ 5 + bitmapBytes t < t.rangeBytes :=
  (prefersBitmap_iff _).mp h

-- Sanity: 40 gapped singletons ascending collapse into one bitmap (as
-- `an_ascending_gapped_list_collapses_to_a_bitmap`), with a size model that
-- matches the tally exactly.
def Rt : Roaring := ⟨fun _ => [], fun _ => 0, fun _ => none⟩
#guard ((pushAllT Rt St.empty restart ((List.range 40).map (· * 2))).map
  (fun p => p.1.segs.length)) == some 1
-- Two consecutive runs stay two ranges (`two_runs_stay_two_ranges_because_a_bitmap_is_dearer`).
#guard ((pushAllT Rt St.empty restart (List.range' 0 100 ++ List.range' 200 100)).map
  (fun p => p.1.segs.length)) == some 2

end IdListCost
