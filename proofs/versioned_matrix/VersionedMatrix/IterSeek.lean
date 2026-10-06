import VersionedMatrix.DeltaBook
/-
# `versioned_matrix::Iter` construction and `seek` (`versioned_matrix.rs:1348-1450`)

A layer's `matrix::Iter` is `LIt`: the layer it is bound to plus the entries
it has yet to yield in the current row range (`GxB_rowIterator_seekRow` +
the `max_row` cut). `detached` (no `GxB_Iterator` attached) yields nothing
until seeked. The abstract state `abs` (buffered element ++ what the layer
iterator has left) is exactly the `VMIter.St` the merge theorem
`VMIter.drain_eq` consumes.

| here | there |
| --- | --- |
| `fromLayersDetaching` | `from_layers_detaching` (:1393) |
| `fromLayers`          | `from_layers` (:1379) |
| `newIt`               | `Iter::new` (:1352), detaching layers `may_hold_rows` rules out |
| `seek`                | `Iter::seek` (:1430) |

Results: `fromLayers_abs` (each stream is the layer restricted to the
range), `detach_harmless` (detaching a layer with nothing in range changes
no stream), `newIt_abs` (with sound row filters, `Iter::new` = `from_layers`),
`seek_abs` (a seek — even after a detached start — is a fresh iterator on
the new range: the #2430 regression `seek_after_a_skipped_layer_still_reads_the_delta`).
-/
namespace VMIterSeek
open VMMat VMRowFilter VMDelta

def inR (lo hi : Nat) (p : Pair) : Bool := decide (lo ≤ p.1) && decide (p.1 ≤ hi)

structure LIt where
  ents : List Pair
  rest : List Pair

def LIt.new (l : List Pair) (lo hi : Nat) : LIt := ⟨l, l.filter (inR lo hi)⟩
def LIt.detached (l : List Pair) : LIt := ⟨l, []⟩
def LIt.seek (it : LIt) (lo hi : Nat) : LIt := LIt.new it.ents lo hi
def LIt.next (it : LIt) : Option Pair × LIt :=
  match it.rest with
  | [] => (none, it)
  | x :: xs => (some x, { it with rest := xs })

structure It where
  mit : LIt
  mNext : Option Pair
  dpit : Option LIt
  dpNext : Option Pair
  dmit : Option LIt
  dmNext : Option Pair

def optIt (l : List Pair) (det : Bool) (lo hi : Nat) : Option LIt :=
  if l = [] then none else if det then some (LIt.detached l) else some (LIt.new l lo hi)

/-- `let dm_next = dmit.as_mut().and_then(Iterator::next)`. -/
def startDm : Option LIt → Option Pair × Option LIt
  | none => (none, none)
  | some it => (it.next.1, some it.next.2)

def fromLayersDetaching (m dp : List Pair) (ddp : Bool) (dm : List Pair) (ddm : Bool) (lo hi : Nat) : It :=
  let r := startDm (optIt dm ddm lo hi)
  ⟨LIt.new m lo hi, none, optIt dp ddp lo hi, none, r.2, r.1⟩

def fromLayers (m dp dm : List Pair) (lo hi : Nat) : It := fromLayersDetaching m dp false dm false lo hi

/-- `Iter::new` (:1352). -/
def newIt (m : Mat Unit) (dp dm : Delta Unit) (lo hi : Nat) : It :=
  fromLayersDetaching (keys m) (keys dp.layer) (!mayHoldRows dp lo hi) (keys dm.layer)
    (!mayHoldRows dm lo hi) lo hi

/-- `Iter::seek` (:1430). -/
def seek (s : It) (lo hi : Nat) : It :=
  let r := match s.dmit with
    | none => (s.dmNext, none)
    | some it => startDm (some (it.seek lo hi))
  ⟨s.mit.seek lo hi, none, s.dpit.map (·.seek lo hi), none, r.2, r.1⟩

def restOf : Option LIt → List Pair
  | none => []
  | some it => it.rest

/-- The observable state: `(M, D, P)` of `VMIter.St`. -/
def abs (s : It) : List Pair × List Pair × List Pair :=
  (s.mNext.toList ++ s.mit.rest, s.dmNext.toList ++ restOf s.dmit, s.dpNext.toList ++ restOf s.dpit)

theorem optIt_rest (l : List Pair) (lo hi : Nat) : restOf (optIt l false lo hi) = l.filter (inR lo hi) := by
  unfold optIt; split
  · next h => subst h; rfl
  · rfl

theorem next_abs (o : Option LIt) : (startDm o).1.toList ++ restOf (startDm o).2 = restOf o := by
  cases o with
  | none => rfl
  | some it =>
    simp only [startDm, LIt.next]; cases h : it.rest <;> simp [restOf, h]

theorem startDm_ents (it : LIt) : ∃ it', (startDm (some it)).2 = some it' ∧ it'.ents = it.ents := by
  simp only [startDm, LIt.next]; cases it.rest <;> exact ⟨_, rfl, rfl⟩

theorem fromLayers_abs (m dp dm : List Pair) (lo hi : Nat) :
    abs (fromLayers m dp dm lo hi) = (m.filter (inR lo hi), dm.filter (inR lo hi), dp.filter (inR lo hi)) := by
  simp only [abs, fromLayers, fromLayersDetaching, LIt.new]
  rw [next_abs, optIt_rest, optIt_rest]; rfl

/-- Nothing of the layer lies in the range. -/
def Quiet (l : List Pair) (lo hi : Nat) : Prop := ∀ p ∈ l, inR lo hi p = false

theorem quiet_filter {l : List Pair} {lo hi : Nat} (h : Quiet l lo hi) : l.filter (inR lo hi) = [] :=
  List.filter_eq_nil_iff.2 (fun p hp => by simp [h p hp])

theorem optIt_rest_det (l : List Pair) (det : Bool) (lo hi : Nat) (h : det = true → Quiet l lo hi) :
    restOf (optIt l det lo hi) = l.filter (inR lo hi) := by
  cases det
  · exact optIt_rest l lo hi
  · unfold optIt; split
    · next he => subst he; rfl
    · simp [restOf, LIt.detached, quiet_filter (h rfl)]

theorem detach_harmless (m dp dm : List Pair) (a b : Bool) (lo hi : Nat)
    (ha : a = true → Quiet dp lo hi) (hb : b = true → Quiet dm lo hi) :
    abs (fromLayersDetaching m dp a dm b lo hi) = abs (fromLayers m dp dm lo hi) := by
  rw [fromLayers_abs]
  simp only [abs, fromLayersDetaching, LIt.new]
  rw [next_abs, optIt_rest_det _ _ _ _ ha, optIt_rest_det _ _ _ _ hb]; rfl

theorem quiet_of_mayHold {d : Delta Unit} (h : RowsOk d) {lo hi : Nat}
    (hm : (!mayHoldRows d lo hi) = true) : Quiet (keys d.layer) lo hi := by
  intro p hp
  obtain ⟨e, he, rfl⟩ := List.mem_map.1 hp
  cases hr : inR lo hi e.1
  · rfl
  · simp [inR] at hr
    have := mayHoldRows_sound h (v := e.2) (by simpa using he) hr.1 hr.2
    simp [this] at hm

/-- `Iter::new` streams exactly what an undetached iterator would. -/
theorem newIt_abs (m : Mat Unit) (dp dm : Delta Unit) (h1 : RowsOk dp) (h2 : RowsOk dm) (lo hi : Nat) :
    abs (newIt m dp dm lo hi) = abs (fromLayers (keys m) (keys dp.layer) (keys dm.layer) lo hi) :=
  detach_harmless _ _ _ _ _ _ _ (quiet_of_mayHold h1) (quiet_of_mayHold h2)

/-- Which layers an iterator is bound to (`None` only for an empty layer). -/
def Bound (s : It) (m dp dm : List Pair) : Prop :=
  s.mit.ents = m ∧ (s.dpit = none ∧ dp = [] ∨ ∃ it, s.dpit = some it ∧ it.ents = dp) ∧
    (s.dmit = none ∧ s.dmNext = none ∧ dm = [] ∨ ∃ it, s.dmit = some it ∧ it.ents = dm)

theorem fromLayersDetaching_bound (m dp dm : List Pair) (a b : Bool) (lo hi : Nat) :
    Bound (fromLayersDetaching m dp a dm b lo hi) m dp dm := by
  refine ⟨rfl, ?_, ?_⟩
  · simp only [fromLayersDetaching, optIt]
    by_cases h : dp = []
    · exact Or.inl ⟨by simp [h], h⟩
    · rw [if_neg h]; split <;> exact Or.inr ⟨_, rfl, rfl⟩
  · simp only [fromLayersDetaching, optIt]
    by_cases h : dm = []
    · exact Or.inl ⟨by simp [h, startDm], by simp [h, startDm], h⟩
    · rw [if_neg h]; split
      · obtain ⟨it, h1, h2⟩ := startDm_ents (LIt.detached dm); exact Or.inr ⟨it, h1, h2⟩
      · obtain ⟨it, h1, h2⟩ := startDm_ents (LIt.new dm lo hi); exact Or.inr ⟨it, h1, h2⟩

/-- `seek` is a fresh iterator on the new range, whatever state (or detached
start) it is called from. -/
theorem seek_abs {s : It} {m dp dm : List Pair} (hb : Bound s m dp dm) (lo hi : Nat) :
    abs (seek s lo hi) = abs (fromLayers m dp dm lo hi) ∧ Bound (seek s lo hi) m dp dm := by
  obtain ⟨hm, hdp, hdm⟩ := hb
  rw [fromLayers_abs]
  refine ⟨?_, ?_⟩
  · simp only [abs, seek, Option.toList, List.nil_append, LIt.seek, LIt.new, hm]
    refine Prod.ext rfl (Prod.ext ?_ ?_)
    · rcases hdm with ⟨h1, h2, rfl⟩ | ⟨it, h1, rfl⟩
      · simp [h1, h2, restOf]
      · simp only [h1]
        exact next_abs (some _)
    · rcases hdp with ⟨h1, rfl⟩ | ⟨it, h1, rfl⟩
      · simp [h1, restOf]
      · simp [h1, restOf]
  · refine ⟨by simp [seek, LIt.seek, LIt.new, hm], ?_, ?_⟩
    · rcases hdp with ⟨h1, h2⟩ | ⟨it, h1, rfl⟩
      · exact Or.inl ⟨by simp [seek, h1], h2⟩
      · exact Or.inr ⟨it.seek lo hi, by simp [seek, h1], rfl⟩
    · rcases hdm with ⟨h1, h2, h3⟩ | ⟨it, h1, rfl⟩
      · exact Or.inl ⟨by simp [seek, h1], by simp [seek, h1, h2], h3⟩
      · refine Or.inr ?_
        obtain ⟨it', h4, h5⟩ := startDm_ents (it.seek lo hi)
        exact ⟨it', by simp only [seek, h1]; exact h4, h5⟩

end VMIterSeek
