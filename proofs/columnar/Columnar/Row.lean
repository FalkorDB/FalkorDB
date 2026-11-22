import Columnar.Batch

/-
# `Row` and its interplay with `BatchRow`

| here | there |
| --- | --- |
| `Row` | `Row` (`runtime/row.rs`); `bound` = the `BitSet` as a predicate (BitSet itself: proofs/runtime_ds) |
| `Row.insert` | `Row::insert_by_id` / `insert` (`row.rs`) |
| `Row.get` | `Row::get_by_id` / `get` / `RowView::value_at` (`row.rs`) |
| `Row.unbind` | `Row::unbind_by_id` / `unbind` (`row.rs`) |
| `Row.take` | `Row::take` (`row.rs`) |
| `Row.cloneWith` | `Row::clone_with` (`row.rs`) |
| `fillRow` | the loop of `Row::merge` (`BatchRow::to_owned_row` no longer shares it: it pushes one slot per column, `pushSlots`) |
| `Row.merge` | `Row::merge` (`row.rs`) |
| `viewAt` | `BatchRow::value_at` (`batch.rs`) |
| `toOwned`, `pushSlots`, `Row.pushSlot` | `BatchRow::to_owned_row` (`batch.rs`), `Row::push_slot` (`row.rs`) |
-/

namespace Columnar
open FalkorCore (V)

variable {F : Type}

structure Row (F : Type) where
  values : List (V F)
  bound : Nat → Bool
  origin : Nat

namespace Row

def empty : Row F := ⟨[], fun _ => false, 0⟩

/-- `insert_by_id`: grow with `Null` to `id + 1`, write, set the bound bit. -/
def insert (r : Row F) (id : Nat) (v : V F) : Row F :=
  let vals := if r.values.length ≤ id then r.values ++ List.replicate (id + 1 - r.values.length) .null
    else r.values
  { r with values := vals.set id v, bound := fun j => if j = id then true else r.bound j }

def get (r : Row F) (id : Nat) : Option (V F) := r.values[id]?

def unbind (r : Row F) (id : Nat) : Row F := { r with bound := fun j => if j = id then false else r.bound j }

/-- `Row::take`: the value, replaced by `Null`; `None` if absent or already `Null`. -/
def take (r : Row F) (id : Nat) : Option (V F) × Row F :=
  match r.values[id]? with
  | none => (none, r)
  | some v => (if v.isNull then none else some v, { r with values := r.values.set id .null })

def cloneWith (r : Row F) (id : Nat) (v : V F) : Row F := r.insert id v

/-- The `resize_with(idx + 1, Null)` padding step of `insert_by_id`. -/
theorem pad_getElem? (l : List (V F)) (id j : Nat) :
    (if l.length ≤ id then l ++ List.replicate (id + 1 - l.length) V.null else l)[j]? =
      if j < l.length then l[j]? else if j ≤ id then some .null else none := by
  by_cases h : l.length ≤ id
  · rw [ite_eq_left h]
    by_cases hj : j < l.length
    · rw [ite_eq_left hj, List.getElem?_append_left hj]
    · rw [ite_eq_right hj, List.getElem?_append_right (by omega), List.getElem?_replicate]
      by_cases hj2 : j ≤ id
      · rw [ite_eq_left hj2, ite_eq_left (by omega)]
      · rw [ite_eq_right hj2, ite_eq_right (by omega)]
  · rw [ite_eq_right h]
    by_cases hj : j < l.length
    · rw [ite_eq_left hj]
    · rw [ite_eq_right hj, ite_eq_right (by omega)]
      exact List.getElem?_eq_none (by omega)

theorem pad_length (l : List (V F)) (id : Nat) :
    (if l.length ≤ id then l ++ List.replicate (id + 1 - l.length) V.null else l).length =
      max l.length (id + 1) := by
  by_cases h : l.length ≤ id
  · rw [ite_eq_left h]; simp; omega
  · rw [ite_eq_right h]; omega

theorem insert_length (r : Row F) (id : Nat) (v : V F) :
    (r.insert id v).values.length = max r.values.length (id + 1) := by
  simp only [insert, List.length_set, pad_length]

theorem insert_get (r : Row F) (id : Nat) (v : V F) (j : Nat) :
    (r.insert id v).get j =
      if j = id then some v else if j < r.values.length then r.get j
      else if j < id then some .null else none := by
  simp only [insert, get]
  by_cases h : j = id
  · subst h
    rw [ite_eq_left rfl, List.getElem?_set_self (by rw [pad_length]; omega)]
  · rw [ite_eq_right h, List.getElem?_set_ne (Ne.symm h), pad_getElem?]
    by_cases hj : j < r.values.length
    · rw [ite_eq_left hj, ite_eq_left hj]
    · rw [ite_eq_right hj, ite_eq_right hj]
      by_cases h2 : j < id
      · rw [ite_eq_left (by omega), ite_eq_left h2]
      · rw [ite_eq_right (by omega), ite_eq_right h2]

theorem insert_bound (r : Row F) (id : Nat) (v : V F) (k : Nat) :
    (r.insert id v).bound k = if k = id then true else r.bound k := rfl

theorem unbind_bound (r : Row F) (id : Nat) (k : Nat) :
    (r.unbind id).bound k = if k = id then false else r.bound k := rfl

theorem unbind_get (r : Row F) (id : Nat) (k : Nat) : (r.unbind id).get k = r.get k := rfl

theorem unbind_values (r : Row F) (id : Nat) : (r.unbind id).values = r.values := rfl

/-- Reading back an inserted slot, and its bound bit. -/
theorem get_insert_self (r : Row F) (id : Nat) (v : V F) :
    (r.insert id v).get id = some v ∧ (r.insert id v).bound id = true := by
  rw [insert_get, insert_bound]; simp

theorem get_cloneWith (r : Row F) (id : Nat) (v : V F) (j : Nat) (hj : j ≠ id) (hl : j < r.values.length) :
    (r.cloneWith id v).get j = r.get j := by
  simp [cloneWith, insert_get, hj, hl]

/-- `take` returns the value iff it is non-null and leaves `Null` behind. -/
theorem take_spec (r : Row F) (id : Nat) (v : V F) (h : r.get id = some v) :
    (r.take id).1 = (if v.isNull then none else some v) ∧ (r.take id).2.get id = some .null := by
  unfold take get at *
  rw [h]
  have hl : id < r.values.length := by
    rcases Nat.lt_or_ge id r.values.length with h' | h'
    · exact h'
    · simp [List.getElem?_eq_none h'] at h
  simp [List.getElem?_set_self hl]


/-! ### The remaining `Row` accessors (`row.rs`)

`bound` is the `BitSet` as a predicate (its `test`/`set`/`clear`/`is_empty` are proved in
proofs/runtime_ds), so a bit read is `r.bound id` and `BitSet::is_empty` is "no `id` is set". -/

/-- `Row::from_raw` (row.rs): the given slots and bound set, `origin_row = 0`. `Row::new`
(row.rs) and `Row::with_capacity` (row.rs) are `fromRaw [] (fun _ => false)`: the
capacity is an allocation hint with no semantic content. -/
def fromRaw (values : List (V F)) (bound : Nat → Bool) : Row F := ⟨values, bound, 0⟩

theorem fromRaw_fields (values : List (V F)) (bound : Nat → Bool) :
    (fromRaw values bound).values = values ∧ (fromRaw values bound).bound = bound ∧
      (fromRaw values bound).origin = 0 := ⟨rfl, rfl, rfl⟩

/-- `Row::new` / `Row::with_capacity` build `Row.empty`: no slots, nothing bound, origin 0. -/
theorem new_spec : (empty : Row F).values = [] ∧ (∀ k, (empty : Row F).bound k = false) ∧
    (empty : Row F).origin = 0 := ⟨rfl, fun _ => rfl, rfl⟩

theorem fromRaw_nil : fromRaw ([] : List (V F)) (fun _ => false) = empty := rfl

/-- `Row::len` (row.rs). -/
def len (r : Row F) : Nat := r.values.length

/-- `Row::is_empty` (row.rs). -/
def isEmpty (r : Row F) : Bool := r.values.isEmpty

theorem len_isEmpty (r : Row F) : r.isEmpty = true ↔ r.len = 0 := by
  simp [isEmpty, len, List.isEmpty_iff_length_eq_zero]

theorem len_empty : (empty : Row F).len = 0 := rfl

/-- `Row::get_by_id` / `Row::get` (row.rs) read `values.get(id)`: `Row.get`, `none`
exactly when `id` is out of range. `RowView::value_at` for `Row` (row.rs) is the same read,
cloned. -/
theorem get_eq_none_iff (r : Row F) (id : Nat) : r.get id = none ↔ r.len ≤ id := by
  simp [get, len, List.getElem?_eq_none_iff]

theorem get_isSome_iff (r : Row F) (id : Nat) : (r.get id).isSome = true ↔ id < r.len := by
  constructor
  · intro h
    cases hv : r.values[id]? with
    | none => simp [get, hv] at h
    | some v => exact (List.getElem?_eq_some_iff.mp hv).1
  · intro h
    simp [get, len, List.getElem?_eq_getElem h]

/-- `Row::is_bound_by_id` (row.rs): the bound bit. -/
def isBoundById (r : Row F) (id : Nat) : Bool := r.bound id

theorem isBoundById_insert (r : Row F) (id : Nat) (v : V F) (k : Nat) :
    (r.insert id v).isBoundById k = (k == id || r.isBoundById k) := by
  simp only [isBoundById, insert]
  by_cases h : k = id <;> simp [h]

theorem isBoundById_unbind (r : Row F) (id : Nat) (k : Nat) :
    (r.unbind id).isBoundById k = (k != id && r.isBoundById k) := by
  simp only [isBoundById, unbind]
  by_cases h : k = id <;> simp [h]

/-- `Row::has_bindings` (row.rs): `!bound.is_empty()`, i.e. some bit is set. -/
def hasBindings (r : Row F) : Prop := ∃ k, r.bound k = true

theorem hasBindings_empty : ¬ (empty : Row F).hasBindings := by
  rintro ⟨k, hk⟩; exact absurd hk (by simp [empty])

theorem hasBindings_insert (r : Row F) (id : Nat) (v : V F) : (r.insert id v).hasBindings :=
  ⟨id, by simp [insert]⟩

theorem hasBindings_of_bound (r : Row F) (id : Nat) (h : r.bound id = true) : r.hasBindings :=
  ⟨id, h⟩

/-- `RowView::to_owned_row` for `Row` (row.rs) is `self.clone()`: the same row. -/
def toOwnedRow (r : Row F) : Row F := r

theorem toOwnedRow_get (r : Row F) (id : Nat) : r.toOwnedRow.get id = r.get id := rfl

theorem filterMap_range_congr {α : Type} (f g : Nat → Option α) (n : Nat)
    (h : ∀ k, k < n → f k = g k) : (List.range n).filterMap f = (List.range n).filterMap g := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [List.range_succ, List.filterMap_append]
    rw [ih (fun k hk => h k (by omega))]
    have := h n (by omega)
    simp [List.filterMap, this]

/-- The `(key, value)` pairs `impl Hash for Row` (row.rs) feeds the hasher, in order: every
slot except an unbound `Null` (padding / a deliberately unbound slot). -/
def hashItems (r : Row F) : List (Nat × V F) :=
  (List.range r.values.length).filterMap fun k =>
    match r.values[k]? with
    | none => none
    | some v => if v.isNull && !r.bound k then none else some (k, v)

theorem mem_hashItems (r : Row F) (k : Nat) (v : V F) :
    (k, v) ∈ r.hashItems ↔ r.values[k]? = some v ∧ ¬ (v.isNull = true ∧ r.bound k = false) := by
  simp only [hashItems, List.mem_filterMap, List.mem_range, Prod.mk.injEq]
  constructor
  · rintro ⟨a, ha, h⟩
    cases hv : r.values[a]? with
    | none => simp [hv] at h
    | some w =>
      simp only [hv] at h
      by_cases hc : (w.isNull && !r.bound a) = true
      · simp [hc] at h
      · simp only [hc] at h; simp at h
        obtain ⟨rfl, rfl⟩ := h
        refine ⟨hv, ?_⟩
        intro ⟨h1, h2⟩; apply hc; simp [h1, h2]
  · rintro ⟨hv, hn⟩
    have hk : k < r.values.length := (List.getElem?_eq_some_iff.mp hv).1
    refine ⟨k, hk, ?_⟩
    simp only [hv]
    have : ¬ ((v.isNull && !r.bound k) = true) := by
      intro h; apply hn; simp at h; simpa using h
    simp [this]

/-- An unbound `Null` slot is skipped: it contributes nothing to the hash. -/
theorem hashItems_skips_unbound_null (r : Row F) (k : Nat) (hn : r.get k = some .null)
    (hb : r.bound k = false) : (k, (V.null : V F)) ∉ r.hashItems := by
  rw [mem_hashItems]; intro ⟨_, h⟩; exact h ⟨rfl, hb⟩

/-- A bound slot (even a bound `Null`) and every non-`Null` slot is hashed. -/
theorem hashItems_bound (r : Row F) (k : Nat) (v : V F) (hv : r.get k = some v)
    (hb : r.bound k = true) : (k, v) ∈ r.hashItems := by
  rw [mem_hashItems]; exact ⟨hv, fun ⟨_, h⟩ => by simp [hb] at h⟩

/-- Appending an unbound `Null` slot does not change the hashed items (so `Row`s that differ
only by trailing unbound padding hash alike). -/
theorem hashItems_pad (r : Row F) (hb : r.bound r.values.length = false) :
    ({ r with values := r.values ++ [V.null] } : Row F).hashItems = r.hashItems := by
  unfold hashItems
  simp only [List.length_append, List.length_singleton, List.range_succ,
    List.filterMap_append]
  have hmap : (List.range r.values.length).filterMap (fun k =>
      match (r.values ++ [(V.null : V F)])[k]? with
      | none => none
      | some v => if v.isNull && !r.bound k then none else some (k, v)) =
      (List.range r.values.length).filterMap (fun k =>
      match r.values[k]? with
      | none => none
      | some v => if v.isNull && !r.bound k then none else some (k, v)) := by
    apply filterMap_range_congr
    intro k hk'
    simp [List.getElem?_append_left hk']
  rw [hmap]
  simp [List.getElem?_append_right, hb, V.isNull]

end Row

open Row

/-- The loop of `Row::merge` (`to_owned_row` uses `pushSlots`, not this): for each slot `id < n`
in order, `f id = some (v, keep)` inserts `v` and then unbinds unless `keep`. -/
def fillRow (n : Nat) (f : Nat → Option (V F × Bool)) (acc : Row F) : Row F :=
  (List.range n).foldl (fun acc id =>
    match f id with
    | none => acc
    | some (v, keep) => let a := acc.insert id v; if keep then a else a.unbind id) acc

theorem fillRow_succ (n : Nat) (f : Nat → Option (V F × Bool)) (acc : Row F) :
    fillRow (n + 1) f acc =
      match f n with
      | none => fillRow n f acc
      | some (v, keep) => let a := (fillRow n f acc).insert n v; if keep then a else a.unbind n := by
  simp [fillRow, List.range_succ, List.foldl_append]

/-- One step of `fillRow`, as reads. -/
def fillStep (R : Row F) (id : Nat) (v : V F) (keep : Bool) : Row F :=
  let a := R.insert id v; if keep then a else a.unbind id

theorem fillStep_get (R : Row F) (id : Nat) (v : V F) (keep : Bool) (k : Nat) :
    (fillStep R id v keep).get k = (R.insert id v).get k := by
  cases keep <;> rfl

theorem fillStep_length (R : Row F) (id : Nat) (v : V F) (keep : Bool) :
    (fillStep R id v keep).values.length = max R.values.length (id + 1) := by
  cases keep <;> simp [fillStep, unbind_values, insert_length]

theorem fillStep_bound (R : Row F) (id : Nat) (v : V F) (keep : Bool) (k : Nat) :
    (fillStep R id v keep).bound k = if k = id then keep else R.bound k := by
  cases keep <;> by_cases h : k = id <;> simp [fillStep, unbind_bound, insert_bound, h]

theorem fillRow_succ' (n : Nat) (f : Nat → Option (V F × Bool)) (acc : Row F) :
    fillRow (n + 1) f acc =
      match f n with
      | none => fillRow n f acc
      | some (v, keep) => fillStep (fillRow n f acc) n v keep := by
  rw [fillRow_succ]; rfl

/-- **`fillRow` spec**: a filled slot reads its value and bound bit; an unfilled
slot keeps the accumulator's (or reads `Null`/absent padding) and its bound bit. -/
theorem fillRow_spec (f : Nat → Option (V F × Bool)) (acc : Row F) :
    ∀ n j,
      ((fillRow n f acc).values.length ≥ acc.values.length) ∧
      (∀ v keep, j < n → f j = some (v, keep) →
        (fillRow n f acc).get j = some v ∧ (fillRow n f acc).bound j = keep) ∧
      ((j < n → f j = none) →
        (j < acc.values.length → (fillRow n f acc).get j = acc.get j) ∧
        ((fillRow n f acc).get j = acc.get j ∨ (fillRow n f acc).get j = some .null) ∧
        (fillRow n f acc).bound j = acc.bound j)
  | 0, j => by simp [fillRow]
  | n + 1, j => by
    obtain ⟨ihl, ihf, ihu⟩ := fillRow_spec f acc n j
    rw [fillRow_succ']
    cases hfn : f n with
    | none =>
      simp only
      refine ⟨ihl, fun v keep hj hf => ?_, fun hn => ihu (fun hj => hn (by omega))⟩
      rcases Nat.lt_or_ge j n with h | h
      · exact ihf v keep h hf
      · have : j = n := by omega
        subst this; rw [hfn] at hf; simp at hf
    | some p =>
      obtain ⟨v0, keep0⟩ := p
      simp only
      refine ⟨by rw [fillStep_length]; omega, fun v keep hj hf => ?_, fun hn => ?_⟩
      · rw [fillStep_get, fillStep_bound, insert_get]
        rcases Nat.lt_or_ge j n with h | h
        · have hjn : j ≠ n := by omega
          obtain ⟨g, bnd⟩ := ihf v keep h hf
          have hjl : j < (fillRow n f acc).values.length := by
            rcases Nat.lt_or_ge j (fillRow n f acc).values.length with h' | h'
            · exact h'
            · simp [Row.get, List.getElem?_eq_none h'] at g
          rw [ite_eq_right hjn, ite_eq_left hjl, ite_eq_right hjn]
          exact ⟨g, bnd⟩
        · have : j = n := by omega
          subst this
          rw [hfn] at hf; simp only [Option.some.injEq, Prod.mk.injEq] at hf
          obtain ⟨rfl, rfl⟩ := hf
          simp
      · have hjn : j ≠ n := by intro e; subst e; rw [hn (by omega)] at hfn; simp at hfn
        obtain ⟨u1, u2, u3⟩ := ihu (fun hj => hn (by omega))
        rw [fillStep_get, fillStep_bound, insert_get, ite_eq_right hjn, ite_eq_right hjn]
        refine ⟨fun hl => ?_, ?_, u3⟩
        · rw [ite_eq_left (by omega), u1 hl]
        · by_cases hl : j < (fillRow n f acc).values.length
          · rw [ite_eq_left hl]; exact u2
          · rw [ite_eq_right hl]
            by_cases h2 : j < n
            · rw [ite_eq_left h2]; right; rfl
            · rw [ite_eq_right h2]
              have : (fillRow n f acc).get j = none := by
                simp [Row.get, List.getElem?_eq_none (Nat.le_of_not_lt hl)]
              rw [← this]; exact u2

namespace Row

/-- `Row::merge`: copy every *bound* slot of `other`, in slot order. -/
def merge (a b : Row F) : Row F :=
  fillRow b.values.length (fun id => if b.bound id then some ((b.get id).getD .null, true) else none) a

/-- **`merge` = overlay of the bound slots**: a slot bound in `other` reads
`other`'s value and is bound; any other slot keeps `self`'s value and bound bit
(so a value-present-but-unbound slot of `other` never overwrites). -/
theorem merge_spec (a b : Row F) (j : Nat) :
    ((j < b.values.length ∧ b.bound j = true) → (a.merge b).get j = b.get j ∧ (a.merge b).bound j = true) ∧
    (¬(j < b.values.length ∧ b.bound j = true) → j < a.values.length →
      (a.merge b).get j = a.get j ∧ (a.merge b).bound j = a.bound j) := by
  obtain ⟨-, hf, hu⟩ := fillRow_spec (fun id => if b.bound id then some ((b.get id).getD .null, true) else none)
    a b.values.length j
  refine ⟨fun ⟨hl, hb⟩ => ?_, fun hn hl => ?_⟩
  · have hf' := hf ((b.get j).getD .null) true hl (by rw [ite_eq_left hb])
    refine ⟨?_, hf'.2⟩
    unfold merge
    rw [hf'.1]
    simp [Row.get, List.getElem?_eq_getElem hl]
  · have hnone : j < b.values.length →
        (if b.bound j = true then some ((b.get j).getD .null, true) else none) = none := by
      intro hj
      have : b.bound j = false := by
        cases h : b.bound j
        · rfl
        · exact absurd ⟨hj, h⟩ hn
      rw [this]; rfl
    obtain ⟨u1, -, u3⟩ := hu hnone
    exact ⟨u1 hl, u3⟩

end Row

/-! ## `BatchRow`: the borrowed view and its owned snapshot -/

/-- `BatchRow::value_at`: an in-range unbound slot reads `Some(Null)`, a slot
beyond the column space reads `None`. -/
def viewAt (b : Batch F) (r i : Nat) : Option (V F) :=
  match b.valueAt i r with
  | some v => some v
  | none => if i < b.cols.length then some .null else none

/-- `Row::push_slot` (`row.rs`): append one slot after the last, holding `v`, with the
bound bit set to `b` (a `false` leaves the bitmap alone). -/
def Row.pushSlot (r : Row F) (v : V F) (b : Bool) : Row F :=
  { r with values := r.values ++ [v],
           bound := if b then (fun j => if j = r.values.length then true else r.bound j) else r.bound }

/-- The slot `to_owned_row` pushes for column `id` (`batch.rs`): an `Unbound` column
is an unbound `Null`; any other keeps its value, bound unless the column is value-only. -/
def slotOf (b : Batch F) (r id : Nat) : V F × Bool :=
  if (b.column id).isUnbound then (.null, false)
  else (((b.column id).get? r).getD .null, !(b.vo id))

/-- The loop of `to_owned_row`: `Row::with_capacity(n)` then one `push_slot` per column. -/
def pushSlots (b : Batch F) (r : Nat) (n : Nat) : Row F :=
  (List.range n).foldl (fun acc id => acc.pushSlot (slotOf b r id).1 (slotOf b r id).2) Row.empty

theorem pushSlots_succ (b : Batch F) (r n : Nat) :
    pushSlots b r (n + 1) = (pushSlots b r n).pushSlot (slotOf b r n).1 (slotOf b r n).2 := by
  simp [pushSlots, List.range_succ, List.foldl_append]

/-- **`pushSlots` spec**: exactly `n` slots; slot `i < n` reads `slotOf`'s value and bound bit,
everything past `n` is absent and unbound. -/
theorem pushSlots_spec (b : Batch F) (r : Nat) : ∀ n,
    (pushSlots b r n).values.length = n ∧
    (∀ i, (pushSlots b r n).get i = if i < n then some (slotOf b r i).1 else none) ∧
    (∀ i, (pushSlots b r n).bound i = (decide (i < n) && (slotOf b r i).2))
  | 0 => by
    refine ⟨rfl, fun i => ?_, fun i => ?_⟩ <;> simp [pushSlots, Row.empty, Row.get]
  | n + 1 => by
    obtain ⟨hl, hg, hb⟩ := pushSlots_spec b r n
    rw [pushSlots_succ]
    refine ⟨by simp [Row.pushSlot, hl], fun i => ?_, fun i => ?_⟩
    · have hgi := hg i
      simp only [Row.get] at hgi ⊢
      simp only [Row.pushSlot, hl]
      by_cases h2 : i < n
      · rw [List.getElem?_append_left (by omega), hgi, ite_eq_left h2, ite_eq_left (by omega)]
      · by_cases hi : i = n
        · subst hi
          rw [List.getElem?_append_right (by omega), hl]; simp
        · rw [List.getElem?_append_right (by omega), List.getElem?_eq_none (by simp; omega),
            ite_eq_right (by omega)]
    · have hbi := hb i
      have key : (slotOf b r n).2 = true ∨ (slotOf b r n).2 = false := by cases (slotOf b r n).2 <;> simp
      by_cases hi : i = n
      · subst hi
        have h0 : (pushSlots b r i).bound i = false := by rw [hb i]; simp
        rcases key with hs | hs <;> simp [Row.pushSlot, hs, hl, h0]
      · rcases key with hs | hs
        · simp only [Row.pushSlot, hs, hl, ite_true, ite_eq_right hi]
          rw [hbi]
          by_cases h2 : i < n
          · simp [h2, show i < n + 1 by omega]
          · simp [h2, show ¬ i < n + 1 by omega]
        · simp only [Row.pushSlot, hs, hl, Bool.false_eq_true, ite_false]
          rw [hbi]
          by_cases h2 : i < n
          · simp [h2, show i < n + 1 by omega]
          · simp [h2, show ¬ i < n + 1 by omega]

theorem pushSlots_origin (b : Batch F) (r : Nat) : ∀ n, (pushSlots b r n).origin = 0
  | 0 => rfl
  | n + 1 => by rw [pushSlots_succ]; exact pushSlots_origin b r n

/-- `BatchRow::to_owned_row`: one slot per column (`pushSlots`), the origin tag copied. -/
def toOwned (b : Batch F) (r : Nat) : Row F :=
  let row := pushSlots b r b.cols.length
  match b.origins with
  | some o => { row with origin := o[r]?.getD 0 }
  | none => row

theorem toOwned_row (b : Batch F) (r : Nat) :
    ∃ o, toOwned b r = { pushSlots b r b.cols.length with origin := o } := by
  unfold toOwned
  cases b.origins with
  | some o => exact ⟨_, rfl⟩
  | none => exact ⟨_, rfl⟩

theorem column_isUnbound_of_ge {b : Batch F} {i : Nat} (h : b.cols.length ≤ i) :
    (b.column i).isUnbound = true := by
  simp [Batch.column, List.getElem?_eq_none h, Column.isUnbound]

/-- **The owned snapshot agrees with the view on every bound slot** (value and
bound bit = not value-only), for any row with an in-range read. -/
theorem toOwned_bound_slot {b : Batch F} (h : b.WF) {r i : Nat} (hr : r ∈ b.active)
    (hu : (b.column i).isUnbound = false) :
    (toOwned b r).get i = viewAt b r i ∧ (toOwned b r).bound i = !(b.vo i) := by
  have hi : i < b.cols.length := by
    rcases Nat.lt_or_ge i b.cols.length with h' | h'
    · exact h'
    · rw [column_isUnbound_of_ge h'] at hu; simp at hu
  obtain ⟨v, hv⟩ := Option.isSome_iff_exists.mp (Batch.valueAt_isSome h hr hu)
  have hget : (b.column i).get? r = some v := by simpa [Batch.valueAt, hu] using hv
  obtain ⟨o, ho⟩ := toOwned_row b r
  obtain ⟨-, hg, hb⟩ := pushSlots_spec b r b.cols.length
  rw [ho]
  refine ⟨?_, ?_⟩
  · simp only [Row.get] at hg ⊢
    rw [hg i, ite_eq_left hi]; simp [slotOf, hu, hget, viewAt, hv]
  · simp only [] at hb ⊢
    rw [hb i]; simp [slotOf, hu, hi]

/-- An unbound slot is never bound in the snapshot, and reads `Null` (or nothing past the
column space). -/
theorem toOwned_unbound_slot (b : Batch F) (r i : Nat) (hu : (b.column i).isUnbound = true) :
    (toOwned b r).bound i = false ∧
      ((toOwned b r).get i = none ∨ (toOwned b r).get i = some .null) := by
  obtain ⟨o, ho⟩ := toOwned_row b r
  obtain ⟨-, hg, hb⟩ := pushSlots_spec b r b.cols.length
  rw [ho]
  refine ⟨?_, ?_⟩
  · simp only [] at hb ⊢
    rw [hb i]; simp [slotOf, hu]
  · simp only [Row.get] at hg ⊢
    rw [hg i]
    by_cases hi : i < b.cols.length
    · right; simp [hi, slotOf, hu]
    · left; simp [hi]

/-- **The owned snapshot answers `value_at` exactly like the view on every slot**, trailing
`Unbound` columns included (`to_owned_row` pushes one slot per column, so an unbound
column stays in scope as `Some(Null)` and a slot past the column space is `None`). -/
theorem toOwned_get_eq_view {b : Batch F} (h : b.WF) {r : Nat} (hr : r ∈ b.active) (i : Nat) :
    (toOwned b r).get i = viewAt b r i := by
  cases hu : (b.column i).isUnbound
  · exact (toOwned_bound_slot h hr hu).1
  · obtain ⟨o, ho⟩ := toOwned_row b r
    obtain ⟨-, hg, -⟩ := pushSlots_spec b r b.cols.length
    rw [ho]
    simp only [Row.get] at hg ⊢
    rw [hg i]
    simp only [viewAt, Batch.valueAt, hu]
    by_cases hi : i < b.cols.length
    · simp [hi, slotOf, hu]
    · simp [hi]

/-- The former latent divergence, now closed: for a trailing `Unbound` column the view
answers `Some(Null)` and so does the owned snapshot (it used to answer `None`, which
`ExprEval::resolve_var` (`eval.rs`) turned into "Variable not found"). -/
def trailingEx (F : Type) : Batch F :=
  { len := 1, sel := none, cols := [.ints [1], .unbound, .unbound], origins := none, vo := fun _ => false }

theorem toOwned_trailing_unbound (F : Type) :
    viewAt (trailingEx F) 0 2 = some .null ∧ (toOwned (trailingEx F) 0).get 2 = some .null ∧
    viewAt (trailingEx F) 0 3 = none ∧ (toOwned (trailingEx F) 0).get 3 = none ∧
    (toOwned (trailingEx F) 0).bound 0 = true ∧ (toOwned (trailingEx F) 0).bound 1 = false ∧
    (toOwned (trailingEx F) 0).bound 2 = false := by
  obtain ⟨-, hg, hb⟩ := pushSlots_spec (trailingEx F) 0 3
  have e : (toOwned (trailingEx F) 0) = pushSlots (trailingEx F) 0 3 := rfl
  rw [e]
  refine ⟨rfl, ?_, rfl, ?_, ?_, ?_, ?_⟩
  · simp only [Row.get] at hg ⊢; rw [hg 2]; simp [slotOf, trailingEx, Batch.column, Column.isUnbound]
  · simp only [Row.get] at hg ⊢; rw [hg 3]; simp
  · rw [hb]; simp [slotOf, trailingEx, Batch.column, Column.isUnbound]
  · rw [hb]; simp [slotOf, trailingEx, Batch.column, Column.isUnbound]
  · rw [hb]; simp [slotOf, trailingEx, Batch.column, Column.isUnbound]

end Columnar
