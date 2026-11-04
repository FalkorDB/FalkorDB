import FalkorRuntimeDS.Row
import FalkorRuntimeDS.Cow
/-
# Remaining `Row` methods, the `RowView` contract, `Cow::new`, `string_pool::global`

| here | there |
| --- | --- |
| `RowView`          | `trait RowView { value_at; to_owned_row }` (row.rs:44-57) |
| `withCapacity`     | `Row::with_capacity` (row.rs:82) |
| `pushSlot`         | `Row::push_slot` (row.rs:224, #3059): append one slot after the last, bound bit as given |
| `fromRaw`          | `Row::from_raw` (row.rs:92) |
| `lenR` / `isEmptyR`| `Row::len` / `Row::is_empty` (row.rs:169, :175) |
| `hasBindings`      | `Row::has_bindings` (row.rs:181) |
| `toOwnedRow`       | `impl RowView for Row :: to_owned_row` (row.rs:238) — `self.clone()` |
| `cowNew`           | `Cow::new` (cow.rs:84) |
| `globalPool`       | `string_pool::global` (string_pool.rs:66) — `OnceLock::get_or_init` |
-/
namespace FalkorRuntimeDS.RowModel
open FalkorRuntimeDS.BitSetModel

variable {β : Type}

/-- row.rs:44-57. The trait's two methods, as a structure of functions over a row type `R`. -/
structure RowView (R β : Type) where
  valueAt : R → Nat → Option (Option β)
  toOwnedRow : R → Row β

/-- The trait contract (row.rs:38-43): the owned snapshot answers `value_at` identically. The
snapshot is what defines the row a view stands for (`R` is abstract), so this is the contract
relative to it; the out-of-range clause (`value_at` is `None` exactly when the slot is out of the
snapshot's range) follows (`RowView.Lawful.valueAt_none_iff`). A view that always answers `none`
with `Row.new` as its snapshot is lawful: it stands for the empty row. -/
def RowView.Lawful (v : RowView R β) : Prop :=
  ∀ r i, getById (v.toOwnedRow r) i = v.valueAt r i

/-- The out-of-range clause of the contract: `value_at` is `None` exactly past the snapshot's slots. -/
theorem RowView.Lawful.valueAt_none_iff {v : RowView R β} (h : v.Lawful) (r : R) (i : Nat) :
    v.valueAt r i = none ↔ (v.toOwnedRow r).values.length ≤ i := by
  rw [← h r i]; unfold getById; simp

/-- `impl RowView for Row` (row.rs:230-240). -/
def rowView : RowView (Row β) β := ⟨getById, fun r => r⟩

/-- row.rs:47 / row.rs:56 (trait declarations): `Row`'s implementation is lawful. -/
theorem rowView_lawful : (rowView : RowView (Row β) β).Lawful := fun _ _ => rfl

/-- row.rs:238 — `to_owned_row` is `clone`: identical values and bound bits. `Row::origin_row` is
not in the `Row` model, so that field's preservation is not stated here (README, "What the model does
not cover"). -/
def toOwnedRow (r : Row β) : Row β := r
theorem toOwnedRow_eq (r : Row β) : toOwnedRow r = r ∧ ∀ i,
    getById (toOwnedRow r) i = getById r i ∧ isBound (toOwnedRow r) i = isBound r i :=
  ⟨rfl, fun _ => ⟨rfl, rfl⟩⟩

/-- row.rs:76 — `Row::default()`. -/
theorem new_spec (i : Nat) : getById (Row.new : Row β) i = none ∧ isBound (Row.new : Row β) i = false :=
  ⟨rfl, test_empty i⟩

/-- row.rs:82 — capacity only, no slots, no bindings. -/
def withCapacity (_n : Nat) : Row β := Row.new
theorem withCapacity_eq (n : Nat) : withCapacity n = (Row.new : Row β) := rfl

/-- row.rs:224 — `Row::push_slot(value, bound)`: `if bound { self.bound.set(self.values.len()) }`,
then `self.values.push(value)`. -/
def pushSlot (r : Row β) (v : Option β) (b : Bool) : Row β :=
  { values := r.values ++ [v], bound := if b then set r.bound r.values.length else r.bound }

/-- **`push_slot` appends exactly one slot**: the new slot (at the old length) reads `v`, every
earlier slot is untouched, and the row grows by one. -/
theorem getById_pushSlot (r : Row β) (v : Option β) (b : Bool) (i : Nat) :
    getById (pushSlot r v b) i = if i = r.values.length then some v else getById r i := by
  unfold getById pushSlot
  by_cases h : i = r.values.length
  · subst h; simp
  · rw [if_neg h]
    by_cases h2 : i < r.values.length
    · simp [List.getElem?_append_left h2]
    · have h3 : r.values.length ≤ i := by omega
      rw [List.getElem?_append_right h3, List.getElem?_eq_none h3]
      simp
      intro h5; omega

theorem length_pushSlot (r : Row β) (v : Option β) (b : Bool) :
    (pushSlot r v b).values.length = r.values.length + 1 := by
  simp [pushSlot]

/-- The bound bit: `bound = true` sets the new slot's bit (every other bit is kept), `bound =
false` leaves the bitmap alone, so a fresh slot stays unbound. -/
theorem isBound_pushSlot (r : Row β) (v : Option β) (b : Bool) (i : Nat) :
    isBound (pushSlot r v b) i = if b then (isBound r i || decide (i = r.values.length)) else isBound r i := by
  unfold isBound pushSlot
  cases b <;> simp [test_set]

/-- On a row whose bitmap has no stale bit at the new index (e.g. one built only by
`withCapacity` + `pushSlot`), an unbound push leaves the slot in scope but unbound: `Some Null`
and not bound. -/
theorem pushSlot_unbound (r : Row β) (hfree : isBound r r.values.length = false) :
    getById (pushSlot r none false) r.values.length = some none ∧
    isBound (pushSlot r none false) r.values.length = false := by
  refine ⟨by simp [getById_pushSlot], ?_⟩
  rw [isBound_pushSlot]; simpa using hfree

/-- row.rs:92 -/
def fromRaw (values : List (Option β)) (bound : BitSet) : Row β := ⟨values, bound⟩
theorem fromRaw_spec (values : List (Option β)) (bound : BitSet) (i : Nat) :
    getById (fromRaw values bound) i = values[i]? ∧ isBound (fromRaw values bound) i = test bound i :=
  ⟨rfl, rfl⟩

/-- row.rs:169 / :175 -/
def lenR (r : Row β) : Nat := r.values.length
def isEmptyR (r : Row β) : Bool := r.values.isEmpty
theorem isEmptyR_iff (r : Row β) : isEmptyR r = true ↔ lenR r = 0 := by
  cases r with | mk vs b => cases vs <;> simp [isEmptyR, lenR]
theorem lenR_new : lenR (Row.new : Row β) = 0 := rfl
theorem getById_none_iff (r : Row β) (i : Nat) : getById r i = none ↔ lenR r ≤ i := by
  simp [getById, lenR]

/-- row.rs:181 -/
def hasBindings (r : Row β) : Bool := !isEmpty r.bound
theorem hasBindings_iff (r : Row β) (hs : WF r.bound) :
    hasBindings r = true ↔ ∃ c, isBound r c = true := by
  have h := isEmpty_iff r.bound hs
  unfold hasBindings isBound
  constructor
  · intro hb
    cases he : isEmpty r.bound
    · apply Classical.byContradiction
      intro hn
      have : isEmpty r.bound = true := h.mpr (fun c => by
        cases hc : test r.bound c
        · rfl
        · exact absurd ⟨c, hc⟩ hn)
      rw [he] at this; cases this
    · rw [he] at hb; cases hb
  · rintro ⟨c, hc⟩
    cases he : isEmpty r.bound
    · rfl
    · have := (h.mp he) c; rw [this] at hc; cases hc
theorem hasBindings_new : hasBindings (Row.new : Row β) = false := rfl

end FalkorRuntimeDS.RowModel

namespace FalkorRuntimeDS.CowModel
/-- cow.rs:84 — `Self { inner, dup: false, .. }` (plus a fresh debug-only `branched` flag): no pending duplication. -/
def cowNew (h : Nat) : Cow := { h, dup := false }
/-- `Cow::new` builds an unshared handle: reads see `inner`, the first `deref_mut`
neither allocates nor moves. -/
theorem cowNew_spec {V : Type} (H : Heap V) (h : Nat) :
    read H (cowNew h) = H.mem h ∧ derefMut H (cowNew h) = (H, cowNew h) := ⟨rfl, rfl⟩
end CowModel

/-- string_pool.rs:66 — `POOL.get_or_init(StringPool::new)` over the cell state. -/
def globalPool {P : Type} (init : P) : Option P → P × Option P
  | none => (init, some init)
  | some p => (p, some p)

/-- The first call installs `StringPool::new()`, every call returns the installed pool and
leaves the cell unchanged afterwards (one process-wide pool). -/
theorem globalPool_spec {P : Type} (init : P) (c : Option P) :
    (globalPool init c).2 = some (globalPool init c).1 ∧
    (globalPool init (globalPool init c).2) = globalPool init c ∧
    (c = none → (globalPool init c).1 = init) ∧
    (∀ p, c = some p → (globalPool init c).1 = p) := by
  cases c <;> simp [globalPool]

end FalkorRuntimeDS
