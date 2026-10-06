import FalkorRuntimeDS.Row
import FalkorRuntimeDS.Cow
/-
# Remaining `Row` methods, the `RowView` contract, `Cow::new`, `string_pool::global`

| here | there |
| --- | --- |
| `RowView`          | `trait RowView { value_at; to_owned_row }` (row.rs:44-57) |
| `withCapacity`     | `Row::with_capacity` (row.rs:82) |
| `fromRaw`          | `Row::from_raw` (row.rs:92) |
| `lenR` / `isEmptyR`| `Row::len` / `Row::is_empty` (row.rs:169, :175) |
| `hasBindings`      | `Row::has_bindings` (row.rs:181) |
| `toOwnedRow`       | `impl RowView for Row :: to_owned_row` (row.rs:238) — `self.clone()` |
| `cowNew`           | `Cow::new` (cow.rs:50) |
| `globalPool`       | `string_pool::global` (string_pool.rs:66) — `OnceLock::get_or_init` |
-/
namespace FalkorRuntimeDS.RowModel
open FalkorRuntimeDS.BitSetModel

variable {β : Type}

/-- row.rs:44-57. The trait's two methods, as a structure of functions over a row type `R`. -/
structure RowView (R β : Type) where
  valueAt : R → Nat → Option (Option β)
  toOwnedRow : R → Row β

/-- The trait contract (row.rs:38-43): `value_at` is `None` exactly when the slot is out of
range, and the owned snapshot answers `value_at` identically. -/
def RowView.Lawful (v : RowView R β) : Prop :=
  ∀ r i, getById (v.toOwnedRow r) i = v.valueAt r i

/-- `impl RowView for Row` (row.rs:230-240). -/
def rowView : RowView (Row β) β := ⟨getById, fun r => r⟩

/-- row.rs:47 / row.rs:56 (trait declarations): `Row`'s implementation is lawful. -/
theorem rowView_lawful : (rowView : RowView (Row β) β).Lawful := fun _ _ => rfl

/-- row.rs:238 — `to_owned_row` is `clone`: identical values, bound bits (and origin). -/
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
/-- cow.rs:50 — `Self { inner, dup: false }`: no pending duplication. -/
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
