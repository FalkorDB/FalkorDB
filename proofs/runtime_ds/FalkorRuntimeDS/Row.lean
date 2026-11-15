import FalkorRuntimeDS.BitSet
/-
# `Row` (`graph/src/runtime/row.rs`): dense slots + a bound bitmap

`Value` is abstracted as `Option β` with `none` = `Value::Null`. `id: u32`
widens to `usize` losslessly on the 64-bit targets FalkorDB builds for, so ids
are `Nat` (a model `u32` is any `Nat < 2^32`; nothing here depends on it).

| here | there |
| --- | --- |
| `Row`            | `struct Row { values: SmallVec<[Value; 4]>, bound: BitSet, origin_row }` (row.rs:65) |
| `insertById`     | `Row::insert_by_id` (row.rs:113) / `insert` (row.rs:104) |
| `getById`        | `Row::get_by_id` (row.rs:128), `get` (row.rs:138), `RowView::value_at` (row.rs:231) |
| `take`           | `Row::take` (row.rs:147) |
| `unbind`         | `Row::unbind` / `unbind_by_id` (row.rs:160, 222) |
| `isBound`        | `Row::is_bound_by_id` (row.rs:199) |
| `merge`          | `Row::merge` (row.rs:209) |
| `cloneWith`      | `Row::clone_with` (row.rs:187) |
| `hashed`         | the `(key, value)` sequence fed to the hasher by `Hash for Row` (row.rs:243) |
-/
namespace FalkorRuntimeDS.RowModel
open FalkorRuntimeDS.BitSetModel

variable {β : Type}

structure Row (β : Type) where
  values : List (Option β)
  bound : BitSet

def Row.new : Row β := ⟨[], empty⟩

/-- row.rs:113 — `resize_with(idx + 1, || Null)` when too short, then write & mark bound. -/
def insertById (r : Row β) (id : Nat) (v : Option β) : Row β :=
  let vals := if r.values.length ≤ id then r.values ++ List.replicate (id + 1 - r.values.length) none
              else r.values
  { values := vals.set id v, bound := set r.bound id }

/-- row.rs:128 — `None` iff out of range; `Some(Null)` for an in-range padded slot. -/
def getById (r : Row β) (id : Nat) : Option (Option β) := r.values[id]?

def isBound (r : Row β) (id : Nat) : Bool := test r.bound id

def unbind (r : Row β) (id : Nat) : Row β := { r with bound := clear r.bound id }

/-- row.rs:147 — take the value, leave `Null`; `None` if out of range or already `Null`.
The bound bit is **not** touched. -/
def take (r : Row β) (id : Nat) : Row β × Option β :=
  match r.values[id]? with
  | none => (r, none)
  | some x => ({ r with values := r.values.set id none }, x)

def cloneWith (r : Row β) (id : Nat) (v : Option β) : Row β := insertById r id v

/-- row.rs:209 — `for id in 0..other.values.len() { if other.bound.test(id) { insert } }`. -/
def mergeFrom (self other : Row β) : List Nat → Row β
  | [] => self
  | id :: ids =>
    let self' := if test other.bound id then insertById self id ((other.values[id]?).getD none) else self
    mergeFrom self' other ids

def merge (self other : Row β) : Row β := mergeFrom self other (List.range other.values.length)

/-- row.rs:243 — the `(key, value)` pairs hashed: every slot except unbound `Null`s. -/
def hashed (r : Row β) : List (Nat × Option β) :=
  (List.range r.values.length).filterMap fun i =>
    match r.values[i]? with
    | some v => if v.isNone && !test r.bound i then none else some (i, v)
    | none => none

/-! ## set / get -/

theorem length_insertById (r : Row β) (id : Nat) (v : Option β) :
    (insertById r id v).values.length = max r.values.length (id + 1) := by
  unfold insertById; split <;> simp <;> omega

/-- `get` after `insert_by_id`: the new value at `id`; old slots unchanged; the gap
padded with `Null`; beyond it still out of range. -/
theorem getById_insertById (r : Row β) (id j : Nat) (v : Option β) :
    getById (insertById r id v) j =
      if j = id then some v
      else if j < r.values.length then getById r j
      else if j < id then some none
      else none := by
  unfold getById insertById
  simp only
  rw [List.getElem?_set]
  by_cases hj : j = id
  · subst hj; simp only [ite_true]
    split <;> simp <;> omega
  · simp only [show ¬ id = j from fun e => hj e.symm, ite_false, hj]
    split
    · rw [List.getElem?_append]
      by_cases hl : j < r.values.length
      · simp [hl]
      · simp only [hl, dite_false, ite_false, List.getElem?_replicate]
        by_cases hji : j < id
        · simp [hji]; omega
        · simp [hji]; omega
    · by_cases hl : j < r.values.length
      · simp [hl]
      · have : r.values[j]? = none := by rw [List.getElem?_eq_none_iff]; omega
        simp [hl, this]; omega

theorem isBound_insertById (r : Row β) (id j : Nat) (v : Option β) :
    isBound (insertById r id v) j = (isBound r j || decide (j = id)) := by
  simp [isBound, insertById, test_set]

theorem isBound_unbind (r : Row β) (id j : Nat) :
    isBound (unbind r id) j = (isBound r j && !decide (j = id)) := by
  simp [isBound, unbind, test_clear]

/-- `unbind` leaves the stored value in place. -/
theorem getById_unbind (r : Row β) (id j : Nat) : getById (unbind r id) j = getById r j := rfl

/-- `clone_with` does not touch the original (values are immutable in the model;
`Row: Clone` is a deep `SmallVec` + `BitSet` clone in Rust). -/
theorem cloneWith_get (r : Row β) (id : Nat) (v : Option β) :
    getById (cloneWith r id v) id = some v := by
  rw [cloneWith, getById_insertById]; simp

/-! ## take -/

theorem take_returns (r : Row β) (id : Nat) :
    (take r id).2 = (r.values[id]?).getD none := by
  unfold take; split <;> simp_all

theorem take_leaves_null (r : Row β) (id : Nat) (h : id < r.values.length) :
    getById (take r id).1 id = some none := by
  unfold take getById
  have : r.values[id]? = some r.values[id] := List.getElem?_eq_getElem h
  rw [this]; simp [h]

/-- `take` keeps the bound bit, so a taken slot is "bound `Null`" — the
distinction `Hash for Row` then sees (`hashed_take`). -/
theorem take_keeps_bound (r : Row β) (id j : Nat) : isBound (take r id).1 j = isBound r j := by
  unfold take; split <;> rfl

/-! ## merge -/

theorem mergeFrom_preserve (other : Row β) (j : Nat) (x : Option β) :
    ∀ (ids : List Nat) (s : Row β), j ∉ ids → getById s j = some x →
      getById (mergeFrom s other ids) j = some x
  | [], _, _, hs => hs
  | k :: ks, s, hn, hs => by
    simp only [mergeFrom]
    apply mergeFrom_preserve other j x ks _ (fun h => hn (List.mem_cons_of_mem _ h))
    split
    · rw [getById_insertById]
      have : j ≠ k := fun e => hn (e ▸ List.mem_cons_self ..)
      simp only [this, ite_false]
      have hlt : j < s.values.length := by
        unfold getById at hs
        exact (List.getElem?_eq_some_iff.mp hs).1
      simp [hlt, hs]
    · exact hs

theorem mergeFrom_get (other : Row β) (j : Nat) (hb : test other.bound j = true) :
    ∀ (ids : List Nat) (self : Row β), ids.Nodup → j ∈ ids →
      getById (mergeFrom self other ids) j = some ((other.values[j]?).getD none)
  | [], _, _, hin => by simp at hin
  | id :: ids, self, hnd, hin => by
    simp only [mergeFrom]
    rw [List.nodup_cons] at hnd
    by_cases hj : j = id
    · subst hj
      apply mergeFrom_preserve _ _ _ _ _ hnd.1
      simp only [hb, ite_true]; rw [getById_insertById]; simp
    · exact mergeFrom_get other j hb ids _ hnd.2 (by simpa [hj] using hin)

/-- `merge` copies exactly the bound slots of `other`. -/
theorem merge_get_bound (self other : Row β) (j : Nat) (hj : j < other.values.length)
    (hb : isBound other j = true) :
    getById (merge self other) j = other.values[j]? := by
  unfold merge
  rw [mergeFrom_get _ j hb _ _ List.nodup_range (List.mem_range.mpr hj)]
  simp [List.getElem?_eq_getElem hj]

/-! ## Hash -/

/-- Trailing padding (unbound `Null` slots) is invisible to `Hash for Row`. -/
theorem hashed_pad (r : Row β) (n : Nat)
    (hpad : ∀ i, r.values.length ≤ i → test r.bound i = false) :
    hashed { r with values := r.values ++ List.replicate n none } = hashed r := by
  unfold hashed
  simp only [List.length_append, List.length_replicate]
  rw [List.range_add, List.filterMap_append]
  have : (List.map (fun x => r.values.length + x) (List.range n)).filterMap (fun i =>
      match (r.values ++ List.replicate n none)[i]? with
      | some v => if v.isNone && !test r.bound i then none else some (i, v)
      | none => none) = [] := by
    rw [List.filterMap_eq_nil_iff]
    intro i hi
    simp only [List.mem_map, List.mem_range] at hi
    obtain ⟨k, hk, rfl⟩ := hi
    rw [List.getElem?_append_right (by omega)]
    have e : r.values.length + k - r.values.length = k := by omega
    have hp := hpad (r.values.length + k) (by omega)
    simp [e, List.getElem?_replicate, hk, hp]
  rw [this, List.append_nil]
  suffices ∀ l : List Nat, (∀ i ∈ l, i < r.values.length) →
      l.filterMap (fun i => match (r.values ++ List.replicate n none)[i]? with
        | some v => if v.isNone && !test r.bound i then none else some (i, v)
        | none => none) =
      l.filterMap (fun i => match r.values[i]? with
        | some v => if v.isNone && !test r.bound i then none else some (i, v)
        | none => none) from this _ (fun i hi => List.mem_range.mp hi)
  intro l hl
  induction l with
  | nil => rfl
  | cons i t ih =>
    simp only [List.filterMap_cons]
    rw [List.getElem?_append_left (hl i (List.mem_cons_self ..))]
    rw [ih (fun x hx => hl x (List.mem_cons_of_mem _ hx))]

/-- …but a slot emptied by `take` (bound `Null`) *is* hashed, so a row that bound
then took a variable hashes differently from one that never bound it. -/
example :
    hashed (take (insertById (Row.new : Row Nat) 0 (some 5)) 0).1 = [(0, none)] ∧
    hashed (Row.new : Row Nat) = [] := by decide

/-- …and a stale value left behind by `unbind` is hashed too. -/
example : hashed (unbind (insertById (Row.new : Row Nat) 0 (some 5)) 0) = [(0, some 5)] := by
  decide

end FalkorRuntimeDS.RowModel
