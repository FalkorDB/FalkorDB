/-
# `OrderSet` (`graph/src/runtime/orderset.rs`) is an insertion-ordered set

Elements are compared with an arbitrary `BEq` (Rust `PartialEq`); a duplicate
insert *replaces* the stored element with the new one (`std::mem::replace`)
and returns the old one, keeping its slot.

| here | there |
| --- | --- |
| `List α`        | `struct OrderSet<T> { vec: Vec<T> }` (orderset.rs:22); `from_vec` (orderset.rs:34) is the identity |
| `insertS`       | `OrderSet::insert` (orderset.rs:38) |
| `removeS`       | `OrderSet::remove` (orderset.rs:52) — `position` then `Vec::remove` |
| `containsS`     | `OrderSet::contains` (orderset.rs:75) |
| `indexOf`       | `OrderSet::get_index_of` (orderset.rs:100) |
| `fromListS`     | `FromIterator` (orderset.rs:121), `extend` (orderset.rs:87) |
| `List` `=`      | `#[derive(PartialEq)]` (orderset.rs:21) — order-*sensitive* |

There is no `swap_remove` in either `OrderSet` or `OrderMap`: every removal is
an order-preserving `Vec::remove`, so the only index invalidation is the shift
of every later element by one (`indexOf_removeS`).
-/
namespace FalkorRuntimeDS.OrderSetModel

variable {α : Type}

section Defs
variable [BEq α]

/-- orderset.rs:38 -/
def insertS : List α → α → List α × Option α
  | [], x => ([x], none)
  | v :: t, x => if v == x then (x :: t, some v) else let r := insertS t x; (v :: r.1, r.2)

/-- orderset.rs:52 -/
def removeS : List α → α → List α
  | [], _ => []
  | v :: t, x => if v == x then t else v :: removeS t x

/-- orderset.rs:75 -/
def containsS : List α → α → Bool
  | [], _ => false
  | v :: t, x => if v == x then true else containsS t x

/-- orderset.rs:100 -/
def indexOf : List α → α → Option Nat
  | [], _ => none
  | v :: t, x => if v == x then some 0 else (indexOf t x).map (· + 1)

def fromListS (l : List α) : List α := l.foldl (fun acc x => (insertS acc x).1) []

def Nodup (l : List α) : Prop := l.Pairwise (fun a b => (a == b) = false)

end Defs

section Props
variable [BEq α] [EquivBEq α]

theorem containsS_eq_any (l : List α) (y : α) : containsS l y = l.any (· == y) := by
  induction l with
  | nil => rfl
  | cons v t ih => simp only [containsS, List.any_cons]; split <;> simp_all

theorem containsS_insertS (l : List α) (x y : α) :
    containsS (insertS l x).1 y = (containsS l y || x == y) := by
  induction l with
  | nil => simp [insertS, containsS]
  | cons v t ih =>
    simp only [insertS]
    by_cases h : (v == x) = true
    · simp only [h, ite_true, containsS]
      rw [BEq.congr_left h]
      by_cases hx : (x == y) = true <;> simp [hx]
    · simp only [h, Bool.false_eq_true, ite_false, containsS, ih]
      by_cases hv : (v == y) = true <;> simp [hv]

theorem insertS_nodup {l : List α} (hn : Nodup l) (x : α) : Nodup (insertS l x).1 := by
  induction l with
  | nil => simp [insertS, Nodup]
  | cons v t ih =>
    unfold Nodup at hn; rw [List.pairwise_cons] at hn
    simp only [insertS]
    by_cases h : (v == x) = true
    · simp only [h, ite_true, Nodup, List.pairwise_cons]
      refine ⟨fun b hb => ?_, hn.2⟩
      exact BEq.neq_of_beq_of_neq (BEq.symm h) (hn.1 b hb)
    · simp only [h, Bool.false_eq_true, ite_false, Nodup, List.pairwise_cons]
      refine ⟨fun b hb => ?_, ih hn.2⟩
      -- b is an element of t, or x itself
      have : containsS (insertS t x).1 b = true := by
        rw [containsS_eq_any]; simp; exact ⟨b, hb, BEq.rfl⟩
      rw [containsS_insertS, containsS_eq_any] at this
      simp only [Bool.or_eq_true, List.any_eq_true] at this
      rcases this with ⟨c, hc, hcb⟩ | hxb
      · exact BEq.neq_of_neq_of_beq (hn.1 c hc) hcb
      · exact BEq.neq_of_neq_of_beq (by simpa using h) hxb

theorem removeS_sublist (l : List α) (x : α) : (removeS l x).Sublist l := by
  induction l with
  | nil => simp [removeS]
  | cons v t ih =>
    simp only [removeS]; split
    · exact List.Sublist.cons _ (List.Sublist.refl _)
    · exact List.Sublist.cons_cons _ ih

theorem removeS_nodup {l : List α} (hn : Nodup l) (x : α) : Nodup (removeS l x) :=
  List.Pairwise.sublist (removeS_sublist l x) hn

theorem fromListS_nodup (l : List α) : Nodup (fromListS l) := by
  unfold fromListS
  suffices ∀ acc, Nodup acc → Nodup (l.foldl (fun acc x => (insertS acc x).1) acc) from
    this [] (by simp [Nodup])
  induction l with
  | nil => intro acc h; exact h
  | cons p t ih => intro acc h; exact ih _ (insertS_nodup h _)

/-- `contains` after `remove` (no duplicates): exactly `x` (and its `==`-class) is gone. -/
theorem containsS_removeS {l : List α} (hn : Nodup l) (x y : α) :
    containsS (removeS l x) y = (containsS l y && !(x == y)) := by
  induction l with
  | nil => simp [removeS, containsS]
  | cons v t ih =>
    unfold Nodup at hn; rw [List.pairwise_cons] at hn
    simp only [removeS]
    by_cases h : (v == x) = true
    · simp only [h, ite_true, containsS]
      by_cases hxy : (x == y) = true
      · have hvy : (v == y) = true := BEq.trans h hxy
        simp only [hvy, hxy, ite_true, Bool.not_true, Bool.and_false]
        rw [containsS_eq_any]
        simp only [List.any_eq_false]
        intro c hc hcy
        have := hn.1 c hc
        rw [BEq.trans hvy (BEq.symm hcy)] at this
        exact absurd this (by simp)
      · have : (v == y) = false := BEq.neq_of_beq_of_neq h (by simpa using hxy)
        simp [this, hxy]
    · simp only [h, Bool.false_eq_true, ite_false, containsS, ih hn.2]
      by_cases hv : (v == y) = true
      · have : (x == y) = false :=
          BEq.symm_false (BEq.neq_of_beq_of_neq (BEq.symm hv) (by simpa using h))
        simp [hv, this]
      · simp [hv]

/-- `get_index_of` after `insert`: existing indices never move; a new element lands at `len`. -/
theorem indexOf_insertS (l : List α) (x y : α) :
    indexOf (insertS l x).1 y =
      match indexOf l y with
      | some i => some i
      | none => if x == y then some l.length else none := by
  induction l with
  | nil => simp [insertS, indexOf]
  | cons v t ih =>
    simp only [insertS]
    by_cases h : (v == x) = true
    · simp only [h, ite_true, indexOf]
      rw [BEq.congr_left h]
      by_cases hx : (x == y) = true
      · simp [hx]
      · simp only [hx, Bool.false_eq_true, ite_false]
        cases indexOf t y <;> simp
    · simp only [h, Bool.false_eq_true, ite_false, indexOf, ih]
      by_cases hv : (v == y) = true
      · simp [hv]
      · simp only [hv, Bool.false_eq_true, ite_false]
        cases indexOf t y <;> simp

/-- The index `get_index_of` returns points at an `==` element, and it is the first. -/
theorem indexOf_spec (l : List α) (y : α) (i : Nat) (h : indexOf l y = some i) :
    ∃ hi : i < l.length, (l[i] == y) = true ∧ ∀ j (hj : j < i), (l[j]'(by omega) == y) = false := by
  induction l generalizing i with
  | nil => simp [indexOf] at h
  | cons v t ih =>
    simp only [indexOf] at h
    split at h
    · rename_i hv; simp at h; subst h; exact ⟨by simp, hv, fun j hj => by omega⟩
    · rename_i hv
      obtain ⟨i', hi', rfl⟩ : ∃ i', indexOf t y = some i' ∧ i = i' + 1 := by
        cases hh : indexOf t y <;> simp [hh] at h; exact ⟨_, rfl, h.symm⟩
      obtain ⟨hlt, heq, hfirst⟩ := ih i' hi'
      refine ⟨by simp; omega, by simpa using heq, ?_⟩
      intro j hj
      cases j with
      | zero => simpa using hv
      | succ j => simpa using hfirst j (by omega)

/-- Index invalidation after `remove(x)` (no duplicates): elements before `x`'s slot
keep their index, elements after it shift down by exactly one, `x` itself is gone. -/
theorem indexOf_removeS {l : List α} (hn : Nodup l) (x y : α) (p : Nat)
    (hp : indexOf l x = some p) :
    indexOf (removeS l x) y =
      if x == y then none
      else (indexOf l y).map (fun j => if p < j then j - 1 else j) := by
  induction l generalizing p with
  | nil => simp [indexOf] at hp
  | cons v t ih =>
    unfold Nodup at hn; rw [List.pairwise_cons] at hn
    simp only [indexOf] at hp
    simp only [removeS]
    by_cases h : (v == x) = true
    · simp only [h, ite_true] at hp ⊢
      simp at hp; subst hp
      simp only [indexOf]
      by_cases hxy : (x == y) = true
      · have hvy : (v == y) = true := BEq.trans h hxy
        simp only [hxy, ite_true]
        cases hi : indexOf t y with
        | none => rfl
        | some i =>
          obtain ⟨hlt, heq, _⟩ := indexOf_spec t y i hi
          have := hn.1 t[i] (List.getElem_mem hlt)
          exact absurd (BEq.trans hvy (BEq.symm heq)) (by simp [this])
      · have : (v == y) = false := BEq.neq_of_beq_of_neq h (by simpa using hxy)
        simp only [hxy, this, Bool.false_eq_true, ite_false]
        cases indexOf t y <;> simp
    · simp only [h, Bool.false_eq_true, ite_false] at hp ⊢
      obtain ⟨p', hp', rfl⟩ : ∃ p', indexOf t x = some p' ∧ p = p' + 1 := by
        cases hh : indexOf t x <;> simp [hh] at hp; exact ⟨_, rfl, hp.symm⟩
      simp only [indexOf, ih hn.2 p' hp']
      by_cases hv : (v == y) = true
      · have : (x == y) = false :=
          BEq.symm_false (BEq.neq_of_beq_of_neq (BEq.symm hv) (by simpa using h))
        simp [hv, this]
      · simp only [hv, Bool.false_eq_true, ite_false]
        by_cases hxy : (x == y) = true
        · simp [hxy]
        · simp only [hxy, Bool.false_eq_true, ite_false]
          cases indexOf t y with
          | none => rfl
          | some j =>
            simp only [Option.map_some]
            by_cases hj : p' < j
            · simp [hj]; omega
            · simp [hj]

end Props

/-! ## Counterexamples

* `from_vec` does not deduplicate (orderset.rs:34). With a duplicate, `remove`
  leaves the element in: `containsS_removeS` needs `Nodup`.
  (Rust: `graph/tests/lean_runtime_ds.rs::orderset_from_vec_duplicates`.)
  The one in-tree caller (`reorder_labels.rs:41`) passes a permutation of an
  existing `OrderSet`, so it is `Nodup` and safe.
* The derived `PartialEq` is list equality, so `{a, b} ≠ {b, a}` — unlike
  `OrderMap`, whose `==` ignores order (`OrderMapModel.eqM_iff_perm`). -/

example : containsS (removeS [1, 1] 1) 1 = true := by decide
example : ([1, 2] : List Nat) ≠ [2, 1] := by decide

end FalkorRuntimeDS.OrderSetModel
