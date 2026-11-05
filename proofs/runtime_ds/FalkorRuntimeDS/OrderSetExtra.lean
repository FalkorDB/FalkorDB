import FalkorRuntimeDS.OrderSet
/-
# `OrderSet` (`graph/src/runtime/orderset.rs`) — the remaining methods

Same list model as `OrderSet.lean` (the `Vec<T>` *is* the list).

| here | there |
| --- | --- |
| `emptyS`     | `Default::default` (orderset.rs:27) |
| `fromVecGo`/`fromVec` | `OrderSet::from_vec` (orderset.rs:42) — since #2905 drops later duplicates in every build, first wins (loop state `acc` = `vec[..i]`) |
| `iterS`      | `OrderSet::iter` (orderset.rs:77) |
| `lenS`       | `OrderSet::len` (orderset.rs:82) |
| `isEmptyS`   | `OrderSet::is_empty` (orderset.rs:87) |
| `clearS`     | `OrderSet::clear` (orderset.rs:112) |
| `getS`       | `OrderSet::get` (orderset.rs:129) — `Vec::get` |
| `indexS`     | `Index<usize>` (orderset.rs:148) — `.error` = the `expect` panic |
| `intoIterS`  | `IntoIterator` (orderset.rs:160) |
-/
set_option linter.unusedSectionVars false
namespace FalkorRuntimeDS.OrderSetModel

variable {α : Type}

/-- orderset.rs:27 -/
def emptyS : List α := []
/-- orderset.rs:77 -/
def iterS (l : List α) : List α := l
/-- orderset.rs:82 -/
def lenS (l : List α) : Nat := l.length
/-- orderset.rs:87 -/
def isEmptyS (l : List α) : Bool := l.isEmpty
/-- orderset.rs:112 -/
def clearS (_l : List α) : List α := []
/-- orderset.rs:129 -/
def getS (l : List α) (i : Nat) : Option α := l[i]?
/-- orderset.rs:148 — `self.vec.get(index).expect("no entry found for key")`. -/
def indexS (l : List α) (i : Nat) : Except String α :=
  match l[i]? with
  | some x => .ok x
  | none => .error "no entry found for key"
/-- orderset.rs:160 -/
def intoIterS (l : List α) : List α := l

section Props
variable [BEq α]

theorem containsS_emptyS (y : α) : containsS (emptyS : List α) y = false := rfl
theorem emptyS_nodup : Nodup (emptyS : List α) := by simp [Nodup, emptyS]

/-- `from_iter` = `extend` into `default()`. -/
theorem fromListS_eq_foldl_emptyS (l : List α) :
    fromListS l = l.foldl (fun acc x => (insertS acc x).1) emptyS := rfl

theorem insertS_fresh (l : List α) (x : α) (h : ∀ v ∈ l, (v == x) = false) :
    insertS l x = (l ++ [x], none) := by
  induction l with
  | nil => rfl
  | cons v t ih =>
    have hv := h v (List.mem_cons_self ..)
    simp only [insertS, hv, Bool.false_eq_true, ite_false]
    rw [ih (fun w hw => h w (List.mem_cons_of_mem _ hw))]
    rfl

theorem foldl_insertS_unique (acc l : List α) (h : Nodup (acc ++ l)) :
    l.foldl (fun acc x => (insertS acc x).1) acc = acc ++ l := by
  induction l generalizing acc with
  | nil => simp
  | cons x t ih =>
    simp only [List.foldl_cons]
    have hx : ∀ v ∈ acc, (v == x) = false := by
      intro v hv
      unfold Nodup at h
      rw [List.pairwise_append] at h
      exact h.2.2 v hv x (by simp)
    rw [insertS_fresh _ _ hx]
    simp only
    rw [ih (acc ++ [x]) (by simpa using h)]
    simp

/-! ### `from_vec` (orderset.rs:42, #2905): drop later duplicates, first wins -/

section FromVec
variable [EquivBEq α]

/-- orderset.rs:43-51, the `while i < vec.len()` loop. `acc` is the kept
prefix `vec[..i]`, the second list the unvisited suffix: a visited element
already `contains`ed in the prefix is `vec.remove(i)`d, else `i += 1`. -/
def fromVecGo : List α → List α → List α
  | acc, [] => acc
  | acc, x :: t => if containsS acc x then fromVecGo acc t else fromVecGo (acc ++ [x]) t

/-- orderset.rs:42 -/
def fromVec (l : List α) : List α := fromVecGo [] l

theorem fromVecGo_nodup (acc l : List α) (h : Nodup acc) : Nodup (fromVecGo acc l) := by
  induction l generalizing acc with
  | nil => exact h
  | cons x t ih =>
    simp only [fromVecGo]; split
    · exact ih acc h
    · next hc =>
      apply ih
      unfold Nodup at h ⊢
      rw [List.pairwise_append]
      refine ⟨h, by simp, fun a ha b hb => ?_⟩
      simp only [List.mem_singleton] at hb; subst hb
      rw [containsS_eq_any] at hc
      simp only [Bool.not_eq_true, List.any_eq_false] at hc
      simpa using hc a ha

theorem fromVecGo_of_nodup (acc l : List α) (h : Nodup (acc ++ l)) : fromVecGo acc l = acc ++ l := by
  induction l generalizing acc with
  | nil => simp [fromVecGo]
  | cons x t ih =>
    have hx : containsS acc x = false := by
      rw [containsS_eq_any]
      simp only [List.any_eq_false]
      intro v hv hvx
      unfold Nodup at h; rw [List.pairwise_append] at h
      have := h.2.2 v hv x (by simp)
      rw [hvx] at this; exact absurd this (by simp)
    simp only [fromVecGo, hx, Bool.false_eq_true, ite_false]
    rw [ih (acc ++ [x]) (by simpa using h)]; simp

theorem fromVecGo_sublist (acc l : List α) : (fromVecGo acc l).Sublist (acc ++ l) := by
  induction l generalizing acc with
  | nil => simp [fromVecGo]
  | cons x t ih =>
    simp only [fromVecGo]; split
    · exact (ih acc).trans (List.Sublist.append_left (List.sublist_cons_self x t) acc)
    · have := ih (acc ++ [x]); simpa using this

theorem containsS_fromVecGo (acc l : List α) (y : α) :
    containsS (fromVecGo acc l) y = (containsS acc y || containsS l y) := by
  induction l generalizing acc with
  | nil => simp [fromVecGo, containsS]
  | cons x t ih =>
    simp only [fromVecGo]; split
    · next hc =>
      rw [ih acc]
      simp only [containsS_eq_any, List.any_cons] at hc ⊢
      by_cases hxy : (x == y) = true
      · obtain ⟨v, hv, hvx⟩ := List.any_eq_true.mp hc
        have : acc.any (· == y) = true := List.any_eq_true.mpr ⟨v, hv, BEq.trans hvx hxy⟩
        simp [this]
      · simp [hxy]
    · rw [ih (acc ++ [x])]
      simp only [containsS_eq_any, List.any_append, List.any_cons, List.any_nil]
      cases acc.any (· == y) <;> cases (x == y) <;> simp

/-- `from_vec` always yields a set (no `==`-duplicates), in every build. -/
theorem fromVec_nodup (l : List α) : Nodup (fromVec l) := fromVecGo_nodup [] l (by simp [Nodup])

/-- On a duplicate-free vector (every in-tree caller) `from_vec` is the identity. -/
theorem fromVec_of_nodup (l : List α) (h : Nodup l) : fromVec l = l := by
  unfold fromVec; rw [fromVecGo_of_nodup [] l (by simpa using h)]; simp

/-- It only drops elements, keeping the order of the survivors. -/
theorem fromVec_sublist (l : List α) : (fromVec l).Sublist l := by
  simpa [fromVec] using fromVecGo_sublist [] l

/-- It drops nothing a `contains` could see. -/
theorem containsS_fromVec (l : List α) (y : α) : containsS (fromVec l) y = containsS l y := by
  simp [fromVec, containsS_fromVecGo, containsS]

/-- **The #2903 fix.** `remove` after `from_vec` really removes, whatever the
input: no `Nodup` hypothesis. -/
theorem containsS_removeS_fromVec (l : List α) (x y : α) :
    containsS (removeS (fromVec l) x) y = (containsS l y && !(x == y)) := by
  rw [containsS_removeS (fromVec_nodup l), containsS_fromVec]

/-- On a duplicate-free vector it agrees with `from_iter`. -/
theorem fromVec_eq_fromListS (l : List α) (h : Nodup l) : fromVec l = fromListS l := by
  rw [fromVec_of_nodup l h]
  unfold fromListS
  rw [foldl_insertS_unique [] l (by simpa using h)]
  simp

end FromVec

/-- Main's unit test `orderset.rs::from_vec_drops_duplicates`: first wins, order kept. -/
theorem fromVec_drops_duplicates :
    fromVec [1, 2, 1, 3, 2] = [1, 2, 3] ∧ containsS (removeS (fromVec [1, 2, 1, 3, 2]) 1) 1 = false := by
  decide

theorem fromVec_dup_removed : containsS (removeS (fromVec [1, 1]) 1) 1 = false := by decide

theorem iterS_eq (l : List α) : iterS l = l := rfl
theorem intoIterS_eq (l : List α) : intoIterS l = l := rfl

/-- Iteration order is insertion order. -/
theorem iterS_insert_fresh (l : List α) (x : α) (h : ∀ v ∈ l, (v == x) = false) :
    iterS (insertS l x).1 = iterS l ++ [x] := by
  rw [insertS_fresh _ _ h]; rfl

theorem lenS_insertS (l : List α) (x : α) :
    lenS (insertS l x).1 = if (insertS l x).2.isSome then lenS l else lenS l + 1 := by
  induction l with
  | nil => rfl
  | cons v t ih =>
    unfold lenS at *
    simp only [insertS]
    split
    · simp
    · simp only [List.length_cons, ih]; split <;> rfl

theorem lenS_removeS (l : List α) (x : α) :
    lenS (removeS l x) = if containsS l x then lenS l - 1 else lenS l := by
  induction l with
  | nil => rfl
  | cons v t ih =>
    unfold lenS at *
    simp only [removeS, containsS]
    split
    · simp
    · simp only [List.length_cons, ih]
      split
      · rename_i hc
        have : 0 < t.length := by cases t with | nil => simp [containsS] at hc | cons => simp
        omega
      · rfl

theorem isEmptyS_iff (l : List α) : isEmptyS l = true ↔ lenS l = 0 := by
  cases l <;> simp [isEmptyS, lenS]

theorem clearS_eq (l : List α) : clearS l = emptyS := rfl
theorem containsS_clearS (l : List α) (y : α) : containsS (clearS l) y = false := rfl
theorem lenS_clearS (l : List α) : lenS (clearS l) = 0 := rfl

theorem getS_eq (l : List α) (i : Nat) : getS l i = l[i]? := rfl

/-- `get(get_index_of(y))` returns an element `==` to `y`. -/
theorem getS_indexOf [EquivBEq α] (l : List α) (y : α) (i : Nat) (h : indexOf l y = some i) :
    ∃ x, getS l i = some x ∧ (x == y) = true := by
  obtain ⟨hi, heq, _⟩ := indexOf_spec l y i h
  exact ⟨l[i], List.getElem?_eq_getElem hi, heq⟩

theorem getS_none_iff (l : List α) (i : Nat) : getS l i = none ↔ lenS l ≤ i := by
  simp [getS, lenS]

/-- `s[i]` is `get(i)` and panics exactly when `i >= len`. -/
theorem indexS_ok_iff (l : List α) (i : Nat) (x : α) : indexS l i = .ok x ↔ getS l i = some x := by
  unfold indexS getS; split <;> simp_all

theorem indexS_panics_iff (l : List α) (i : Nat) : (∃ e, indexS l i = .error e) ↔ lenS l ≤ i := by
  unfold indexS lenS; split
  · rename_i x hx
    have := (List.getElem?_eq_some_iff.mp hx).1
    constructor
    · rintro ⟨e, he⟩; cases he
    · intro h; omega
  · rename_i hx
    exact ⟨fun _ => List.getElem?_eq_none_iff.mp hx, fun _ => ⟨_, rfl⟩⟩

end Props

end FalkorRuntimeDS.OrderSetModel
