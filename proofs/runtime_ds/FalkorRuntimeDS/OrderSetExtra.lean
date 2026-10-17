import FalkorRuntimeDS.OrderSet
/-
# `OrderSet` (`graph/src/runtime/orderset.rs`) — the remaining methods

Same list model as `OrderSet.lean` (the `Vec<T>` *is* the list).

| here | there |
| --- | --- |
| `emptyS`     | `Default::default` (orderset.rs:27) |
| `fromVec`    | `OrderSet::from_vec` (orderset.rs:34) — no deduplication |
| `iterS`      | `OrderSet::iter` (orderset.rs:61) |
| `lenS`       | `OrderSet::len` (orderset.rs:66) |
| `isEmptyS`   | `OrderSet::is_empty` (orderset.rs:71) |
| `clearS`     | `OrderSet::clear` (orderset.rs:96) |
| `getS`       | `OrderSet::get` (orderset.rs:113) — `Vec::get` |
| `indexS`     | `Index<usize>` (orderset.rs:132) — `.error` = the `expect` panic |
| `intoIterS`  | `IntoIterator` (orderset.rs:144) |
-/
set_option linter.unusedSectionVars false
namespace FalkorRuntimeDS.OrderSetModel

variable {α : Type}

/-- orderset.rs:27 -/
def emptyS : List α := []
/-- orderset.rs:34 -/
def fromVec (l : List α) : List α := l
/-- orderset.rs:61 -/
def iterS (l : List α) : List α := l
/-- orderset.rs:66 -/
def lenS (l : List α) : Nat := l.length
/-- orderset.rs:71 -/
def isEmptyS (l : List α) : Bool := l.isEmpty
/-- orderset.rs:96 -/
def clearS (_l : List α) : List α := []
/-- orderset.rs:113 -/
def getS (l : List α) (i : Nat) : Option α := l[i]?
/-- orderset.rs:132 — `self.vec.get(index).expect("no entry found for key")`. -/
def indexS (l : List α) (i : Nat) : Except String α :=
  match l[i]? with
  | some x => .ok x
  | none => .error "no entry found for key"
/-- orderset.rs:144 -/
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

/-- `from_vec` is the identity; on a duplicate-free vector it agrees with `from_iter`. -/
theorem fromVec_eq (l : List α) : fromVec l = l := rfl
theorem fromVec_eq_fromListS (l : List α) (h : Nodup l) : fromVec l = fromListS l := by
  unfold fromVec fromListS
  rw [foldl_insertS_unique [] l (by simpa using h)]
  simp

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
