/-!
# Finite sums over lists (no Mathlib)

`lsum l f` is `Σ_{a ∈ l} f a` in `Int`. The GraphBLAS `PLUS` monoid over a range of inner
indices, the host passes' running sums and the weight totals are all stated with it. The lemmas
are the ones the equalities need: linearity, reindexing by a permutation, swapping two sums,
and collapsing an indicator sum over `range n` to a sum over a sub-list.
-/

namespace CchCoarsen

/-- `Σ_{a ∈ l} f a` over `Int`. -/
def lsum {α : Type} : List α → (α → Int) → Int
  | [], _ => 0
  | a :: l, f => f a + lsum l f

@[simp] theorem lsum_nil {α : Type} (f : α → Int) : lsum [] f = 0 := rfl
@[simp] theorem lsum_cons {α : Type} (a : α) (l : List α) (f : α → Int) :
    lsum (a :: l) f = f a + lsum l f := rfl

theorem lsum_append {α : Type} (l₁ l₂ : List α) (f : α → Int) :
    lsum (l₁ ++ l₂) f = lsum l₁ f + lsum l₂ f := by
  induction l₁ with
  | nil => simp
  | cons a l ih => simp [ih, Int.add_assoc]

theorem lsum_congr {α : Type} {l : List α} {f g : α → Int} (h : ∀ a ∈ l, f a = g a) :
    lsum l f = lsum l g := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    simp only [lsum_cons]
    rw [h a (by simp), ih (fun x hx => h x (by simp [hx]))]

theorem lsum_zero {α : Type} {l : List α} {f : α → Int} (h : ∀ a ∈ l, f a = 0) : lsum l f = 0 := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    simp only [lsum_cons]
    rw [h a (by simp), ih (fun x hx => h x (by simp [hx]))]; rfl

theorem lsum_add {α : Type} (l : List α) (f g : α → Int) :
    lsum l (fun a => f a + g a) = lsum l f + lsum l g := by
  induction l with
  | nil => simp
  | cons a l ih => simp only [lsum_cons, ih]; omega

theorem lsum_mul_left {α : Type} (l : List α) (c : Int) (f : α → Int) :
    lsum l (fun a => c * f a) = c * lsum l f := by
  induction l with
  | nil => simp
  | cons a l ih => simp only [lsum_cons, ih, Int.mul_add]

theorem lsum_mul_right {α : Type} (l : List α) (c : Int) (f : α → Int) :
    lsum l (fun a => f a * c) = lsum l f * c := by
  induction l with
  | nil => simp
  | cons a l ih => simp only [lsum_cons, ih, Int.add_mul]

/-- A sum over a permuted list is the same sum (`sort_unstable` reorders, sums do not care). -/
theorem lsum_perm {α : Type} {l₁ l₂ : List α} (h : l₁.Perm l₂) (f : α → Int) :
    lsum l₁ f = lsum l₂ f := by
  induction h with
  | nil => rfl
  | cons x _ ih => simp [ih]
  | swap x y l => simp only [lsum_cons]; omega
  | trans _ _ ih₁ ih₂ => rw [ih₁, ih₂]

theorem lsum_swap {α β : Type} (l₁ : List α) (l₂ : List β) (f : α → β → Int) :
    lsum l₁ (fun a => lsum l₂ (fun b => f a b)) = lsum l₂ (fun b => lsum l₁ (fun a => f a b)) := by
  induction l₁ with
  | nil => simp [lsum_zero]
  | cons a l ih =>
    simp only [lsum_cons, ih]
    rw [← lsum_add]

theorem lsum_filter {α : Type} (l : List α) (p : α → Bool) (f : α → Int) :
    lsum (l.filter p) f = lsum l (fun a => if p a then f a else 0) := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    by_cases hp : p a <;> simp [hp, ih]

theorem lsum_flatMap {α β : Type} (l : List α) (g : α → List β) (f : β → Int) :
    lsum (l.flatMap g) f = lsum l (fun a => lsum (g a) f) := by
  induction l with
  | nil => rfl
  | cons a l ih => simp [List.flatMap_cons, lsum_append, ih]

theorem lsum_filterMap {α β : Type} (l : List α) (g : α → Option β) (f : β → Int) :
    lsum (l.filterMap g) f = lsum l (fun a => match g a with | some b => f b | none => 0) := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    cases hg : g a <;> simp [hg, ih]

theorem lsum_map {α β : Type} (l : List α) (g : α → β) (f : β → Int) :
    lsum (l.map g) f = lsum l (fun a => f (g a)) := by
  induction l with
  | nil => rfl
  | cons a l ih => simp [ih]

theorem lsum_nonneg {α : Type} {l : List α} {f : α → Int} (h : ∀ a ∈ l, 0 ≤ f a) : 0 ≤ lsum l f := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [lsum_cons]
    have := h a (by simp); have := ih (fun x hx => h x (by simp [hx])); omega

theorem lsum_le {α : Type} {l : List α} {f g : α → Int} (h : ∀ a ∈ l, f a ≤ g a) :
    lsum l f ≤ lsum l g := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [lsum_cons]
    have := h a (by simp); have := ih (fun x hx => h x (by simp [hx])); omega

/-- The sum of an indicator that is nonzero at a single member of a duplicate-free list. -/
theorem lsum_single {α : Type} [DecidableEq α] {l : List α} (hl : l.Nodup) {a : α} (ha : a ∈ l)
    (f : α → Int) : lsum l (fun t => if t = a then f t else 0) = f a := by
  induction l with
  | nil => simp at ha
  | cons b l ih =>
    rw [List.nodup_cons] at hl
    simp only [lsum_cons]
    by_cases hb : b = a
    · subst hb
      rw [lsum_zero (fun t ht => by
        have : t ≠ b := fun h => hl.1 (h ▸ ht)
        simp [this])]
      simp
    · have ha' : a ∈ l := by
        rcases List.mem_cons.1 ha with h | h
        · exact absurd h.symm hb
        · exact h
      simp [hb, ih hl.2 ha']

/-- The same, for an element absent from the list: the indicator sums to zero. -/
theorem lsum_single_absent {α : Type} [DecidableEq α] {l : List α} {a : α} (ha : a ∉ l)
    (f : α → Int) : lsum l (fun t => if t = a then f t else 0) = 0 :=
  lsum_zero (fun t ht => by
    have : t ≠ a := fun h => ha (h ▸ ht)
    simp [this])

/-- `Σ_{t < n} [t ∈ l] f t = Σ_{t ∈ l} f t` for a duplicate-free `l` inside `range n`: an
indicator sum over the inner dimension is the sum over the stored indices. -/
theorem lsum_range_indicator {l : List Nat} (hl : l.Nodup) {n : Nat} (hn : ∀ x ∈ l, x < n)
    (f : Nat → Int) : lsum (List.range n) (fun t => if t ∈ l then f t else 0) = lsum l f := by
  induction l with
  | nil => simp [lsum_zero]
  | cons a l ih =>
    rw [List.nodup_cons] at hl
    have hsplit : ∀ t, (if t ∈ a :: l then f t else 0) =
        (if t = a then f t else 0) + (if t ∈ l then f t else 0) := by
      intro t
      by_cases h1 : t = a
      · subst h1; simp [hl.1]
      · simp [h1]
    rw [lsum_congr (fun t _ => hsplit t), lsum_add,
      lsum_single List.nodup_range (List.mem_range.2 (hn a (by simp))),
      ih hl.2 (fun x hx => hn x (by simp [hx]))]
    simp

end CchCoarsen
