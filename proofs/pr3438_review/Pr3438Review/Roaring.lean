/-
# The `RoaringTreemap` operations PR #3438 relies on, and sorted-list facts

`RoaringTreemap` is the `roaring` crate (0.11.5, `Cargo.lock`): a **Rust
dependency, not FFI**. It has no Lean model in `proofs/`, so what the PR uses of
it is stated here as a hypothesis structure, `RoaringContract`, read off the
crate's documented semantics (`treemap/inherent.rs:504-563`,
`treemap/iter.rs:133-180`):

| field | crate | stated as |
| --- | --- | --- |
| `elems` | `iter()` | the set, ascending, each `< 2^64` |
| `rank`  | `rank(x)` — "the number of integers that are <= value" | `(elems.filter (· ≤ x)).length` |
| `select`| `select(n)` — "the `n`th integer in the set or `None` if `n >= len()`" | `elems[n]?` |
| `iterFrom` | `let mut it = iter(); it.advance_to(a);` then `it.next()`… — "advance the iterator to the first position where the item has a value >= `n`" | `elems.filter (a ≤ ·)` |

Every theorem that takes a `RoaringContract` is therefore **CONDITIONAL** on it
(an undischarged premise about other Rust code). `listContract` shows the
contract is satisfiable (not vacuous). It was also checked against the real
crate by `lean3438_roaring_contract` (scratch copy of the PR,
`attribute_store.rs` tests: 3000 random sets spanning four 2^32 high keys
including `u64::MAX`, every `rank`, `advance_to` and `select` compared with
the definitions above — all pass).

main's walk uses `deleted.contains(id)`; it is `decide (id ∈ elems r)`.
-/
namespace Pr3438

structure RoaringContract (R : Type) where
  elems : R → List Nat
  rank : R → Nat → Nat
  select : R → Nat → Option Nat
  iterFrom : R → Nat → List Nat
  sorted : ∀ r, (elems r).Pairwise (· < ·)
  bound : ∀ r, ∀ x ∈ elems r, x < 2 ^ 64
  rank_spec : ∀ r x, rank r x = ((elems r).filter (fun d => decide (d ≤ x))).length
  select_spec : ∀ r n, select r n = (elems r)[n]?
  iterFrom_spec : ∀ r a, iterFrom r a = (elems r).filter (fun d => decide (a ≤ d))

/-- A sorted list of `u64`s satisfies the contract: it is not vacuous. -/
def SortedSet : Type := { l : List Nat // l.Pairwise (· < ·) ∧ ∀ x ∈ l, x < 2 ^ 64 }

def listContract : RoaringContract SortedSet where
  elems r := r.1
  rank r x := (r.1.filter (fun d => decide (d ≤ x))).length
  select r n := r.1[n]?
  iterFrom r a := r.1.filter (fun d => decide (a ≤ d))
  sorted r := r.2.1
  bound r := r.2.2
  rank_spec _ _ := rfl
  select_spec _ _ := rfl
  iterFrom_spec _ _ := rfl

/-! ## Sorted-list facts -/

/-- One step of the deleted-id cursor: the deleted ids `≥ id` are `id` followed
by those `≥ id+1` when `id` is deleted, and just those `≥ id+1` otherwise. -/
theorem filter_ge_step : ∀ (l : List Nat), l.Pairwise (· < ·) → ∀ id : Nat,
    l.filter (fun d => decide (id ≤ d)) =
      if id ∈ l then id :: l.filter (fun d => decide (id + 1 ≤ d))
      else l.filter (fun d => decide (id + 1 ≤ d))
  | [], _, id => by simp
  | d :: t, h, id => by
    have ht : t.Pairwise (· < ·) := (List.pairwise_cons.1 h).2
    have hgt : ∀ x ∈ t, d < x := (List.pairwise_cons.1 h).1
    have ih := filter_ge_step t ht id
    by_cases hlt : d < id
    · have h1 : ¬ id ≤ d := by omega
      have h2 : ¬ id + 1 ≤ d := by omega
      have hne : id ≠ d := by omega
      simp only [List.filter_cons, h1, h2, decide_false, Bool.false_eq_true, ite_false,
        List.mem_cons, hne, false_or]
      exact ih
    · by_cases heq : d = id
      · subst heq
        have hall1 : t.filter (fun x => decide (d ≤ x)) = t :=
          List.filter_eq_self.2 (fun x hx => by have := hgt x hx; simp; omega)
        have hall2 : t.filter (fun x => decide (d + 1 ≤ x)) = t :=
          List.filter_eq_self.2 (fun x hx => by have := hgt x hx; simp; omega)
        simp [hall1, hall2]
      · have hd : id < d := by omega
        have hnot : id ∉ d :: t := by
          intro hm; simp only [List.mem_cons] at hm
          rcases hm with hm | hm
          · omega
          · have := hgt id hm; omega
        have hall1 : t.filter (fun x => decide (id ≤ x)) = t :=
          List.filter_eq_self.2 (fun x hx => by have := hgt x hx; simp; omega)
        have hall2 : t.filter (fun x => decide (id + 1 ≤ x)) = t :=
          List.filter_eq_self.2 (fun x hx => by have := hgt x hx; simp; omega)
        have h1 : id ≤ d := by omega
        have h2 : id + 1 ≤ d := by omega
        simp [h1, h2, hall1, hall2, hnot]

/-- Seeking to the `n`-th element of a sorted list leaves exactly its suffix
from `n`: `advance_to(select(n))` ≡ `skip(n)`. -/
theorem filter_ge_select : ∀ (l : List Nat), l.Pairwise (· < ·) → ∀ (n a : Nat), l[n]? = some a →
    l.filter (fun d => decide (a ≤ d)) = l.drop n
  | [], _, n, a, h => by simp at h
  | d :: t, h, 0, a, hs => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at hs
    subst hs
    have hgt : ∀ x ∈ t, d < x := (List.pairwise_cons.1 h).1
    have : t.filter (fun x => decide (d ≤ x)) = t :=
      List.filter_eq_self.2 (fun x hx => by have := hgt x hx; simp; omega)
    simp [this]
  | d :: t, h, n + 1, a, hs => by
    simp only [List.getElem?_cons_succ] at hs
    have ht : t.Pairwise (· < ·) := (List.pairwise_cons.1 h).2
    have hgt : ∀ x ∈ t, d < x := (List.pairwise_cons.1 h).1
    have ha : a ∈ t := List.mem_of_getElem? hs
    have hda : d < a := hgt a ha
    have : ¬ a ≤ d := by omega
    simp only [List.filter_cons, this, decide_false, Bool.false_eq_true, ite_false, List.drop_succ_cons]
    exact filter_ge_select t ht n a hs

/-- Every element of `l.filter (a ≤ ·)` is `≥ a`, so its head is not below `a`. -/
theorem head_filter_ge (l : List Nat) (a : Nat) :
    ∀ x, (l.filter (fun d => decide (a ≤ d))).head? = some x → a ≤ x := by
  intro x hx
  have hm := List.mem_of_mem_head? hx
  simp only [List.mem_filter, decide_eq_true_eq] at hm
  exact hm.2

end Pr3438
