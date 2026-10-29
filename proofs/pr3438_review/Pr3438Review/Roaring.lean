/-
# `RoaringTreemap`: the owners' abstraction, and `rank`/`select`/`advance_to` on it

The owners model a `RoaringTreemap` abstractly and bucket their rows PROVEN on
that abstraction: graph_persist as the sorted list its `iter()` yields
(`G.delNodes`/`G.delEdges`, read by `encRoar` as `iter().skip(offset).take(count)`),
pending_commit as membership (`deleted : Nat → Bool` = `contains`). `RSet` is
that abstraction, defined once here (it mirrors graph_persist's deleted-id list:
ascending, distinct, `< 2^64`; `contains` is `decide (x ∈ elems)`, pending_commit's
predicate). The three operations the PR adds are **definitions** on it, from the
crate's documented semantics (roaring 0.11.5 `treemap/inherent.rs:504-563`,
`treemap/iter.rs:133-180`):

| here | crate |
| --- | --- |
| `RSet.elems` | `iter()` — ascending |
| `RSet.rank x` = `|{d ∈ s | d ≤ x}|` | `rank(x)`: "the number of integers that are <= value" |
| `RSet.select n` = the `n`-th smallest | `select(n)`: "the `n`th integer in the set or `None` if `n >= len()`" |
| `RSet.iterFrom a` = the elements `≥ a`, ascending | `let mut it = iter(); it.advance_to(a);` then `next()`…: "the first position where the item has a value >= `n`" |

**Trust.** The roaring crate is trusted to implement this abstraction, at the
same level as the owners' `contains`/`iter`: no hypothesis is named, and rows
resting on it are PROVEN. The check against the real crate is the randomized
scratch test `lean3438_roaring_contract` (a copy of the PR, `attribute_store.rs`
tests): 3000 random sets spanning four 2^32 high keys up to `u64::MAX`; every
`iter` is strictly ascending and every `rank`, `advance_to` and `select` equals
the definitions above — all pass.
-/
namespace Pr3438

/-- A `RoaringTreemap`: its ascending, distinct `u64` elements. -/
structure RSet where
  elems : List Nat
  sorted : elems.Pairwise (· < ·)
  bound : ∀ x ∈ elems, x < 2 ^ 64

namespace RSet
/-- `rank(x)`. -/
def rank (r : RSet) (x : Nat) : Nat := (r.elems.filter (fun d => decide (d ≤ x))).length
/-- `select(n)`. -/
def select (r : RSet) (n : Nat) : Option Nat := r.elems[n]?
/-- `iter()` after `advance_to(a)`. -/
def iterFrom (r : RSet) (a : Nat) : List Nat := r.elems.filter (fun d => decide (a ≤ d))

theorem rank_spec (r : RSet) (x : Nat) : r.rank x = (r.elems.filter (fun d => decide (d ≤ x))).length := rfl
theorem select_spec (r : RSet) (n : Nat) : r.select n = r.elems[n]? := rfl
theorem iterFrom_spec (r : RSet) (a : Nat) : r.iterFrom a = r.elems.filter (fun d => decide (a ≤ d)) := rfl
end RSet

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
