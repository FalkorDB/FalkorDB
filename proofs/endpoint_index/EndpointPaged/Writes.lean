import EndpointPaged.Layout
/-! # Page-level writes, `grow_to`, `reserve_exact` -/
namespace Falkor.EndpointIndex.Paged
open Falkor.EndpointIndex

variable {α : Type}

/-! ## Page-level write addressing -/

/-- `page_mut(at)[at % P] = x`: overwrite slot `at % P` of page `at / P`. -/
def pagePut (P : Nat) (pages : List (List α)) (at_ : Nat) (x : α) : List (List α) :=
  pages.modify (at_ / P) (fun pg => pg.set (at_ % P) x)

theorem length_pagePut (P : Nat) (pages : List (List α)) (at_ : Nat) (x : α) :
    ((pagePut P pages at_ x).map List.length) = pages.map List.length := by
  apply List.ext_getElem
  · simp [pagePut]
  · intro i h1 h2
    simp only [pagePut, List.getElem_map, List.getElem_modify]
    split <;> simp

theorem shape_of_lengths (P : Nat) :
    ∀ (ps qs : List (List α)), ps.map List.length = qs.map List.length → Shape P ps → Shape P qs
  | [], [], _, h => h
  | [p], [q], e, h => by simp at e; simp only [Shape] at *; omega
  | p :: p' :: ps, q :: q' :: qs, e, h => by
    simp only [List.map_cons, List.cons.injEq] at e
    simp only [Shape] at *
    exact ⟨by omega, shape_of_lengths P (p' :: ps) (q' :: qs) (by simp [e.2.1, e.2.2]) h.2⟩
  | [], _ :: _, e, _ => by simp at e
  | _ :: _, [], e, _ => by simp at e
  | [_], _ :: _ :: _, e, _ => by simp at e
  | _ :: _ :: _, [_], e, _ => by simp at e

/-- **Page-level addressing.** Under `Shape`, writing slot `at % P` of page `at / P` is
writing index `at` of the flat list — `TierOps::put` / `vacate` are the flat `putAt` /
`vacate` the main model uses. -/
theorem flatten_pagePut (P : Nat) (hP : 0 < P) :
    ∀ (pages : List (List α)), Shape P pages → ∀ at_ x, at_ < pages.flatten.length →
    (pagePut P pages at_ x).flatten = pages.flatten.set at_ x := by
  intro pages
  induction pages with
  | nil => intro _ _ _ h; simp at h
  | cons p ps ih =>
    intro hs at_ x hi
    simp only [pagePut, List.flatten_cons] at hi ⊢
    cases ps with
    | nil =>
      simp only [Shape] at hs
      simp only [List.flatten_nil, List.append_nil] at hi ⊢
      rw [Nat.div_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)]
      simp
    | cons q qs =>
      simp only [Shape] at hs
      obtain ⟨hp, hs'⟩ := hs
      by_cases hiP : at_ < P
      · rw [Nat.div_eq_of_lt hiP, Nat.mod_eq_of_lt hiP, List.set_append, if_pos (by omega)]
        simp
      · have hd : at_ / P = (at_ - P) / P + 1 := by
          rw [Nat.div_eq_sub_div hP (by omega)]
        have hm : at_ % P = (at_ - P) % P := Nat.mod_eq_sub_mod (by omega)
        rw [hd, List.modify_succ_cons, List.flatten_cons, List.set_append, if_neg (by omega), hp, hm]
        have := ih hs' (at_ - P) x (by simp [List.length_append] at hi ⊢; omega)
        simp only [pagePut] at this
        rw [this]

namespace PG
/-- `TierOps::put` (:741): no bounds check of its own; callers pass `at < len`. -/
def put (P : Nat) (g : PG α) (at_ : Nat) (x : α) : PG α := ⟨pagePut P g.pages at_ x, g.len⟩
/-- `TierOps::vacate` (:750). -/
def vacate (P : Nat) (fill : α) (g : PG α) (at_ : Nat) : PG α :=
  if at_ < g.len then g.put P at_ fill else g
end PG

theorem put_spec (P : Nat) (hP : 0 < P) (fill : α) (g : PG α) (hw : g.WF P fill) (at_ : Nat) (x : α)
    (hat : at_ < g.len) :
    (g.put P at_ x).WF P fill ∧ (g.put P at_ x).view = g.view.set at_ x := by
  obtain ⟨hs, hl, hd⟩ := hw
  have hf := flatten_pagePut P hP g.pages hs at_ x (by omega)
  refine ⟨⟨shape_of_lengths P _ _ (length_pagePut P g.pages at_ x).symm hs, ?_, ?_⟩, ?_⟩
  · simp only [PG.put, hf, List.length_set]; omega
  · simp only [PG.put, hf, List.length_set]
    rw [List.drop_set_of_lt hat]; exact hd
  · simp [PG.view, PG.put, hf, List.take_set]

theorem vacate_spec (P : Nat) (hP : 0 < P) (fill : α) (g : PG α) (hw : g.WF P fill) (at_ : Nat) :
    (g.vacate P fill at_).WF P fill ∧
    (g.vacate P fill at_).view = (if at_ < g.view.length then g.view.set at_ fill else g.view) := by
  have hlv := length_view g hw.2.1
  simp only [PG.length] at hlv
  unfold PG.vacate
  split
  · rename_i h; rw [if_pos (by omega)]; exact put_spec P hP fill g hw at_ fill h
  · rename_i h; rw [if_neg (by omega)]; exact ⟨hw, rfl⟩

/-! ## `grow_to`: `ensure_pages` then `len` -/

namespace PG
/-- `TierOps::grow_to` (:732). -/
def growTo (P : Nat) (fill : α) (g : PG α) (n : Nat) : PG α :=
  if n > g.len then ⟨ensurePages P fill g.pages n, n⟩ else g
end PG

/-- **`grow_to` is the flat `growTo`.** The slots it exposes read `EMPTY`: whether they
come from new pages, a regrown tail, or slack an earlier doubling left past `len`. -/
theorem growTo_spec (P : Nat) (hP : 0 < P) (fill : α) (g : PG α) (hw : g.WF P fill) (n : Nat) :
    (g.growTo P fill n).WF P fill ∧
    (g.growTo P fill n).view =
      (if n > g.view.length then g.view ++ List.replicate (n - g.view.length) fill else g.view) := by
  have hlv := length_view g hw.2.1
  simp only [PG.length] at hlv
  obtain ⟨hs, hl, hd⟩ := hw
  unfold PG.growTo
  split
  · rename_i hn
    rw [if_pos (by omega)]
    obtain ⟨hs', ⟨k, hk⟩, hcov⟩ := ensurePages_spec P hP fill g.pages n hs
    have hsplit := (List.take_append_drop g.len g.pages.flatten).symm
    rw [hd] at hsplit
    have hall : (ensurePages P fill g.pages n).flatten =
        g.view ++ List.replicate (g.pages.flatten.length - g.len + k) fill := by
      rw [hk]
      conv => lhs; rw [hsplit]
      simp only [PG.view, List.append_assoc, List.replicate_append_replicate]
    have hcov' := hcov
    rw [hall, List.length_append, List.length_replicate, hlv] at hcov'
    refine ⟨⟨hs', hcov, ?_⟩, ?_⟩
    · show List.drop n (ensurePages P fill g.pages n).flatten =
        List.replicate ((ensurePages P fill g.pages n).flatten.length - n) fill
      rw [hall, List.drop_append, List.drop_eq_nil_of_le (by omega), List.drop_replicate,
        List.nil_append, List.length_append, List.length_replicate, hlv]
      congr 1; omega
    · show List.take n (ensurePages P fill g.pages n).flatten = _
      rw [hall, List.take_append, List.take_of_length_le (by omega), List.take_replicate, hlv]
      congr 2; omega
  · rename_i hn
    rw [if_neg (by omega)]
    exact ⟨⟨hs, hl, hd⟩, rfl⟩
/-! ## `reserve_exact` (:761) -/

/-- `pages.reserve_exact((len + extra).div_ceil(P).saturating_sub(pages.len()))`, with
`Vec::reserve_exact`'s guarantee `capacity ≥ pages.len() + additional`. -/
def reserveCap (P cap pagesLen len extra : Nat) : Nat :=
  max cap (pagesLen + ((len + extra + P - 1) / P - pagesLen))

/-- After `reserve_exact(extra)` the page vector can hold every page `len + extra` slots
need, and the saturating subtraction never asks for a wrapped-around capacity (it is 0
when enough pages already exist). The pages and `len` themselves are untouched. -/
theorem reserve_covers (P cap pagesLen len extra : Nat) :
    (len + extra + P - 1) / P ≤ reserveCap P cap pagesLen len extra ∧
    ((len + extra + P - 1) / P ≤ pagesLen → (len + extra + P - 1) / P - pagesLen = 0) := by
  simp only [reserveCap]; omega

end Falkor.EndpointIndex.Paged
