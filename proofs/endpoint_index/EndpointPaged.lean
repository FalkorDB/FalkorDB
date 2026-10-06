import EndpointPaged.Arc
/-!
# `Paged<T>` and its `Arc` pages: layout, page-level writes, copy-on-write isolation

Companion to `FalkorEndpointIndex` (which treats a tier as its flat slot list and
proves the page arithmetic of `Paged::at` / `ensure_pages`). This file closes the rest
of `graph/src/graph/endpoint_index.rs` (origin/main `3fec7d7c9`, worktree):

| here | there |
| --- | --- |
| `u24Put`/`u24Get`, `get_put_contract` | `impl Endpoint for U24/u16/u32/u64` (:81-124) and the trait decls `Endpoint::get/put` (:64-65) |
| `PG`, `PG.view`, `PG.WF`          | `struct Paged<T> { pages, len }` (:150-153); `WF` = every page but the last full, `len` within the pages, slack past `len` all `EMPTY` |
| `PG.dflt`, `PG.isEmpty`, `PG.len` | `Paged::default` (:165), `is_empty` (:187), `len` (:183, `TierOps::len` :758) |
| `emptyPage`                       | `empty_page` (:178) |
| `PG.at`, `PG.iter`                | `Paged::at` (:192), `iter` (:199) |
| `allocBytes`, `usedBytes`, `sz`   | `allocated_bytes` (:204), `used_bytes` (:213), `size_of::<(T, T)>()` |
| `fromSlots`                       | `Paged::from_slots` (:234) |
| `promoteSlots`                    | the iterator `promote` (:336) hands `from_slots` |
| `pagePut`, `PG.put`, `PG.vacate`  | `TierOps::put` (:741) / `vacate` (:750): `page_mut(at)[at % PAGE_SLOTS] = …` |
| `PG.growTo`                       | `TierOps::grow_to` (:732): `ensure_pages(len)` then `self.len = len` |
| `reserveCap`                      | `TierOps::reserve_exact` (:761) |
| `Heap`, `pageMut`, `cloneV`, `dropV`, `pushFresh`, `replace` | `Arc` strong counts: `page_mut` (:293, `Arc::get_mut`-or-copy), `Paged::clone` (:156, clones page pointers), drop, `empty_page`/`from_slots` (fresh `Arc::from`), `regrow` (:218, `*page = Arc::from(next)`); the same shape is `Arc::make_mut` in `with_tier` / `raise_to` (tier granularity) |
| `bytesOf`, `memUsage`             | `EndpointIndex::memory_usage` (:427), `bytes_per_edge` (:437) |

Everything is proved without `sorry`/`axiom`. Not modelled: weak counts (no `Weak` to a
page or tier is ever created, so `get_mut`'s "strong = 1 and weak = 0" is "strong = 1"),
the actual allocator, and the `f64` division at the end of `bytes_per_edge` (the
numerator and denominator are proved; the quotient is IEEE).
-/
namespace Falkor.EndpointIndex.Paged
open Falkor.EndpointIndex

/-! ## `set` only writes inside the tier it addresses

`TierOps::put` does not bounds-check against `len` (it indexes the page and would panic,
or write slack). The flat model's `List.set` is a no-op out of range, so this closes the
gap: every arm of `EndpointIndex::set` passes an offset that is in range. -/

theorem set_inplace_in_range (ix : Ix) {e r at_ : Nat} (h : ix.locate e = some (r, at_)) :
    at_ < ((ix.tier r).getD []).length := by
  obtain ⟨_, hat, _⟩ := locate_some ix h
  simp only [Ix.tierLen] at hat
  cases hx : ix.tier r <;> simp_all

theorem set_promote_in_range (ix : Ix) (hwf : WF ix) {e r at_ n : Nat}
    (h : ix.locate e = some (r, at_)) (hgt : r < n) (hn3 : n ≤ 3) :
    e < (((ix.raiseTo n).tier n).getD []).length := by
  obtain ⟨_, hat, he⟩ := locate_some ix h
  obtain ⟨hvw, _, hpre, hsuf⟩ := raiseTo_spec ix (by omega) hn3 hwf
  have hsp := splice (ix.raiseTo n) hn3
  rw [hpre, hsuf, hvw] at hsp
  have h1 := pre_mono ix hgt hn3
  have h2 := len_splice ix hn3
  have h3 := congrArg List.length hsp
  rw [← len_view] at h3
  simp [viewT] at h3
  omega

theorem set_append_in_range (r at_ : Nat) (v : List Slot) : at_ < (growTo r (at_ + 1) v).length := by
  unfold growTo; split <;> (try simp) <;> omega

#print axioms pageMut_spec
#print axioms growTo_spec
#print axioms fromSlots_spec
#print axioms flatten_pagePut

end Falkor.EndpointIndex.Paged
