import FalkorEndpointIndex
/-! # Field types, `Paged<T>` shape, memory accounting, `from_slots` -/
namespace Falkor.EndpointIndex.Paged
open Falkor.EndpointIndex

/-! ## `Endpoint` impls and the trait contract -/

/-- `U24::put` (:98): the three low little-endian bytes of `v`. -/
def u24Put (v : Nat) : Nat × Nat × Nat := (v % 256, v / 256 % 256, v / 65536 % 256)
/-- `U24::get` (:95): `u32::from_le_bytes([b0, b1, b2, 0])`. -/
def u24Get (b : Nat × Nat × Nat) : Nat := b.1 + 256 * b.2.1 + 65536 * b.2.2

/-- The byte layout of `U24` is exactly the truncating cast the flat model uses (`putF 1`). -/
theorem u24_get_put (v : Nat) : u24Get (u24Put v) = putF 1 v := by
  simp only [u24Get, u24Put, putF, bits]
  omega

/-- **Trait contract** (`Endpoint::get`/`put`, :64-65, every impl): `get (put v)` is `v`
truncated to the width, which is `v` itself whenever `v` fits, and `EMPTY` (= `CEILING`)
is the width's all-ones value, so it always fits. -/
theorem get_put_contract (r v : Nat) :
    putF r v < 2 ^ bits r ∧ (v < 2 ^ bits r → putF r v = v) ∧ EMPTY r + 1 = 2 ^ bits r := by
  refine ⟨Nat.mod_lt _ (Nat.two_pow_pos _), fun h => Nat.mod_eq_of_lt h, ?_⟩
  match r with
  | 0 | 1 | 2 => decide
  | _ + 3 => simp [EMPTY, bits]

/-! ## `Paged<T>` -/

variable {α : Type}

structure PG (α : Type) where
  pages : List (List α)
  len : Nat

namespace PG

/-- The slots a tier holds: the first `len` of the concatenated pages. -/
def view (g : PG α) : List α := g.pages.flatten.take g.len

/-- Page shape, `len` within the pages, and the slack past `len` all `fill` (`EMPTY`). -/
def WF (P : Nat) (fill : α) (g : PG α) : Prop :=
  Shape P g.pages ∧ g.len ≤ g.pages.flatten.length ∧
  g.pages.flatten.drop g.len = List.replicate (g.pages.flatten.length - g.len) fill

/-- `Paged::default` (:165). -/
def dflt : PG α := ⟨[], 0⟩
/-- `Paged::len` (:183) and `TierOps::len` (:758). -/
def length (g : PG α) : Nat := g.len
/-- `Paged::is_empty` (:187). -/
def isEmpty (g : PG α) : Bool := g.len == 0
/-- `Paged::at` (:192): `pages[at / P][at % P]` (`none` = the index would panic). -/
def at_ (P : Nat) (g : PG α) (i : Nat) : Option α := (g.pages[i / P]?).bind (·[i % P]?)
/-- `Paged::iter` (:199): `(0..len).map(at)`. -/
def iter (P : Nat) (g : PG α) : List (Option α) := (List.range g.len).map (g.at_ P)

end PG

theorem dflt_wf (P : Nat) (fill : α) : (PG.dflt : PG α).WF P fill ∧ (PG.dflt : PG α).view = [] := by
  simp [PG.WF, PG.dflt, PG.view, Shape]

theorem length_view (g : PG α) (h : g.len ≤ g.pages.flatten.length) : g.view.length = g.length := by
  simp only [PG.view, PG.length, List.length_take]; omega

theorem isEmpty_iff (g : PG α) (h : g.len ≤ g.pages.flatten.length) :
    g.isEmpty = true ↔ g.view = [] := by
  have := length_view g h
  simp only [PG.isEmpty, PG.length, beq_iff_eq] at *
  rw [← List.length_eq_zero_iff, this]

/-- `empty_page` (:178): `slots` copies of `(EMPTY, EMPTY)`. -/
def emptyPage (fill : α) (slots : Nat) : List α := List.replicate slots fill

theorem emptyPage_spec (fill : α) (n : Nat) :
    (emptyPage fill n).length = n ∧ ∀ x ∈ emptyPage fill n, x = fill := by
  exact ⟨by simp [emptyPage], fun x hx => List.eq_of_mem_replicate hx⟩

/-- `Paged::iter` yields exactly the slots of the flat view (every index in range). -/
theorem iter_eq (P : Nat) (hP : 0 < P) (fill : α) (g : PG α) (hw : g.WF P fill) :
    g.iter P = g.view.map some := by
  obtain ⟨hs, hl, _⟩ := hw
  apply List.ext_getElem
  · simp only [PG.iter, PG.view, List.length_map, List.length_range, List.length_take]; omega
  · intro i h1 h2
    simp only [PG.iter, List.getElem_map, List.getElem_range, PG.at_]
    have hi : i < g.pages.flatten.length := by simp [PG.iter] at h1; omega
    have := paged_at P hP g.pages hs i hi
    rw [← this]
    simp [PG.view, List.getElem?_eq_getElem hi]

/-! ## Memory accounting -/

/-- `allocated_bytes` (:204): page lengths summed, times `size_of::<(T, T)>()`. -/
def allocBytes (sz : Nat) (g : PG α) : Nat := (g.pages.map List.length).sum * sz
/-- `used_bytes` (:213). -/
def usedBytes (sz : Nat) (g : PG α) : Nat := g.len * sz

theorem allocBytes_eq (sz : Nat) (g : PG α) : allocBytes sz g = g.pages.flatten.length * sz := by
  simp [allocBytes, List.length_flatten]

theorem used_le_alloc (sz : Nat) (g : PG α) (h : g.len ≤ g.pages.flatten.length) :
    usedBytes sz g ≤ allocBytes sz g := by
  rw [allocBytes_eq]; exact Nat.mul_le_mul_right _ h

/-- `size_of::<(T, T)>()` for `u16`, `U24` (`[u8; 3]`, alignment 1), `u32`, `u64`. -/
def sz : Nat → Nat
  | 0 => 4
  | 1 => 6
  | 2 => 8
  | _ => 16

/-- The pair is exactly two fields of the width: no padding at any tier. -/
theorem sz_eq (r : Nat) : sz r = 2 * (bits r / 8) := by
  match r with
  | 0 | 1 | 2 => decide
  | _ + 3 => simp [sz, bits]

/-- The numerator `bytes_per_edge` (:437) divides by `len()`: used bytes over the four tiers. -/
def bytesOf (ix : Ix) : Nat := ix.tierLen 0 * sz 0 + ix.tierLen 1 * sz 1 + ix.tierLen 2 * sz 2 + ix.tierLen 3 * sz 3

/-- `bytes_per_edge` is between 4 and 16 bytes per stored slot, whatever the tier mix. -/
theorem bytes_per_edge_bounds (ix : Ix) : 4 * ix.len ≤ bytesOf ix ∧ bytesOf ix ≤ 16 * ix.len := by
  simp only [bytesOf, Ix.len, sz]; omega

/-- A single active tier `r` costs exactly `sz r` per slot (the Rust test's 4.0/6.0/8.0/16.0). -/
theorem bytes_single_tier (ix : Ix) (r : Nat) (hr : r ≤ 3) (h : ∀ k ≤ 3, k ≠ r → ix.tierLen k = 0) :
    bytesOf ix = sz r * ix.len := by
  rcases rank_cases hr with rfl | rfl | rfl | rfl <;>
  · have h0 := h 0; have h1 := h 1; have h2 := h 2; have h3 := h 3
    simp only [bytesOf, Ix.len, sz] at *
    omega

/-- `memory_usage` (:427) over paged tiers: the four `allocated_bytes` summed. -/
def memUsage (t : List (Nat × PG α)) : Nat := (t.map fun (s, g) => allocBytes s g).sum

theorem memUsage_ge_used (t : List (Nat × PG α)) (h : ∀ p ∈ t, p.2.len ≤ p.2.pages.flatten.length) :
    (t.map fun (s, g) => usedBytes s g).sum ≤ memUsage t := by
  induction t with
  | nil => simp [memUsage]
  | cons p t ih =>
    simp only [memUsage, List.map_cons, List.sum_cons] at *
    have := used_le_alloc p.1 p.2 (h p (by simp))
    have := ih (fun q hq => h q (by simp [hq]))
    omega

/-- `EndpointIndex::is_empty` (:421): no slot at all, so every id reads absent. -/
theorem ix_isEmpty (ix : Ix) : ix.len = 0 ↔ view ix = [] := by
  rw [len_view]; exact List.length_eq_zero_iff

theorem ix_isEmpty_get (ix : Ix) (h : ix.len = 0) (i : Nat) : ix.get i = none := by
  rw [get_view, (ix_isEmpty ix).1 h]; simp [lk]

/-! ## `from_slots` -/

/-- `Paged::from_slots(it, len)` (:234): while `left > 0`, take `min(left, P)` items as
one page. -/
def fromSlots (P : Nat) (hP : 0 < P) (it : List α) (left : Nat) : List (List α) :=
  if h : left = 0 then [] else
  it.take (min left P) :: fromSlots P hP (it.drop (min left P)) (left - min left P)
termination_by left
decreasing_by have : 0 < min left P := Nat.lt_min.2 ⟨by omega, hP⟩; omega

/-- **Page layout of `from_slots`.** Given an iterator of at least `len` items, the pages
have the `Shape` every other method relies on (all full but a non-empty last), hold
exactly the first `len` items in order, and number `⌈len / P⌉` — the `with_capacity`. -/
theorem fromSlots_spec (P : Nat) (hP : 0 < P) :
    ∀ (left : Nat) (it : List α), left ≤ it.length →
    Shape P (fromSlots P hP it left) ∧ (fromSlots P hP it left).flatten = it.take left ∧
    (fromSlots P hP it left).length = (left + P - 1) / P ∧ ∀ p ∈ fromSlots P hP it left, p ≠ [] := by
  intro left
  induction left using Nat.strongRecOn with
  | _ left ih =>
    intro it hit
    rw [fromSlots]
    by_cases h0 : left = 0
    · subst h0; simp [Shape]; rw [Nat.div_eq_of_lt (by omega)]
    rw [dif_neg h0]
    have hm : 0 < min left P := Nat.lt_min.2 ⟨by omega, hP⟩
    have hlt : left - min left P < left := by omega
    obtain ⟨hs, hf, hn, hne⟩ := ih _ hlt (it.drop (min left P)) (by simp; omega)
    refine ⟨?_, ?_, ?_, ?_⟩
    · -- shape
      generalize hR : fromSlots P hP (it.drop (min left P)) (left - min left P) = R at *
      cases R with
      | nil =>
        simp only [Shape]; simp; omega
      | cons q qs =>
        simp only [Shape]
        refine ⟨?_, hs⟩
        -- a page that is followed by another is full: left - P > 0
        have hpos : left - min left P ≠ 0 := by
          intro h; rw [h, fromSlots] at hR; simp at hR
        simp; omega
    · simp only [List.flatten_cons, hf]
      rw [← List.take_add]
      congr 1; omega
    · simp only [List.length_cons, hn]
      by_cases hPl : P ≤ left
      · rw [Nat.min_eq_right hPl, show left + P - 1 = (left - P + P - 1) + P by omega,
          Nat.add_div_right _ hP]
      · rw [Nat.min_eq_left (by omega), Nat.sub_self, Nat.zero_add, Nat.div_eq_of_lt (by omega),
          show left + P - 1 = (left - 1) + P by omega, Nat.add_div_right _ hP,
          Nat.div_eq_of_lt (by omega)]
    · intro p hp
      simp only [List.mem_cons] at hp
      rcases hp with rfl | hp
      · intro h
        have h2 : (it.take (min left P)).length = min left P := by
          rw [List.length_take]; omega
        rw [h] at h2; simp at h2; omega
      · exact hne p hp

/-- `Paged { pages: from_slots(it, len), len }` is well formed and its view is the first
`len` items: the debug-asserted "iterator shorter than the stated length" never matters
when the caller passes an iterator of exactly `len` items, as `promote` does. -/
theorem fromSlots_wf (P : Nat) (hP : 0 < P) (fill : α) (it : List α) (len : Nat) (h : len ≤ it.length) :
    (PG.mk (fromSlots P hP it len) len).WF P fill ∧ (PG.mk (fromSlots P hP it len) len).view = it.take len := by
  obtain ⟨hs, hf, _, _⟩ := fromSlots_spec P hP len it h
  refine ⟨⟨hs, by simp [hf]; omega, by simp [hf]; omega⟩, by simp [PG.view, hf, List.take_take]⟩

/-- The iterator `promote` (:336) builds: the widened narrow tier, then every slot of
`into`; `total = old.len() + into.len()` is exactly its length. -/
def promoteSlots (a b : Nat) (old into : List Slot) : List Slot := old.map (widen a b) ++ into

theorem promote_paged (P : Nat) (hP : 0 < P) (a b : Nat) (old into : List Slot) :
    let g : PG Slot := ⟨fromSlots P hP (promoteSlots a b old into) (old.length + into.length),
      old.length + into.length⟩
    g.WF P (EMPTY b, EMPTY b) ∧ g.view = old.map (widen a b) ++ into := by
  have hl : old.length + into.length ≤ (promoteSlots a b old into).length := by simp [promoteSlots]
  obtain ⟨hw, hv⟩ := fromSlots_wf P hP (EMPTY b, EMPTY b) _ _ hl
  refine ⟨hw, ?_⟩
  rw [hv, promoteSlots, List.take_of_length_le (by simp)]

end Falkor.EndpointIndex.Paged
