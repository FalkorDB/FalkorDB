/-
# `<[T]>::binary_search_by_key`, as the Rust std of this toolchain implements it

Every sorted-span lookup in the two target files goes through it:
`SpanRef::get` (attribute_store.rs:856), `merge_span` fast path 1 (:648-667),
`lookup_sorted` (pending.rs:100), `set_node_attribute` /
`set_relationship_attribute` (pending.rs:444, :813, which use the `Err`
insertion point). `bsearch` is the loop of rustc 1.98.1
`core/src/slice/mod.rs:2976-3028` line for line (branchless, no early exit),
over the slice's keys.

* `bsearch_ok_iff` — on strictly sorted keys `Ok i` is returned iff `ks[i] = k`.
* `bsearch_err` — otherwise `Err p` is the insertion point: everything before
  `p` is smaller, everything from `p` on is larger, `p ≤ len`.
-/
namespace PendingCommit.BinSearch

inductive Res where
  | ok (i : Nat)
  | err (i : Nat)
  deriving DecidableEq, Repr, BEq

/-- The `while size > 1` loop; returns the final `base`. -/
def loop (ks : List Nat) (k base size : Nat) : Nat :=
  if h : size > 1 then
    let half := size / 2
    let mid := base + half
    let base' := if ks.getD mid 0 > k then base else mid   -- `cmp == Greater`
    loop ks k base' (size - half)
  else base
termination_by size
decreasing_by omega

/-- `binary_search_by(|e| e.key().cmp(&k))`. -/
def bsearch (ks : List Nat) (k : Nat) : Res :=
  if ks.length = 0 then .err 0
  else
    let base := loop ks k 0 ks.length
    let c := ks.getD base 0
    if c = k then .ok base
    else .err (base + if c < k then 1 else 0)

def SSorted (ks : List Nat) : Prop := List.Pairwise (· < ·) ks

theorem getD_lt {ks : List Nat} (h : SSorted ks) {i j : Nat} (hij : i < j) (hj : j < ks.length) :
    ks.getD i 0 < ks.getD j 0 := by
  have := (List.pairwise_iff_getElem.1 h) i j (by omega) hj hij
  simp only [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (by omega : i < ks.length),
    List.getElem?_eq_getElem hj, Option.getD_some]
  exact this

theorem getD_le {ks : List Nat} (h : SSorted ks) {i j : Nat} (hij : i ≤ j) (hj : j < ks.length) :
    ks.getD i 0 ≤ ks.getD j 0 := by
  rcases Nat.lt_or_eq_of_le hij with h1 | rfl
  · exact Nat.le_of_lt (getD_lt h h1 hj)
  · exact Nat.le_refl _

/-- Loop invariant. -/
structure Inv (ks : List Nat) (k base size : Nat) : Prop where
  pos : 1 ≤ size
  bound : base + size ≤ ks.length
  low : 0 < base → ks.getD base 0 ≤ k
  high : ∀ j, base + size ≤ j → j < ks.length → k < ks.getD j 0

theorem loop_inv (ks : List Nat) (hs : SSorted ks) (k : Nat) :
    ∀ size base, Inv ks k base size → Inv ks k (loop ks k base size) 1 := by
  intro size
  induction size using Nat.strongRecOn with
  | _ size ih =>
    intro base hI
    unfold loop
    by_cases h : size > 1
    · simp only [h, dite_true]
      apply ih (size - size / 2) (by omega)
      by_cases hg : ks.getD (base + size / 2) 0 > k
      · simp only [hg, ite_true]
        refine ⟨by omega, by have := hI.bound; omega, hI.low, ?_⟩
        intro j hj hjn
        have := getD_le hs (i := base + size / 2) (j := j) (by omega) hjn
        omega
      · simp only [hg, ite_false]
        refine ⟨by omega, by have := hI.bound; omega, fun _ => by omega, ?_⟩
        intro j hj hjn; exact hI.high j (by omega) hjn
    · simp only [h, dite_false]
      have : size = 1 := by have := hI.pos; omega
      subst this; exact hI

theorem final_inv (ks : List Nat) (hs : SSorted ks) (k : Nat) (hn : ks.length ≠ 0) :
    Inv ks k (loop ks k 0 ks.length) 1 :=
  loop_inv ks hs k _ 0 ⟨by omega, by omega, fun h => absurd h (by omega), fun j hj hjn => by omega⟩

theorem getD_mem_eq {ks : List Nat} {i : Nat} (hi : i < ks.length) : ks.getD i 0 = ks[i] := by
  simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi]

/-- **`bsearch_ok_iff`**: on strictly sorted keys, `Ok i` exactly when `ks[i] = k`. -/
theorem bsearch_ok_iff (ks : List Nat) (hs : SSorted ks) (k i : Nat) :
    bsearch ks k = .ok i ↔ (i < ks.length ∧ ks.getD i 0 = k) := by
  unfold bsearch
  by_cases hn : ks.length = 0
  · simp [hn]
  · simp only [hn, ite_false]
    have I := final_inv ks hs k hn
    generalize loop ks k 0 ks.length = b at I
    constructor
    · intro h
      split at h
      · cases h; exact ⟨by have := I.bound; omega, by assumption⟩
      · cases h
    · rintro ⟨hi, hk⟩
      -- the key sits exactly at `b`
      have hib : i = b := by
        rcases Nat.lt_trichotomy i b with h1 | h1 | h1
        · have := getD_lt hs h1 (by have := I.bound; omega)
          have := I.low (by omega); omega
        · exact h1
        · have := I.high i (by omega) hi; omega
      subst hib; rw [if_pos hk]

/-- **`bsearch_err`**: `Err p` is the insertion point. -/
theorem bsearch_err (ks : List Nat) (hs : SSorted ks) (k p : Nat) (h : bsearch ks k = .err p) :
    p ≤ ks.length ∧ (∀ j, j < p → ks.getD j 0 < k) ∧
      (∀ j, p ≤ j → j < ks.length → k < ks.getD j 0) := by
  unfold bsearch at h
  by_cases hn : ks.length = 0
  · simp [hn] at h; subst h; refine ⟨by omega, fun j hj => by omega, fun j _ hj => by omega⟩
  · simp only [hn, ite_false] at h
    have I := final_inv ks hs k hn
    generalize loop ks k 0 ks.length = b at I h
    have hb := I.bound
    split at h
    · cases h
    · rename_i hne
      cases h
      by_cases hlt : ks.getD b 0 < k
      · simp only [hlt, ite_true]
        refine ⟨by omega, ?_, fun j hj hjn => I.high j (by omega) hjn⟩
        intro j hj
        rcases Nat.lt_or_eq_of_le (Nat.le_of_lt_succ hj) with h1 | rfl
        · have := getD_lt hs h1 (by omega); omega
        · exact hlt
      · simp only [hlt, ite_false, Nat.add_zero]
        have hgt : k < ks.getD b 0 := by omega
        have hb0 : b = 0 := by
          rcases Nat.eq_zero_or_pos b with h0 | h0
          · exact h0
          · have := I.low h0; omega
        subst hb0
        refine ⟨by omega, fun j hj => by omega, ?_⟩
        intro j hj hjn
        have := getD_le hs (i := 0) (j := j) (by omega) hjn; omega

/-- With `k` absent the result is never `Ok`. -/
theorem bsearch_absent (ks : List Nat) (hs : SSorted ks) (k : Nat) (h : k ∉ ks) :
    ∃ p, bsearch ks k = .err p := by
  cases e : bsearch ks k with
  | err p => exact ⟨p, rfl⟩
  | ok i =>
    have ⟨hi, hk⟩ := (bsearch_ok_iff ks hs k i).1 e
    rw [getD_mem_eq hi] at hk
    exact absurd (hk ▸ List.getElem_mem hi) h

-- Rust's loop on concrete slices (std's own doc examples).
#guard bsearch [0, 1, 1, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55] 13 == .ok 9
#guard bsearch [0, 1, 1, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55] 4 == .err 7
#guard bsearch [0, 1, 1, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55] 100 == .err 13

end PendingCommit.BinSearch
