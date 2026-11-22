import FalkorNativeIndex.Rows
set_option linter.unusedSimpArgs false

/-!
# CCH topology: the chordal supergraph (`cch/order.rs:381`, `chordal_supergraph`)

Over ranks. `adj r` is the skeleton's row of the node of rank `r`, as ranks (`Csr` is symmetric and simple:
`Cch.lean`, `isSymmetricSimple_iff`). `chordal_supergraph` builds each rank's upward row in rank order from
its own higher neighbours and its elimination-tree children's rows (the children of `r` are the ranks whose
row starts with `r`; the Rust keeps them as linked lists, which this model replaces by that set), sorted and
deduplicated (`sort_unstable`, `dedup`).

Proven: the **soundness** half of "the rows are exactly the fill-in": every entry of a row is a rank reachable
from it through lower ranks (`row_sound`), and `up_closure`, the closure property the elimination tree
rests on. The completeness half (every such reachable rank is in the row) is **not proven**.
-/

namespace NI

theorem ifP {α : Type _} {c : Prop} [Decidable c] {a b : α} (h : c) : (if c then a else b) = a := by simp [h]
theorem ifN {α : Type _} {c : Prop} [Decidable c] {a b : α} (h : ¬c) : (if c then a else b) = b := by simp [h]

/-- A path from `u` to `w` through vertices all below `bound` (the endpoints are free). -/
inductive Pth (adj : Nat → List Nat) (bound : Nat) : Nat → Nat → Prop where
  | edge {u v : Nat} : v ∈ adj u → Pth adj bound u v
  | step {u v w : Nat} : v ∈ adj u → v < bound → Pth adj bound v w → Pth adj bound u w

theorem Pth.mono {adj : Nat → List Nat} {b b' : Nat} (h : b ≤ b') {u w : Nat} (p : Pth adj b u w) :
    Pth adj b' u w := by
  induction p with
  | edge hv => exact .edge hv
  | step hv hb _ ih => exact .step hv (by omega) ih

theorem Pth.trans {adj : Nat → List Nat} {b : Nat} {u m w : Nat} (p : Pth adj b u m) (hm : m < b)
    (q : Pth adj b m w) : Pth adj b u w := by
  induction p with
  | edge hv => exact .step hv hm q
  | step hv hb _ ih => exact .step hv hb (ih hm q)

theorem Pth.symm {adj : Nat → List Nat} (hs : ∀ u v, v ∈ adj u → u ∈ adj v) {b : Nat} {u w : Nat}
    (p : Pth adj b u w) : Pth adj b w u := by
  induction p with
  | edge hv => exact .edge (hs _ _ hv)
  | @step u v w hv hb _ ih => exact Pth.trans ih hb (.edge (hs _ _ hv))

/-- Rank `r`'s upward row (`chordal_supergraph`'s loop body, `order.rs:403-428`): its higher neighbours, and
    each child's row without `r`, sorted and deduplicated. `rows` are the rows of the ranks before `r`. -/
def rowAt (adj : Nat → List Nat) (rows : List (List Nat)) (r : Nat) : List Nat :=
  sortDedup ((adj r).filter (fun h => decide (r < h)) ++
    ((List.range r).filter (fun c => decide ((rows.getD c []).head? = some r))).flatMap
      (fun c => (rows.getD c []).filter (fun h => h != r)))

/-- The rows of ranks `0 .. n-1`. -/
def buildRows (adj : Nat → List Nat) : Nat → List (List Nat)
  | 0 => []
  | n + 1 => buildRows adj n ++ [rowAt adj (buildRows adj n) n]

theorem buildRows_length (adj : Nat → List Nat) (n : Nat) : (buildRows adj n).length = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [buildRows, ih]

theorem buildRows_getD (adj : Nat → List Nat) : ∀ (n r : Nat), r < n →
    (buildRows adj n).getD r [] = rowAt adj (buildRows adj r) r := by
  intro n
  induction n with
  | zero => intro r h; omega
  | succ n ih =>
    intro r h
    simp only [buildRows]
    by_cases hr : r < n
    · have hl : r < (buildRows adj n).length := by rw [buildRows_length]; exact hr
      rw [List.getD_eq_getElem?_getD, List.getElem?_append_left hl, ← List.getD_eq_getElem?_getD]
      exact ih r hr
    · have : r = n := by omega
      subst this
      rw [List.getD_eq_getElem?_getD, List.getElem?_append_right (by rw [buildRows_length]; exact Nat.le_refl _)]
      simp [buildRows_length]

/-- Row `r` (a function of `r`). -/
def rowOf (adj : Nat → List Nat) (r : Nat) : List Nat := rowAt adj (buildRows adj r) r

/-- Earlier rows seen from `rowAt` are `rowOf`. -/
theorem rows_getD (adj : Nat → List Nat) (r c : Nat) (h : c < r) : (buildRows adj r).getD c [] = rowOf adj c :=
  buildRows_getD adj r c h

/-- **Membership in a row**: a higher neighbour, or an entry of a child's row other than `r`. -/
theorem mem_rowOf (adj : Nat → List Nat) (r w : Nat) :
    w ∈ rowOf adj r ↔ (w ∈ adj r ∧ r < w) ∨
      ∃ c, c < r ∧ (rowOf adj c).head? = some r ∧ w ∈ rowOf adj c ∧ w ≠ r := by
  unfold rowOf rowAt
  rw [(sortDedup_spec _).2]
  simp only [List.mem_append, List.mem_filter, decide_eq_true_eq, List.mem_flatMap, List.mem_range, bne_iff_ne,
    ne_eq]
  constructor
  · rintro (⟨h1, h2⟩ | ⟨c, ⟨hc, hh⟩, hw, hne⟩)
    · exact Or.inl ⟨h1, h2⟩
    · refine Or.inr ⟨c, hc, ?_, ?_, hne⟩
      · rwa [rows_getD adj r c hc] at hh
      · rwa [rows_getD adj r c hc] at hw
  · rintro (⟨h1, h2⟩ | ⟨c, hc, hh, hw, hne⟩)
    · exact Or.inl ⟨h1, h2⟩
    · refine Or.inr ⟨c, ⟨hc, ?_⟩, ?_, hne⟩
      · rwa [rows_getD adj r c hc]
      · rwa [rows_getD adj r c hc]

theorem rowOf_sorted (adj : Nat → List Nat) (r : Nat) : Sorted (rowOf adj r) := (sortDedup_spec _).1

/-- A row's first entry is its least. -/
theorem head_le (adj : Nat → List Nat) (r p w : Nat) (hp : (rowOf adj r).head? = some p) (hw : w ∈ rowOf adj r) :
    p ≤ w := by
  have hs := rowOf_sorted adj r
  cases hl : rowOf adj r with
  | nil => rw [hl] at hw; simp at hw
  | cons a t =>
    rw [hl] at hp hs hw
    simp at hp; subst hp
    rcases List.mem_cons.1 hw with rfl | hw
    · exact Nat.le_refl _
    · exact Nat.le_of_lt (List.rel_of_pairwise_cons hs hw)

/-- **Soundness: every entry of a row is a fill-in**: rank `w` is above `r` and reachable from it through ranks
    below `r` (`hs`: the skeleton is symmetric). -/
theorem row_sound (adj : Nat → List Nat) (hs : ∀ u v, v ∈ adj u → u ∈ adj v) :
    ∀ r w, w ∈ rowOf adj r → r < w ∧ Pth adj r r w := by
  intro r
  induction r using Nat.strongRecOn with
  | _ r ih =>
    intro w hw
    rcases (mem_rowOf adj r w).1 hw with ⟨h1, h2⟩ | ⟨c, hc, hh, hwc, hne⟩
    · exact ⟨h2, .edge h1⟩
    · have hrc : r ∈ rowOf adj c := List.mem_of_mem_head? hh
      obtain ⟨_, p1⟩ := ih c hc r hrc
      obtain ⟨hcw, p2⟩ := ih c hc w hwc
      have hr : r ≤ w := head_le adj c r w hh hwc
      refine ⟨by omega, ?_⟩
      have p1' : Pth adj r r c := (Pth.symm hs p1).mono (by omega)
      exact p1'.trans hc (p2.mono (by omega))

/-- **The closure the elimination tree rests on**: if rank `x < r` has `r` and `w > r` in its row, then `r` has
    `w` in its row (a child's row, less its parent, is in the parent's). -/
theorem up_closure (adj : Nat → List Nat) : ∀ (d x r w : Nat), r - x = d → x < r → r ∈ rowOf adj x →
    w ∈ rowOf adj x → r < w → w ∈ rowOf adj r := by
  intro d
  induction d using Nat.strongRecOn with
  | _ d ih =>
    intro x r w hd hxr hr hw hrw
    obtain ⟨p, hp⟩ : ∃ p, (rowOf adj x).head? = some p := by
      cases h : rowOf adj x with
      | nil => rw [h] at hr; simp at hr
      | cons a t => exact ⟨a, rfl⟩
    have hpr : p ≤ r := head_le adj x p r hp hr
    by_cases hpe : p = r
    · subst hpe
      exact (mem_rowOf adj p w).2 (Or.inr ⟨x, hxr, hp, hw, by omega⟩)
    · have hpm : p ∈ rowOf adj x := List.mem_of_mem_head? hp
      have hxp : x < p := (row_sound_aux adj x p hpm)
      have hpr' : p < r := by omega
      have hr' : r ∈ rowOf adj p := (mem_rowOf adj p r).2 (Or.inr ⟨x, hxp, hp, hr, by omega⟩)
      have hw' : w ∈ rowOf adj p := (mem_rowOf adj p w).2 (Or.inr ⟨x, hxp, hp, hw, by omega⟩)
      exact ih (r - p) (by omega) p r w rfl hpr' hr' hw' hrw
where
  row_sound_aux (adj : Nat → List Nat) (x p : Nat) (h : p ∈ rowOf adj x) : x < p := by
    rcases (mem_rowOf adj x p).1 h with ⟨_, h2⟩ | ⟨c, hc, hh, hwc, hne⟩
    · exact h2
    · have := head_le adj c x p hh hwc
      omega

/-! ## `Rank` (`order.rs:40-60`) -/

/-- `Rank::new(index)` stores `index + 1` (a `NonZeroUsize`); `Rank::index` subtracts it again. -/
def rankStore (index : Nat) : Nat := index + 1
def rankIndex (stored : Nat) : Nat := stored - 1

theorem rank_roundtrip (index : Nat) : rankIndex (rankStore index) = index ∧ rankStore index ≠ 0 := by
  simp [rankIndex, rankStore]

theorem rank_order (a b : Nat) : rankStore a < rankStore b ↔ a < b := by simp [rankStore]

/-! ## Completeness: every fill-in is in the row -/

/-- A path through vertices below `k + 1` either stays below `k`, or passes through `k`, in which case it is a path
    to `k` and a path from `k`, both below `k` (a loop at `k` is dropped). -/
theorem Pth.split {adj : Nat → List Nat} {k u w : Nat} (p : Pth adj (k + 1) u w) :
    Pth adj k u w ∨ (Pth adj k u k ∧ Pth adj k k w) := by
  induction p with
  | edge hv => exact Or.inl (.edge hv)
  | @step u v w hv hb _ ih =>
    by_cases hvk : v < k
    · rcases ih with h | ⟨h1, h2⟩
      · exact Or.inl (.step hv hvk h)
      · exact Or.inr ⟨.step hv hvk h1, h2⟩
    · have hv' : v = k := by omega
      subst hv'
      refine Or.inr ⟨.edge hv, ?_⟩
      rcases ih with h | ⟨_, h2⟩
      · exact h
      · exact h2

/-- **Completeness, by induction on the bound**: a rank `u ≥ k` and a higher rank `w` joined by a path through
    ranks below `k` have `w` in `u`'s row. (`k = 0`: the path is an edge. `k + 1`: a path through `k` puts both
    ends in `k`'s row, and a row is closed upward, `up_closure`.) -/
theorem row_complete_aux (adj : Nat → List Nat) (hs : ∀ u v, v ∈ adj u → u ∈ adj v) :
    ∀ (k u w : Nat), k ≤ u → u < w → Pth adj k u w → w ∈ rowOf adj u := by
  intro k
  induction k with
  | zero =>
    intro u w _ huw p
    cases p with
    | edge hv => exact (mem_rowOf adj u w).2 (Or.inl ⟨hv, huw⟩)
    | step _ hb _ => omega
  | succ k ih =>
    intro u w hku huw p
    rcases p.split with h | ⟨h1, h2⟩
    · exact ih u w (by omega) huw h
    · have hu : u ∈ rowOf adj k := ih k u (Nat.le_refl _) (by omega) (Pth.symm hs h1)
      have hw : w ∈ rowOf adj k := ih k w (Nat.le_refl _) (by omega) h2
      exact up_closure adj (u - k) k u w rfl (by omega) hu hw huw

/-- **Completeness**: a rank above `r` reachable from it through ranks below `r` is in `r`'s row. -/
theorem row_complete (adj : Nat → List Nat) (hs : ∀ u v, v ∈ adj u → u ∈ adj v) (r w : Nat) (hrw : r < w)
    (p : Pth adj r r w) : w ∈ rowOf adj r :=
  row_complete_aux adj hs r r w (Nat.le_refl _) hrw p

/-- **`chordal_supergraph` computes exactly the fill-in** (the elimination game's graph): `w` is in rank `r`'s upward
    row iff `w > r` and `w` is reachable from `r` through ranks below `r` (`row_sound` and `row_complete`). -/
theorem row_iff (adj : Nat → List Nat) (hs : ∀ u v, v ∈ adj u → u ∈ adj v) (r w : Nat) :
    w ∈ rowOf adj r ↔ r < w ∧ Pth adj r r w :=
  ⟨row_sound adj hs r w, fun ⟨h1, h2⟩ => row_complete adj hs r w h1 h2⟩

end NI
