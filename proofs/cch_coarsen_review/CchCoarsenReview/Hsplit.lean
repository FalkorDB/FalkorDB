import CchCoarsenReview.Cpack
import GraphblasWrappers.DeltaLmxm
/-!
# `hsplit`: the fine split per depth as one filter pass over `A_0`'s CSR

**Rust** (m1opt.patch @7df8e5887): `split2`, m1_reuse_bench.rs:1856-1910. Under `hsplit` (:1865) row
`u` of the new matrix is empty when `labels[u] == SEP` (:1874-1877), and otherwise keeps the columns
`v` of `A_0`'s row `u` with `labels[v] == labels[u]` (:1878-1882). The output is iso 1 (`emit_csr`
with `iso = true`, :1870).

**Replaced** (:1889-1909): `M = GxB_Matrix_build_Scalar(ii, jj, 1)` with `(ii, jj) = (v, labels[v])`
for every non-separator `v` (:1889-1897, :1898-1903), then
`GrB_mxm(an, A_0, NULL, GxB_ANY_PAIR_INT32, M, M, GrB_DESC_ST1)` (:1904-1909): `A_0` is a structural
mask, the second operand is transposed, so `an = A_0⟨struct⟩ (M ANY_PAIR Mᵀ)`. The `ANY_PAIR` product's
pattern is graphblas_wrappers' `GBWDelta.mxmAny`; `ANY_PAIR` stores the value 1.
-/

namespace CchCoarsen

/-- `SEP` (m1_reuse_bench.rs:33): `u32::MAX`, the separator label. -/
def SEP : Nat := 4294967295

/-- `split2` under `hsplit`: the host pass. -/
def splitRows (A : Rows Int) (lab : Nat → Nat) : Rows Int := fun u =>
  if lab u = SEP then [] else ((A u).filter (fun e => decide (lab e.1 = lab u))).map (fun e => (e.1, (1 : Int)))

/-- `M` (n × cells) as a pattern: `M(v, c)` iff `v < n`, `v` is not a separator and `labels[v] = c`. -/
def splitM (n : Nat) (lab : Nat → Nat) : GBWDelta.Pat := fun v c => decide (v < n ∧ lab v ≠ SEP ∧ lab v = c)

/-- The switch-off result: `A_0⟨struct⟩ = M ANY_PAIR Mᵀ`, a fresh output (`an` is new), value 1. -/
def splitGB (n cells : Nat) (A : Mat Int) (lab : Nat → Nat) : Mat Int := fun u v =>
  if (A u v).isSome && GBWDelta.mxmAny cells (splitM n lab) (fun c v => splitM n lab v c) u v then some 1 else none

/-- `A_0` is the unloaded CSR of an `n × n` matrix, and every non-separator label is a cell below
`cells` (`GxB_Matrix_build_Scalar` needs the column index in range). -/
structure SplitPre (n cells : Nat) (A : Rows Int) (lab : Nat → Nat) : Prop where
  rowStrict : ∀ u, StrictRow (A u)
  colLt : ∀ u, ∀ e ∈ A u, e.1 < n
  rowsBeyond : ∀ u, n ≤ u → A u = []
  labLt : ∀ v, v < n → lab v ≠ SEP → lab v < cells

theorem lookup_filter_map {row : List (Nat × Int)} (hn : (row.map (·.1)).Nodup) (q : Nat → Bool)
    (x : Int) (v : Nat) :
    ((row.filter (fun e => q e.1)).map (fun e => (e.1, x))).lookup v =
      if (row.lookup v).isSome ∧ q v = true then some x else none := by
  induction row with
  | nil => simp
  | cons e row ih =>
    obtain ⟨a, b⟩ := e
    simp only [List.map_cons, List.nodup_cons, List.mem_map] at hn
    have hna : row.lookup a = none := lookup_eq_none_of (fun e he hea => hn.1 ⟨e, he, hea⟩)
    rw [List.filter_cons, List.lookup_cons]
    by_cases hva : v = a
    · subst hva
      by_cases hq : q v = true
      · simp [hq]
      · have : ((row.filter (fun e => q e.1)).map (fun e => (e.1, x))).lookup v = none := by
          rw [ih hn.2]; simp [hna]
        simp [hq, this]
    · have hb : (v == a) = false := by simp [hva]
      rw [hb]
      by_cases hq : q a = true
      · simp only [hq, ite_true, List.map_cons, List.lookup_cons, hb]
        exact ih hn.2
      · simp only [hq, Bool.false_eq_true, ite_false]
        exact ih hn.2

/-- **`hsplit` = `A_0⟨struct⟩ = M ANY_PAIR Mᵀ`**: the host pass keeps `(u, v)` exactly when `A_0` stores
it and both ends carry the same non-separator label; value 1 either way. -/
theorem hsplit_eq {n cells : Nat} {A : Rows Int} {lab : Nat → Nat} (hp : SplitPre n cells A lab) (u v : Nat) :
    den (splitRows A lab) u v = splitGB n cells (den A) lab u v := by
  have hspec : (GBWDelta.mxmAny cells (splitM n lab) (fun c v => splitM n lab v c) u v = true) ↔
      (u < n ∧ v < n ∧ lab u ≠ SEP ∧ lab v = lab u) := by
    simp only [GBWDelta.mxmAny, decide_eq_true_eq, GBWDelta.cnt_pos_iff, splitM,
      decide_eq_true_eq]
    constructor
    · rintro ⟨c, _, ⟨hu, hsu, rfl⟩, ⟨hv, _, hlv⟩⟩; exact ⟨hu, hv, hsu, hlv⟩
    · rintro ⟨hu, hv, hsu, hlv⟩
      exact ⟨lab u, hp.labLt u hu hsu, ⟨hu, hsu, rfl⟩, ⟨hv, by rw [hlv]; exact hsu, hlv⟩⟩
  unfold splitGB
  by_cases hun : n ≤ u
  · have : A u = [] := hp.rowsBeyond u hun
    simp [den, splitRows, this]
  · have hu : u < n := by omega
    by_cases hs : lab u = SEP
    · have : ¬ (GBWDelta.mxmAny cells (splitM n lab) (fun c v => splitM n lab v c) u v = true) :=
        fun h => ((hspec.1 h).2.2.1 hs)
      simp [den, splitRows, hs, this]
    · simp only [den, splitRows, hs, ite_false]
      rw [lookup_filter_map (hp.rowStrict u).keys_nodup (fun j => decide (lab j = lab u)) 1 v]
      by_cases hA : ((A u).lookup v).isSome
      · obtain ⟨w, hw⟩ := (lookup_isSome_iff _ _).1 hA
        have hv : v < n := hp.colLt u (v, w) hw
        by_cases hl : lab v = lab u
        · have := hspec.2 ⟨hu, hv, hs, hl⟩
          simp [hA, hl, this]
        · have : ¬ (GBWDelta.mxmAny cells (splitM n lab) (fun c v => splitM n lab v c) u v = true) :=
            fun h => hl (hspec.1 h).2.2.2
          simp [hA, hl, this]
      · simp [hA]

/-- The split's rows are strictly ascending (a sub-list of `A_0`'s), a separator's row is empty, and no
row holds a separator column: every column carries its row's label. -/
theorem hsplit_shape {A : Rows Int} {lab : Nat → Nat} (hA : ∀ u, StrictRow (A u)) (u : Nat) :
    StrictRow (splitRows A lab u) ∧ (lab u = SEP → splitRows A lab u = []) ∧
    ∀ e ∈ splitRows A lab u, lab e.1 = lab u ∧ lab e.1 ≠ SEP ∧ e.2 = 1 := by
  refine ⟨?_, fun h => by simp [splitRows, h], fun e he => ?_⟩
  · unfold splitRows
    split
    · simp [StrictRow]
    · simp only [StrictRow, List.pairwise_map]
      exact (hA u).sublist List.filter_sublist
  · unfold splitRows at he
    split at he
    · simp at he
    · next hs =>
      simp only [List.mem_map, List.mem_filter, decide_eq_true_eq] at he
      obtain ⟨x, ⟨_, hx⟩, rfl⟩ := he
      exact ⟨hx, by rw [hx]; exact hs, rfl⟩

/-- **The split discharges `hcon`'s precondition** at R's first level: R passes the weights
`ones_vec(n, labels)` (m1_reuse_bench.rs:1948-1958, :2196), so `live v ↔ labels[v] ≠ SEP`, and every
column of the split is live. -/
theorem hsplit_cols_live {A : Rows Int} {lab : Nat → Nat} (u : Nat) (e : Nat × Int)
    (he : e ∈ splitRows A lab u) : live (fun v => if lab v ≠ SEP then 1 else 0) e.1 := by
  unfold splitRows at he
  split at he
  · simp at he
  · next hs =>
    simp only [List.mem_map, List.mem_filter, decide_eq_true_eq] at he
    obtain ⟨x, ⟨_, hx⟩, rfl⟩ := he
    simp [live, hx, hs]

/-- The split keeps a symmetric `A_0` symmetric (`hcon` and the matching rounds need a symmetric `A`). -/
theorem hsplit_symm {n cells : Nat} {A : Rows Int} {lab : Nat → Nat} (hp : SplitPre n cells A lab)
    (hsym : ∀ u v, (den A u v).isSome ↔ (den A v u).isSome) (u v : Nat) :
    (den (splitRows A lab) u v).isSome ↔ (den (splitRows A lab) v u).isSome := by
  rw [hsplit_eq hp, hsplit_eq hp]
  have hspec : ∀ a b, (GBWDelta.mxmAny cells (splitM n lab) (fun c v => splitM n lab v c) a b = true) ↔
      (a < n ∧ b < n ∧ lab a ≠ SEP ∧ lab b = lab a) := by
    intro a b
    simp only [GBWDelta.mxmAny, decide_eq_true_eq, GBWDelta.cnt_pos_iff, splitM,
      decide_eq_true_eq]
    constructor
    · rintro ⟨c, _, ⟨hu, hsu, rfl⟩, ⟨hv, _, hlv⟩⟩; exact ⟨hu, hv, hsu, hlv⟩
    · rintro ⟨hu, hv, hsu, hlv⟩
      exact ⟨lab a, hp.labLt a hu hsu, ⟨hu, hsu, rfl⟩, ⟨hv, by rw [hlv]; exact hsu, hlv⟩⟩
  have : ∀ a b, (splitGB n cells (den A) lab a b).isSome ↔
      ((den A a b).isSome ∧ (a < n ∧ b < n ∧ lab a ≠ SEP ∧ lab b = lab a)) := by
    intro a b
    unfold splitGB
    rw [← hspec a b]
    by_cases h1 : (den A a b).isSome <;>
    by_cases h2 : GBWDelta.mxmAny cells (splitM n lab) (fun c v => splitM n lab v c) a b = true <;>
    simp [h1, h2]
  rw [this, this, hsym u v]
  constructor
  · rintro ⟨h, hu, hv, hs, hl⟩; exact ⟨h, hv, hu, by rw [hl]; exact hs, hl.symm⟩
  · rintro ⟨h, hv, hu, hs, hl⟩; exact ⟨h, hu, hv, by rw [hl]; exact hs, hl.symm⟩

end CchCoarsen
