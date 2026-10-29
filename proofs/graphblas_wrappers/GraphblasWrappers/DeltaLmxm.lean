import VersionedMatrix.Delta
import VersionedMatrix.DeltaProofs
import VersionedMatrix.Tensor
/-
# `Matrix::delta_lmxm` (matrix.rs:1520-1655): `F * (M - DM + DP)` on the pattern

`delta_lmxm` multiplies a frontier `F` by a versioned matrix's three layers
(`m` base, `dp` additions, `dm` deletions) without merging them. Since #3492
(a3a99bdbc) it keeps an output that one deleted base edge reaches but another,
live one still does.

The GraphBLAS calls are *definitions* here, with the meaning the GraphBLAS spec
gives them (as in `versioned_matrix/Delta.lean`): a matrix is the function from a
coordinate to its stored value (`Pat`: stored or not, for a `bool` / pattern use of a
`u64` matrix; `UMat`: the stored `UINT64`).

| primitive (Rust) | here |
| --- | --- |
| `GrB_mxm` over `GxB_ANY_PAIR_BOOL`, no mask | `mxmAny`: `C(i,j)` stored iff `∃k. A(i,k) ∧ B(k,j)` |
| `GrB_mxm` over `GxB_PLUS_PAIR_UINT64`, no mask | `mxmCount`: `C(i,j) = #{k. A(i,k) ∧ B(k,j)}`, stored iff that is `> 0` |
| the same with mask `dm_paths`, `GrB_DESC_S` (structural, fresh output) | `maskS dm_paths (mxmCount ..)` |
| `GrB_mxm(self, dm_paths, .., ANY_PAIR, self, m, GrB_DESC_RSC)` | `main`: stored where `dm_paths` is *not* stored and `mxmAny` is |
| `GrB_Matrix_eWiseMult_BinaryOp(.., GrB_GT_UINT64, m_paths, dm_paths)` | `ewGT`: where both are stored, `a > b` |
| `GrB_Matrix_select_BOOL(.., GrB_VALUEEQ_BOOL, .., true)` | `selTrue` |
| `Matrix::element_wise_add` on `bool` (`GxB_ANY_BOOL`) | `eadd`: union of patterns |
| `.nvals() > 0` | `nz` |

`self` is both input and output of the main `GrB_mxm`; the spec reads the input
before writing, and `dm_paths`, `m_paths`, `live` and `ac` are all computed from the
original `self` before it is overwritten (matrix.rs:1545-1648).
-/

namespace GBWDelta

abbrev Pat := Nat → Nat → Bool
abbrev UMat := Nat → Nat → Option Nat

/-- Number of inner indices `k < n` through which `(i,j)` is reached: `PLUS_PAIR`. -/
def cnt (n : Nat) (A B : Pat) (i j : Nat) : Nat :=
  ((List.range n).filter fun k => A i k && B k j).length

theorem cnt_pos_iff {n : Nat} {A B : Pat} {i j : Nat} :
    0 < cnt n A B i j ↔ ∃ k, k < n ∧ A i k = true ∧ B k j = true := by
  simp only [cnt, List.length_pos_iff_exists_mem, List.mem_filter, List.mem_range,
    Bool.and_eq_true]

def mxmCount (n : Nat) (A B : Pat) : UMat :=
  fun i j => if cnt n A B i j = 0 then none else some (cnt n A B i j)

def mxmAny (n : Nat) (A B : Pat) : Pat := fun i j => decide (0 < cnt n A B i j)

def maskS (mk X : UMat) : UMat := fun i j => if (mk i j).isSome then X i j else none

def ewGT (a b : UMat) : Nat → Nat → Option Bool :=
  fun i j => match a i j, b i j with
    | some x, some y => some (decide (y < x))
    | _, _ => none

def selTrue (x : Nat → Nat → Option Bool) : Pat := fun i j => x i j == some true

def eadd (A B : Pat) : Pat := fun i j => A i j || B i j

open Classical in
/-- `nvals() > 0`. -/
noncomputable def nz (X : Pat) : Bool := decide (∃ i j, X i j = true)

noncomputable def nzU (X : UMat) : Bool := nz fun i j => (X i j).isSome

open Classical in
theorem nz_iff (X : Pat) : nz X = true ↔ ∃ i j, X i j = true := by
  unfold nz; exact decide_eq_true_iff

theorem nz_false {X : Pat} (h : nz X = false) (i j : Nat) : X i j = false := by
  refine Classical.byContradiction fun hc => ?_
  have : nz X = true := (nz_iff X).2 ⟨i, j, by simpa using hc⟩
  rw [h] at this; exact Bool.noConfusion this

theorem nz_true_of {X : Pat} {i j : Nat} (h : X i j = true) : nz X = true :=
  (nz_iff X).2 ⟨i, j, h⟩

/-- `Matrix::delta_lmxm`, branch for branch (matrix.rs:1520-1655). -/
noncomputable def deltaLmxm (n : Nat) (F M DP DM : Pat) : Pat :=
  if (!nz DP && !nz DM) = true then
    mxmAny n F M                                              -- :1537-1540 hot path: `lmxm(m)`
  else
    let dmPaths : UMat := mxmCount n F DM                      -- :1559
    let mask : Option UMat := if nz DM = true ∧ nzU dmPaths = true then some dmPaths else none
    let restored : Option Pat :=
      if nz DM = true ∧ nzU dmPaths = true then
        let mPaths := maskS dmPaths (mxmCount n F M)           -- :1575
        let live : Pat := selTrue (ewGT mPaths dmPaths)        -- :1586, :1595
        if nz live = true then some live else none
      else none
    let accum : Option Pat :=
      if nz DP = true then
        let ac := mxmAny n F DP                                -- :1612
        if nz ac = true then some ac else none
      else none
    let main : Pat :=                                          -- :1626, mask `!dm_paths`, replace
      match mask with
      | some mk => fun i j => !(mk i j).isSome && mxmAny n F M i j
      | none => mxmAny n F M
    let r1 := match restored with | some l => eadd main l | none => main   -- :1650
    match accum with | some a => eadd r1 a | none => r1                    -- :1653

/-! ## What it computes, with no assumption about the layers -/

def opt (o : Option Pat) (i j : Nat) : Bool :=
  match o with | some l => l i j | none => false

theorem addOpt (o : Option Pat) (X : Pat) (i j : Nat) :
    (match o with | some l => eadd X l | none => X) i j = (X i j || opt o i j) := by
  cases o <;> simp [eadd, opt]

theorem cnt_zero_of_nz_false {n : Nat} {F X : Pat} (h : nz X = false) (i j : Nat) :
    cnt n F X i j = 0 := by
  refine Classical.byContradiction fun hc => ?_
  obtain ⟨k, _, _, h2⟩ := cnt_pos_iff.1 (Nat.pos_of_ne_zero hc)
  rw [nz_false h k j] at h2; exact Bool.noConfusion h2

theorem cnt_zero_of_nzU_false {n : Nat} {F DM : Pat} (h : nzU (mxmCount n F DM) = false) (i j : Nat) :
    cnt n F DM i j = 0 := by
  have := nz_false h i j
  simp only [mxmCount] at this
  refine Classical.byContradiction fun hc => ?_
  simp [hc] at this

/-- The result stored at `(i,j)`, in terms of the path counts alone. `a = |F*M|`,
`b = |F*DM|`, `c = |F*DP|` at `(i,j)`: kept when the output is not under the deletion
mask (`b = 0`) and reached by `m`; or under it but `m` reaches it by strictly more
paths than `dm` does; or reached by `dp`. -/
theorem deltaLmxm_spec (n : Nat) (F M DP DM : Pat) (i j : Nat) :
    deltaLmxm n F M DP DM i j = true ↔
      (0 < cnt n F M i j ∧ cnt n F DM i j = 0) ∨
      (0 < cnt n F DM i j ∧ cnt n F DM i j < cnt n F M i j) ∨
      0 < cnt n F DP i j := by
  unfold deltaLmxm
  by_cases hhot : (!nz DP && !nz DM) = true
  · rw [if_pos hhot]
    simp only [Bool.and_eq_true, Bool.not_eq_true'] at hhot
    have h1 := cnt_zero_of_nz_false (n := n) (F := F) hhot.2 i j
    have h2 := cnt_zero_of_nz_false (n := n) (F := F) hhot.1 i j
    simp only [mxmAny, decide_eq_true_eq]
    omega
  · rw [if_neg hhot]
    simp only []
    -- value of the optional pieces
    have hacc : opt (if nz DP = true then
          (let ac := mxmAny n F DP; if nz ac = true then some ac else none) else none) i j = true ↔
        0 < cnt n F DP i j := by
      by_cases hd : nz DP = true
      · simp only [hd, if_true]
        by_cases ha : nz (mxmAny n F DP) = true
        · simp [ha, opt, mxmAny]
        · have := nz_false (by simpa using ha) i j
          simp only [Bool.not_eq_true] at ha
          simp [ha, opt]
          simpa [mxmAny] using this
      · have hd' : nz DP = false := by simpa using hd
        simp [hd', opt, cnt_zero_of_nz_false hd']
    rw [addOpt, Bool.or_eq_true, hacc]
    by_cases hG : (nz DM = true ∧ nzU (mxmCount n F DM) = true)
    · simp only [hG, and_self, if_true]
      have hb := hG
      rw [addOpt, Bool.or_eq_true]
      have hlive : opt (let live := selTrue (ewGT (maskS (mxmCount n F DM) (mxmCount n F M))
            (mxmCount n F DM)); if nz live = true then some live else none) i j = true ↔
          (0 < cnt n F DM i j ∧ cnt n F DM i j < cnt n F M i j) := by
        simp only []
        by_cases hl : nz (selTrue (ewGT (maskS (mxmCount n F DM) (mxmCount n F M)) (mxmCount n F DM))) = true
        · simp only [hl, if_true, opt, selTrue, ewGT, maskS, mxmCount, beq_iff_eq]
          by_cases hb0 : cnt n F DM i j = 0
          · simp [hb0]
          · by_cases ha0 : cnt n F M i j = 0
            · simp [hb0, ha0] <;> omega
            · simp [hb0, ha0] <;> omega
        · have hl' : nz (selTrue (ewGT (maskS (mxmCount n F DM) (mxmCount n F M)) (mxmCount n F DM))) = false := by
            simpa using hl
          have := nz_false hl' i j
          simp only [hl, if_false, opt, Bool.false_eq_true, false_iff, not_and]
          intro hb0
          simp only [selTrue, ewGT, maskS, mxmCount, beq_iff_eq] at this
          by_cases ha0 : cnt n F M i j = 0
          · omega
          · intro hlt
            simp [Nat.pos_iff_ne_zero.1 hb0, ha0, Nat.pos_iff_ne_zero.1 hb0] at this
            omega
      rw [hlive]
      simp only [mxmCount, mxmAny, Option.isSome_some, Bool.not_true, Bool.false_and,
        Bool.and_eq_true, decide_eq_true_eq]
      by_cases hb0 : cnt n F DM i j = 0 <;> simp [hb0] <;> omega
    · have hb0 : cnt n F DM i j = 0 := by
        by_cases h1 : nz DM = true
        · have h2 : nzU (mxmCount n F DM) = false := by
            by_cases h3 : nzU (mxmCount n F DM) = true
            · exact absurd ⟨h1, h3⟩ hG
            · simpa using h3
          exact cnt_zero_of_nzU_false h2 i j
        · exact cnt_zero_of_nz_false (n := n) (F := F) (by simpa using h1) i j
      simp only [hG, if_false, mxmAny, hb0]
      simp

/-! ## Correctness: `F * (M - DM + DP)`, given `dm ⊆ m` -/

theorem len_filter_split (l : List Nat) (p q : Nat → Bool) :
    (l.filter p).length =
      (l.filter fun x => p x && q x).length + (l.filter fun x => p x && !q x).length := by
  induction l with
  | nil => simp
  | cons a t ih =>
    by_cases hp : p a = true <;> by_cases hq : q a = true <;>
      simp [List.filter_cons, hp, hq, ih] <;> omega

/-- With `dm ⊆ m`, the paths through `m` split into those through the live entries
`m ∖ dm` and those through `dm`. -/
theorem cnt_split (n : Nat) (F M DM : Pat) (hsub : ∀ k j, DM k j = true → M k j = true)
    (i j : Nat) :
    cnt n F M i j = cnt n F (fun k j => M k j && !DM k j) i j + cnt n F DM i j := by
  unfold cnt
  rw [len_filter_split (List.range n) (fun k => F i k && M k j) (fun k => DM k j)]
  have h1 : (List.range n).filter (fun k => (F i k && M k j) && DM k j) =
      (List.range n).filter (fun k => F i k && DM k j) := by
    apply List.filter_congr
    intro k _
    by_cases hd : DM k j = true
    · simp [hd, hsub k j hd]
    · simp [hd]
  have h2 : (List.range n).filter (fun k => (F i k && M k j) && !DM k j) =
      (List.range n).filter (fun k => F i k && (M k j && !DM k j)) := by
    apply List.filter_congr
    intro k _
    simp [Bool.and_assoc]
  show _ = (List.filter (fun k => F i k && (M k j && !DM k j)) (List.range n)).length +
    (List.filter (fun k => F i k && DM k j) (List.range n)).length
  rw [h1, h2]; omega

/-- **`delta_lmxm = F * (M - DM + DP)`** (the pattern), whenever the deletion layer
sits inside the base (`dm ⊆ m`, the `VersionedMatrix` invariant). In particular an
output reached by one deleted and one live base edge is kept (#3492). -/
theorem deltaLmxm_correct (n : Nat) (F M DP DM : Pat) (hsub : ∀ k j, DM k j = true → M k j = true)
    (i j : Nat) :
    deltaLmxm n F M DP DM i j = true ↔
      ∃ k, k < n ∧ F i k = true ∧
        ((M k j = true ∧ DM k j = false) ∨ DP k j = true) := by
  rw [deltaLmxm_spec, cnt_split n F M DM hsub i j]
  have hc := cnt_pos_iff (n := n) (A := F) (B := fun k j => M k j && !DM k j) (i := i) (j := j)
  simp only [Bool.and_eq_true, Bool.not_eq_true'] at hc
  constructor
  · intro h
    have : 0 < cnt n F (fun k j => M k j && !DM k j) i j ∨ 0 < cnt n F DP i j := by omega
    rcases this with h | h
    · obtain ⟨k, hk, h1, h2, h3⟩ := hc.1 h
      exact ⟨k, hk, h1, Or.inl ⟨h2, h3⟩⟩
    · obtain ⟨k, hk, h1, h2⟩ := (cnt_pos_iff (A := F) (B := DP)).1 h
      exact ⟨k, hk, h1, Or.inr h2⟩
  · rintro ⟨k, hk, h1, (⟨h2, h3⟩ | h2)⟩
    · have := hc.2 ⟨k, hk, h1, h2, h3⟩
      omega
    · have := (cnt_pos_iff (A := F) (B := DP)).2 ⟨k, hk, h1, h2⟩
      omega

/-! ## Over a versioned matrix's layers -/

open VM in
/-- Layer sets as patterns. -/
def patOf (l : Layer) : Pat := fun i j => decide ((i, j) ∈ l)

open VM in
/-- **The result for a versioned matrix with its layer invariant**: `(i,j)` is stored iff
some frontier entry `(i,k)`, `k < n`, steps to `j` in the logical matrix `(m ∖ dm) ∪ dp`.
`Inv.dm_m` (`dm ⊆ m`) is the only invariant used; `VM.Inv` is preserved by every
mutation (`VM.run_repr` / `VM.set_inv` / `VM.remove_inv` / ...). -/
theorem deltaLmxm_eff (n : Nat) (F : Pat) (v : VM) (h : Inv v) (i j : Nat) :
    deltaLmxm n F (patOf v.m) (patOf v.dp) (patOf v.dm) i j = true ↔
      ∃ k, k < n ∧ F i k = true ∧ eff v (k, j) := by
  rw [deltaLmxm_correct n F _ _ _ (fun k j hd => by
    simpa [patOf] using h.dm_m (k, j) (by simpa [patOf] using hd))]
  unfold eff
  simp only [patOf, decide_eq_true_eq, Bool.not_eq_true', decide_eq_false_iff_not]

open VM in
/-- A `Tensor`'s forward layers (`m`, `dp`, `dm` of a pair-state grid `st`, each pair obeying
`VMTensor.Inv`): the same statement with `effV`. -/
theorem deltaLmxm_tensor (n : Nat) (F : Pat) (st : Nat → Nat → VMTensor.PairSt)
    (h : ∀ k j, VMTensor.Inv (st k j)) (i j : Nat) :
    deltaLmxm n F (fun k j => (st k j).m.isSome) (fun k j => (st k j).dp.isSome)
        (fun k j => (st k j).dm) i j = true ↔
      ∃ k, k < n ∧ F i k = true ∧ (VMTensor.effV (st k j)).isSome = true := by
  rw [deltaLmxm_correct n F _ _ _ (fun k j hd => (h k j).dm_m hd)]
  apply exists_congr; intro k
  apply and_congr_right; intro _
  apply and_congr_right; intro _
  have hi := h k j
  unfold VMTensor.effV
  cases hdp : (st k j).dp <;> cases hdm : (st k j).dm <;> cases hm : (st k j).m <;> simp [hdp, hdm, hm]

/-! ## History: before #3492 (a3a99bdbc) -/

/-- Node `1 = x`, `2 = y`, `3 = z`; frontier row `0 = {x, y}`; base edges `x→z`, `y→z`
(`delta_lmxm_keeps_an_output_one_live_edge_still_reaches`, matrix.rs:1954). -/
def cxF : Pat := fun i k => decide ((i, k) ∈ [(0, 1), (0, 2)])
def cxM : Pat := fun k j => decide ((k, j) ∈ [(1, 3), (2, 3)])
def cxNone : Pat := fun _ _ => false
def cxDM1 : Pat := fun k j => decide ((k, j) ∈ [(1, 3)])
def cxDM2 : Pat := fun k j => decide ((k, j) ∈ [(1, 3), (2, 3)])

/-- The same input through the current function keeps it, and still drops it when both
paths are deleted. -/
theorem delta_lmxm_keeps_live_path :
    deltaLmxm 8 cxF cxM cxNone cxDM1 0 3 = true := by
  rw [deltaLmxm_spec]; decide

theorem delta_lmxm_drops_when_all_paths_deleted :
    deltaLmxm 8 cxF cxM cxNone cxDM2 0 3 = false := by
  rw [Bool.eq_false_iff, Ne, deltaLmxm_spec]; decide

/-- `dm ⊆ m` is what makes the counting exact: a stray tombstone (`y→z` in `dm` but not in
`m`) cancels a live path. `VM.Inv.dm_m` rules this layout out. -/
def cxMx : Pat := fun k j => decide ((k, j) ∈ [(1, 3)])
def cxDMy : Pat := fun k j => decide ((k, j) ∈ [(2, 3)])
theorem needs_dm_sub_m :
    deltaLmxm 8 cxF cxMx cxNone cxDMy 0 3 = false := by
  rw [Bool.eq_false_iff, Ne, deltaLmxm_spec]; decide

end GBWDelta
