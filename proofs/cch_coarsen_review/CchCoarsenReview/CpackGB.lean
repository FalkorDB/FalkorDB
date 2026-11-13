import CchCoarsenReview.Cpack
/-!
# `cpack` against the GraphBLAS numbering it replaced; conservation; `pk`

The switch-off path (m1_coarsen_bench.rs:1031-1112 (m1opt.patch @7df8e5887)):

| Rust | here |
| --- | --- |
| `l = eWiseMult(MIN, idx, mate)` (:1034); both full | `min i (mt i)` |
| `lead = select(eWiseMult(EQ, l, idx) == true)`, masked by `w`'s structure when `partial` (:1035-1044) | `leadGB` |
| `extractTuples(lead)` (:1052) | `GBContract.li`: the indices ascending |
| `cid = GrB_Vector_build_UINT64(li, 0..nc)` (:1053-1055) | `GBContract.cid` with `cid (li[k]) = k` |
| `ca = GxB_Vector_extract_Vector(cid, l)` (:1059-1064) | `cid (min i (mt i))` |
| `Cᵀ = GxB_Matrix_build_Scalar_Vector(idl, ca, 1)` (:1065-1082) | `buildS 1 (ctTuplesGB ..)` |
| `w_c = mxv(PLUS_TIMES, Cᵀ, w, GrB_DESC_T0)` (:1106-1112) | `mxmPT n (tr Cᵀ) w` |

The two C contracts used (SuiteSparse:GraphBLAS user guide): `GrB_Vector_extractTuples` of a
non-jumbled vector lists its indices ascending; `GrB_Vector_build` / `GxB_Matrix_build_Scalar_Vector`
without duplicates store each tuple, pairing the `k`-th values of the index vectors (both listed in
index order, and here with the same pattern, the live set).
-/

namespace CchCoarsen

variable {n : Nat} {wh : Nat → Int} {mt : Nat → Nat}

/-- `lead` as GraphBLAS computes it: `l(i) = i`, masked by `w` when `partial` (some vertex dead). When
nothing is dead, `w` has all `n` entries, so every `i < n` is live. -/
def leadGB (n : Nat) (wh : Nat → Int) (mt : Nat → Nat) (part : Bool) (i : Nat) : Prop :=
  i < n ∧ min i (mt i) = i ∧ (part = true → live wh i)

/-- The C contracts of `extractTuples` then `build` (above). -/
structure GBContract (n : Nat) (wh : Nat → Int) (mt : Nat → Nat) (part : Bool) where
  li : List Nat
  li_eq : ∀ [DecidablePred (leadGB n wh mt part)], li = (List.range n).filter (fun i => decide (leadGB n wh mt part i))
  cid : Nat → Nat
  cid_spec : ∀ k (h : k < li.length), cid li[k] = k

instance (part : Bool) : DecidablePred (leadGB n wh mt part) :=
  fun _ => inferInstanceAs (Decidable (_ ∧ _ ∧ _))

theorem leadGB_iff {part : Bool} (hp : part = false → ∀ i, i < n → live wh i) (i : Nat) :
    leadGB n wh mt part i ↔ i < n ∧ lead wh mt i := by
  unfold leadGB lead
  constructor
  · rintro ⟨hi, hmin, hl⟩
    have hl' : live wh i := by
      cases hpart : part
      · exact hp hpart i hi
      · exact hl hpart
    refine ⟨hi, hl', ?_⟩
    rcases Nat.le_total i (mt i) with h | h
    · exact h
    · rw [Nat.min_eq_right h] at hmin; omega
  · rintro ⟨hi, hl, hle⟩
    exact ⟨hi, Nat.min_eq_left hle, fun _ => hl⟩

/-- `extractTuples(lead)` lists exactly the host's leaders. -/
theorem li_eq_leaders {part : Bool} (G : GBContract n wh mt part)
    (hp : part = false → ∀ i, i < n → live wh i) : G.li = leaders wh mt n := by
  rw [G.li_eq, leaders]
  apply List.filter_congr
  intro i hi
  have hin : i < n := List.mem_range.1 hi
  have := leadGB_iff (mt := mt) hp i
  simp only [decide_eq_decide]
  constructor
  · intro h; exact (this.1 h).2
  · intro h; exact this.2 ⟨hin, h⟩

/-- `ca = cid(l)` is the host's id: `cid` of a leader is its rank. -/
theorem cid_eq_ca {part : Bool} (G : GBContract n wh mt part) (hm : MatePre n wh mt)
    (hp : part = false → ∀ i, i < n → live wh i) {i : Nat} (hi : i < n) (hl : live wh i) :
    G.cid (min i (mt i)) = (cpack n wh mt).ca i := by
  obtain ⟨hld, hlt⟩ := min_lead hm hi hl
  have hg := leaders_get_rank hld hlt
  rw [← li_eq_leaders G hp] at hg
  obtain ⟨hk, hget⟩ := List.getElem?_eq_some_iff.1 hg
  have := G.cid_spec _ hk
  rw [hget] at this
  rw [this, (cpack_inv hm).ca i hi hl]; rfl

/-- The tuples `GxB_Matrix_build_Scalar_Vector(idl, ca)` stores: `(i, ca i)` for the live `i`. -/
def ctTuplesGB {part : Bool} (G : GBContract n wh mt part) : List (Nat × Nat) :=
  ((List.range n).filter (fun i => decide (live wh i))).map (fun i => (i, G.cid (min i (mt i))))

/-- `cpack`'s `Cᵀ`, as rows: row `i` of `tp/tj`. -/
def ctCpack (n : Nat) (wh : Nat → Int) (mt : Nat → Nat) : Rows Int :=
  fun i => ((cpack n wh mt).trows[i]?.getD []).map (fun c => (c, (1 : Int)))

/-- `cpack`'s `C`, as member lists: row `r` of `cp/cj`. -/
def cCpack (n : Nat) (wh : Nat → Int) (mt : Nat → Nat) : Nat → List Nat :=
  fun r => (cpack n wh mt).crows[r]?.getD []

/-- The rows `cpack` packs are `ctRowsOf` of its `ca`, over the live vertices below `n`. -/
theorem ctCpack_eq (hm : MatePre n wh mt) :
    ctCpack n wh mt = ctRowsOf (fun i => i < n ∧ live wh i) (cpack n wh mt).ca := by
  funext i
  simp only [ctCpack, ctRowsOf]
  by_cases hi : i < n
  · rw [cpack_ct_rows hm hi]
    by_cases hl : live wh i <;> simp [hl, hi]
  · have : (cpack n wh mt).trows[i]? = none := by
      rw [(cpack_inv hm).trows]; simp; omega
    simp [this, hi]

/-- **`cpack`'s `Cᵀ` is the one GraphBLAS built.** -/
theorem cpack_ct_eq_gb {part : Bool} (G : GBContract n wh mt part) (hm : MatePre n wh mt)
    (hp : part = false → ∀ i, i < n → live wh i) (i c : Nat) :
    den (ctCpack n wh mt) i c = buildS 1 (ctTuplesGB G) i c := by
  rw [ctCpack_eq hm, den_ctRowsOf]
  unfold buildS
  have key : (i, c) ∈ ctTuplesGB G ↔ (i < n ∧ live wh i) ∧ (cpack n wh mt).ca i = c := by
    simp only [ctTuplesGB, List.mem_map, List.mem_filter, List.mem_range, decide_eq_true_eq,
      Prod.mk.injEq]
    constructor
    · rintro ⟨j, ⟨hj, hlj⟩, rfl, hc⟩
      exact ⟨⟨hj, hlj⟩, by rw [← cid_eq_ca G hm hp hj hlj]; exact hc⟩
    · rintro ⟨⟨hi, hl⟩, hc⟩
      exact ⟨i, ⟨hi, hl⟩, rfl, by rw [cid_eq_ca G hm hp hi hl]; exact hc⟩
  by_cases h : (i < n ∧ live wh i) ∧ (cpack n wh mt).ca i = c
  · have h' := key.2 h
    simp only [h, h', and_self, ite_true]
  · have h' : (i, c) ∉ ctTuplesGB G := fun h' => h (key.1 h')
    simp only [h, h', ite_false]

/-- **`C` is `Cᵀ` transposed**: row `r` of `cp/cj` holds `i` exactly when row `i` of `tp/tj` holds `r`.
So the `cpack` contraction `c_rows · A · ctp` (:1228-1236) is the switch-off `Cᵀᵀ · A · Cᵀ`. -/
theorem cpack_c_is_ct_transposed (hm : MatePre n wh mt) (r i : Nat) :
    den (cRowsOf (cCpack n wh mt)) r i = tr (den (ctCpack n wh mt)) r i := by
  simp only [tr]
  rw [den_cRowsOf, ctCpack_eq hm, den_ctRowsOf]
  by_cases hr : r < (cpack n wh mt).nc
  · obtain ⟨row, hrow, _, hiff⟩ := cpack_rows_iff hm hr i
    simp only [cCpack, hrow, Option.getD_some]
    by_cases h : i ∈ row
    · simp [h, (hiff.1 h).1, (hiff.1 h).2.1, (hiff.1 h).2.2]
    · have : ¬ ((i < n ∧ live wh i) ∧ (cpack n wh mt).ca i = r) :=
        fun ⟨⟨a, b⟩, c⟩ => h (hiff.2 ⟨a, b, c⟩)
      simp [h, this]
  · have hnone : (cpack n wh mt).crows[r]? = none := by
      rw [(cpack_inv hm).crows]; simp
      have := (cpack_inv hm).nc; simp [rank] at this; omega
    have : ¬ ((i < n ∧ live wh i) ∧ (cpack n wh mt).ca i = r) := by
      rintro ⟨⟨hi, hl⟩, hc⟩
      exact hr (hc ▸ cpack_ca_lt hm hi hl)
    simp [cCpack, hnone, this]

/-- `cpack`'s `C` meets `hcon`'s preconditions on `C`: each row lists distinct members below `n`. -/
theorem cCpack_nodup_lt (hm : MatePre n wh mt) (r : Nat) :
    (cCpack n wh mt r).Nodup ∧ ∀ i ∈ cCpack n wh mt r, i < n := by
  by_cases hr : r < (cpack n wh mt).nc
  · obtain ⟨row, hrow, _, _⟩ := cpack_rows_iff hm hr 0
    have h := cpack_inv hm
    have hc := hr; rw [h.nc] at hc
    obtain ⟨ℓ, hℓ⟩ : ∃ ℓ, (leaders wh mt n)[r]? = some ℓ :=
      ⟨_, List.getElem?_eq_getElem (show r < (leaders wh mt n).length from hc)⟩
    have hrow' := hrow
    rw [h.crows, List.getElem?_map, hℓ] at hrow'
    cases hrow'
    simp only [cCpack, hrow, Option.getD_some]
    refine ⟨memb_nodup ℓ, fun i hi => ?_⟩
    have hmem := List.mem_of_getElem? hℓ
    rw [mem_leaders] at hmem
    exact ((memb_mem hm hmem.2 hmem.1).1 hi).1
  · have hnone : (cpack n wh mt).crows[r]? = none := by
      rw [(cpack_inv hm).crows]; simp
      have := (cpack_inv hm).nc; simp [rank] at this; omega
    simp [cCpack, hnone]

/-- `cpack`'s `C` rows are exactly the classes of its `ca` on the live vertices below `n` (the premise of
`hcon_symm`). -/
theorem cCpack_mem_iff (hm : MatePre n wh mt) (r i : Nat) :
    i ∈ cCpack n wh mt r ↔ ((i < n ∧ live wh i) ∧ (cpack n wh mt).ca i = r) := by
  by_cases hr : r < (cpack n wh mt).nc
  · obtain ⟨row, hrow, _, hiff⟩ := cpack_rows_iff hm hr i
    simp only [cCpack, hrow, Option.getD_some, hiff, and_assoc]
  · have hnone : (cpack n wh mt).crows[r]? = none := by
      rw [(cpack_inv hm).crows]; simp
      have := (cpack_inv hm).nc; simp [rank] at this; omega
    simp only [cCpack, hnone, Option.getD_none, List.not_mem_nil, false_iff]
    rintro ⟨⟨hi, hl⟩, hc⟩
    exact hr (hc ▸ cpack_ca_lt hm hi hl)

/-- **`cpack`'s `C` rows are ascending**: `[ℓ]` or `[ℓ, m]` with `ℓ < m` (:1196-1203; packed with
`jumbled = false`, :1220). -/
theorem cCpack_sorted (hm : MatePre n wh mt) (r : Nat) : (cCpack n wh mt r).Pairwise (· < ·) := by
  by_cases hr : r < (cpack n wh mt).nc
  · have h := cpack_inv hm
    have hc := hr; rw [h.nc] at hc
    obtain ⟨ℓ, hℓ⟩ : ∃ ℓ, (leaders wh mt n)[r]? = some ℓ :=
      ⟨_, List.getElem?_eq_getElem (show r < (leaders wh mt n).length from hc)⟩
    have hmem := List.mem_of_getElem? hℓ
    rw [mem_leaders] at hmem
    have hrow : (cpack n wh mt).crows[r]? = some (memb mt ℓ) := by rw [h.crows, List.getElem?_map, hℓ]; rfl
    simp only [cCpack, hrow, Option.getD_some, memb]
    by_cases he : mt ℓ = ℓ
    · simp [he]
    · have : ℓ < mt ℓ := by have := hmem.2.2; omega
      simp [he, this]
  · have hnone : (cpack n wh mt).crows[r]? = none := by
      rw [(cpack_inv hm).crows]; simp
      have := (cpack_inv hm).nc; simp [rank] at this; omega
    simp [cCpack, hnone]

/-! ## Conservation -/

theorem wsum_eq (ℓ : Nat) : wsum wh mt ℓ = lsum (memb mt ℓ) wh := by
  unfold wsum memb; by_cases he : mt ℓ = ℓ <;> simp [he]

/-- **The total vertex weight is conserved**: `Σ_c w_c(c) = Σ_{live i} w(i)`. -/
theorem cpack_conserves (hm : MatePre n wh mt) :
    lsum (cpack n wh mt).wch id = lsum (List.range n) (fun i => if live wh i then wh i else 0) := by
  rw [(cpack_inv hm).wch, lsum_map]
  simp only [id]
  have hmlt : ∀ ℓ ∈ leaders wh mt n, ∀ i ∈ memb mt ℓ, i < n := by
    intro ℓ hℓ i hi
    rw [mem_leaders] at hℓ
    exact ((memb_mem hm hℓ.2 hℓ.1).1 hi).1
  calc lsum (leaders wh mt n) (wsum wh mt)
      = lsum (leaders wh mt n) (fun ℓ => lsum (List.range n) (fun i => if i ∈ memb mt ℓ then wh i else 0)) := by
        refine lsum_congr (fun ℓ hℓ => ?_)
        rw [wsum_eq, lsum_range_indicator (memb_nodup ℓ) (hmlt ℓ hℓ)]
    _ = lsum (List.range n) (fun i => lsum (leaders wh mt n) (fun ℓ => if i ∈ memb mt ℓ then wh i else 0)) :=
        lsum_swap _ _ _
    _ = lsum (List.range n) (fun i => if live wh i then wh i else 0) := by
        refine lsum_congr (fun i hi => ?_)
        have hin := List.mem_range.1 hi
        by_cases hl : live wh i
        · obtain ⟨hld, hlt⟩ := min_lead hm hin hl
          have : ∀ ℓ ∈ leaders wh mt n, (if i ∈ memb mt ℓ then wh i else 0) =
              (if ℓ = min i (mt i) then wh i else 0) := by
            intro ℓ hℓ
            rw [mem_leaders] at hℓ
            have := memb_mem (i := i) hm hℓ.2 hℓ.1
            by_cases h : ℓ = min i (mt i)
            · have hmm : i ∈ memb mt ℓ := this.2 ⟨hin, hl, h.symm⟩
              subst h; simp [hmm]
            · have : i ∉ memb mt ℓ := fun hm' => h ((this.1 hm').2.2).symm
              simp [h, this]
          rw [lsum_congr this, lsum_single (leaders_nodup n) (mem_leaders.2 ⟨hlt, hld⟩) (fun _ => wh i)]
          simp [hl]
        · refine (lsum_zero (fun ℓ hℓ => ?_)).trans (by simp [hl])
          rw [mem_leaders] at hℓ
          have : i ∉ memb mt ℓ := fun hm' => hl ((memb_mem hm hℓ.2 hℓ.1).1 hm').2.1
          simp [this]

/-! ## `pk`: a reused level's `Cᵀ` packed from the numbering -/

/-- `pack_ct` (m1_reuse_bench.rs:808-837): row `x` is `[cmap[x]]` unless it is `NONE` (:825-831). The
`npar` branch (:813-821) emits the same rows through `emit_csr`. -/
def packCt (cmap : Nat → Nat) : Rows Int := fun x => if cmap x ≠ NONE then [(cmap x, 1)] else []

/-- The tuples the switch-off path builds from (`number`, m1_reuse_bench.rs:355-365): `(x, cmap[x])` for every
`x < nprev` with `cmap[x] ≠ NONE`. -/
def ctTuples (nprev : Nat) (cmap : Nat → Nat) : List (Nat × Nat) :=
  ((List.range nprev).filter (fun x => decide (cmap x ≠ NONE))).map (fun x => (x, cmap x))

/-- **`pk` = `GxB_Matrix_build_Scalar(ct_i, ct_j, 1)`** (m1_reuse_bench.rs:946-949), on the
`nprev` rows of `Cᵀ`. Each row has at most one entry, and exactly one when `cmap[x] ≠ NONE`. -/
theorem pk_eq_build (nprev : Nat) (cmap : Nat → Nat) (x c : Nat) (hx : x < nprev) :
    den (packCt cmap) x c = buildS 1 (ctTuples nprev cmap) x c := by
  have key : (x, c) ∈ ctTuples nprev cmap ↔ cmap x ≠ NONE ∧ c = cmap x := by
    simp only [ctTuples, List.mem_map, List.mem_filter, List.mem_range, decide_eq_true_eq,
      Prod.mk.injEq]
    constructor
    · rintro ⟨y, ⟨_, hy⟩, rfl, rfl⟩; exact ⟨hy, rfl⟩
    · rintro ⟨hy, rfl⟩; exact ⟨x, ⟨hx, hy⟩, rfl, rfl⟩
  have hl : den (packCt cmap) x c = if cmap x ≠ NONE ∧ c = cmap x then some 1 else none := by
    simp only [den, packCt]
    by_cases h : cmap x ≠ NONE
    · by_cases hc : c = cmap x
      · subst hc; simp [h]
      · have : (c == cmap x) = false := by simp [hc]
        simp [h, hc, List.lookup_cons, this]
    · simp [h]
  rw [hl]; unfold buildS
  split <;> split
  · rfl
  · next h1 h2 => exact absurd (key.2 h1) h2
  · next h1 h2 => exact absurd (key.1 h2) h1
  · rfl

theorem lookup_zip_range' (gw : List Int) (s k : Nat) :
    ((List.range' s gw.length).zip gw).lookup (s + k) = gw[k]? := by
  induction gw generalizing s k with
  | nil => simp
  | cons a l ih =>
    simp only [List.length_cons, List.range'_succ, List.zip_cons_cons, List.lookup_cons]
    cases k with
    | zero => simp
    | succ k =>
      have : (s + (k + 1) == s) = false := by simp
      rw [this]
      have := ih (s + 1) k
      rw [show s + 1 + k = s + (k + 1) by omega] at this
      simpa using this

/-- `pk`'s gather weights (m1_reuse_bench.rs:1382-1387): `pack_full_i32(gw)` stores `gw[k]` at every
`k < off`; `vec_i32(off, 0..off, gw)` (`GrB_Vector_build_INT32`, no duplicates) stores the value paired
with index `k`, which is the same. -/
theorem pk_weights (gw : List Int) (k : Nat) :
    ((List.range gw.length).zip gw).lookup k = gw[k]? := by
  rw [List.range_eq_range']
  simpa using lookup_zip_range' gw 0 k

end CchCoarsen
