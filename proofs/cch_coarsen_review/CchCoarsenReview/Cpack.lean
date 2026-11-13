import CchCoarsenReview.Hcon
/-!
# `cpack` and `pk`: the leader numbering, `C`, `Cᵀ` and `w_c` from one pass over `mate`

**Rust** (m1opt.patch @7df8e5887): `contract_packed`, m1_coarsen_bench.rs:1174-1223 (`c_host`). One
pass over `i in 0..n` (:1189): a dead `i` (`wh[i] <= 0`, :1191) gets an empty `Cᵀ` row; a live `i`
whose mate is not below it is a leader (:1195): it opens coarse row `nc` with members `[i]` or
`[i, m]` (:1196-1205), pushes `w_c = wh[i] (+ wh[m])` (:1198, :1202, :1206) and takes id `nc - 1`
(:1207); a live `i` with `m < i` takes its mate's id `ca[m]` (:1209). Either way `Cᵀ` row `i` is
`[id]` (:1211-1213).

**Replaced** (the switch off, :1031-1112): `l = MIN(idx, mate)`; `lead = select(l == idx)`, masked
by `w`'s structure when some vertex is dead; `extractTuples(lead)`; `cid = build(li, 0..nc)`;
`ca = cid(l)` (`GxB_Vector_extract_Vector`); `Cᵀ = build_Scalar_Vector(live, ca)`;
`w_c = mxv(PLUS_TIMES, Cᵀ, w, GrB_DESC_T0)`.

**`pk`** (m1_reuse_bench.rs:808-837 `pack_ct`): a reused level's `Cᵀ` packed as row `x` =
`[cmap[x]]` when `cmap[x] ≠ NONE`, in place of `GxB_Matrix_build_Scalar(ct_i, ct_j)` (`build_level`,
:942-951; `ct_i/ct_j` filled by `number` at :355-365).
-/

namespace CchCoarsen

section Numbering

variable (n : Nat) (wh : Nat → Int) (mt : Nat → Nat)

/-- A vertex is live when its host weight is positive (`wh[i] <= 0` skips it, :1191). -/
def live (i : Nat) : Prop := 0 < wh i
instance : DecidablePred (live wh) := fun i => inferInstanceAs (Decidable (0 < wh i))

/-- A leader: live, and its mate is not below it (`m >= i`, :1195). -/
def lead (i : Nat) : Prop := live wh i ∧ i ≤ mt i
instance : DecidablePred (lead wh mt) := fun _ => inferInstanceAs (Decidable (_ ∧ _))

/-- The leaders below `k`, ascending. -/
def leaders (k : Nat) : List Nat := (List.range k).filter (fun i => decide (lead wh mt i))

/-- Number of leaders below `x`: the id the numbering gives leader `x`. -/
def rank (x : Nat) : Nat := (leaders wh mt x).length

/-- C's row of a leader (:1196-1203): `[ℓ]`, or `[ℓ, m]` with `m > ℓ`. -/
def memb (ℓ : Nat) : List Nat := if mt ℓ ≠ ℓ then [ℓ, mt ℓ] else [ℓ]

/-- `w_c` of a leader (:1198, :1202). -/
def wsum (ℓ : Nat) : Int := if mt ℓ ≠ ℓ then wh ℓ + wh (mt ℓ) else wh ℓ

/-- The coarse id of a vertex: the rank of the smaller of it and its mate. -/
def caOf (i : Nat) : Nat := rank wh mt (min i (mt i))

end Numbering

/-- The pass's state: `ca`, C's rows (`cp/cj`), Cᵀ's rows (`tp/tj`), `wch`, `nc`. -/
structure CpState where
  ca : Nat → Nat
  crows : List (List Nat)
  trows : List (List Nat)
  wch : List Int
  nc : Nat

/-- `u32::MAX`, the `NONE` id `ca` starts with (:1176). -/
def NONE : Nat := 4294967295

def upd (f : Nat → Nat) (i v : Nat) : Nat → Nat := fun x => if x = i then v else f x

/-- One iteration of the loop, :1189-1214. -/
def cpStep (wh : Nat → Int) (mt : Nat → Nat) (s : CpState) (i : Nat) : CpState :=
  if wh i ≤ 0 then { s with trows := s.trows ++ [[]] }
  else
    let m := mt i
    if i ≤ m then
      let id := s.nc
      { ca := upd s.ca i id, crows := s.crows ++ [if m ≠ i then [i, m] else [i]],
        trows := s.trows ++ [[id]], wch := s.wch ++ [if m ≠ i then wh i + wh m else wh i],
        nc := s.nc + 1 }
    else
      let id := s.ca m
      { s with ca := upd s.ca i id, trows := s.trows ++ [[id]] }

def cpInit : CpState := ⟨fun _ => NONE, [], [], [], 0⟩

/-- `c_host`: the loop over `0..n`. -/
def cpack (n : Nat) (wh : Nat → Int) (mt : Nat → Nat) : CpState := (List.range n).foldl (cpStep wh mt) cpInit

/-- What the numbering needs from the matching: `mate` is an involution on the live vertices, and the
mate of a live vertex is live and in range. `Matching.lean` proves it for the rounds and the 2-hop
pass (`mate_involution`), given that `A`'s pattern stays among live vertices. -/
def MatePre (n : Nat) (wh : Nat → Int) (mt : Nat → Nat) : Prop :=
  ∀ i, i < n → live wh i → mt i < n ∧ live wh (mt i) ∧ mt (mt i) = i

section Lemmas

variable {n : Nat} {wh : Nat → Int} {mt : Nat → Nat}

theorem leaders_succ (k : Nat) :
    leaders wh mt (k + 1) = leaders wh mt k ++ (if lead wh mt k then [k] else []) := by
  simp only [leaders, List.range_succ, List.filter_append]
  by_cases h : lead wh mt k <;> simp [h]

theorem rank_succ (k : Nat) : rank wh mt (k + 1) = rank wh mt k + (if lead wh mt k then 1 else 0) := by
  simp only [rank, leaders_succ, List.length_append]
  by_cases h : lead wh mt k <;> simp [h]

theorem mem_leaders {k x : Nat} : x ∈ leaders wh mt k ↔ x < k ∧ lead wh mt x := by
  simp [leaders]

theorem leaders_nodup (k : Nat) : (leaders wh mt k).Nodup :=
  List.nodup_range.sublist List.filter_sublist

theorem rank_mono {x y : Nat} (h : x ≤ y) : rank wh mt x ≤ rank wh mt y := by
  induction y with
  | zero => have : x = 0 := by omega
            subst this; exact Nat.le_refl _
  | succ y ih =>
    by_cases hx : x = y + 1
    · subst hx; exact Nat.le_refl _
    · have := ih (by omega); rw [rank_succ]; omega

theorem rank_lt {x y : Nat} (hl : lead wh mt x) (h : x < y) : rank wh mt x < rank wh mt y := by
  have h1 : rank wh mt (x + 1) = rank wh mt x + 1 := by rw [rank_succ]; simp [hl]
  have h2 := rank_mono (wh := wh) (mt := mt) (show x + 1 ≤ y by omega)
  omega

/-- The leader with rank `c` sits at index `c`. -/
theorem leaders_get_rank {ℓ k : Nat} (hl : lead wh mt ℓ) (hk : ℓ < k) :
    (leaders wh mt k)[rank wh mt ℓ]? = some ℓ := by
  induction k with
  | zero => omega
  | succ k ih =>
    rw [leaders_succ]
    by_cases hℓ : ℓ = k
    · subst hℓ
      rw [List.getElem?_append_right (by simp [rank])]
      simp [hl, rank]
    · rw [List.getElem?_append_left (by
        have := rank_lt (wh := wh) (mt := mt) hl (show ℓ < k by omega)
        simpa [rank] using this)]
      exact ih (by omega)

/-- Ranks of the leaders are exactly `0..rank k`: the `c`-th leader has rank `c`. -/
theorem rank_get {k c ℓ : Nat} (h : (leaders wh mt k)[c]? = some ℓ) : rank wh mt ℓ = c := by
  have hm := List.mem_of_getElem? h
  rw [mem_leaders] at hm
  have h2 := leaders_get_rank (wh := wh) (mt := mt) hm.2 hm.1
  have hc : c < (leaders wh mt k).length := by
    rcases List.getElem?_eq_some_iff.1 h with ⟨hc, _⟩; exact hc
  exact ((List.Nodup.getElem?_inj hc (leaders_nodup k)).1 (h.trans h2.symm)).symm

end Lemmas

/-- The loop's invariant after `0..k`, in closed form. -/
structure CpInv (n : Nat) (wh : Nat → Int) (mt : Nat → Nat) (k : Nat) (s : CpState) : Prop where
  nc : s.nc = rank wh mt k
  crows : s.crows = (leaders wh mt k).map (memb mt)
  wch : s.wch = (leaders wh mt k).map (wsum wh mt)
  trows : s.trows = (List.range k).map (fun i => if live wh i then [caOf wh mt i] else [])
  ca : ∀ i, i < k → live wh i → s.ca i = caOf wh mt i

theorem cpInv_step {n : Nat} {wh : Nat → Int} {mt : Nat → Nat} (hm : MatePre n wh mt) {k : Nat}
    (hk : k < n) {s : CpState} (h : CpInv n wh mt k s) : CpInv n wh mt (k + 1) (cpStep wh mt s k) := by
  unfold cpStep
  by_cases hd : wh k ≤ 0
  · have hnl : ¬ live wh k := by simp [live]; omega
    have hnlead : ¬ lead wh mt k := fun h => hnl h.1
    simp only [hd, ite_true]
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    · rw [h.nc, rank_succ]; simp [hnlead]
    · rw [h.crows, leaders_succ]; simp [hnlead]
    · rw [h.wch, leaders_succ]; simp [hnlead]
    · rw [h.trows, List.range_succ]; simp [hnl]
    · intro i hi hl
      have : i ≠ k := fun he => hnl (he ▸ hl)
      exact h.ca i (by omega) hl
  · have hl : live wh k := by simp [live]; omega
    simp only [hd, ite_false]
    by_cases hle : k ≤ mt k
    · have hld : lead wh mt k := ⟨hl, hle⟩
      have hca : caOf wh mt k = rank wh mt k := by simp [caOf, Nat.min_eq_left hle]
      simp only [hle, ite_true]
      refine ⟨?_, ?_, ?_, ?_, ?_⟩
      · show s.nc + 1 = _; rw [h.nc, rank_succ]; simp [hld]
      · show s.crows ++ _ = _; rw [h.crows, leaders_succ]
        simp only [hld, ite_true, List.map_append, List.map_cons, List.map_nil, memb]
      · show s.wch ++ _ = _; rw [h.wch, leaders_succ]
        simp only [hld, ite_true, List.map_append, List.map_cons, List.map_nil, wsum]
      · show s.trows ++ _ = _; rw [h.trows, List.range_succ]; simp [hl, hca, h.nc]
      · intro i hi hli
        show upd s.ca k s.nc i = _
        by_cases hik : i = k
        · subst hik; simp [upd, hca, h.nc]
        · simp only [upd, hik, ite_false]; exact h.ca i (by omega) hli
    · have hlt : mt k < k := by omega
      have hnlead : ¬ lead wh mt k := fun h => hle h.2
      obtain ⟨_, hml, hmm⟩ := hm k hk hl
      have hcam : s.ca (mt k) = caOf wh mt k := by
        rw [h.ca (mt k) hlt hml]
        simp only [caOf, hmm]
        rw [Nat.min_comm]
      simp only [hle, ite_false]
      refine ⟨?_, ?_, ?_, ?_, ?_⟩
      · show s.nc = _; rw [h.nc, rank_succ]; simp [hnlead]
      · show s.crows = _; rw [h.crows, leaders_succ]; simp [hnlead]
      · show s.wch = _; rw [h.wch, leaders_succ]; simp [hnlead]
      · show s.trows ++ _ = _; rw [h.trows, List.range_succ]; simp [hl, hcam]
      · intro i hi hli
        show upd s.ca k (s.ca (mt k)) i = _
        by_cases hik : i = k
        · subst hik; simp [upd, hcam]
        · simp only [upd, hik, ite_false]; exact h.ca i (by omega) hli

theorem cpInv_range {n : Nat} {wh : Nat → Int} {mt : Nat → Nat} (hm : MatePre n wh mt) :
    ∀ k, k ≤ n → CpInv n wh mt k ((List.range k).foldl (cpStep wh mt) cpInit) := by
  intro k
  induction k with
  | zero => intro _; exact ⟨rfl, rfl, rfl, rfl, fun i hi => absurd hi (Nat.not_lt_zero _)⟩
  | succ k ih =>
    intro hk
    rw [List.range_succ, List.foldl_append]
    exact cpInv_step hm (by omega) (ih (by omega))

/-- The closed form of `cpack`'s output. -/
theorem cpack_inv {n : Nat} {wh : Nat → Int} {mt : Nat → Nat} (hm : MatePre n wh mt) :
    CpInv n wh mt n (cpack n wh mt) := cpInv_range hm n (Nat.le_refl _)

section Results

variable {n : Nat} {wh : Nat → Int} {mt : Nat → Nat}

theorem min_lead (hm : MatePre n wh mt) {i : Nat} (hi : i < n) (hl : live wh i) :
    lead wh mt (min i (mt i)) ∧ min i (mt i) < n := by
  obtain ⟨hmn, hml, hmm⟩ := hm i hi hl
  by_cases h : i ≤ mt i
  · rw [Nat.min_eq_left h]; exact ⟨⟨hl, h⟩, hi⟩
  · rw [Nat.min_eq_right (by omega)]; exact ⟨⟨hml, by rw [hmm]; omega⟩, hmn⟩

/-- **The ids are in range**: every live vertex's coarse id is below `nc`. -/
theorem cpack_ca_lt (hm : MatePre n wh mt) {i : Nat} (hi : i < n) (hl : live wh i) :
    (cpack n wh mt).ca i < (cpack n wh mt).nc := by
  have h := cpack_inv hm
  rw [h.ca i hi hl, h.nc]
  obtain ⟨hld, hlt⟩ := min_lead hm hi hl
  exact rank_lt hld hlt

/-- **`Cᵀ` has exactly one entry per live row, and none per dead row**: row `i` is `[ca i]`. -/
theorem cpack_ct_rows (hm : MatePre n wh mt) {i : Nat} (hi : i < n) :
    (cpack n wh mt).trows[i]? = some (if live wh i then [(cpack n wh mt).ca i] else []) := by
  have h := cpack_inv hm
  rw [h.trows, List.getElem?_map, List.getElem?_range hi]
  by_cases hl : live wh i
  · simp [hl, h.ca i hi hl]
  · simp [hl]

theorem memb_mem (hm : MatePre n wh mt) {ℓ i : Nat} (hℓ : lead wh mt ℓ) (hℓn : ℓ < n) :
    i ∈ memb mt ℓ ↔ (i < n ∧ live wh i ∧ min i (mt i) = ℓ) := by
  obtain ⟨hmn, hml, hmm⟩ := hm ℓ hℓn hℓ.1
  have hle := hℓ.2
  unfold memb
  by_cases he : mt ℓ = ℓ
  · simp only [he, ne_eq, not_true_eq_false, ite_false, List.mem_singleton]
    constructor
    · rintro rfl; exact ⟨hℓn, hℓ.1, by rw [he]; simp⟩
    · rintro ⟨hi, hli, hmin⟩
      obtain ⟨_, _, hmi⟩ := hm i hi hli
      by_cases h : i ≤ mt i
      · rw [Nat.min_eq_left h] at hmin; exact hmin
      · rw [Nat.min_eq_right (by omega)] at hmin
        have : mt (mt i) = mt i := by rw [hmin]; exact he
        rw [hmi] at this; omega
  · simp only [he, ne_eq, not_false_eq_true, ite_true, List.mem_cons,
      List.not_mem_nil, or_false]
    constructor
    · rintro (rfl | rfl)
      · exact ⟨hℓn, hℓ.1, Nat.min_eq_left hle⟩
      · exact ⟨hmn, hml, by rw [hmm]; exact Nat.min_eq_right hle⟩
    · rintro ⟨hi, hli, hmin⟩
      obtain ⟨_, _, hmi⟩ := hm i hi hli
      by_cases h : i ≤ mt i
      · rw [Nat.min_eq_left h] at hmin; exact Or.inl hmin
      · rw [Nat.min_eq_right (by omega)] at hmin
        right; rw [← hmin, hmi]

/-- **The numbering is a bijection from the matched groups onto `0..nc`.** Coarse row `c < nc` holds
exactly the live vertices whose id is `c`; it is never empty (it holds its leader); and every live
vertex is in the row of its id. -/
theorem cpack_rows_iff (hm : MatePre n wh mt) {c : Nat} (hc : c < (cpack n wh mt).nc) (i : Nat) :
    ∃ row, (cpack n wh mt).crows[c]? = some row ∧ row ≠ [] ∧
      (i ∈ row ↔ i < n ∧ live wh i ∧ (cpack n wh mt).ca i = c) := by
  have h := cpack_inv hm
  rw [h.nc] at hc
  obtain ⟨ℓ, hℓ⟩ : ∃ ℓ, (leaders wh mt n)[c]? = some ℓ := by
    have : c < (leaders wh mt n).length := hc
    exact ⟨_, List.getElem?_eq_getElem this⟩
  have hmem := List.mem_of_getElem? hℓ
  rw [mem_leaders] at hmem
  have hr := rank_get hℓ
  refine ⟨memb mt ℓ, by rw [h.crows, List.getElem?_map, hℓ]; rfl, ?_, ?_⟩
  · unfold memb; split <;> simp
  · rw [memb_mem hm hmem.2 hmem.1]
    constructor
    · rintro ⟨hi, hli, hmin⟩
      refine ⟨hi, hli, ?_⟩
      rw [h.ca i hi hli, caOf, hmin, hr]
    · rintro ⟨hi, hli, hca⟩
      refine ⟨hi, hli, ?_⟩
      rw [h.ca i hi hli, caOf] at hca
      obtain ⟨hld, hlt⟩ := min_lead hm hi hli
      have h1 := leaders_get_rank hld hlt
      rw [hca, hℓ] at h1
      exact (Option.some.inj h1).symm

theorem memb_nodup (ℓ : Nat) : (memb mt ℓ).Nodup := by
  unfold memb
  by_cases he : mt ℓ = ℓ
  · simp [he]
  · simp [he, Ne.symm he]

/-- `w_c` of row `c` is the sum of its members' weights. -/
theorem cpack_wch (hm : MatePre n wh mt) {c : Nat} (hc : c < (cpack n wh mt).nc) :
    ∃ row, (cpack n wh mt).crows[c]? = some row ∧ (cpack n wh mt).wch[c]? = some (lsum row wh) := by
  have h := cpack_inv hm
  rw [h.nc] at hc
  obtain ⟨ℓ, hℓ⟩ : ∃ ℓ, (leaders wh mt n)[c]? = some ℓ :=
    ⟨_, List.getElem?_eq_getElem (show c < (leaders wh mt n).length from hc)⟩
  refine ⟨memb mt ℓ, by rw [h.crows, List.getElem?_map, hℓ]; rfl, ?_⟩
  rw [h.wch, List.getElem?_map, hℓ]
  simp only [Option.map_some, Option.some.injEq, wsum, memb]
  by_cases he : mt ℓ = ℓ <;> simp [he]

/-- The weight vector `w` the GraphBLAS path reads, from the host array: an entry exactly at the
live vertices (`dense_i32`, :719-731; `ones_vec`, m1_reuse_bench.rs:1948-1958). -/
def wvec (wh : Nat → Int) : Mat Int := fun i _ => if live wh i then some (wh i) else none

/-- **`w_c = Cᵀᵀ·w`**: the host's `w_c` of row `c` equals `mxv(PLUS_TIMES, Cᵀ, w, GrB_DESC_T0)` at
`c` (the switch-off `Cw`, :1106-1112), with `Cᵀ` the matrix `cpack` packs. -/
theorem cpack_wc_eq_mxv (hm : MatePre n wh mt) {c : Nat} (hc : c < (cpack n wh mt).nc) :
    ∃ row, (cpack n wh mt).crows[c]? = some row ∧
      mxmPT n (tr (den (ctRowsOf (live wh) (cpack n wh mt).ca))) (wvec wh) c 0 = some (lsum row wh) := by
  obtain ⟨row, hrow, hne, hiff⟩ := cpack_rows_iff hm hc 0
  refine ⟨row, hrow, ?_⟩
  have hiff' : ∀ i, i ∈ row ↔ i < n ∧ live wh i ∧ (cpack n wh mt).ca i = c := fun i => by
    obtain ⟨row', hrow', _, h'⟩ := cpack_rows_iff hm hc i
    rw [hrow] at hrow'; cases hrow'; exact h'
  have hnd : row.Nodup := by
    have h := cpack_inv hm
    rw [h.nc] at hc
    obtain ⟨ℓ, hℓ⟩ : ∃ ℓ, (leaders wh mt n)[c]? = some ℓ :=
      ⟨_, List.getElem?_eq_getElem (show c < (leaders wh mt n).length from hc)⟩
    rw [h.crows, List.getElem?_map, hℓ] at hrow
    cases hrow; exact memb_nodup ℓ
  refine (option_eq_ite (P := True) (v := lsum row wh) ?_ ?_).trans (by simp)
  · rw [mxmPT_isSome]
    simp only [iff_true]
    obtain ⟨i, hi⟩ := List.exists_mem_of_ne_nil row hne
    obtain ⟨hin, hil, hica⟩ := (hiff' i).1 hi
    refine ⟨i, hin, ?_, ?_⟩
    · show (den (ctRowsOf (live wh) (cpack n wh mt).ca) i c).isSome
      rw [den_ctRowsOf]; simp [hil, hica]
    · simp [wvec, hil]
  · rw [mxmPT_getD]
    have hpt : ∀ t ∈ List.range n,
        (tr (den (ctRowsOf (live wh) (cpack n wh mt).ca)) c t).getD 0 * (wvec wh t 0).getD 0 =
        if t ∈ row then wh t else 0 := by
      intro t htr
      have htn : t < n := List.mem_range.1 htr
      show (den (ctRowsOf (live wh) (cpack n wh mt).ca) t c).getD 0 * _ = _
      rw [den_ctRowsOf]
      by_cases ht : t ∈ row
      · obtain ⟨_, hl, hca⟩ := (hiff' t).1 ht; simp [ht, hl, hca, wvec]
      · have : ¬ (live wh t ∧ (cpack n wh mt).ca t = c) :=
          fun ⟨hl, hca⟩ => ht ((hiff' t).2 ⟨htn, hl, hca⟩)
        simp [this, ht]
    rw [lsum_congr hpt, lsum_range_indicator hnd (fun i hi => ((hiff' i).1 hi).1)]

end Results

end CchCoarsen
