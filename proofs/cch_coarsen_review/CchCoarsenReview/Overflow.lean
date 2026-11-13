import CchCoarsenReview.CpackGB
/-!
# Overflow: the `u32` offsets and ids and the `i32` values and weights fit

The host passes add `i32` values (`sm += tmp[q].1`, m1_coarsen_bench.rs:1423; `v += tmp[q].1`,
m1_reuse_bench.rs:879), `i32` weights (`wsum += wh[m]`, :1202; `ws[qi] += p.w[x]`,
m1_reuse_bench.rs:418) and write `u32` offsets (`cnt.push(cols.len() as u32)`) and ids (`nc - 1`,
`np + k`). The release profile has no overflow checks (Cargo.toml `[profile.release]`), so an overflow
would wrap silently. GraphBLAS's `PLUS_TIMES_INT32` is C `int32_t` arithmetic.

The argument, for the levels `hcon` builds: their values are nonnegative and their total never grows
(`hcon_total_le`), and the weights' total is conserved (`cpack_conserves`). Depth 0 is iso 1 (`to_grb`,
m1_coarsen_bench.rs:1515) with unit weights (:1577), so on those levels every value, partial sum, weight
and pair sum is at most `nnz(A_0)` or `2·n`, and every offset at most `nnz(A_0)`. B2 levels are bounded
only relative to their parent (`hb2_vals`, `prow_bounds`), and their offsets are not bounded: gap G7 in
the root header. `bench_sizes_fit` checks the three road graphs.
-/

namespace CchCoarsen

/-- The sizes after the loader (m1-optimize README "Data"): `(n, nnz)`. -/
def benchGraphs : List (Nat × Nat) := [(1965206, 5533214), (1088092, 3083796), (1379917, 3843320)]

/-- **B2's id spaces at depth `d` are at most `(d+1)·n`.** A child level's ids are `np + nparts`
(`let n = np + nparts`, m1_reuse_bench.rs:305-306, :451-452, :631-632; `b2_splice` :848-850), with `np` the
parent level's whole id space, holes included, and the parent at depth `d` is depth `d−1`'s B2 or capstop
hierarchy (:2304-2306). So each depth adds at most `nparts ≤ n` ids per level. The premise `nparts ≤ n`
(each part has a member, members are distinct live previous-level vertices, and a level has at most `n`
live vertices: weights are ≥ 1 and their total is the depth's live fine count) is not proved from the
numbering (gap G2). -/
theorem ids_bound (n : Nat) (f : Nat → Nat) (h0 : f 0 ≤ n) (hs : ∀ d, f (d + 1) ≤ f d + n) :
    ∀ d, f d ≤ (d + 1) * n := by
  intro d
  induction d with
  | zero => simpa using h0
  | succ d ih => have := hs d; rw [Nat.succ_mul]; omega

/-- **The bench's sizes fit**: values and their sums (`≤ nnz`) and pair weights (`≤ 2n`) below
`i32::MAX`; offsets (`≤ nnz`) below `2³²`; ids at the deepest depth `D = 14`, including B2's appended
part ids (`≤ (D+1)·n = 15n`, `ids_bound`), below `NONE = u32::MAX` (CA: 29,478,090). -/
theorem bench_sizes_fit : ∀ g ∈ benchGraphs,
    g.2 < 2 ^ 31 - 1 ∧ 2 * g.1 < 2 ^ 31 - 1 ∧ g.2 < 2 ^ 32 ∧ (14 + 1) * g.1 < NONE := by
  decide

/-- The merge keeps the total. -/
theorem grp_total (o : Option (Nat × Int)) (l : List (Nat × Int)) :
    lsum (grp o l) Prod.snd = (match o with | some g => g.2 | none => 0) + lsum l Prod.snd := by
  induction l generalizing o with
  | nil => cases o <;> simp [grp]
  | cons e l ih =>
    cases o with
    | none => simp only [grp]; rw [ih]; simp
    | some g =>
      obtain ⟨c, s⟩ := g
      simp only [grp]
      by_cases h : e.1 = c
      · simp only [h, ite_true]; rw [ih]; simp only [lsum_cons]; omega
      · simp only [h, ite_false, lsum_cons]; rw [ih]

theorem grp_nonneg (o : Option (Nat × Int)) (l : List (Nat × Int)) (ho : ∀ g, o = some g → 0 ≤ g.2)
    (hl : ∀ e ∈ l, 0 ≤ e.2) : ∀ x ∈ grp o l, 0 ≤ x.2 := by
  induction l generalizing o with
  | nil => cases o <;> simp [grp]; exact ho _ rfl
  | cons e l ih =>
    have he := hl e (by simp)
    have hl' := fun x hx => hl x (List.mem_cons_of_mem e hx)
    cases o with
    | none => simp only [grp]; exact ih (some e) (fun g hg => by cases hg; exact he) hl'
    | some g =>
      obtain ⟨c, s⟩ := g
      have hs := ho _ rfl
      simp only [grp]
      by_cases h : e.1 = c
      · simp only [h, ite_true]
        exact ih _ (fun g hg => by cases hg; simp at hs ⊢; omega) hl'
      · simp only [h, ite_false]
        intro x hx
        rcases List.mem_cons.1 hx with rfl | hx
        · exact hs
        · exact ih (some e) (fun g hg => by cases hg; exact he) hl' x hx

/-- Each stored value of a merged nonnegative list is at most the list's total. -/
theorem grp_val_le (l : List (Nat × Int)) (hl : ∀ e ∈ l, 0 ≤ e.2) :
    ∀ x ∈ grp none l, 0 ≤ x.2 ∧ x.2 ≤ lsum l Prod.snd := by
  intro x hx
  have hnn := grp_nonneg none l (by simp) hl
  refine ⟨hnn x hx, ?_⟩
  have htot := grp_total none l
  simp at htot
  rw [← htot]
  -- one term of a nonnegative sum is at most the sum
  have : ∀ (m : List (Nat × Int)), (∀ y ∈ m, 0 ≤ y.2) → x ∈ m → x.2 ≤ lsum m Prod.snd := by
    intro m hm hxm
    induction m with
    | nil => simp at hxm
    | cons y m ih =>
      simp only [lsum_cons]
      rcases List.mem_cons.1 hxm with rfl | h
      · have := lsum_nonneg (fun z hz => hm z (List.mem_cons_of_mem _ hz)); omega
      · have := ih (fun z hz => hm z (List.mem_cons_of_mem _ hz)) h
        have := hm y (by simp); omega
  exact this _ hnn hx

theorem tmp_total_le (A : Rows Int) (C : Nat → List Nat) (ca : Nat → Nat) (r : Nat)
    (hnn : ∀ i, ∀ e ∈ A i, 0 ≤ e.2) :
    lsum (hconTmp A C ca r) Prod.snd ≤ lsum (C r) (fun i => lsum (A i) Prod.snd) := by
  simp only [hconTmp, lsum_flatMap, lsum_filterMap]
  refine lsum_le (fun i _ => lsum_le (fun e he => ?_))
  by_cases h : ca e.1 = r
  · simp only [h, ne_eq, not_true_eq_false, ite_false]; exact hnn i e he
  · simp [h]

/-- **One coarse row**: every stored value of `hcon`'s row `r` is nonnegative and at most the total of
its members' rows; the row has at most as many entries as its members' rows together. -/
theorem hcon_row_bounds {A : Rows Int} {C : Nat → List Nat} {ca : Nat → Nat}
    {srt : List (Nat × Int) → List (Nat × Int)} (hS : SortSpec srt) (hnn : ∀ i, ∀ e ∈ A i, 0 ≤ e.2) (r : Nat) :
    (∀ x ∈ hconRows srt A C ca r, 0 ≤ x.2 ∧ x.2 ≤ lsum (C r) (fun i => lsum (A i) Prod.snd)) ∧
    (hconRows srt A C ca r).length ≤ ((C r).flatMap A).length := by
  have hnnT : ∀ e ∈ srt (hconTmp A C ca r), 0 ≤ e.2 := by
    intro e he
    have := ((hS _).1.mem_iff).1 he
    simp only [hconTmp, List.mem_flatMap, List.mem_filterMap] at this
    obtain ⟨i, _, e', he', hx⟩ := this
    by_cases h : ca e'.1 = r
    · simp [h] at hx
    · have hx' : (ca e'.1, e'.2) = e := by simpa [h] using hx
      rw [← hx']; exact hnn i e' he'
  refine ⟨fun x hx => ?_, ?_⟩
  · obtain ⟨h1, h2⟩ := grp_val_le _ hnnT x hx
    refine ⟨h1, Int.le_trans h2 ?_⟩
    rw [lsum_perm (hS _).1]
    exact tmp_total_le A C ca r hnn
  · refine Nat.le_trans (grp_none_length_le _) ?_
    rw [(hS _).1.length_eq]
    simp only [hconTmp]
    induction C r with
    | nil => simp
    | cons i l ih =>
      simp only [List.flatMap_cons, List.length_append]
      have := List.length_filterMap_le (fun e => if ca e.1 ≠ r then some (ca e.1, e.2) else none) (A i)
      omega

/-! ## The level totals -/

variable {n : Nat} {wh : Nat → Int} {mt : Nat → Nat}

/-- Summing over the coarse groups is summing over the live vertices. -/
theorem lsum_leaders_memb (hm : MatePre n wh mt) (f : Nat → Int) :
    lsum (leaders wh mt n) (fun ℓ => lsum (memb mt ℓ) f) =
      lsum (List.range n) (fun i => if live wh i then f i else 0) := by
  have hmlt : ∀ ℓ ∈ leaders wh mt n, ∀ i ∈ memb mt ℓ, i < n := by
    intro ℓ hℓ i hi
    rw [mem_leaders] at hℓ
    exact ((memb_mem hm hℓ.2 hℓ.1).1 hi).1
  calc lsum (leaders wh mt n) (fun ℓ => lsum (memb mt ℓ) f)
      = lsum (leaders wh mt n) (fun ℓ => lsum (List.range n) (fun i => if i ∈ memb mt ℓ then f i else 0)) := by
        refine lsum_congr (fun ℓ hℓ => ?_)
        rw [lsum_range_indicator (memb_nodup ℓ) (hmlt ℓ hℓ)]
    _ = lsum (List.range n) (fun i => lsum (leaders wh mt n) (fun ℓ => if i ∈ memb mt ℓ then f i else 0)) :=
        lsum_swap _ _ _
    _ = lsum (List.range n) (fun i => if live wh i then f i else 0) := by
        refine lsum_congr (fun i hi => ?_)
        have hin := List.mem_range.1 hi
        by_cases hl : live wh i
        · obtain ⟨hld, hlt⟩ := min_lead hm hin hl
          have : ∀ ℓ ∈ leaders wh mt n, (if i ∈ memb mt ℓ then f i else 0) =
              (if ℓ = min i (mt i) then f i else 0) := by
            intro ℓ hℓ
            rw [mem_leaders] at hℓ
            have := memb_mem (i := i) hm hℓ.2 hℓ.1
            by_cases h : ℓ = min i (mt i)
            · have hmm : i ∈ memb mt ℓ := this.2 ⟨hin, hl, h.symm⟩
              subst h; simp [hmm]
            · have : i ∉ memb mt ℓ := fun hm' => h ((this.1 hm').2.2).symm
              simp [h, this]
          rw [lsum_congr this, lsum_single (leaders_nodup n) (mem_leaders.2 ⟨hlt, hld⟩) (fun _ => f i)]
          simp [hl]
        · refine (lsum_zero (fun ℓ hℓ => ?_)).trans (by simp [hl])
          rw [mem_leaders] at hℓ
          have : i ∉ memb mt ℓ := fun hm' => hl ((memb_mem hm hℓ.2 hℓ.1).1 hm').2.1
          simp [this]

theorem lsum_range_getD {β : Type} (l : List β) (d : β) (g : β → Int) :
    lsum (List.range l.length) (fun r => g (l[r]?.getD d)) = lsum l g := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    simp only [List.length_cons, List.range_succ_eq_map, lsum_cons, List.getElem?_cons_zero,
      Option.getD_some, lsum_map, Nat.succ_eq_add_one, List.getElem?_cons_succ]
    rw [ih]

/-- **The total of `A_c`'s values is at most `A`'s**: with `cpack`'s `C`, the rows of `hcon`'s output
sum to at most the sum of `A`'s rows (nonnegative values). So no level's total exceeds depth 0's. -/
theorem hcon_total_le (hm : MatePre n wh mt) {A : Rows Int} {ca : Nat → Nat}
    {srt : List (Nat × Int) → List (Nat × Int)} (hS : SortSpec srt) (hnn : ∀ i, ∀ e ∈ A i, 0 ≤ e.2) :
    lsum (List.range (cpack n wh mt).nc) (fun r => lsum (hconRows srt A (cCpack n wh mt) ca r) Prod.snd) ≤
      lsum (List.range n) (fun i => lsum (A i) Prod.snd) := by
  have hrow : ∀ r, lsum (hconRows srt A (cCpack n wh mt) ca r) Prod.snd ≤
      lsum (cCpack n wh mt r) (fun i => lsum (A i) Prod.snd) := by
    intro r
    simp only [hconRows]
    have := grp_total none (srt (hconTmp A (cCpack n wh mt) ca r))
    simp only [Int.zero_add] at this
    rw [this, lsum_perm (hS _).1]
    exact tmp_total_le A _ ca r hnn
  refine Int.le_trans (lsum_le (fun r _ => hrow r)) ?_
  have h := cpack_inv hm
  have hlen : (cpack n wh mt).nc = (cpack n wh mt).crows.length := by
    rw [h.nc, h.crows, List.length_map]; rfl
  rw [hlen]
  simp only [cCpack]
  rw [lsum_range_getD (cpack n wh mt).crows [] (fun row => lsum row (fun i => lsum (A i) Prod.snd))]
  rw [h.crows, lsum_map, lsum_leaders_memb hm]
  refine lsum_le (fun i _ => ?_)
  by_cases hl : live wh i
  · simp [hl]
  · simp only [hl, ite_false]; exact lsum_nonneg (fun e he => hnn i e he)

/-- **Weights stay in range**: each `w_c` is positive and at most the total live weight, which is
conserved (`cpack_conserves`). -/
theorem cpack_wch_bounds (hm : MatePre n wh mt) {c : Nat} (hc : c < (cpack n wh mt).nc) :
    ∃ x, (cpack n wh mt).wch[c]? = some x ∧ 0 < x ∧
      x ≤ lsum (List.range n) (fun i => if live wh i then wh i else 0) := by
  obtain ⟨row, hrow, hw⟩ := cpack_wch hm hc
  obtain ⟨row', hrow', hne, hiff⟩ := cpack_rows_iff hm hc 0
  rw [hrow] at hrow'; cases hrow'
  refine ⟨lsum row wh, hw, ?_, ?_⟩
  · obtain ⟨i, hi⟩ := List.exists_mem_of_ne_nil row hne
    have hiff' : ∀ j, j ∈ row → live wh j := by
      intro j hj
      obtain ⟨r2, hr2, _, h2⟩ := cpack_rows_iff hm hc j
      rw [hrow] at hr2; cases hr2; exact (h2.1 hj).2.1
    have : ∀ (m : List Nat), (∀ j ∈ m, live wh j) → m ≠ [] → 0 < lsum m wh := by
      intro m hm' hne'
      induction m with
      | nil => exact absurd rfl hne'
      | cons a m ih =>
        simp only [lsum_cons]
        have ha : 0 < wh a := hm' a (by simp)
        have : 0 ≤ lsum m wh := lsum_nonneg (fun j hj => Int.le_of_lt (hm' j (List.mem_cons_of_mem _ hj)))
        omega
    exact this row hiff' hne
  · have hcons := cpack_conserves hm
    have h := cpack_inv hm
    -- `w_c(c)` is one term of the conserved total, all terms positive
    have hwl : ∀ x ∈ (cpack n wh mt).wch, 0 ≤ x := by
      intro x hx
      rw [h.wch, List.mem_map] at hx
      obtain ⟨ℓ, hℓ, rfl⟩ := hx
      rw [mem_leaders] at hℓ
      rw [wsum_eq]
      refine lsum_nonneg (fun j hj => Int.le_of_lt ((memb_mem hm hℓ.2 hℓ.1).1 hj).2.1)
    have hmem : lsum row wh ∈ (cpack n wh mt).wch := List.mem_of_getElem? hw
    rw [← hcons]
    have : ∀ (m : List Int), (∀ y ∈ m, 0 ≤ y) → ∀ x ∈ m, x ≤ lsum m id := by
      intro m hm' x hx
      induction m with
      | nil => simp at hx
      | cons y m ih =>
        simp only [lsum_cons, id]
        rcases List.mem_cons.1 hx with rfl | h'
        · have := lsum_nonneg (f := id) (fun z hz => hm' z (List.mem_cons_of_mem _ hz))
          omega
        · have := ih (fun z hz => hm' z (List.mem_cons_of_mem _ hz)) h'
          have := hm' y (by simp); omega
    exact this _ hwl _ hmem

end CchCoarsen
