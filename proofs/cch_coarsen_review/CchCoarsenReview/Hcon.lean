import CchCoarsenReview.Rows
/-!
# `hcon`: the contraction `A_c = select(OFFDIAG, C·A·Cᵀ)` as one host pass

**Rust** (m1opt.patch @7df8e5887): `host_contract`, m1_coarsen_bench.rs:1392-1437, called from
`contract_packed` (:1224-1225) when `hcon` is on. For coarse row `r` it walks C's row `r` (the
members `i`, :1409), each member's row of `A` (:1410), maps the column through `ca` and drops the
diagonal (:1411-1414), sorts by coarse column (:1417) and merges equal columns (:1418-1431).

**Replaced** (the switch off, :1226-1247): `GrB_mxm(tm, PLUS_TIMES_INT32, C, A)` (:1229),
`GrB_mxm(ac, PLUS_TIMES_INT32, tm, Cᵀ)` (:1235) and `GrB_select(ac, GrB_OFFDIAG)` (:1242). Here `C` is the packed
`c_rows` (nc × n, row `r` = its members, iso 1) and `Cᵀ` is the packed `ctp` (n × nc, row `i` =
`[ca i]` for a live `i`, iso 1); `Cpack.lean` proves those are what `cpack` builds.
-/

namespace CchCoarsen

/-- The members of each coarse vertex as an iso-1 CSR (`c_rows`, :1177-1179, packed :1220). -/
def cRowsOf (C : Nat → List Nat) : Rows Int := fun r => (C r).map (fun i => (i, (1 : Int)))

/-- `Cᵀ` as an iso-1 CSR: row `i` holds `ca i` when `i` is live (`ctp`, :1180-1182, packed :1221). -/
def ctRowsOf (live : Nat → Prop) [DecidablePred live] (ca : Nat → Nat) : Rows Int :=
  fun i => if live i then [(ca i, (1 : Int))] else []

/-- `tmp` for coarse row `r` (:1408-1416): every `(ca j, A(i,j))` with `i` a member of `r` and
`ca j ≠ r`, in the order read. -/
def hconTmp (A : Rows Int) (C : Nat → List Nat) (ca : Nat → Nat) (r : Nat) : List (Nat × Int) :=
  (C r).flatMap fun i => (A i).filterMap fun e => if ca e.1 ≠ r then some (ca e.1, e.2) else none

/-- `host_contract`'s output row `r`: sort by key (:1417), then merge (:1418-1431). `srt` is any
function meeting `sort_unstable_by_key`'s contract (`SortSpec`). -/
def hconRows (srt : List (Nat × Int) → List (Nat × Int)) (A : Rows Int) (C : Nat → List Nat)
    (ca : Nat → Nat) : Rows Int := fun r => grp none (srt (hconTmp A C ca r))

/-- The preconditions the pass relies on. `A` is the unloaded CSR (`Csr::new`: rows strictly
ascending after `GrB_wait`), square of size `n`; `C` lists each coarse vertex's members once; every
column of `A` is a live vertex. The last one is the verifier's `split` finding: `host_contract` maps a
dead column through `ca` to `NONE` (`u32::MAX`) and emits it, where GraphBLAS drops the entry
(`hcon_dead_column_differs`). `Hsplit.lean` discharges it for the split graph. -/
structure HconPre (n : Nat) (A : Rows Int) (C : Nat → List Nat) (live : Nat → Prop) : Prop where
  rowStrict : ∀ i, StrictRow (A i)
  colLt : ∀ i, ∀ e ∈ A i, e.1 < n
  cNodup : ∀ r, (C r).Nodup
  cLt : ∀ r, ∀ i ∈ C r, i < n
  colLive : ∀ i, ∀ e ∈ A i, live e.1

theorem den_cRowsOf (C : Nat → List Nat) (r i : Nat) :
    den (cRowsOf C) r i = if i ∈ C r then some 1 else none := by
  simp only [den, cRowsOf]
  induction C r with
  | nil => simp
  | cons a l ih =>
    simp only [List.map_cons, List.lookup_cons, List.mem_cons]
    by_cases h : i = a
    · subst h; simp
    · have : (i == a) = false := by simp [h]
      rw [this]; simp [h, ih]

theorem den_ctRowsOf (live : Nat → Prop) [DecidablePred live] (ca : Nat → Nat) (i c : Nat) :
    den (ctRowsOf live ca) i c = if live i ∧ ca i = c then some 1 else none := by
  simp only [den, ctRowsOf]
  by_cases hl : live i
  · by_cases hc : c = ca i
    · subst hc; simp [hl]
    · have : (c == ca i) = false := by simp [hc]
      simp [hl, List.lookup_cons, this, Ne.symm hc]
  · simp [hl]

theorem mem_keys_hconTmp (A : Rows Int) (C : Nat → List Nat) (ca : Nat → Nat) (r c : Nat) :
    c ∈ (hconTmp A C ca r).map (·.1) ↔ ∃ i ∈ C r, ∃ e ∈ A i, ca e.1 ≠ r ∧ ca e.1 = c := by
  simp only [hconTmp, List.mem_map, List.mem_flatMap, List.mem_filterMap]
  constructor
  · rintro ⟨x, ⟨i, hi, e, he, hx⟩, rfl⟩
    by_cases h : ca e.1 = r
    · simp [h] at hx
    · have hx' : (ca e.1, e.2) = x := by simpa [h] using hx
      exact ⟨i, hi, e, he, h, by rw [← hx']⟩
  · rintro ⟨i, hi, e, he, h1, h2⟩
    exact ⟨(ca e.1, e.2), ⟨i, hi, e, he, by simp [h1]⟩, h2⟩

theorem ksum_hconTmp (A : Rows Int) (C : Nat → List Nat) (ca : Nat → Nat) (r c : Nat) (hrc : r ≠ c) :
    ksum (hconTmp A C ca r) c = lsum (C r) (fun i => lsum (A i) (fun e => if ca e.1 = c then e.2 else 0)) := by
  simp only [ksum, hconTmp, lsum_flatMap, lsum_filterMap]
  refine lsum_congr (fun i _ => lsum_congr (fun e _ => ?_))
  by_cases h : ca e.1 = c
  · have : c ≠ r := Ne.symm hrc
    simp [h, this]
  · by_cases h' : ca e.1 ≠ r <;> simp [h, h']

/-- An `Option` is determined by when it is stored and its value. -/
theorem option_eq_ite {o : Option Int} {P : Prop} [Decidable P] {v : Int} (h1 : o.isSome ↔ P)
    (h2 : o.getD 0 = v) : o = if P then some v else none := by
  cases o with
  | none => have : ¬P := fun hp => by simpa using h1.2 hp
            simp [this]
  | some x => have : P := h1.1 rfl
              simp only [Option.getD_some] at h2; simp [this, h2]

/-- **`hcon` = `select(OFFDIAG, C·A·Cᵀ)`.** For every coarse row `r` and column `c`, the host pass's
row `r` stores at `c` exactly what `OFFDIAG(PLUS_TIMES(PLUS_TIMES(C, A), Cᵀ))` stores there, for any
order `sort_unstable_by_key` leaves equal keys in. -/
theorem hcon_eq {n : Nat} {A : Rows Int} {C : Nat → List Nat} {live : Nat → Prop} [DecidablePred live]
    {ca : Nat → Nat} (hp : HconPre n A C live) {srt : List (Nat × Int) → List (Nat × Int)}
    (hS : SortSpec srt) (r c : Nat) :
    den (hconRows srt A C ca) r c =
      offdiag (mxmPT n (mxmPT n (den (cRowsOf C)) (den A)) (den (ctRowsOf live ca))) r c := by
  show (grp none (srt (hconTmp A C ca r))).lookup c = _
  rw [(sortGrp_spec hS _).2 c]
  simp only [offdiag]
  by_cases hrc : r = c
  · subst hrc
    have : r ∉ (hconTmp A C ca r).map (·.1) := by
      rw [mem_keys_hconTmp]; rintro ⟨i, _, e, _, h1, h2⟩; exact h1 h2
    simp [this]
  · simp only [hrc, ite_false]
    symm
    refine option_eq_ite ?_ ?_
    · -- stored: some `t` reached from a member of `r` with `ca t = c`
      rw [mxmPT_isSome, mem_keys_hconTmp]
      constructor
      · rintro ⟨t, _, h1, h2⟩
        rw [mxmPT_isSome] at h1
        obtain ⟨i, _, hi, ht⟩ := h1
        rw [den_cRowsOf] at hi
        rw [den_ctRowsOf] at h2
        have hiC : i ∈ C r := by by_cases h : i ∈ C r <;> simp_all
        have h2' : live t ∧ ca t = c := by by_cases h : live t ∧ ca t = c <;> simp_all
        obtain ⟨v, hv⟩ := (lookup_isSome_iff (A i) t).1 ht
        exact ⟨i, hiC, (t, v), hv, by simp [h2'.2]; exact fun h => hrc (h ▸ rfl), h2'.2⟩
      · rintro ⟨i, hi, e, he, _, h2⟩
        refine ⟨e.1, hp.colLt i e he, ?_, ?_⟩
        · rw [mxmPT_isSome]
          refine ⟨i, hp.cLt r i hi, ?_, ?_⟩
          · show (den (cRowsOf C) r i).isSome; rw [den_cRowsOf]; simp [hi]
          · show ((A i).lookup e.1).isSome
            exact (lookup_isSome_iff (A i) e.1).2 ⟨e.2, by simpa using he⟩
        · show (den (ctRowsOf live ca) e.1 c).isSome
          rw [den_ctRowsOf]; simp [hp.colLive i e he, h2]
    · -- value: swap the two sums, then collapse the indicators
      rw [ksum_hconTmp A C ca r c hrc, mxmPT_getD]
      have hCt : ∀ t, ((den (ctRowsOf live ca)) t c).getD 0 = if live t ∧ ca t = c then 1 else 0 := by
        intro t; rw [den_ctRowsOf]; by_cases h : live t ∧ ca t = c <;> simp [h]
      have hCm : ∀ i, ((den (cRowsOf C)) r i).getD 0 = if i ∈ C r then 1 else 0 := by
        intro i; rw [den_cRowsOf]; by_cases h : i ∈ C r <;> simp [h]
      calc lsum (List.range n) (fun t => (mxmPT n (den (cRowsOf C)) (den A) r t).getD 0 *
              (den (ctRowsOf live ca) t c).getD 0)
          = lsum (List.range n) (fun t => lsum (List.range n) (fun i =>
              (if i ∈ C r then (A i).lookup t |>.getD 0 else 0) * (if live t ∧ ca t = c then 1 else 0))) := by
            refine lsum_congr (fun t _ => ?_)
            rw [mxmPT_getD, hCt, ← lsum_mul_right]
            refine lsum_congr (fun i _ => ?_)
            rw [hCm]; by_cases h : i ∈ C r <;> simp [h, den]
        _ = lsum (List.range n) (fun i => if i ∈ C r then lsum (List.range n) (fun t =>
              ((A i).lookup t).getD 0 * (if live t ∧ ca t = c then 1 else 0)) else 0) := by
            rw [lsum_swap]
            refine lsum_congr (fun i _ => ?_)
            by_cases h : i ∈ C r
            · simp [h]
            · simp [h, lsum_zero]
        _ = lsum (C r) (fun i => lsum (List.range n) (fun t =>
              ((A i).lookup t).getD 0 * (if live t ∧ ca t = c then 1 else 0))) :=
            lsum_range_indicator (hp.cNodup r) (hp.cLt r) _
        _ = lsum (C r) (fun i => lsum (A i) (fun e => if ca e.1 = c then e.2 else 0)) := by
            refine lsum_congr (fun i _ => ?_)
            rw [lsum_range_lookup (hp.rowStrict i).keys_nodup (hp.colLt i)]
            refine lsum_congr (fun e he => ?_)
            have hl := hp.colLive i e he
            by_cases h : ca e.1 = c <;> simp [h, hl]

/-- The output rows are strictly ascending (sorted, no duplicates) and have no diagonal. -/
theorem hcon_row_shape {A : Rows Int} {C : Nat → List Nat} {ca : Nat → Nat}
    {srt : List (Nat × Int) → List (Nat × Int)} (hS : SortSpec srt) (r : Nat) :
    StrictRow (hconRows srt A C ca r) ∧ ∀ x ∈ hconRows srt A C ca r, x.1 ≠ r := by
  refine ⟨(sortGrp_spec hS _).1, fun x hx hxr => ?_⟩
  have := sortGrp_keys hS _ x hx
  rw [mem_keys_hconTmp] at this
  obtain ⟨_, _, _, _, h1, h2⟩ := this
  exact h1 (h2.trans hxr)

/-- Every output column is some `ca j` (so `< nc` when `ca` maps live vertices below `nc`). -/
theorem hcon_cols {A : Rows Int} {C : Nat → List Nat} {ca : Nat → Nat}
    {srt : List (Nat × Int) → List (Nat × Int)} (hS : SortSpec srt) (r : Nat) (x : Nat × Int)
    (hx : x ∈ hconRows srt A C ca r) : ∃ i ∈ C r, ∃ e ∈ A i, ca e.1 = x.1 := by
  have := sortGrp_keys hS _ x hx
  rw [mem_keys_hconTmp] at this
  obtain ⟨i, hi, e, he, _, h2⟩ := this
  exact ⟨i, hi, e, he, h2⟩

/-- **`A_c` is symmetric in pattern when `A` is**, given that `C`'s rows are exactly the classes of `ca`
on the live vertices (what `cpack` builds: `cCpack_mem_iff`). The next level's matching (`KSpec.symm`)
and `hb2` (`B2Pre.pSymm`) need this. -/
theorem hcon_symm {A : Rows Int} {C : Nat → List Nat} {live : Nat → Prop} [DecidablePred live]
    {ca : Nat → Nat} {srt : List (Nat × Int) → List (Nat × Int)} (hS : SortSpec srt)
    (hC : ∀ r i, i ∈ C r ↔ (live i ∧ ca i = r))
    (hAsym : ∀ i j, ((A i).lookup j).isSome = true → ((A j).lookup i).isSome = true)
    (hlive : ∀ i, ∀ e ∈ A i, live e.1) (r c : Nat)
    (h : (den (hconRows srt A C ca) r c).isSome = true) : (den (hconRows srt A C ca) c r).isSome = true := by
  have key : ∀ a b, (den (hconRows srt A C ca) a b).isSome = true ↔
      ∃ i ∈ C a, ∃ e ∈ A i, ca e.1 ≠ a ∧ ca e.1 = b := by
    intro a b
    show ((grp none (srt (hconTmp A C ca a))).lookup b).isSome = true ↔ _
    rw [(sortGrp_spec hS _).2 b, ← mem_keys_hconTmp]
    by_cases hm : b ∈ (hconTmp A C ca a).map (·.1) <;> simp [hm]
  obtain ⟨i, hi, e, he, hne, hc⟩ := (key r c).1 h
  rw [key]
  have hir := (hC r i).1 hi
  obtain ⟨v, hv⟩ := (lookup_isSome_iff _ _).1
    (hAsym i e.1 ((lookup_isSome_iff _ _).2 ⟨e.2, by simpa using he⟩))
  refine ⟨e.1, (hC c e.1).2 ⟨hlive i e he, hc⟩, (i, v), hv, ?_, hir.2⟩
  show ca i ≠ c
  rw [hir.2]; intro h'; exact hne (hc.trans h'.symm)

/-- **The precondition is needed** (the verifier's `split` probe, m1-optimize.md "Verification"): a
column that is not live has no `Cᵀ` row, so GraphBLAS drops its entry, while `host_contract` maps it
through `ca` (here `NONE`) and emits it. One fine vertex `0` (coarse `0`), one dead vertex `1`
(`ca 1 = NONE = 2^32-1`), `A(0,1) = 1`. -/
theorem hcon_dead_column_differs :
    let A : Rows Int := fun i => if i = 0 then [(1, 1)] else []
    let C : Nat → List Nat := fun r => if r = 0 then [0] else []
    let ca : Nat → Nat := fun j => if j = 0 then 0 else 4294967295
    den (hconRows (fun l => l) A C ca) 0 4294967295 = some 1 ∧
      offdiag (mxmPT 2 (mxmPT 2 (den (cRowsOf C)) (den A)) (den (ctRowsOf (fun j => j = 0) ca))) 0 4294967295 = none := by
  constructor
  · decide
  · simp only [offdiag]
    simp only [show (0 : Nat) ≠ 4294967295 from by decide, ite_false]
    cases h : mxmPT 2 (mxmPT 2 (den (cRowsOf (fun r => if r = 0 then [0] else []))) (den fun i => if i = 0 then [(1, 1)] else []))
        (den (ctRowsOf (fun j => j = 0) fun j => if j = 0 then 0 else 4294967295)) 0 4294967295 with
    | none => rfl
    | some v =>
      have h' := congrArg Option.isSome h
      simp only [Option.isSome_some] at h'
      obtain ⟨t, ht, _, h2⟩ := mxmPT_isSome.1 h'
      rw [den_ctRowsOf] at h2
      rcases (by omega : t = 0 ∨ t = 1) with rfl | rfl <;> simp at h2

end CchCoarsen
