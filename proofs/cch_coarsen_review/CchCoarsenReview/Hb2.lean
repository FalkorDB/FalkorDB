import CchCoarsenReview.Hsplit
/-!
# `hb2`: the B2 level in one pass over the parent level, with the straddler rows spliced in

**Rust** (m1opt.patch @7df8e5887): `b2_splice`, m1_reuse_bench.rs:843-930.
* Part rows (:861-888): part `s = np + k` with members `part_mem[k]` (vertices of the child's
  previous level); `jj = cmap[j]` for each entry `(j, v)` of a member's row in `p.a`, kept when
  `jj ≠ NONE && jj ≠ s` (:865-871); sort, merge (:873-883). Each merged `(jj, v)` with `jj < np` is
  also pushed as `(jj, s, v)` to `tcol` (:884-886), sorted by `(jj, s)` (:890).
* Rows `i < np` (:902-917): when `cell[i] ≠ SEP`, the parent's row `i` keeps its entries `(q2, v)` with
  `cell[q2] == cell[i]` (:903-912); then `tcol`'s entries for row `i` are appended (:913-917).
* Rows `i ≥ np` (:918-922): the part rows.

**Replaced** (the switch off, `build_level`'s `Mode::B2`, :976-1032):
`M = build(cl_q, cl_c)` (the clean parent ids and their cells); `z = par.a⟨struct⟩ = M ANY_PAIR Mᵀ`;
`x = eWiseMult(FIRST, par.a, z)`; resize to `n × n`; `cs = build(ps_i, ps_j)` (each straddler member
and its part); `R = select(OFFDIAG, (csᵀ·p.a)·Cᵀ)` over `PLUS_TIMES`; `y = eWiseAdd(FIRST, R, Rᵀ)`;
the level is `eWiseAdd(FIRST, x, y)`.

So **B2's level** is: the parent's level restricted to pairs of clean ids in one cell, plus the
straddler rows `R` and their transposes. `hb2_eq` proves the host pass equal to it at every coordinate.
-/

namespace CchCoarsen

/-! ## The contraction with dropped columns -/

/-- `tmp` with a liveness guard on the column (B2's `jj != NONE`, :868). -/
def contrTmp (live : Nat → Prop) [DecidablePred live] (A : Rows Int) (C : Nat → List Nat) (ca : Nat → Nat)
    (r : Nat) : List (Nat × Int) :=
  (C r).flatMap fun i => (A i).filterMap fun e => if live e.1 ∧ ca e.1 ≠ r then some (ca e.1, e.2) else none

theorem mem_keys_contrTmp (live : Nat → Prop) [DecidablePred live] (A : Rows Int) (C : Nat → List Nat)
    (ca : Nat → Nat) (r c : Nat) :
    c ∈ (contrTmp live A C ca r).map (·.1) ↔ ∃ i ∈ C r, ∃ e ∈ A i, live e.1 ∧ ca e.1 ≠ r ∧ ca e.1 = c := by
  simp only [contrTmp, List.mem_map, List.mem_flatMap, List.mem_filterMap]
  constructor
  · rintro ⟨x, ⟨i, hi, e, he, hx⟩, rfl⟩
    by_cases h : live e.1 ∧ ca e.1 ≠ r
    · have hx' : (ca e.1, e.2) = x := by simpa [h] using hx
      exact ⟨i, hi, e, he, h.1, h.2, by rw [← hx']⟩
    · simp [h] at hx
  · rintro ⟨i, hi, e, he, h0, h1, h2⟩
    exact ⟨(ca e.1, e.2), ⟨i, hi, e, he, by simp [h0, h1]⟩, h2⟩

theorem ksum_contrTmp (live : Nat → Prop) [DecidablePred live] (A : Rows Int) (C : Nat → List Nat)
    (ca : Nat → Nat) (r c : Nat) (hrc : r ≠ c) :
    ksum (contrTmp live A C ca r) c =
      lsum (C r) (fun i => lsum (A i) (fun e => if live e.1 ∧ ca e.1 = c then e.2 else 0)) := by
  simp only [ksum, contrTmp, lsum_flatMap, lsum_filterMap]
  refine lsum_congr (fun i _ => lsum_congr (fun e _ => ?_))
  by_cases h : live e.1 ∧ ca e.1 = c
  · have : c ≠ r := Ne.symm hrc
    simp [h.1, h.2, this]
  · by_cases h' : live e.1 ∧ ca e.1 ≠ r
    · have : ¬ ca e.1 = c := fun hc => h ⟨h'.1, hc⟩
      simp [h', this]
    · simp [h', h]

/-- What the contraction needs: `A` is an unloaded `n`-column CSR, `C`'s rows list distinct members. -/
structure ContrPre (n : Nat) (A : Rows Int) (C : Nat → List Nat) : Prop where
  rowStrict : ∀ i, StrictRow (A i)
  colLt : ∀ i, ∀ e ∈ A i, e.1 < n
  cNodup : ∀ r, (C r).Nodup
  cLt : ∀ r, ∀ i ∈ C r, i < n

/-- **The guarded contraction = `select(OFFDIAG, C·A·Cᵀ)`**, with `Cᵀ` empty on the dropped columns: no
precondition on `A`'s columns. (`hcon_eq` is the unguarded pass, which needs every column live.) -/
theorem contr_eq {n : Nat} {A : Rows Int} {C : Nat → List Nat} (hp : ContrPre n A C)
    (live : Nat → Prop) [DecidablePred live] (ca : Nat → Nat)
    {srt : List (Nat × Int) → List (Nat × Int)} (hS : SortSpec srt) (r c : Nat) :
    (grp none (srt (contrTmp live A C ca r))).lookup c =
      offdiag (mxmPT n (mxmPT n (den (cRowsOf C)) (den A)) (den (ctRowsOf live ca))) r c := by
  rw [(sortGrp_spec hS _).2 c]
  simp only [offdiag]
  by_cases hrc : r = c
  · subst hrc
    have : r ∉ (contrTmp live A C ca r).map (·.1) := by
      rw [mem_keys_contrTmp]; rintro ⟨i, _, e, _, _, h1, h2⟩; exact h1 h2
    simp [this]
  · simp only [hrc, ite_false]
    symm
    refine option_eq_ite ?_ ?_
    · rw [mxmPT_isSome, mem_keys_contrTmp]
      constructor
      · rintro ⟨t, _, h1, h2⟩
        rw [mxmPT_isSome] at h1
        obtain ⟨i, _, hi, ht⟩ := h1
        rw [den_cRowsOf] at hi
        rw [den_ctRowsOf] at h2
        have hiC : i ∈ C r := by by_cases h : i ∈ C r <;> simp_all
        have h2' : live t ∧ ca t = c := by by_cases h : live t ∧ ca t = c <;> simp_all
        obtain ⟨v, hv⟩ := (lookup_isSome_iff (A i) t).1 ht
        exact ⟨i, hiC, (t, v), hv, h2'.1, by simp [h2'.2]; exact fun h => hrc (h ▸ rfl), h2'.2⟩
      · rintro ⟨i, hi, e, he, h0, _, h2⟩
        refine ⟨e.1, hp.colLt i e he, ?_, ?_⟩
        · rw [mxmPT_isSome]
          refine ⟨i, hp.cLt r i hi, ?_, ?_⟩
          · show (den (cRowsOf C) r i).isSome; rw [den_cRowsOf]; simp [hi]
          · show ((A i).lookup e.1).isSome
            exact (lookup_isSome_iff (A i) e.1).2 ⟨e.2, by simpa using he⟩
        · show (den (ctRowsOf live ca) e.1 c).isSome
          rw [den_ctRowsOf]; simp [h0, h2]
    · rw [ksum_contrTmp live A C ca r c hrc, mxmPT_getD]
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
        _ = lsum (C r) (fun i => lsum (A i) (fun e => if live e.1 ∧ ca e.1 = c then e.2 else 0)) := by
            refine lsum_congr (fun i _ => ?_)
            rw [lsum_range_lookup (hp.rowStrict i).keys_nodup (hp.colLt i)]
            refine lsum_congr (fun e _ => ?_)
            by_cases h : live e.1 ∧ ca e.1 = c <;> simp [h]

/-! ## The B2 level -/

/-- The numbering's data `b2_splice` reads (`Num`, m1_reuse_bench.rs:206-229): `np` parent ids, `n`
child ids (`np` plus the parts), `cmap` (previous-level id → child id or `NONE`), `cell` (a clean
parent id's cell, `SEP` for holes; :324-325) and each part's members (`part_start/part_mem`). -/
structure B2Num where
  np : Nat
  n : Nat
  cmap : Nat → Nat
  cell : Nat → Nat
  mem : Nat → List Nat

/-- A previous-level vertex is kept when `cmap` maps it (`jj != NONE`). -/
def kept (N : B2Num) (j : Nat) : Prop := N.cmap j ≠ NONE
instance (N : B2Num) : DecidablePred (kept N) := fun _ => inferInstanceAs (Decidable (_ ≠ _))

/-- Part row `s` (:861-883). -/
def prow (srt : List (Nat × Int) → List (Nat × Int)) (pA : Rows Int) (N : B2Num) (s : Nat) : List (Nat × Int) :=
  grp none (srt (contrTmp (kept N) pA N.mem N.cmap s))

/-- `tcol`'s entries for row `i` (:884-890, read at :899 and :913-917): `(s, v)` for every part `s` whose row has
`(i, v)`, in ascending `s` (`tcol` is sorted by the unique key `(jj, s)`). -/
def tcolRow (srt : List (Nat × Int) → List (Nat × Int)) (pA : Rows Int) (N : B2Num) (i : Nat) : List (Nat × Int) :=
  (List.range' N.np (N.n - N.np)).filterMap (fun s => ((prow srt pA N s).lookup i).map (fun v => (s, v)))

/-- `b2_splice`'s row `i` (:897-925, the closure passed to `emit_csr`). -/
def b2Rows (srt : List (Nat × Int) → List (Nat × Int)) (parA pA : Rows Int) (N : B2Num) : Rows Int := fun i =>
  if i < N.np then
    (if N.cell i ≠ SEP then (parA i).filter (fun e => decide (N.cell e.1 = N.cell i)) else []) ++ tcolRow srt pA N i
  else prow srt pA N i

/-- `M`'s pattern (n_p × ncells): the clean parent ids and their cells (`cl_q/cl_c`, :324-331: a parent id
is clean exactly when the numbering gives it a cell, `cell ≠ SEP`). -/
def mB2 (N : B2Num) : GBWDelta.Pat := fun q c => decide (q < N.np ∧ N.cell q ≠ SEP ∧ N.cell q = c)

/-- `z = par.a⟨struct⟩ = M ANY_PAIR Mᵀ` (`m_build`, :979-984; `split_mxm`, :985-990). -/
def zB2 (ncells : Nat) (parA : Mat Int) (N : B2Num) : Mat Int := fun q q' =>
  if (parA q q').isSome && GBWDelta.mxmAny ncells (mB2 N) (fun c q' => mB2 N q' c) q q' then some 1 else none

/-- `x = eWiseMult(FIRST, par.a, z)` (`split_mult`, :992-997), then resized (:999-1004; no entry changes). -/
def xB2 (ncells : Nat) (parA : Mat Int) (N : B2Num) : Mat Int := ewMult (fun a _ => a) parA (zB2 ncells parA N)

/-- `R = select(OFFDIAG, (csᵀ·p.a)·Cᵀ)` (`strad`, :1009-1015): `cs` holds each part's members
(`ps_i/ps_j`), `Cᵀ` is `cmap` (`pk_eq_build`). -/
def RGB (nprev : Nat) (pA : Mat Int) (N : B2Num) : Mat Int :=
  offdiag (mxmPT nprev (mxmPT nprev (den (cRowsOf N.mem)) pA) (den (ctRowsOf (kept N) N.cmap)))

/-- The switch-off level (:976-1032): `y = eWiseAdd(FIRST, R, Rᵀ)` (:1020-1021), then
`eWiseAdd(FIRST, x, y)` (`union`, :1025-1030). -/
def b2GB (ncells nprev : Nat) (parA pA : Mat Int) (N : B2Num) : Mat Int :=
  ewAdd (fun a _ => a) (xB2 ncells parA N) (ewAdd (fun a _ => a) (RGB nprev pA N) (tr (RGB nprev pA N)))

/-- The preconditions. `p.a` (the child's previous level) is symmetric; the parent's level is an
`np × np` CSR; part `s`'s members are exactly the previous-level vertices `cmap` sends to `s`; clean
cells are below `ncells`. -/
structure B2Pre (ncells nprev : Nat) (parA pA : Rows Int) (N : B2Num) : Prop where
  parStrict : ∀ q, StrictRow (parA q)
  parColLt : ∀ q, ∀ e ∈ parA q, e.1 < N.np
  parBeyond : ∀ q, N.np ≤ q → parA q = []
  pStrict : ∀ x, StrictRow (pA x)
  pColLt : ∀ x, ∀ e ∈ pA x, e.1 < nprev
  pSymm : ∀ x y, ((pA x).lookup y).isSome = true → ((pA y).lookup x).isSome = true
  memNodup : ∀ s, (N.mem s).Nodup
  memIff : ∀ s x, x ∈ N.mem s ↔ (N.np ≤ s ∧ s < N.n ∧ x < nprev ∧ N.cmap x = s)
  cmapLt : ∀ x, N.cmap x ≠ NONE → N.cmap x < N.n
  nLtNone : N.n ≤ NONE
  npLe : N.np ≤ N.n
  cellLt : ∀ q, q < N.np → N.cell q ≠ SEP → N.cell q < ncells

section
variable {ncells nprev : Nat} {parA pA : Rows Int} {N : B2Num} {srt : List (Nat × Int) → List (Nat × Int)}

theorem B2Pre.contr (hp : B2Pre ncells nprev parA pA N) : ContrPre nprev pA N.mem :=
  ⟨hp.pStrict, hp.pColLt, hp.memNodup, fun s x hx => ((hp.memIff s x).1 hx).2.2.1⟩

/-- `R`, the straddler rows, of `p.a`'s CSR. -/
abbrev Rm (nprev : Nat) (pA : Rows Int) (N : B2Num) : Mat Int := RGB nprev (den pA) N

theorem prow_eq (hp : B2Pre ncells nprev parA pA N) (hS : SortSpec srt) (s c : Nat) :
    (prow srt pA N s).lookup c = Rm nprev pA N s c :=
  contr_eq hp.contr (kept N) N.cmap hS s c

/-- `R(s, t)` is stored iff some member of `s` has a kept neighbour mapped to `t ≠ s`. -/
theorem Rm_isSome (hp : B2Pre ncells nprev parA pA N) (hS : SortSpec srt) (s t : Nat) :
    (Rm nprev pA N s t).isSome = true ↔
      ∃ x ∈ N.mem s, ∃ e ∈ pA x, kept N e.1 ∧ N.cmap e.1 ≠ s ∧ N.cmap e.1 = t := by
  rw [← prow_eq hp hS, prow, (sortGrp_spec hS _).2 t, ← mem_keys_contrTmp]
  by_cases h : t ∈ (contrTmp (kept N) pA N.mem N.cmap s).map (·.1) <;> simp [h]

/-- No straddler row below `np`: those ids are parent ids, not parts. -/
theorem Rm_low (hp : B2Pre ncells nprev parA pA N) (hS : SortSpec srt) {s : Nat} (hs : s < N.np) (t : Nat) :
    Rm nprev pA N s t = none := by
  cases h : Rm nprev pA N s t with
  | none => rfl
  | some v =>
    have := (Rm_isSome hp hS s t).1 (by simp [h])
    obtain ⟨x, hx, _⟩ := this
    have := ((hp.memIff s x).1 hx).1; omega

/-- **`R` is symmetric in pattern on the part rows** (from `p.a`'s symmetry): for a part `s`, if `R(t, s)`
is stored so is `R(s, t)`. So `eWiseAdd(FIRST, R, Rᵀ)` on a part row is `R`'s row. -/
theorem Rm_symm (hp : B2Pre ncells nprev parA pA N) (hS : SortSpec srt) {s t : Nat} (hsnp : N.np ≤ s)
    (h : (Rm nprev pA N t s).isSome = true) : (Rm nprev pA N s t).isSome = true := by
  obtain ⟨x, hx, e, he, hk, hne, hs⟩ := (Rm_isSome hp hS t s).1 h
  obtain ⟨_, htn, _, hcx⟩ := (hp.memIff t x).1 hx
  rw [Rm_isSome hp hS]
  have hy : e.1 < nprev := hp.pColLt x e he
  have hsym := hp.pSymm x e.1 ((lookup_isSome_iff _ _).2 ⟨e.2, by simpa using he⟩)
  obtain ⟨v, hv⟩ := (lookup_isSome_iff _ _).1 hsym
  have hsn : s < N.n := by rw [← hs]; exact hp.cmapLt e.1 hk
  refine ⟨e.1, (hp.memIff s e.1).2 ⟨hsnp, hsn, hy, hs⟩, (x, v), hv, ?_, ?_, hcx⟩
  · show N.cmap x ≠ NONE
    rw [hcx]; have := hp.nLtNone; omega
  · show N.cmap x ≠ s
    rw [hcx]; intro h'; exact hne (hs.trans h'.symm)

theorem lookup_filter_key {row : List (Nat × Int)} (hn : (row.map (·.1)).Nodup) (q : Nat → Bool) (c : Nat) :
    (row.filter (fun e => q e.1)).lookup c = if q c = true then row.lookup c else none := by
  induction row with
  | nil => simp
  | cons e row ih =>
    obtain ⟨a, b⟩ := e
    simp only [List.map_cons, List.nodup_cons, List.mem_map] at hn
    have hna : row.lookup a = none := lookup_eq_none_of (fun e he hea => hn.1 ⟨e, he, hea⟩)
    rw [List.filter_cons, List.lookup_cons]
    by_cases hca : c = a
    · subst hca
      by_cases hq : q c = true
      · simp [hq]
      · simp only [hq, Bool.false_eq_true, ite_false]
        rw [ih hn.2]; simp [hq]
    · have hb : (c == a) = false := by simp [hca]
      rw [hb]
      by_cases hq : q a = true
      · simp only [hq, ite_true, List.lookup_cons, hb]; exact ih hn.2
      · simp only [hq, Bool.false_eq_true, ite_false]; exact ih hn.2

theorem lookup_keyed {L : List Nat} (hn : L.Nodup) (f : Nat → Option Int) (c : Nat) :
    (L.filterMap (fun s => (f s).map (fun v => (s, v)))).lookup c = if c ∈ L then f c else none := by
  induction L with
  | nil => simp
  | cons a L ih =>
    rw [List.nodup_cons] at hn
    rw [List.filterMap_cons]
    cases hf : f a with
    | none =>
      simp only [Option.map_none]
      rw [ih hn.2]
      by_cases hc : c = a
      · subst hc; simp [hn.1, hf]
      · simp [hc]
    | some v =>
      simp only [Option.map_some, List.lookup_cons]
      by_cases hc : c = a
      · subst hc; simp [hf]
      · have hb : (c == a) = false := by simp [hc]
        rw [hb]; simp only
        rw [ih hn.2]; simp [hc]

/-- `R(c, i)` is stored only for a part `c` (`np ≤ c < n`). -/
theorem Rm_range (hp : B2Pre ncells nprev parA pA N) (hS : SortSpec srt) {c i : Nat}
    (h : (Rm nprev pA N c i).isSome = true) : N.np ≤ c ∧ c < N.n := by
  obtain ⟨x, hx, _⟩ := (Rm_isSome hp hS c i).1 h
  have := (hp.memIff c x).1 hx; exact ⟨this.1, this.2.1⟩

theorem tcolRow_lookup (hp : B2Pre ncells nprev parA pA N) (hS : SortSpec srt) (i c : Nat) :
    (tcolRow srt pA N i).lookup c = Rm nprev pA N c i := by
  unfold tcolRow
  rw [lookup_keyed List.nodup_range' (fun s => (prow srt pA N s).lookup i) c]
  by_cases hc : c ∈ List.range' N.np (N.n - N.np)
  · simp only [hc, ite_true]; exact prow_eq hp hS c i
  · simp only [hc, ite_false]
    cases h : Rm nprev pA N c i with
    | none => rfl
    | some v =>
      obtain ⟨h1, h2⟩ := Rm_range hp hS (c := c) (i := i) (by simp [h])
      exact absurd (List.mem_range'.2 ⟨c - N.np, by omega, by omega⟩) hc

theorem zB2_isSome (hp : B2Pre ncells nprev parA pA N) (q q' : Nat) :
    (zB2 ncells (den parA) N q q').isSome = true ↔
      ((den parA q q').isSome = true ∧ q < N.np ∧ q' < N.np ∧ N.cell q ≠ SEP ∧ N.cell q' = N.cell q) := by
  have hz : (GBWDelta.mxmAny ncells (mB2 N) (fun c q' => mB2 N q' c) q q' = true) ↔
      (q < N.np ∧ q' < N.np ∧ N.cell q ≠ SEP ∧ N.cell q' = N.cell q) := by
    simp only [GBWDelta.mxmAny, decide_eq_true_eq, GBWDelta.cnt_pos_iff, mB2, decide_eq_true_eq]
    constructor
    · rintro ⟨c, _, ⟨hq, hs, rfl⟩, ⟨hq', _, hl⟩⟩; exact ⟨hq, hq', hs, hl⟩
    · rintro ⟨hq, hq', hs, hl⟩
      exact ⟨N.cell q, hp.cellLt q hq hs, ⟨hq, hs, rfl⟩, ⟨hq', by rw [hl]; exact hs, hl⟩⟩
  unfold zB2
  rw [← hz]
  by_cases h1 : (den parA q q').isSome = true <;>
  by_cases h2 : GBWDelta.mxmAny ncells (mB2 N) (fun c q' => mB2 N q' c) q q' = true <;> simp [h1, h2]

/-- The clean part of row `i < np` is `x`'s row. -/
theorem xrow_eq (hp : B2Pre ncells nprev parA pA N) {i : Nat} (hi : i < N.np) (c : Nat) :
    (if N.cell i ≠ SEP then (parA i).filter (fun e => decide (N.cell e.1 = N.cell i)) else []).lookup c =
      xB2 ncells (den parA) N i c := by
  have hcases : ∀ v, den parA i c = some v → c < N.np := by
    intro v hv
    exact hp.parColLt i (c, v) (by
      obtain ⟨w, hw⟩ := (lookup_isSome_iff _ _).1 (by simp [den] at hv; simp [hv] : ((parA i).lookup c).isSome = true)
      have := lookup_of_mem (hp.parStrict i).keys_nodup hw
      simp [den] at hv; rw [hv] at this; cases this; exact hw)
  unfold xB2 ewMult
  by_cases hs : N.cell i = SEP
  · simp only [hs, ne_eq, not_true_eq_false, ite_false, List.lookup_nil]
    cases hz : zB2 ncells (den parA) N i c with
    | none => cases den parA i c <;> rfl
    | some _ =>
      have := ((zB2_isSome hp i c).1 (by simp [hz])).2.2.2.1
      exact absurd hs this
  · simp only [hs, ne_eq, not_false_eq_true, ite_true]
    rw [lookup_filter_key (hp.parStrict i).keys_nodup (fun j => decide (N.cell j = N.cell i)) c]
    cases hl : (parA i).lookup c with
    | none =>
      have : den parA i c = none := hl
      rw [this]; simp
    | some v =>
      have hd : den parA i c = some v := hl
      have hc := hcases v hd
      by_cases hcc : N.cell c = N.cell i
      · have := (zB2_isSome hp i c).2 ⟨by simp [hd], hi, hc, hs, hcc⟩
        obtain ⟨u, hu⟩ := Option.isSome_iff_exists.1 this
        simp [hcc, hd, hu]
      · have : zB2 ncells (den parA) N i c = none := by
          cases hz : zB2 ncells (den parA) N i c with
          | none => rfl
          | some _ => exact absurd ((zB2_isSome hp i c).1 (by simp [hz])).2.2.2.2 hcc
        simp [hcc, hd, this]

/-- **`hb2` = the switch-off B2 level**, at every coordinate: B2's level is the parent's level kept
between clean ids of one cell, plus the straddler rows `R` and, in the clean rows, their transposes. -/
theorem hb2_eq (hp : B2Pre ncells nprev parA pA N) (hS : SortSpec srt) (i c : Nat) :
    den (b2Rows srt parA pA N) i c = b2GB ncells nprev (den parA) (den pA) N i c := by
  show (b2Rows srt parA pA N i).lookup c = _
  unfold b2Rows b2GB ewAdd tr
  by_cases hi : i < N.np
  · simp only [hi, ite_true]
    rw [List.lookup_append, tcolRow_lookup hp hS, xrow_eq hp hi c]
    simp only [Rm]
    have hRi : RGB nprev (den pA) N i c = none := Rm_low hp hS hi c
    rw [hRi]
    cases hx : xB2 ncells (den parA) N i c with
    | none => cases RGB nprev (den pA) N c i <;> rfl
    | some a =>
      have hc : c < N.np := by
        unfold xB2 ewMult at hx
        cases hpa : den parA i c with
        | none => rw [hpa] at hx; simp at hx
        | some v =>
          have : ((parA i).lookup c) = some v := hpa
          obtain ⟨w, hw⟩ := (lookup_isSome_iff _ _).1 (by simp [this] : ((parA i).lookup c).isSome = true)
          exact hp.parColLt i (c, w) hw
      have hR : RGB nprev (den pA) N c i = none := Rm_low hp hS hc i
      rw [hR]; rfl
  · simp only [hi, ite_false]
    rw [prow_eq hp hS]
    simp only [Rm]
    have hpi : parA i = [] := hp.parBeyond i (by omega)
    have hx0 : xB2 ncells (den parA) N i c = none := by simp [xB2, ewMult, den, hpi]
    rw [hx0]
    cases hR : RGB nprev (den pA) N i c with
    | some v => cases RGB nprev (den pA) N c i <;> rfl
    | none =>
      have : RGB nprev (den pA) N c i = none := by
        cases h : RGB nprev (den pA) N c i with
        | none => rfl
        | some w =>
          have := Rm_symm hp hS (s := i) (t := c) (by omega) (by show (RGB nprev (den pA) N c i).isSome = true; rw [h]; rfl)
          have h2 : (RGB nprev (den pA) N i c).isSome = true := this
          rw [hR] at h2; exact absurd h2 (by simp)
      rw [this]

end

end CchCoarsen
