import FalkorNativeIndex.CchRoute
import FalkorNativeIndex.CchTopo
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-!
# Small functions of `cch/`: `gallop_to`, `arc`, the query wrapper
-/
namespace NI

/-! ## `gallop_to` (`query.rs:156-170`) -/

/-- The first position at or after `from` holding a rank `≥ rank`, as a count: `from` plus the leading smaller ranks. -/
def firstAtLeast (ranks : List Nat) (from_ rank : Nat) : Nat :=
  from_ + ((ranks.drop from_).takeWhile (fun r => decide (r < rank))).length

/-- The doubling probe: `while high < len && ranks[high] < rank { low = high + 1; high += step; step *= 2 }`. -/
def gallopLoop (ranks : List Nat) (rank : Nat) : Nat → Nat → Nat → Nat → Nat × Nat
  | 0, low, high, _ => (low, high)
  | fuel + 1, low, high, step =>
    if high < ranks.length ∧ ranks.getD high 0 < rank then gallopLoop ranks rank fuel (high + 1) (high + step) (step * 2)
    else (low, high)

/-- `gallop_to`: probe 1, 2, 4, ... positions ahead, then `partition_point` on the last stretch (std's contract on a
    partitioned slice: the number of leading elements the predicate holds for). -/
def gallopTo (ranks : List Nat) (from_ rank : Nat) : Nat :=
  let r := gallopLoop ranks rank (ranks.length + 1) from_ from_ 1
  let high := min r.2 ranks.length
  r.1 + (((ranks.drop r.1).take (high - r.1)).takeWhile (fun x => decide (x < rank))).length

theorem tw_drop_split (ranks : List Nat) (rank : Nat) (hs : ranks.Pairwise (· ≤ ·)) :
    ∀ (k a : Nat), a + k ≤ ranks.length → (∀ i, a ≤ i → i < a + k → ranks.getD i 0 < rank) →
      ((ranks.drop a).takeWhile (fun r => decide (r < rank))).length =
        k + ((ranks.drop (a + k)).takeWhile (fun r => decide (r < rank))).length := by
  intro k
  induction k with
  | zero => intro a _ _; simp
  | succ k ih =>
    intro a hk hall
    have hlt : a < ranks.length := by omega
    have hx : ranks.getD a 0 < rank := hall a (Nat.le_refl _) (by omega)
    have hd : ranks.drop a = ranks[a] :: ranks.drop (a + 1) := List.drop_eq_getElem_cons hlt
    have hx' : ranks[a] < rank := by simpa [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hlt] using hx
    rw [hd, List.takeWhile_cons, ifP (by simpa using hx')]
    simp only [List.length_cons]
    rw [ih (a + 1) (by omega) (fun i h1 h2 => hall i (by omega) (by omega))]
    rw [show a + 1 + k = a + (k + 1) by omega]
    omega

theorem tw_len_le_len (l : List Nat) (p : Nat → Bool) : (l.takeWhile p).length ≤ l.length := by
  induction l with
  | nil => simp
  | cons x l ih => simp only [List.takeWhile_cons]; split <;> simp <;> omega

theorem tw_fail (p : Nat → Bool) : ∀ (l : List Nat) (i : Nat) (hi : i < l.length), p l[i] = false →
    (l.takeWhile p).length ≤ i := by
  intro l
  induction l with
  | nil => intro i hi; simp at hi
  | cons x l ih =>
    intro i hi hp
    simp only [List.takeWhile_cons]
    by_cases hx : p x = true
    · rw [ifP hx]
      cases i with
      | zero => simp at hp; rw [hp] at hx; cases hx
      | succ i => simp only [List.length_cons]; have := ih i (by simpa using hi) (by simpa using hp); omega
    · rw [ifN hx]; simp

theorem tw_take (p : Nat → Bool) : ∀ (l : List Nat) (k : Nat),
    ((l.take k).takeWhile p).length = min (l.takeWhile p).length k := by
  intro l
  induction l with
  | nil => intro k; simp
  | cons x l ih =>
    intro k
    cases k with
    | zero => simp
    | succ k =>
      simp only [List.take_succ_cons, List.takeWhile_cons]
      split
      · simp [ih]
      · simp

/-- The loop's invariant, and its exit. -/
theorem gallopLoop_spec (ranks : List Nat) (rank from_ : Nat) (hs : ranks.Pairwise (· ≤ ·)) :
    ∀ (fuel low high step : Nat), 1 ≤ step → from_ ≤ low → low ≤ high → low ≤ ranks.length →
      (∀ i, from_ ≤ i → i < low → ranks.getD i 0 < rank) → ranks.length - high < fuel →
      let r := gallopLoop ranks rank fuel low high step
      from_ ≤ r.1 ∧ r.1 ≤ r.2 ∧ r.1 ≤ ranks.length ∧ (∀ i, from_ ≤ i → i < r.1 → ranks.getD i 0 < rank) ∧
        (ranks.length ≤ r.2 ∨ (r.2 < ranks.length ∧ rank ≤ ranks.getD r.2 0)) := by
  intro fuel
  induction fuel with
  | zero => intro low high step _ _ _ _ _ hf; omega
  | succ fuel ih =>
    intro low high step hstep h1 h2 h3 h4 hf
    simp only [gallopLoop]
    by_cases hc : high < ranks.length ∧ ranks.getD high 0 < rank
    · rw [ifP hc]
      obtain ⟨hhl, hhr⟩ := hc
      have g1 : 1 ≤ step * 2 := by omega
      have g2 : from_ ≤ high + 1 := by omega
      have g3 : high + 1 ≤ high + step := by omega
      have g4 : high + 1 ≤ ranks.length := by omega
      have g5 : ranks.length - (high + step) < fuel := by omega
      refine ih (high + 1) (high + step) (step * 2) g1 g2 g3 g4 ?_ g5
      intro i hi1 hi2
      by_cases hil : i < low
      · exact h4 i hi1 hil
      · -- by sortedness, everything up to `high` is below `rank`
        have hi : i ≤ high := by omega
        have hih : i < ranks.length := by omega
        rcases Nat.lt_or_ge i high with hlt | hge
        · have := List.pairwise_iff_getElem.1 hs i high hih hhl hlt
          have e1 : ranks.getD i 0 = ranks[i] := by simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hih]
          have e2 : ranks.getD high 0 = ranks[high] := by simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hhl]
          omega
        · have : i = high := by omega
          subst this; exact hhr
    · rw [ifN hc]
      refine ⟨h1, h2, h3, h4, ?_⟩
      by_cases hh : high < ranks.length
      · right; refine ⟨hh, ?_⟩; show rank ≤ ranks.getD high 0
        by_cases hr : ranks.getD high 0 < rank
        · exact absurd ⟨hh, hr⟩ hc
        · omega
      · left; show ranks.length ≤ high; omega

/-- **`gallop_to` finds the first position at or after `from` holding a rank `≥ rank`** (or `len`), on an ascending
    list (`partition_point` by std's contract on a partitioned slice). -/
theorem gallopTo_spec (ranks : List Nat) (from_ rank : Nat) (hs : ranks.Pairwise (· ≤ ·)) (hf : from_ ≤ ranks.length) :
    gallopTo ranks from_ rank = firstAtLeast ranks from_ rank := by
  have hl := gallopLoop_spec ranks rank from_ hs (ranks.length + 1) from_ from_ 1 (Nat.le_refl _) (Nat.le_refl _)
    (Nat.le_refl _) hf (fun i h1 h2 => by omega) (by omega)
  unfold gallopTo firstAtLeast
  generalize hr : gallopLoop ranks rank (ranks.length + 1) from_ from_ 1 = r at hl ⊢
  obtain ⟨hl1, hl2, hl3, hl4, hl5⟩ := hl
  simp only
  -- the leading smaller ranks from `from` split at `r.1`
  have hsplit := tw_drop_split ranks rank hs (r.1 - from_) from_ (by omega) (fun i h1 h2 => hl4 i h1 (by omega))
  rw [show from_ + (r.1 - from_) = r.1 by omega] at hsplit
  have hTle : ((ranks.drop r.1).takeWhile (fun x => decide (x < rank))).length ≤ min r.2 ranks.length - r.1 := by
    rcases hl5 with h | ⟨h1, h2⟩
    · have := tw_len_le_len (ranks.drop r.1) (fun x => decide (x < rank))
      simp at this; omega
    · have hdl : r.2 - r.1 < (ranks.drop r.1).length := by simp; omega
      have := tw_fail (fun x => decide (x < rank)) (ranks.drop r.1) (r.2 - r.1) hdl (by
        have e : (ranks.drop r.1)[r.2 - r.1] = ranks.getD r.2 0 := by
          simp [List.getD_eq_getElem?_getD, List.getElem?_drop, List.getElem?_eq_getElem h1]
          congr 1; omega
        rw [e]; exact decide_eq_false (by omega))
      omega
  rw [tw_take, Nat.min_eq_left hTle]
  omega

/-! ## `arc` (`query.rs:179-190`) and `Cch::query` (`mod.rs:309-315`) -/

/-- `arc(cch, from, to)`: the weight and middle of the arc directed `from -> to`, `None` if the ranks are not
    adjacent in the chordal supergraph. (Pair-indexed; the slot arrays are `CchMetric`/`mkArr` below.) -/
def arcOf (up : Nat → List Nat) (A : Nat → Nat → Wt) (M : Nat → Nat → Option Nat) (a b : Nat) :
    Option (Wt × Option Nat) :=
  if a ≠ b ∧ ((a < b ∧ b ∈ up a) ∨ (b < a ∧ a ∈ up b)) then some (A a b, M a b) else none

theorem arcOf_spec (up A M a b) :
    arcOf up A M a b = some (A a b, M a b) ↔ ArcP up a b := by
  unfold arcOf ArcP
  by_cases h : a ≠ b ∧ ((a < b ∧ b ∈ up a) ∨ (b < a ∧ a ∈ up b))
  · simp [h]
  · simp [h]

/-- `Cch::query`'s guards: an id outside the hierarchy gives `None`, `src == dst` the single-node path of weight
    `0`, otherwise the search (`run`) answers. -/
def queryWrap (n src dst : Nat) (run : Nat → Nat → Option (Wt × List Nat)) : Option (Wt × List Nat) :=
  if ¬ src < n ∨ ¬ dst < n then none
  else if src = dst then some (some 0, [src]) else run src dst

theorem queryWrap_spec (n src dst run) :
    (¬ src < n ∨ ¬ dst < n → queryWrap n src dst run = none) ∧
    (src < n → src = dst → queryWrap n src dst run = some (some 0, [src])) ∧
    (src < n → dst < n → src ≠ dst → queryWrap n src dst run = run src dst) := by
  refine ⟨fun h => by unfold queryWrap; rw [ifP h], fun h1 h2 => ?_, fun h1 h2 h3 => by simp [queryWrap, h1, h2, h3]⟩
  subst h2; simp [queryWrap, h1]

/-! ## Weight validation and the entry iterator (`customize.rs:136-170`) -/

/-- The weight check `weight.is_nan() || *weight < 0.0` on an `f64` given as its bits (`Float.lean`). -/
def badWeight (w : Nat) : Bool := isNaN w || fpcmp w 0 == some .lt

/-- **What the weight check accepts**: exactly the non-NaN encodings that are not negative: `+0`, `-0`, every positive
    finite value and `+inf`. NaN, `-inf` and every negative value are refused. -/
theorem badWeight_false_iff (w : Nat) (hw : w < W) :
    badWeight w = false ↔ isNaN w = false ∧ (w < SIGN ∨ w = SIGN) := by
  unfold badWeight
  by_cases hn : isNaN w = true
  · simp [hn]
  · have hn' : isNaN w = false := by simpa using hn
    simp only [hn', Bool.false_or]
    have hz : isNaN 0 = false := by decide
    have hpos0 : pos 0 = SIGN := by decide
    unfold fpcmp
    simp only [hn', hz, Bool.or_self, Bool.false_eq_true, ite_false, beq_iff_eq, Option.some.injEq, hpos0]
    have hWS : W = 2 * SIGN := by unfold W SIGN; omega
    rcases pos_cases w with ⟨h1, h2⟩ | ⟨h1, h2⟩
    · rw [h2]; unfold cmpN
      have hn1 : ¬ (SIGN + w < SIGN) := by omega
      have hn2 : ¬ (SIGN + w = SIGN → False) ∨ True := Or.inr trivial
      rw [ifN hn1]
      by_cases he : SIGN + w = SIGN
      · rw [ifP he]; simp; omega
      · rw [ifN he]; simp; omega
    · rw [h2]; unfold cmpN
      by_cases h : w = SIGN
      · subst h; have : W - SIGN = SIGN := by omega
        rw [this]; simp
      · have hlt : W - w < SIGN := by omega
        rw [ifP hlt]; simp; omega

inductive VErr where
  | needsRebuild | internal | invalidWeight
deriving DecidableEq

/-- `validate` (`customize.rs:136-160`), in its three checks' order. `n` is the topology's node count; `csr` the
    weight CSR with its `values`. -/
def validate (n : Nat) (csr : Cch.Csr) (values : List Nat) : Option VErr :=
  if csr.n ≠ n then some .needsRebuild
  else if csr.isWellFormed = false ∨ values.length ≠ csr.i.length ∨
      ¬ (List.range n).all (fun u => Cch.strictWin (csr.rowD u)) then some .internal
  else if values.any badWeight then some .invalidWeight
  else none

/-- **`validate` accepts exactly a well-shaped, valid input, and reports the first failing check**:
    a different node count is `NeedsRebuild` whatever else is wrong; then a malformed CSR is `Internal`; then a
    negative or NaN weight is `InvalidWeight`. -/
theorem validate_spec (n : Nat) (csr : Cch.Csr) (values : List Nat) :
    (validate n csr values = none ↔ csr.n = n ∧ csr.isWellFormed = true ∧ values.length = csr.i.length ∧
      (∀ u, u < n → (csr.rowD u).Pairwise (· < ·)) ∧ ∀ w ∈ values, badWeight w = false) ∧
    (csr.n ≠ n → validate n csr values = some .needsRebuild) ∧
    (csr.n = n → (csr.isWellFormed = false ∨ values.length ≠ csr.i.length ∨
        ¬ (∀ u, u < n → (csr.rowD u).Pairwise (· < ·))) → validate n csr values = some .internal) ∧
    (csr.n = n → csr.isWellFormed = true → values.length = csr.i.length →
        (∀ u, u < n → (csr.rowD u).Pairwise (· < ·)) → (∃ w ∈ values, badWeight w = true) →
        validate n csr values = some .invalidWeight) := by
  have hrows : (List.range n).all (fun u => Cch.strictWin (csr.rowD u)) = true ↔
      ∀ u, u < n → (csr.rowD u).Pairwise (· < ·) := by
    simp only [List.all_eq_true, List.mem_range, Cch.strictWin_iff]
  refine ⟨?_, ?_, ?_, ?_⟩
  · unfold validate
    by_cases h1 : csr.n = n
    · simp only [h1, ne_eq, not_true_eq_false, ite_false]
      by_cases h2 : csr.isWellFormed = false ∨ values.length ≠ csr.i.length ∨
          ¬ (List.range n).all (fun u => Cch.strictWin (csr.rowD u)) = true
      · rw [ifP (by simpa using h2)]
        simp only [reduceCtorEq, false_iff, true_and]
        intro ⟨h3, h4, h5, h6⟩
        rcases h2 with h2 | h2 | h2
        · simp [h2] at h3
        · exact h2 h4
        · exact h2 (hrows.2 h5)
      · rw [ifN (by simpa using h2)]
        simp only [not_or, Classical.not_not, ne_eq] at h2
        obtain ⟨h3, h4, h5⟩ := h2
        by_cases h6 : values.any badWeight = true
        · rw [ifP h6]
          simp only [reduceCtorEq, false_iff, h1, true_and]
          intro ⟨_, _, _, h7⟩
          obtain ⟨w, hw, hb⟩ := List.any_eq_true.1 h6
          rw [h7 w hw] at hb; cases hb
        · rw [ifN h6]
          simp only [true_iff, h1, true_and]
          refine ⟨by simpa using h3, h4, hrows.1 h5, fun w hw => ?_⟩
          cases hb : badWeight w
          · rfl
          · exact absurd (List.any_eq_true.2 ⟨w, hw, hb⟩) h6
    · simp [validate, h1]
  · intro h; simp [validate, h]
  · intro h1 h2
    unfold validate
    rw [ifN (by simpa using h1)]
    rw [ifP]
    rcases h2 with h2 | h2 | h2
    · exact Or.inl h2
    · exact Or.inr (Or.inl h2)
    · exact Or.inr (Or.inr (fun h => h2 (hrows.1 h)))
  · intro h1 h2 h3 h4 ⟨w, hw, hb⟩
    unfold validate
    rw [ifN (by simpa using h1), ifN, ifP]
    · exact List.any_eq_true.2 ⟨w, hw, hb⟩
    · simp [h2, h3, hrows.2 h4]

/-- `WeightCsr::entries`: every stored entry as `(row, column, weight)`, row by row. -/
def entries (csr : Cch.Csr) (values : List Nat) : List (Nat × Nat × Nat) :=
  (List.range csr.n).flatMap (fun u =>
    ((csr.rowD u).zip ((values.drop (csr.p.getD u 0)).take (csr.p.getD (u + 1) 0 - csr.p.getD u 0))).map
      (fun vw => (u, vw.1, vw.2)))

theorem mem_zip_idx : ∀ (l1 l2 : List Nat) (v w : Nat), l1.length = l2.length →
    ((v, w) ∈ l1.zip l2 ↔ ∃ j : Nat, l1[j]? = some v ∧ l2[j]? = some w)
  | [], [], v, w, _ => by simp
  | [], _ :: _, _, _, h => by simp at h
  | _ :: _, [], _, _, h => by simp at h
  | a :: l1, b :: l2, v, w, h => by
    have ih := mem_zip_idx l1 l2 v w (by simpa using h)
    simp only [List.zip_cons_cons, List.mem_cons]
    constructor
    · rintro (h | h)
      · simp only [Prod.mk.injEq] at h
        exact ⟨0, by simp [h.1], by simp [h.2]⟩
      · obtain ⟨j, hj1, hj2⟩ := ih.1 h
        exact ⟨j + 1, by simpa using hj1, by simpa using hj2⟩
    · rintro ⟨j, hj1, hj2⟩
      cases j with
      | zero => left; simp at hj1 hj2; simp [hj1, hj2]
      | succ j => right; exact ih.2 ⟨j, by simpa using hj1, by simpa using hj2⟩

theorem slice_getElem? {α : Type} (l : List α) (a b j : Nat) :
    ((l.drop a).take (b - a))[j]? = if j < b - a then l[a + j]? else none := by
  rw [List.getElem?_take, List.getElem?_drop]

/-- **`entries` returns exactly the stored entries**: for a well-formed CSR with one weight per entry,
    `(u, v, w)` is returned iff `u < n` and some position `k` of row `u` holds column `v` and weight `w`. -/
theorem entries_mem (csr : Cch.Csr) (values : List Nat) (hwf : csr.isWellFormed = true)
    (hl : values.length = csr.i.length) (u v w : Nat) :
    (u, v, w) ∈ entries csr values ↔ u < csr.n ∧ ∃ k, csr.p.getD u 0 ≤ k ∧ k < csr.p.getD (u + 1) 0 ∧
      csr.i[k]? = some v ∧ values[k]? = some w := by
  obtain ⟨hpl, h0, hn, hm, hid⟩ := (Cch.isWellFormed_iff csr).1 hwf
  have key : ∀ u', u' < csr.n → ∃ a b, csr.p.getD u' 0 = a ∧ csr.p.getD (u' + 1) 0 = b ∧ a ≤ b ∧
      b ≤ csr.i.length ∧ csr.rowD u' = (csr.i.drop a).take (b - a) := by
    intro u' hu'
    obtain ⟨a, ha⟩ : ∃ a, csr.p[u']? = some a := ⟨csr.p[u']'(by omega), List.getElem?_eq_getElem (by omega)⟩
    obtain ⟨b, hb⟩ : ∃ b, csr.p[u' + 1]? = some b := ⟨csr.p[u' + 1]'(by omega), List.getElem?_eq_getElem (by omega)⟩
    have hab : a ≤ b := hm u' a b ha hb
    have hbn : b ≤ csr.i.length := Cch.mono_le csr.p hm csr.n (u' + 1) b csr.i.length (by omega) hb hn
    refine ⟨a, b, by simp [List.getD_eq_getElem?_getD, ha], by simp [List.getD_eq_getElem?_getD, hb], hab, hbn, ?_⟩
    unfold Cch.Csr.rowD
    rw [Cch.row?_spec csr u' a b ha hb, ifP ⟨hab, hbn⟩]; rfl
  unfold entries
  simp only [List.mem_flatMap, List.mem_range, List.mem_map, Prod.mk.injEq]
  constructor
  · rintro ⟨u', hu', ⟨⟨v', w'⟩, hmem, rfl, rfl, rfl⟩⟩
    refine ⟨hu', ?_⟩
    obtain ⟨a, b, hga, hgb, hab, hbn, hrow⟩ := key u' hu'
    rw [hga, hgb, hrow] at hmem
    have hlen : ((csr.i.drop a).take (b - a)).length = ((values.drop a).take (b - a)).length := by
      simp [List.length_take, List.length_drop, hl]
    obtain ⟨j, hj1, hj2⟩ := (mem_zip_idx _ _ _ _ hlen).1 hmem
    rw [slice_getElem?] at hj1 hj2
    by_cases hj : j < b - a
    · rw [ifP hj] at hj1 hj2
      exact ⟨a + j, by rw [hga]; omega, by rw [hgb]; omega, hj1, hj2⟩
    · rw [ifN hj] at hj1; cases hj1
  · rintro ⟨hu, k, hk1, hk2, hv, hw⟩
    refine ⟨u, hu, ⟨v, w⟩, ?_, rfl, rfl, rfl⟩
    obtain ⟨a, b, hga, hgb, hab, hbn, hrow⟩ := key u hu
    rw [hga] at hk1; rw [hgb] at hk2
    rw [hga, hgb, hrow]
    have hlen : ((csr.i.drop a).take (b - a)).length = ((values.drop a).take (b - a)).length := by
      simp [List.length_take, List.length_drop, hl]
    refine (mem_zip_idx _ _ _ _ hlen).2 ⟨k - a, ?_, ?_⟩
    · rw [slice_getElem?, ifP (by omega), show a + (k - a) = k by omega]; exact hv
    · rw [slice_getElem?, ifP (by omega), show a + (k - a) = k by omega]; exact hw

/-! ## `order_connected_nodes` (`nd.rs:91-124`)

The skeleton's connected nodes (non-empty rows), renumbered `0..m` in id order, with their rows copied into the
backend's offsets and neighbours (`row_offsets.push(neighbors.len())` after each row). The backend itself is not
modelled (see `ndOrder`). -/

/-- The connected nodes, in id order. -/
def connectedNodes (skel : Cch.Csr) : List Nat := (List.range skel.n).filter (fun node => !(skel.rowD node).isEmpty)

/-- The backend's CSR: offsets and neighbours of the renumbered rows. -/
def backendCsr (skel : Cch.Csr) : List Nat × List Nat :=
  flatRows ((connectedNodes skel).map (fun node => (skel.rowD node).map (fun nb => (connectedNodes skel).idxOf nb)))

/-- **The renumbered CSR is the skeleton on its connected nodes, relabelled**: `m + 1` offsets, and row `i` is the
    row of the `i`-th connected node with every neighbour replaced by its new id, which is below `m` and names that
    very neighbour (every neighbour of a connected node is connected, because the skeleton is symmetric). -/
theorem backendCsr_spec (skel : Cch.Csr) (hw : skel.isWellFormed = true)
    (hsym : ∀ u, u < skel.n → ∀ v ∈ skel.rowD u, v ≠ u ∧ u ∈ skel.rowD v) :
    let c := connectedNodes skel
    (backendCsr skel).1.length = c.length + 1 ∧
    (∀ i (hi : i < c.length), rowOfFlat (backendCsr skel).1 (backendCsr skel).2 i =
        (skel.rowD c[i]).map (fun nb => c.idxOf nb)) ∧
    (∀ i (hi : i < c.length), ∀ nb ∈ skel.rowD c[i], c.idxOf nb < c.length ∧ c[c.idxOf nb]? = some nb) := by
  intro c
  obtain ⟨_, _, _, _, hid⟩ := (Cch.isWellFormed_iff skel).1 hw
  have hcmem : ∀ x, x ∈ c ↔ x < skel.n ∧ skel.rowD x ≠ [] := by
    intro x
    simp [c, connectedNodes, List.mem_filter, List.mem_range]
  refine ⟨?_, ?_, ?_⟩
  · unfold backendCsr
    rw [flatRows_eq]; simp [stateOf]; rfl
  · intro i hi
    unfold backendCsr
    have hl : ((c.map (fun node => (skel.rowD node).map (fun nb => c.idxOf nb)))).length = c.length := by simp
    rw [rowOfFlat_flatRows _ i (by rw [hl]; exact hi)]
    simp; rfl
  · intro i hi nb hnb
    have hcn : c[i] ∈ c := List.getElem_mem hi
    obtain ⟨hcn1, _⟩ := (hcmem _).1 hcn
    have hnbn : nb < skel.n := hid nb (Cch.rowD_subset skel c[i] nb hnb)
    have hne : skel.rowD nb ≠ [] := by
      intro he
      have := (hsym c[i] hcn1 nb hnb).2
      rw [he] at this; simp at this
    have hnbc : nb ∈ c := (hcmem nb).2 ⟨hnbn, hne⟩
    have hlt : c.idxOf nb < c.length := List.idxOf_lt_length_iff.2 hnbc
    refine ⟨hlt, ?_⟩
    rw [List.getElem?_eq_getElem hlt]
    simp [List.getElem_idxOf hlt]

/-! ## The slot arrays (`Metric`, `customize.rs:60-80`, `:382-410`)

An arc slot is `up_range(lower).start + position of higher in lower's row`. The four per-arc arrays are indexed
by it. `slotArr rows n f` is the array whose slot of the arc `(r, x)` holds `f r x`: rows in rank order, each row's
entries in order, matching `neighbors` (`flatRows`). -/

def slotArr {β : Type} (rows : List (List Nat)) (n : Nat) (f : Nat → Nat → β) : List β :=
  ((List.range n).map (fun r => (rows.getD r []).map (f r))).flatten

theorem offAt_map_len {α β : Type} (rows : List (List α)) (rows' : List (List β)) (hl : rows.length = rows'.length)
    (h : ∀ j (hj : j < rows.length), (rows'.getD j []).length = (rows.getD j []).length) (r : Nat) (hr : r ≤ rows.length) :
    offAt rows' r = offAt rows r := by
  unfold offAt
  rw [List.length_flatten, List.length_flatten]
  congr 1
  apply List.ext_getElem
  · simp; omega
  · intro j hj1 hj2
    simp only [List.getElem_map, List.getElem_take]
    have hj : j < rows.length := by simp at hj1; omega
    have := h j hj
    rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj,
      List.getElem?_eq_getElem (by omega)] at this
    simpa using this

/-- **A slot holds its arc's value**: in `slotArr rows n f`, slot `offAt rows r + i` (the `i`-th arc of row `r`)
    holds `f r` of that arc's higher end. -/
theorem slotArr_get {β : Type} (rows : List (List Nat)) (n : Nat) (f : Nat → Nat → β) (hn : rows.length = n)
    (r i : Nat) (hr : r < n) (hi : i < (rows.getD r []).length) :
    (slotArr rows n f)[offAt rows r + i]? = some (f r ((rows.getD r [])[i]'hi)) := by
  obtain ⟨rows', hrows'⟩ : ∃ rows', rows' = (List.range n).map (fun r => (rows.getD r []).map (f r)) := ⟨_, rfl⟩
  have hl' : rows'.length = n := by simp [hrows']
  have hrow : ∀ j (hj : j < n), rows'.getD j [] = (rows.getD j []).map (f j) := by
    intro j hj
    simp [hrows', List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range, hj]
  have hoff : offAt rows' r = offAt rows r := by
    refine offAt_map_len rows rows' (by omega) (fun j hj => ?_) r (by omega)
    rw [hrow j (by omega)]; simp
  have hslice := slice_row rows' r (by omega)
  have hsl2 : offAt rows' (r + 1) - offAt rows' r = ((rows.getD r []).map (f r)).length := by
    rw [offAt_succ rows' r (by omega)]
    have : rows'[r] = (rows.getD r []).map (f r) := by
      have := hrow r hr
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (by omega)] at this
      simpa using this
    rw [this]; omega
  unfold slotArr
  rw [← hrows']
  have hsl := slice_getElem? (rows'.flatten) (offAt rows' r) (offAt rows' (r + 1)) i
  rw [hslice] at hsl
  have : rows'[r] = (rows.getD r []).map (f r) := by
    have := hrow r hr
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (by omega)] at this
    simpa using this
  rw [this] at hsl
  have hlt : i < offAt rows' (r + 1) - offAt rows' r := by rw [hsl2]; simpa using hi
  rw [ifP hlt] at hsl
  rw [← hoff, ← hsl]
  simp only [List.getElem?_map]
  rw [List.getElem?_eq_getElem hi]
  rfl

/-- The four per-arc arrays of a customization, from its arc weights `A` and middles `M`
    (`upward_weight[k] = A lower higher`, `downward_weight[k] = A higher lower`, likewise the middles). -/
def metricArrays (rows : List (List Nat)) (n : Nat) (A : Nat → Nat → Wt) (M : Nat → Nat → Option Nat) :
    List Wt × List Wt × List (Option Nat) × List (Option Nat) :=
  (slotArr rows n (fun r x => A r x), slotArr rows n (fun r x => A x r),
   slotArr rows n (fun r x => M r x), slotArr rows n (fun r x => M x r))

theorem upRange_flat (rows : List (List Nat)) (r : Nat) (hr : r < rows.length) :
    rowSlots (flatRows rows).1 r = (offAt rows r, offAt rows (r + 1)) := by
  rw [flatRows_eq]
  unfold rowSlots stateOf
  simp only
  have e1 : ((List.range (rows.length + 1)).map (offAt rows)).getD r 0 = offAt rows r := by
    simp [List.getD_eq_getElem?_getD, List.getElem?_range, show r < rows.length + 1 by omega]
  have e2 : ((List.range (rows.length + 1)).map (offAt rows)).getD (r + 1) 0 = offAt rows (r + 1) := by
    simp [List.getD_eq_getElem?_getD, List.getElem?_range, show r + 1 < rows.length + 1 by omega]
  rw [e1, e2]

/-- **The slot arrays hold the pair-indexed arc weights**: for an arc `lower < higher` of the chordal rows, the slot
    `arc_slot` finds — the same from either order — holds in the upward arrays the arc `lower -> higher`
    and in the downward arrays `higher -> lower`; so `arc(from, to)` (upward arrays if `from < to`, else downward)
    returns exactly `A from to` and its middle. -/
theorem arc_via_slots (bs : List Nat → Nat → Option Nat) (hbs : BsIdxSpec bs) (rows : List (List Nat)) (n : Nat)
    (hn : rows.length = n) (A : Nat → Nat → Wt) (M : Nat → Nat → Option Nat) (t : Topo)
    (hoff : t.upOff = (flatRows rows).1) (hnb : t.upNb = (flatRows rows).2)
    (hsorted : ∀ r, r < n → (rows.getD r []).Pairwise (· < ·)) (a b : Nat) (hab : a < b) (han : a < n)
    (hb : b ∈ rows.getD a []) :
    ∃ k, t.arcSlot bs a b = some k ∧ t.arcSlot bs b a = some k ∧
      (metricArrays rows n A M).1[k]? = some (A a b) ∧ (metricArrays rows n A M).2.1[k]? = some (A b a) ∧
      (metricArrays rows n A M).2.2.1[k]? = some (M a b) ∧ (metricArrays rows n A M).2.2.2[k]? = some (M b a) := by
  have hup : t.up a = rows.getD a [] := by
    unfold Topo.up
    rw [hoff, hnb, rowOfFlat_flatRows rows a (by omega)]
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (by omega)]; rfl
  have hrange : (t.upRange a).1 = offAt rows a := by
    unfold Topo.upRange; rw [hoff, upRange_flat rows a (by omega)]
  have hspec := arcSlot_spec bs hbs t a b (by rw [Nat.min_eq_left (Nat.le_of_lt hab), hup]; exact hsorted a han)
  have hspec' := arcSlot_spec bs hbs t b a (by rw [Nat.min_eq_right (Nat.le_of_lt hab), hup]; exact hsorted a han)
  rw [Nat.min_eq_left (Nat.le_of_lt hab), Nat.max_eq_right (Nat.le_of_lt hab), hup] at hspec
  rw [Nat.min_eq_right (Nat.le_of_lt hab), Nat.max_eq_left (Nat.le_of_lt hab), hup] at hspec'
  rw [ifP hb, hrange] at hspec
  rw [ifP hb, hrange] at hspec'
  have hidx : (rows.getD a []).idxOf b < (rows.getD a []).length := List.idxOf_lt_length_iff.2 hb
  have hel : (rows.getD a [])[(rows.getD a []).idxOf b] = b := List.getElem_idxOf hidx
  refine ⟨offAt rows a + (rows.getD a []).idxOf b, hspec, hspec', ?_, ?_, ?_, ?_⟩ <;>
  · unfold metricArrays
    simp only
    rw [slotArr_get rows n _ hn a _ han hidx, hel]
  all_goals rfl

end NI
