/-
Aggregation functions (`graph/src/runtime/functions/aggregation.rs`) and the
two ways the Aggregate operator drives them (`graph/src/runtime/ops/aggregate.rs`):

* per row  — `run_agg_expr` (aggregate.rs:981): skip a `Null` argument
  (aggregate.rs:1030), else call `batch_fn(runtime, &[v], 1, prev)`
  (aggregate.rs:1066);
* per batch — the keyless fast path (aggregate.rs:586-689): one
  `batch_fn(runtime, column, num_active, prev)` call per input batch.

Every `*_batch` kernel is a left fold over its input slice that skips `Null`.
The headline theorems say the two drivers agree, and that splitting the input
into batches at any boundary gives the same accumulator (`*_hom`).
-/

namespace FalkorAggOps

/-- The slice of `runtime::value::Value` the numeric aggregates see.
`other n` stands for every non-numeric, non-null value (validated away before
`sum`/`avg`/`stDev` are called — aggregate.rs:657-669, :1045). -/
inductive V where
  | null
  | int (i : Int)
  | flt (f : Float)
  | other (n : Nat)
  deriving Inhabited

def V.isNull : V → Bool
  | .null => true
  | _ => false

/-- `Value::get_numeric` / the `as f64` casts in the kernels. -/
def V.num : V → Float
  | .int i => Float.ofInt i
  | .flt f => f
  | _ => 0.0

/-! ## A generic null-skipping fold, and the row/batch agreement theorem -/

/-- Shape of every `*_batch` kernel: `for val in inputs { if Null continue; step }`. -/
def skipFold {S : Type} (step : S → V → S) (acc : S) (xs : List V) : S :=
  xs.foldl (fun a v => if v.isNull then a else step a v) acc

/-- The per-row driver `run_agg_expr`: null arguments return early
(aggregate.rs:1030), others call the kernel on a one-element slice. -/
def rowDriver {S : Type} (kernel : S → List V → S) (acc : S) (xs : List V) : S :=
  xs.foldl (fun a v => if v.isNull then a else kernel a [v]) acc

/-- The keyless fast path: one kernel call per batch (aggregate.rs:676). -/
def batchDriver {S : Type} (kernel : S → List V → S) (acc : S) (bs : List (List V)) : S :=
  bs.foldl kernel acc

theorem skipFold_append {S : Type} (step : S → V → S) (acc : S) (xs ys : List V) :
    skipFold step acc (xs ++ ys) = skipFold step (skipFold step acc xs) ys := by
  simp [skipFold, List.foldl_append]

/-- Batch-boundary independence: any batching of the input column gives the
same accumulator as one big batch. -/
theorem batchDriver_eq_join {S : Type} (step : S → V → S) (acc : S) (bs : List (List V)) :
    batchDriver (skipFold step) acc bs = skipFold step acc bs.flatten := by
  induction bs generalizing acc with
  | nil => rfl
  | cons b bs ih =>
    simp [batchDriver] at *
    rw [ih, skipFold_append]

/-- Per-row driver = one batch. -/
theorem skipFold_cons {S : Type} (step : S → V → S) (acc : S) (x : V) (xs : List V) :
    skipFold step acc (x :: xs) = skipFold step (if x.isNull then acc else step acc x) xs := rfl

theorem rowDriver_eq_batch {S : Type} (step : S → V → S) (acc : S) (xs : List V) :
    rowDriver (skipFold step) acc xs = skipFold step acc xs := by
  unfold rowDriver
  have : (fun a v => if v.isNull = true then a else skipFold step a [v]) =
      (fun a v => if v.isNull = true then a else step a v) := by
    funext a v; by_cases h : v.isNull <;> simp [h, skipFold]
  rw [this]; rfl

/-- Headline: vectorized (any batching) = row-at-a-time. -/
theorem vectorized_eq_rowwise {S : Type} (step : S → V → S) (acc : S) (bs : List (List V)) :
    batchDriver (skipFold step) acc bs = rowDriver (skipFold step) acc bs.flatten := by
  rw [batchDriver_eq_join, rowDriver_eq_batch]

/-! ## The kernels, line by line -/

/-- `count_batch` (aggregation.rs:343) with a column (`count(x)`). -/
def countStep (n : Int) (_ : V) : Int := n + 1
def countBatch (acc : Int) (xs : List V) : Int := skipFold countStep acc xs

/-- `count_batch` with no column (`count(*)`, aggregation.rs:354): adds `num_rows`. -/
def countStarBatch (acc : Int) (numRows : Nat) : Int := acc + numRows

theorem countBatch_spec (acc : Int) (xs : List V) :
    countBatch acc xs = acc + (xs.filter (fun v => !v.isNull)).length := by
  induction xs generalizing acc with
  | nil => simp [countBatch, skipFold]
  | cons x xs ih =>
    unfold countBatch; rw [skipFold_cons]
    by_cases h : x.isNull
    · simp only [h, ite_true]
      rw [show skipFold countStep acc xs = countBatch acc xs from rfl, ih]; simp [List.filter, h]
    · have h' : x.isNull = false := by simpa using h
      simp only [h', Bool.false_eq_true, if_false]
      rw [show skipFold countStep (countStep acc x) xs = countBatch (countStep acc x) xs from rfl, ih]
      simp [List.filter, h, countStep]; omega

theorem countStar_hom (acc : Int) (m n : Nat) :
    countStarBatch (countStarBatch acc m) n = countStarBatch acc (m + n) := by
  simp [countStarBatch]; omega

/-- `count(*)` on the per-row path is `count(1)`: a non-null constant column
(cypher.rs `count(*)` → `Constant(Int 1)`), so it counts every row. -/
theorem countStar_rowwise (acc : Int) (n : Nat) :
    countBatch acc (List.replicate n (V.int 1)) = countStarBatch acc n := by
  rw [countBatch_spec]
  simp [countStarBatch, V.isNull]

/-- `sum_batch` (aggregation.rs:387): the accumulator is always `f64`. -/
def sumStep (t : Float) (v : V) : Float := t + v.num
def sumBatch (acc : Float) (xs : List V) : Float := skipFold sumStep acc xs

/-- `collect_batch` (aggregation.rs:364). -/
def collectStep (l : List V) (v : V) : List V := l ++ [v]
def collectBatch (acc : List V) (xs : List V) : List V := skipFold collectStep acc xs

theorem collectBatch_spec (acc xs : List V) :
    collectBatch acc xs = acc ++ xs.filter (fun v => !v.isNull) := by
  induction xs generalizing acc with
  | nil => simp [collectBatch, skipFold]
  | cons x xs ih =>
    unfold collectBatch; rw [skipFold_cons]
    by_cases h : x.isNull
    · simp only [h, ite_true]
      rw [show skipFold collectStep acc xs = collectBatch acc xs from rfl, ih]; simp [List.filter, h]
    · have h' : x.isNull = false := by simpa using h
      simp only [h', Bool.false_eq_true, if_false]
      rw [show skipFold collectStep (collectStep acc x) xs
          = collectBatch (collectStep acc x) xs from rfl, ih]
      simp [List.filter, h, collectStep]

/-- An abstract `compare_value`: ordering + "compared against null" flag. -/
structure Cmp (α : Type) where
  ord : α → α → Ordering
  nullFlag : α → α → Bool

/-- `max_batch` / `min_batch` (aggregation.rs:412, :487): `best = Null` takes
the first value, then `val` replaces `best` when strictly greater (less) and
not a null comparison. `want = .gt` for max, `.lt` for min. -/
def mmStep {α : Type} (c : Cmp α) (want : Ordering) (best : Option α) (v : α) : Option α :=
  match best with
  | none => some v
  | some b => if c.ord v b == want && !c.nullFlag v b then some v else some b

def mmBatch {α : Type} (c : Cmp α) (want : Ordering) (best : Option α) (xs : List (Option α)) :
    Option α :=
  xs.foldl (fun a v => match v with | none => a | some v => mmStep c want a v) best

/-- The scalar `max` function body (aggregation.rs:119-128): keep `a` (the new
value) when `b.compare_value(a)` is `Less` or a null comparison. -/
def maxScalar {α : Type} (c : Cmp α) (a : α) (b : Option α) : Option α :=
  match b with
  | none => some a
  | some b => if c.ord b a == .lt || c.nullFlag b a then some a else some b

/-- For a comparison that is antisymmetric (`ord b a = lt ↔ ord a b = gt`) and
never flags two non-null values, the scalar `max` and `max_batch` agree. -/
theorem maxScalar_eq_batch {α : Type} (c : Cmp α)
    (anti : ∀ a b, c.ord b a = .lt ↔ c.ord a b = .gt)
    (noflag : ∀ a b, c.nullFlag a b = false) (a : α) (b : Option α) :
    maxScalar c a b = mmStep c .gt b a := by
  cases b with
  | none => rfl
  | some b =>
    simp only [maxScalar, mmStep, noflag, Bool.or_false, Bool.not_false, Bool.and_true]
    by_cases h : c.ord b a = .lt
    · have := (anti a b).mp h; simp [h, this]
    · have : c.ord a b ≠ .gt := fun h' => h ((anti a b).mpr h')
      simp [h, this]

/-- A NaN-like element breaks antisymmetry (`value.rs` compare: NaN is `Less`
both ways, known #2891/#2913): then scalar `max` and `max_batch` disagree. -/
def nanCmp : Cmp Nat where
  -- 0 plays NaN: Less against everything, both directions.
  ord a b := if a == 0 || b == 0 then .lt else compare a b
  nullFlag _ _ := false

theorem maxScalar_ne_batch_nan : maxScalar nanCmp 0 (some 5) ≠ mmStep nanCmp .gt (some 5) 0 := by
  decide

/-- `avg_batch` (aggregation.rs:440) state `[sum, count, had_overflow]`. -/
structure AvgS where
  sum : Float
  count : Int
  ovf : Bool

/-- `about_to_overflow` (aggregation.rs:331). -/
def aboutToOverflow (a b : Float) : Bool :=
  (if a < 0 then -1.0 else if a > 0 then 1.0 else if a.isNaN then a else
    -- signum(+0.0) = 1, signum(-0.0) = -1
    (if 1.0 / a < 0 then -1.0 else 1.0)) ==
  (if b < 0 then -1.0 else if b > 0 then 1.0 else if b.isNaN then b else
    (if 1.0 / b < 0 then -1.0 else 1.0)) &&
  a.abs >= (1.7976931348623157e308 - b.abs)

def avgStep (s : AvgS) (v : V) : AvgS :=
  let val := v.num
  let count := s.count + 1
  if s.ovf || aboutToOverflow s.sum val then
    let sum := s.sum / Float.ofInt count
    let sum := if s.ovf then sum * Float.ofInt (count - 1) else sum
    { sum := sum + val / Float.ofInt count, count, ovf := true }
  else
    { sum := s.sum + val, count, ovf := s.ovf }

def avgBatch (acc : AvgS) (xs : List V) : AvgS := skipFold avgStep acc xs

/-- `finalize_avg` (aggregation.rs:583). -/
def finalizeAvg (s : AvgS) : Option Float :=
  if s.count == 0 then none else if s.ovf then some s.sum else some (s.sum / Float.ofInt s.count)

def avgInit : AvgS := { sum := 0.0, count := 0, ovf := false }

theorem avg_count (acc : AvgS) (xs : List V) :
    (avgBatch acc xs).count = acc.count + (xs.filter (fun v => !v.isNull)).length := by
  induction xs generalizing acc with
  | nil => simp [avgBatch, skipFold]
  | cons x xs ih =>
    unfold avgBatch; rw [skipFold_cons]
    by_cases h : x.isNull
    · simp only [h, ite_true]
      rw [show skipFold avgStep acc xs = avgBatch acc xs from rfl, ih]; simp [List.filter, h]
    · have h' : x.isNull = false := by simpa using h
      simp only [h', Bool.false_eq_true, if_false]
      rw [show skipFold avgStep (avgStep acc x) xs = avgBatch (avgStep acc x) xs from rfl, ih]
      have : (avgStep acc x).count = acc.count + 1 := by
        simp only [avgStep]; split <;> rfl
      rw [this]; simp [List.filter, h]; omega

/-- Empty (or all-null) input: `avg` is `null` (openCypher, C). -/
theorem avg_empty (xs : List V) (h : ∀ v ∈ xs, v.isNull = true) :
    finalizeAvg (avgBatch avgInit xs) = none := by
  have hc := avg_count avgInit xs
  have : xs.filter (fun v => !v.isNull) = [] := by
    rw [List.filter_eq_nil_iff]; intro v hv; simp [h v hv]
  rw [this] at hc
  have hc0 : (avgBatch avgInit xs).count = 0 := by rw [hc]; simp [avgInit]
  unfold finalizeAvg; simp [hc0]

/-- Below the overflow threshold `avg` accumulates a plain running sum. -/
theorem avg_no_overflow_is_sum (acc : AvgS) (xs : List V)
    (h : (avgBatch acc xs).ovf = false) :
    (avgBatch acc xs).sum = sumBatch acc.sum xs := by
  induction xs generalizing acc with
  | nil => rfl
  | cons x xs ih =>
    simp only [avgBatch, sumBatch, skipFold, List.foldl] at *
    by_cases hx : x.isNull
    · simp [hx] at *; exact ih acc h
    · simp only [hx, Bool.false_eq_true, ↓reduceIte] at *
      -- ovf is monotone: once set it stays set
      have mono : ∀ (s : AvgS) (ys : List V), s.ovf = true →
          (ys.foldl (fun a v => if v.isNull = true then a else avgStep a v) s).ovf = true := by
        intro s ys hs
        induction ys generalizing s with
        | nil => exact hs
        | cons y ys ihy =>
          simp only [List.foldl]
          apply ihy
          split
          · exact hs
          · simp [avgStep, hs]
      by_cases hov : (acc.ovf || aboutToOverflow acc.sum x.num) = true
      · exfalso
        have : (avgStep acc x).ovf = true := by simp [avgStep, hov]
        have := mono _ xs this
        rw [this] at h; exact Bool.noConfusion h
      · simp only [Bool.not_eq_true] at hov
        have e : avgStep acc x = { sum := acc.sum + x.num, count := acc.count + 1, ovf := acc.ovf } := by
          simp [avgStep, hov]
        rw [e] at h ⊢
        exact ih _ h

/-- `stdev_batch` (aggregation.rs:549) state `[sum, values]`. -/
def stdevStep (s : Float × List Float) (v : V) : Float × List Float := (s.1 + v.num, s.2 ++ [v.num])
def stdevBatch (acc : Float × List Float) (xs : List V) := skipFold stdevStep acc xs

/-- `finalize_stdev` (aggregation.rs:675): `n ≤ 1` → `0.0` (C and openCypher). -/
def finalizeStdev (s : Float × List Float) : Float :=
  if s.2.length ≤ 1 then 0.0 else
    let mean := s.1 / Float.ofNat s.2.length
    Float.sqrt ((s.2.map (fun v => (v - mean) * (v - mean))).foldl (· + ·) 0.0
      / Float.ofNat (s.2.length - 1))

theorem stdev_empty : finalizeStdev (stdevBatch (0.0, []) []) = 0.0 := rfl

/-- `percentile_batch` (aggregation.rs:518) state `[percentile, values]`:
every non-null row OVERWRITES the stored percentile (aggregation.rs:539). -/
def pctStep (s : Float × List Float) (vp : V × Float) : Float × List Float :=
  if vp.1.isNull then s else (vp.2, s.2 ++ [vp.1.num])

def pctFold (acc : Float × List Float) (xs : List (V × Float)) := xs.foldl pctStep acc

/-- Last-wins: the stored percentile is the last non-null row's. C keeps the
first row's (`agg_precentile.c`). Confirmed:
`lean_ops_aggregate::bug_percentile_row_dependent_argument_last_wins`. -/
theorem pct_last_wins (acc : Float × List Float) (xs : List (V × Float)) (v : V) (p : Float)
    (hv : v.isNull = false) : (pctFold acc (xs ++ [(v, p)])).1 = p := by
  simp [pctFold, List.foldl_append, pctStep, hv]

/-! ### Batch homomorphisms for each concrete kernel -/

theorem count_hom (a : Int) (bs : List (List V)) :
    batchDriver countBatch a bs = rowDriver countBatch a bs.flatten :=
  vectorized_eq_rowwise countStep a bs
theorem sum_hom (a : Float) (bs : List (List V)) :
    batchDriver sumBatch a bs = rowDriver sumBatch a bs.flatten :=
  vectorized_eq_rowwise sumStep a bs
theorem collect_hom (a : List V) (bs : List (List V)) :
    batchDriver collectBatch a bs = rowDriver collectBatch a bs.flatten :=
  vectorized_eq_rowwise collectStep a bs
theorem avg_hom (a : AvgS) (bs : List (List V)) :
    batchDriver avgBatch a bs = rowDriver avgBatch a bs.flatten :=
  vectorized_eq_rowwise avgStep a bs
theorem stdev_hom (a : Float × List Float) (bs : List (List V)) :
    batchDriver stdevBatch a bs = rowDriver stdevBatch a bs.flatten :=
  vectorized_eq_rowwise stdevStep a bs

theorem minmax_hom {α : Type} (c : Cmp α) (w : Ordering) (a : Option α)
    (bs : List (List (Option α))) :
    bs.foldl (mmBatch c w) a = mmBatch c w a bs.flatten := by
  induction bs generalizing a with
  | nil => rfl
  | cons b bs ih => simp [ih, mmBatch, List.foldl_append]

/-! ### Concrete checks mirroring live C-vs-Rust runs (both engines agree) -/

-- sum of two i64::MAX-ish ints is an f64 in both engines (C: AGG_SUM doubleval):
-- no integer overflow is possible, only precision loss.
#eval sumBatch 0.0 [.int 9223372036854775807, .int 1]           -- 9223372036854775808
-- avg overflow guard: [MAX, MAX] → MAX, [MAX, MAX, -MAX] → 5.99e307 (C: same)
#eval finalizeAvg (avgBatch avgInit [.flt 1.7976931348623157e308, .flt 1.7976931348623157e308])
#eval finalizeAvg (avgBatch avgInit
  [.flt 1.7976931348623157e308, .flt 1.7976931348623157e308, .flt (-1.7976931348623157e308)])
#eval finalizeAvg (avgBatch avgInit [])                         -- none
#eval finalizeStdev (stdevBatch (0.0, []) [.int 1, .int 2, .int 3, .int 4])  -- 1.29099…

end FalkorAggOps
