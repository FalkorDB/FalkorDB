/-!
# Scalar aggregation bodies (functions/aggregation.rs:49-720) and math.rs `register`/`e`/`rand`

Each aggregate's per-row `func(value, acc)` folded over a column. (The vectorised operator
prefers `batch_agg`, see ops_aggregate; these bodies are what `func` computes and what the
per-row path runs when `batch_agg` is absent.) f64 is abstract: `FM` lists the operations
and the facts used, as structure fields — no axioms.
-/
namespace ValueMath.Agg

structure FM (F : Type) where
  zero : F
  one : F
  add : F → F → F
  sub : F → F → F
  mul : F → F → F
  div : F → F → F
  sqrt : F → F
  ofInt : Int → F          -- `i as f64`
  ofNat : Nat → F          -- `usize as f64`
  abs : F → F
  signum : F → F
  max : F                  -- `f64::MAX`
  le : F → F → Bool
  beq : F → F → Bool
  trunc : F → F
  fract : F → F
  e : F                    -- `std::f64::consts::E`
  /-- `x.trunc() + x.fract() = x` (exact in IEEE for finite x; std docs). -/
  trunc_fract : ∀ x, add (trunc x) (fract x) = x
  sqrt_zero : sqrt zero = zero

variable {F : Type} (M : FM F)

inductive AV (F : Type) where
  | null | bool (b : Bool) | int (i : Int) | float (f : F) | list (vs : List (AV F)) | other

inductive Res (α : Type) where
  | ok (a : α) | unreachable

/-- `get_numeric` -/
def num : AV F → F
  | .int i => M.ofInt i
  | .float f => f
  | _ => M.zero

/-! ## collect (aggregation.rs:55) -/

def collect : AV F → AV F → Res (AV F)
  | a, .null => .ok (.list [a])
  | .null, .list l => .ok (.list l)
  | a, .list l => .ok (.list (l ++ [a]))
  | _, _ => .unreachable

def foldAgg (f : AV F → AV F → Res (AV F)) : AV F → List (AV F) → Res (AV F)
  | acc, [] => .ok acc
  | acc, v :: vs => match f v acc with
    | .ok acc' => foldAgg f acc' vs
    | .unreachable => .unreachable

def isNull : AV F → Bool
  | .null => true
  | _ => false

/-- `collect` from its initial `[]` gathers the non-null inputs in order. -/
theorem collect_fold (vs : List (AV F)) :
    ∀ l, foldAgg collect (.list l) vs = .ok (.list (l ++ vs.filter (! isNull ·))) := by
  induction vs with
  | nil => intro l; simp [foldAgg]
  | cons v vs ih =>
    intro l
    cases v <;> simp [foldAgg, collect, ih, isNull]

/-! ## count (aggregation.rs:77) -/

def count : AV F → AV F → Res (AV F)
  | .null, s => .ok s
  | _, .int a => .ok (.int (a + 1))
  | _, _ => .unreachable

theorem count_fold (vs : List (AV F)) :
    ∀ n : Int, foldAgg count (.int n) vs = .ok (.int (n + (vs.filter (! isNull ·)).length)) := by
  induction vs with
  | nil => intro n; simp [foldAgg]
  | cons v vs ih =>
    intro n
    cases v <;> simp [foldAgg, count, ih, isNull] <;> omega

/-! ## sum (aggregation.rs:95) -/

def sum : AV F → AV F → Res (AV F)
  | .null, acc => .ok acc
  | .int a, .float b => .ok (.float (M.add (M.ofInt a) b))
  | .float a, .int b => .ok (.float (M.add a (M.ofInt b)))
  | .float a, .float b => .ok (.float (M.add a b))
  | _, _ => .unreachable

/-- The declared argument type `Int | Float | Null`. -/
def numOrNull : AV F → Bool
  | .null | .int _ | .float _ => true
  | _ => false

/-- From the `Float(0.0)` initial value the accumulator stays a Float, so the
`unreachable!` arm (e.g. `(Int, Int)`) is dead; the result is the left fold of `+`. -/
theorem sum_fold (vs : List (AV F)) (h : vs.all numOrNull) :
    ∀ acc, foldAgg (sum M) (.float acc) vs =
      .ok (.float ((vs.filter (! isNull ·)).foldl (fun s v =>
        match v with | .int i => M.add (M.ofInt i) s | .float f => M.add f s | _ => s) acc)) := by
  induction vs with
  | nil => intro acc; simp [foldAgg]
  | cons v vs ih =>
    intro acc
    simp only [List.all_cons, Bool.and_eq_true] at h
    cases v <;> simp_all [foldAgg, sum, isNull, numOrNull]

/-! ## min (aggregation.rs:140) -/

/-- `b.compare_value(&a)` abstracted: ordering + `ComparedNull` flag. -/
def min (cmp : AV F → AV F → Ordering × Bool) : AV F → AV F → Res (AV F)
  | a, b => let (o, cn) := cmp b a; if o = .gt || cn then .ok a else .ok b

/-- With the accumulator starting at `Null` (compared-null with everything), the first
non-null value is taken; afterwards `min` keeps the smaller under `compare_value`. -/
theorem min_step (cmp : AV F → AV F → Ordering × Bool) (a b : AV F) :
    min cmp a b = if (cmp b a).1 = .gt || (cmp b a).2 then .ok a else .ok b := rfl

theorem min_from_null (cmp : AV F → AV F → Ordering × Bool) (a : AV F)
    (hn : (cmp .null a).2 = true) : min cmp a .null = .ok a := by
  simp [min, hn]

/-! ## avg (aggregation.rs:162) and `about_to_overflow` (:331) -/

/-- `a.signum() == b.signum() && a.abs() >= (f64::MAX - b.abs())` -/
def aboutToOverflow (a b : F) : Bool := M.beq (M.signum a) (M.signum b) && M.le (M.sub M.max (M.abs b)) (M.abs a)

/-- One avg step on `[sum, count, had_overflow]`. -/
def avgStep (val : F) : F × Int × Bool → F × Int × Bool
  | (s, c, ov) =>
    let c := c + 1
    if ov || aboutToOverflow M s val then
      let s1 := M.div s (M.ofInt c)
      let s2 := if ov then M.mul s1 (M.ofInt (c - 1)) else s1
      (M.add s2 (M.div val (M.ofInt c)), c, true)
    else (M.add s val, c, false)

def avg : AV F → AV F → Res (AV F)
  | .null, ctx => .ok ctx
  | v@(.int _), .list [.float s, .int c, .bool ov]
  | v@(.float _), .list [.float s, .int c, .bool ov] =>
    let r := avgStep M (num M v) (s, c, ov)
    .ok (.list [.float r.1, .int r.2.1, .bool r.2.2])
  | _, _ => .unreachable

/-- The count slot counts the non-null inputs; without overflow the sum slot is the plain
running sum (the incremental-mean mode is entered only when `about_to_overflow`). -/
theorem avg_count (vs : List (AV F)) (h : vs.all numOrNull) :
    ∀ s c ov, ∃ s' ov', foldAgg (avg M) (.list [.float s, .int c, .bool ov]) vs =
      .ok (.list [.float s', .int (c + (vs.filter (! isNull ·)).length), .bool ov']) := by
  induction vs with
  | nil => intro s c ov; exact ⟨s, ov, by simp [foldAgg]⟩
  | cons v vs ih =>
    intro s c ov
    simp only [List.all_cons, Bool.and_eq_true] at h
    cases v with
    | null => obtain ⟨s', ov', e⟩ := ih h.2 s c ov; exact ⟨s', ov', by simpa [foldAgg, avg, isNull] using e⟩
    | int i =>
      obtain ⟨s', ov', e⟩ := ih h.2 (avgStep M (num M (.int i)) (s, c, ov)).1
        (avgStep M (num M (.int i)) (s, c, ov)).2.1 (avgStep M (num M (.int i)) (s, c, ov)).2.2
      refine ⟨s', ov', ?_⟩
      have hc : (avgStep M (num M (.int i)) (s, c, ov)).2.1 = c + 1 := by
        simp only [avgStep]; split <;> rfl
      simp only [foldAgg, avg]
      rw [e, hc]
      simp [isNull]; omega
    | float f =>
      obtain ⟨s', ov', e⟩ := ih h.2 (avgStep M (num M (.float f)) (s, c, ov)).1
        (avgStep M (num M (.float f)) (s, c, ov)).2.1 (avgStep M (num M (.float f)) (s, c, ov)).2.2
      refine ⟨s', ov', ?_⟩
      have hc : (avgStep M (num M (.float f)) (s, c, ov)).2.1 = c + 1 := by
        simp only [avgStep]; split <;> rfl
      simp only [foldAgg, avg]
      rw [e, hc]
      simp [isNull]; omega
    | _ => simp [numOrNull] at h

theorem avgStep_plain (val s : F) (c : Int) (h : aboutToOverflow M s val = false) :
    avgStep M val (s, c, false) = (M.add s val, c + 1, false) := by
  simp [avgStep, h]

/-! ## percentile (aggregation.rs:226) and stdev (:286) -/

/-- `percentileDisc`/`percentileCont` step: store `p`, push the value as a Float. -/
def percentile (v p : AV F) : AV F → Res (AV F)
  | ctx => match v with
    | .null => .ok ctx
    | _ => match ctx with
      | .list [.float _, .list xs] => .ok (.list [.float (num M p), .list (xs ++ [.float (num M v)])])
      | _ => .unreachable

theorem percentile_collects (p : AV F) (vs : List (AV F)) :
    ∀ q xs, ∃ q', foldAgg (fun v acc => percentile M v p acc) (.list [.float q, .list xs]) vs =
      .ok (.list [.float q', .list (xs ++ (vs.filter (! isNull ·)).map (fun v => .float (num M v)))]) := by
  induction vs with
  | nil => intro q xs; exact ⟨q, by simp [foldAgg]⟩
  | cons v vs ih =>
    intro q xs
    cases v with
    | null => obtain ⟨q', e⟩ := ih q xs; exact ⟨q', by simpa [foldAgg, percentile, isNull] using e⟩
    | _ =>
      first
      | (obtain ⟨q', e⟩ := ih (num M p) (xs ++ [.float (num M _)])
         exact ⟨q', by simpa [foldAgg, percentile, isNull] using e⟩)

/-- `stDev`/`stDevP` step on `[sum, values]`. -/
def stdev : AV F → AV F → Res (AV F)
  | .null, ctx => .ok ctx
  | v@(.int _), .list [.float s, .list xs]
  | v@(.float _), .list [.float s, .list xs] =>
    .ok (.list [.float (M.add s (num M v)), .list (xs ++ [.float (num M v)])])
  | _, _ => .unreachable

theorem stdev_collects (vs : List (AV F)) (h : vs.all numOrNull) :
    ∀ s xs, ∃ s', foldAgg (stdev M) (.list [.float s, .list xs]) vs =
      .ok (.list [.float s', .list (xs ++ (vs.filter (! isNull ·)).map (fun v => .float (num M v)))]) := by
  induction vs with
  | nil => intro s xs; exact ⟨s, by simp [foldAgg]⟩
  | cons v vs ih =>
    intro s xs
    simp only [List.all_cons, Bool.and_eq_true] at h
    cases v with
    | null => obtain ⟨s', e⟩ := ih h.2 s xs; exact ⟨s', by simpa [foldAgg, stdev, isNull] using e⟩
    | int i =>
      obtain ⟨s', e⟩ := ih h.2 (M.add s (num M (.int i))) (xs ++ [.float (num M (.int i))])
      exact ⟨s', by simpa [foldAgg, stdev, isNull] using e⟩
    | float f =>
      obtain ⟨s', e⟩ := ih h.2 (M.add s (num M (.float f))) (xs ++ [.float (num M (.float f))])
      exact ⟨s', by simpa [foldAgg, stdev, isNull] using e⟩
    | _ => simp [numOrNull] at h

/-! ## modf (:669), finalize_stdev (:675), finalize_stdevp (:697) -/

/-- `modf(x) = (x.fract(), x.trunc())` -/
def modf (x : F) : F × F := (M.fract x, M.trunc x)

theorem modf_recombine (x : F) : M.add (modf M x).2 (modf M x).1 = x := M.trunc_fract x

def sqDevSum (xs : List F) (mean : F) : F :=
  xs.foldl (fun acc v => M.add acc (M.mul (M.sub v mean) (M.sub v mean))) M.zero

/-- `finalize_stdev`: sample std-dev (`n − 1`); 0 for fewer than 2 values. -/
def finalizeStdev (s : F) (xs : List F) : F :=
  if xs.length ≤ 1 then M.zero
  else M.sqrt (M.div (sqDevSum M xs (M.div s (M.ofNat xs.length))) (M.ofNat (xs.length - 1)))

/-- `finalize_stdevp`: population std-dev (`n`); 0 for no values. -/
def finalizeStdevp (s : F) (xs : List F) : F :=
  if xs.length = 0 then M.zero
  else M.sqrt (M.div (sqDevSum M xs (M.div s (M.ofNat xs.length))) (M.ofNat xs.length))

theorem finalizeStdev_small (s : F) (xs : List F) (h : xs.length ≤ 1) : finalizeStdev M s xs = M.zero := by
  simp [finalizeStdev, h]

theorem finalizeStdevp_empty (s : F) : finalizeStdevp M s [] = M.zero := by simp [finalizeStdevp]

/-- The two finalizers differ only in the divisor (and the 1-value case: sample is 0,
population is `sqrt(0 / 1)`). -/
theorem finalize_divisors (s : F) (xs : List F) (h : 2 ≤ xs.length) :
    finalizeStdev M s xs = M.sqrt (M.div (sqDevSum M xs (M.div s (M.ofNat xs.length))) (M.ofNat (xs.length - 1))) ∧
    finalizeStdevp M s xs = M.sqrt (M.div (sqDevSum M xs (M.div s (M.ofNat xs.length))) (M.ofNat xs.length)) := by
  have h1 : ¬ xs.length ≤ 1 := by omega
  have h2 : xs.length ≠ 0 := by omega
  simp [finalizeStdev, finalizeStdevp, h1, h2]

/-! ## register (aggregation.rs:49) -/

def aggRegistered : List String :=
  ["collect", "count", "sum", "max", "min", "avg", "percentileDisc", "percentileCont", "stDev", "stDevP"]

theorem aggRegistered_nodup : aggRegistered.Nodup := by decide

/-! ## math.rs: register (:42), e (:77), rand (:186) -/

def mathRegistered : List String :=
  ["abs", "ceil", "e", "exp", "floor", "log", "log10", "randomUUID", "pow", "rand", "round", "sign",
   "sqrt", "range", "coalesce"]

theorem mathRegistered_nodup : mathRegistered.Nodup := by decide

/-- `e()`: the constant, no arguments. -/
def eFn (args : List (AV F)) : Res (AV F) := if args = [] then .ok (.float M.e) else .unreachable

theorem eFn_spec : eFn M [] = .ok (.float M.e) := rfl

/-- `rng.random_range(0.0..1.0)`: the RNG is an argument, with the rand-crate contract. -/
structure Rng (F : Type) (M : FM F) where
  draw : Nat → F
  in_range : ∀ s, M.le M.zero (draw s) = true ∧ M.le M.one (draw s) = false

def randFn (R : Rng F M) (seed : Nat) (args : List (AV F)) : Res (AV F) :=
  if args = [] then .ok (.float (R.draw seed)) else .unreachable

theorem randFn_range (R : Rng F M) (seed : Nat) :
    ∃ x, randFn M R seed [] = .ok (.float x) ∧ M.le M.zero x = true ∧ M.le M.one x = false :=
  ⟨R.draw seed, rfl, R.in_range seed⟩

end ValueMath.Agg
