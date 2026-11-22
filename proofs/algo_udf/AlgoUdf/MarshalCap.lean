import AlgoUdf.MarshalThm
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false
/-! # UDF value conversion is depth-capped

Both conversions are instances of `Step` (`Nesting.lean`): `fromStep H` is `js_to_value_level` over
the QuickJS heap `H`, `toStep` is `value_to_js_at`'s check + `value_to_js_level`.

Main results (`MAX = MAX_UDF_VALUE_NESTING = 1024`):
* `fromJs_frames`, `toJs_frames`: **`MAX + 1` conversion levels always suffice**, for every input —
  including a self-containing JS array (`Hself`).
* `fromJs_past_cap`, `toJs_past_cap`: an input nested past the cap is an `Err`, never an overflow;
  exactly `udf_nesting_error()` for the Rust tests' shapes (`fromJs_nestArr_past`, `self_array_error`,
  `toJs_nestList_past`).
* `fromJs_needs_all_frames`: the bound is tight (`MAX` frames do not suffice for `MAX` nested arrays);
  the uncapped conversion (`cap = none`) needs `n + 1` frames for `n` nested arrays, which is why
  the bound is exactly `MAX + 1`.
* `cap_asymmetry_vecf32`: the two caps count differently — `value_to_js` does not count a VecF32
  leaf as a level, `js_to_value` does, so `MAX` lists around a vecf32 convert into JS but not
  back. (A Path is a leaf both ways: `js_to_value` reads it from the entity class.)
-/
namespace AlgoUdf.Marshal

/-! ## `R.bind` helpers -/

theorem bind_ne_none {α β} {x : R α} {k : α → R β} (hx : x ≠ none) (hk : ∀ a, x = R.ok a → k a ≠ none) :
    R.bind x k ≠ none := by
  match x, hx with
  | some (.error _), _ => exact fun h => nomatch h
  | some (.ok a), _ => exact hk a rfl

theorem bind_ne_none_left {α β} {x : R α} {k : α → R β} (h : R.bind x k ≠ none) : x ≠ none := by
  intro hx; subst hx; exact h rfl

theorem bind_ok_ne {α β} {a : α} {x : R α} {k : α → R β} (h : R.bind x k ≠ none) (hx : x = R.ok a) :
    k a ≠ none := by
  subst hx; exact h

/-! ## The element loop -/

theorem listLoop_total {α β} (conv : α → R β) : ∀ l : List α, (∀ x ∈ l, conv x ≠ none) → listLoop conv l ≠ none
  | [], _ => fun h => nomatch h
  | x :: xs, h => by
    simp only [listLoop]
    refine bind_ne_none (h x (by simp)) fun a _ => bind_ne_none
      (listLoop_total conv xs fun y hy => h y (by simp [hy])) fun _ _ => fun h => nomatch h

theorem listLoop_ext {α β} (conv conv' : α → R β) : ∀ l : List α, (∀ x ∈ l, conv x = conv' x) →
    listLoop conv l = listLoop conv' l
  | [], _ => rfl
  | x :: xs, h => by
    simp only [listLoop]
    rw [h x (by simp), listLoop_ext conv conv' xs fun y hy => h y (by simp [hy])]

theorem listLoop_err {α β} (conv : α → R β) : ∀ (l : List α) (x : α) (e : String), (∀ y ∈ l, conv y ≠ none) →
    x ∈ l → conv x = R.err e → ∃ e', listLoop conv l = R.err e'
  | [], _, _, _, hx, _ => by simp at hx
  | y :: ys, x, e, hn, hx, he => by
    simp only [listLoop]
    cases hy : conv y with
    | none => exact absurd hy (hn y (by simp))
    | some r =>
      cases r with
      | error e' => exact ⟨e', rfl⟩
      | ok v =>
        simp only [R.bind_ok]
        rcases List.mem_cons.1 hx with rfl | hx
        · rw [hy] at he; cases he
        · obtain ⟨e', he'⟩ := listLoop_err conv ys x e (fun z hz => hn z (by simp [hz])) hx he
          rw [he']; exact ⟨e', rfl⟩

theorem listLoop_mono {α β} (conv conv' : α → R β) (hc : ∀ x, conv x ≠ none → conv' x = conv x) :
    ∀ l : List α, listLoop conv l ≠ none → listLoop conv' l = listLoop conv l
  | [], _ => rfl
  | x :: xs, h => by
    simp only [listLoop] at h ⊢
    have hx := bind_ne_none_left h
    rw [hc x hx]
    cases hcx : conv x with
    | none => exact absurd hcx hx
    | some r =>
      cases r with
      | error e => rfl
      | ok v =>
        simp only [R.bind_ok]
        have h2 := bind_ok_ne h hcx
        rw [listLoop_mono conv conv' hc xs (bind_ne_none_left h2)]

/-! ## `js_to_value` as a `Step` -/

/-- The arms that check the depth (array :321, plain object :399). -/
def needsS : Shape → Bool
  | .done _ => false
  | _ => true

/-- The children `js_to_value_level` converts at `depth + 1`. -/
def kidsS (H : Heap) : Shape → List JsV
  | .done _ => []
  | .list items => items
  | .vec _ => []
  | .map ps => (keptProps ps).map (·.2)

section
variable (H : Heap)

theorem shape_leaf_indep (b b' : Bool) (conv conv' : JsV → R RV) (s : Shape) (h : needsS s = false) :
    runShape H b conv s = runShape H b' conv' s := by
  cases s <;> simp [needsS] at h; rfl

theorem shape_leaf_some (b : Bool) (conv : JsV → R RV) (s : Shape) (h : needsS s = false) :
    runShape H b conv s ≠ none := by
  cases s <;> simp [needsS] at h; exact fun h => nomatch h

theorem shape_chk_err (conv : JsV → R RV) (s : Shape) (h : needsS s = true) :
    runShape H true conv s = R.err nestErr := by
  cases s <;> simp [needsS] at h <;> rfl

theorem shape_ext (b : Bool) (conv conv' : JsV → R RV) (s : Shape) (h : ∀ x ∈ kidsS H s, conv x = conv' x) :
    runShape H b conv s = runShape H b conv' s := by
  cases s with
  | done r => rfl
  | list items => simp only [runShape]; rw [listLoop_ext conv conv' items h]
  | vec items => rfl
  | map ps => simp only [runShape]; rw [listLoop_ext conv conv' ((keptProps ps).map (fun x : List Char × JsV => x.2)) h]

theorem shape_total (b : Bool) (conv : JsV → R RV) (s : Shape) (h : ∀ x, conv x ≠ none) :
    runShape H b conv s ≠ none := by
  cases s with
  | done r => exact fun h => nomatch h
  | vec items => simp only [runShape]; split <;> exact fun h => nomatch h
  | list items =>
    simp only [runShape]; split
    · exact fun h => nomatch h
    · exact bind_ne_none (listLoop_total conv items fun x _ => h x) fun _ _ => fun h => nomatch h
  | map ps =>
    simp only [runShape]; split
    · exact fun h => nomatch h
    · exact bind_ne_none (listLoop_total conv _ fun x _ => h x) fun _ _ => fun h => nomatch h

theorem shape_kid_err (conv : JsV → R RV) (s : Shape) (x : JsV) (e : String) (hn : ∀ y, conv y ≠ none)
    (hx : x ∈ kidsS H s) (he : conv x = R.err e) : ∃ e', runShape H false conv s = R.err e' := by
  cases s with
  | done r => simp [kidsS] at hx
  | vec items => simp [kidsS] at hx
  | list items =>
    obtain ⟨e', he'⟩ := listLoop_err conv items x e (fun y _ => hn y) hx he
    simp only [runShape, Bool.false_eq_true, ite_false, he', R.bind_err]; exact ⟨e', rfl⟩
  | map ps =>
    obtain ⟨e', he'⟩ := listLoop_err conv ((keptProps ps).map (fun x : List Char × JsV => x.2)) x e (fun y _ => hn y) hx he
    simp only [runShape, Bool.false_eq_true, ite_false, he', R.bind_err]; exact ⟨e', rfl⟩

theorem shape_mono (b : Bool) (conv conv' : JsV → R RV) (s : Shape) (h : runShape H b conv s ≠ none)
    (hc : ∀ x, conv x ≠ none → conv' x = conv x) : runShape H b conv' s = runShape H b conv s := by
  cases s with
  | done r => rfl
  | vec items => rfl
  | list items =>
    simp only [runShape] at h ⊢
    split
    · rfl
    · rename_i hb; simp only [hb] at h
      rw [listLoop_mono conv conv' hc items (bind_ne_none_left h)]
  | map ps =>
    simp only [runShape] at h ⊢
    split
    · rfl
    · rename_i hb; simp only [hb] at h
      rw [listLoop_mono conv conv' hc _ (bind_ne_none_left h)]

/-- **`js_to_value_level` is a `Step`** over the heap `H`. -/
def fromStep : Step JsV RV where
  step b conv v := fromLevel H b conv v
  needs v := needsS (classify (deref H v))
  kids v := kidsS H (classify (deref H v))
  leaf_indep b b' conv conv' v h := shape_leaf_indep H b b' conv conv' _ h
  leaf_some b conv v h := shape_leaf_some H b conv _ h
  chk_err conv v h := shape_chk_err H conv _ h
  ext b conv conv' v h := shape_ext H b conv conv' _ h
  total b conv v h := shape_total H b conv _ h
  kid_err conv v x e hn hx he := shape_kid_err H conv _ x e hn hx he
  mono b conv conv' v h hc := shape_mono H b conv conv' _ h hc

@[simp] theorem fromStep_needs (v : JsV) : (fromStep H).needs v = needsS (classify (deref H v)) := rfl
@[simp] theorem fromStep_kids (v : JsV) : (fromStep H).kids v = kidsS H (classify (deref H v)) := rfl

theorem fromF_eq_run (cap : Option Nat) : ∀ f d v, fromF H cap f d v = (fromStep H).run cap f d v
  | 0, _, _ => rfl
  | f + 1, d, v => by
    simp only [fromF, Step.run]
    show fromLevel H _ _ v = fromLevel H _ _ v
    congr 1; funext x; exact fromF_eq_run cap f (d + 1) x

end

/-! ## `value_to_js` as a `Step` -/

/-- `value_to_js_at`'s body (:116-121): the depth check, then `value_to_js_level`. -/
def toStepFn (b : Bool) (conv : RV → R JsV) (v : RV) : R JsV :=
  if needsV v && b then R.err nestErr else toLevel conv v

theorem toLevel_leaf_indep (conv conv' : RV → R JsV) (v : RV) (h : needsV v = false) :
    toLevel conv v = toLevel conv' v := by
  cases v <;> simp [needsV] at h <;> rfl

theorem toLevel_leaf_some (conv : RV → R JsV) (v : RV) (h : needsV v = false) : toLevel conv v ≠ none := by
  cases v <;> simp [needsV] at h <;> simp only [toLevel] <;> (try split) <;> exact fun h => nomatch h

def toStep : Step RV JsV where
  step := toStepFn
  needs := needsV
  kids := kidsV
  leaf_indep b b' conv conv' v h := by
    simp only [toStepFn, h, Bool.false_and, Bool.false_eq_true, ite_false]; exact toLevel_leaf_indep _ _ v h
  leaf_some b conv v h := by
    simp only [toStepFn, h, Bool.false_and, Bool.false_eq_true, ite_false]; exact toLevel_leaf_some _ v h
  chk_err conv v h := by simp [toStepFn, h]
  ext b conv conv' v h := by
    cases v with
    | list xs => simp only [toStepFn, toLevel]; rw [listLoop_ext conv conv' xs h]
    | map kv =>
      simp only [toStepFn, toLevel]
      rw [listLoop_ext conv conv' (kv.map (fun x : List Char × RV => x.2)) h]
    | _ => rfl
  total b conv v h := by
    unfold toStepFn; split
    · exact fun h => nomatch h
    · cases hv : needsV v
      · exact toLevel_leaf_some conv v hv
      · cases v <;> simp [needsV] at hv
        · exact bind_ne_none (listLoop_total conv _ fun x _ => h x) fun _ _ => fun h => nomatch h
        · exact bind_ne_none (listLoop_total conv _ fun x _ => h x) fun _ _ => fun h => nomatch h
  kid_err conv v x e hn hx he := by
    cases v with
    | list xs =>
      obtain ⟨e', he'⟩ := listLoop_err conv xs x e (fun y _ => hn y) hx he
      simp only [toStepFn, toLevel, Bool.and_false, Bool.false_eq_true, ite_false, he', R.bind_err]
      exact ⟨e', rfl⟩
    | map kv =>
      obtain ⟨e', he'⟩ := listLoop_err conv (kv.map (fun x : List Char × RV => x.2)) x e (fun y _ => hn y) hx he
      simp only [toStepFn, toLevel, Bool.and_false, Bool.false_eq_true, ite_false, he', R.bind_err]
      exact ⟨e', rfl⟩
    | _ => simp [kidsV] at hx
  mono b conv conv' v h hc := by
    unfold toStepFn at h ⊢
    split
    · rfl
    · rename_i hb
      simp only [hb, Bool.false_eq_true, ite_false] at h
      cases v with
      | list xs =>
        simp only [toLevel] at h ⊢
        rw [listLoop_mono conv conv' hc xs (bind_ne_none_left h)]
      | map kv =>
        simp only [toLevel] at h ⊢
        rw [listLoop_mono conv conv' hc _ (bind_ne_none_left h)]
      | _ => rfl

theorem toF_eq_run (cap : Option Nat) : ∀ f d v, toF cap f d v = toStep.run cap f d v
  | 0, _, _ => rfl
  | f + 1, d, v => by
    simp only [toF, Step.run]
    show toStepFn _ _ v = toStepFn _ _ v
    congr 1; funext x; exact toF_eq_run cap f (d + 1) x

@[simp] theorem toStep_needs (v : RV) : toStep.needs v = needsV v := rfl
@[simp] theorem toStep_kids (v : RV) : toStep.kids v = kidsV v := rfl

/-- A marker-less array is a list. -/
theorem classify_arr_nil (items : List JsV) : classify (.arr items []) = .list items := rfl
theorem deref_arr (H : Heap) (items ps) : deref H (.arr items ps) = .arr items ps := rfl

/-! ## `js_to_value` -/

/-- `v` nests at most `MAX` levels (as `js_to_value` counts them). -/
def fitsJs (H : Heap) (v : JsV) : Bool := (fromStep H).within MAX v

/-- **`js_to_value` never needs more than `MAX + 1` conversion levels, for any heap** — cyclic
    ones included: the stack can no longer be exhausted by the value. -/
theorem fromJs_frames (H : Heap) (v : JsV) : fromF H (some MAX) (MAX + 1) 0 v ≠ none := by
  rw [fromF_eq_run]; exact (fromStep H).run_frames MAX (MAX + 1) 0 v (by omega) (by omega)

/-- More stack never changes the answer: `fromJs` is the result on any stack of at least `MAX + 1`. -/
theorem fromF_big_stack (H : Heap) (v : JsV) (k : Nat) :
    fromF H (some MAX) (MAX + 1 + k) 0 v = some (fromJs H v) := by
  rw [fromF_eq_run, (fromStep H).run_mono_add _ _ _ _ (by rw [← fromF_eq_run]; exact fromJs_frames H v)]
  have := fromJs_frames H v
  unfold fromJs; rw [fromF_eq_run] at this ⊢
  revert this; cases (fromStep H).run (some MAX) (MAX + 1) 0 v <;> simp [Rget]

/-- **Past the cap, `js_to_value` is an error** (never a stack overflow). -/
theorem fromJs_past_cap (H : Heap) (v : JsV) (h : fitsJs H v = false) : ∃ e, fromJs H v = .error e := by
  obtain ⟨e, he⟩ := (fromStep H).run_past_cap MAX (MAX + 1) 0 v (by omega) (by omega) h
  refine ⟨e, ?_⟩
  unfold fromJs; rw [fromF_eq_run, he]; rfl

/-! ### A self-containing array (`const a = []; a.push(a)`) -/

/-- The heap of `a = [a]`: cell 0 is an array whose only item is a reference to itself. -/
def Hself : Heap := fun _ => .arr [.ref 0] []

/-- **#3336: it is now exactly `udf_nesting_error()`** (Rust test `a_self_containing_array_is_an_error`). -/
theorem self_array_error : fromJs Hself (.ref 0) = .error nestErr := by
  suffices ∀ f d, d ≤ MAX → MAX + 1 ≤ f + d → fromF Hself (some MAX) f d (.ref 0) = R.err nestErr by
    unfold fromJs; rw [this (MAX + 1) 0 (by omega) (by omega)]; rfl
  intro f
  induction f with
  | zero => intro d h1 h2; simp [MAX] at *; omega
  | succ f ih =>
    intro d h1 h2
    simp only [fromF, fromLevel, deref, Hself, classify_arr_nil]
    by_cases hd : MAX ≤ d
    · simp only [runShape, chk, decide_eq_true hd, ite_true]
    · simp only [runShape, chk, decide_eq_false hd, Bool.false_eq_true, ite_false, listLoop,
        ih (d + 1) (by omega) (by omega), R.bind_err]
      rfl

/-! ### `n` nested arrays (the Rust tests' `js_nested_array`) -/

def one : JsV := .num (ofInt 1)
/-- `n` single-element arrays around `1`. -/
def nestArr : Nat → JsV
  | 0 => one
  | n + 1 => .arr [nestArr n] []
/-- `n` single-element lists around `1`. -/
def nestList : Nat → RV
  | 0 => .int 1
  | n + 1 => .list [nestList n]

theorem fits_nestArr (H : Heap) : ∀ r n, (fromStep H).within r (nestArr n) = decide (n ≤ r)
  | 0, 0 => rfl
  | 0, n + 1 => by simp [Step.within, nestArr, deref_arr, classify_arr_nil, needsS]
  | r + 1, 0 => rfl
  | r + 1, n + 1 => by
    simp [Step.within, nestArr, deref_arr, classify_arr_nil, kidsS, fits_nestArr H r n]

theorem fromF_nestArr_ok (H : Heap) (c : Nat) : ∀ n f d, n + d ≤ c → n + 1 ≤ f →
    fromF H (some c) f d (nestArr n) = R.ok (nestList n)
  | 0, f + 1, d, _, _ => rfl
  | n + 1, f + 1, d, h1, h2 => by
    simp only [fromF, fromLevel, nestArr, deref_arr, classify_arr_nil]
    have hd : ¬ c ≤ d := by omega
    simp only [runShape, chk, decide_eq_false hd, Bool.false_eq_true, ite_false, listLoop,
      fromF_nestArr_ok H c n f (d + 1) (by omega) (by omega), R.bind_ok]
    rfl

theorem fromF_nestArr_err (H : Heap) (c : Nat) : ∀ n f d, c < n + d → d ≤ c → c + 1 ≤ f + d →
    fromF H (some c) f d (nestArr n) = R.err nestErr
  | 0, _, d, h1, h2, _ => by omega
  | n + 1, 0, d, _, h2, h3 => by omega
  | n + 1, f + 1, d, h1, h2, h3 => by
    simp only [fromF, fromLevel, nestArr, deref_arr, classify_arr_nil]
    by_cases hd : c ≤ d
    · simp only [runShape, chk, decide_eq_true hd, ite_true]
    · simp only [runShape, chk, decide_eq_false hd, Bool.false_eq_true, ite_false, listLoop,
        fromF_nestArr_err H c n f (d + 1) (by omega) (by omega) (by omega), R.bind_err]
      rfl

/-- **At the cap, an array of `MAX` levels converts** (Rust test `js_array_at_the_cap_converts`). -/
theorem fromJs_nestArr_ok (H : Heap) (n : Nat) (h : n ≤ MAX) : fromJs H (nestArr n) = .ok (nestList n) := by
  unfold fromJs; rw [fromF_nestArr_ok H MAX n (MAX + 1) 0 (by omega) (by omega)]; rfl

/-- **Past it, exactly `udf_nesting_error()`** (Rust tests `js_array_past_the_cap_is_an_error` and,
    with `n = 300000`, `the_advisory_array_is_an_error`). -/
theorem fromJs_nestArr_past (H : Heap) (n : Nat) (h : MAX < n) : fromJs H (nestArr n) = .error nestErr := by
  unfold fromJs; rw [fromF_nestArr_err H MAX n (MAX + 1) 0 (by omega) (by omega) (by omega)]; rfl

theorem advisory_array_error : fromJs H0 (nestArr 300000) = .error nestErr :=
  fromJs_nestArr_past H0 300000 (by decide)

/-- The `MAX + 1` bound is tight: `MAX` levels of stack do not convert `MAX` nested arrays. -/
theorem fromJs_needs_all_frames (H : Heap) : fromF H (some MAX) MAX 0 (nestArr MAX) = none := by
  rw [fromF_eq_run, ← (fromStep H).run_eq_uncapped MAX MAX 0 _ (by rw [fits_nestArr]; simp), ← fromF_eq_run]
  -- without the cap, `n` nested arrays need `n + 1` levels, so `MAX` levels overflow
  suffices ∀ n f d, f ≤ n → fromF H none f d (nestArr n) = none from this MAX MAX 0 (Nat.le_refl _)
  intro n
  induction n with
  | zero => intro f d h; rw [Nat.le_zero.1 h]; rfl
  | succ n ih =>
    intro f d h
    cases f with
    | zero => rfl
    | succ f =>
      simp only [fromF, fromLevel, nestArr, deref_arr, classify_arr_nil]
      simp only [runShape, chk, Bool.false_eq_true, ite_false, listLoop, ih f (d + 1) (by omega), R.bind_none]

/-! ## `value_to_js` -/

/-- `v` nests at most `MAX` lists / maps. -/
def fitsV (v : RV) : Bool := toStep.within MAX v

theorem toJs_frames (v : RV) : toF (some MAX) (MAX + 1) 0 v ≠ none := by
  rw [toF_eq_run]; exact toStep.run_frames MAX (MAX + 1) 0 v (by omega) (by omega)

theorem toF_big_stack (v : RV) (k : Nat) : toF (some MAX) (MAX + 1 + k) 0 v = some (toJs v) := by
  rw [toF_eq_run, toStep.run_mono_add _ _ _ _ (by rw [← toF_eq_run]; exact toJs_frames v)]
  have := toJs_frames v
  unfold toJs; rw [toF_eq_run] at this ⊢
  revert this; cases toStep.run (some MAX) (MAX + 1) 0 v <;> simp [Rget]

theorem toJs_past_cap (v : RV) (h : fitsV v = false) : ∃ e, toJs v = .error e := by
  obtain ⟨e, he⟩ := toStep.run_past_cap MAX (MAX + 1) 0 v (by omega) (by omega) h
  refine ⟨e, ?_⟩
  unfold toJs; rw [toF_eq_run, he]; rfl

/-- **A value within the cap converts exactly as before #3336** (the uncapped code, `cap = none`),
for every stack size: the new depth check never fires on a JS value that fits. -/
theorem fromF_fits_eq_uncapped (H : Heap) (v : JsV) (f : Nat) (h : fitsJs H v = true) :
    fromF H none f 0 v = fromF H (some MAX) f 0 v := by
  rw [fromF_eq_run, fromF_eq_run]
  exact (fromStep H).run_eq_uncapped MAX f 0 v (by simpa [fitsJs] using h)

/-- The same for `value_to_js`: a Cypher value that nests at most `MAX` lists / maps converts as before. -/
theorem toF_fits_eq_uncapped (v : RV) (f : Nat) (h : fitsV v = true) :
    toF none f 0 v = toF (some MAX) f 0 v := by
  rw [toF_eq_run, toF_eq_run]
  exact toStep.run_eq_uncapped MAX f 0 v (by simpa [fitsV] using h)

theorem toF_nestList_ok (c : Nat) : ∀ n f d, n + d ≤ c → n + 1 ≤ f →
    toF (some c) f d (nestList n) = R.ok (nestArr n)
  | 0, f + 1, d, _, _ => rfl
  | n + 1, f + 1, d, h1, h2 => by
    have hd : ¬ c ≤ d := by omega
    simp only [toF, nestList, needsV, chk, decide_eq_false hd, Bool.and_false, Bool.false_eq_true,
      ite_false, toLevel, listLoop, toF_nestList_ok c n f (d + 1) (by omega) (by omega), R.bind_ok]
    rfl

theorem toF_nestList_err (c : Nat) : ∀ n f d, c < n + d → d ≤ c → c + 1 ≤ f + d →
    toF (some c) f d (nestList n) = R.err nestErr
  | 0, _, d, h1, h2, _ => by omega
  | n + 1, 0, d, _, h2, h3 => by omega
  | n + 1, f + 1, d, h1, h2, h3 => by
    by_cases hd : c ≤ d
    · simp only [toF, nestList, needsV, chk, decide_eq_true hd, Bool.and_self, ite_true]
    · simp only [toF, nestList, needsV, chk, decide_eq_false hd, Bool.and_false, Bool.false_eq_true,
        ite_false, toLevel, listLoop, toF_nestList_err c n f (d + 1) (by omega) (by omega) (by omega), R.bind_err]
      rfl

/-- **Rust test `value_to_js_respects_the_cap`**: `MAX` nested lists convert and round-trip ... -/
theorem nestList_rt (n : Nat) (h : n ≤ MAX) : rt (nestList n) := by
  unfold rt toJs
  rw [toF_nestList_ok MAX n (MAX + 1) 0 (by omega) (by omega)]
  exact fromJs_nestArr_ok H0 n h

/-- ... and `MAX + 1` are exactly `udf_nesting_error()`. -/
theorem toJs_nestList_past (n : Nat) (h : MAX < n) : toJs (nestList n) = .error nestErr := by
  unfold toJs; rw [toF_nestList_err MAX n (MAX + 1) 0 (by omega) (by omega) (by omega)]; rfl

/-! ## The two caps count a VecF32 leaf differently -/

/-- `n` single-element lists around `v`, and `n` single-element arrays around `j`. -/
def wrapL : Nat → RV → RV
  | 0, v => v
  | n + 1, v => .list [wrapL n v]
def wrapA : Nat → JsV → JsV
  | 0, j => j
  | n + 1, j => .arr [wrapA n j] []

def vecJs : JsV := .arr [] [("__falkor_type".toList, .str "vecf32")]

theorem toF_wrap_vec (c : Nat) : ∀ n f d, n + d ≤ c → n + 1 ≤ f →
    toF (some c) f d (wrapL n (.vecf32 [])) = R.ok (wrapA n vecJs)
  | 0, f + 1, d, _, _ => rfl
  | n + 1, f + 1, d, h1, h2 => by
    have hd : ¬ c ≤ d := by omega
    simp only [toF, wrapL, needsV, chk, decide_eq_false hd, Bool.and_false, Bool.false_eq_true,
      ite_false, toLevel, listLoop, toF_wrap_vec c n f (d + 1) (by omega) (by omega), R.bind_ok]
    rfl

theorem fits_wrap_vec (H : Heap) : ∀ n, (fromStep H).within n (wrapA n vecJs) = false
  | 0 => rfl
  | n + 1 => by
    simp [Step.within, wrapA, deref_arr, classify_arr_nil, kidsS, fits_wrap_vec H n]

/-- FINDING (model; not a crash): `value_to_js` checks only List / Map against the cap, but
    `js_to_value` also counts the array that carries a VecF32 as a level. So `MAX` lists around a vecf32 — a value inside the cap — convert into JS, and the
    unchanged JS value does not convert back: an identity UDF fails on it with
    `udf_nesting_error()`. -/
theorem cap_asymmetry_vecf32 :
    toJs (wrapL MAX (.vecf32 [])) = .ok (wrapA MAX vecJs) ∧
    fitsV (wrapL MAX (.vecf32 [])) = true ∧
    ∃ e, fromJs H0 (wrapA MAX vecJs) = .error e := by
  refine ⟨?_, ?_, fromJs_past_cap H0 _ (by
    unfold fitsJs; exact fits_wrap_vec H0 MAX)⟩
  · unfold toJs; rw [toF_wrap_vec MAX MAX (MAX + 1) 0 (by omega) (by omega)]; rfl
  · unfold fitsV
    suffices ∀ r n, n ≤ r → toStep.within r (wrapL n (.vecf32 [])) = true from this MAX MAX (Nat.le_refl _)
    intro r
    induction r with
    | zero => intro n h; rw [Nat.le_zero.1 h]; rfl
    | succ r ih =>
      intro n h
      cases n with
      | zero => rfl
      | succ n => simp [Step.within, wrapL, kidsV, ih n (by omega)]

end AlgoUdf.Marshal
