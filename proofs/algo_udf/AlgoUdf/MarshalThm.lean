import AlgoUdf.Marshal
/-! # UDF marshalling: round trip per kind, and exactly where it breaks

`rt v` := `toJs v >>= fromJs H0 = .ok v` (`js_to_value(value_to_js(v)) == v`).
-/
namespace AlgoUdf.Marshal

def rt (v : RV) : Prop := (toJs v >>= fromJs H0) = .ok v

/-! ## One-level unfoldings (scalars never reach the depth check) -/

theorem toJs_int (i : Int) : toJs (.int i) = .ok (if i.natAbs < two53.toNat then .num (ofInt i) else .big i) := rfl
theorem toJs_float (f : F64) : toJs (.float f) = .ok (.num f) := rfl
theorem toJs_eq (v : RV) : toJs v = Rget (toF (some MAX) (MAX + 1) 0 v) := rfl
theorem fromJs_eq (H : Heap) (v : JsV) : fromJs H v = Rget (fromF H (some MAX) (MAX + 1) 0 v) := rfl

theorem toJs_datetime (ts : Int) : toJs (.datetime ts) =
    if (ts * 1000).natAbs > i64max.toNat then .error "i64 overflow (debug panic)"
    else .ok (.obj [("__falkor_temporal_type".toList, .str "datetime")] (.date (mkDate (ts * 1000)))) := by
  rw [toJs_eq]
  by_cases h : (ts * 1000).natAbs > i64max.toNat <;> simp [toF, needsV, toLevel, h, Rget, R.err, R.ok]

/-- An object with no `__falkor_type` marker and a Date / RegExp class is converted by `fromObjWith`. -/
theorem classify_kind (ps : List (List Char × JsV)) (k : JsKind) (ht : tag ps = none) (hk : k ≠ .plain) :
    classify (.obj ps k) = .done (fromObjWith ps k (.ok [])) := by
  cases k with
  | plain => exact absurd rfl hk
  | _ => simp [classify, ht]

theorem fromJs_kind (H : Heap) (ps : List (List Char × JsV)) (k : JsKind) (ht : tag ps = none) (hk : k ≠ .plain) :
    fromJs H (.obj ps k) = fromObjWith ps k (.ok []) := by
  rw [fromJs_eq]; simp only [fromF, fromLevel, deref, classify_kind ps k ht hk, runShape, Rget]

theorem toJs_vecf32 (xs : List F64) : toJs (.vecf32 xs) = .ok (.arr (xs.map .num) [("__falkor_type".toList, .str "vecf32")]) := rfl

theorem fromJs_num (H : Heap) (f : F64) :
    fromJs H (.num f) = .ok (if f.integral && f.small && !f.negZero then .int f.toInt else .float f) := rfl
theorem fromJs_big (H : Heap) (i : Int) :
    fromJs H (.big i) = if i.natAbs ≤ i64max.toNat then .ok (.int i) else .error "BigInt out of i64 range" := by
  rw [fromJs_eq]; by_cases h : i.natAbs ≤ i64max.toNat <;> simp [fromF, fromLevel, deref, classify, runShape, h, Rget]

theorem null_rt : rt .null := rfl
theorem bool_rt (b : Bool) : rt (.bool b) := rfl
theorem str_rt (s : String) : rt (.str s) := rfl

/-- i64 round trips: |i| < 2^53 as a Number, otherwise as a BigInt. -/
theorem int_roundtrip (i : Int) (h : i.natAbs ≤ i64max.toNat) : rt (.int i) := by
  unfold rt
  rw [toJs_int]
  by_cases hs : i.natAbs < two53.toNat
  · simp only [hs, ite_true, bind, Except.bind, fromJs_num]
    unfold ofInt
    by_cases h0 : i = 0
    · subst h0; rfl
    · simp only [h0, ite_false, F64.integral, F64.small, F64.toInt, F64.negZero, hs, decide_true,
        Bool.and_self, Bool.not_false, ite_true]
  · simp only [hs, ite_false, bind, Except.bind, fromJs_big, h, ite_true]

/-- Float round trips iff it is not an integral value below 2^53 in magnitude other
than -0.0 (#3074). -/
theorem float_rt_iff (f : F64) :
    rt (.float f) ↔ ¬ (f.integral = true ∧ f.small = true ∧ f.negZero = false) := by
  unfold rt
  simp only [toJs_float, bind, Except.bind, fromJs_num]
  by_cases h : f.integral = true ∧ f.small = true ∧ f.negZero = false
  · obtain ⟨h1, h2, h3⟩ := h
    simp only [h1, h2, h3, Bool.and_self, Bool.not_false, ite_true]
    refine ⟨fun e => ?_, fun hn => (hn ⟨trivial, trivial, trivial⟩).elim⟩
    cases e
  · have : (f.integral && f.small && !f.negZero) = false := by
      cases hi : f.integral <;> cases hs : f.small <;> cases hn : f.negZero <;> simp_all
    simp [this, h]

/-- **#3074: -0.0 round trips as a Float** (as in C). -/
theorem neg_zero_rt : rt (.float (.zero true)) := rfl

/-- Same mechanism, shared with C (live: both return `2` for `id(2.0)`): 2.0 → Int 2. -/
theorem integral_float_becomes_int : (toJs (.float (.integ 2)) >>= fromJs H0) = .ok (.int 2) := rfl

theorem node_rt (id : Nat) : rt (.node id) := by
  unfold rt; cases id <;> rfl
theorem rel_rt (id : Nat) : rt (.rel id) := by
  unfold rt; cases id <;> rfl
theorem point_rt (a b : F64) : rt (.point a b) := rfl

/-- Datetime round trips inside the JS Date range (|ts| ≤ 8.64e12 s). -/
theorem datetime_rt (ts : Int) (h : (ts * 1000).natAbs ≤ jsDateMax.toNat) : rt (.datetime ts) := by
  unfold rt
  have h1 : ¬ (ts * 1000).natAbs > i64max.toNat := by
    simp only [i64max, jsDateMax] at *; omega
  rw [toJs_datetime]
  simp only [h1, ite_false, bind, Except.bind, mkDate, h, ite_true]
  unfold ofInt
  by_cases h0 : ts * 1000 = 0
  · have : ts = 0 := by omega
    subst this; rfl
  · simp only [h0, ite_false]
    rw [fromJs_kind H0 _ _ rfl (by intro h; cases h)]
    simp only [fromObjWith, get, List.find?]
    simp [F64.finite, F64.toInt]

/-- Time and Duration are rejected on the way in (C rejects them too). -/
theorem time_rejected : (toJs .time).toOption = none := rfl

/-! ## VecF32 -/

/-- The vecf32 element loop accepts every number (#3074: inf / NaN included). -/
theorem vecf32_fromVec (xs : List F64) : fromVec (xs.map .num) = .ok (.vecf32 xs) := by
  induction xs with
  | nil => rfl
  | cons x t ih =>
    simp only [List.map, fromVec]
    rw [ih]
    rfl

/-- **#3074: every vecf32 round trips**, non-finite elements included (as in C). -/
theorem vecf32_rt (xs : List F64) : rt (.vecf32 xs) := by
  unfold rt
  rw [toJs_vecf32]
  show fromJs H0 (.arr (xs.map .num) [("__falkor_type".toList, .str "vecf32")]) = _
  have ht : tag [("__falkor_type".toList, JsV.str "vecf32")] = some "vecf32" := rfl
  rw [fromJs_eq]
  simp only [fromF, fromLevel, deref, classify, ht, ite_true, runShape, chk, MAX]
  rw [vecf32_fromVec xs]; rfl

/-- Kept for the earlier name: finite vectors round trip (now a special case). -/
theorem vecf32_finite_rt (xs : List F64) (_h : ∀ x ∈ xs, x.finite = true) :
    fromVec (xs.map .num) = .ok (.vecf32 xs) := vecf32_fromVec xs

/-- `vecf32([1e39])` holds +inf and now comes back unchanged. -/
theorem vecf32_inf_rt : rt (.vecf32 [.inf false]) := vecf32_rt _

/-! ## Maps -/

theorem escPre_prefix (k : List Char) : escPre.isPrefixOf (escPre ++ k) = true := by
  simp [List.isPrefixOf_iff_prefix]

theorem pre_of_escPre : pre.isPrefixOf escPre = true := by decide

theorem unesc_esc (k : List Char) : unesc (esc k) = k := by
  unfold esc
  split
  · simp only [unesc, escPre_prefix, ite_true]
    show List.drop escPre.length (escPre ++ k) = k
    simp
  · rename_i h
    unfold unesc
    split
    · rename_i h2
      exfalso; apply h
      rw [List.isPrefixOf_iff_prefix] at *
      exact List.IsPrefix.trans (List.isPrefixOf_iff_prefix.mp pre_of_escPre) h2
    · rfl

theorem keep_esc (k : List Char) : keep (esc k) = true := by
  unfold keep esc; split
  · simp [escPre_prefix]
  · rename_i h; simp [h]

/-- An escaped user key is never one of the marker keys. -/
theorem esc_ne_marker (k : List Char) : esc k ≠ "__falkor_type".toList := by
  unfold esc; split
  · rename_i hp
    intro h; have h1 := congrArg List.length h
    have h2 := (List.isPrefixOf_iff_prefix.mp hp).length_le
    have h3 : escPre.length = 13 := by decide
    have h4 : "__falkor_type".toList.length = 13 := by decide
    have h5 : pre.length = 9 := by decide
    simp only [List.length_append] at h1; omega
  · rename_i hp; intro h; subst h; exact hp (by decide)

/-- The Cypher-map direction can never produce a node/edge/point marker
(`agrees_udf_reserved_keys_escaped`): no own key of the object `value_to_js` builds for a map is
`__falkor_type`. -/
theorem map_never_marked (kv : List (List Char × RV)) (js : List JsV) :
    get (assembleObj kv js) "__falkor_type".toList = none := by
  induction kv generalizing js with
  | nil => cases js <;> rfl
  | cons p kvs ih =>
    obtain ⟨k, v⟩ := p
    cases js with
    | nil => rfl
    | cons j js =>
      simp only [assembleObj]
      split
      · exact ih js
      · simp only [get, List.find?]
        have : (esc k == "__falkor_type".toList) = false := by
          simpa using fun h => esc_ne_marker k h
        simp only [this]
        exact ih js

/-- **#3074: a plain object is a map whatever its keys** — a `constructor` (or any
other) own property can no longer make it a Date or a RegExp. -/
theorem plain_obj_is_map (ps : List (List Char × JsV)) (h1 : tag ps ≠ some "node") (h2 : tag ps ≠ some "edge")
    (h3 : tag ps ≠ some "path") (h4 : tag ps ≠ some "point") :
    classify (.obj ps .plain) = .map ps := by
  simp [classify, h1, h2, h3, h4]

/-- `{constructor: {name: 'Date'}}` round trips as a map (C returns the map too). -/
theorem constructor_key_roundtrips :
    rt (.map [("constructor".toList, .map [("name".toList, .str "Date")])]) := rfl

/-- ... and so does `{constructor: {name: 'RegExp'}}`. -/
theorem constructor_regexp_roundtrips :
    rt (.map [("constructor".toList, .map [("name".toList, .str "RegExp")])]) := rfl

/-- Shared with C (live: both return `{}`): a `__proto__` key is lost. -/
theorem proto_key_dropped :
    (toJs (.map [("__proto__".toList, .int 1)]) >>= fromJs H0).toOption = some (.map []) := rfl

/-! ## JS → Rust: what is accepted that should not be -/

/-- BUG (confirmed, `bug_udf_forged_edge_marker_panics`, `..._forged_node_creates_dangling_edge`):
any JS object carrying the marker becomes an entity reference, with no liveness check. -/
theorem forged_node_marker (id : Nat) :
    fromJs H0 (.obj [("__falkor_type".toList, .str "node"), ("__falkor_node_id".toList, .num (ofInt id))] .plain)
      = .ok (.node id) := by cases id <;> rfl

theorem forged_edge_marker :
    fromJs H0 (.obj [("__falkor_type".toList, .str "edge"), ("__falkor_edge_id".toList, .num (.integ 1000))] .plain)
      = .ok (.rel 1000) := rfl

/-- A JS object key starting `__falkor_` (not `__falkor_esc_`) is silently dropped. -/
theorem internal_keys_dropped :
    fromJs H0 (.obj [("__falkor_x".toList, .num (.integ 1))] .plain) = .ok (.map []) := rfl

theorem symbol_rejected : fromJs H0 .sym = .error "Symbol values are not supported" := rfl
theorem bigint_out_of_range : (fromJs H0 (.big (2 ^ 63))).toOption = none := rfl

/-- Array-index keys are enumerated first (shared with C: `{b:1, `1`:2}` → `{1:2, b:1}`). -/
theorem index_keys_first :
    jsKeys [("b".toList, .null), ("1".toList, .null)] = ["1".toList, "b".toList] := by decide

end AlgoUdf.Marshal
