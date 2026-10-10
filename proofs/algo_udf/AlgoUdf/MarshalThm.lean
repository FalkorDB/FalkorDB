import AlgoUdf.Marshal
/-! # UDF marshalling: round trip per kind, and exactly where it breaks

`rt v` := `toJs v >>= fromJs = .ok v` (`js_to_value(value_to_js(v)) == v`).
-/
namespace AlgoUdf.Marshal

def rt (v : RV) : Prop := (toJs v >>= fromJs) = .ok v

theorem null_rt : rt .null := rfl
theorem bool_rt (b : Bool) : rt (.bool b) := rfl
theorem str_rt (s : String) : rt (.str s) := rfl

/-- i64 round trips: |i| < 2^53 as a Number, otherwise as a BigInt. -/
theorem int_roundtrip (i : Int) (h : i.natAbs ≤ i64max.toNat) : rt (.int i) := by
  unfold rt
  by_cases hs : i.natAbs < two53.toNat
  · simp only [toJs, hs, ite_true, bind, Except.bind]
    unfold ofInt
    by_cases h0 : i = 0
    · subst h0; rfl
    · simp only [h0, ite_false, fromJs, F64.integral, F64.small, F64.toInt, F64.negZero, hs, decide_true,
        Bool.and_self, Bool.not_false, ite_true]
  · simp only [toJs, hs, ite_false, bind, Except.bind, fromJs, h, ite_true]

/-- Float round trips iff it is not an integral value below 2^53 in magnitude other
than -0.0 (#3074). -/
theorem float_rt_iff (f : F64) :
    rt (.float f) ↔ ¬ (f.integral = true ∧ f.small = true ∧ f.negZero = false) := by
  unfold rt
  simp only [toJs, bind, Except.bind, fromJs]
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

/-- HISTORICAL (`bug_udf_negative_zero_becomes_integer_zero`, fixed by #3074 `7a81c83b0`):
the number arm used to turn -0.0 into Int 0. -/
theorem pre3074_neg_zero_not_preserved : pre3074_num (.zero true) = .int 0 := rfl

/-- #3074 changed the number arm at -0.0 only. -/
theorem num_eq_pre3074_off_negZero (f : F64) (h : f.negZero = false) :
    (fromJs (.num f)) = .ok (pre3074_num f) := by
  simp [fromJs, pre3074_num, h]

/-- Same mechanism, shared with C (live: both return `2` for `id(2.0)`): 2.0 → Int 2. -/
theorem integral_float_becomes_int : (toJs (.float (.integ 2)) >>= fromJs) = .ok (.int 2) := rfl

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
  simp only [toJs, h1, ite_false, bind, Except.bind, mkDate, h, ite_true, fromJs]
  unfold ofInt
  by_cases h0 : ts * 1000 = 0
  · have : ts = 0 := by omega
    subst this; rfl
  · simp only [h0, ite_false]
    show fromObjWith _ _ _ = _
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
  show fromJs (.arr (xs.map .num) [("__falkor_type".toList, .str "vecf32")]) = _
  simp only [fromJs, get, List.find?]
  exact vecf32_fromVec xs

/-- Kept for the earlier name: finite vectors round trip (now a special case). -/
theorem vecf32_finite_rt (xs : List F64) (_h : ∀ x ∈ xs, x.finite = true) :
    fromVec (xs.map .num) = .ok (.vecf32 xs) := vecf32_fromVec xs

/-- `vecf32([1e39])` holds +inf and now comes back unchanged. -/
theorem vecf32_inf_rt : rt (.vecf32 [.inf false]) := vecf32_rt _

/-- HISTORICAL (`bug_udf_vecf32_inf_rejected`, fixed by #3074 `7a81c83b0`): the element
loop refused a non-finite element. C returns `[inf]`. -/
theorem pre3074_vecf32_inf_not_roundtrip :
    (pre3074_fromVec [.num (.inf false)]).toOption = none := rfl

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
(`agrees_udf_reserved_keys_escaped`). -/
theorem map_never_marked (kv : List (List Char × RV)) (ps : List (List Char × JsV))
    (h : toJsMap kv = .ok ps) : get ps "__falkor_type".toList = none := by
  induction kv generalizing ps with
  | nil => simp [toJsMap] at h; subst h; rfl
  | cons p kvs ih =>
    obtain ⟨k, v⟩ := p
    simp only [toJsMap] at h
    cases hv : toJs v with
    | error e => rw [hv] at h; cases h
    | ok j =>
      cases hr : toJsMap kvs with
      | error e => rw [hv, hr] at h; cases h
      | ok ps' =>
        rw [hv, hr] at h; simp only [bind, Except.bind] at h
        cases h
        split
        · exact ih ps' hr
        · simp only [get, List.find?]
          have : (esc k == "__falkor_type".toList) = false := by
            simpa using fun h => esc_ne_marker k h
          simp only [this]
          exact ih ps' hr

/-- **#3074: a plain object is a map whatever its keys** — a `constructor` (or any
other) own property can no longer make it a Date or a RegExp. -/
theorem plain_obj_is_map (ps : List (List Char × JsV)) (props : Except String (List (List Char × RV))) :
    fromObjWith ps .plain props = (props >>= fun kv => pure (.map kv)) := rfl

/-- `{constructor: {name: 'Date'}}` round trips as a map (C returns the map too). -/
theorem constructor_key_roundtrips :
    rt (.map [("constructor".toList, .map [("name".toList, .str "Date")])]) := rfl

/-- ... and so does `{constructor: {name: 'RegExp'}}`. -/
theorem constructor_regexp_roundtrips :
    rt (.map [("constructor".toList, .map [("name".toList, .str "RegExp")])]) := rfl

/-- HISTORICAL (`bug_udf_constructor_key_mistaken_for_date`, fixed by #3074 `7a81c83b0`):
the constructor-name test failed a map whose own `constructor.name` is 'Date' ... -/
theorem pre3074_constructor_key_breaks_roundtrip :
    (pre3074_fromObjWith [("constructor".toList, .obj [("name".toList, .str "Date")] .plain)] .plain
      (.ok [])).toOption = none := rfl

/-- ... and turned one with 'RegExp' into the string "[object Object]". -/
theorem pre3074_constructor_regexp_becomes_string :
    (pre3074_fromObjWith [("constructor".toList, .obj [("name".toList, .str "RegExp")] .plain)] .plain
      (.ok [])).toOption = some (.str "[object Object]") := rfl

/-- Shared with C (live: both return `{}`): a `__proto__` key is lost. -/
theorem proto_key_dropped :
    (toJs (.map [("__proto__".toList, .int 1)]) >>= fromJs).toOption = some (.map []) := rfl

/-! ## JS → Rust: what is accepted that should not be -/

/-- BUG (confirmed, `bug_udf_forged_edge_marker_panics`, `..._forged_node_creates_dangling_edge`):
any JS object carrying the marker becomes an entity reference, with no liveness check. -/
theorem forged_node_marker (id : Nat) :
    fromJs (.obj [("__falkor_type".toList, .str "node"), ("__falkor_node_id".toList, .num (ofInt id))] .plain)
      = .ok (.node id) := by cases id <;> rfl

theorem forged_edge_marker :
    fromJs (.obj [("__falkor_type".toList, .str "edge"), ("__falkor_edge_id".toList, .num (.integ 1000))] .plain)
      = .ok (.rel 1000) := rfl

/-- A JS object key starting `__falkor_` (not `__falkor_esc_`) is silently dropped. -/
theorem internal_keys_dropped :
    fromJs (.obj [("__falkor_x".toList, .num (.integ 1))] .plain) = .ok (.map []) := rfl

theorem symbol_rejected : fromJs .sym = .error "Symbol values are not supported" := rfl
theorem bigint_out_of_range : (fromJs (.big (2 ^ 63))).toOption = none := rfl

/-- Array-index keys are enumerated first (shared with C: `{b:1, `1`:2}` → `{1:2, b:1}`). -/
theorem index_keys_first :
    jsKeys [("b".toList, .null), ("1".toList, .null)] = ["1".toList, "b".toList] := by decide

end AlgoUdf.Marshal
