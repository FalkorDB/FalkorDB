import FalkorValueMath.AggScalar
import FalkorValueMath.Registry
import FalkorValueMath.Int64
import FalkorValueMath.Value
import FalkorValueMath.CArith
import FalkorValueMath.TypeCheck
import FalkorValueMath.MathFns
import FalkorValueMath.Conversion
import FalkorValueMath.SelfCompare
import FalkorValueMath.Misc
import FalkorValueMath.Trig
import FalkorValueMath.Procedures
import FalkorValueMath.ConvReg
import FalkorValueMath.FloatToString
import FalkorValueMath.IndexDdl
import FalkorValueMath.Nesting
/-!
# Value arithmetic, math/conversion functions and function type checking (FalkorDB-rs)

**Re-synced to main 256ab4d3f** (from 43936bab9). Behaviour changes in scope: `toString(Float)`
(`float_to_string`, #3609, `FloatToString.lean`, already modelled), `List(Any)` skip in
`value_of_type` (#3861, `vot_list_any`). `apply_pow` now takes `&Value`, `index_ddl.rs` takes
`IndexType`/`EntityType` by value and calls `Runtime::upgrade_to_write` (the wrapper over
`write_escalation().upgrade_to_write`), `debug_assert_eq!(args, [])` and `map_or_default`
refactors in math/trig/mod.rs: no behaviour change, models stand. Every file:line cite in
this header and the module headers re-pinned to HEAD; every `value.rs` NOT COVERED/MODELLED row
(trait declarations, constructors, JSON wrappers, chrono accessors) now points at the proof
that covers it in `proofs/value_order` or `proofs/functions_temporal`.

Lean 4 (core only) models of `graph/src/runtime/value.rs` (arithmetic, formatting,
memory accounting, attribute access, self-comparison, v19 codec),
`functions/math.rs`, `functions/conversion.rs` and `functions/mod.rs` (arity, argument
type checking, registry). Zero `sorry`/`admit`/`axiom`; 384 theorems/examples.
Coverage per Rust fn: `COVERAGE.tsv` (all rows PROVEN).
Rust repros: `graph/tests/lean_value_math.rs`
(`cargo test -p graph --test lean_value_math`; tests assert today's behaviour, C output in
comments). Live: Rust module on port 18410 vs C `bin/macos-arm64v8-release/falkordb.so`
on 18411. Compare/eq/hash laws are in `proofs/value_order` (reused, not redone).

## Lean ↔ Rust

| Lean (file) | Rust |
| --- | --- |
| `wrap`, `wAdd/wSub/wMul/wDiv/wRem` (Int64) | `wrapping_*` in `Add/Sub/Mul/Div/Rem for Value` value.rs |
| `roundF`, `satTrunc`, `cDivExact` (Int64) | `i as f64`; C `SIValue_Divide` (src/value.c, master) |
| `V`, `add`, `addSlow`, `sub`, `mul`, `div`, `rem`, `omInsert/omExtend` (Value) | value.rs; `OrderMap::insert/extend` ordermap.rs |
| `cAdd`, `cMul`, `siMultiply`, `cNumeric` (CArith) | C `SIValue_Add/_Multiply` + `AR_ADD/AR_MUL` type checks (`cAddType`/`cMulType`); `siMultiply` is written from `value.c`, independent of `mul` |
| `Ty`, `valueOfType`, `Accepts`, `validate`, `validateArgsType`, `isCompatibleWith`, `canReturnBoolean`, `unionDisplay`, `regGet`, `regIsAggregate` (TypeCheck) | value.rs; mod.rs |
| `FC`, `mAbs…mSqrt`, `coalesce`, `uuidLayout`, `applyPow` (+ `pre2904_applyPow`), `cAbs` (MathFns) | math.rs; C `AR_ABS` |
| `parseI64`, `f64Grammar`, `cStrtoll`, `rustToIntegerStr`, `toInteger/toFloat/toStr/toBooleanV`, list forms, `isEmpty` (Conversion) | conversion.rs; C `AR_TOINTEGER/AR_TOFLOAT` (numeric_funcs.c) |
| `splitOnce`, `digitCount`, `sciFixed`, `floatToString`, `toStr_float` (FloatToString) | conversion.rs `float_to_string` (PR #3609; std formatters/`parse` = `FmtOps` parameters) |
| `cmpV`, `cmpList`, `cmpMap`, `chunked`, `firstDiff`, `mayFail`, `isNeverEqual` (SelfCompare) | value.rs |
| `fmtDuration`, `HV/heapSize/amortized`, `getAttr`, `pointComponent`, `durationComponentPre`, `encode`/`pre3339_decode`, `dedupStep`, `jsonFloat` (Misc) | value.rs/:2177 (decode before #3339), :2072-2079, :1700 |
| `MAX_VALUE_NESTING`, `valueNestingError`, `decodeValue`, `decode`, `validateProperty`/`validateListProperty`/`isValidProperty` (Nesting) | value.rs; pending.rs |

## Main theorems

Int64: `div_rem_identity` ((a/b)*b + a%b = a for i64, wrapping), `wDiv_exact`,
`tdiv_inI64`, `wRem_exact`, `rem_bounds`, `wAdd_assoc/comm`, `c_rust_div_one_agree`.
Value: null propagation for all five operators, `mul_ok_iff`/`div_ok_iff`/`rem_ok_iff`
(exact error conditions), `lookup_omExtend` (map `+` is a right-biased merge),
`nodup_omExtend` (no duplicate keys), `keys_omInsert`, list append/prepend laws,
temporal `+ Duration` symmetric; `Time ± Duration` keeps only the time of day (`add_time_dur_in_day`, `sub_time_dur_in_day`,
`time_sub_add_dur`, `time_add_whole_days`, `time_add_dur_mod`); `apply_pow` rejects a non-numeric, non-null operand, base first
(`pow_nonnumeric_base_errors`, `pow_nonnumeric_exp_errors`, `pow_bool_errors`, `pow_null`, `pow_numeric_unchanged`,
`pow_fn_in_domain`; `pre2904_applyPow` in MathFns is the earlier model, kept as history); the scalar `percentile` step stores
the percentile only while nothing is collected (`percentile_first_row`, `percentile_keeps`). CArith: `mul_agree_on_numeric`, `add_agree_str_scalar`,
`add_agree_list` (for a declared-type operand), `add_agree_int_int`, `add_agree_float` (Rust = C on the shared domain).
TypeCheck: **`valueOfType_none_iff`** (`value_of_type` returns None iff the clean
acceptance spec holds), **`validateArgsType_go_none`**, `validate_fixed_iff`,
`numeric_body_cases`, `canReturnBoolean_of_accepts`, `isCompatibleWith_*`,
`unionDisplay_*`, `regGet_builtin_ci`, `regGet_udf_case_sensitive`. MathFns: **`unary_no_panic`** (the nine `unreachable!()`
arms are dead after validation), `sqrt_float_eq_ieee`, `sqrt_nan_iff`,
`abs_agree_except`, `coalesce_spec`, `uuid_shape`. Conversion:
`parseI64_sub_cStrtoll`, `toInteger_int_literals_agree`, **`toInteger_int_literal_agrees_c`**, `toBooleanList_eq`,
`toFloatList_eq`, `isEmpty_no_panic`. SelfCompare: **`mayFail_sound`** (the
`may_fail_self_match` pre-filter is sound), `isNeverEqual_iff`,
**`chunked_eq_firstDiff`** (16-lane block scan = first-difference scan). Misc:
`pre3339_decode_encode` (v19 codec roundtrip, old decoder), `readVec_vecBytes`,
**`heap_le_amortized`**, `shares_le`, `fmtDuration_fields`.

## Value nesting (Nesting.lean)

`decodeValue_refuses_past_cap` (a call past `MAX_VALUE_NESTING` reads nothing and returns the
nesting error), **`decode_reach`** (whatever the input, no `decode_value` call is made deeper
than `MAX + 1`), **`decodeValue_agrees`** (otherwise exactly the decoder without a cap: `Ok` iff
it succeeds with a value `within` the cap), `decode_encode` /
`decodeValue_encode_deep` (on encoded storable values: round-trip iff `within`, else the
nesting error), `within_of_depth` / `not_within_of_depth` (depth ≤ MAX always loads, ≥ MAX+2
never), **`validateProperty_depth`** (every accepted property has depth ≤ MAX), **`stored_roundtrip`**
(every accepted property — vectors < 2³² — saves and loads back), `validateProperty_nest`.
`decode_nest_deep` is a current theorem: the decoder refuses depth past the cap.
Note (`within_nest_empty`): the decoder's cut is one level more lenient than the writers' —
`MAX` arrays around `[]` (depth MAX+1) decode; harmless (still bounded, never stored).
`GRAPH.BULKINSERT`'s bound (bulk_insert.rs) is in proofs/redis_layer (`Bulk`).

## Confirmed bugs (Rust repro + live Rust vs C)

No open confirmed bugs. Fixed ones are in this file's git history. The latent ones are under Suspicions.

## Divergences from C (live; Rust often the better answer — decide per case)

- `MIN / -1`: Rust errors (`ValueMath.div_min_neg_one`), C MAX; `(2^53+1)/1` Rust exact, C 2^53 (C divides through double).
- `abs(MIN)` Rust error, C MIN; `abs(-0.0)` Rust 0, C -0.
- `+` typing: `'a'+date(..)`, `'a'+{b:1}`, `'a'+localtime(..)` Rust error, C concatenates;
  `true+1`, `1.5+true`, `duration+1`, `date+1` Rust error, C Float; `null*'a'`, `'a'/null`
  Rust null, C type error; error texts differ for `-` (C lists accepted types).
- `'a'+(0.0/0)` 'aNaN' vs 'anan'; `toJSON(NaN)` 'null' vs 'nan'; `toJSON(0.1+0.2)`
  '0.30000000000000004' vs '0.3'; `toJSON(1e300)` 301 digits vs '1e+300'; point JSON
  `"height": null` vs `"height":null`; map JSON `{"a":[1,…]}` vs `{"a": [1, …]}`; C
  `toJSON('a"b')` does not escape (C bug); C toJSON of temporals/vecf32 errors.
- `toFloat('0.1')` 0.1 vs C 0.100000001490116 (C uses `strtof`, float32 — C bug);
  `toFloat('16777217')` exact vs 16777216; `toFloat('1e39')` 1e39 vs null.
- `toInteger(NaN)` null vs 0, `toInteger(inf)` null vs MAX; `toStringList([1.5])`
  '1.5' vs '1.500000' (PR #3609: `toString(1.0)` '1.0', `toString(1e20)` '1.0E20'); `toFloatList([' 3'])`/`toIntegerList([' 3'])` null vs 3.
- `p.LATITUDE` Rust 1.5, C null (Rust component names are ASCII case-insensitive).
- `toString(duration('PT0S'))` 'PT0S' vs C 'P' (C bug).
- Also seen (known):
  `toString(1.0)` '1', `toInteger('1e3')`, leading whitespace (functions_temporal).

## Suspicions / latent

- `Functions::is_aggregate(name)` (mod.rs) does not lowercase; no caller today
  (`isAggregate_case_sensitive`).
- `heap_size`/`amortized_heap_size` charge a `Relationship` 24 bytes it does not own.
- `Encode<19>` writes Map/Node/Rel/Path as NULL after only a `debug_assert!`
  (`encode_map_is_null`); `Decode<19>` `with_capacity(len)` trusts the length.
- `add_var_len` stores the lowered name, `add` the original (error text casing).

## Modelling gaps

f64 is abstract (`FOps`, `FC` classes, `FloatCmp` laws); libm `exp/ln/log10/floor/ceil/
round` are parameters; `f64::from_str` value (only its grammar and integer-literal
rounding are modelled); chrono calendar accessors and formatting, `json_escape`,
`rand`, the QuickJS UDF bridge and registry locking are trusted (NOT COVERED rows).
Temporal helpers are parameters here and proved in proofs/functions_temporal.

## Wave 5 (`Registry.lean`, `AggScalar.lean`)

functions/mod.rs (dispatch, `GraphFn` constructors, `Functions::add*`, `set_*_fn`, `iter`,
`init_functions`, the UDF registry and version counter), math.rs `register`/`e`/`rand` and
every scalar aggregation body: all PROVEN. QuickJS bridge = argument; `to_lowercase`,
f64 ops and the RNG are structures (`Lower`, `FM`, `Rng`). Highlights: `add_spec`,
`registerAll_ok` (init never panics iff lowered names are distinct), `initFunctions_once`,
`setStructFn_spec`, `canReturnEntity_sound/complete`, `version_mono`,
`udf_uninit_noop`, `collect_fold`, `count_fold`, `sum_fold`, `avg_count`,
`stdev_collects`, `percentile_collects`, `finalize_divisors`, `randFn_range`.
## Union display and UDF names

- `dbms.functions` union spelling: `type_to_dbms_string` flattens
  (`collect_dbms_types`, procedures.rs), sorts by `DBMS_TYPE_ORDER` and dedups.
  `union_order_is_c` (the member
  list is strictly increasing in C's type order with exactly the collected members),
  `canon_eq_of`, `union_perm_invariant` (declaration order irrelevant), and the concrete rows
  `union_point_null` (`Null or Point`), `union_tofloat` (`String, Integer, Float, or Null`),
  `union_nested`, `union_any`.
- UDF names are case-sensitive: `register_udf`/`unregister_udf`
  (mod.rs/:1142) and the `Functions::get` UDF fallback (:1048) key by the exact name.
  `Registry.udf_case_sensitive` and `TypeCheck.regGet_udf_case_sensitive`; built-ins stay
  case-insensitive (`regGet_builtin_ci`). `dbms_functions` drops a UDF whose lowercase name equals a
  built-in's (`keepUdf`).

## Wave 6 (`Trig.lean`, `Procedures.lean`, `ConvReg.lean`, `IndexDdl.lean`)

trig.rs, procedures.rs, path.rs, entity_type.rs, index_ddl.rs and the rest of conversion.rs:
all PROVEN (libm/`f64` parse+Display/`to_json_string`/graph index store = parameters).
Highlights: `unary_rust_eq_c` + `atan2_spec` (every trig function is C's expression tree,
null/type/arity handling identical; live 25 calls identical), `cot/haversin/degrees/radians_formula`;
`yields_*` (every procedure batch has one column per declared YIELD, rectangular),
`dbmsProcedures_sorted`, `mergePicks_eq` + `dbmsFunctions_merge_sorted` (the hand-written
built-in/UDF merge is `List.merge`: sorted, a permutation), `tyValue_eq` (interning
invisible), `any_string` (= C), `typesOfFields_spec`, `metaStats_labels`, `path_split`,
`create_ok`/`ro_rejects`/`emit_counts` (DDL stats/effects bookkeeping).
Live (Rust 18920 vs C 18921, empty + populated graph): `db.labels`, `db.relationshipTypes`,
`db.propertyKeys`, `db.meta.stats` (also after deletes), `dbms.procedures` identical.
Divergences: `db.indexes`/`db.constraints` same rows, different row order; `db.indexes`
`info` is only `{fields:[{name}]}` (C: full RediSearch info, and edge indexes list
`range:_src_id`/`range:_dest_id`); `reducible` differs
for hasLabels/length/nodes/relationships, internal-op rows differ; `createNodeIndex`
C form `('L','p')` rejected (known stub, test_effects_shapes.py:670) and alone under
GRAPH.QUERY it fails with the misleading "graph.RO_QUERY is to be executed only on read-only
queries" (procedure_call.rs: a lone write procedure is planned as read-only);
`db.idx.fulltext.drop` of a missing index errors (C: no-op); `queryNodes(1, 2)` errors
(C: empty).

## Re-pinned to 5e475e6c9 (#3436)
Re-pin only. `procedures.rs` (`runtime.g.borrow()` -> `runtime.graph()`, -3 lines) and
`index_ddl.rs` rows shifted; `value.rs`, `math.rs`, `conversion.rs` are unchanged since b84473a7f.
-/
