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
import FalkorValueMath.IndexDdl
/-!
# Value arithmetic, math/conversion functions and function type checking (FalkorDB-rs)

Lean 4 (core only) models of `graph/src/runtime/value.rs` (arithmetic, formatting,
memory accounting, attribute access, self-comparison, v19 codec),
`functions/math.rs`, `functions/conversion.rs` and `functions/mod.rs` (arity, argument
type checking, registry). Zero `sorry`/`admit`/`axiom`; 184 theorems/examples.
Coverage per Rust fn: `COVERAGE.tsv` (72 PROVEN, 11 MODELLED, 46 NOT COVERED of 129).
Rust repros: `graph/tests/lean_value_math.rs`
(`cargo test -p graph --test lean_value_math`; tests assert today's behaviour, C output in
comments). Live: Rust module on port 18410 vs C `bin/macos-arm64v8-release/falkordb.so`
on 18411. Compare/eq/hash laws are in `proofs/value_order` (reused, not redone).

## Lean ↔ Rust

| Lean (file) | Rust |
| --- | --- |
| `wrap`, `wAdd/wSub/wMul/wDiv/wRem` (Int64) | `wrapping_*` in `Add/Sub/Mul/Div/Rem for Value` value.rs:915,1048,1097,1130,1158 |
| `roundF`, `satTrunc`, `cDivExact` (Int64) | `i as f64`; C `SIValue_Divide` (src/value.c, master) |
| `V`, `add`, `addSlow`, `sub`, `mul`, `div`, `rem`, `omInsert/omExtend` (Value) | value.rs:180, :910, :925, :1042, :1091, :1120, :1148; `OrderMap::insert/extend` ordermap.rs:76,218 |
| `cAdd`, `cMul` (CArith) | C `SIValue_Add/_Multiply` + `AR_ADD/AR_MUL` type checks |
| `Ty`, `valueOfType`, `Accepts`, `validate`, `validateArgsType`, `isCompatibleWith`, `canReturnBoolean`, `unionDisplay`, `regGet`, `regIsAggregate` (TypeCheck) | value.rs:1282; mod.rs:531, :787, :828, :590, :569, :610, :1048, :1079 |
| `FC`, `mAbs…mSqrt`, `coalesce`, `uuidLayout`, `applyPow`, `cAbs` (MathFns) | math.rs:46-345; C `AR_ABS` |
| `parseI64`, `f64Grammar`, `cStrtoll`, `rustToIntegerStr`, `toInteger/toFloat/toStr/toBooleanV`, list forms, `isEmpty` (Conversion) | conversion.rs:44-285; C `AR_TOINTEGER/AR_TOFLOAT` (numeric_funcs.c) |
| `cmpV`, `cmpList`, `cmpMap`, `chunked`, `firstDiff`, `mayFail`, `isNeverEqual` (SelfCompare) | value.rs:1224, :1399, :1472, :1817, :1821, :1554, :1531 |
| `fmtDuration`, `HV/heapSize/amortized`, `getAttr`, `pointComponent`, `durationComponentPre`, `encode/decode`, `dedupStep`, `jsonFloat` (Misc) | value.rs:296, :342, :388, :439, :458, :608, :1899/:1985, :1887, :1591 |

## Main theorems

Int64: `div_rem_identity` ((a/b)*b + a%b = a for i64, wrapping), `wDiv_exact`,
`tdiv_inI64`, `wRem_exact`, `rem_bounds`, `wAdd_assoc/comm`, `c_rust_div_one_agree`.
Value: null propagation for all five operators, `mul_ok_iff`/`div_ok_iff`/`rem_ok_iff`
(exact error conditions), `lookup_omExtend` (map `+` is a right-biased merge),
`nodup_omExtend` (no duplicate keys), `keys_omInsert`, list append/prepend laws,
temporal `+ Duration` symmetric. CArith: `mul_agree_on_numeric`, `add_agree_str_scalar`,
`add_agree_list`, `add_agree_int_int` (Rust = C on the shared domain).
TypeCheck: **`valueOfType_none_iff`** (`value_of_type` returns None iff the clean
acceptance spec holds), **`validateArgsType_go_none`**, `validate_fixed_iff`,
`numeric_body_cases`, `canReturnBoolean_of_accepts`, `isCompatibleWith_*`,
`unionDisplay_*`, `regGet_lower`. MathFns: **`unary_no_panic`** (the nine `unreachable!()`
arms are dead after validation), `sqrt_float_eq_ieee`, `sqrt_nan_iff`,
`abs_agree_except`, `coalesce_spec`, `uuid_shape`. Conversion:
`parseI64_sub_cStrtoll`, `toInteger_int_literals_agree`, **`toInteger_int_literal_agrees_c`**, `toBooleanList_eq`,
`toFloatList_eq`, `isEmpty_no_panic`. SelfCompare: **`mayFail_sound`** (the
`may_fail_self_match` pre-filter is sound), `isNeverEqual_iff`,
**`chunked_eq_firstDiff`** (16-lane block scan = first-difference scan). Misc:
**`decode_encode`** (v19 codec roundtrip for every storable value), `readVec_vecBytes`,
**`heap_le_amortized`**, `shares_le`, `fmtDuration_fields`.

## Confirmed bugs (Rust repro + live Rust vs C)

1. **FIXED — `toInteger` clamped out-of-range negatives to i64::MIN** (#2955; fixed by
   `ac5c27b76`, PR #2989, conversion.rs:53-59): `toInteger('-9223372036854775809')` …
   `'-9223372036854776832'` returned -9223372036854775808 (C null). Now an integer-only
   string that fails `parse::<i64>` is null without the f64 retry. Proved:
   `toInteger_neg_overflow_null` (the historical window, now null in Rust and C) and the
   general `toInteger_int_literal_agrees_c` (every `[+-]?digits+` string: Rust = C strtoll).
2. **FIXED — `coalesce()` accepted with zero arguments** (#2956; fixed by `648b41c5c`,
   PR #2990, mod.rs:813-819): built-in var-length functions now require ≥ 1 argument with
   C's message; UDFs still take any count. Proved: `coalesce_zero_args_rejected`,
   `validate_varLength` (iff), `udf_zero_args_ok`.

## Divergences from C (live; Rust often the better answer — decide per case)

- `MIN / -1` Rust MIN, C MAX; `(2^53+1)/1` Rust exact, C 2^53 (C divides through double).
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
  '1.5' vs '1.500000'; `toFloatList([' 3'])`/`toIntegerList([' 3'])` null vs 3.
- `p.LATITUDE` Rust 1.5, C null (Rust component names are ASCII case-insensitive).
- `toString(duration('PT0S'))` 'PT0S' vs C 'P' (C bug).
- Also seen (known): `sign(-2.5)` Float (#2906), `^` not type-checked (#2902),
  `toString(1.0)` '1', `toInteger('1e3')`, leading whitespace (functions_temporal).

## Suspicions / latent

- `Functions::is_aggregate(name)` (mod.rs:1079) does not lowercase; no caller today
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
`range:_src_id`/`range:_dest_id`); `dbms.functions` union spellings follow declaration
order (`union_order_visible`: Rust `Point or Null`, C `Null or Point`), `reducible` differs
for hasLabels/length/nodes/relationships, internal-op rows differ; `createNodeIndex`
C form `('L','p')` rejected (known stub, test_effects_shapes.py:670) and alone under
GRAPH.QUERY it fails with the misleading "graph.RO_QUERY is to be executed only on read-only
queries" (procedure_call.rs:67: a lone write procedure is planned as read-only);
`db.idx.fulltext.drop` of a missing index errors (C: no-op); `queryNodes(1, 2)` errors
(C: empty).
-/
