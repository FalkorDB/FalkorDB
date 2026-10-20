# value_math: arithmetic on values, built-in functions and their argument checks

This project proves what the engine does when it computes with values. It covers four areas:

- the `+ - * / %` operators on `Value`;
- the math, trig and conversion functions;
- how function arguments are type-checked before a function body runs;
- the function registry, the scalar aggregation bodies and the built-in procedures (`db.labels`, `dbms.functions`, …).

It also covers value memory accounting, the v19 value codec, and the nesting cap that #3339 put on stored values. Throughout, Rust is compared with C FalkorDB on the inputs where both have an answer.

User-facing behaviour that depends on it:

- every arithmetic expression, for example `RETURN 7 / 2, -7 % 3, {a:1} + {a:2}, [1] + 2`;
- `abs`, `sqrt`, `sin` and the other math and trig functions, `toInteger`, `toFloat`, `toString` and their list forms, and `coalesce`;
- the "Type mismatch" and "Received N arguments" errors every function call can raise;
- `count`, `sum`, `avg`, `collect`, `stDev` and `percentile*`;
- `CALL db.labels()`, `db.propertyKeys()`, `db.meta.stats()`, `dbms.procedures()`, `dbms.functions()` and the index DDL procedures;
- `GRAPH.MEMORY`, and saving and loading property values through RDB and `RESTORE`.

The Rust is in `graph/src/runtime/value.rs` and `graph/src/runtime/functions/` (`mod.rs`, `math.rs`, `trig.rs`, `conversion.rs`, `aggregation.rs`, `procedures.rs`, `path.rs`), plus `index_ddl.rs` and `entity_type.rs`. The full Lean↔Rust mapping (every Lean name next to the Rust `file:line` it models) is in the header of [`FalkorValueMath.lean`](FalkorValueMath.lean). This README explains *what* is proven and *why it matters*.

Comparison, equality and hashing laws are not here: they are in [`../value_order`](../value_order).

## The questions this project answers

**1. Can a built-in function panic on a bad argument?**

No. Every numeric function body ends in `_ => unreachable!()`. `unary_no_panic` proves those arms are dead: once argument validation accepts `Int | Float | Null`, only those three variants reach the body. The type check itself is proven to mean what it says. `valueOfType_none_iff` shows that `value_of_type` returns `None` exactly when the value matches the declared type:

- `Any` accepts everything;
- a plain type needs the same variant;
- a list type checks its elements;
- `Optional` checks the inner type;
- `Union` needs some member to match.

**2. What do `/` and `%` do on integers?**

They wrap, as in a Rust release build. They agree with exact arithmetic everywhere except `i64::MIN / -1` (`wDiv_exact`, `wRem_exact`). The division identity `(a / b) * b + a % b = a` always holds (`div_rem_identity`). Every operator propagates null, and each one has an exact error condition. For example, `mul_ok_iff` says `*` fails exactly when a non-null operand isn't a number. Map `+` is a right-biased merge: `{a:1} + {a:2}` is `{a:2}` (`lookup_omExtend`), and the result never has duplicate keys (`nodup_omExtend`).

**3. Does `toInteger('…')` agree with C?**

On every integer literal it does: an optional sign followed by ASCII digits, of any length. `toInteger_int_literal_agrees_c` proves that Rust's answer equals C's `strtoll` branch, whatever the `f64` parser does. In range both return the value, and out of range both return `null`. Before #2989, Rust clamped `toInteger('-9223372036854775809')` to `i64::MIN`; `toInteger_neg_overflow_null` covers that window.

**4. Can a stored value be nested deep enough to crash the server on load?**

No, since #3339:

- `decode_reach`: whatever the input, the RDB/`RESTORE` decoder never recurses deeper than `MAX_VALUE_NESTING + 1`;
- `decodeValue_agrees`: below the cap, the decoder accepts exactly what the pre-#3339 decoder accepted;
- `validateProperty_depth`: every value a property write accepts is within the cap;
- `stored_roundtrip`: every accepted value saves and loads back unchanged, for vectors under 2³² dimensions.

**5. Are the trig functions and the procedures the same as C?**

- **Trig.** `unary_rust_eq_c` proves that every unary trig function is C's expression tree, with the same null, type and arity handling. `atan2_spec` covers `atan2`.
- **Procedures.** The `yields_*` theorems prove that every procedure returns one column per declared `YIELD`. `dbms.functions()` merges built-ins and UDFs by hand; `mergePicks_eq` proves that merge is `List.merge`, so the output is sorted and is a permutation of the inputs.

The divergences that remain are listed under [Bugs found](#bugs-found).

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `valueOfType_none_iff` | The argument type check accepts exactly the values that match the declared type | `TypeCheck.lean` |
| `unary_no_panic` | No math function reaches its `unreachable!()` after validation | `MathFns.lean` |
| `div_rem_identity`, `wDiv_exact`, `wRem_exact` | Integer `/` and `%` are exact except `MIN / -1`, and `(a/b)*b + a%b = a` | `Int64.lean` |
| `mul_ok_iff`, `div_ok_iff`, `rem_ok_iff` | The exact conditions under which `*`, `/`, `%` return an error | `Value.lean` |
| `lookup_omExtend`, `nodup_omExtend` | Map `+` is a right-biased merge with no duplicate keys | `Value.lean` |
| `mul_agree_on_numeric`, `add_agree_int_int` | Rust `+` and `*` equal C's on the inputs both accept | `CArith.lean` |
| `toInteger_int_literal_agrees_c` | `toInteger` on any integer string equals C's `strtoll` (#2989) | `Conversion.lean` |
| `coalesce_zero_args_rejected` | `coalesce()` with no arguments is an arity error, as in C (#2990) | `TypeCheck.lean` |
| `mayFail_sound` | The `may_fail_self_match` pre-filter is sound: when it says "can't fail", `v = v` is conclusively true | `SelfCompare.lean` |
| `chunked_eq_firstDiff` | The 16-lane vector comparison finds the same first difference as an element-by-element scan | `SelfCompare.lean` |
| `decode_reach`, `decodeValue_agrees`, `stored_roundtrip` | Decoding is depth-bounded, unchanged below the cap, and every accepted property round-trips (#3339) | `Nesting.lean` |
| `heap_le_amortized` | For unshared values, the amortized memory figure is at least the plain heap size | `Misc.lean` |
| `registerAll_ok` | Registering the built-ins never panics when the lowered names are distinct | `Registry.lean` |
| `unary_rust_eq_c` | Every unary trig function equals C's | `Trig.lean` |
| `mergePicks_eq` | `dbms.functions()`'s built-in/UDF merge is a sorted merge | `Procedures.lean` |

### How `valueOfType_none_iff` is proved

1. **Model the type.** `V F` is the engine `Value` from [`../core`](../core) (all 16 variants), with the float payload kept abstract. `Ty` is the declared argument type: `Any`, a plain type, `List t`, `Optional t` or `Union [t…]`.
2. **Model the function.** `valueOfType` is `Value::value_of_type` (`value.rs`), arm for arm. It returns `none` for a match and an error text otherwise.
3. **Write the spec separately.** `Accepts v t` is the clean acceptance relation, one rule per type former.
4. **Induct on the type.** The proof is a nested induction over `Ty`, because `Union` holds a list of types. The list case needs the element-wise check, and the union case needs "some member accepts".

The result is used by `validateArgsType_go_none` (the whole argument list), and from there by `unary_no_panic` and `unary_rust_eq_c`. There are no hypotheses about other Rust code.

## Bugs found

Each bug was confirmed with a Rust repro and live against C (see [`../FINDINGS.md`](../FINDINGS.md)).

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| 1 | `toInteger_neg_overflow_null` | `toInteger('-9223372036854775809')` and the strings just below it returned `i64::MIN` (C: `null`) | Fixed by #2989 (closes #2955) |
| 2 | `coalesce_zero_args_rejected` | `coalesce()` with zero arguments was accepted and returned `null` (C: arity error). Same for `indegree`/`outdegree` | Fixed by #2990 (closes #2956) |
| 3 | `pre3339_decode_any_depth`, `pre3339_isValid_any_depth` | Stored values had no nesting limit, so a deeply nested value could overflow the stack on load (GHSA-mwvh-8fq5-x2j8, #2722) | Fixed by #3339 (`decode_reach`, `stored_roundtrip`) |
| 4 | `pre2907_sign_float_is_float` (in `expr_semantics`) | `sign(-2.5)` returned a Float | Fixed by #2907 |
| — | `isAggregate_case_sensitive` | `Functions::is_aggregate("COUNT")` is false although `count` is registered, because it doesn't lowercase | Latent: no caller today |
| — | `encode_map_is_null` | `Encode<19>` writes a Map, Node, Relationship or Path as `NULL` behind only a `debug_assert!` | Latent: property writes never store those types |
| — | `within_nest_empty` | The decoder accepts one level more than writers produce (`MAX` arrays around `[]`) | Harmless: still bounded, never stored |
| — | `union_order_visible` | `dbms.functions()` prints union types in declaration order (`Point or Null`), while C uses its own order (`Null or Point`) | Divergence from C, open |

Other live divergences from C are recorded in the header, not as bugs, because Rust is often the better answer:

- `MIN / -1` gives `MIN` in Rust and `MAX` in C. `(2^53+1)/1` is exact in Rust; C divides through a double.
- `abs(MIN)` errors in Rust. `abs(-0.0)` is `0` in Rust and `-0` in C.
- Some `+` typing differs. `'a' + {b:1}` is an error in Rust, while C concatenates. `true + 1` is an error in Rust, while C returns a Float.
- `toJSON` of floats and maps is formatted differently.
- `toFloat('0.1')` is exact in Rust. C goes through `strtof`, which is a C bug.
- Rust accepts `p.LATITUDE` case-insensitively.
- `db.indexes` and `db.constraints` return rows in a different order.

This project has no CONDITIONAL rows.

## Files

| File | What's in it |
|---|---|
| `Int64.lean` | i64 wrapping `/` and `%`, `div_rem_identity`, `i as f64`, C's division for comparison |
| `Value.lean` | `Value` `+ - * / %` with null propagation and exact error conditions; `OrderMap` insert/extend for map `+` |
| `CArith.lean` | C's `SIValue_Add`/`_Multiply` and their agreement with Rust on the shared domain |
| `TypeCheck.lean` | `value_of_type`, `validate_args_type`, arity, `is_compatible_with`, union display, registry lookup; `coalesce()` |
| `MathFns.lean` | `abs` … `sqrt`, `coalesce`, `randomUUID` layout, `^`; `unary_no_panic` |
| `Trig.lean` | `trig.rs` against C's expression trees |
| `Conversion.lean` | `toInteger`/`toFloat`/`toString`/`toBoolean` and list forms, integer parsing against C `strtoll` |
| `ConvReg.lean` | The rest of `conversion.rs` (registration and wrappers) |
| `SelfCompare.lean` | `compare_value` on lists and maps, the chunked vector scan, `may_fail_self_match`, `is_never_equal` |
| `Misc.lean` | Duration formatting, heap-size accounting, attribute and point/duration accessors, the v19 codec before #3339, JSON floats |
| `Nesting.lean` | #3339: `MAX_VALUE_NESTING`, the capped decoder, property validation, round-trip |
| `Registry.lean` | `functions/mod.rs` dispatch, `GraphFn` constructors, `add*`, `init_functions`, the UDF registry and version counter |
| `AggScalar.lean` | The scalar aggregation bodies (`count`, `sum`, `avg`, `collect`, `stDev`, `percentile*`) and `rand` |
| `Procedures.lean` | `procedures.rs`, `path.rs` and `entity_type.rs`: YIELD shapes, `dbms.procedures`/`dbms.functions` ordering, `db.meta.stats` |
| `IndexDdl.lean` | `index_ddl.rs` (stats and effects bookkeeping) |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

Shared foundations (the engine `Value` as `V F`, `wrap` and the `wrapping_*` helpers, `InI64`) come from [`../core`](../core) through a Lake path dependency.

## Coverage

Line numbers are pinned to main 289c201fe.

`COVERAGE.tsv` has 184 rows: 67 in `value.rs`, 36 in `functions/mod.rs`, 20 in `procedures.rs`, 17 in `math.rs`, 14 in `trig.rs`, 12 in `aggregation.rs`, 11 in `conversion.rs`, 3 each in `path.rs` and `index_ddl.rs`, and 1 in `entity_type.rs`.

| Bucket | Rows |
|---|---|
| PROVEN | 161 |
| MODELLED | 3 |
| NOT COVERED | 20 |

All non-PROVEN rows are in `value.rs`:

- **MODELLED:** `Point::distance` (proven in `functions_temporal`, `Spatial.lean`), `Value::name` (used by the error-message theorems) and `fmt_json` (only non-finite floats are proven; the rest is `write!` formatting, live-checked against C).
- **NOT COVERED:** trait method declarations (`order`, `compare_value`, `value_of_type`, `get_type`, `contains`, `fmt_json`); trivial constructors and `clone`; chrono formatting and calendar accessors (`format_datetime`, `get_date_component`, …), which are trusted and live-checked against C; `json_escape` and node JSON writing.

The header's own count ("72 PROVEN, 11 MODELLED, 46 NOT COVERED of 129") predates waves 5 and 6; the numbers above are computed from the current `COVERAGE.tsv`. `bash proofs/coverage.sh` counts each function once, in its best bucket across all projects.

## What the model does not cover

- **Floats.** `f64` is abstract (`FOps`, the `FC` classes, `FloatCmp` laws). libm `exp`, `ln`, `log10`, `floor`, `ceil` and `round` are parameters, so their results are trusted.
- **Parsing.** `f64::from_str` is modelled only by its grammar and its integer-literal rounding, not by the value it returns.
- **Trusted libraries.** Chrono calendar accessors and formatting, `json_escape`, the RNG behind `rand`, the QuickJS UDF bridge and registry locking.
- **Temporal helpers.** These are parameters here; they are proven in [`../functions_temporal`](../functions_temporal). The exception is `Time ± Duration`, which since #2985 no longer calls them: `timeAddDur`/`timeSubDur` model it directly, and `add_time_dur_in_day`, `sub_time_dur_in_day`, `time_sub_add_dur`, `time_add_whole_days` prove it never fails, always lands in `[0, 86400)`, and ignores whole days.
- **Ordering and hashing.** Value ordering, equality and hashing are proven in `value_order`, not here.
- **Memory accounting.** `heap_size` charges a `Relationship` 24 bytes it doesn't own; the model records this but doesn't treat it as wrong.

## Build and check

```bash
cd proofs/value_math && lake build    # this project
bash proofs/lean_ci.sh value_math     # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh               # repo-wide coverage report
```

**When you change the `Value` operators, the codec or the formatting in `value.rs`, or anything under `runtime/functions/`:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match.
3. Re-run the build.

A new built-in function needs a row and, if it is numeric, an entry in `unaryFns` so that `unary_no_panic` covers it. If a Rust-equals-C theorem stops holding, either the change introduced a divergence from C or the theorem needs a new hypothesis. Either way, say which in the PR.
