import FalkorTemporal.Calendar
import FalkorTemporal.EraAll
import FalkorTemporal.CalendarLaws
import FalkorTemporal.Duration
import FalkorTemporal.Constructors
import FalkorTemporal.VectorExpr
import FalkorTemporal.Spatial
import FalkorTemporal.Fns
import FalkorTemporal.Parse
import FalkorTemporal.Pure
import FalkorTemporal.PadDefs
import FalkorTemporal.PadAux
import FalkorTemporal.Components
import FalkorTemporal.DurationComponent
import FalkorTemporal.SpatialFns
import FalkorTemporal.VecDistance
/-!
# Temporal, spatial, conversion and columnar-expression functions of FalkorDB-rs

Lean 4 (core only, no Mathlib) models of `graph/src/runtime/functions/temporal.rs`,
the temporal half of `graph/src/runtime/value.rs`, `functions/spatial.rs`,
`vec_distance.rs`, `vector_expr.rs` and `functions/conversion.rs`. No `sorry`, `admit`
or `axiom`. The only enumeration is the 400-year Gregorian era (146 097 days and the
148 800-point date grid), checked by the kernel with `decide +kernel` in `EraFwd*.lean`
/ `EraBwd*.lean` (≈ 10 CPU-minutes; no `native_decide`).

Line-by-line coverage of every Rust function is in `COVERAGE.tsv` (PROVEN / MODELLED /
AXIOMATISED / NOT COVERED). Rust repros: `graph/tests/lean_functions_temporal.rs`
(`cargo test -p graph --test lean_functions_temporal -- --nocapture`; each `bug_*` test
asserts today's behaviour and must be flipped when fixed). Live checks: Rust module
(`target/release/libfalkordb.dylib`) on port 18280 vs C module
(`bin/macos-arm64v8-release/falkordb.so`) on 18281, `redis-cli GRAPH.QUERY g "<q>"`.

## Lean ↔ Rust

| Lean | Rust |
| --- | --- |
| `isLeap`, `daysInMonth` | `is_leap` `value.rs`, `days_in_month` `value.rs` |
| `daysFromCivil` | `days_from_civil` `value.rs` |
| `daysFromCivilT` | `days_from_civil` `temporal.rs` (second copy, truncating i32 `/`) |
| `civilFromDaysI` / `civilFromDays` | `civil_from_days_wide` `value.rs` (i64 year) / `civil_from_days` `value.rs` (with `y as i32`) |
| `constructDuration` | `construct_duration_secs` `temporal.rs` (ends in `check_date_range`) |
| `MAX_DATE_YEAR`, `checkDateRange` | `MAX_DATE_YEAR` `temporal.rs`, `check_date_range` `temporal.rs` |
| `decomposeDuration` | `decompose_duration` `temporal.rs` |
| `shiftMonths` / `shiftMonthsR` | `shift_months` `value.rs` (shared by `+`/`-`); arithmetic / with the `MAX_DATE_YEAR` guard |
| `addDur` / `subDur` (`addDurRaw` / `subDurRaw`: the unchecked arithmetic) | `add_duration_to_timestamp` `value.rs` / `sub_duration_from_timestamp` `value.rs` (range-checked) |
| `timeAddDur` / `timeSubDur` | `(Time, Duration)` arms of `add_slow` `value.rs` / `Sub` `value.rs` |
| `dateFromComponents`, `weekPath`, `quarterPath`, `ymdPath` | `date_from_components` `temporal.rs` (range-checked) |
| `checkRange`, `inRange`, `i32Ok`, `u32Ok` | `check_range` `temporal.rs`, `i32::try_from`/`u32::try_from` |
| `checkedAddDays`, `tdDaysOk`, `chronoYearOk` | chrono 0.4.45 `checked_add_signed` / `NaiveDate + TimeDelta`, `TimeDelta::days`, `MIN_YEAR`/`MAX_YEAR` |
| `weekSlice`, `sliceOk` | the `rest.len() <= 2` branch and `rest.split_at_checked(2)` in `parse_week_date` `temporal.rs` |
| `Fns.dateLoop`, `chk64` | `W`/`D` arms of `parse_duration_string` `temporal.rs` (the trailing-number-without-unit errors at and) (`checked_mul`/`checked_add`, #2987) |
| `formatDate`, `formatDatetime`, `pushDate`, `twoDigits` (`Components.lean`) | `format_date` `value.rs`, `format_datetime` `value.rs`, `push_date` `value.rs`, `write_two_digits` `value.rs` (i64 year, all digits, sign) |
| `SpatialFns.pointFromMapEntries`, `numeric`, `point` | `point_from_map_entries` `spatial.rs`, `numeric` `spatial.rs`, `point` `spatial.rs` |
| `VExpr.colEval` / `VExpr.rowEval` | `VectorEval::eval_and_or` `vector_expr.rs` / `ExprIR::Or` `eval.rs` / `ExprIR::And` `eval.rs` |
| `VExpr.intLane` / `valueArith` | `arithmetic` int lane `vector_expr.rs` / `Add..Rem for Value` `value.rs` |
| `Spatial2.distance` (`Spatial` holds only the `#guard`ed rounding mechanism) | `Point::distance` `value.rs` (clamped `a`) |

## Proven (160 theorems + 30 kernel-checked era chunks)

Calendar: `dfc_civil` (days_from_civil ∘ civil_from_days = id on all Int),
`civil_dfc` (civil_from_days ∘ days_from_civil = id on every real date), `civil_valid`
(civil_from_days only yields real dates, leap years included), `dfc_injective`,
`daysFromCivilT_eq` (the two Rust copies agree), `daysFromCivil_day`, `isLeap_add400`,
`isLeap_tmod` (Rust's truncating `%` is fine), `civilFromDays_exact` (i32 cast exact when
the year fits), `dim_link`, `era_fwd`/`era_bwd`/`dfcEra_range`/`dfcEra_of`/`yoe_bounds`.
Durations: `construct_decompose` (construct ∘ decompose = id — the law `Duration ±
Duration` relies on), `construct_ym_s`, `decompose_monthsDur`, `add_sub_months`
(`(d + PnM) − PnM = d` for day ≤ 28), Date/Time/Duration ordering is a total order
(`temporal_order_*`, `dateTs_mono_day`). Constructors: `dateFromComponents_total` (no
input of `date_from_components` panics), `weekPath_total`, `quarterPath_total`,
`weekPath_ok`, `quarterPath_ok`,
`weekSlice_err_iff`, `weekSlice_short_ok`, `sliceOk_ascii` (the split step errors exactly for a
long `rest` whose byte 2 is not a char boundary; the no-panic guarantee itself is the `Option`
type of `split_at_checked`).
Dates: `checkDateRange_iff`, `checkDateRange_boundary`, `shiftMonthsR_eq`, `shiftMonthsR_out_of_range`, `shiftMonthsR_no_wrap`, `shift_past_limit_errors`, `addDur_eq_raw`, `addDur_in_range`, `subDur_in_range`; `formatDate_spec`, `formatDatetime_spec`, `pushDate_eq`, `format_extremes` (the unit test's four `i64::MIN/MAX` strings), `format_wide_years`, `formatDate_wide_distinct`; `pointFromMapEntries`: `point_key_count`, `point_ok_iff`, `point_nan_refused`, `point_range_msgs`.
Columnar: `colEval_eq` (columnar AND/OR with row narrowing = per-row Kleene AND/OR; batch
fails iff some row fails), `rowEval_short`, `intLane_eq` (int lane = `Value` arithmetic,
incl. the divide-by-zero fallback), `mapM_bind_mapM`, `colEval_rows`, `rowK_eq`.
`shiftMonths_total`, `sub_eq_add_neg_months` (`ts - P{Y}Y{M}M = ts + P{-Y}Y{-M}M`),
`subDur_eq_shift`.
Counterexamples as theorems: `neg_days_encoding`, `civilFromDays_trunc`.
On main: `dayOfWeek_seven_is_sunday`, `dayOfWeek_zero_rejected`, `dayOfWeek_out_of_range_errors`.

## Confirmed divergences (Rust repro + live Rust-vs-C)

10. **Divergences, conversions (live only; `conversion.rs` is private)**:
    `toString(1.0)` = "1" (C "1.000000", openCypher "1.0" — looks like an Integer);
    `toString(0.1+0.2)` = "0.30000000000000004" (C "0.300000"); `toString(0.0/0)` "NaN"
    vs "nan"; `toString(1e300)` differs in every digit past 17 (`conversion.rs`
    `format!("{f}")` vs C `%f`). `toInteger('1e3')` = 1000 (C null);
    `toInteger(' 12')`, `toFloat(' 1')` null (C 12, 1); `toFloat('0x10')` null (C 16).
11. **Divergences, constructor validation** (Rust stricter or looser than C):
    `duration('P5')`, `duration('P-')` → `PT0S` (C null); `duration('PT1.5S')` → `PT1S`
    (C null); `duration({years:200000000})` errors (C `P200000000Y`);
    `date('2020-01-01') + duration({years:178956970, months:7})` = "8990-08-01" (year
    printed mod 10000; C "178958988-08-01").

Shared with C (not Rust-specific, openCypher divergence): negative day/second durations
are encoded as an instant before 1970 and decompose as `P-1Y11M30D…`, so
`date('2020-03-01') + duration({days:-1})` = 2020-03-02 (spec: 2020-02-29) —
`neg_days_encoding`. C also accepts 2021-02-29 (normalises to 03-01); Rust errors (spec).
C's `date('2021-01-03').dayOfQuarter` is 4 (Rust 3, correct).

## Suspected, not confirmed

- `decompose_duration`: `y - 1970` in i32 and `ya + yb` in `Duration + Duration`
  (`value.rs`) overflow in debug for durations whose year wraps (`civilFromDays_trunc`);
  release wraps silently. Not triggered in release (construct rejects afterwards).
- `shift_months` (shared by `+`/`-`): `y + years` (i32), and in
  `sub_duration_from_timestamp` `-years` (i32::MIN), plus `days * 86400` (i64), unchecked;
  only reachable through the year-wrap above.
- `hour: 4294967297` is truncated by `as u32` in `time_from_components` (C does the same).
  The date components are not truncated (`i32::try_from`/`u32::try_from`,
  `ymd_no_truncation`).
- `vec.*` distance via simsimd: not modelled (FFI).

## Modelling gaps

f64 is abstract (`toString`, `toInteger`/`toFloat` parsing, haversine are only
`#eval`/live-tested); chrono's formatting, `from_ymd_opt`, `from_isoywd_opt`, component
accessors and `Utc::now` are trusted; `addDur`/`subDur` are range-checked at
`MAX_DATE_YEAR` (the round-trip laws are proved on the unchecked arithmetic over `SmallYear`); the columnar proof covers AND/OR and the int lane, not CASE,
comparisons or the float lane; which row's error message a batch reports is not modelled.

## Wave 5 (`Fns.lean`, `Parse.lean`, `Pure.lean`, `Components.lean`, `SpatialFns.lean`)

Every fn of temporal.rs and spatial.rs plus the temporal accessors/formatters and
`Point::new`/`distance` of value.rs is PROVEN. chrono, `str::parse`, f64 `as` casts, the
clock, f64/f32 trig and simsimd are structures (`Num`, `Chrono`, `ChronoDT`, `Flt`, `Simd`)
or arguments — no axioms. Highlights: `parseDate_ok` (a parsed date is a valid calendar
date, or an ordinal / ISO-week result that the abstract chrono parser accepted), `parseTime_ok`, `parseDuration_weeks`,
`durationStruct_eq_map` / `dateStruct_eq_map` / `localtimeStruct_eq_map` (the binder's
positional-slot rewrite preserves values), `datePure_midnight`, `localtimePure_range`,
`clock_fns_shape`, `transaction_consistent`, `calendar_ranges`, `fields_roundtrip`,
`formatTime_length`, `distance_self`, `point_eq_struct`, `vecDist_kernel`.
The compact date/time branches reject
any non-digit (`allDigits` guard, temporal.rs) instead of silently dropping it —
`parseDate_compact_rejects`, `parseDate_compact_ok_allDigits`, `parseTime_compact_rejects`,
`compact_junk_examples`. The `W` path is separate (`parseDate_week_first`; `date('2020W1é')`
is covered by `weekSlice_err_iff`).
Suspicion (message only): `point({latitude: null, ..})` errors "requires 'latitude'
field" for a literal map (slot form) but "must be a number, got Null" for a map value
(`point_null_lat_msgs`); C uses one message for both.

## Duration parsing and components

`parse_duration_string` errors on a trailing number with
no unit (`P5`, `P1Y5`, `PT5`, `P1DT2H3`: "Duration component has no unit: 5") in both the date and
the time loop (`Fns.dateLoop` / `timeLoop` end-of-input arms; theorems
`dateLoop_trailing_number`, `timeLoop_trailing_number`, `dateLoop_end_ok`, `timeLoop_end_ok`,
`parseDuration_P5`; live: all four strings error, `P1D` still parses).
`toString(Float)` is `conversion.rs`, proved in proofs/value_math.

**`DurationComponent.lean` — `Value::get_duration_component` (value.rs) is chrono-bounded.**
`durComp_agrees_decompose`: inside chrono's year range (-262143..=262142, `chronoK`) every component
equals the `decompose_duration` one; `durComp_weeks`, `durComp_unknown`, `durComp_invalid_null`.
**BUG** (`durComp_null_but_representable`, `bug_duration_component_null`): a duration whose instant
leaves chrono's range (about 260 174 years and up) is still accepted by `check_date_range` (up to
~2.1e9 years) and printed/added by `decompose_duration`, but `.years`/`.days`/... return `null`.
Live (Rust main 256ab4d3f vs C master build): `RETURN duration({years: 262000, days: 3}).years` is
`null` in Rust and `262000` in C; `toString(duration({years: 1000000, days: 3}))` is `P1000000Y3D` in
both; 100000 and 200000 years agree. (C returns 0/0/3/6 for the negative `years: -1000000`; Rust
`null`.)

## Wave 6 (`VecDistance`)
vec_distance.rs PROVEN against the exact simsimd 6.5.16 wrapper (length check, then the
C kernel as a parameter): `euclidean = sqrt ∘ l2sq`, `inner_product = -dot`,
`distance_table` (None ⇒ euclidean, names case-sensitive), `distance_none_iff`
(None ⇔ unknown metric ∨ length mismatch).
-/
