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
import FalkorTemporal.Components
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
| `isLeap`, `daysInMonth` | `is_leap` `value.rs:764`, `days_in_month` `value.rs:752` |
| `daysFromCivil` | `days_from_civil` `value.rs:771` |
| `daysFromCivilT` | `days_from_civil` `temporal.rs:413` (second copy, truncating i32 `/`) |
| `civilFromDaysI` / `civilFromDays` | `civil_from_days` `value.rs:788` (without / with `y as i32`) |
| `constructDuration` | `construct_duration_secs` `temporal.rs:429` |
| `decomposeDuration` | `decompose_duration` `temporal.rs:470` |
| `addDur` / `subDur` | `add_duration_to_timestamp` `value.rs:696` / `sub_duration_from_timestamp` `value.rs:730` |
| `weekPath`, `quarterPath` | `date_from_components` `temporal.rs:62` (week / quarter branches) |
| `dateAddDays`, `tdDaysOk`, `chronoYearOk` | chrono 0.4.45 `NaiveDate + TimeDelta`, `TimeDelta::days`, `MIN_YEAR`/`MAX_YEAR` |
| `sliceOk` | `&rest[..2]` in `parse_week_date` `temporal.rs:248` |
| `wrapMul` | `n * 7` in `parse_duration_string` `temporal.rs:376` (release) |
| `formatDateDigits` | `format_date` `value.rs:274` + `write_date_into` `value.rs:803` |
| `VExpr.colEval` / `VExpr.rowEval` | `VectorEval::eval_and_or` `vector_expr.rs:392` / `ExprIR::And`,`Or` in `eval.rs:629-697` |
| `VExpr.intLane` / `valueArith` | `arithmetic` int lane `vector_expr.rs:878` / `Add..Rem for Value` `value.rs:903-1160` |
| `Spatial.haversine` | `Point::distance` `value.rs:121` |

## Proven (51 theorems + 30 kernel-checked era chunks)

Calendar: `dfc_civil` (days_from_civil ∘ civil_from_days = id on all Int),
`civil_dfc` (civil_from_days ∘ days_from_civil = id on every real date), `civil_valid`
(civil_from_days only yields real dates, leap years included), `dfc_injective`,
`daysFromCivilT_eq` (the two Rust copies agree), `daysFromCivil_day`, `isLeap_add400`,
`isLeap_tmod` (Rust's truncating `%` is fine), `civilFromDays_exact` (i32 cast exact when
the year fits), `dim_link`, `era_fwd`/`era_bwd`/`dfcEra_range`/`dfcEra_of`/`yoe_bounds`.
Durations: `construct_decompose` (construct ∘ decompose = id — the law `Duration ±
Duration` relies on), `construct_ym_s`, `decompose_monthsDur`, `add_sub_months`
(`(d + PnM) − PnM = d` for day ≤ 28), Date/Time/Duration ordering is a total order
(`temporal_order_*`, `dateTs_mono_day`). Constructors: `weekPath_total` (no panic for
|week| ≤ 10⁶ and |year| ≤ 200 000), `sliceOk_ascii`, `formatDate_injective_4digit`.
Columnar: `colEval_eq` (columnar AND/OR with row narrowing = per-row Kleene AND/OR; batch
fails iff some row fails), `rowEval_short`, `intLane_eq` (int lane = `Value` arithmetic,
incl. the divide-by-zero fallback), `mapM_bind_mapM`, `colEval_rows`, `rowK_eq`.
Counterexamples as theorems: `sub_vs_add_neg`, `neg_days_encoding`,
`time_not_normalised`, `week_string_wraps`, `week_panics`, `week_panics_delta`,
`quarter_panics`, `week_slice_panics`, `formatDate_not_injective`,
`dayOfWeek_seven_rejected`, `dayOfWeek_zero_is_sunday`, `civilFromDays_trunc`.

## Confirmed bugs (Rust repro + live Rust-vs-C)

1. **Server crash** `RETURN date({year:2020, week:10000000000})` — panic
   "`NaiveDate + TimeDelta` overflowed" at `temporal.rs:83`; C: "Invalid value for week
   (valid values 1 - 53)". `week: -9223372036854775807` → `TimeDelta::days out of bounds`
   (debug: multiply overflow at `:83`). Test `bug_date_week_huge_panics`. Fix: validate
   `week ∈ 1..=53` and use `checked_add_signed`/`try_days`.
2. **Server crash** `date({year:2020, quarter:2, dayOfQuarter: 9223372036854775807})` —
   `TimeDelta::days` panics (`temporal.rs:95`, `doq - 1` also overflows for i64::MIN);
   C: "Invalid value for dayOfQuarter (valid values 1 - 92)". Test
   `bug_date_day_of_quarter_panics`. Fix: `TimeDelta::try_days`, validate 1..=92.
3. **Server crash** `RETURN date('2020W1é')` — `&rest[..2]` not a char boundary
   (`temporal.rs:248`); C: "Failed to parse date". Test
   `bug_date_week_string_char_boundary_panics`. Fix: `rest.get(..2)` / require ASCII.
4. **Wrong result** `distance(point({latitude:-88.3, longitude:-180}),
   point({latitude:88.3, longitude:0}))` is `nan` (Rust) vs 20037518 (C); 35 428 of
   6 485 401 antipodal pairs on a 0.1° grid. `(1.0 - a).sqrt()` with `a` rounded above 1
   (`value.rs:137`). Test `bug_distance_antipodal_nan`. Fix: clamp `a` to `[0, 1]`.
5. **Wrong result** `date('2020-03-31') - duration({months:1})` = 2020-02-29 (Rust) vs
   2020-03-02 (C), while Rust's `date + duration({months:-1})` = 2020-03-02
   (`sub` clamps `value.rs:746`, `add` overflows `value.rs:712-722`). Test
   `bug_date_minus_month_disagrees_with_plus_negative_month`.
6. **Wrong result** `localtime('01:00') + duration({days:1}) = localtime('01:00')` →
   false (C true); same for `{months:1}`. Time arithmetic is not reduced mod 86400
   (`value.rs:1027`, `:1072`). Test `bug_time_plus_day_not_normalised`.
7. **Wrong result** `toString(date({year:12345}))` = "2345-01-01" (C "12345-01-01");
   `date({year:-5})` prints "0005-01-01" (`write_date_into` `value.rs:803`). Test
   `bug_format_date_truncates_year`. Relatedly `toString(localdatetime(...) +
   duration({years:300000}))` = "<invalid timestamp: 9468663474030>" (C
   "302020-01-01T10:20:30") — `format_datetime` via chrono range.
8. **Wrong result** `duration('P2635249153387078803W')` = `P5D` (release wrap of `n * 7`,
   `temporal.rs:376`; debug panics). C returns a (garbage) non-P5D value; the map form
   errors. Test `bug_duration_week_string_wraps`. Fix: `checked_mul`/`checked_add`.
9. **Divergence** map `dayOfWeek` is 0..6 with 0 = Sunday (`temporal.rs:75,84`):
   `date({year:2021, week:1, dayOfWeek:7})` errors (C: 2021-01-10); `dayOfWeek:0` accepted
   (C: error). The string form uses 1..7. Test `bug_date_map_day_of_week_is_zero_based`.
10. **Divergences, conversions (live only; `conversion.rs` is private)**:
    `toString(1.0)` = "1" (C "1.000000", openCypher "1.0" — looks like an Integer);
    `toString(0.1+0.2)` = "0.30000000000000004" (C "0.300000"); `toString(0.0/0)` "NaN"
    vs "nan"; `toString(1e300)` differs in every digit past 17 (`conversion.rs:131`
    `format!("{f}")` vs C `%f`). `toInteger('1e3')` = 1000 (C null);
    `toInteger(' 12')`, `toFloat(' 1')` null (C 12, 1); `toFloat('0x10')` null (C 16).
11. **Divergences, constructor validation** (Rust stricter or looser than C):
    `duration('P5')`, `duration('P-')` → `PT0S` (C null); `duration('PT1.5S')` → `PT1S`
    (C null); `date({year:2021, quarter:1, dayOfQuarter:200})` = 2021-07-19 (C error);
    `date({year:2021, week:60, dayOfWeek:0})` accepted (C error);
    `duration({years:200000000})` errors (C `P200000000Y`);
    `date('2020-01-01') + duration({years:178956970, months:7})` = "8990-08-01" (year
    printed mod 10000; C "178958988-08-01").

Shared with C (not Rust-specific, openCypher divergence): negative day/second durations
are encoded as an instant before 1970 and decompose as `P-1Y11M30D…`, so
`date('2020-03-01') + duration({days:-1})` = 2020-03-02 (spec: 2020-02-29) —
`neg_days_encoding`. C also accepts 2021-02-29 (normalises to 03-01); Rust errors (spec).
C's `date('2021-01-03').dayOfQuarter` is 4 (Rust 3, correct).

## Suspected, not confirmed

- `decompose_duration`: `y - 1970` in i32 and `ya + yb` in `Duration + Duration`
  (`value.rs:1014`) overflow in debug for durations whose year wraps (`civilFromDays_trunc`);
  release wraps silently. Not triggered in release (construct rejects afterwards).
- `add_duration_to_timestamp`: `y + years` (i32) and `new_days * 86400` (i64) unchecked;
  only reachable through the year-wrap above.
- `date({year: 4294969266})`, `month: 4294967298`, `hour: 4294967297` are truncated by
  `as i32`/`as u32` (C does the same).
- `vec.*` distance via simsimd: not modelled (FFI).

## Modelling gaps

f64 is abstract (`toString`, `toInteger`/`toFloat` parsing, haversine are only
`#eval`/live-tested); chrono's formatting, `from_ymd_opt`, `from_isoywd_opt`, component
accessors and `Utc::now` are trusted; i32/i64 overflow in `addDur`/`subDur` is outside the
proved range (`SmallYear`); the columnar proof covers AND/OR and the int lane, not CASE,
comparisons or the float lane; which row's error message a batch reports is not modelled.

## Wave 5 (`Fns.lean`, `Parse.lean`, `Pure.lean`, `Components.lean`, `SpatialFns.lean`)

Every fn of temporal.rs and spatial.rs plus the temporal accessors/formatters and
`Point::new`/`distance` of value.rs is PROVEN. chrono, `str::parse`, f64 `as` casts, the
clock, f64/f32 trig and simsimd are structures (`Num`, `Chrono`, `ChronoDT`, `Flt`, `Simd`)
or arguments — no axioms. Highlights: `parseDate_ok` (a parsed date is a valid calendar
date / ordinal / ISO-week date), `parseTime_ok`, `parseDuration_weeks`,
`durationStruct_eq_map` / `dateStruct_eq_map` / `localtimeStruct_eq_map` (the binder's
positional-slot rewrite preserves values), `datePure_midnight`, `localtimePure_range`,
`clock_fns_shape`, `transaction_consistent`, `calendar_ranges`, `fields_roundtrip`,
`formatTime_length`, `distance_self`, `point_eq_struct`, `vecDist_kernel`.
Suspicion (message only): `point({latitude: null, ..})` errors "requires 'latitude'
field" for a literal map (slot form) but "must be a number, got Null" for a map value
(`point_null_lat_msgs`); C uses one message for both.

## Wave 6 (`VecDistance`)
vec_distance.rs PROVEN against the exact simsimd 6.5.16 wrapper (length check, then the
C kernel as a parameter): `euclidean = sqrt ∘ l2sq`, `inner_product = -dot`,
`distance_table` (None ⇒ euclidean, names case-sensitive), `distance_none_iff`
(None ⇔ unknown metric ∨ length mismatch).
-/
