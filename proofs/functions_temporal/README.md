# functions_temporal: dates, times, durations, points and vector distances

This project proves that the calendar arithmetic underneath every temporal value is correct, and records where the temporal constructors crash or disagree with C. Every `Date`, `LocalTime`, `LocalDateTime` and `Duration` in FalkorDB-rs is one `i64` count of seconds. Turning that number into a year, month and day, and back, goes through two small calendar functions. If either one is off by a day for some date, every temporal feature is off by a day for that date.

User-facing behaviour that depends on it:

- `date(…)`, `localtime(…)`, `localdatetime(…)` and `duration(…)`, in both their map and string forms;
- `date + duration`, `date - duration` and `duration + duration`;
- `.year`, `.month`, `.week`, `.dayOfWeek` and the other component accessors, and `toString` of temporal values;
- `ORDER BY` over dates, times and durations;
- `point(…)`, `distance(…)`, and the `vec.*` distance functions.

It also proves one small columnar result: `AND`/`OR` and the integer arithmetic lane give the same answer as per-row evaluation.

The Rust is in `graph/src/runtime/functions/temporal.rs`, the temporal half of `graph/src/runtime/value.rs`, `functions/spatial.rs`, `vec_distance.rs`, and parts of `vector_expr.rs` and `functions/conversion.rs`. The full Lean↔Rust mapping (every Lean name next to the Rust `file:line` it models) is in the header of [`FalkorTemporal.lean`](FalkorTemporal.lean). This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. Is the calendar right for every date?**

Yes. `days_from_civil` turns (year, month, day) into a day number, and `civil_from_days` turns it back (Howard Hinnant's algorithms). The project proves:

- `dfc_civil`: day number → date → day number is the identity on **every** integer;
- `civil_dfc`: date → day number → date is the identity on every real date;
- `civil_valid`: `civil_from_days` only produces real dates, with leap years included;
- `dfc_injective`: two different dates never share a day number.

Rust has a second copy of `days_from_civil` in `temporal.rs`, which uses `i32` and truncating division. `daysFromCivilT_eq` proves the two copies agree. The final `y as i32` cast is exact whenever the year fits (`civilFromDays_exact`).

**2. Is `Duration + Duration` correct?**

A duration is stored as "the instant 1970-01-01 plus the years and months plus the days and seconds", and its components are recovered by decomposing that instant. `construct_decompose` proves, under range hypotheses on the components (they fit `i64`/`i32`), that construct ∘ decompose is the identity. That encoding round-trip is the law `Duration ± Duration` relies on, but the `(Duration, Duration)` arm of `Add for Value` and its overflow behaviour are not proved here: only the round-trip it rests on. `add_sub_months` proves that `(d + P1M) - P1M = d` for every day of the month up to 28. `+` and `-` share `shift_months` (`shiftMonths`), and `sub_eq_add_neg_months` proves `ts - P{Y}Y{M}M = ts + P{-Y}Y{-M}M` for every timestamp, for a pure year/month duration built from a month total `t` with `-1000000 ≤ t ≤ 1000000`; `shiftMonths_total` shows the shift depends only on the total month count.

**3. Is ordering of temporal values a valid sort?**

Yes. Date, time and duration ordering is a total order (`temporal_order_total`, `temporal_order_trans`, `temporal_order_antisymm`), so `ORDER BY` over them can't trip Rust's sort check. A later day always has a larger timestamp (`dateTs_mono_day`).

**4. Can a temporal constructor crash the server?**

No. `date_from_components` range-checks every component before chrono sees it: week 1–53, quarter 1–4, dayOfQuarter 1–92, and `i32`/`u32` conversions that fail instead of truncating. The week string is split with `split_at_checked`. The project proves:

- `dateFromComponents_total`: `date_from_components` returns a date or an error for **every** input, any integer in any of its seven slots. This is what `date(map)`, `localdatetime(map)` and their slot forms call;
- `weekSlice_err_iff`, `weekSlice_short_ok`: the `W` string split errors exactly when `rest` is longer than two bytes and byte 2 is not a char boundary, so `date('2020W1é')` is an error (`week_slice_errors`). That the split cannot panic is the type of `split_at_checked` (an `Option`), not a theorem; the numeric parses behind it are abstract;
- `weekPath_ok`, `quarterPath_ok`: the week and quarter paths for in-range inputs.

An in-range week whose date falls outside chrono's year range, such as `date({year:-262143, week:1})`, is an error (`week_edge_errors`).

**5. Do the parsers produce real dates?**

Yes:

- a successful date parse yields a valid calendar date, or a result that the abstract chrono ordinal / ISO-week parser accepted (`parseDate_ok`; chrono's own validity laws are not modelled);
- a successfully parsed time string is below 86 400 seconds (`parseTime_ok`);
- compact strings with a non-digit are rejected (`parseDate_compact_rejects`, `parseTime_compact_rejects`);
- the binder's rewrite of `date({year: …})` into positional slots preserves values (`dateStruct_eq_map`, `localtimeStruct_eq_map`, `durationStruct_eq_map`).

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `dfc_civil`, `civil_dfc` | Day number ↔ (year, month, day) are mutually inverse: on all integers one way, on all real dates the other | `CalendarLaws.lean` |
| `civil_valid`, `dfc_injective` | Every day number maps to a real date, and no two dates share a day number | `CalendarLaws.lean` |
| `fwdOk_all`, `bwdOk_all` | The 400-year Gregorian era checked day by day by the Lean kernel | `EraAll.lean` |
| `daysFromCivilT_eq` | The two Rust copies of `days_from_civil` agree | `Calendar.lean` |
| `construct_decompose` | Decomposing a duration and rebuilding it gives the same duration | `Duration.lean` |
| `temporal_order_total` | Date/time/duration ordering is total | `Duration.lean` |
| `dateFromComponents_total`, `weekSlice_err_iff` | `date_from_components` is total, and the week-string split errors exactly when it would not be a valid slice | `Constructors.lean` |
| `parseDate_ok`, `parseTime_ok` | A parsed date is a calendar date or a chrono-accepted ordinal / ISO-week result; a parsed time is below 86 400 s | `Parse.lean` |
| `durationStruct_eq_map`, `dateStruct_eq_map` | The positional-slot constructor forms equal the map forms | `Pure.lean` |
| `distance_finite_coords`, `distance_finite` | `distance()` is finite (never NaN) for points with finite coordinates, antipodal ones included; `distance_finite` is the same for any point whose haversine term is not NaN. Both rest on binary64 laws stated as `Flt` / `NanLaws` fields | `SpatialFns.lean` |
| `timeAddDur_eq_tod`, `timeSub_timeAdd` | `localtime ± duration` is the time of day of the date-arithmetic result, and `-` undoes `+` | `Duration.lean` |
| `dateLoop_days_i64` | The `duration('P…W…D')` day accumulator never wraps; overflow is an error | `Parse.lean` |
| `colEval_eq`, `intLane_eq` | Columnar `AND`/`OR` and the int arithmetic lane equal per-row evaluation | `VectorExpr.lean` |
| `distance_none_iff`, `vecDist_kernel` | The internal `vec_distance::distance` returns `None` exactly for an unknown metric or a length mismatch (`VecDistance.lean`); the public `vec.*` functions turn a length mismatch into an error (`vecDist_mismatch`) and otherwise return the simsimd kernel's value (`vecDist_kernel`, `SpatialFns.lean`) | `VecDistance.lean`, `SpatialFns.lean` |

### How `dfc_civil` and `civil_dfc` are proved

1. **Model the functions.** `daysFromCivil` and `civilFromDaysI` are the Rust functions over `Int`, with `div_euclid`/`rem_euclid` as Lean's `/`/`%` (which agree for a positive divisor). Their sub-expressions get names (`yoeOf`, `doyOf`, `mpOf`, `dOf`, `dfcEra`), so proofs can talk about them.
2. **Split off the era.** Both algorithms first reduce to a position inside a 400-year era of 146 097 days. Outside the era the arithmetic is linear, and `omega` handles it.
3. **Check the era exhaustively.** Inside one era there are finitely many cases. `fwdOk n` checks that era day `n` decodes to an in-range month and a real day of that month. `bwdOk k` checks that a real date on the 148 800-point (year-of-era, month, day) grid encodes and decodes back to itself. Both are checked for *every* point by the Lean kernel with `decide +kernel`, split into 30 chunks across `EraFwd*.lean` and `EraBwd*.lean`. That takes about 10 CPU-minutes and doesn't use `native_decide`.
4. **Combine.** `era_fwd` and `era_bwd` lift the finite checks to statements about the era pieces, and `dfc_civil` and `civil_dfc` assemble them. `civil_dfc` needs `ValidDate y m d`: a month in 1..12 and a day in the month's range. Rust only ever calls it on such dates.

## Bugs found

Each bug has a Rust repro test (`graph/tests/lean_functions_temporal.rs`, run locally and not committed) and was compared live against C. See [`../FINDINGS.md`](../FINDINGS.md). The first row is an open divergence from C; the others are shared with C, suspected, or message-only.

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| — | `durComp_null_but_representable`, `bug_duration_component_null` (`DurationComponent.lean`) | `Value::get_duration_component` is chrono-bounded: a duration whose instant leaves chrono's year range (about 260,174 years and up) is accepted by `check_date_range` (up to ~2.1e9 years) and printed by `decompose_duration`, but `.years`/`.days`/… return `null`. `RETURN duration({years: 262000, days: 3}).years` is `null` in Rust, `262000` in C (re-checked live on 256ab4d3f); `years: -1000000` gives `null` in Rust, 0/0/3/6 in C | **Open** (confirmed live) |
| — | `neg_days_encoding` | `date('2020-03-01') + duration({days:-1})` is 2020-03-02 (spec: 02-29), because negative durations are encoded as an instant before 1970 | Shared with C |
| — | `civilFromDays_trunc` | `y - 1970` in `i32` inside `decompose_duration` overflows for durations whose year wraps | Suspected, not triggered in release |
| — | `point_null_lat_msgs` | `point({latitude:null, …})` gives two different error messages for a literal map and a map value | Message only |

The header also records conversion divergences found live: `toString(1.0)` is `"1"` (C `"1.000000"`), `toInteger('1e3')` is 1000 (C `null`), and `toFloat(' 1')` is `null` (C 1). They aren't modelled, since `f64` formatting and parsing are abstract. More live-only conversion divergences (`conversion.rs` is private): `toString(0.1+0.2)` is `"0.30000000000000004"` (C `"0.300000"`), `toString(0.0/0)` is `"NaN"` (C `"nan"`), `toString(1e300)` differs past 17 digits (`format!("{f}")` vs C `%f`), `toInteger(' 12')` is `null` (C 12), `toFloat('0x10')` is `null` (C 16).

Constructor validation, Rust stricter or looser than C: `duration('P5')` and `duration('P-')` give `PT0S` (C null); `duration('PT1.5S')` gives `PT1S` (C null); `duration({years:200000000})` errors (C `P200000000Y`); `date('2020-01-01') + duration({years:178956970, months:7})` prints "8990-08-01" (year mod 10000; C "178958988-08-01"). C also accepts 2021-02-29 (normalises to 03-01; Rust errors, as the spec says), and C's `date('2021-01-03').dayOfQuarter` is 4 (Rust 3, correct).

Suspected, not confirmed: `ya + yb` in `Duration + Duration` and `shift_months` (`y + years` in i32, `-years` at `i32::MIN`, `days * 86400` in i64) are unchecked and overflow in debug only for year-wrapping durations; `hour: 4294967297` is truncated by `as u32` in `time_from_components` (C does the same).

This project has no CONDITIONAL rows.

## Files

| File | What's in it |
|---|---|
| `Calendar.lean` | `isLeap`, `daysInMonth`, both `days_from_civil` copies, `civil_from_days`, and the named era pieces |
| `EraDefs.lean`, `EraFwd0-3.lean`, `EraBwd0-3.lean`, `EraAll.lean` | The kernel-checked enumeration of one 400-year era (`fwdOk_all`, `bwdOk_all`) |
| `CalendarLaws.lean` | `dfc_civil`, `civil_dfc`, `civil_valid`, `dfc_injective`, the `i32` cast |
| `Duration.lean` | Duration construct/decompose, `add_duration_to_timestamp`/`sub_duration_from_timestamp`, `Time ± Duration`, temporal ordering |
| `Constructors.lean` | `date_from_components` (week, quarter and year/month/day branches) and its totality, the `W` string split |
| `Fns.lean` | Field extractors, `*_pure` constructors, the clock functions, `register` |
| `Parse.lean` | `parse_date_string`, `parse_time_string`, `parse_datetime_string`, `parse_duration_string` (checked `W`/`D`) |
| `Pure.lean` | The slot-based `*_struct_pure` constructors equal the map forms |
| `Components.lean` | Temporal accessors and date/time formatting (`value.rs`) |
| `Spatial.lean`, `SpatialFns.lean` | `Point::distance` (haversine, clamped) and `spatial.rs`; the antipodal NaN, by evaluation and by proof over `NanLaws` |
| `VecDistance.lean` | `vec_distance.rs` over the simsimd kernel as a parameter |
| `VectorExpr.lean` | Columnar `AND`/`OR` and the int arithmetic lane against per-row evaluation |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

The project is self-contained: its `lakefile.toml` has no dependencies, and it does not import `../core`.

## Coverage

`COVERAGE.tsv` has 66 rows, all PROVEN: 33 in `temporal.rs`, 21 in `value.rs`, 8 in `spatial.rs` and 4 in `vec_distance.rs`. Two `value.rs` rows are arm rows (the `Time ± Duration` arms of `add_slow` and `Sub::sub`), pinned to the enclosing function. `get_duration_component` is one of the `value.rs` rows (`DurationComponent.lean`), and `format_duration` is a PROVEN row of [`../value_math`](../value_math) (`fmtDuration_*`, using this project's `decompose_duration`).

| Bucket | Rows |
|---|---|
| PROVEN | 66 |

The columnar evaluator (`vector_expr.rs`) and the `f64` conversion functions (`conversion.rs`) are no longer rows of this ledger: they are covered by [`../columnar`](../columnar), [`../expr_semantics`](../expr_semantics) and [`../value_math`](../value_math).

`bash proofs/coverage.sh` counts each function once, in its best bucket across all projects.

## What the model does not cover

- **Floats.** `f64` is abstract. `toString` and `toInteger`/`toFloat` parsing are only live-tested. The antipodal-NaN theorems assume the binary64 facts they use (`NanLaws`/`Flt`: `f64::clamp`, `sqrt`, `atan2`, NaN propagation, magnitude bounds of `sin`/`cos`/`atan2`) rather than deriving them, and `mul_add`'s fused rounding is not modelled.
- **Chrono.** Formatting, `from_ymd_opt`, `from_isoywd_opt`, the component accessors and `Utc::now` are trusted, as structures (`Chrono`, `ChronoDT`) whose laws come from chrono's docs.
- **Overflow.** `add_duration_to_timestamp`/`sub_duration_from_timestamp` are range-checked (`addDur`, `shiftMonthsR`, `checkDateRange_iff`); the round-trip laws (`add_sub_months`) are proved on the unchecked arithmetic (`addDurRaw`) over `SmallYear`, and `addDur_eq_raw` links the two.
- **Columnar coverage.** The columnar proof here covers `AND`/`OR` and the int lane only, not `CASE`, comparisons or the float lane (those are in `columnar` and `expr_semantics`). Which row's error message a batch reports isn't modelled.
- **simsimd.** The `vec.*` kernels are an FFI parameter.

## Build and check

```bash
(cd proofs/functions_temporal && lake build)    # this project (the era check takes ~10 CPU-minutes)
bash proofs/lean_ci.sh functions_temporal     # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                       # repo-wide coverage report
```

**When you change `functions/temporal.rs`, `spatial.rs`, `vec_distance.rs` or the calendar and duration code in `value.rs`:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match.
3. Re-run the build.

A change to `days_from_civil` or `civil_from_days` re-runs the era enumeration, so expect the long build. When you fix one of the open W2-temporal bugs, replace its counterexample by a proof of the fixed behaviour (no `pre<PR>_` history is kept: note the fix in prose, and the last revision that has the counterexample), and flip the matching `bug_*` repro test.
