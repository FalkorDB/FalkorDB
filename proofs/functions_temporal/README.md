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

A duration is stored as "the instant 1970-01-01 plus the years and months plus the days and seconds", and its components are recovered by decomposing that instant. `construct_decompose` proves that construct ∘ decompose is the identity, which is the law `Duration ± Duration` relies on. `add_sub_months` proves that `(d + P1M) - P1M = d` for every day of the month up to 28. Since #2984 `+` and `-` share `shift_months` (`shiftMonths`), and `sub_eq_add_neg_months` proves `ts - P{Y}Y{M}M = ts + P{-Y}Y{-M}M` for every timestamp; `shiftMonths_total` shows the shift depends only on the total month count.

**3. Is ordering of temporal values a valid sort?**

Yes. Date, time and duration ordering is a total order (`temporal_order_total`, `temporal_order_trans`, `temporal_order_antisymm`), so `ORDER BY` over them can't trip Rust's sort check. A later day always has a larger timestamp (`dateTs_mono_day`).

**4. Can a temporal constructor crash the server?**

Yes, and three inputs that do so are confirmed (see [Bugs found](#bugs-found)). The project proves where the safe region is. `weekPath_total` shows that the week branch of `date({year, week, dayOfWeek})` never panics for `|week| ≤ 10⁶` and `|year| ≤ 200 000`. `week_panics` gives a `week` outside that range that does panic. The missing guard is C's "valid values 1 - 53".

**5. Do the parsers produce real dates?**

Yes:

- a successfully parsed date string is a valid calendar, ordinal or ISO-week date (`parseDate_ok`);
- the same holds for times (`parseTime_ok`);
- since #3125, compact strings with a non-digit are rejected (`parseDate_compact_rejects`, `parseTime_compact_rejects`);
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
| `weekPath_total` | The `week` constructor branch is panic-free on a sane domain | `Constructors.lean` |
| `parseDate_ok`, `parseTime_ok` | The string parsers only accept real dates and times | `Parse.lean` |
| `durationStruct_eq_map`, `dateStruct_eq_map` | The positional-slot constructor forms equal the map forms | `Pure.lean` |
| `distance_finite` | `distance()` is never NaN for finite points, antipodal ones included (#2983) | `SpatialFns.lean` |
| `timeAddDur_eq_tod`, `timeSub_timeAdd` | `localtime ± duration` is the time of day of the date-arithmetic result, and `-` undoes `+` (#2985) | `Duration.lean` |
| `dateLoop_days_i64` | The `duration('P…W…D')` day accumulator never wraps; overflow is an error (#2987) | `Parse.lean` |
| `colEval_eq`, `intLane_eq` | Columnar `AND`/`OR` and the int arithmetic lane equal per-row evaluation | `VectorExpr.lean` |
| `distance_none_iff`, `vecDist_kernel` | `vec.*` distance returns `null` exactly for an unknown metric or a length mismatch, and otherwise calls the right simsimd kernel | `VecDistance.lean` |

### How `dfc_civil` and `civil_dfc` are proved

1. **Model the functions.** `daysFromCivil` and `civilFromDaysI` are the Rust functions over `Int`, with `div_euclid`/`rem_euclid` as Lean's `/`/`%` (which agree for a positive divisor). Their sub-expressions get names (`yoeOf`, `doyOf`, `mpOf`, `dOf`, `dfcEra`), so proofs can talk about them.
2. **Split off the era.** Both algorithms first reduce to a position inside a 400-year era of 146 097 days. Outside the era the arithmetic is linear, and `omega` handles it.
3. **Check the era exhaustively.** Inside one era there are finitely many cases. `fwdOk n` checks that era day `n` decodes to an in-range month and a real day of that month. `bwdOk k` checks that a real date on the 148 800-point (year-of-era, month, day) grid encodes and decodes back to itself. Both are checked for *every* point by the Lean kernel with `decide +kernel`, split into 30 chunks across `EraFwd*.lean` and `EraBwd*.lean`. That takes about 10 CPU-minutes and doesn't use `native_decide`.
4. **Combine.** `era_fwd` and `era_bwd` lift the finite checks to statements about the era pieces, and `dfc_civil` and `civil_dfc` assemble them. `civil_dfc` needs `ValidDate y m d`: a month in 1..12 and a day in the month's range. Rust only ever calls it on such dates.

## Bugs found

Each bug has a Rust repro test (`graph/tests/lean_functions_temporal.rs`, run locally and not committed) and was compared live against C. See [`../FINDINGS.md`](../FINDINGS.md). Re-checked against main 8621212fc: #2983, #2984, #2985 and #2987 fixed W2-temporal-4, -5, -6 and -8; the others are still present.

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| W2-temporal-1 | `week_panics`, `week_panics_delta` | `RETURN date({year:2020, week:10000000000})` **crashes the server** (chrono `NaiveDate + TimeDelta` overflow). C: "Invalid value for week" | **Open** |
| W2-temporal-2 | `quarter_panics` | `date({year:2020, quarter:2, dayOfQuarter:9223372036854775807})` **crashes the server** (`TimeDelta::days` panics). C: error | **Open** |
| W2-temporal-3 | `week_slice_panics` | `RETURN date('2020W1é')` **crashes the server**: `&rest[..2]` is not on a char boundary. #3125 doesn't cover the `W` path (`parseDate_week_first`) | **Open** |
| W2-temporal-4 | `pre2983_distance_nan` | `distance()` between near-antipodal points was `NaN` (C: 20037518): `(1.0 - a).sqrt()` with `a` rounded above 1. 35 428 of 6 485 401 antipodal pairs on a 0.1° grid. Proven over an abstract float with the binary64 facts as hypotheses (`NanLaws`); the mechanism is also `#eval`'d in `Spatial.lean` | Fixed by #2983 (`distance_finite`: finite whenever the haversine term is not NaN) |
| W2-temporal-5 | `pre2984_sub_vs_add_neg` | `date('2020-03-31') - duration({months:1})` was 2020-02-29 (subtraction clamped the day), while `+ duration({months:-1})` was 2020-03-02 (C: 03-02) | Fixed by #2984 (`sub_eq_add_neg_months`: `ts - P{Y}Y{M}M = ts + P{-Y}Y{-M}M` for every timestamp; `sub_vs_add_neg`: both 03-02; `subDur_eq_pre2984`: unchanged for day ≤ 28) |
| W2-temporal-6 | `pre2985_time_not_normalised` | `localtime('01:00') + duration({days:1}) = localtime('01:00')` was `false` (C: `true`): time arithmetic wasn't reduced mod 86400 | Fixed by #2985 (`timeAddDur_eq_tod`, `timeSub_timeAdd`, `time_normalised`) |
| W2-temporal-7 | `formatDate_not_injective` | `toString(date({year:12345}))` is `"2345-01-01"` (C: `"12345-01-01"`); `formatDate_injective_4digit` proves it is right for 4-digit years | **Open** |
| W2-temporal-8 | `pre2987_week_string_wraps` | `duration('P2635249153387078803W')` was `P5D` in release (`n * 7` wrapped) and panicked in debug | Fixed by #2987 (`dateLoop_week_overflow`, `dateLoop_days_i64`: now an error, never a wrap) |
| W2-temporal-9 | `dayOfWeek_seven_rejected`, `dayOfWeek_zero_is_sunday` | The map form's `dayOfWeek` is 0..6 with 0 = Sunday, so `dayOfWeek:7` errors (C: accepted); the string form uses 1..7 | **Open** (divergence) |
| — | `neg_days_encoding` | `date('2020-03-01') + duration({days:-1})` is 2020-03-02 (spec: 02-29), because negative durations are encoded as an instant before 1970 | Shared with C |
| — | `civilFromDays_trunc` | `y - 1970` in `i32` inside `decompose_duration` overflows for durations whose year wraps | Suspected, not triggered in release |
| — | `point_null_lat_msgs` | `point({latitude:null, …})` gives two different error messages for a literal map and a map value | Message only |

The header also records conversion divergences found live: `toString(1.0)` is `"1"` (C `"1.000000"`), `toInteger('1e3')` is 1000 (C `null`), and `toFloat(' 1')` is `null` (C 1). They aren't modelled, since `f64` formatting and parsing are abstract.

This project has no CONDITIONAL rows.

## Files

| File | What's in it |
|---|---|
| `Calendar.lean` | `isLeap`, `daysInMonth`, both `days_from_civil` copies, `civil_from_days`, and the named era pieces |
| `EraDefs.lean`, `EraFwd0-3.lean`, `EraBwd0-3.lean`, `EraAll.lean` | The kernel-checked enumeration of one 400-year era (`fwdOk_all`, `bwdOk_all`) |
| `CalendarLaws.lean` | `dfc_civil`, `civil_dfc`, `civil_valid`, `dfc_injective`, the `i32` cast |
| `Duration.lean` | Duration construct/decompose, `add_duration_to_timestamp`/`sub_duration_from_timestamp`, `Time ± Duration`, temporal ordering; bug 5, history of 6 and 8 |
| `Constructors.lean` | The `date({week|quarter …})` branches, the `W` string slice, date formatting; bugs 1–3, 7, 9 |
| `Fns.lean` | Field extractors, `*_pure` constructors, the clock functions, `register` |
| `Parse.lean` | `parse_date_string`, `parse_time_string`, `parse_datetime_string`, `parse_duration_string` (checked `W`/`D` since #2987) |
| `Pure.lean` | The slot-based `*_struct_pure` constructors equal the map forms |
| `Components.lean` | Temporal accessors and formatting (`value.rs`), `Point` |
| `Spatial.lean`, `SpatialFns.lean` | `Point::distance` (haversine, clamped since #2983) and `spatial.rs`; the antipodal NaN, by evaluation and by proof over `NanLaws` |
| `VecDistance.lean` | `vec_distance.rs` over the simsimd kernel as a parameter |
| `VectorExpr.lean` | Columnar `AND`/`OR` and the int arithmetic lane against per-row evaluation |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

The project is self-contained: its `lakefile.toml` has no dependencies, and it does not import `../core`.

## Coverage

`COVERAGE.tsv` has 100 rows: 31 in `temporal.rs`, 26 in `vector_expr.rs`, 21 in `value.rs`, 11 in `conversion.rs`, 7 in `spatial.rs` and 4 in `vec_distance.rs`. Two `value.rs` rows are arm rows (the `Time ± Duration` arms of `add_slow` and `Sub::sub`), pinned to the enclosing function.

| Bucket | Rows |
|---|---|
| PROVEN | 63 |
| MODELLED | 3 |
| NOT COVERED | 34 |

- **MODELLED:** `ExprColumn::get`, `union_nulls` and `eval_per_row`, which this project's columnar proof uses as its reference but doesn't state a theorem about.
- **NOT COVERED (`vector_expr.rs`, 20 rows):** the rest of the columnar evaluator: dispatch, `CASE`, comparisons, the float lane and bitmap plumbing. All of these are proven in [`../columnar`](../columnar) and [`../expr_semantics`](../expr_semantics).
- **NOT COVERED (`conversion.rs`, 11 rows):** `f64` parsing and formatting, checked live against C only. `value_math` proves most of these.
- **NOT COVERED (`value.rs`, 3 rows):** `Point::validate` (f32 range checks), `format_duration` and `get_duration_component`, all live-tested.

`bash proofs/coverage.sh` counts each function once, in its best bucket across all projects.

## What the model does not cover

- **Floats.** `f64` is abstract. `toString` and `toInteger`/`toFloat` parsing are only live-tested. The antipodal-NaN theorems assume the binary64 facts they use (`NanLaws`: `f64::clamp`, `sqrt`, `atan2`, NaN propagation) rather than deriving them, and `mul_add`'s fused rounding is not modelled.
- **Chrono.** Formatting, `from_ymd_opt`, `from_isoywd_opt`, the component accessors and `Utc::now` are trusted, as structures (`Chrono`, `ChronoDT`) whose laws come from chrono's docs.
- **Overflow.** `i32`/`i64` overflow in `add_duration_to_timestamp`/`sub_duration_from_timestamp` is outside the proven range (`SmallYear`).
- **Columnar coverage.** The columnar proof here covers `AND`/`OR` and the int lane only, not `CASE`, comparisons or the float lane (those are in `columnar` and `expr_semantics`). Which row's error message a batch reports isn't modelled.
- **simsimd.** The `vec.*` kernels are an FFI parameter.

## Build and check

```bash
cd proofs/functions_temporal && lake build    # this project (the era check takes ~10 CPU-minutes)
bash proofs/lean_ci.sh functions_temporal     # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                       # repo-wide coverage report
```

**When you change `functions/temporal.rs`, `spatial.rs`, `vec_distance.rs` or the calendar and duration code in `value.rs`:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match.
3. Re-run the build.

A change to `days_from_civil` or `civil_from_days` re-runs the era enumeration, so expect the long build. When you fix one of the W2-temporal bugs, rename its counterexample to `pre<PR>_…`, prove the fixed behaviour, and flip the matching `bug_*` repro test.
