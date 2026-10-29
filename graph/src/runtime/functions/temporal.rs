//! Temporal constructor functions.
//!
//! Constructs Cypher temporal values from maps or ISO 8601 strings.
//! All temporal types are stored internally as `i64` Unix timestamps
//! (seconds from epoch).
//!
//! ```text
//!  Cypher                 Function          Input formats
//! ────────────────────────────────────────────────────────────────────
//!  timestamp()            timestamp_fn()    (none) -> current millis
//!  date(x)                date_fn()         Map{year,month,day,week,quarter,...}
//!                                           String "YYYY-MM-DD", "YYYYMMDD",
//!                                                  "YYYY-Www-D", "YYYYDDD", ...
//!  localtime(x)           localtime_fn()    Map{hour,minute,second}
//!                                           String "HH:MM:SS", "HHMMSS", ...
//!  localdatetime(x)       localdatetime_fn() Map (date + time fields combined)
//!                                           String "YYYY-MM-DDThh:mm:ss"
//!  duration(x)            duration_fn()     Map{years,months,weeks,days,hours,...}
//!                                           String "P[nY][nM][nD]T[nH][nM][nS]"
//! ```
//!
//! ## Internal representation
//!
//! ```text
//!  Value::Date(ts)      -- seconds at midnight UTC of that date
//!  Value::Time(ts)      -- seconds since epoch for 1970-01-01 + time
//!  Value::Datetime(ts)  -- full UTC timestamp in seconds
//!  Value::Duration(p)   -- signed months, days and seconds packed in an i64
//! ```
//!
//! A duration keeps its months, days and seconds as independent signed
//! components (openCypher / Neo4j semantics); see `pack_duration`.
//!
//! ## Date parsing modes
//!
//! The `date()` constructor supports calendar dates (YYYY-MM-DD),
//! ordinal dates (YYYYDDD), ISO week dates (YYYY-Www-D), and
//! quarter dates (via `quarter` + `dayOfQuarter` map fields).

#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use super::{FnType, Functions, Type};
use crate::runtime::{ordermap::OrderMap, runtime::Runtime, value::Value};
use chrono::{Datelike, NaiveDate, NaiveDateTime, NaiveTime, Utc};
use std::sync::Arc;

fn get_int_field(
    map: &OrderMap<Arc<String>, Value>,
    field: &str,
) -> Option<i64> {
    map.get_str(field).and_then(|v| match v {
        Value::Int(i) => Some(*i),
        Value::Float(f) => Some(*f as i64),
        _ => None,
    })
}

/// Validate an optional date component against its inclusive range, with the
/// same message as the C implementation.
fn check_range(
    name: &str,
    value: Option<i64>,
    min: i64,
    max: i64,
) -> Result<(), String> {
    match value {
        Some(v) if !(min..=max).contains(&v) => Err(format!(
            "Invalid value for {name} (valid values {min} - {max})"
        )),
        _ => Ok(()),
    }
}

/// Build a NaiveDate from optional components. Supports ymd, week, and quarter modes.
///
/// Every component is range-checked before it reaches chrono: chrono's
/// `NaiveDate + TimeDelta` and `TimeDelta::days` panic on overflow, and a
/// panic here takes the whole server down.
fn date_from_components(
    year: Option<i64>,
    month: Option<i64>,
    day: Option<i64>,
    week: Option<i64>,
    day_of_week: Option<i64>,
    quarter: Option<i64>,
    day_of_quarter: Option<i64>,
) -> Result<NaiveDate, String> {
    let year = year.unwrap_or(1970);
    let year = i32::try_from(year).map_err(|_| format!("Invalid year: {year}"))?;

    if let Some(week) = week {
        check_range("week", Some(week), 1, 53)?;
        // ISO day of week, as in the string form and C: 1 = Monday .. 7 = Sunday
        check_range("dayOfWeek", day_of_week, 1, 7)?;
        let day_offset = day_of_week.unwrap_or(1) - 1;
        let jan4 =
            NaiveDate::from_ymd_opt(year, 1, 4).ok_or_else(|| format!("Invalid year: {year}"))?;
        let weekday_of_jan4 = jan4.weekday().num_days_from_monday();
        // week is in 1..=53, so the offset is at most 370 days
        let days = (week - 1) * 7 + day_offset - i64::from(weekday_of_jan4);
        return jan4
            .checked_add_signed(chrono::Duration::days(days))
            .ok_or_else(|| format!("Invalid date: year={year}, week={week}"));
    }

    if let Some(quarter) = quarter {
        check_range("quarter", Some(quarter), 1, 4)?;
        check_range("dayOfQuarter", day_of_quarter, 1, 92)?;
        let doq = day_of_quarter.unwrap_or(1);
        let quarter_start_month = ((quarter - 1) * 3 + 1) as u32;
        let base = NaiveDate::from_ymd_opt(year, quarter_start_month, 1)
            .ok_or_else(|| format!("Invalid quarter: year={year}, quarter={quarter}"))?;
        return base
            .checked_add_signed(chrono::Duration::days(doq - 1))
            .ok_or_else(|| format!("Invalid dayOfQuarter: {doq}"));
    }

    let month = month.unwrap_or(1);
    let day = day.unwrap_or(1);
    u32::try_from(month)
        .ok()
        .zip(u32::try_from(day).ok())
        .and_then(|(m, d)| NaiveDate::from_ymd_opt(year, m, d))
        .ok_or_else(|| format!("Invalid date: year={year}, month={month}, day={day}"))
}

fn date_from_map(map: &OrderMap<Arc<String>, Value>) -> Result<NaiveDate, String> {
    date_from_components(
        get_int_field(map, "year"),
        get_int_field(map, "month"),
        get_int_field(map, "day"),
        get_int_field(map, "week"),
        get_int_field(map, "dayOfWeek"),
        get_int_field(map, "quarter"),
        get_int_field(map, "dayOfQuarter"),
    )
}

fn time_from_components(
    hour: Option<i64>,
    minute: Option<i64>,
    second: Option<i64>,
) -> Result<NaiveTime, String> {
    let hour = hour.unwrap_or(0) as u32;
    let minute = minute.unwrap_or(0) as u32;
    let second = second.unwrap_or(0) as u32;
    NaiveTime::from_hms_opt(hour, minute, second)
        .ok_or_else(|| format!("Invalid time: hour={hour}, minute={minute}, second={second}"))
}

fn time_from_map(map: &OrderMap<Arc<String>, Value>) -> Result<NaiveTime, String> {
    time_from_components(
        get_int_field(map, "hour"),
        get_int_field(map, "minute"),
        get_int_field(map, "second"),
    )
}

/// Extract i64 from positional slot arg (Int/Float/Null).
#[inline]
const fn slot_to_int(v: &Value) -> Option<i64> {
    match v {
        Value::Int(i) => Some(*i),
        Value::Float(f) => Some(*f as i64),
        _ => None,
    }
}

// Parse date from ISO string formats.
fn parse_date_string(s: &str) -> Result<NaiveDate, String> {
    // Try formats with hyphens first
    if s.contains('W') {
        return parse_week_date(s);
    }
    if s.starts_with('-') {
        // Negative year - not supported
        return Err(format!("Unsupported date string: {s}"));
    }
    let digits_only: String = s.chars().filter(char::is_ascii_digit).collect();
    let has_hyphens = s.contains('-');

    if has_hyphens {
        // YYYY-MM-DD or YYYY-MM
        let parts: Vec<&str> = s.split('-').collect();
        match parts.len() {
            1 => {
                let year: i32 = parts[0].parse().map_err(|_| format!("Invalid year: {s}"))?;
                NaiveDate::from_ymd_opt(year, 1, 1).ok_or_else(|| format!("Invalid date: {s}"))
            }
            2 => {
                let year: i32 = parts[0].parse().map_err(|_| format!("Invalid year: {s}"))?;
                let month: u32 = parts[1]
                    .parse()
                    .map_err(|_| format!("Invalid month: {s}"))?;
                NaiveDate::from_ymd_opt(year, month, 1).ok_or_else(|| format!("Invalid date: {s}"))
            }
            3 => {
                let year: i32 = parts[0].parse().map_err(|_| format!("Invalid year: {s}"))?;
                let month: u32 = parts[1]
                    .parse()
                    .map_err(|_| format!("Invalid month: {s}"))?;
                let day: u32 = parts[2].parse().map_err(|_| format!("Invalid day: {s}"))?;
                NaiveDate::from_ymd_opt(year, month, day)
                    .ok_or_else(|| format!("Invalid date: {s}"))
            }
            _ => Err(format!("Invalid date string: {s}")),
        }
    } else {
        // Compact formats: YYYY, YYYYMM, YYYYDDD, YYYYMMDD.
        // `digits_only` drops non-digits rather than rejecting them, so
        // `date('2o15o7o21')` would otherwise dispatch on `20150721` and answer
        // 2015-07-21. Every documented compact form is already all digits, so
        // nothing valid is lost by refusing the rest. A closure, not
        // `char::is_ascii_digit` directly: that takes `&char`, and `all` hands
        // the predicate a `char` by value (`filter`, above, is handed a `&char`,
        // which is why it can be written bare).
        if !s.chars().all(|c| c.is_ascii_digit()) {
            return Err(format!("Invalid date string: {s}"));
        }
        match digits_only.len() {
            4 => {
                let year: i32 = digits_only
                    .parse()
                    .map_err(|_| format!("Invalid year: {s}"))?;
                NaiveDate::from_ymd_opt(year, 1, 1).ok_or_else(|| format!("Invalid date: {s}"))
            }
            6 => {
                let year: i32 = digits_only[..4]
                    .parse()
                    .map_err(|_| format!("Invalid year: {s}"))?;
                let month: u32 = digits_only[4..6]
                    .parse()
                    .map_err(|_| format!("Invalid month: {s}"))?;
                NaiveDate::from_ymd_opt(year, month, 1).ok_or_else(|| format!("Invalid date: {s}"))
            }
            7 => {
                // YYYYDDD - ordinal day
                let year: i32 = digits_only[..4]
                    .parse()
                    .map_err(|_| format!("Invalid year: {s}"))?;
                let ordinal: u32 = digits_only[4..7]
                    .parse()
                    .map_err(|_| format!("Invalid ordinal: {s}"))?;
                NaiveDate::from_yo_opt(year, ordinal)
                    .ok_or_else(|| format!("Invalid ordinal date: {s}"))
            }
            8 => {
                let year: i32 = digits_only[..4]
                    .parse()
                    .map_err(|_| format!("Invalid year: {s}"))?;
                let month: u32 = digits_only[4..6]
                    .parse()
                    .map_err(|_| format!("Invalid month: {s}"))?;
                let day: u32 = digits_only[6..8]
                    .parse()
                    .map_err(|_| format!("Invalid day: {s}"))?;
                NaiveDate::from_ymd_opt(year, month, day)
                    .ok_or_else(|| format!("Invalid date: {s}"))
            }
            _ => Err(format!("Invalid date string: {s}")),
        }
    }
}

fn parse_week_date(s: &str) -> Result<NaiveDate, String> {
    // Formats: YYYY-Www, YYYY-Www-D, YYYYWww, YYYYWwwD
    let s = s.replace('-', "");
    let w_pos = s
        .find('W')
        .ok_or_else(|| format!("Invalid week date: {s}"))?;
    let year: i32 = s[..w_pos]
        .parse()
        .map_err(|_| format!("Invalid year in week date: {s}"))?;
    let rest = &s[w_pos + 1..];

    let (week, dow) = if rest.len() <= 2 {
        let week: u32 = rest.parse().map_err(|_| format!("Invalid week: {s}"))?;
        (week, chrono::Weekday::Mon)
    } else {
        // `split_at_checked` rather than `rest[..2]`: byte 2 may fall inside a
        // multi-byte character, and slicing there panics.
        let (week, d) = rest
            .split_at_checked(2)
            .ok_or_else(|| format!("Invalid week: {s}"))?;
        let week: u32 = week.parse().map_err(|_| format!("Invalid week: {s}"))?;
        let d: u32 = d.parse().map_err(|_| format!("Invalid day of week: {s}"))?;
        let weekday = match d {
            1 => chrono::Weekday::Mon,
            2 => chrono::Weekday::Tue,
            3 => chrono::Weekday::Wed,
            4 => chrono::Weekday::Thu,
            5 => chrono::Weekday::Fri,
            6 => chrono::Weekday::Sat,
            7 => chrono::Weekday::Sun,
            _ => return Err(format!("Invalid day of week: {d}")),
        };
        (week, weekday)
    };

    NaiveDate::from_isoywd_opt(year, week, dow)
        .ok_or_else(|| format!("Invalid ISO week date: year={year}, week={week}"))
}

// Parse time from string formats.
fn parse_time_string(s: &str) -> Result<NaiveTime, String> {
    // Strip fractional part
    let s = s.split('.').next().unwrap_or(s);
    let has_colons = s.contains(':');

    if has_colons {
        let parts: Vec<&str> = s.split(':').collect();
        let hour: u32 = parts[0].parse().map_err(|_| format!("Invalid hour: {s}"))?;
        let minute: u32 = if parts.len() > 1 {
            parts[1]
                .parse()
                .map_err(|_| format!("Invalid minute: {s}"))?
        } else {
            0
        };
        let second: u32 = if parts.len() > 2 {
            parts[2]
                .parse()
                .map_err(|_| format!("Invalid second: {s}"))?
        } else {
            0
        };
        NaiveTime::from_hms_opt(hour, minute, second).ok_or_else(|| format!("Invalid time: {s}"))
    } else {
        let digits: String = s.chars().filter(char::is_ascii_digit).collect();
        // Same as the compact date branch: `localtime('xx123456yy')` must not
        // parse as 12:34:56. The fraction was already stripped above.
        if !s.chars().all(|c| c.is_ascii_digit()) {
            return Err(format!("Invalid time string: {s}"));
        }
        match digits.len() {
            2 => {
                let hour: u32 = digits.parse().map_err(|_| format!("Invalid hour: {s}"))?;
                NaiveTime::from_hms_opt(hour, 0, 0).ok_or_else(|| format!("Invalid time: {s}"))
            }
            4 => {
                let hour: u32 = digits[..2]
                    .parse()
                    .map_err(|_| format!("Invalid hour: {s}"))?;
                let min: u32 = digits[2..4]
                    .parse()
                    .map_err(|_| format!("Invalid minute: {s}"))?;
                NaiveTime::from_hms_opt(hour, min, 0).ok_or_else(|| format!("Invalid time: {s}"))
            }
            6 => {
                let hour: u32 = digits[..2]
                    .parse()
                    .map_err(|_| format!("Invalid hour: {s}"))?;
                let min: u32 = digits[2..4]
                    .parse()
                    .map_err(|_| format!("Invalid minute: {s}"))?;
                let sec: u32 = digits[4..6]
                    .parse()
                    .map_err(|_| format!("Invalid second: {s}"))?;
                NaiveTime::from_hms_opt(hour, min, sec).ok_or_else(|| format!("Invalid time: {s}"))
            }
            _ => Err(format!("Invalid time string: {s}")),
        }
    }
}

// Parse ISO 8601 datetime string.
fn parse_datetime_string(s: &str) -> Result<NaiveDateTime, String> {
    // Split on T to separate date and time
    let (date_part, time_part) = s
        .find('T')
        .map_or((s, None), |t_pos| (&s[..t_pos], Some(&s[t_pos + 1..])));

    let date = parse_date_string(date_part)?;

    let time = if let Some(tp) = time_part {
        parse_time_string(tp)?
    } else {
        NaiveTime::from_hms_opt(0, 0, 0).unwrap()
    };

    Ok(NaiveDateTime::new(date, time))
}

// Parse ISO 8601 duration string: P[nY][nM][nD][T[nH][nM][nS]]
fn parse_duration_string(s: &str) -> Result<(i64, i64, i64, i64, i64, i64, i64), String> {
    let s = s
        .strip_prefix('P')
        .ok_or_else(|| format!("Duration string must start with 'P': {s}"))?;

    let (date_part, time_part) = s
        .find('T')
        .map_or((s, None), |t_pos| (&s[..t_pos], Some(&s[t_pos + 1..])));

    let mut years = 0i64;
    let mut months = 0i64;
    let mut days = 0i64;
    let mut hours = 0i64;
    let mut minutes = 0i64;
    let mut seconds = 0i64;

    // Parse date part
    let mut num_buf = String::new();
    for ch in date_part.chars() {
        if ch.is_ascii_digit() || ch == '-' {
            num_buf.push(ch);
        } else {
            let n: i64 = num_buf
                .parse()
                .map_err(|_| format!("Invalid number in duration: {num_buf}"))?;
            num_buf.clear();
            match ch {
                'Y' => years = n,
                'M' => months = n,
                'W' => {
                    days = n
                        .checked_mul(7)
                        .and_then(|w| days.checked_add(w))
                        .ok_or_else(|| format!("Duration overflow: weeks={n}"))?;
                }
                'D' => {
                    days = days
                        .checked_add(n)
                        .ok_or_else(|| format!("Duration overflow: days={n}"))?;
                }
                _ => return Err(format!("Unknown duration component: {ch}")),
            }
        }
    }

    // Parse time part
    if let Some(tp) = time_part {
        num_buf.clear();
        for ch in tp.chars() {
            if ch.is_ascii_digit() || ch == '-' || ch == '.' {
                num_buf.push(ch);
            } else {
                let n: i64 = num_buf
                    .split('.')
                    .next()
                    .unwrap_or(&num_buf)
                    .parse()
                    .map_err(|_| format!("Invalid number in duration: {num_buf}"))?;
                num_buf.clear();
                match ch {
                    'H' => hours = n,
                    'M' => minutes = n,
                    'S' => seconds = n,
                    _ => return Err(format!("Unknown duration time component: {ch}")),
                }
            }
        }
    }

    Ok((years, months, 0, days, hours, minutes, seconds))
}

/// Days from 1970-01-01 to the given (year, month, day) — Howard Hinnant's civil-from-days.
/// Avoids chrono allocation/validation on the duration hot path.
#[inline]
const fn days_from_civil(
    y: i32,
    m: u32,
    d: u32,
) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u32; // [0, 399]
    let mp = if m > 2 { m - 3 } else { m + 9 }; // [0, 11]
    let doy = (153 * mp + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era as i64 * 146_097 + doe as i64 - 719_468
}

// A duration is three independent signed components, as in openCypher / Neo4j:
// months (years * 12 + months), days (weeks * 7 + days) and seconds
// (hours, minutes and seconds). No component ever borrows from or carries into
// another, so `-1 day` stays `-1 day` and `P1M-3D` stays `P1M-3D`.
//
// They are packed into the single `i64` of `Value::Duration` (8 bytes, like the
// C engine's `time_t`): months in the top 14 bits, days in the next 18 and
// seconds in the low 32, each two's complement. Every field is limited to a
// symmetric range so that negating a duration always fits.

const DUR_SEC_BITS: u32 = 32;
const DUR_DAY_BITS: u32 = 18;
const DUR_MONTH_BITS: u32 = 14;
/// Largest absolute value of the seconds component (about 68 years).
pub const DUR_MAX_SECONDS: i64 = (1 << (DUR_SEC_BITS - 1)) - 1;
/// Largest absolute value of the days component (about 358 years).
pub const DUR_MAX_DAYS: i64 = (1 << (DUR_DAY_BITS - 1)) - 1;
/// Largest absolute value of the months component (about 682 years).
pub const DUR_MAX_MONTHS: i64 = (1 << (DUR_MONTH_BITS - 1)) - 1;

/// Pack signed (months, days, seconds) into a `Value::Duration` payload.
pub fn pack_duration(
    months: i64,
    days: i64,
    seconds: i64,
) -> Result<i64, String> {
    if months.unsigned_abs() > DUR_MAX_MONTHS as u64
        || days.unsigned_abs() > DUR_MAX_DAYS as u64
        || seconds.unsigned_abs() > DUR_MAX_SECONDS as u64
    {
        return Err(format!(
            "Duration out of range: months={months}, days={days}, seconds={seconds}"
        ));
    }
    Ok((months << (DUR_DAY_BITS + DUR_SEC_BITS))
        | ((days & ((1 << DUR_DAY_BITS) - 1)) << DUR_SEC_BITS)
        | (seconds & ((1 << DUR_SEC_BITS) - 1)))
}

/// Unpack a `Value::Duration` payload into signed (months, days, seconds).
#[inline]
pub const fn unpack_duration(packed: i64) -> (i64, i64, i64) {
    let months = packed >> (DUR_DAY_BITS + DUR_SEC_BITS);
    let days = (packed << DUR_MONTH_BITS) >> (64 - DUR_DAY_BITS);
    let seconds = (packed << (DUR_MONTH_BITS + DUR_DAY_BITS)) >> (64 - DUR_SEC_BITS);
    (months, days, seconds)
}

/// Build a duration from the user-facing components, normalising them the way
/// Neo4j does: years into months, weeks into days, hours/minutes into seconds.
pub fn construct_duration(
    years: i64,
    months: i64,
    weeks: i64,
    days: i64,
    hours: i64,
    minutes: i64,
    seconds: i64,
) -> Result<i64, String> {
    let overflow = || {
        format!(
            "Duration overflow: years={years}, months={months}, weeks={weeks}, days={days}, hours={hours}, minutes={minutes}, seconds={seconds}"
        )
    };
    let total_months = years
        .checked_mul(12)
        .and_then(|y| y.checked_add(months))
        .ok_or_else(overflow)?;
    let total_days = weeks
        .checked_mul(7)
        .and_then(|w| w.checked_add(days))
        .ok_or_else(overflow)?;
    let total_seconds = hours
        .checked_mul(3600)
        .and_then(|h| minutes.checked_mul(60).and_then(|m| h.checked_add(m)))
        .and_then(|s| s.checked_add(seconds))
        .ok_or_else(overflow)?;
    pack_duration(total_months, total_days, total_seconds)
}

/// Convert a duration stored with the legacy encoding (seconds from the Unix
/// epoch of `epoch + years/months + days + time`, as written by the C engine
/// and by earlier Rust versions) into the packed component form. The legacy
/// decomposition is replayed exactly, so a stored value reads back as it did.
pub fn duration_from_legacy(legacy: i64) -> Result<i64, String> {
    let days = legacy.div_euclid(86400);
    let time_of_day = legacy.rem_euclid(86400);
    let (y, m, d) = crate::runtime::value::civil_from_days(days);
    let months = i64::from(y - 1970) * 12 + i64::from(m) - 1;
    let remaining = (i64::from(d) - 1) * 86400 + time_of_day;
    pack_duration(months, remaining / 86400, remaining % 86400)
}

/// The wire form of a duration: `(is_v2, payload)`. A duration the legacy
/// epoch-offset encoding holds losslessly is written in it, byte for byte what
/// the C engine writes; any other (negative components, a day count past the
/// month length, ...) is written as the packed form under `T_DURATION_V2`.
pub fn duration_wire(packed: i64) -> (bool, i64) {
    let legacy = duration_to_legacy(packed);
    if duration_from_legacy(legacy) == Ok(packed) {
        (false, legacy)
    } else {
        (true, packed)
    }
}

/// Encode a duration in the legacy epoch-offset form, which is what the compact
/// result-set protocol carries (C clients read it that way). Components that
/// cannot be expressed non-negatively borrow exactly as the C engine does.
pub fn duration_to_legacy(packed: i64) -> i64 {
    let (months, days, seconds) = unpack_duration(packed);
    let base_year = 1970 + months.div_euclid(12) as i32;
    let base_month = (months.rem_euclid(12) + 1) as u32;
    days_from_civil(base_year, base_month, 1) * 86400 + days * 86400 + seconds
}

// ---------------------------------------------------------------------------
// Pure (Runtime-free) constructors used by the optimizer's constant-folding
// pass. The Runtime-bound versions registered below delegate to these for the
// non-zero-arg branch.
// ---------------------------------------------------------------------------

pub fn date_pure(args: &[Value]) -> Result<Value, String> {
    match args.first() {
        Some(Value::Map(map)) => {
            let d = date_from_map(map)?;
            let ts = d.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp();
            Ok(Value::Date(ts))
        }
        Some(Value::String(s)) => {
            let d = parse_date_string(s)?;
            let ts = d.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp();
            Ok(Value::Date(ts))
        }
        Some(Value::Null) => Ok(Value::Null),
        _ => unreachable!(),
    }
}

pub fn localtime_pure(args: &[Value]) -> Result<Value, String> {
    match args.first() {
        Some(Value::Map(map)) => {
            let t = time_from_map(map)?;
            let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
            let dt = NaiveDateTime::new(epoch, t);
            Ok(Value::Time(dt.and_utc().timestamp()))
        }
        Some(Value::String(s)) => {
            let t = parse_time_string(s)?;
            let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
            let dt = NaiveDateTime::new(epoch, t);
            Ok(Value::Time(dt.and_utc().timestamp()))
        }
        Some(Value::Null) => Ok(Value::Null),
        _ => unreachable!(),
    }
}

pub fn localdatetime_pure(args: &[Value]) -> Result<Value, String> {
    match args.first() {
        Some(Value::Map(map)) => {
            let d = date_from_map(map)?;
            let t = time_from_map(map)?;
            let dt = NaiveDateTime::new(d, t);
            Ok(Value::Datetime(dt.and_utc().timestamp()))
        }
        Some(Value::String(s)) => {
            let dt = parse_datetime_string(s)?;
            Ok(Value::Datetime(dt.and_utc().timestamp()))
        }
        Some(Value::Null) => Ok(Value::Null),
        _ => unreachable!(),
    }
}

pub fn duration_pure(args: &[Value]) -> Result<Value, String> {
    match args.first() {
        Some(Value::Map(map)) => {
            let mut years = 0i64;
            let mut months = 0i64;
            let mut weeks = 0i64;
            let mut days = 0i64;
            let mut hours = 0i64;
            let mut minutes = 0i64;
            let mut seconds = 0i64;
            for (k, v) in map.iter() {
                let n = match v {
                    Value::Int(i) => *i,
                    Value::Float(f) => *f as i64,
                    _ => continue,
                };
                match k.as_str() {
                    "years" => years = n,
                    "months" => months = n,
                    "weeks" => weeks = n,
                    "days" => days = n,
                    "hours" => hours = n,
                    "minutes" => minutes = n,
                    "seconds" => seconds = n,
                    _ => {}
                }
            }
            let ts = construct_duration(years, months, weeks, days, hours, minutes, seconds)?;
            Ok(Value::Duration(ts))
        }
        Some(Value::String(s)) => {
            let (years, months, weeks, days, hours, minutes, seconds) = parse_duration_string(s)?;
            let ts = construct_duration(years, months, weeks, days, hours, minutes, seconds)?;
            Ok(Value::Duration(ts))
        }
        Some(Value::Null) => Ok(Value::Null),
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// Positional-slot constructors. The binder rewrites Map-literal calls with
// constant string keys (e.g. `duration({months: i})`) into invocations of these
// internal functions, avoiding per-row Map+Arc allocation. Slot order is
// fixed per constructor; missing fields are passed as Value::Null.
// ---------------------------------------------------------------------------

/// Slot order: [years, months, weeks, days, hours, minutes, seconds]
const DURATION_SLOTS: &[&str] = &[
    "years", "months", "weeks", "days", "hours", "minutes", "seconds",
];

/// Slot order: [year, month, day, week, dayOfWeek, quarter, dayOfQuarter]
const DATE_SLOTS: &[&str] = &[
    "year",
    "month",
    "day",
    "week",
    "dayOfWeek",
    "quarter",
    "dayOfQuarter",
];

/// Slot order: [hour, minute, second]
const LOCALTIME_SLOTS: &[&str] = &["hour", "minute", "second"];

/// Slot order: [year, month, day, week, dayOfWeek, quarter, dayOfQuarter, hour, minute, second]
const LOCALDATETIME_SLOTS: &[&str] = &[
    "year",
    "month",
    "day",
    "week",
    "dayOfWeek",
    "quarter",
    "dayOfQuarter",
    "hour",
    "minute",
    "second",
];

pub fn duration_struct_pure(args: &[Value]) -> Result<Value, String> {
    debug_assert_eq!(args.len(), 7);
    let years = slot_to_int(&args[0]).unwrap_or(0);
    let months = slot_to_int(&args[1]).unwrap_or(0);
    let weeks = slot_to_int(&args[2]).unwrap_or(0);
    let days = slot_to_int(&args[3]).unwrap_or(0);
    let hours = slot_to_int(&args[4]).unwrap_or(0);
    let minutes = slot_to_int(&args[5]).unwrap_or(0);
    let seconds = slot_to_int(&args[6]).unwrap_or(0);
    let ts = construct_duration(years, months, weeks, days, hours, minutes, seconds)?;
    Ok(Value::Duration(ts))
}

pub fn date_struct_pure(args: &[Value]) -> Result<Value, String> {
    debug_assert_eq!(args.len(), 7);
    let d = date_from_components(
        slot_to_int(&args[0]),
        slot_to_int(&args[1]),
        slot_to_int(&args[2]),
        slot_to_int(&args[3]),
        slot_to_int(&args[4]),
        slot_to_int(&args[5]),
        slot_to_int(&args[6]),
    )?;
    let ts = d.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp();
    Ok(Value::Date(ts))
}

pub fn localtime_struct_pure(args: &[Value]) -> Result<Value, String> {
    debug_assert_eq!(args.len(), 3);
    let t = time_from_components(
        slot_to_int(&args[0]),
        slot_to_int(&args[1]),
        slot_to_int(&args[2]),
    )?;
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
    let dt = NaiveDateTime::new(epoch, t);
    Ok(Value::Time(dt.and_utc().timestamp()))
}

pub fn localdatetime_struct_pure(args: &[Value]) -> Result<Value, String> {
    debug_assert_eq!(args.len(), 10);
    let d = date_from_components(
        slot_to_int(&args[0]),
        slot_to_int(&args[1]),
        slot_to_int(&args[2]),
        slot_to_int(&args[3]),
        slot_to_int(&args[4]),
        slot_to_int(&args[5]),
        slot_to_int(&args[6]),
    )?;
    let t = time_from_components(
        slot_to_int(&args[7]),
        slot_to_int(&args[8]),
        slot_to_int(&args[9]),
    )?;
    let dt = NaiveDateTime::new(d, t);
    Ok(Value::Datetime(dt.and_utc().timestamp()))
}

pub fn register(funcs: &mut Functions) {
    // ── timestamp() ──
    cypher_fn!(funcs, "timestamp",
        args: [],
        ret: Type::Int,
        non_deterministic,
        fn timestamp_fn(_, args) {
            debug_assert!(args.is_empty());
            let now = Utc::now();
            Ok(Value::Int(now.timestamp_millis()))
        }
    );

    // ── date() ──
    cypher_fn!(funcs, "date",
        args: [Type::Optional(Box::new(Type::union([
            Type::Map,
            Type::String,
            Type::Null,
        ])))],
        ret: Type::union([Type::Date, Type::Null]),
        non_deterministic,
        fn date_fn(_, args) {
            if args.is_empty() {
                // Zero args: return current date
                let now = Utc::now().date_naive();
                let ts = now.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp();
                Ok(Value::Date(ts))
            } else {
                date_pure(args)
            }
        }
    );

    // ── localtime() ──
    cypher_fn!(funcs, "localtime",
        args: [Type::Optional(Box::new(Type::union([
            Type::Map,
            Type::String,
            Type::Null,
        ])))],
        ret: Type::union([Type::Time, Type::Null]),
        non_deterministic,
        fn localtime_fn(_, args) {
            if args.is_empty() {
                // Zero args: return current local time
                let now = Utc::now().time();
                let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
                let dt = NaiveDateTime::new(epoch, now);
                let ts = dt.and_utc().timestamp();
                Ok(Value::Time(ts))
            } else {
                localtime_pure(args)
            }
        }
    );

    // ── localdatetime() ──
    cypher_fn!(funcs, "localdatetime",
        args: [Type::Optional(Box::new(Type::union([
            Type::Map,
            Type::String,
            Type::Null,
        ])))],
        ret: Type::union([Type::Datetime, Type::Null]),
        non_deterministic,
        fn localdatetime_fn(_, args) {
            if args.is_empty() {
                // Zero args: return current local datetime
                let now = Utc::now().naive_utc();
                let ts = now.and_utc().timestamp();
                Ok(Value::Datetime(ts))
            } else {
                localdatetime_pure(args)
            }
        }
    );

    // ── duration() ──
    cypher_fn!(funcs, "duration",
        args: [Type::union([Type::Map, Type::String, Type::Null])],
        ret: Type::union([Type::Duration, Type::Null]),
        fn duration_fn(_, args) {
            duration_pure(args)
        }
    );

    // ── date.transaction() ──
    cypher_fn!(funcs, "date.transaction",
        args: [],
        ret: Type::Date,
        non_deterministic,
        fn date_transaction_fn(rt, _args) {
            let now = rt.transaction_timestamp.date_naive();
            let ts = now.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp();
            Ok(Value::Date(ts))
        }
    );

    // ── localtime.transaction() ──
    cypher_fn!(funcs, "localtime.transaction",
        args: [],
        ret: Type::Time,
        non_deterministic,
        fn localtime_transaction_fn(rt, _args) {
            let now = rt.transaction_timestamp.time();
            let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
            let dt = NaiveDateTime::new(epoch, now);
            let ts = dt.and_utc().timestamp();
            Ok(Value::Time(ts))
        }
    );

    // ── localdatetime.transaction() ──
    cypher_fn!(funcs, "localdatetime.transaction",
        args: [],
        ret: Type::Datetime,
        non_deterministic,
        fn localdatetime_transaction_fn(rt, _args) {
            let now = rt.transaction_timestamp.naive_utc();
            let ts = now.and_utc().timestamp();
            Ok(Value::Datetime(ts))
        }
    );

    // Mark temporal constructors as safe to constant-fold when called with
    // concrete arguments, and attach the positional-slot form invoked by
    // eval.rs after the binder rewrites `duration({months: i})` into
    // positional children.
    let constructor_args = || vec![Type::union([Type::Map, Type::String, Type::Null])];
    funcs.set_pure_fn("date", date_pure, constructor_args());
    funcs.set_pure_fn("localtime", localtime_pure, constructor_args());
    funcs.set_pure_fn("localdatetime", localdatetime_pure, constructor_args());
    funcs.set_pure_fn("duration", duration_pure, constructor_args());
    funcs.set_struct_fn("date", date_struct_pure, DATE_SLOTS);
    funcs.set_struct_fn("localtime", localtime_struct_pure, LOCALTIME_SLOTS);
    funcs.set_struct_fn(
        "localdatetime",
        localdatetime_struct_pure,
        LOCALDATETIME_SLOTS,
    );
    funcs.set_struct_fn("duration", duration_struct_pure, DURATION_SLOTS);
}

#[cfg(test)]
mod tests {
    use super::{date_pure, date_struct_pure, localdatetime_pure};
    use crate::runtime::{ordermap::OrderMap, value::Value};
    use std::sync::Arc;

    fn map(entries: &[(&str, i64)]) -> Value {
        Value::Map(Arc::new(OrderMap::from_vec(
            entries
                .iter()
                .map(|(k, v)| (Arc::new((*k).to_string()), Value::Int(*v)))
                .collect(),
        )))
    }

    fn date_slots(
        year: i64,
        week: Option<i64>,
        quarter: Option<i64>,
        day_of_quarter: Option<i64>,
    ) -> Vec<Value> {
        let slot = |v: Option<i64>| v.map_or(Value::Null, Value::Int);
        vec![
            Value::Int(year),
            Value::Null,
            Value::Null,
            slot(week),
            Value::Null,
            slot(quarter),
            slot(day_of_quarter),
        ]
    }

    // Out-of-range week / quarter / dayOfQuarter components used to panic in
    // chrono's `NaiveDate + TimeDelta` / `TimeDelta::days` and crash the server.
    #[test]
    fn date_out_of_range_components_error_instead_of_panicking() {
        for week in [
            0,
            54,
            10_000_000_000,
            i64::MAX,
            i64::MIN,
            -9_223_372_036_854_775_807,
        ] {
            assert!(date_pure(&[map(&[("year", 2020), ("week", week)])]).is_err());
            assert!(date_struct_pure(&date_slots(2020, Some(week), None, None)).is_err());
            assert!(localdatetime_pure(&[map(&[("year", 2020), ("week", week)])]).is_err());
        }
        for doq in [0, 93, 200, i64::MAX, i64::MIN] {
            assert!(
                date_pure(&[map(&[
                    ("year", 2020),
                    ("quarter", 2),
                    ("dayOfQuarter", doq)
                ])])
                .is_err()
            );
            assert!(date_struct_pure(&date_slots(2020, None, Some(2), Some(doq))).is_err());
        }
        for quarter in [0, 5, i64::MAX, i64::MIN, 4_294_967_297] {
            assert!(date_pure(&[map(&[("year", 2020), ("quarter", quarter)])]).is_err());
        }
        // Years outside chrono's range, at both ends, with week arithmetic that
        // would step past the representable range.
        for year in [262_143, -262_144, 262_144, i64::MAX, 4_294_969_316] {
            for week in [1, 53] {
                let _ = date_pure(&[map(&[("year", year), ("week", week)])]);
            }
        }
        assert!(date_pure(&[map(&[("year", 4_294_969_316)])]).is_err());
    }

    #[test]
    fn date_week_string_with_multibyte_char_errors_instead_of_panicking() {
        for s in ["2020W1é", "2020-W1é", "2020Wé1", "2020W€", "2020W12é"] {
            assert!(date_pure(&[Value::String(Arc::new(s.to_string()))]).is_err());
        }
    }

    // dayOfWeek is ISO 1..=7 (7 = Sunday), like the string form and C.
    #[test]
    fn date_day_of_week_is_iso() {
        let sunday = date_pure(&[map(&[("year", 2021), ("week", 1), ("dayOfWeek", 7)])]).unwrap();
        assert_eq!(
            sunday,
            date_pure(&[Value::String(Arc::new("2021-W01-7".to_string()))]).unwrap()
        );
        let monday = date_pure(&[map(&[("year", 2021), ("week", 1), ("dayOfWeek", 1)])]).unwrap();
        assert_eq!(
            monday,
            date_pure(&[Value::String(Arc::new("2021-W01-1".to_string()))]).unwrap()
        );
        for dow in [0, 8, -1] {
            assert!(date_pure(&[map(&[("year", 2021), ("week", 1), ("dayOfWeek", dow)])]).is_err());
        }
    }

    #[test]
    fn date_in_range_components_still_work() {
        let d = |y: i32, m: u32, day: u32| {
            Value::Date(
                chrono::NaiveDate::from_ymd_opt(y, m, day)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap()
                    .and_utc()
                    .timestamp(),
            )
        };
        assert_eq!(
            date_pure(&[map(&[("year", 1984), ("week", 10)])]).unwrap(),
            d(1984, 3, 5)
        );
        assert_eq!(
            date_pure(&[map(&[("year", 1918), ("week", 53)])]).unwrap(),
            d(1918, 12, 30)
        );
        assert_eq!(
            date_pure(&[map(&[("year", 1984), ("quarter", 3), ("dayOfQuarter", 45)])]).unwrap(),
            d(1984, 8, 14)
        );
        assert_eq!(
            date_pure(&[map(&[("year", 1984), ("quarter", 4), ("dayOfQuarter", 92)])]).unwrap(),
            d(1984, 12, 31)
        );
        assert_eq!(
            date_pure(&[Value::String(Arc::new("2015-W30-2".to_string()))]).unwrap(),
            d(2015, 7, 21)
        );
    }
}

#[cfg(test)]
mod compact_parser_rejects_junk_tests {
    use super::{parse_date_string, parse_time_string};
    use chrono::{Datelike, Timelike};

    // The compact parsers built their digit buffer with
    // `filter(char::is_ascii_digit)`, which silently dropped every other
    // character, so junk around a valid date or time parsed as the valid one.
    #[test]
    fn compact_date_rejects_embedded_letters() {
        assert!(parse_date_string("2o15o7o21").is_err());
        assert!(parse_date_string("hello20230506world").is_err());
    }

    #[test]
    fn compact_date_rejects_other_punctuation() {
        assert!(parse_date_string("+2015").is_err());
        assert!(parse_date_string("2015 07 21").is_err());
    }

    #[test]
    fn compact_date_keeps_every_documented_form() {
        let y = parse_date_string("2015").unwrap();
        assert_eq!((y.year(), y.month(), y.day()), (2015, 1, 1));
        let ym = parse_date_string("201507").unwrap();
        assert_eq!((ym.year(), ym.month(), ym.day()), (2015, 7, 1));
        let ymd = parse_date_string("20150721").unwrap();
        assert_eq!((ymd.year(), ymd.month(), ymd.day()), (2015, 7, 21));
        let ord = parse_date_string("2015202").unwrap();
        assert_eq!((ord.year(), ord.month(), ord.day()), (2015, 7, 21));
    }

    #[test]
    fn the_hyphenated_and_week_forms_are_untouched() {
        let ym = parse_date_string("2015-07").unwrap();
        assert_eq!((ym.year(), ym.month(), ym.day()), (2015, 7, 1));
        let ymd = parse_date_string("2015-07-21").unwrap();
        assert_eq!((ymd.year(), ymd.month(), ymd.day()), (2015, 7, 21));
        assert!(parse_date_string("2015-W30-2").is_ok());
    }

    #[test]
    fn colon_less_time_rejects_embedded_letters() {
        assert!(parse_time_string("xx123456yy").is_err());
    }

    #[test]
    fn colon_less_time_keeps_every_documented_form() {
        assert_eq!(parse_time_string("21").unwrap().hour(), 21);
        assert_eq!(parse_time_string("2140").unwrap().minute(), 40);
        assert_eq!(parse_time_string("214032").unwrap().second(), 32);
        // The fraction is stripped before the digits are inspected.
        assert_eq!(parse_time_string("214032.142").unwrap().second(), 32);
    }

    #[test]
    fn the_colon_time_form_is_untouched() {
        let t = parse_time_string("21:40:32").unwrap();
        assert_eq!((t.hour(), t.minute(), t.second()), (21, 40, 32));
    }
}

#[cfg(test)]
mod duration_tests {
    use super::{
        DUR_MAX_DAYS, DUR_MAX_MONTHS, DUR_MAX_SECONDS, construct_duration, duration_from_legacy,
        duration_to_legacy, duration_wire, pack_duration, unpack_duration,
    };
    use crate::runtime::value::Value;

    fn dur(
        months: i64,
        days: i64,
        secs: i64,
    ) -> i64 {
        pack_duration(months, days, secs).unwrap()
    }

    #[test]
    fn components_round_trip_with_every_sign() {
        for m in [-DUR_MAX_MONTHS, -13, -1, 0, 1, 13, DUR_MAX_MONTHS] {
            for d in [-DUR_MAX_DAYS, -3, 0, 3, DUR_MAX_DAYS] {
                for s in [-DUR_MAX_SECONDS, -86_401, -1, 0, 1, 86_401, DUR_MAX_SECONDS] {
                    assert_eq!(unpack_duration(dur(m, d, s)), (m, d, s));
                }
            }
        }
        assert_eq!(dur(0, 0, 0), 0);
        assert!(pack_duration(DUR_MAX_MONTHS + 1, 0, 0).is_err());
        assert!(pack_duration(0, -DUR_MAX_DAYS - 1, 0).is_err());
        assert!(pack_duration(0, 0, DUR_MAX_SECONDS + 1).is_err());
    }

    #[test]
    fn constructor_never_borrows_between_units() {
        let p = construct_duration(0, 0, 0, -1, 0, 0, 0).unwrap();
        assert_eq!(unpack_duration(p), (0, -1, 0));
        let p = construct_duration(1, -14, 1, 0, 1, -90, 0).unwrap();
        assert_eq!(unpack_duration(p), (-2, 7, -1800));
    }

    #[test]
    fn printing_uses_each_components_own_sign() {
        for (m, d, s, want) in [
            (0, -1, 0, "P-1D"),
            (1, -3, 0, "P1M-3D"),
            (0, 0, -3600, "PT-1H"),
            (0, 0, -5400, "PT-1H-30M"),
            (-13, 0, 0, "P-1Y-1M"),
            (149, -14, 57_600, "P12Y5M-14DT16H"),
            (0, 0, 90_000, "PT25H"),
            (0, 0, 0, "PT0S"),
        ] {
            assert_eq!(Value::format_duration(dur(m, d, s)), want);
        }
    }

    #[test]
    fn issue_2745_rows() {
        let a = Value::Duration(construct_duration(0, 0, 0, 2, 0, 0, 0).unwrap());
        let b = Value::Duration(construct_duration(0, 0, 0, -3, 0, 0, 0).unwrap());
        let Ok(Value::Duration(sum)) = a + b else {
            panic!()
        };
        assert_eq!(Value::format_duration(sum), "P-1D");
        // 2020-03-01 + (-1 day) = 2020-02-29
        let date = Value::Date(1_583_020_800);
        let Ok(Value::Date(d)) = date + Value::Duration(sum) else {
            panic!()
        };
        assert_eq!(Value::format_date(d), "2020-02-29");
    }

    #[test]
    fn add_and_sub_a_negative_duration_are_calendar_correct() {
        let date = Value::Date(1_583_020_800); // 2020-03-01
        let minus_month = Value::Duration(dur(-1, 0, 0));
        let Ok(Value::Date(d)) = date.clone() + minus_month.clone() else {
            panic!()
        };
        assert_eq!(Value::format_date(d), "2020-02-01");
        let Ok(Value::Date(d)) = date - minus_month else {
            panic!()
        };
        assert_eq!(Value::format_date(d), "2020-04-01");
        let Ok(Value::Datetime(t)) =
            Value::Datetime(1_583_056_800) + Value::Duration(dur(0, -1, -39_600))
        else {
            panic!()
        };
        assert_eq!(Value::format_datetime(t), "2020-02-28T23:00:00");
    }

    #[test]
    fn duration_difference_keeps_components() {
        let Ok(Value::Duration(p)) = Value::Duration(dur(1, 0, 0)) - Value::Duration(dur(0, 3, 0))
        else {
            panic!()
        };
        assert_eq!(Value::format_duration(p), "P1M-3D");
    }

    #[test]
    fn legacy_encoding_converts_losslessly_where_canonical() {
        // Round trip for everything the legacy form holds.
        for p in [
            dur(0, 0, 0),
            dur(13, 2, 3_661),
            dur(0, 1, 0),
            dur(1, 0, 86_399),
        ] {
            let legacy = duration_to_legacy(p);
            assert_eq!(duration_from_legacy(legacy), Ok(p));
            assert_eq!(duration_wire(p), (false, legacy));
        }
        // What C stored for P1D / P1Y1M1DT1H1M1S.
        assert_eq!(duration_from_legacy(86_400), Ok(dur(0, 1, 0)));
        // Components the legacy form cannot hold use the packed tag.
        for p in [dur(0, -1, 0), dur(0, 40, 0), dur(0, 0, 90_000)] {
            assert!(duration_wire(p).0, "{:?}", unpack_duration(p));
        }
    }

    #[test]
    fn ordering_follows_length() {
        let lt = |a: i64, b: i64| Value::Duration(a).sort_cmp(&Value::Duration(b)).is_lt();
        assert!(lt(dur(0, -1, 0), dur(0, 1, 0)));
        assert!(lt(dur(0, 1, 0), dur(0, 0, 90_000)));
        assert!(lt(dur(0, 40, 0), dur(2, 0, 0)));
    }
}
