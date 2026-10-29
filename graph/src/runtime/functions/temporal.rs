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
// `Dur` holds them in one 8-byte word so `Value` does not grow. Bit 0 tells the
// two forms apart:
//   * 1: inline. Months (13 bits), days (18) and seconds (32) packed, two's
//     complement. This is every duration of a realistic size.
//   * 0: the word is a pointer from `Arc::into_raw` to the three components as
//     `i64`s, for a duration with a component too large to pack.
// The form is canonical (inline exactly when the components fit), but every
// consumer reads the components through `Dur::components` and never looks at
// the form, so the two behave identically.

const DUR_SEC_BITS: u32 = 32;
const DUR_DAY_BITS: u32 = 18;
const DUR_MONTH_BITS: u32 = 13;
const DUR_MAX_SECONDS: i64 = (1 << (DUR_SEC_BITS - 1)) - 1;
const DUR_MAX_DAYS: i64 = (1 << (DUR_DAY_BITS - 1)) - 1;
const DUR_MAX_MONTHS: i64 = (1 << (DUR_MONTH_BITS - 1)) - 1;
/// The most months a duration can hold (an `i32`, as before).
const DUR_LIMIT_MONTHS: i64 = i32::MAX as i64;

pub struct Dur(u64);

// SAFETY: a `Dur` is either plain bits or an owned `Arc<[i64; 3]>`.
unsafe impl Send for Dur {}
unsafe impl Sync for Dur {}

impl Dur {
    /// Build a duration, refusing components whose length in seconds does not
    /// fit an `i64` (or whose months do not fit an `i32`).
    pub fn new(
        months: i64,
        days: i64,
        seconds: i64,
    ) -> Result<Self, String> {
        let err =
            || format!("Duration out of range: months={months}, days={days}, seconds={seconds}");
        if months.unsigned_abs() > DUR_LIMIT_MONTHS as u64 {
            return Err(err());
        }
        // The components must also be addable to a timestamp.
        days.checked_mul(86400)
            .and_then(|d| d.checked_add(seconds))
            .and_then(|t| t.checked_abs())
            .ok_or_else(err)?;
        if months.unsigned_abs() <= DUR_MAX_MONTHS as u64
            && days.unsigned_abs() <= DUR_MAX_DAYS as u64
            && seconds.unsigned_abs() <= DUR_MAX_SECONDS as u64
        {
            let bits = ((months as u64) << (DUR_DAY_BITS + DUR_SEC_BITS + 1))
                | ((days as u64 & ((1 << DUR_DAY_BITS) - 1)) << (DUR_SEC_BITS + 1))
                | ((seconds as u64 & ((1 << DUR_SEC_BITS) - 1)) << 1)
                | 1;
            Ok(Self(bits))
        } else {
            let ptr = Arc::into_raw(Arc::new([months, days, seconds]));
            Ok(Self(ptr as usize as u64))
        }
    }

    /// The signed (months, days, seconds). The one way to read a duration.
    #[inline]
    pub fn components(&self) -> (i64, i64, i64) {
        if self.0 & 1 == 1 {
            Self::unpack_inline(self.0)
        } else {
            // SAFETY: bit 0 clear means a live `Arc<[i64; 3]>` pointer.
            let a = unsafe { &*(self.0 as usize as *const [i64; 3]) };
            (a[0], a[1], a[2])
        }
    }

    #[inline]
    const fn unpack_inline(bits: u64) -> (i64, i64, i64) {
        let b = bits as i64;
        let months = b >> (DUR_DAY_BITS + DUR_SEC_BITS + 1);
        let days = (b << DUR_MONTH_BITS) >> (64 - DUR_DAY_BITS);
        let seconds = (b << (DUR_MONTH_BITS + DUR_DAY_BITS)) >> (64 - DUR_SEC_BITS);
        (months, days, seconds)
    }

    /// The word, when the duration is stored inline (for the attribute store).
    #[inline]
    pub const fn inline_bits(&self) -> Option<u64> {
        if self.0 & 1 == 1 { Some(self.0) } else { None }
    }

    /// Rebuild a duration from `inline_bits`.
    #[inline]
    pub const fn from_inline_bits(bits: u64) -> Self {
        debug_assert!(bits & 1 == 1);
        Self(bits)
    }
}

impl Clone for Dur {
    #[inline]
    fn clone(&self) -> Self {
        if self.0 & 1 == 0 {
            // SAFETY: a live pointer from `Arc::into_raw`.
            unsafe { Arc::increment_strong_count(self.0 as usize as *const [i64; 3]) };
        }
        Self(self.0)
    }
}

impl Drop for Dur {
    #[inline]
    fn drop(&mut self) {
        if self.0 & 1 == 0 {
            // SAFETY: a live pointer from `Arc::into_raw`, released once.
            unsafe { drop(Arc::from_raw(self.0 as usize as *const [i64; 3])) };
        }
    }
}

impl PartialEq for Dur {
    #[inline]
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        self.components() == other.components()
    }
}
impl Eq for Dur {}

impl std::hash::Hash for Dur {
    fn hash<H: std::hash::Hasher>(
        &self,
        state: &mut H,
    ) {
        self.components().hash(state);
    }
}

impl std::fmt::Debug for Dur {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        let (m, d, s) = self.components();
        write!(f, "Dur({m}m {d}d {s}s)")
    }
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
) -> Result<Dur, String> {
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
    Dur::new(total_months, total_days, total_seconds)
}

/// Convert a duration stored with the legacy encoding (seconds from the Unix
/// epoch of `epoch + years/months + days + time`, as written by the C engine
/// and by earlier Rust versions) into components. The legacy decomposition is
/// replayed exactly, so a stored value reads back as it did.
pub fn duration_from_legacy(legacy: i64) -> Result<Dur, String> {
    let days = legacy.div_euclid(86400);
    let time_of_day = legacy.rem_euclid(86400);
    let (y, m, d) = crate::runtime::value::civil_from_days(days);
    let months = i64::from(y - 1970) * 12 + i64::from(m) - 1;
    let remaining = (i64::from(d) - 1) * 86400 + time_of_day;
    Dur::new(months, remaining / 86400, remaining % 86400)
}

/// Encode a duration in the legacy epoch-offset form (what the compact
/// result-set protocol carries, and what C writes). Components that cannot be
/// expressed non-negatively borrow exactly as the C engine does.
pub fn duration_to_legacy(dur: &Dur) -> i64 {
    let (months, days, seconds) = dur.components();
    let base_year = 1970i32.saturating_add(months.div_euclid(12) as i32);
    let base_month = (months.rem_euclid(12) + 1) as u32;
    days_from_civil(base_year, base_month, 1)
        .wrapping_mul(86400)
        .wrapping_add(days.wrapping_mul(86400))
        .wrapping_add(seconds)
}

/// How a duration goes on the wire (RDB and effects).
pub enum DurationWire {
    /// The C engine's epoch-offset `time_t`, under `T_DURATION`.
    Legacy(i64),
    /// Signed (months, days, seconds), under `T_DURATION_V2`.
    V2(i64, i64, i64),
}

/// A duration the legacy encoding holds losslessly is written in it, byte for
/// byte what the C engine writes; any other (negative components, a day count
/// past the month length, ...) is written as its components.
pub fn duration_wire(dur: &Dur) -> DurationWire {
    let legacy = duration_to_legacy(dur);
    if duration_from_legacy(legacy).as_ref() == Ok(dur) {
        DurationWire::Legacy(legacy)
    } else {
        let (m, d, s) = dur.components();
        DurationWire::V2(m, d, s)
    }
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
        Dur, DurationWire, construct_duration, duration_from_legacy, duration_to_legacy,
        duration_wire,
    };
    use crate::runtime::value::Value;
    use std::hash::{DefaultHasher, Hash, Hasher};

    fn dur(
        months: i64,
        days: i64,
        secs: i64,
    ) -> Dur {
        Dur::new(months, days, secs).unwrap()
    }

    fn v(
        months: i64,
        days: i64,
        secs: i64,
    ) -> Value {
        Value::Duration(dur(months, days, secs))
    }

    fn is_inline(d: &Dur) -> bool {
        d.inline_bits().is_some()
    }

    fn hash_of(d: &Value) -> u64 {
        let mut h = DefaultHasher::new();
        d.hash(&mut h);
        h.finish()
    }

    #[test]
    fn value_stays_sixteen_bytes() {
        assert_eq!(std::mem::size_of::<Value>(), 16);
        assert_eq!(std::mem::size_of::<Dur>(), 8);
    }

    #[test]
    fn components_round_trip_in_both_forms() {
        for m in [-4095, -13, -1, 0, 1, 13, 4095, 4096, -20_000, 2_000_000_000] {
            for d in [
                -131_071,
                -3,
                0,
                3,
                131_071,
                131_072,
                200_000,
                -4_000_000_000_000,
            ] {
                for s in [
                    -2_147_483_647,
                    -86_401,
                    0,
                    86_401,
                    2_147_483_647,
                    2_147_483_648,
                    9_000_000_000_000_000,
                    -9_000_000_000_000_000,
                ] {
                    let x = dur(m, d, s);
                    assert_eq!(x.components(), (m, d, s));
                    assert_eq!(x.clone().components(), (m, d, s));
                    let fits = m.abs() <= 4095 && d.abs() <= 131_071 && s.abs() <= 2_147_483_647;
                    assert_eq!(is_inline(&x), fits, "{m} {d} {s}");
                }
            }
        }
        assert!(Dur::new(0, 0, i64::MAX).is_ok());
        assert!(Dur::new(0, i64::MAX, 0).is_err());
        assert!(Dur::new(i64::MAX, 0, 0).is_err());
    }

    #[test]
    fn constructor_never_borrows_between_units() {
        let p = construct_duration(0, 0, 0, -1, 0, 0, 0).unwrap();
        assert_eq!(p.components(), (0, -1, 0));
        let p = construct_duration(1, -14, 1, 0, 1, -90, 0).unwrap();
        assert_eq!(p.components(), (-2, 7, -1800));
    }

    #[test]
    fn large_components_keep_the_full_range() {
        let h = construct_duration(0, 0, 0, 0, 600_000, 0, 0).unwrap();
        assert_eq!(h.components(), (0, 0, 2_160_000_000));
        assert!(!is_inline(&h));
        assert_eq!(Value::format_duration(&h), "PT600000H");
        let s = construct_duration(0, 0, 0, 0, 0, 0, 9_000_000_000_000_000).unwrap();
        assert_eq!(Value::format_duration(&s), "PT2500000000000H");
        let d = construct_duration(0, 0, 0, 200_000, 0, 0, 0).unwrap();
        assert_eq!(Value::format_duration(&d), "P200000D");
        let m = construct_duration(0, 10_000, 0, 0, 0, 0, 0).unwrap();
        assert_eq!(Value::format_duration(&m), "P833Y4M");
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
            (
                -10_000,
                -200_000,
                -2_160_000_000,
                "P-833Y-4M-200000DT-600000H",
            ),
        ] {
            assert_eq!(Value::format_duration(&dur(m, d, s)), want);
        }
    }

    #[test]
    fn issue_2745_rows() {
        let Ok(Value::Duration(sum)) = v(0, 2, 0) + v(0, -3, 0) else {
            panic!()
        };
        assert_eq!(Value::format_duration(&sum), "P-1D");
        let date = Value::Date(1_583_020_800);
        let Ok(Value::Date(d)) = date + Value::Duration(sum) else {
            panic!()
        };
        assert_eq!(Value::format_date(d), "2020-02-29");
    }

    #[test]
    fn arithmetic_crosses_between_forms() {
        // big - big that fits is the packed value, equal and hashing alike
        let big_a = v(0, 0, 3_000_000_000);
        let big_b = v(0, 0, 3_000_000_000 - 90);
        let Ok(diff) = big_a.clone() - big_b else {
            panic!()
        };
        let packed = v(0, 0, 90);
        assert!(matches!(&diff, Value::Duration(d) if is_inline(d)));
        assert_eq!(diff, packed);
        assert_eq!(hash_of(&diff), hash_of(&packed));
        // packed + packed that overflows the inline range becomes big
        let Ok(sum) = v(0, 0, 2_000_000_000) + v(0, 0, 2_000_000_000) else {
            panic!()
        };
        assert!(matches!(&sum, Value::Duration(d) if !is_inline(d)));
        assert_eq!(sum, v(0, 0, 4_000_000_000));
        assert_eq!(hash_of(&sum), hash_of(&v(0, 0, 4_000_000_000)));
        // mixed big and negative
        let Ok(Value::Duration(m)) =
            v(-20_000, 300_000, -5_000_000_000) + v(20_000, -300_000, 5_000_000_001)
        else {
            panic!()
        };
        assert_eq!(m.components(), (0, 0, 1));
        // overflow is an error
        assert!((v(0, 0, i64::MAX) + v(0, 0, 1)).is_err());
    }

    #[test]
    fn add_and_sub_a_negative_duration_are_calendar_correct() {
        let date = Value::Date(1_583_020_800); // 2020-03-01
        let Ok(Value::Date(d)) = date.clone() + v(-1, 0, 0) else {
            panic!()
        };
        assert_eq!(Value::format_date(d), "2020-02-01");
        let Ok(Value::Date(d)) = date.clone() - v(-1, 0, 0) else {
            panic!()
        };
        assert_eq!(Value::format_date(d), "2020-04-01");
        let Ok(Value::Datetime(t)) = Value::Datetime(1_583_056_800) + v(0, -1, -39_600) else {
            panic!()
        };
        assert_eq!(Value::format_datetime(t), "2020-02-28T23:00:00");
        // a big-form duration shifts dates the same way
        let Ok(Value::Date(d)) = date + v(0, 200_000, 0) else {
            panic!()
        };
        assert_eq!(Value::format_date(d), "2567-09-30");
    }

    #[test]
    fn duration_difference_keeps_components() {
        let Ok(Value::Duration(p)) = v(1, 0, 0) - v(0, 3, 0) else {
            panic!()
        };
        assert_eq!(Value::format_duration(&p), "P1M-3D");
    }

    #[test]
    fn legacy_encoding_converts_losslessly_where_canonical() {
        for p in [
            dur(0, 0, 0),
            dur(13, 2, 3_661),
            dur(0, 1, 0),
            dur(1, 0, 86_399),
        ] {
            let legacy = duration_to_legacy(&p);
            assert_eq!(duration_from_legacy(legacy), Ok(p.clone()));
            assert!(matches!(duration_wire(&p), DurationWire::Legacy(l) if l == legacy));
        }
        assert_eq!(duration_from_legacy(86_400), Ok(dur(0, 1, 0)));
        for p in [
            dur(0, -1, 0),
            dur(0, 40, 0),
            dur(0, 0, 90_000),
            dur(0, 0, 3_000_000_000),
        ] {
            assert!(matches!(duration_wire(&p), DurationWire::V2(..)), "{p:?}");
        }
    }

    #[test]
    fn ordering_follows_length() {
        let lt = |a: Value, b: Value| a.sort_cmp(&b).is_lt();
        assert!(lt(v(0, -1, 0), v(0, 1, 0)));
        assert!(lt(v(0, 1, 0), v(0, 0, 90_000)));
        assert!(lt(v(0, 40, 0), v(2, 0, 0)));
        assert!(lt(v(0, 0, 2_000_000_000), v(0, 0, 3_000_000_000)));
    }
}
