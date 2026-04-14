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
//!  Value::Duration(ts)  -- encoded as epoch + year/month offset + day/time
//! ```
//!
//! Duration encoding uses `construct_duration_secs` to anchor the
//! year/month components to a concrete date, then adds days and
//! sub-day offsets.  `decompose_duration` reverses this for display.
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
use chrono::{Datelike, NaiveDate, NaiveDateTime, NaiveTime, TimeZone, Utc};
use std::sync::Arc;
use thin_vec::ThinVec;

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

// Build a NaiveDate from map components: supports year/month/day, week, quarter modes.
fn date_from_map(map: &OrderMap<Arc<String>, Value>) -> Result<NaiveDate, String> {
    let year = get_int_field(map, "year").unwrap_or(1970) as i32;

    if let Some(week) = get_int_field(map, "week") {
        let dow_raw = get_int_field(map, "dayOfWeek").unwrap_or(1);
        if !(0..=6).contains(&dow_raw) {
            return Err(format!("Invalid dayOfWeek: {dow_raw}, expected 0..6"));
        }
        let dow = dow_raw as u32;
        // Find Monday of ISO week 1 for the given year
        let jan4 =
            NaiveDate::from_ymd_opt(year, 1, 4).ok_or_else(|| format!("Invalid year: {year}"))?;
        let weekday_of_jan4 = jan4.weekday().num_days_from_monday(); // Mon=0..Sun=6
        let iso_week1_monday = jan4 - chrono::Duration::days(i64::from(weekday_of_jan4));
        // Add (week-1)*7 days to get to target week Monday
        let target_monday = iso_week1_monday + chrono::Duration::days((week - 1) * 7);
        // Add dayOfWeek offset: dow 0=Sun,1=Mon,...,6=Sat
        // Monday is already our base, so offset from Monday
        let day_offset = if dow == 0 { 6 } else { i64::from(dow) - 1 };
        let result = target_monday + chrono::Duration::days(day_offset);
        return Ok(result);
    }

    if let Some(quarter) = get_int_field(map, "quarter") {
        let doq = get_int_field(map, "dayOfQuarter").unwrap_or(1);
        let quarter_start_month = ((quarter - 1) * 3 + 1) as u32;
        let base = NaiveDate::from_ymd_opt(year, quarter_start_month, 1)
            .ok_or_else(|| format!("Invalid quarter: year={year}, quarter={quarter}"))?;
        return base
            .checked_add_signed(chrono::Duration::days(doq - 1))
            .ok_or_else(|| format!("Invalid dayOfQuarter: {doq}"));
    }

    let month = get_int_field(map, "month").unwrap_or(1) as u32;
    let day = get_int_field(map, "day").unwrap_or(1) as u32;
    NaiveDate::from_ymd_opt(year, month, day)
        .ok_or_else(|| format!("Invalid date: year={year}, month={month}, day={day}"))
}

fn time_from_map(map: &OrderMap<Arc<String>, Value>) -> Result<NaiveTime, String> {
    let hour = get_int_field(map, "hour").unwrap_or(0) as u32;
    let minute = get_int_field(map, "minute").unwrap_or(0) as u32;
    let second = get_int_field(map, "second").unwrap_or(0) as u32;
    NaiveTime::from_hms_opt(hour, minute, second)
        .ok_or_else(|| format!("Invalid time: hour={hour}, minute={minute}, second={second}"))
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
        // Compact formats: YYYY, YYYYMM, YYYYDDD, YYYYMMDD
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
        let week: u32 = rest[..2]
            .parse()
            .map_err(|_| format!("Invalid week: {s}"))?;
        let d: u32 = rest[2..]
            .parse()
            .map_err(|_| format!("Invalid day of week: {s}"))?;
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
                'W' => days += n * 7,
                'D' => days += n,
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

/// Construct a duration i64 (seconds from epoch) from components.
/// The encoding stores the target datetime = epoch + years/months + days + time.
pub fn construct_duration_secs(
    years: i64,
    months: i64,
    weeks: i64,
    days: i64,
    hours: i64,
    minutes: i64,
    seconds: i64,
) -> Result<i64, String> {
    let total_month_offset = years * 12 + months;
    let base_year = 1970i32 + (total_month_offset as i32).div_euclid(12);
    let base_month = ((total_month_offset as i32).rem_euclid(12) + 1) as u32;

    let anchor = NaiveDate::from_ymd_opt(base_year, base_month, 1)
        .ok_or_else(|| format!("Invalid duration: years={years}, months={months}"))?
        .and_hms_opt(0, 0, 0)
        .ok_or("Invalid anchor time")?;

    let anchor_ts = anchor.and_utc().timestamp();
    let extra = (weeks * 7 + days) * 86400 + hours * 3600 + minutes * 60 + seconds;
    Ok(anchor_ts + extra)
}

/// Decompose a duration (seconds from epoch) into (years, months, remaining_seconds).
pub fn decompose_duration(dur_secs: i64) -> Result<(i32, i32, i64), String> {
    let dt = Utc
        .timestamp_opt(dur_secs, 0)
        .single()
        .ok_or("Invalid duration timestamp")?;

    let year_diff = dt.year() - 1970;
    let month_diff = dt.month() as i32 - 1;

    let anchor = Utc
        .with_ymd_and_hms(1970 + year_diff, (1 + month_diff) as u32, 1, 0, 0, 0)
        .single()
        .ok_or("Invalid duration anchor")?;

    let remaining_secs = dur_secs - anchor.timestamp();
    Ok((year_diff, month_diff, remaining_secs))
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
        var_arg: Type::Union(vec![Type::Map, Type::String, Type::Null]),
        ret: Type::Union(vec![Type::Date, Type::Null]),
        non_deterministic,
        fn date_fn(_, args) {
            let mut iter = args.into_iter();
            match iter.next() {
                Some(Value::Map(map)) => {
                    let d = date_from_map(&map)?;
                    let ts = d.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp();
                    Ok(Value::Date(ts))
                }
                Some(Value::String(s)) => {
                    let d = parse_date_string(&s)?;
                    let ts = d.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp();
                    Ok(Value::Date(ts))
                }
                Some(Value::Null) => Ok(Value::Null),
                None => {
                    // Zero args: return current date
                    let now = Utc::now().date_naive();
                    let ts = now.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp();
                    Ok(Value::Date(ts))
                }
                _ => unreachable!(),
            }
        }
    );

    // ── localtime() ──
    cypher_fn!(funcs, "localtime",
        var_arg: Type::Union(vec![Type::Map, Type::String, Type::Null]),
        ret: Type::Union(vec![Type::Time, Type::Null]),
        non_deterministic,
        fn localtime_fn(_, args) {
            let mut iter = args.into_iter();
            match iter.next() {
                Some(Value::Map(map)) => {
                    let t = time_from_map(&map)?;
                    // Store as seconds from epoch (base date 1970-01-01)
                    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
                    let dt = NaiveDateTime::new(epoch, t);
                    let ts = dt.and_utc().timestamp();
                    Ok(Value::Time(ts))
                }
                Some(Value::String(s)) => {
                    let t = parse_time_string(&s)?;
                    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
                    let dt = NaiveDateTime::new(epoch, t);
                    let ts = dt.and_utc().timestamp();
                    Ok(Value::Time(ts))
                }
                Some(Value::Null) => Ok(Value::Null),
                None => {
                    // Zero args: return current local time
                    let now = Utc::now().time();
                    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
                    let dt = NaiveDateTime::new(epoch, now);
                    let ts = dt.and_utc().timestamp();
                    Ok(Value::Time(ts))
                }
                _ => unreachable!(),
            }
        }
    );

    // ── localdatetime() ──
    cypher_fn!(funcs, "localdatetime",
        var_arg: Type::Union(vec![Type::Map, Type::String, Type::Null]),
        ret: Type::Union(vec![Type::Datetime, Type::Null]),
        non_deterministic,
        fn localdatetime_fn(_, args) {
            let mut iter = args.into_iter();
            match iter.next() {
                Some(Value::Map(map)) => {
                    let d = date_from_map(&map)?;
                    let t = time_from_map(&map)?;
                    let dt = NaiveDateTime::new(d, t);
                    let ts = dt.and_utc().timestamp();
                    Ok(Value::Datetime(ts))
                }
                Some(Value::String(s)) => {
                    let dt = parse_datetime_string(&s)?;
                    let ts = dt.and_utc().timestamp();
                    Ok(Value::Datetime(ts))
                }
                Some(Value::Null) => Ok(Value::Null),
                None => {
                    // Zero args: return current local datetime
                    let now = Utc::now().naive_utc();
                    let ts = now.and_utc().timestamp();
                    Ok(Value::Datetime(ts))
                }
                _ => unreachable!(),
            }
        }
    );

    // ── duration() ──
    cypher_fn!(funcs, "duration",
        args: [Type::Union(vec![Type::Map, Type::String, Type::Null])],
        ret: Type::Union(vec![Type::Duration, Type::Null]),
        fn duration_fn(_, args) {
            let mut iter = args.into_iter();
            match iter.next() {
                Some(Value::Map(map)) => {
                    let years = get_int_field(&map, "years").unwrap_or(0);
                    let months = get_int_field(&map, "months").unwrap_or(0);
                    let weeks = get_int_field(&map, "weeks").unwrap_or(0);
                    let days = get_int_field(&map, "days").unwrap_or(0);
                    let hours = get_int_field(&map, "hours").unwrap_or(0);
                    let minutes = get_int_field(&map, "minutes").unwrap_or(0);
                    let seconds = get_int_field(&map, "seconds").unwrap_or(0);
                    let ts = construct_duration_secs(years, months, weeks, days, hours, minutes, seconds)?;
                    Ok(Value::Duration(ts))
                }
                Some(Value::String(s)) => {
                    let (years, months, weeks, days, hours, minutes, seconds) = parse_duration_string(&s)?;
                    let ts = construct_duration_secs(years, months, weeks, days, hours, minutes, seconds)?;
                    Ok(Value::Duration(ts))
                }
                Some(Value::Null) => Ok(Value::Null),
                _ => unreachable!(),
            }
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
}
