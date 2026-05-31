//! Graph slowlog: tracks the N slowest queries per graph.
//!
//! Thread-safe via `Mutex`. Each entry records the command name, query text,
//! optional parameters, latency, and timestamp. Queries faster than
//! [`MIN_LATENCY_MS`] are ignored. Duplicate queries (same command + query
//! text) update in place rather than creating a new entry.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use redis_module::raw;

use crate::reply::format_g;

/// Maximum number of slowlog entries.
const MAX_ENTRIES: usize = 10;

/// Queries faster than this (in ms) are not logged.
const MIN_LATENCY_MS: f64 = 10.0;

/// Maximum stored string length for query/params. Longer strings are truncated
/// with a trailing `"..."`.
const STR_MAX_LEN: usize = 2048;

/// Truncate `s` to at most `STR_MAX_LEN` characters, appending `"..."` if
/// truncated.
fn truncate(s: &str) -> String {
    if s.len() > STR_MAX_LEN {
        let mut t = String::with_capacity(STR_MAX_LEN + 3);
        t.push_str(&s[..STR_MAX_LEN]);
        t.push_str("...");
        t
    } else {
        s.to_owned()
    }
}

/// Compute a deduplication key from command + query.
fn entry_hash(
    cmd: &str,
    query: &str,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    cmd.hash(&mut hasher);
    // Hash at most STR_MAX_LEN bytes of the query (matches C behaviour).
    let n = query.len().min(STR_MAX_LEN);
    query[..n].hash(&mut hasher);
    hasher.finish()
}

struct SlowLogEntry {
    cmd: String,
    query: String,
    params: Option<String>,
    latency: f64,
    timestamp: f64,
    hash: u64,
}

pub struct SlowLog {
    inner: Mutex<SlowLogInner>,
}

struct SlowLogInner {
    entries: Vec<SlowLogEntry>,
    min_latency: f64,
}

impl SlowLog {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(SlowLogInner {
                entries: Vec::with_capacity(MAX_ENTRIES),
                min_latency: 0.0,
            }),
        }
    }

    /// Record a query. Called after execution completes.
    ///
    /// - `cmd`: command name (e.g. `"GRAPH.QUERY"`)
    /// - `full_query`: the raw query string including any `CYPHER param=val` prefix
    /// - `params_offset`: byte offset where the actual query starts (params are
    ///   `full_query[..params_offset]`)
    /// - `latency_ms`: execution time in milliseconds
    pub fn add(
        &self,
        cmd: &str,
        full_query: &str,
        params_offset: usize,
        latency_ms: f64,
    ) {
        if latency_ms < MIN_LATENCY_MS {
            return;
        }

        let query_text = full_query[params_offset..].trim_start();
        let params_text = full_query[..params_offset].trim();

        let mut inner = self.inner.lock();

        // Quick pre-check (no truncation / hashing needed if we can't fit).
        if inner.entries.len() >= MAX_ENTRIES && latency_ms <= inner.min_latency {
            return;
        }

        let hash = entry_hash(cmd, query_text);

        // Check for existing entry with same hash.
        if let Some(existing) = inner.entries.iter_mut().find(|e| e.hash == hash) {
            if latency_ms > existing.latency {
                // Update latency, timestamp, and params.
                existing.latency = latency_ms;
                existing.timestamp = unix_now();
                existing.params = if params_text.is_empty() {
                    None
                } else {
                    Some(truncate(params_text))
                };
            }
            return;
        }

        // Build truncated strings.
        let truncated_query = truncate(query_text);
        let truncated_params = if params_text.is_empty() {
            None
        } else {
            Some(truncate(params_text))
        };
        let timestamp = unix_now();

        if inner.entries.len() < MAX_ENTRIES {
            inner.entries.push(SlowLogEntry {
                cmd: cmd.to_owned(),
                query: truncated_query,
                params: truncated_params,
                latency: latency_ms,
                timestamp,
                hash,
            });
        } else {
            // Replace the fastest (min-latency) entry.
            let idx = inner
                .entries
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.latency.total_cmp(&b.1.latency))
                .map(|(i, _)| i)
                .unwrap();
            inner.entries[idx] = SlowLogEntry {
                cmd: cmd.to_owned(),
                query: truncated_query,
                params: truncated_params,
                latency: latency_ms,
                timestamp,
                hash,
            };
        }

        // Recompute min_latency.
        inner.min_latency = inner
            .entries
            .iter()
            .map(|e| e.latency)
            .fold(f64::MAX, f64::min);
    }

    /// Clear all entries.
    pub fn reset(&self) {
        let mut inner = self.inner.lock();
        inner.entries.clear();
        inner.min_latency = 0.0;
    }

    /// Reply to the Redis client with the slowlog contents.
    ///
    /// # Safety
    /// `ctx` must be a valid Redis module context.
    pub unsafe fn reply(
        &self,
        ctx: *mut raw::RedisModuleCtx,
    ) {
        let inner = self.inner.lock();
        raw::reply_with_array(ctx, inner.entries.len() as _);
        for entry in &inner.entries {
            raw::reply_with_array(ctx, 5);

            // 1. timestamp (double)
            raw::reply_with_double(ctx, entry.timestamp);

            // 2. command
            raw::reply_with_string_buffer(ctx, entry.cmd.as_ptr().cast(), entry.cmd.len());

            // 3. query
            raw::reply_with_string_buffer(ctx, entry.query.as_ptr().cast(), entry.query.len());

            // 4. latency (formatted with ~5 significant digits, matching C's %.5g)
            let latency_str = format_g(entry.latency, 5);
            raw::reply_with_string_buffer(ctx, latency_str.as_ptr().cast(), latency_str.len());

            // 5. params (string or null)
            match &entry.params {
                Some(p) => {
                    raw::reply_with_string_buffer(ctx, p.as_ptr().cast(), p.len());
                }
                None => {
                    raw::reply_with_null(ctx);
                }
            }
        }
    }
}

/// Current time as integer UNIX seconds, returned as f64 for
/// `reply_with_double`. Matches C FalkorDB slowlog timestamp granularity.
fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
        .floor()
}
