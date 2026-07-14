"""Shared per-query metric extraction from a benchmark run's `result`.

Two per-query sources in the aggregate JSON (verified against
FalkorDB/benchmark@deca7752, release v2.2):

  * `telemetry_for_type[query]` = {wait-ms, exec-ms, report-ms} — server-side.
    `exec-ms` is the clean per-query server execution signal.
  * `histogram_for_type[query]` = 11 client-latency percentiles (ms) in this
    fixed order: [p10, p20, p30, p40, p50, p60, p70, p80, p90, p95, p99].

So the metrics we expose per query: server `exec`, and client `p50`/`p95`/`p99`.
Used by both format-summary.py (comment movers) and build-trend.py (trend data).
"""

# Index into histogram_for_type for each client percentile.
_PCT_INDEX = {"p50": 4, "p90": 8, "p95": 9, "p99": 10}

# Public metric keys, in display order.
METRICS = ("exec", "p50", "p95", "p99")

METRIC_LABEL = {
    "exec": "server exec ms",
    "p50": "client p50 ms",
    "p95": "client p95 ms",
    "p99": "client p99 ms",
}


def parse_latency_ms(value):
    """Parse an aggregate-latency value into float milliseconds.

    The benchmark emits these as unit-suffixed strings whose unit VARIES with
    magnitude — "6.463ms", but "2.490s" once it crosses a second (and "…us"
    below a millisecond). A bare number is assumed already-ms. Returns None if
    unparseable. Order matters: "ms"/"us"/"µs" all end in "s", so check them
    before the bare "s". (Fixes p95/p99 silently dropping whenever they're ≥1s.)
    """
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    s = value.strip()
    try:
        if s.endswith("ms"):
            return float(s[:-2])
        if s.endswith("us") or s.endswith("µs"):
            return float(s[:-2]) / 1000.0
        if s.endswith("s"):
            return float(s[:-1]) * 1000.0
    except ValueError:
        return None
    return None


def per_query(run, metric):
    """Return {query: value_ms} for `metric` on one run's result.

    `metric` is one of METRICS. Missing/malformed entries are skipped rather
    than raising, so a partial run still yields what it can.
    """
    result = (run or {}).get("result", {}) or {}
    out = {}
    if metric == "exec":
        for q, tb in (result.get("telemetry_for_type") or {}).items():
            v = (tb or {}).get("exec-ms")
            if isinstance(v, (int, float)):
                out[q] = float(v)
        return out

    idx = _PCT_INDEX[metric]
    for q, arr in (result.get("histogram_for_type") or {}).items():
        if isinstance(arr, list) and len(arr) > idx and isinstance(arr[idx], (int, float)):
            out[q] = float(arr[idx])
    return out
