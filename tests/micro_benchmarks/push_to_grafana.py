#!/usr/bin/env python3
"""
Push micro-benchmark results to Grafana via Prometheus remote write.

Required environment variables:
  GRAFANA_REMOTE_WRITE_URL  - e.g. https://<id>.grafana.net/api/prom/push
  GRAFANA_METRICS_USER      - Grafana Cloud metrics user ID (numeric string)
  GRAFANA_METRICS_TOKEN     - API token with MetricsPublisher role

If any credential is missing the script exits with code 0 (silently skipped),
so the workflow step can run unconditionally and be activated by adding the
GitHub repository secrets/variables when ready.
"""

import os
import sys
import glob
import json
import time
import struct
import argparse


# ---------------------------------------------------------------------------
# Minimal Prometheus remote write protobuf encoder (no external proto deps)
# ---------------------------------------------------------------------------

def _varint(value):
    """Encode a non-negative integer as a protobuf varint."""
    result = b''
    while True:
        bits = value & 0x7F
        value >>= 7
        if value:
            result += bytes([bits | 0x80])
        else:
            result += bytes([bits])
            break
    return result


def _field_string(field_num, value):
    """Encode a length-delimited (wire type 2) string field."""
    encoded = value.encode('utf-8')
    return _varint((field_num << 3) | 2) + _varint(len(encoded)) + encoded


def _field_double(field_num, value):
    """Encode a 64-bit float (wire type 1) field."""
    return _varint((field_num << 3) | 1) + struct.pack('<d', float(value))


def _field_int64(field_num, value):
    """Encode a varint (wire type 0) int64 field."""
    return _varint((field_num << 3) | 0) + _varint(int(value))


def _field_bytes(field_num, data):
    """Encode an embedded message (wire type 2) field."""
    return _varint((field_num << 3) | 2) + _varint(len(data)) + data


def _encode_label(name, value):
    """Encode a Label{name, value} message."""
    return _field_string(1, name) + _field_string(2, value)


def _encode_sample(value, timestamp_ms):
    """Encode a Sample{value, timestamp} message."""
    return _field_double(1, value) + _field_int64(2, timestamp_ms)


def _encode_timeseries(labels, value, timestamp_ms):
    """Encode a TimeSeries{labels[], samples[]} message."""
    data = b''
    for k, v in sorted(labels.items()):
        data += _field_bytes(1, _encode_label(k, v))  # labels field
    data += _field_bytes(2, _encode_sample(value, timestamp_ms))  # samples field
    return data


def _encode_write_request(timeseries_list):
    """Encode a WriteRequest{timeseries[]} message."""
    data = b''
    for labels, value, timestamp_ms in timeseries_list:
        data += _field_bytes(1, _encode_timeseries(labels, value, timestamp_ms))
    return data


# ---------------------------------------------------------------------------
# Time unit normalization
# ---------------------------------------------------------------------------

_NS_FACTORS = {
    'ns': 1.0,
    'us': 1_000.0,
    'ms': 1_000_000.0,
    's':  1_000_000_000.0,
}


def _to_ns(value, unit):
    return value * _NS_FACTORS.get(unit, 1.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Push micro-benchmark results to Grafana via Prometheus remote write."
    )
    parser.add_argument('--branch',       required=True, help='Git branch name')
    parser.add_argument('--commit-sha',   required=True, help='Full commit SHA')
    parser.add_argument('--event-name',   required=True, help='GitHub event name')
    parser.add_argument('--results-dir',  default='tests/micro_benchmarks/results',
                        help='Directory containing *_results.json files')
    args = parser.parse_args()

    url   = os.environ.get('GRAFANA_REMOTE_WRITE_URL', '').strip()
    user  = os.environ.get('GRAFANA_METRICS_USER', '').strip()
    token = os.environ.get('GRAFANA_METRICS_TOKEN', '').strip()

    if not (url and user and token):
        print("Grafana credentials not configured — skipping push.", file=sys.stderr)
        sys.exit(0)

    try:
        import snappy
        import requests
    except ImportError as e:
        print(f"Missing dependency: {e}. Install grafana_requirements.txt first.", file=sys.stderr)
        sys.exit(1)

    result_files = sorted(glob.glob(os.path.join(args.results_dir, '*_results.json')))
    if not result_files:
        print(f"No result files found in {args.results_dir}.", file=sys.stderr)
        sys.exit(1)

    timestamp_ms = int(time.time() * 1000)
    short_sha = args.commit_sha[:8]

    timeseries = []
    for path in result_files:
        suite = os.path.basename(path).replace('_results.json', '')
        with open(path) as f:
            data = json.load(f)

        for b in data.get('benchmarks', []):
            if b.get('run_type') != 'iteration':
                continue

            value_ns = _to_ns(b['real_time'], b.get('time_unit', 'ns'))
            labels = {
                '__name__':   'falkordb_micro_benchmark_real_time_ns',
                'benchmark':  b['name'].replace('/', '_'),
                'suite':      suite,
                'branch':     args.branch,
                'commit_sha': short_sha,
                'event':      args.event_name,
            }
            timeseries.append((labels, value_ns, timestamp_ms))

    if not timeseries:
        print("No benchmark iterations found.", file=sys.stderr)
        sys.exit(1)

    payload = _encode_write_request(timeseries)
    compressed = snappy.compress(payload)

    headers = {
        'Content-Type':                       'application/x-protobuf',
        'Content-Encoding':                   'snappy',
        'X-Prometheus-Remote-Write-Version':  '0.1.0',
    }

    resp = requests.post(
        url,
        data=compressed,
        headers=headers,
        auth=(user, token),
        timeout=30,
    )
    resp.raise_for_status()
    print(f"Pushed {len(timeseries)} time-series to Grafana (HTTP {resp.status_code}).")


if __name__ == '__main__':
    main()
