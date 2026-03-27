#!/usr/bin/env python3
# Compare Google Benchmark JSON results between master and a PR branch
# and generate a markdown summary for posting to the PR description.

import os
import glob
import json
import argparse

try:
    import mdformat
    HAS_MDFORMAT = True
except ImportError:
    HAS_MDFORMAT = False

warnings = []
comparisons = []

# nanoseconds per unit
_NS_FACTORS = {
    'ns': 1.0,
    'us': 1_000.0,
    'ms': 1_000_000.0,
    's':  1_000_000_000.0,
}


def _to_ns(value, unit):
    return value * _NS_FACTORS.get(unit, 1.0)


def _fmt_time(value, unit):
    """Format a benchmark time value for display."""
    return f"{value:.4g} {unit}"


def _pct_diff(baseline_ns, new_ns):
    if baseline_ns == 0:
        return 0.0
    return 100.0 * (new_ns - baseline_ns) / baseline_ns


def _classify(pct):
    if pct > 5:
        return "DEGRADATION", "red"
    if pct > 3:
        return "POTENTIAL DEGRADATION", "orange"
    if pct < -5:
        return "IMPROVEMENT", "lime"
    if pct < -3:
        return "POTENTIAL IMPROVEMENT", "green"
    return "STABLE", "cadetblue"


def _load_file(path):
    """Load a Google Benchmark result file. Returns dict: name -> benchmark."""
    with open(path, 'r') as f:
        data = json.load(f)
    return {
        b['name']: b
        for b in data.get('benchmarks', [])
        if b.get('run_type') == 'iteration'
    }


def compare(master_dir, branch_dir):
    master_files = {
        os.path.basename(p): p
        for p in glob.glob(os.path.join(master_dir, '*_results.json'))
    }
    branch_files = {
        os.path.basename(p): p
        for p in glob.glob(os.path.join(branch_dir, '*_results.json'))
    }

    common       = set(master_files) & set(branch_files)
    only_branch  = set(branch_files) - common
    only_master  = set(master_files) - common

    for f in sorted(only_branch):
        warnings.append(
            f"Suite `{f}` is only in the PR branch — no baseline to compare against."
        )
    for f in sorted(only_master):
        warnings.append(
            f"Suite `{f}` exists in master but is missing from the PR branch."
        )

    for fname in sorted(common):
        suite = fname.replace('_results.json', '')
        master_bms = _load_file(master_files[fname])
        branch_bms = _load_file(branch_files[fname])

        for bname in sorted(set(master_bms) & set(branch_bms)):
            mb = master_bms[bname]
            bb = branch_bms[bname]

            m_ns = _to_ns(mb['real_time'], mb.get('time_unit', 'ns'))
            b_ns = _to_ns(bb['real_time'], bb.get('time_unit', 'ns'))

            pct = _pct_diff(m_ns, b_ns)
            label, color = _classify(pct)

            comparisons.append({
                'suite':        suite,
                'name':         bname,
                'master_time':  mb['real_time'],
                'master_unit':  mb.get('time_unit', 'ns'),
                'branch_time':  bb['real_time'],
                'branch_unit':  bb.get('time_unit', 'ns'),
                'pct':          pct,
                'label':        label,
                'color':        color,
            })


def _summary_table(branch_name):
    md = "| Suite | Benchmark | Master | Branch | Δ% | Status |\n"
    md += "|---|---|---:|---:|---:|---|\n"
    for c in comparisons:
        md += (
            f"| {c['suite']} "
            f"| `{c['name']}` "
            f"| {_fmt_time(c['master_time'], c['master_unit'])} "
            f"| {_fmt_time(c['branch_time'], c['branch_unit'])} "
            f"| {c['pct']:+.1f}% "
            f"| {c['label']} |\n"
        )
    return md


def _details_section(branch_name):
    md = ""
    for c in comparisons:
        md += (
            f"<details>\n"
            f"<summary><code>{c['name']}</code> — "
            f"{c['label']} ({c['pct']:+.1f}%)</summary>\n\n"
        )
        md += f"| | `master` | `{branch_name}` | Δ% |\n"
        md += "|---|---:|---:|---:|\n"
        md += (
            f"| real\\_time "
            f"| {_fmt_time(c['master_time'], c['master_unit'])} "
            f"| {_fmt_time(c['branch_time'], c['branch_unit'])} "
            f"| {c['pct']:+.2f}% |\n"
        )
        md += "\n</details>\n\n"
    return md


def generate_markdown(branch_name):
    md = f"## Micro-Benchmark Comparison: `master` ← `{branch_name}`\n\n"
    md += "> **real\\_time** as reported by Google Benchmark. Lower is better.\n\n"

    if warnings:
        md += "### ⚠️ Warnings\n\n"
        for w in warnings:
            md += f"- {w}\n"
        md += "\n"

    if not comparisons:
        md += "_No benchmark results could be compared._\n"
        return md

    md += "### Summary\n\n"
    md += _summary_table(branch_name)
    md += "\n### Details\n\n"
    md += _details_section(branch_name)

    return md


def main():
    parser = argparse.ArgumentParser(
        description="Compare Google Benchmark JSON results and produce a markdown report."
    )
    parser.add_argument('--master_dir', required=True,
                        help='Directory containing master *_results.json files')
    parser.add_argument('--branch_dir', required=True,
                        help='Directory containing PR branch *_results.json files')
    parser.add_argument('--branch_name', required=True,
                        help='Name of the PR branch')
    parser.add_argument('--output', default='micro_benchmark_compare.md',
                        help='Output markdown file (default: micro_benchmark_compare.md)')
    parser.add_argument('--dryrun', action='store_true',
                        help='Print to stdout without writing a file')
    args = parser.parse_args()

    compare(args.master_dir, args.branch_dir)
    md = generate_markdown(args.branch_name)

    if HAS_MDFORMAT:
        md = mdformat.text(md, extensions=["gfm"])

    print(md)
    if not args.dryrun:
        with open(args.output, 'w') as f:
            f.write(md)


if __name__ == '__main__':
    main()
