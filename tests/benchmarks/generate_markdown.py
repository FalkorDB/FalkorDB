#! /usr/bin/env python3

import glob
import json
import argparse
import mdformat

benchmark_jsons = []

warnings = []


def load_benchmarks(sot_branch: str, new_branch: str):
    all_sot_files = set(glob.glob("*-results.json", root_dir=f"compare/{sot_branch}"))
    all_new_files = set(glob.glob("*-results.json", root_dir=f"compare/{new_branch}"))

    benchmarks_to_test = all_sot_files.intersection(all_new_files)

    benchmarks_only_in_new = all_new_files.difference(benchmarks_to_test)
    for benchmark in benchmarks_only_in_new:
        benchmark_name = benchmark.replace("-results.json", "")
        warnings.append(f"**Benchmark '{benchmark_name}' has no counterpart in baseline benchmarks.**")

    benchmarks_missing_in_new = all_sot_files.difference(benchmarks_to_test)
    for benchmark in benchmarks_missing_in_new:
        benchmark_name = benchmark.replace("-results.json", "")
        warnings.append(f"**Benchmark '{benchmark_name}' "
                        f"exists in baseline but is missing in the new branch benchmarks.**")

    for benchmark_file_name in benchmarks_to_test:
        # I really don't want the over indentation, so not using the context pattern again
        sot_benchmark = json.load(open(f"compare/{sot_branch}/{benchmark_file_name}", "r"))
        new_benchmark = json.load(open(f"compare/{new_branch}/{benchmark_file_name}", "r"))

        new_benchmark_clients = new_benchmark["Clients"]
        if new_benchmark_clients != sot_benchmark["Clients"]:
            warnings.append(f"Number of concurrent clients for benchmark '{benchmark_file_name}' differs between the tests")

        new_benchmark_rps = new_benchmark["MaxRps"]
        if new_benchmark_rps != sot_benchmark["MaxRps"]:
            warnings.append(f"Requests per second limit for benchmark '{benchmark_file_name}' differs between the tests")

        new_benchmark_commands_issued = new_benchmark["IssuedCommands"]
        if new_benchmark_commands_issued != sot_benchmark["IssuedCommands"]:
            warnings.append(f"Number of commands issued for benchmark '{benchmark_file_name}' differs between the tests")

        benchmark_jsons.append({
            "Benchmark": str.replace(benchmark_file_name, "-results.json", ""),
            "sot Benchmark Duration": new_benchmark["DurationMillis"],
            "new Benchmark Duration": sot_benchmark["DurationMillis"],
            "Number of Concurrent Clients": new_benchmark_clients,
            "Requests Per Second Limit": new_benchmark_rps,
            "Commands Issued": new_benchmark_commands_issued,
            "sot": {
                "Average Internal Latency": sot_benchmark["OverallGraphInternalLatencies"]["Total"]["avg"],
                "50th Percentile Internal Latency": new_benchmark["OverallGraphInternalLatencies"]["Total"]["q50"],
                "99.9th Percentile Internal Latency": sot_benchmark["OverallGraphInternalLatencies"]["Total"][
                    "q999"],
                "Average Client Latency": sot_benchmark["OverallClientLatencies"]["Total"]["avg"],
                "50th Percentile Client Latency": new_benchmark["OverallClientLatencies"]["Total"]["q50"],
                "99.9th Percentile Client Latency": sot_benchmark["OverallClientLatencies"]["Total"]["q999"],
                "Overall Request Rate": sot_benchmark["OverallQueryRates"]["Total"]
            },
            "new": {
                "Average Internal Latency": new_benchmark["OverallGraphInternalLatencies"]["Total"]["avg"],
                "50th Percentile Internal Latency": new_benchmark["OverallGraphInternalLatencies"]["Total"]["q50"],
                "99.9th Percentile Internal Latency": new_benchmark["OverallGraphInternalLatencies"]["Total"][
                    "q999"],
                "Average Client Latency": new_benchmark["OverallClientLatencies"]["Total"]["avg"],
                "50th Percentile Client Latency": new_benchmark["OverallClientLatencies"]["Total"]["q50"],
                "99.9th Percentile Client Latency": new_benchmark["OverallClientLatencies"]["Total"]["q999"],
                "Overall Request Rate": new_benchmark["OverallQueryRates"]["Total"]
            }
        })


def generate_table(dict_list: list[dict]) -> str:
    if not dict_list:
        return ""

    # Extract headers from the keys of the first dictionary
    headers = dict_list[0].keys()

    # Create the header row
    header_row = f"| {' | '.join(headers)} |"

    # Create the separator row
    separator_row = f"| {' | '.join([':---:'] * len(headers))} |".replace(":---:", "---", 1)

    # Create the data rows
    data_rows = [f"| {' | '.join(str(d[h]) for h in headers)} |" for d in dict_list]

    # Combine all parts into the final markdown table string
    markdown_table = "\n".join([header_row, separator_row] + data_rows)

    return f"{markdown_table}\n"


def generate_diff(old_val, new_val):
    return 100 * (new_val - old_val) / old_val


def generate_benchmark_list_table(sot_branch: str, new_branch: str) -> str:
    markdown_str = f"## Benchmark Comparison '{sot_branch}' <---> '{new_branch}'\n"
    markdown_str += "### Benchmark List:\n"
    markdown_str += "The following benchmarks were ran with the following settings: \n\n"
    markdown_str += generate_table([{
        "Benchmark": benchmark["Benchmark"],
        f"  Branch '_{new_branch}_' Benchmark Duration  ": benchmark["new Benchmark Duration"],
        f"  Branch '_{sot_branch}_' Benchmark Duration  ": benchmark["sot Benchmark Duration"],
        "  Number of Concurrent Clients  ": benchmark["Number of Concurrent Clients"],
        "  Commands Issued  ": benchmark["Commands Issued"],
    } for benchmark in benchmark_jsons])
    return markdown_str


def generate_warnings_text() -> str:
    if len(warnings) == 0:
        return ""

    markdown_str = "## Warnings:\n"
    markdown_str += ("The following warnings were generated in the comparison, "
                     "take them into consideration when making decisions:\n")
    markdown_str += '\n'.join([f"+ {warning}" for warning in warnings])

    return markdown_str + "\n"


def main():
    parser = argparse.ArgumentParser(prefix_chars="--")
    parser.add_argument("--new_branch", type=str, required=True,
                        help="The name of the branch to compare against the source of truth, required")
    parser.add_argument("--sot_branch", type=str,
                        help="The name of the source of truth branch, defaults to 'master'", default="master")
    parser.add_argument("--output_file", type=str,
                        help="The name of the output markdown file, defaults to 'compare.md'", default="compare.md")
    parser.add_argument("--dryrun", action="store_true", help="Only print the output")
    args = parser.parse_args()

    load_benchmarks(args.sot_branch, args.new_branch)

    output_markdown_file = None
    if not args.dryrun:
        # I want it to fail if cant open when not in dryrun, so I open it here, instead of using the context pattern
        output_markdown_file = open(args.output_file, "w")

    markdown_str = generate_benchmark_list_table(args.sot_branch, args.new_branch) + "\n"
    markdown_str += generate_warnings_text()

    markdown_str += "### Benchmarks:\n"
    for benchmark in benchmark_jsons:
        benchmark_label = benchmark["Benchmark"]
        sot_inner_avg = float(benchmark["sot"]["Average Internal Latency"])
        new_inner_avg = float(benchmark["new"]["Average Internal Latency"])

        sot_client_avg = float(benchmark["sot"]["Average Client Latency"])
        new_client_avg = float(benchmark["new"]["Average Client Latency"])

        min_val = min([sot_inner_avg, new_inner_avg, sot_client_avg, new_client_avg])
        max_val = max([sot_inner_avg, new_inner_avg, sot_client_avg, new_client_avg])

        internal_diff = generate_diff(sot_inner_avg, new_inner_avg)
        overall_diff = generate_diff(sot_client_avg, new_client_avg)

        match overall_diff:
            case _ if overall_diff > 5:
                summary_label = "DEGRADATION"
                summary_color = "red"
            case _ if overall_diff > 3:
                summary_label = "POTENTIAL DEGRADATION"
                summary_color = "yellow"
            case _ if overall_diff < -3:
                summary_label = "POTENTIAL IMPROVEMENT"
                summary_color = "green"
            case _ if overall_diff < -5:
                summary_label = "IMPROVEMENT"
                summary_color = "lime"
            case _:
                summary_label = "STABLE"
                summary_color = "cadetblue"

        markdown_str += f"""
    <details>
    
    <summary style='color:{summary_color}'>Benchmark '{benchmark_label}' ({summary_label})</summary>
    
    Overall Latency diff between baseline and current branch: **{round(internal_diff, 2)}%** \\
    (Less is better)
    
    Internal Latency diff between baseline and current branch: **{round(overall_diff, 2)}%** \\
    (Less is better)
    
    ```mermaid
    ---
    config:
        xyChart:
            plotReservedSpacePercent: 0
            titleFontSize: 16
            xAxis:
                labelFontSize: 12
        themeVariables:
            xyChart:
                plotColorPalette: "#3498DB, #DB3409"
        
    ---
    xychart-beta
        title "{benchmark_label}"
        x-axis ["Baseline Internal Avg.", "New Internal Avg.", "Baseline Overall Avg.", "New Overall Avg."]
        y-axis "Graph Internal Latency (in ms)" {min_val / 3} --> {max_val}
        bar [{benchmark['sot']['Average Internal Latency']}, 0, {benchmark['sot']['Average Client Latency']}, 0]
        bar [0, {benchmark['new']['Average Internal Latency']}, 0, {benchmark['new']['Average Client Latency']}]
    ```
    
    </details>
    
    """

    markdown_str = mdformat.text(markdown_str, extensions=["gfm"]).replace("````", "")
    print(markdown_str)
    if not args.dryrun:
        output_markdown_file.write(markdown_str)


if __name__ == "__main__":
    main()
