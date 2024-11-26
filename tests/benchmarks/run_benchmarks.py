#! /usr/bin/env python3
import glob
import json
import sys
import subprocess
import os
import hashlib

import yaml
import platform
import jsonpath_ng
from typing import TextIO
from urllib.request import urlretrieve
from http.client import HTTPSConnection


def run_single_benchmark(file_stream: TextIO, bench: str):
    data = yaml.safe_load(file_stream)

    # Always prefer the environment variable over the yaml file
    if platform.system() == "Darwin":
        db_module = os.getenv("DB_MODULE", "../../bin/macos-arm64v8-release/src/falkordb.so")
    else:
        db_module = os.getenv("DB_MODULE", "../../bin/linux-x64-release/src/falkordb.so")
    if db_module is None and "db_module" not in data:
        print("Error! No DB module specified in the yaml file or the environment variable")
        exit(1)

    result_file_name = f"{data['name']}-results.json"
    if os.path.exists(result_file_name):
        os.remove(result_file_name)

    process = subprocess.Popen(["./falkordb-benchmark-go", "--yaml_config", bench,
                                "--output_file", result_file_name,
                                "--override_module", db_module],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Print the output of the benchmark tool
    for line in process.stdout:
        print(line, end='')

    # Wait for the process to finish and get the return code
    process.wait()

    # Check if there were any errors
    if process.returncode != 0:
        error_messages = process.stderr.readlines()
        print("Benchmark run failed with the following errors:", *error_messages, sep="\n")
        print("Aborting")
        exit(1)

    if "kpis" in data:
        with open(result_file_name) as results_file:
            json_results = json.load(results_file)
            for kpi in data["kpis"]:
                parsed_key = jsonpath_ng.parse(kpi["key"])
                kpi_val = parsed_key.find(json_results)
                if kpi_val is not None:
                    failed = False

                    if "min_value" in kpi and not kpi_val[0].value > kpi["min_value"]:
                        print(f'Error! Expected {kpi["key"]} to be greater than {kpi["min_value"]}, '
                              f'is {kpi_val[0].value}')
                        failed = True

                    if "max_value" in kpi and not kpi_val[0].value < kpi["max_value"]:
                        print(f'Error! Expected {kpi["key"]} to be less than {kpi["max_value"]}, '
                              f'is {kpi_val[0].value}')
                        failed = True

                    if failed:
                        exit(1)


def single_iteration(bench: str, idx: int, bench_end: int):
    print(f"========== Benchmark {idx + 1}/{bench_end} Started ================\n")

    if os.path.exists("dataset.rdb"):
        os.remove("dataset.rdb")

    with open(bench, "r") as current_bench_file:
        run_single_benchmark(current_bench_file, bench)

    print("")
    print(f"========== Benchmark {idx + 1}/{bench_end} Completed ==============")
    print("")

    try:
        subprocess.check_output("pidof redis-server", shell=True)
        print("Redis server is still running!")
        exit(1)
    except subprocess.CalledProcessError:
        pass


def verify_sha256_checksum(target_file, checksum_file):
    # Read the existing checksum
    with open(checksum_file, 'r') as f:
        existing_checksum = f.read().strip()

    # Compute the SHA256 checksum of the target file
    computed_checksum = hashlib.sha256()
    with open(target_file, 'rb') as f:
        for block in iter(lambda: f.read(4096), b''):
            computed_checksum.update(block)
    computed_checksum = computed_checksum.hexdigest()

    # Compare the computed checksum with the existing one
    if existing_checksum != computed_checksum:
        print(f"Checksum verification failed for {target_file}.")
        print(f"Existing: {existing_checksum}")
        print(f"Computed: {computed_checksum}")
        return False

    print(f"Checksum verification passed for {target_file}.")
    return True


def get_system_doublet():
    if platform.system() == "Darwin":
        return "darwin-arm64"
    else:
        return "linux-amd64"


def verify_and_download_benchmark_tool():
    if not os.path.exists("./falkordb-benchmark-go"):
        print("Downloading missing benchmark tool")

        try:
            conn = HTTPSConnection("api.github.com")
            conn.request("GET", "/repos/falkordb/falkordb-benchmark-go/releases/latest",
                         headers={"User-Agent": "FalkorDB Benchmarking Tool"})
            response_json = json.loads(conn.getresponse().read())

            system_doublet = get_system_doublet()
            files_downloaded = 0
            for asset in response_json["assets"]:
                if asset["name"] == f"falkordb-benchmark-go-{system_doublet}.tar.gz.sha256":
                    urlretrieve(asset["browser_download_url"], "./falkordb-benchmark-go.tar.gz.sha256")

                    files_downloaded += 1
                    if files_downloaded == 2:
                        break

                if asset["name"] == f"falkordb-benchmark-go-{system_doublet}.tar.gz":
                    urlretrieve(asset["browser_download_url"], "./falkordb-benchmark-go.tar.gz")

                    files_downloaded += 1
                    if files_downloaded == 2:
                        break

            if not files_downloaded == 2:
                raise Exception("Failed to download the benchmark tool or its SHA256 checksum")

            if not verify_sha256_checksum("./falkordb-benchmark-go.tar.gz",
                                          "./falkordb-benchmark-go.tar.gz.sha256"):
                raise Exception("Checksum verification failed for the benchmark tool")

            subprocess.run(["tar", "-xzf", "./falkordb-benchmark-go.tar.gz"])
            os.remove("./falkordb-benchmark-go.tar.gz")
            os.remove("./falkordb-benchmark-go.tar.gz.sha256")

            if not os.path.exists("./falkordb-benchmark-go"):
                raise Exception("Failed to extract the benchmark tool")

        except Exception as e:
            print(f"Failed to download the benchmark tool: {e}")
            exit(1)


def print_help():
    print("Usage: ./run_benchmarks.py <BenchmarkGroup>")
    print("")
    print("Example: ./run_benchmarks.py group_a          # Run group 'group_a'")
    print("")
    print("To print this help message: ./run_benchmarks.py --help or ./run_benches.py -h")
    exit(0)


def main():
    if len(sys.argv) > 1 and (sys.argv[1] == "--help" or sys.argv[1] == "-h"):
        print_help()

    if len(sys.argv) < 2:
        print_help()

    benchmark_group = sys.argv[1]
    if benchmark_group == '':
        print("Error! Empty benchmark group provided")
        exit(1)

    print(f"Starting benchmark Group {benchmark_group}...\n")
    if not os.path.isdir(benchmark_group):
        print(f"Error! Benchmark group {benchmark_group} not found")
        exit(1)

    verify_and_download_benchmark_tool()

    benches = glob.glob(f"{benchmark_group}/*.yml", recursive=False)
    benches_count = len(benches)
    for idx, bench in enumerate(benches):
        single_iteration(f"{bench}", idx, benches_count)


if __name__ == "__main__":
    main()
