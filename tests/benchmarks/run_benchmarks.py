#! /usr/bin/env python3

import json
import sys
import subprocess
import os
import hashlib

import yaml
import jsonpath_ng

from urllib.request import urlretrieve
from http.client import HTTPSConnection


def run_single_benchmark(idx, file_stream):
    data = yaml.safe_load(file_stream)

    # Always prefer the environment variable over the yaml file
    db_image = os.getenv("DB_IMAGE")
    if not db_image:
        if "docker_image" in data:
            db_image = data["docker_image"]
        else:
            db_image = "falkordb/falkordb:edge"  # If nothing is specified, use the latest image

    process = subprocess.Popen(["./falkordb-benchmark-go", "--yaml_config", f"./{idx}.yml",
                                "--output_file", f"{idx}-results.json", "--override_image", db_image],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    for line in process.stdout:
        print(line, end='')

    # Wait for the process to finish and get the return code
    process.wait()

    # Check if there were any errors
    if process.returncode != 0:
        for line in process.stderr:
            print(line, end='')

        print("Benchmark run failed, aborting")
        exit(1)

    if "kpis" in data:
        with open(f"{idx}-results.json", 'r') as results_file:
            json_results = json.load(results_file)
            for kpi in data["kpis"]:
                parsed_key = jsonpath_ng.parse(kpi["key"])
                kpi_val = parsed_key.find(json_results)
                if kpi_val is not None:
                    failed = False

                    if "min_value" in kpi:
                        if not kpi_val[0].value > kpi["min_value"]:
                            print(f'Error! Expected {kpi["key"]} to be greater than {kpi["min_value"]}, '
                                  f'is {kpi_val[0].value}')
                            failed = True

                    if "max_value" in kpi:
                        if not kpi_val[0].value < kpi["max_value"]:
                            print(f'Error! Expected {kpi["key"]} to be less than {kpi["max_value"]}, '
                                  f'is {kpi_val[0].value}')
                            failed = True

                    if failed:
                        exit(1)


def single_iteration(idx: int, bench_start: int, bench_end: int):
    bench_relative_idx = idx - bench_start
    print(f"========== Benchmark {bench_relative_idx + 1}/{bench_end - bench_start} Started ================\n")

    if os.path.exists(f"{idx}-results.json"):
        os.remove(f"{idx}-results.json")

    if os.path.exists("dataset.rdb"):
        os.remove("dataset.rdb")

    with open(f"{idx}.yml", "r") as current_bench_file:
        run_single_benchmark(idx, current_bench_file)

    print("")
    print(f"========== Benchmark {bench_relative_idx + 1}/{bench_end - bench_start} Completed ==============")
    print("")

    try:
        subprocess.check_output("pidof redis-server", shell=True)
        print("Redis server is still running!")
        exit(1)
    except subprocess.CalledProcessError:
        pass


def verify_and_download_graph500():
    if not os.path.exists("./datasets/graph500.rdb"):
        print("Downloading missing dataset")
        try:
            urlretrieve("https://s3.amazonaws.com/benchmarks.redislabs/redisgraph/"
                        "datasets/graph500-scale18-ef16_v2.4.7_dump.rdb", "./datasets/graph500.rdb")
        except Exception as e:
            print(f"Failed to download the dataset: {e}")
            exit(1)


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
                    urlretrieve(asset["browser_download_url"], f"./falkordb-benchmark-go.tar.gz.sha256")

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


def main():
    if len(sys.argv) > 1 and (sys.argv[1] == "--help" or sys.argv[1] == "-h"):
        print("Usage: ./run_benchmarks.py <start-bench>[-<end-bench>] [<benchmark_name>]")
        print("")
        print("Example: ./run_benchmarks.py 1-5 MyBenchmark        # Run benchmarks 1 to 5 and "
              "create a tar 'MyBenchmark-results.tar.gz' with the results")
        print("Example: ./run_benchmarks.py 5 MyBenchmark          # Run only the 5th benchmark")
        print("Example: ./run_benchmarks.py                        # Run all benchmarks")
        print("")
        print("To print this help message: ./run_benchmarks.py --help or ./run_benches.py -h")
        exit(0)

    benchmark_name = None
    if len(sys.argv) > 2:
        benchmark_name = sys.argv[2]

    print(f"Starting benchmark suite{f' "{benchmark_name}"' if benchmark_name else ''}...\n")

    verify_and_download_benchmark_tool()
    verify_and_download_graph500()

    bench_start = 0
    bench_end = 20
    if len(sys.argv) > 1:
        bench_split = sys.argv[1].split("-")

        try:
            bench_start = int(bench_split[0])
            bench_end = bench_start + 1
            if len(bench_split) > 1:
                bench_end = int(bench_split[1])
                if bench_end == bench_start:
                    print("Benchmark range must be at least 1 benchmarks(range is not inclusive)")
                    exit(1)
        except ValueError as e:
            print(f"Benchmark numbers must be Positive integers: {e}")
            exit(1)

    for idx in range(bench_start, bench_end):
        single_iteration(idx, bench_start, bench_end)

    if benchmark_name:
        # Create tar from the results of the benchmarks
        command = ["tar", "-czf", f"{benchmark_name}-results.tar.gz"]
        command.extend([f'{idx}-results.json' for idx in range(bench_start, bench_end)])
        res = subprocess.run(command)
        if res.returncode != 0:
            print("Failed to create tar file")
            if os.path.exists(f"{benchmark_name}-results.tar.gz"):
                os.remove(f"{benchmark_name}-results.tar.gz")
            exit(1)


if __name__ == "__main__":
    main()
