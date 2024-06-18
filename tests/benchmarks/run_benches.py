#! /usr/bin/env python3

import json
import sys
import subprocess
import os

import yaml
import jsonpath_ng


def run_single_benchmark(idx, file_stream):
    data = yaml.safe_load(file_stream)

    # Always prefer the environment variable over the yaml file
    db_image = os.getenv("DB_IMAGE")
    if not db_image:
        if "docker_image" in data:
            db_image = data["db_image"]
        else:
            db_image = "falkordb/falkordb:edge"  # If nothing is specified, use the latest image

    process = subprocess.Popen(["./falkordb-benchmark-go", "--yaml_config", f"./{idx}.yml",
                                "--output_file", f"{idx}-results.json", "--override_image", db_image],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

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


def main():
    print("Starting benchmark suite...")

    for idx in range(20):
        print(f"Benchmark {idx}/19:\n")

        if os.path.exists(f"{idx}-results.json"):
            os.remove(f"{idx}-results.json")

        if os.path.exists("dataset.rdb"):
            os.remove("dataset.rdb")

        with open(f"{idx}.yml", "r") as current_bench_file:
            run_single_benchmark(idx, current_bench_file)

        print(f"\nBenchmark {idx}/19 Completed\n")

        try:
            subprocess.check_output("pidof redis-server", shell=True)
            print("Redis server is still running!")
            exit(1)
        except subprocess.CalledProcessError:
            pass


if __name__ == "__main__":
    if len(sys.argv) > 1:
        bench_num = 0
        try:
            bench_num = int(sys.argv[1])
        except ValueError as e:
            print("Second argument must be an integer")
            exit(1)

        with open(f"{bench_num}.yml", "r+") as bench_file:
            run_single_benchmark(bench_num, bench_file)
    else:
        main()
