#!/usr/bin/env python3

import os
import sys
import site
import argparse
import subprocess
from time import time
from RLTest import Env
from redis import ResponseError
from falkordb import Graph

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../.."))
sys.path.insert(0, os.path.join(ROOT, "deps/readies"))
import paella


def make_connection(module):
    env, db = Env(decodeResponses=True, module=module, logDir="logs")
    return db.select_graph("G")


def issue_queries(graph, timeout):
    os.environ["PATH"] = os.pathsep.join([ENV['PATH'], site.getuserbase() + '/bin', '/venv/bin'])
    os.environ["PYTHONPATH"] = os.pathsep.join([ENV['PYTHONPATH'],  HERE + "/generator"])
    cmd = ["grammarinator-generate", "CustomCypherGenerator.CustomCypherGenerator",
           "--sys-path", "generator/", "--jobs", "1", "-r", "oC_Query", "--stdout", "-d", "30"]

    start = time()
    while time() - start < timeout:
        # Capture generated queries
        try:
            query = subprocess.check_output(cmd, encoding="utf8")
        except subprocess.CalledProcessError as e:
            # Subprocess failed, emit the thrown exception and retry
            print(str(e))
            continue

        # Log query to console
        print(query)
        try:
            graph.query(query, timeout=1000)
        except ResponseError as e:
            print("Encountered exception: %s" % str(e))
        except ConnectionError as e:
            # The previous query crashed the server, exit with exception
            raise(e)


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--timeout", type=int, default=30, help="timeout value in seconds")
parser.add_argument("-m", "--module", type=str, help="module .so file")
args = parser.parse_args()

graph = make_connection(args.module)
issue_queries(graph, args.timeout)
