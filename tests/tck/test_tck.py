import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import common

from behave.__main__ import main as behave_main

def setup_module(module):
    common.start_redis()


def teardown_module(module):
    common.shutdown_redis()

def query(query):
    return common.client.execute_command("GRAPH.QUERY", "x", query)

def test_tck():
    tck_features = os.getenv("TCK_FEATURES", "./tests/tck/features/")
    cmd = [tck_features, '--format=progress', '--tags=-crash', '--tags=-skip', "--no-capture"]
    tck_include = os.getenv("TCK_INCLUDE", "")
    if not tck_include:
        tck_done_file = os.getenv("TCK_DONE", "")
        if tck_done_file and os.path.exists(tck_done_file):
            with open(tck_done_file, "r") as file:
                tck_include = "|".join(line.strip() for line in file if line.strip())
    if tck_include:
        cmd = [tck_features, '--format=progress', '--tags=-crash', '--tags=-skip', "--no-capture", "--stop", "--include", tck_include]
    res = behave_main(cmd)
    res = 'pass' if res == 0 else 'fail'
    assert res == 'pass'
