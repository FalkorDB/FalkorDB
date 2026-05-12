"""
Memory regression tests for FalkorDB.

Unlike tests in `test_memory_usage.py` (which validate that the
`GRAPH.MEMORY USAGE` command reports something *correct*), the tests in this
file act as **true regression tests** for memory consumption: they exercise a
small set of realistic, deterministic workloads, read the reported total
memory consumption, and compare it against a baseline value stored in
`memory_baselines.json`. A workload fails CI if it consumes meaningfully more
memory than the recorded baseline.

The intent is to catch unintentional memory regressions introduced by code
changes (e.g. growing per-node/per-edge overhead, leaks in attribute storage,
matrix bloat, etc.) before they reach a release.

How comparisons work
--------------------
For every workload we measure the `total_graph_sz_mb` value reported by
`GRAPH.MEMORY USAGE`. The current value is considered a regression iff:

    current > baseline * (1 + TOLERANCE_PCT/100) + TOLERANCE_MIN_MB

`TOLERANCE_MIN_MB` provides an absolute floor so that very small workloads
(whose reported size is at or near zero MB) do not produce false positives
from rounding noise.

Configuration (environment variables)
-------------------------------------
* `MEMORY_REGRESSION_UPDATE=1`
    Run in "update" mode: re-measure every workload and overwrite
    `memory_baselines.json` with the freshly measured values. Use this when
    you have *intentionally* changed memory consumption (e.g. a new feature
    that legitimately needs more memory) and want to refresh the baseline.
    The test never fails in this mode.

* `MEMORY_REGRESSION_TOLERANCE_PCT` (default: 10)
    Allowed relative growth, in percent, above the baseline before the test
    flags a regression.

* `MEMORY_REGRESSION_TOLERANCE_MIN_MB` (default: 0.5)
    Absolute floor (in MB) added to the tolerance to avoid false positives
    on very small reported values.

Bootstrap behaviour
-------------------
If a workload has no baseline yet (fresh checkout, newly added workload),
the test records its measured size, prints a message instructing the
maintainer to commit the updated baselines file, and passes. This keeps the
false-positive rate at zero on the very first run while still providing the
plumbing for genuine regression detection on subsequent runs.

Updating baselines after an intentional change
----------------------------------------------
1. Run:
       MEMORY_REGRESSION_UPDATE=1 \
           ./tests/flow/tests.sh TEST=test_memory_regression
2. Inspect the diff of `tests/flow/memory_baselines.json`.
3. Commit it together with the code change that justifies the new values.
"""

import json
import os

from common import *

GRAPH_ID = "memory_regression"

# Location of the persisted baselines, next to this test file
BASELINES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "memory_baselines.json")

# Tolerance configuration -- see module docstring
TOLERANCE_PCT = float(os.getenv("MEMORY_REGRESSION_TOLERANCE_PCT", "10"))
TOLERANCE_MIN_MB = float(os.getenv("MEMORY_REGRESSION_TOLERANCE_MIN_MB", "0.5"))
UPDATE_BASELINES = os.getenv("MEMORY_REGRESSION_UPDATE", "0") == "1"


def _load_baselines():
    """Load `memory_baselines.json`. Returns the full document (so we can
    preserve the leading `_comment` block when rewriting it)."""
    try:
        with open(BASELINES_PATH, "r") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        doc = {}
    if not isinstance(doc, dict):
        doc = {}
    doc.setdefault("workloads", {})
    if not isinstance(doc["workloads"], dict):
        doc["workloads"] = {}
    return doc


def _save_baselines(doc):
    """Persist the baselines document back to disk."""
    with open(BASELINES_PATH, "w") as f:
        json.dump(doc, f, indent=4, sort_keys=False)
        f.write("\n")


class testMemoryRegression(FlowTestsBase):
    """Regression tests asserting that memory usage for a fixed set of
    workloads does not grow beyond a configurable tolerance vs. a checked-in
    baseline."""

    def __init__(self):
        # Use the same environment configuration as the existing memory
        # usage tests so the two test files exercise the server identically.
        self.env, self.db = Env(env='oss-cluster')
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

        # The baselines document is loaded once per test session. When
        # running in `MEMORY_REGRESSION_UPDATE=1` mode the in-memory copy is
        # mutated by each workload and flushed to disk at teardown.
        self._baselines_doc = _load_baselines()
        self._dirty = False

    def tearDown(self):
        # Clean up the workload graph between tests so each workload sees an
        # empty graph (we recreate the dataset inside every test).
        try:
            self.graph.delete()
        except ResponseError:
            pass
        self.graph = self.db.select_graph(GRAPH_ID)

        if UPDATE_BASELINES and self._dirty:
            _save_baselines(self._baselines_doc)
            self._dirty = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _total_memory_mb(self):
        """Return the `total_graph_sz_mb` reported by `GRAPH.MEMORY USAGE`."""
        res = self.conn.execute_command("GRAPH.MEMORY", "USAGE", GRAPH_ID,
                                        "SAMPLES", 100)
        # Result is a flat array of (key, value) pairs; the value associated
        # with "total_graph_sz_mb" lives at index 1 (see test_memory_usage.py).
        return float(res[1])

    def _check_regression(self, workload, measured_mb):
        """Compare `measured_mb` against the recorded baseline for
        `workload`. Records the measurement when missing or when running in
        UPDATE mode; otherwise fails the test on regression."""

        workloads = self._baselines_doc["workloads"]
        baseline = workloads.get(workload)

        if UPDATE_BASELINES:
            workloads[workload] = measured_mb
            self._dirty = True
            print(f"[memory-regression] UPDATE: {workload} -> {measured_mb:.6f} MB")
            return

        if baseline is None:
            # First time we see this workload: bootstrap, pass, and ask the
            # maintainer to commit the new baseline.
            workloads[workload] = measured_mb
            self._dirty = True
            _save_baselines(self._baselines_doc)
            print(
                f"[memory-regression] NOTICE: no baseline for '{workload}'. "
                f"Recorded {measured_mb:.6f} MB. "
                f"Please commit the updated tests/flow/memory_baselines.json."
            )
            return

        baseline = float(baseline)
        limit = baseline * (1.0 + TOLERANCE_PCT / 100.0) + TOLERANCE_MIN_MB

        msg = (
            f"Memory regression in workload '{workload}': "
            f"baseline={baseline:.6f} MB, measured={measured_mb:.6f} MB, "
            f"limit={limit:.6f} MB "
            f"(tolerance={TOLERANCE_PCT}% + {TOLERANCE_MIN_MB} MB). "
            f"If this change is intentional, refresh the baseline by running "
            f"with MEMORY_REGRESSION_UPDATE=1 and commit "
            f"tests/flow/memory_baselines.json."
        )
        self.env.assertTrue(measured_mb <= limit, msg)

    def _run_workload(self, name, build_query, *, repeat=1):
        """Build the workload by running `build_query` `repeat` times, then
        measure total memory and compare against the baseline."""
        for _ in range(repeat):
            self.graph.query(build_query)
        measured = self._total_memory_mb()
        self._check_regression(name, measured)

    # ------------------------------------------------------------------
    # Workloads
    # ------------------------------------------------------------------
    #
    # Each test below defines one deterministic workload. The query payload
    # is intentionally small enough to keep CI fast (< a few seconds) but
    # large enough to produce a stable, non-trivial memory reading.

    def test_baseline_nodes_only_10k(self):
        """10K unlabeled nodes with no attributes."""
        self._run_workload(
            "nodes_only_10k",
            "UNWIND range(1, 10000) AS x CREATE ()"
        )

    def test_baseline_nodes_labeled_10k(self):
        """10K labeled nodes (:A) with no attributes."""
        self._run_workload(
            "nodes_labeled_10k",
            "UNWIND range(1, 10000) AS x CREATE (:A)"
        )

    def test_baseline_nodes_with_int_attr_10k(self):
        """10K labeled nodes with a single integer attribute."""
        self._run_workload(
            "nodes_with_int_attr_10k",
            "UNWIND range(1, 10000) AS x CREATE (:A {v: x})"
        )

    def test_baseline_nodes_with_string_attr_10k(self):
        """10K labeled nodes with a short string attribute."""
        self._run_workload(
            "nodes_with_string_attr_10k",
            "UNWIND range(1, 10000) AS x CREATE (:A {v: 'val_' + toString(x)})"
        )

    def test_baseline_edges_10k(self):
        """10K simple (a)-[:R]->(b) edges (=> 20K nodes, 10K edges)."""
        self._run_workload(
            "edges_10k",
            "UNWIND range(1, 10000) AS x CREATE ()-[:R]->()"
        )

    def test_baseline_edges_with_attr_10k(self):
        """10K edges carrying a single integer attribute."""
        self._run_workload(
            "edges_with_attr_10k",
            "UNWIND range(1, 10000) AS x CREATE ()-[:R {w: x}]->()"
        )

    def test_baseline_indexed_nodes_10k(self):
        """10K indexed labeled nodes with one attribute (range index)."""
        # Create the index first so every CREATE feeds the index.
        try:
            self.graph.query("CREATE INDEX FOR (n:A) ON (n.v)")
        except ResponseError:
            # Index may already exist if a previous teardown left state;
            # in that case the failure is benign.
            pass
        self._run_workload(
            "indexed_nodes_10k",
            "UNWIND range(1, 10000) AS x CREATE (:A {v: x})"
        )

    def test_create_delete_cycles(self):
        """Repeated create/delete cycles should not leak memory: after 5
        rounds of creating then deleting 5K nodes, total memory should
        stay close to the baseline of a freshly populated 5K-node graph."""
        for _ in range(5):
            self.graph.query("UNWIND range(1, 5000) AS x CREATE (:A {v: x})")
            self.graph.query("MATCH (n:A) DELETE n")
        # Finally, recreate the dataset so the measurement reflects a
        # stable, populated state (rather than an empty-after-delete one).
        self.graph.query("UNWIND range(1, 5000) AS x CREATE (:A {v: x})")
        measured = self._total_memory_mb()
        self._check_regression("create_delete_cycles_5k", measured)
