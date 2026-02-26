---
name: Run FalkorDB tests
description: Run unit tests, flow tests, TCK tests, and other test suites with filtering and parallel execution
---

# Run FalkorDB tests

Run the FalkorDB test suites: unit (C), flow (Python/RLTest), TCK (OpenCypher compliance), upgrade, and fuzz.

## Usage

Install Python test dependencies first: `pip install -r tests/requirements.txt`

## Run all tests

    make test

## Individual test suites

    make unit-tests            # C unit tests
    make flow-tests            # Python integration tests (RLTest)
    make tck-tests             # OpenCypher compliance tests (Behave)

## Run a specific test

    make flow-tests TEST=test_aggregation
    make unit-tests TEST=test_delta_matrix
    make tck-tests TEST=test_name

## Parallel execution

    make flow-tests PARALLEL=4
    make tck-tests PARALLEL=4
    make test PARALLEL=4

## List tests without running

    make test LIST=1
    make flow-tests LIST=1
    make unit-tests LIST=1

## Run with sanitizers or Valgrind

    make flow-tests SAN=address
    make unit-tests VG=1
    make flow-tests VG=1 VG_LEAKS=1

## Verbose output

    make flow-tests VERBOSE=1

## Coverage

    make coverage              # Build with coverage and run all tests

## Notes

- Flow tests (`tests/flow/test_*.py`): Python integration tests using RLTest framework; each test class gets a live Redis+FalkorDB instance
- Unit tests (`tests/unit/test_*.c`): Compiled C executables, one per test file
- TCK tests (`tests/tck/`): OpenCypher compliance tests using Python Behave (BDD)
- Flow test pattern: class with `__init__` calling `Env()` and `db.select_graph()`, methods named `test01_*`, `test02_*`, etc.
- Test logs are written to `tests/<suite>/logs/`
