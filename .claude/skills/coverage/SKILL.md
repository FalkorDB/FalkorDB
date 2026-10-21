---
name: coverage
description: Measure FalkorDB code coverage with LLVM source-based instrumentation across Rust unit tests, Python e2e/MVCC/concurrency tests, flow tests, and TCK - producing an lcov report. Use when asked to measure/report test coverage or reproduce the CI coverage job locally.
allowed-tools: Bash
---

# Coverage

LLVM source-based coverage (`-C instrument-coverage`) over the same suites CI
runs. `llvm-profdata`/`llvm-cov` must match the LLVM that built the binary —
the devcontainer ships `llvm-profdata-22`/`llvm-cov-22`; locally use whatever
suffix matches your LLVM (drop the suffix if you only have one). Also needs
`lcov` (`brew install lcov` / `apt install lcov`).

## Full coverage (Rust unit + Python + flow + TCK)

```bash
# clean stale coverage data first
find . -name "*.profraw" -delete
rm -f cov.profdata codecov.txt codecov.txt.all

# build + Rust unit tests, instrumented.
# The link-arg is required on Linux/CI so the embedded RediSearch static libs
# link without duplicate-symbol errors; drop it with macOS's linker. RUSTDOCFLAGS
# instruments doctests too, matching CI.
COV_RUSTFLAGS="-C instrument-coverage -C link-arg=-Wl,--allow-multiple-definition"
RUSTFLAGS="$COV_RUSTFLAGS" RUSTDOCFLAGS="$COV_RUSTFLAGS" cargo build
RUSTFLAGS="$COV_RUSTFLAGS" RUSTDOCFLAGS="$COV_RUSTFLAGS" cargo test -p graph

# instrumented Python + flow + TCK suites (module already built above)
source venv/bin/activate 2>/dev/null || source /data/venv/bin/activate
pytest tests/test_e2e.py tests/test_functions.py tests/test_mvcc.py tests/test_concurrency.py -vv
./flow.sh
TCK_DONE=tck_done.txt pytest tests/tck/test_tck.py -s

# pass each unit-test binary as an extra -object so llvm-cov also credits
# coverage recorded by test-only code paths (CI does this; without it those
# lines under-report)
TEST_OBJECTS=""
for bin in $(RUSTFLAGS="$COV_RUSTFLAGS" cargo test -p graph --no-run 2>&1 \
    | grep -w Executable | grep -oE 'target/debug/deps/[^)]+'); do
  TEST_OBJECTS="$TEST_OBJECTS -object $bin"
done

# merge the .profraw files and export lcov (libfalkordb.dylib on macOS)
llvm-profdata-22 merge --sparse $(find . -name "*.profraw") -o cov.profdata
llvm-cov-22 export --format=lcov --instr-profile cov.profdata target/debug/libfalkordb.so $TEST_OBJECTS > codecov.txt.all
lcov --ignore-errors unused -r codecov.txt.all -o codecov.txt
llvm-cov-22 report --instr-profile cov.profdata target/debug/libfalkordb.so $TEST_OBJECTS   # console summary
```

Narrow the run when you don't need whole-project numbers:
- **Rust unit only** — stop after `cargo test -p graph` and skip straight to
  the merge/export step (fast local check).
- **Quick** — run everything except `./flow.sh` (flow tests dominate the
  wall-clock).

## Running outside a devcontainer

The versioned `llvm-*-22` tools and `venv` at `/data/venv` live in the
devcontainer. To reproduce the exact CI environment from a plain host, run
the same sequence inside the container image:

```bash
docker build -t falkordb-dev -f .devcontainer/Dockerfile .
docker run --rm -v $(pwd):/workspace -w /workspace falkordb-dev bash -c "
  find . -name '*.profraw' -delete
  CF='-C instrument-coverage -C link-arg=-Wl,--allow-multiple-definition'
  RUSTFLAGS=\"\$CF\" RUSTDOCFLAGS=\"\$CF\" cargo build
  RUSTFLAGS=\"\$CF\" RUSTDOCFLAGS=\"\$CF\" cargo test -p graph
  source /data/venv/bin/activate
  pytest tests/test_e2e.py tests/test_functions.py tests/test_mvcc.py tests/test_concurrency.py -vv
  ./flow.sh
  TCK_DONE=tck_done.txt pytest tests/tck/test_tck.py -s
  TEST_OBJECTS=''
  for bin in \$(RUSTFLAGS=\"\$CF\" cargo test -p graph --no-run 2>&1 | grep -w Executable | grep -oE 'target/debug/deps/[^)]+'); do TEST_OBJECTS=\"\$TEST_OBJECTS -object \$bin\"; done
  llvm-profdata-22 merge --sparse \$(find . -name '*.profraw') -o cov.profdata
  llvm-cov-22 export --format=lcov --instr-profile cov.profdata target/debug/libfalkordb.so \$TEST_OBJECTS > codecov.txt.all
  lcov --ignore-errors unused -r codecov.txt.all -o codecov.txt
  llvm-cov-22 report --instr-profile cov.profdata target/debug/libfalkordb.so \$TEST_OBJECTS
"
```

## Output & notes

- `codecov.txt` is the filtered lcov report (upload this to Codecov);
  `codecov.txt.all` is the raw pre-filter export; `cov.profdata` is the merged
  profile; `*.profraw` are per-process files safe to delete afterward.
- Browse locally with `genhtml codecov.txt -o cov_html`.
- Flow tests require the instrumented debug build to exist first.
- Outside the devcontainer, the first run is slow because it builds the image.

If coverage fails, report the errors clearly and help diagnose.
