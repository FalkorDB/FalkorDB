# Proof conventions

## Project layout

```
proofs/<area>/
  lakefile.toml        # name, defaultTargets, [[lean_lib]] (copy an existing one)
  lean-toolchain       # leanprover/lean4:v4.34.0
  .gitignore           # .lake/
  <Root>.lean          # imports the modules; header doc-comment = the report
  <Root>/*.lean        # modules, < ~400 lines each, build after each
  COVERAGE.tsv
  repro.py             # optional: live repros needing RediSearch / a server
```

No Mathlib: use core Lean 4 plus `omega`, `simp`, `decide`, `induction`, and the
`List`/`Array` lemmas in core. Projects are standalone. When you need another
project's definitions, copy them and say where they came from.

## Faithful models

- Follow the Rust line by line: same branch structure, same names, and a `file:line` cite for each modelled fn.
- Model machine integers faithfully where overflow matters: `BitVec 64` or `Int` with Rust's semantics (debug panic vs release wrap, `checked_*`, `wrapping_*`, `as` casts).
- f64 stays abstract: use a float structure or class with only the laws you need. Never add an `axiom` for it.
- FFI (GraphBLAS, LAGraph, RediSearch, the Redis module API, QuickJS, libc): state the C API spec you rely on as a hypothesis structure (preferred), or as an `axiom` with a `-- AXIOM-OK: <C doc>` comment. Those fns are AXIOMATISED.

## COVERAGE.tsv

`file<TAB>function<TAB>line<TAB>bucket<TAB>lean_name_or_reason`, where `file` is the
repo-relative path (`graph/src/...` or `src/...`). There is one row per Rust fn in the
files the project targets; `#` lines are comments.

| bucket | meaning |
| --- | --- |
| PROVEN | a theorem (no sorry) about a faithful model of that fn; for glue or accessors, a small theorem stating exactly what it returns is enough |
| AXIOMATISED | true FFI only |
| MODELLED | modelled, but checked only by `#eval`/`decide`, or no theorem about it |
| NOT COVERED | with a reason; test-only code uses a reason starting `TEST-ONLY:` |

`coverage.sh` keeps each fn's best bucket across projects (PROVEN > AXIOMATISED >
MODELLED > NOT COVERED), and excludes test code (`#[cfg(test)]`, `*_bench.rs`,
`tests.rs`, `test_aux.rs`) from the denominator. The target is 100% of non-FFI,
non-test fns PROVEN, with 0 unlisted.

## Bugs

- Confirmed means a repro against the real Rust shows it. Compare with C where the semantics are C's (C module: the `falkordb/falkordb-server:edge-c` image, or a local `master` build under `bin/`).
- In FINDINGS.md, give each bug an id, `file:line`, the repro, the Rust vs C output, the Lean theorem name, and a suggested fix.
- When main fixes a bug: rename the counterexample to `pre<PR>_…`, add the correctness theorem, and note `fixed by #PR (sha)` in FINDINGS.md.

## Live repros

```bash
cp target/release/libfalkordb.dylib /tmp/fdb_$(date +%s).dylib    # fresh path
redis-server --port 19xyz --loadmodule /tmp/fdb_….dylib --daemonize yes --save ''
# wait for PING (RediSearch init takes a few seconds), run queries, then shutdown nosave
RLTest -t tests/flow/test_x.py --module <dylib> --redis-config-file "$PWD/tests/flow/redis.conf" -p 19xyz
```

Use `-p <port>`, not `--randomize-ports`: `Env()` in `tests/flow/common.py` connects
to the port given by `-p` (default 6379), so randomized runs still hit 6379. Make
sure nothing stale listens on the port you pick.
