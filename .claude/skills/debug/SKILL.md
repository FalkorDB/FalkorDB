---
name: debug
description: Debug FalkorDB crashes, hangs, panics, and incorrect query results - attach lldb/gdb to redis-server with the module loaded, understand panic/crash behavior, step through query execution, and triage failing CI runs. Use when investigating a crash, hang, wrong query result, or a red CI job.
allowed-tools: Bash
---

# Debug

## 1. Attach a debugger to the module

Build a debug binary first (`cargo build`), then launch `redis-server` with
the module loaded under a debugger (mirrors `.vscode/launch.json`):

```bash
# macOS
lldb -- redis-server --loadmodule target/debug/libfalkordb.dylib

# Linux
lldb -- redis-server --loadmodule target/debug/libfalkordb.so
# or: gdb --args redis-server --loadmodule target/debug/libfalkordb.so
```

Then `run` inside the debugger, reproduce the issue from a second terminal
with `redis-cli` (or the failing test/query), and let it trap on
crash/signal. Use `bt` (backtrace), `frame select`/`up`/`down`, and
`thread list` to inspect the failure. Swap `target/debug` for
`target/release` to debug a release build instead.

For a specific failing flow test under the debugger, use RLTest directly
instead of `flow.sh` so it doesn't spawn/manage the server itself (activate
the venv first so `python -m RLTest` resolves — it lives at `/data/venv` in
the devcontainer/CI):
```bash
source venv/bin/activate 2>/dev/null || source /data/venv/bin/activate
lldb -- python -m RLTest --test tests/flow/<file>.py --module target/debug/libfalkordb.so --no-progress -v
```

## 2. Understand crash/panic behavior before you dig in

FalkorDB is a Redis module (`cdylib`) — Redis and the module share one
process, so a Rust panic is not automatically contained the way it would be
in a standalone binary:
- A panic that unwinds across the `extern "C"` boundary back into Redis is
  unsound and typically aborts the whole process rather than just failing a
  command.
- `graph_init` installs a panic hook (`src/module_init.rs`) that captures a
  backtrace unconditionally (`Backtrace::force_capture()` — no
  `RUST_BACKTRACE` needed), logs it via `RedisModule_Log`, then calls
  `std::process::exit(1)`. The thread pool (`graph/src/threadpool.rs`) wraps
  each job in `catch_unwind` to keep a single bad *query* from shrinking the
  worker pool, but for an actual panic the process-wide hook fires first —
  so in practice **"the server crashed" almost always means "something
  panicked"**: check the Redis log (stdout/stderr, or the configured
  `logfile`) for a `FalkorDB panic:` line before assuming memory corruption.

## 3. Step through query execution

- **Query visualizer** (`docs/query_visualizer.md`): an interactive TUI for
  stepping through a query's execution tree and inspecting the environment
  (variable bindings) at each step.
  ```bash
  python tests/record.py
  ```
  Enter a Cypher query, then use the Left/Right arrows to step, or the
  search box with a Python expression (e.g. `x == 5`) to jump to a matching
  step.
- **`GRAPH.EXPLAIN`**: shows the execution plan (the IR tree from
  `planner.rs`/`optimizer.rs`) without running the query — useful to confirm
  which operators (`NodeByLabelScan`, `CondTraverse`, `Filter`, ...) a query
  actually compiles to.
  ```
  GRAPH.EXPLAIN mygraph "MATCH (n:Label) RETURN n"
  ```

## 4. Triage a failing CI run

```bash
gh run list --branch <branch> --limit 5          # find the run
gh run view <run-id> --log-failed                # pull only failed-step logs
```
- Cross-check failures against known-flaky tests before treating them as a
  regression: the `-svc` (replica) flow tests are flaky specifically under
  ASAN/coverage instrumentation due to a pre-existing `atomic_refcell`
  "already mutably borrowed" race — re-run the job once before digging
  further if only those fail and only under `asan`/`coverage` variants.
- If logs mention a fuzz artifact, see the `fuzz` skill for reproducing it
  locally from `fuzz/artifacts/`.
