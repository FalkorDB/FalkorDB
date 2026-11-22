# redis_layer: the Redis-facing code in `src/`

This project covers the code between a Redis client and the graph engine: everything under `src/` except `redis_type.rs` (which is in [`../graph_persist`](../graph_persist)) and `divergence_guard.rs` (which is in [`../replication`](../replication)). That is:

- **argument parsing** for `GRAPH.QUERY`, `RO_QUERY`, `PROFILE`, `EXPLAIN`, `GRAPH.CONFIG`, module-load arguments, `GRAPH.CONSTRAINT`, `GRAPH.MEMORY`;
- **timeouts**, including how the per-query `TIMEOUT` combines with `TIMEOUT_DEFAULT`/`TIMEOUT_MAX`;
- **the reply encoders**: the compact format the client libraries parse, the verbose format, and the RESP bytes;
- **the `RO_QUERY` write guard**;
- **the `GRAPH.BULK` binary parser**;
- **the RDB byte layer** (`buffered_io.rs`) and the graph header and schema blocks (`serializers/mod.rs`);
- **the rest of `src/`**: query sessions, write dispatch, telemetry, the slow log, the per-thread memory accounting.

Most of it is about agreeing with C. A client that works against C FalkorDB should get the same answer, or the same error, from Rust. Each part is modelled line by line and compared with C (`master`) on live servers.

The full Lean↔Rust mapping is in the header of [`RedisLayer.lean`](RedisLayer.lean) and in the table at the top of each file. This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. Do Rust and C accept the same `GRAPH.QUERY` arguments?**

- **Yes, with one exception (#3757).** Rust refuses a missing or malformed `SCHEMA_HASH` value (`Failed to parse schema hash value`) that C skips as an unknown flag: `rust_c_agree_refuted` and `rust_c_same_accept_refuted` (repro `GRAPH.QUERY g "RETURN 1" SCHEMA_HASH abc`). For every other vector, `rust_c_agree_except_schema`: under the invariant `GRAPH.CONFIG` maintains (`CfgOk`), Rust's `parse_query_flags` and C's `_read_flags` either both fail, with corresponding errors, or both succeed with the same compact flag, version and timeout. `rust_ok_c_ok`: whatever Rust accepts, C accepts with the same flags.

Related results about the parser itself:

- `parse_ok_iff`: it accepts exactly 3 to 8 arguments whose tail fits the documented grammar;
- `parse_documented`: every documented command line is accepted with its documented meaning;
- `four_commands_share`: `QUERY`, `RO_QUERY`, `PROFILE` and `EXPLAIN` reject the same vectors and read the same flags.

**2. Does a write query honour `TIMEOUT`?**

- **Yes, once `TIMEOUT_MAX` or `TIMEOUT_DEFAULT` is set.** The per-query value then applies to writes exactly as to reads (`rustTimeout_write_pq`, `rustTimeout_rw_uniform`). Only under the deprecated `TIMEOUT` setting on its own do writes ignore it, as in C (`rustTimeout_write_ignores_pq`).
- The effective timeout never exceeds `TIMEOUT_MAX` (`rustTimeout_le_max`).

**3. Does `GRAPH.CONFIG SET` keep the configuration consistent?**

Yes:

- `rustSet_inv`: every successful `SET` batch leaves `TIMEOUT_DEFAULT ≤ TIMEOUT_MAX`. C checks a multi-pair batch one pair at a time and can break this (`c_breaks_inv`); here Rust is right.
- Rust also validates values the way C does:
  - `VKEY_MAX_ENTITY_COUNT -5` is refused (`vkey_ok_nonneg`);
  - `JS_HEAP_SIZE` below 1 MB is refused (`jsheap_ok_ge`);
  - `CMD_INFO` accepts exactly C's spellings (`cmdinfo_ok_iff`);
  - `GET x extra` is a wrong-arity error (`get_extra_wrong_arity`);
  - names fold case only over ASCII, so `tımeout` is not `TIMEOUT` (`dotless_i_not_folded`).
- Module-load arguments are parsed by the same rules as C's `Config_Init` (`rust_c_same_accept` in `LoadArgs.lean`), and a refused argument applies nothing (`error_applies_nothing`). The theorem assumes no embedded NUL in names and values (C reads C strings and stops at the first NUL, Rust keeps the whole string, so `CACHE_SIZE "10\0junk"` loads on C and is refused by Rust; not live-checked) and `FsOk` (`File::create` agrees with `access(W_OK)`). Two further differences remain:
  - C skips leading whitespace in numbers and Rust does not, so `CACHE_SIZE " 10"` loads on C and is refused by Rust (`leading_space_divergence`);
  - Rust accepts `INDEX_WORKER_THREADS`, `BOLT_PORT`, `EFFECTS_COMPRESSION` and `INDEX_BUILD_THREAD_COUNT` (range 1 to 1024), which C does not know (`rust_only_names`). These are intended.
- A `VKEY_MAX_ENTITY_COUNT` value set by `GRAPH.CONFIG SET` or a module-load argument survives `CONFIG REWRITE` and a restart: the rewritten `graph.VKEY_MAX_ENTITY_COUNT` line is inside the native config's range (`0..i64::MAX`) and reads back the same value (`vkey_rewrite_reloads_config`, `vkey_rewrite_reloads_load` in `ConfigRewrite.lean`). Other native configs can still abort the next start (bug 16 below).

**4. Does the compact reply decode to the right values in the client?**

Yes. `dec_enc` and `decReply_enc`: for every value (scalars, temporal values, lists, maps, nodes, edges, paths, vectors, points) and for a whole reply, the compact encoding that `reply.rs` writes decodes in falkordb-py's parser to the intended value. Floats are compared after formatting, exactly as C formats them. `compact_emits` and `verbose_emits` prove that the `RedisModule_Reply*` call stream, postponed array lengths included, produces exactly that reply tree, and `bytes_of_emits` gives the RESP2 bytes. `postponed_wrong_len` shows that a wrong postponed count would break it.

The verbose format prints floats and points as C does: a NaN inside a list or vector is `nan` (`fmtS_floats_c`, `fmtS_vec_c`), and a top-level point is `point({latitude:…, longitude:…})` without a space after the colons (`pointStr_c`, `verbose_point_c`). Both are stated against C's `%f` as a hypothesis (`CPf`).

**5. Can `RO_QUERY` ever write?**

No: `ro_never_writes`. An `RO_QUERY` that is not rejected performs no mutation, and the query contained neither a mutating clause nor a write procedure. Write detection is split over three places: the parser's write flag, the planner's Commit insertion, and a runtime guard on procedure calls. `write_proc_invisible_to_scan` shows the runtime guard is needed: a query whose only write is a procedure call such as `CALL db.idx.fulltext.createNodeIndex(…)` has a read-only plan, and only the guard stops it.

**6. Does `GRAPH.BULK` read back exactly what the loader wrote?**

Yes: `readProperty_enc`, `go_encs`, `records_enc`. The bulk reader reads back exactly the property values the loader encoded, and a node token imports exactly its records, in order. `ceiling_sound`: the `max_records` pre-check never rejects an honest payload. Nesting deeper than the cap is refused (`deep_refused`).

A header that names a property twice (`p,q,p`) still gives each record one entry per attribute: `readRecord_enc` proves `PropColumns::read_record` returns `keepFirst [] (ids.zip vs)` for every record, whatever the previous record left in `filled`. `keepFirst_nodup`: no attribute appears twice. `keepFirst_find`: the kept value is the first non-null one in header order, so a null in the first `p` column lets the second supply the value (`repeated_header_null_first`). `keepFirst_sublist`: column order is kept. `readRecord_plain`: with no repeated name, the result is the plain per-column read. `keepFirst_plainRead`: the result equals what `import_attrs_resolved` keeps.

**7. Is the RDB byte layer lossless?**

Yes:

- `writeAll_flatten`: buffered writes equal the concatenation of the logical writes for every buffer size, so the flush points are invisible;
- `roundtrip`: anything written reads back exactly, for any chunk source that honours the load contract. `rdb_roundtrip` (RDB string buffers), `pipe_roundtrip` (`GRAPH.COPY`) and `vec_roundtrip` (`GRAPH.RESTORE`) are instances. `i64` and the f64/f32 bit patterns are exact;
- `decH_encH`, `decSchema_encSchema`: the graph header and schema blocks round-trip, the schema up to an explicit normal form (`normSchema`). An index field named `range:x` keeps its name (`field_name_kept`).

`graph_persist` uses the header and schema results as hypotheses.

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `rust_c_agree_except_schema` | For every argument vector, unless Rust refuses a `SCHEMA_HASH` value or a `SCHEMA_HASH` without `--compact`, Rust's query-flag parser and C's agree on success or failure and on every flag; `rust_c_agree_refuted` (#3757) shows the exception is real, `rust_c_agree_except_schemaParse_refuted` that the second one is needed | `QueryArgsAgree.lean` |
| `parse_ok_iff`, `parse_documented`, `four_commands_share` | What the parser accepts, exactly; the four query commands share it | `QueryArgsSpec.lean`, `QueryArgs.lean` |
| `rustTimeout_le_max`, `rustTimeout_rw_uniform` | The effective timeout is bounded and, once configured, applies to writes as to reads | `Args.lean` |
| `rustSet_inv` | `GRAPH.CONFIG SET` keeps `TIMEOUT_DEFAULT ≤ TIMEOUT_MAX` | `Config.lean` |
| `rust_c_same_accept` (`LoadArgs`) | Module-load arguments are accepted exactly as C accepts them, up to two documented differences, for arguments without embedded NUL, given `FsOk` | `LoadArgs.lean` |
| `vkey_rewrite_reloads_config`, `vkey_rewrite_reloads_load` | A `VKEY_MAX_ENTITY_COUNT` value set at run time or at load survives `CONFIG REWRITE` and a restart | `ConfigRewrite.lean` |
| `dec_enc`, `decReply_enc` | The compact reply decodes in the Python client to the intended values | `Compact.lean` |
| `compact_emits`, `verbose_emits`, `compact_reply`, `bytes_of_emits` | The reply call streams produce exactly the intended reply and RESP bytes | `RespCompact.lean`, `RespVerbose.lean`, `Resp.lean` |
| `ro_never_writes` | An `RO_QUERY` that runs performs no write | `RoQuery.lean` |
| `readProperty_enc`, `records_enc`, `ceiling_sound` | `GRAPH.BULK` reads back exactly what was written; the size pre-check is sound | `Bulk.lean` |
| `readRecord_enc`, `keepFirst_nodup`, `keepFirst_find` | A `GRAPH.BULK` record keeps one entry per attribute: the first non-null value in header order | `Bulk.lean` |
| `writeAll_flatten`, `roundtrip` | The RDB byte layer is lossless for every buffer size and chunk source | `BufferedIO.lean`, `RoundTrip.lean` |
| `decH_encH`, `decSchema_encSchema` | Graph header and schema blocks round-trip | `Serial.lean`, `SerialProof.lean` |
| `hashInput_total`, `truncate_total`, `prefix_longest` | The slow log never panics, and keeps the longest prefix of at most 2048 bytes that ends on a character boundary | `SlowLog.lean` |
| `checkFork_spec`, `current_kept`, `executeWrite_first`, `admit_bounded` | Write sessions: a check succeeds only on a fork of the current version, a `Current` one is never refused as stale; a stale attempt is invisible; the queue never exceeds `MAX_QUEUED_QUERIES` | `QuerySession.lean`, `GraphCore.lean` |
| `sync_unserved_silent`, `freed_stream_stripped`, `stripAll_covers`, `iterSteps_delete_before_write` | Telemetry is node-local (#3844): replicated or AOF-replayed queries are not logged, no stream reaches a snapshot, freed streams are deleted before any write | `GraphCore.lean`, `TelemetryFreed.lean`, `TelemetryLoop.lean` |
| `query_accounting`, `other_thread_invisible` | Per-query memory accounting counts exactly the query's own allocations, on its own thread | `Allocator.lean` |

### How `rust_c_agree_except_schema` is proved

1. **Model the bytes.** Arguments are byte lists, so non-UTF-8 input is a legal input. `eqIC` is ASCII case-insensitive comparison, which both engines use for flag names.
2. **Model both parsers.** `parseQueryFlags` and `scanFlags` follow `parse_query_flags` in `query_args.rs` line by line. `cReadFlags` follows C's `_read_flags` plus `_validate_command_arity` from `cmd_dispatcher.c`. Integers go through `string2ll` (Redis's own parser, which both engines end up calling) on both sides.
3. **Relate the results.** `Rel` says what a Rust flag record means in C's terms: same compact flag, same version, same timeout. C replaces `TIMEOUT 0` by the default in the parser, and Rust does it later in `compute_effective_timeout`; `Rel` accounts for that. `QErr.toC` maps each Rust error to the C error it corresponds to.
4. **Induct over the flag loop.** `scan_agree` proves the two loops stay related at every step, flag by flag. The small lemmas `k1` … `k17` say no two flag names are equal ignoring case, so at most one branch matches.
5. **The hypothesis `CfgOk`** says configured timeouts are non-negative and `TIMEOUT_DEFAULT ≤ TIMEOUT_MAX` when a maximum is set. `rustSet_inv` proves `GRAPH.CONFIG SET` keeps it, so it holds on any running server.

## Bugs found

All confirmed live against C (`repro_live.py` in this folder, and the scripts named in [`../FINDINGS.md`](../FINDINGS.md)).

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| 10 | `leading_space_divergence` | `CACHE_SIZE " 10"` and `GRAPH.CONFIG SET RESULTSET_SIZE " 5"` succeed on C and are refused by Rust | **Open** (confirmed live, not filed) |
| 13 | none | `CREATE (a:A) DELETE a` reports no node stats in the verbose format (minor) | **Open**, not filed |
| 14 | `bug2782_same_name_other_db_aborts`, `bug2782_invariant_broken` (`MultiDb.lean`) | `GRAPH_REGISTRY` is keyed by name only, so creating `shared` in db1 displaces db0's `shared`; the next write to db0's graph aborts ("graph was deleted or replaced…"; debug build: assert). Property `KeyspaceRegistered` / `WritesSucceed` | **Open** (#2782) |
| 15 | `bug2653_nonutf8_key_mangled` (`KeyName.lean`) | A non-UTF-8 graph key is replicated and signalled under its `from_utf8_lossy` form: `\xc3(bin` → `\xef\xbf\xbd(bin`, so the replica writes a different key. Property `ReplicatedKeyIsStored` (C: `c_ok`) | **Open** (#2653) |
| 16 | `rewrite_reload_counterexamples` (`ConfigRewrite.lean`) | `CONFIG REWRITE` saves `graph.<NAME> <value>` for what `GRAPH.CONFIG SET` or a load argument stored; a value outside the native config's range aborts the next start | **Open**, not filed. Confirmed live (FINDINGS): `CACHE_SIZE 2000` as a load argument and `GRAPH.CONFIG SET JS_HEAP_SIZE 5000000000`. Model only, not live-checked: `THREAD_COUNT` > 1024, `INDEX_WORKER_THREADS` > 1024, `NODE_CREATION_BUFFER` > 2^30 as load arguments, `JS_HEAP_SIZE` > 2^32 as a load argument, `JS_STACK_SIZE` > 2^32 by either path |
| 19 | `bug3447_compact_cancelled_new_type_panics` (`RelLookup.lean`) | **Crash.** `CREATE (a)-[r:NEW_TYPE]->(b) DETACH DELETE a RETURN r --compact`: the cascaded edge is snapshotted, but its type is never registered (`Pending::commit` registers only surviving groups, `pending.rs:872-873`, `:1242-1251`), so `get_type_id(..).unwrap()` (`reply.rs:260`) panics; verbose works. Property `RegisteredTypes` ⇒ `ReplyTotal` once nothing is pending (`replyTotal_of_registered` also needs `s.pending = []`: a pending relationship can satisfy `RegisteredTypes` while the modelled lookups still fail) | **Open** (#3447, compact-reply part) |

Also in the verbose format: a node deleted earlier in the same query still shows its labels (C shows `[]`). That one is intended: it was decided on 2026-09-25 that a deleted entity keeps all its details for the rest of the query.

C-side bugs seen along the way: C crashes on some malformed bulk tokens, and counts `L:L` twice.

## Files

| File | What's in it |
|---|---|
| `RedisLayer.lean` | Root: the report header (Lean↔Rust tables, theorems, divergences, gaps) |
| `Args.lean` | Integer parsers (`rustParseI64`, `string2ll`) and `compute_effective_timeout` (`rustTimeout`) |
| `QueryArgs.lean`, `QueryArgsSpec.lean`, `QueryArgsAgree.lean` | `parse_query_flags`; what it accepts exactly; agreement with C's `_read_flags` |
| `Config.lean` | `GRAPH.CONFIG`: validation, cross-field checks, `SET`, dispatch; C's `_Config_set` |
| `LoadArgs.lean`, `ModuleInit.lean` | Module-load arguments, the configuration table, event handlers |
| `ConfigRewrite.lean` | `CONFIG REWRITE` of the native configs against what the other paths store |
| `Compact.lean` | Compact value encoding and the falkordb-py decoder |
| `Resp.lean`, `RespCompact.lean`, `RespVerbose.lean` | Reply call streams, postponed lengths, RESP2 bytes; verbose values and statistics |
| `RoQuery.lean` | The `RO_QUERY` write guard |
| `Bulk.lean` | The `GRAPH.BULK` binary format: property reader, records, header columns (`PropColumns`), counts, the `as u16` span length |
| `BufferedIO.lean`, `BufferedReader.lean`, `RoundTrip.lean` | `buffered_io.rs`: the type-tagged byte layer, its readers, the write→read round trip |
| `Serial.lean`, `SerialSchema.lean`, `SerialEntry.lean`, `SerialProof.lean` | `serializers/mod.rs`: graph header, schema blocks, index fields, constraints, and their round trip |
| `QuerySession.lean` | The per-query lock and GIL protocol in `query_session.rs` |
| `GraphCore.lean` | `graph_core.rs`: names, registry, write commit, dispatch, queue admission |
| `MultiDb.lean` | #2782: several Redis databases over the name-keyed registry; `graph_query`'s create/open path |
| `KeyName.lean` | #2653: key names as bytes, a concrete `from_utf8_lossy`, the replicated key |
| `ErrorText.lean` | `str_as_legal_resp_string` (error and simple-string reply text) on bytes |
| `Commands.lean`, `Commands2.lean` | The smaller command handlers: constraint, record, copy, bulk, memory, config get, and others |
| `Telemetry.lean`, `TelemetryReg.lean`, `TelemetryLoop.lean`, `TelemetryFreed.lean` | `telemetry.rs`: stream names, the query registry, batching, the flusher, the persistence strip and freed-graph stream queue |
| `SlowLog.lean` | `slow_log.rs` (`MinCache`, `SlowestKept`, `add`) |
| `RelLookup.lean` | #3447 (open): the relationship lookups behind the reply encoders (endpoints, type id/name over graph, snapshots and pending), the cascade commit and `GRAPH.RECORD` |
| `Allocator.lean` | `allocator.rs`: per-thread allocation counters |
| `repro_live.py` | Live Rust-vs-C repros for the divergences above |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

All `.lean` files except the root are under `RedisLayer/`. The project has no Lake dependencies.

## Coverage

`COVERAGE.tsv` has 297 rows across `src/`. The biggest files are `buffered_io.rs` (55), `graph_core.rs` (46), `telemetry.rs` (36), `query_session.rs` (24) and `bulk_insert.rs` (22).

| Bucket | Rows |
|---|---|
| PROVEN | 253 |
| CONDITIONAL | 1 |
| AXIOMATISED | 34 |
| NOT COVERED | 9 |

- **AXIOMATISED:** wrappers around the Redis module API (`RedisModule_*`), libc (`pipe`, `fork`, `_exit`, `waitpid`, `pthread_atfork`), the OS clock and file-descriptor I/O. Most are in `graph_core.rs` (15).
- **CONDITIONAL:** `QuerySession::check_fork`; its reason column names the hypothesis.
- **NOT COVERED:** `sync_deepbind_environ` (`src/module_init.rs`, Linux-only, bound to ld.so and libc); `RedisBuildHost::publish` and `RedisBuildHost::warn` (`src/graph_core.rs`, the online index build, not modelled yet); and the six functions of `src/divergence_guard.rs`. Those six rows' reason column says "covered by proofs/replication narrative", but `replication` now models them line by line in `Guard.lean` and lists them as PROVEN; `bash proofs/coverage.sh` counts each function once, in its best bucket.

## What the model does not cover

- **The Redis module API, libc, the clock and file I/O** are hypotheses (`Compact.Env`, `Bulk.Codec`) or AXIOMATISED, never Lean axioms.
- **Concurrency** is modelled per operation, or as a single-thread log of events: the telemetry flusher, the write-queue election and GIL nesting are proven as protocols, not under an interleaving semantics.
- **Number formatting** is abstract. Compact vector elements use `RedisModule_ReplyWithDouble` (`Env.vfmt`). Glibc prints a NaN with the sign bit set as `-nan`, which `write_f` does not reproduce (`CPf.nan_text` is stated for `nan`).
- **A refused `open_key_writable(..)?`**: the server's own error text is not modelled.
- **Compact floats**: floats in the compact reply are compared after formatting, which matches C, not as exact values.
- **`serializers/encoder` and `decoder`** are proven in `graph_persist`, **`divergence_guard.rs`** in `replication`, and **`redis_type.rs`** in `graph_persist`.

## Build and check

```bash
(cd proofs/redis_layer && lake build)    # this project
bash proofs/lean_ci.sh redis_layer     # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                # repo-wide coverage report
venv/bin/python proofs/redis_layer/repro_live.py   # live Rust-vs-C repros (Rust on 18430, C on 18431)
```

**When you change anything under `src/` (other than `redis_type.rs` and `divergence_guard.rs`):**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match.
3. Re-run the build.

Changing argument parsing or reply formatting usually means re-checking against C as well: `rust_c_agree_except_schema` and the `LoadArgs` agreement theorem compare against a model of C, so a change that is meant to diverge from C must say so in the theorem. If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR.
