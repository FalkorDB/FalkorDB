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

- **Before #3010, no.** Some examples:
  - a non-UTF-8 argument ended Rust's flag loop, so a later `--compact` was ignored;
  - `TIMEOUT -5`, `TIMEOUT +10` and `TIMEOUT 010` were accepted;
  - `RO_QUERY` and `PROFILE` ignored a garbage or missing timeout;
  - `version 4294967296` was accepted;
  - there was no limit on the number of arguments.
- **On main, yes, for every argument vector.** `rust_c_agree`: under the invariant `GRAPH.CONFIG` maintains (`CfgOk`), Rust's `parse_query_flags` and C's `_read_flags` either both fail, with corresponding errors, or both succeed with the same compact flag, version and timeout. `rust_c_same_accept` is the "accept the same vectors" corollary.

Related results about the parser itself:

- `parse_ok_iff`: it accepts exactly 3 to 8 arguments whose tail fits the documented grammar;
- `parse_documented`: every documented command line is accepted with its documented meaning;
- `four_commands_share`: `QUERY`, `RO_QUERY`, `PROFILE` and `EXPLAIN` reject the same vectors and read the same flags.

**2. Does a write query honour `TIMEOUT`?**

- **Before #3000, no.** With `TIMEOUT_MAX` set to 100000, `UNWIND range(1,300000) AS x CREATE (:N {x:x}) TIMEOUT 1` created all 300000 nodes on Rust; C timed out and created none (`pre3000_write_timeout_divergence`).
- **On main, yes, once `TIMEOUT_MAX` or `TIMEOUT_DEFAULT` is set.** The per-query value then applies to writes exactly as to reads (`rustTimeout_write_pq`, `rustTimeout_rw_uniform`). Only under the deprecated `TIMEOUT` setting on its own do writes ignore it, as in C (`rustTimeout_write_ignores_pq`).
- The effective timeout never exceeds `TIMEOUT_MAX` (`rustTimeout_le_max`).

**3. Does `GRAPH.CONFIG SET` keep the configuration consistent?**

Yes:

- `rustSet_inv`: every successful `SET` batch leaves `TIMEOUT_DEFAULT ≤ TIMEOUT_MAX`. C checks a multi-pair batch one pair at a time and can break this (`c_breaks_inv`); here Rust is right.
- Since #3021 Rust also validates values the way C does:
  - `VKEY_MAX_ENTITY_COUNT -5` is refused (`vkey_ok_nonneg`);
  - `JS_HEAP_SIZE` below 1 MB is refused (`jsheap_ok_ge`);
  - `CMD_INFO` accepts exactly C's spellings (`cmdinfo_ok_iff`);
  - `GET x extra` is a wrong-arity error (`get_extra_wrong_arity`);
  - names fold case only over ASCII, so `tımeout` is not `TIMEOUT` (`dotless_i_not_folded`).
- Since #3328 module-load arguments are parsed by the same rules as C's `Config_Init` (`rust_c_same_accept` in `LoadArgs.lean`), and a refused argument applies nothing (`error_applies_nothing`). Two differences remain:
  - C skips leading whitespace in numbers and Rust does not, so `CACHE_SIZE " 10"` loads on C and is refused by Rust (`leading_space_divergence`);
  - Rust accepts `INDEX_WORKER_THREADS`, `BOLT_PORT` and `EFFECTS_COMPRESSION`, which C does not know (`rust_only_names`). These are intended.
- Since #3419 a `VKEY_MAX_ENTITY_COUNT` value set by `GRAPH.CONFIG SET` or a module-load argument survives `CONFIG REWRITE` and a restart: the rewritten `graph.VKEY_MAX_ENTITY_COUNT` line is inside the native config's range (`0..i64::MAX`) and reads back the same value (`vkey_rewrite_reloads_config`, `vkey_rewrite_reloads_load` in `ConfigRewrite.lean`). Before, `0` and anything above 2^30 (FalkorDB Cloud loads with `4611686000000000000`) made the next start abort (`pre3419_rewrite_refused`, #3413). Other native configs still have this problem (bug 16 below).

**4. Does the compact reply decode to the right values in the client?**

Yes. `dec_enc` and `decReply_enc`: for every value (scalars, temporal values, lists, maps, nodes, edges, paths, vectors, points) and for a whole reply, the compact encoding that `reply.rs` writes decodes in falkordb-py's parser to the intended value. Floats are compared after formatting, exactly as C formats them. `compact_emits` and `verbose_emits` prove that the `RedisModule_Reply*` call stream, postponed array lengths included, produces exactly that reply tree, and `bytes_of_emits` gives the RESP2 bytes. `postponed_wrong_len` shows that a wrong postponed count would break it.

Since #3043 the verbose format prints floats and points as C does: a NaN inside a list or vector is `nan` (`fmtS_floats_c`, `fmtS_vec_c`), and a top-level point is `point({latitude:…, longitude:…})` without a space after the colons (`pointStr_c`, `verbose_point_c`). Both are stated against C's `%f` as a hypothesis (`CPf`).

**5. Can `RO_QUERY` ever write?**

No: `ro_never_writes`. An `RO_QUERY` that is not rejected performs no mutation, and the query contained neither a mutating clause nor a write procedure. Write detection is split over three places: the parser's write flag, the planner's Commit insertion, and a runtime guard on procedure calls. `write_proc_invisible_to_scan` shows the runtime guard is needed: a query whose only write is a procedure call such as `CALL db.idx.fulltext.createNodeIndex(…)` has a read-only plan, and only the guard stops it.

**6. Does `GRAPH.BULK` read back exactly what the loader wrote?**

Yes: `readProperty_enc`, `go_encs`, `records_enc`. The bulk reader reads back exactly the property values the loader encoded, and a node token imports exactly its records, in order. `ceiling_sound`: the `max_records` pre-check never rejects an honest payload. Since #3339 nesting deeper than the cap is refused (`deep_refused`).

Since #3026 a header that names a property twice (`p,q,p`) still gives each record one entry per attribute: `readRecord_enc` proves `PropColumns::read_record` returns `keepFirst [] (ids.zip vs)` for every record, whatever the previous record left in `filled`. `keepFirst_nodup`: no attribute appears twice. `keepFirst_find`: the kept value is the first non-null one in header order, so a null in the first `p` column lets the second supply the value (`repeated_header_null_first`). `keepFirst_sublist`: column order is kept. `readRecord_plain`: with no repeated name, the result is the plain per-column read. `keepFirst_plainRead`: the result equals what #3063's `import_attrs_resolved` already kept from the old read, so no stored entity changes.

**7. Is the RDB byte layer lossless?**

Yes:

- `writeAll_flatten`: buffered writes equal the concatenation of the logical writes for every buffer size, so the flush points are invisible;
- `roundtrip`: anything written reads back exactly, for any chunk source that honours the load contract. `rdb_roundtrip` (RDB string buffers), `pipe_roundtrip` (`GRAPH.COPY`) and `vec_roundtrip` (`GRAPH.RESTORE`) are instances. `i64` and the f64/f32 bit patterns are exact;
- `decH_encH`, `decSchema_encSchema`: the graph header and schema blocks round-trip, the schema up to an explicit normal form (`normSchema`). Since #3252 an index field named `range:x` keeps its name (`field_name_kept`).

`graph_persist` uses the header and schema results as hypotheses.

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `rust_c_agree` | For every argument vector, Rust's query-flag parser and C's agree on success or failure and on every flag | `QueryArgsAgree.lean` |
| `parse_ok_iff`, `parse_documented`, `four_commands_share` | What the parser accepts, exactly; the four query commands share it | `QueryArgsSpec.lean`, `QueryArgs.lean` |
| `rustTimeout_le_max`, `rustTimeout_rw_uniform` | The effective timeout is bounded and, once configured, applies to writes as to reads | `Args.lean` |
| `rustSet_inv` | `GRAPH.CONFIG SET` keeps `TIMEOUT_DEFAULT ≤ TIMEOUT_MAX` | `Config.lean` |
| `rust_c_same_accept` (`LoadArgs`) | Module-load arguments are accepted exactly as C accepts them, up to two documented differences | `LoadArgs.lean` |
| `vkey_rewrite_reloads_config`, `vkey_rewrite_reloads_load` | A `VKEY_MAX_ENTITY_COUNT` value set at run time or at load survives `CONFIG REWRITE` and a restart (#3419) | `ConfigRewrite.lean` |
| `dec_enc`, `decReply_enc` | The compact reply decodes in the Python client to the intended values | `Compact.lean` |
| `compact_emits`, `verbose_emits`, `compact_reply`, `bytes_of_emits` | The reply call streams produce exactly the intended reply and RESP bytes | `RespCompact.lean`, `RespVerbose.lean`, `Resp.lean` |
| `ro_never_writes` | An `RO_QUERY` that runs performs no write | `RoQuery.lean` |
| `readProperty_enc`, `records_enc`, `ceiling_sound` | `GRAPH.BULK` reads back exactly what was written; the size pre-check is sound | `Bulk.lean` |
| `readRecord_enc`, `keepFirst_nodup`, `keepFirst_find` | A `GRAPH.BULK` record keeps one entry per attribute: the first non-null value in header order (#3026) | `Bulk.lean` |
| `writeAll_flatten`, `roundtrip` | The RDB byte layer is lossless for every buffer size and chunk source | `BufferedIO.lean`, `RoundTrip.lean` |
| `decH_encH`, `decSchema_encSchema` | Graph header and schema blocks round-trip | `Serial.lean`, `SerialProof.lean` |
| `hashInput_total`, `truncate_total`, `prefix_longest` | Since #3234 the slow log never panics, and keeps the longest prefix of at most 2048 bytes that ends on a character boundary | `SlowLog.lean` |
| `claimFork_spec`, `executeWrite_first`, `admit_bounded` | Write sessions: a claim succeeds only on a fork of the current version (#3290); a stale attempt is invisible; the queue never exceeds `MAX_QUEUED_QUERIES` (#3014) | `QuerySession.lean`, `GraphCore.lean` |
| `query_accounting`, `other_thread_invisible` | Per-query memory accounting counts exactly the query's own allocations, on its own thread | `Allocator.lean` |

### How `rust_c_agree` is proved

1. **Model the bytes.** Arguments are byte lists, so non-UTF-8 input is a legal input. `eqIC` is ASCII case-insensitive comparison, which both engines use for flag names.
2. **Model both parsers.** `parseQueryFlags` and `scanFlags` follow `parse_query_flags` in `query_args.rs` line by line. `cReadFlags` follows C's `_read_flags` plus `_validate_command_arity` from `cmd_dispatcher.c`. Integers go through `string2ll` (Redis's own parser, which both engines end up calling) on both sides.
3. **Relate the results.** `Rel` says what a Rust flag record means in C's terms: same compact flag, same version, same timeout. C replaces `TIMEOUT 0` by the default in the parser, and Rust does it later in `compute_effective_timeout`; `Rel` accounts for that. `QErr.toC` maps each Rust error to the C error it corresponds to.
4. **Induct over the flag loop.** `scan_agree` proves the two loops stay related at every step, flag by flag. The ten small lemmas `k1` … `k10` say no two flag names are equal ignoring case, so at most one branch matches.
5. **The hypothesis `CfgOk`** says configured timeouts are non-negative and `TIMEOUT_DEFAULT ≤ TIMEOUT_MAX` when a maximum is set. `rustSet_inv` proves `GRAPH.CONFIG SET` keeps it, so it holds on any running server.

## Bugs found

All confirmed live against C (`repro_live.py` in this folder, and the scripts named in [`../FINDINGS.md`](../FINDINGS.md)).

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| 1 | `nonutf8_keeps_compact`, `timeout_noncanonical_rejected`, `garbage_timeout_rejected`, `version_above_uint_rejected`, `arity_capped` (now theorems) | Query flags: a non-UTF-8 flag dropped `--compact`; malformed timeouts and versions accepted; no arity cap | Fixed by #3010 (closes #3009); `rust_c_agree` |
| 2 | `pre3000_write_timeout_divergence` | Writes ignored the per-query `TIMEOUT` | Fixed by #3000 (closes #2998); `write_timeout_agrees` |
| 3 | `pre3021_vkey_negative_accepted`, `pre3021_cmdinfo_extra_spellings`, `pre3021_jsheap_tiny_accepted` | `GRAPH.CONFIG` accepted invalid values, extra arguments, and non-ASCII case folding; `ASYNC_DELETE` reported 0 (C 1) | Fixed by #3021 (closes #3020); `fixed3021_counterexamples` |
| 4 | `pre3026_two_entries_for_one_attr` | A `GRAPH.BULK` header naming a property twice stored both values | Fixed at import by #3063 (`pending_commit` `resolved_duplicate_fixed`), and at the parse by #3026 (closes #3024): `readRecord_enc`, `keepFirst_nodup`, `keepFirst_find` |
| 5 | `slot_len_truncates` | `Slot.len = n as u16`: 65,536 properties on one entity vanish. Root cause of #2539 | Bulk side fixed by #3063 (closes #2539); still reachable only from a malformed RDB |
| 6 | `memArgs_samples`, `cons_label_refused`, `cons_edge_refused`, `cons_count_noncanonical` (now theorems) | `GRAPH.MEMORY … SAMPLES 0` refused; `GRAPH.CONSTRAINT` accepted `LABEL`/`EDGE` and `+1`/`01` counts | Fixed by #3053 (closes #3052) |
| 7 | `pre3234_entry_hash_panics`, `pre3234_truncate_panics` | **Crash.** A query over 2048 bytes, slower than 10 ms, with a multi-byte character across byte 2048 panicked the slow log and killed the server (W4-redis-1) | Fixed by #3234 (closes #3233); `truncate_total` |
| 8 | `pre3252_field_prefix_lost` | An index on an attribute literally named `range:x` came back as an index on `x` after reload (W4-redis-2) | Fixed by #3252 (closes #3239); `field_name_kept` |
| 9 | `pre3328_bool_yes_case`, `pre3328_table_name_case`, `pre3328_load_missing` | Module-load arguments: `CMD_INFO YES` read as 0, `CMD_INFO garbage` loaded, `cache_size 10` and `ASYNC_DELETE yes` ignored (W4-redis-3) | Fixed by #3328 (closes #3268); `fixed3328_counterexamples` |
| 10 | `leading_space_divergence` | `CACHE_SIZE " 10"` and `GRAPH.CONFIG SET RESULTSET_SIZE " 5"` succeed on C and are refused by Rust | **Open** (confirmed live, not filed) |
| 11 | `from_slice_past_end` | Any truncated `GRAPH.RESTORE` payload calls `RedisModule_LoadStringBuffer(NULL)`; a cut at a record boundary SIGSEGVs. C crashes too | **Open** (#2537; external PR #3167) |
| 12 | `bulk_reply_differs` | The `GRAPH.BULK` reply says "relations created"; C says "edges created" (W4-redis-4) | **Open** (minor) |
| 13 | `pre3043_point_space`, `pre3043_nan_in_list` | Verbose format: a top-level point printed `latitude: 1.000000` with a space (C none); NaN inside a list printed `NaN` (C `nan`) | Fixed by #3043 (closes #3042); `pointStr_c`, `fmtS_floats_c`. Still open (minor, not filed): `CREATE (a:A) DELETE a` reports no node stats |
| 14 | `bug2782_same_name_other_db_aborts`, `bug2782_invariant_broken` (`MultiDb.lean`) | `GRAPH_REGISTRY` is keyed by name only, so creating `shared` in db1 displaces db0's `shared`; the next write to db0's graph aborts ("graph was deleted or replaced…"; debug build: assert). Property `KeyspaceRegistered` / `WritesSucceed` | **Open** (#2782) |
| 15 | `bug2653_nonutf8_key_mangled` (`KeyName.lean`) | A non-UTF-8 graph key is replicated and signalled under its `from_utf8_lossy` form: `\xc3(bin` → `\xef\xbf\xbd(bin`, so the replica writes a different key. Property `ReplicatedKeyIsStored` (C: `c_ok`) | **Open** (#2653) |
| 16 | `pre3419_rewrite_refused`; `rewrite_reload_counterexamples` (`ConfigRewrite.lean`) | `CONFIG REWRITE` saves `graph.<NAME> <value>` for what `GRAPH.CONFIG SET` or a load argument stored; a value outside the native config's range aborts the next start. `VKEY_MAX_ENTITY_COUNT` 0 or above 2^30 did (#3413) | `VKEY_MAX_ENTITY_COUNT` fixed by #3419: `vkey_rewrite_reloads_config`, `vkey_rewrite_reloads_load`. **Open** (not live-checked, not filed): `CACHE_SIZE` > 1000, `THREAD_COUNT` > 1024, `NODE_CREATION_BUFFER` > 2^30 as load arguments, `JS_HEAP_SIZE` > 2^32 by either path |
| 17 | `pre3428_cjk_panics`, `pre3428_latin1_garbled` (`ErrorText.lean`) | **Crash.** Error and simple-string replies went through redismodule-rs `str_as_legal_resp_string`, which truncated each char to `c as u8`: `é` was sent as the lone byte 0xE9, and a char whose low byte is 0 (`一`) made `CString::new(..).unwrap()` panic (#3185) | Fixed by #3428 (fork branch `fix/resp-string-bytes`, main b84473a7f): `legalResp_safe`, `legalResp_high_bytes`, `legalResp_cjk` — error text is now byte-preserving apart from CR/LF/NUL → space |
| 18 | `bug3442_update_breaks_min_cache`, `bug3442_faster_evicts_slower` (`SlowLog.lean`) | `SlowLog::add`'s update branch (`slow_log.rs:120-132`) returns before the `min_latency` recompute (`:172-177`), so the cached minimum goes stale: Q1..Q10 at 20..110 ms, Q1 again at 200 ms, then Q11 at 25 ms is admitted and evicts Q2 (30 ms). Properties `MinCache`, `SlowestKept` (C recomputes in `cleanup`); fix shape `addFixed_minCache`, `fixed3442_trace` | **Open** (#3442) |
| 19 | `bug3447_compact_cancelled_new_type_panics` (`RelLookup.lean`) | **Crash.** `CREATE (a)-[r:NEW_TYPE]->(b) DETACH DELETE a RETURN r --compact`: the cascaded edge is snapshotted, but its type is never registered (`Pending::commit` registers only surviving groups, `pending.rs:872-873`, `:1242-1251`), so `get_type_id(..).unwrap()` (`reply.rs:260`) panics; verbose works. Property `RegisteredTypes` ⇒ `ReplyTotal` (`replyTotal_of_registered`) | **Open** (#3447, compact-reply part) |
| 20 | `bug3449_record_renders_dead_rel` (`RelLookup.lean`) | **Crash.** `GRAPH.RECORD` of a top-level `DELETE r` renders `r` after commit with no snapshot (`snapshot_required` false, `delete.rs:69-81,99`): `graph.rs:3381` panics; of `CREATE … SET a.x = 1/0` renders the pending `r` with `bg.get_relationship_type_id` (`reply.rs:487`, `graph.rs:3348`) panics. Properties `RecordedLive`, `ReplyTotal`; fix shape `fixed3449_record_snapshot`, `verboseRelFixed_total` | **Open** (#3449; C has no `GRAPH.RECORD`) |

Also in the verbose format: a node deleted earlier in the same query still shows its labels (C shows `[]`). That one is intended: it was decided on 2026-09-25 that a deleted entity keeps all its details for the rest of the query.

C-side bugs seen along the way: C crashes on some malformed bulk tokens, and counts `L:L` twice.

## Files

| File | What's in it |
|---|---|
| `RedisLayer.lean` | Root: the report header (Lean↔Rust tables, theorems, divergences, gaps) |
| `Args.lean` | Integer parsers (`rustParseI64`, `string2ll`) and `compute_effective_timeout` (`rustTimeout`) |
| `QueryArgs.lean`, `QueryArgsSpec.lean`, `QueryArgsAgree.lean` | `parse_query_flags`; what it accepts exactly; agreement with C's `_read_flags` |
| `Config.lean` | `GRAPH.CONFIG`: validation, cross-field checks, `SET`, dispatch; C's `_Config_set` |
| `LoadArgs.lean`, `ModuleInit.lean` | Module-load arguments (#3328), the configuration table, event handlers |
| `ConfigRewrite.lean` | `CONFIG REWRITE` of the native configs against what the other paths store (#3419) |
| `Compact.lean` | Compact value encoding and the falkordb-py decoder |
| `Resp.lean`, `RespCompact.lean`, `RespVerbose.lean` | Reply call streams, postponed lengths, RESP2 bytes; verbose values and statistics |
| `RoQuery.lean` | The `RO_QUERY` write guard |
| `Bulk.lean` | The `GRAPH.BULK` binary format: property reader, records, header columns (`PropColumns`, #3026), counts, the `as u16` span length |
| `BufferedIO.lean`, `BufferedReader.lean`, `RoundTrip.lean` | `buffered_io.rs`: the type-tagged byte layer, its readers, the write→read round trip |
| `Serial.lean`, `SerialSchema.lean`, `SerialEntry.lean`, `SerialProof.lean` | `serializers/mod.rs`: graph header, schema blocks, index fields, constraints, and their round trip |
| `QuerySession.lean` | The per-query lock and GIL protocol in `query_session.rs` |
| `GraphCore.lean` | `graph_core.rs`: names, registry, write commit, dispatch, queue admission |
| `MultiDb.lean` | #2782: several Redis databases over the name-keyed registry; `graph_query`'s create/open path |
| `KeyName.lean` | #2653: key names as bytes, a concrete `from_utf8_lossy`, the replicated key |
| `ErrorText.lean` | #3428: `str_as_legal_resp_string` (error and simple-string reply text) on bytes; history `pre3428_*` |
| `Commands.lean`, `Commands2.lean` | The smaller command handlers: constraint, record, copy, bulk, memory, config get, and others |
| `Telemetry.lean`, `TelemetryReg.lean`, `TelemetryLoop.lean` | `telemetry.rs`: stream names, the query registry, batching, the flusher |
| `SlowLog.lean` | `slow_log.rs` (incl. #3442: `MinCache`, `SlowestKept`, `addFixed`) |
| `RelLookup.lean` | #3447/#3449: the relationship lookups behind the reply encoders (endpoints, type id/name over graph, snapshots and pending), the cascade commit and `GRAPH.RECORD` |
| `Allocator.lean` | `allocator.rs`: per-thread allocation counters |
| `repro_live.py` | Live Rust-vs-C repros for the divergences above |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

All `.lean` files except the root are under `RedisLayer/`. The project has no Lake dependencies.

## Coverage

`COVERAGE.tsv` has 294 rows across `src/`. The biggest files are `buffered_io.rs` (55), `graph_core.rs` (44), `telemetry.rs` (36), `query_session.rs` (25) and `bulk_insert.rs` (22).

| Bucket | Rows |
|---|---|
| PROVEN | 254 |
| AXIOMATISED | 34 |
| NOT COVERED | 6 |

- **AXIOMATISED:** wrappers around the Redis module API (`RedisModule_*`), libc (`pipe`, `fork`, `_exit`, `waitpid`, `pthread_atfork`), the OS clock and file-descriptor I/O. Most are in `graph_core.rs` (15).
- **NOT COVERED:** the six functions of `src/divergence_guard.rs`. Their reason column says "covered by proofs/replication narrative", but `replication` now models them line by line in `Guard.lean` and lists them as PROVEN; `bash proofs/coverage.sh` counts each function once, in its best bucket.

## What the model does not cover

- **The Redis module API, libc, the clock and file I/O** are hypotheses (`Compact.Env`, `Bulk.Codec`) or AXIOMATISED, never Lean axioms.
- **Concurrency** is modelled per operation, or as a single-thread log of events: the telemetry flusher, the write-queue election and GIL nesting are proven as protocols, not under an interleaving semantics.
- **Number formatting** is abstract. Floats in the compact reply are compared after formatting, which matches C, not as exact values.
- **`serializers/encoder` and `decoder`** are proven in `graph_persist`, **`divergence_guard.rs`** in `replication`, and **`redis_type.rs`** in `graph_persist`.

## Build and check

```bash
cd proofs/redis_layer && lake build    # this project
bash proofs/lean_ci.sh redis_layer     # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                # repo-wide coverage report
venv/bin/python proofs/redis_layer/repro_live.py   # live Rust-vs-C repros (Rust on 18430, C on 18431)
```

**When you change anything under `src/` (other than `redis_type.rs` and `divergence_guard.rs`):**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match.
3. Re-run the build.

Changing argument parsing or reply formatting usually means re-checking against C as well: `rust_c_agree` and the `LoadArgs` agreement theorem compare against a model of C, so a change that is meant to diverge from C must say so in the theorem. If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR.
