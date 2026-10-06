import RedisLayer.Allocator
import RedisLayer.Args
import RedisLayer.QueryArgsSpec
import RedisLayer.Config
import RedisLayer.Compact
import RedisLayer.RoQuery
import RedisLayer.Bulk
import RedisLayer.RoundTrip
import RedisLayer.SerialProof
import RedisLayer.RespVerbose
import RedisLayer.TelemetryLoop
import RedisLayer.ModuleInit
import RedisLayer.QuerySession
import RedisLayer.GraphCore
import RedisLayer.Commands2
import RedisLayer.SlowLog

/-!
# The Redis-facing layer (`src/`, excluding `redis_type.rs`)

Argument parsing, `GRAPH.CONFIG`, the compact result encoding, the RO_QUERY write
guard and the `GRAPH.BULK` binary parser, each modelled line by line and checked
against the C engine (`master`, `bin/macos-arm64v8-release/falkordb.so`) on live servers.
`lake build` is clean: no `sorry`, no `admit`, no `axiom`. The Redis module API and Rust
`core` byte conversions enter only as hypothesis structures (`Compact.Env`,
`Bulk.Codec`), never as axioms.

| here | there |
| --- | --- |
| `QueryArgs.parseQueryFlags`/`scanFlags`/`upToNul` | `parse_query_flags`/`up_to_nul`, `src/commands/query_args.rs:44-84` (#3010, `557f18868`) |
| `QueryArgs.queryFront`/`roQueryFront`/`profileFront`/`explainFront` | `query.rs:46-51`, `ro_query.rs:34-39`, `profile.rs:22`, `explain.rs:65` |
| `Args.rustParseI64`     | `str::parse::<i64>` (GRAPH.CONFIG) |
| `Args.string2ll`        | Redis `string2ll` (`RedisModule_StringToLongLong`, `parse_integer`) |
| `QueryArgs.cReadFlags`  | C `_read_flags` + `_validate_command_arity` (`cmd_dispatcher.c`) |
| `Args.rustTimeout`      | `compute_effective_timeout`, `src/graph_core.rs:896-931` |
| `Config.validate/cross/apply1/rustSet` | `config_cmd.rs:114-202`, `:210-242`, `:251-291`, `:342-390` (as of #3021; `validatePre3021` historical) |
| `Config.dispatch`, `upA` | `graph_config` arity / sub-command / `to_ascii_uppercase`, `config_cmd.rs:308-343` (#3021) |
| `Config.cSet`           | C `_Config_set` (`cmd_config.c`) |
| `Compact.encPair/encV/encReply` | `reply_compact_value` / `reply_result::<true>`, `src/reply.rs:134-344`, `:619-666` |
| `Compact.dec/decReply`  | falkordb-py `query_result.py` `parse_scalar`, `__parse_records` |
| `Ro.plan`               | parser `write` flag (`graph/src/parser/cypher.rs:780-910`) + Commit insertion (`planner/mod.rs:2416`, `:2749`) |
| `Ro.isWritePlan/roQuery/runRead` | `execute_query`, `graph_core.rs:597-608`; guards `runtime/eval.rs:936,956` |
| `Bulk.go/attach`        | `read_property`, `src/commands/bulk_insert.rs:94-169` |
| `Bulk.cstr`             | `read_cstring`, `:31-45` |
| `Bulk.records/readN`    | `process_node_token` record loop, `:386-410` |
| `Bulk.parseCount`       | `parse_count`, `:643-665` |
| `Bulk.getOrCreate/slotLen` | `attribute_store.rs:224-238`, `:598-610` |

## Proven (headline theorems)

* `rust_c_agree` (QueryArgsAgree) — for **every** argument vector, under the
  GRAPH.CONFIG invariant, `parse_query_flags` and C's `_read_flags` accept the same
  vectors, fail with the corresponding error, and read the same compact flag, version
  and timeout; `rust_c_same_accept`. `parse_ok_iff` (QueryArgsSpec) — the parser accepts
  exactly 3..8 arguments whose tail is in `Grammar`; `parse_total`; `parse_documented`
  — every documented command line is accepted with its documented meaning;
  `four_commands_share` — QUERY, RO_QUERY, PROFILE, EXPLAIN reject the same vectors and
  read the same flags; `parse_timeout_nonneg` — accepted timeouts are in `0..=TIMEOUT_MAX`.
* `string2ll_dec`, `rustParseI64_dec` — every canonical decimal round-trips.
* `rustTimeout_le_max`, `rustTimeout_pos`, `rustTimeout_exceeds`,
  `rustTimeout_write_ignores_pq` — the effective timeout never exceeds TIMEOUT_MAX
  (under the config invariant), is positive when set, and ignores the per-query value
  for writes.
* `rustSet_inv` — every successful `GRAPH.CONFIG SET` batch leaves
  `TIMEOUT_DEFAULT ≤ TIMEOUT_MAX` true; `rustSet_timeout_deprecated`; `c_breaks_inv`
  — C's sequential dry-run lets one batch break it (Rust is right here).
* `dec_enc`, `decReply_enc` — the compact encoding of every value (scalars, temporal,
  list, map, node, edge, path, vector, point) and of a whole reply decodes in the
  Python client to its spec; floats only modulo `parseF ∘ fmt` (`%.15g`, same as C).
* `ro_never_writes` — an RO_QUERY that is not rejected performs no mutation and
  contains neither a plan-visible mutation (`plan_write`) nor a write procedure
  (`plan_proc`); `write_proc_invisible_to_scan` shows the runtime guard is load-bearing.
* `readProperty_enc`, `go_encs`, `records_enc` — the bulk reader reads back exactly
  what the loader wrote, and a node token imports exactly its encoded records in order;
  `ceiling_sound` — the `max_records` pre-check never rejects an honest payload;
  `go`/`attach` totality is Lean's termination proof (each step consumes a byte or pops
  a frame).

## Confirmed divergences from C (all reproduced live: `repro_live.py`)

1. **FIXED by #3010 (`557f18868`, issue #3009)** — formerly: a non-UTF-8 flag ended
   the Rust flag loop (`--compact` dropped); `TIMEOUT -5/+10/010` accepted; RO_QUERY and
   PROFILE ignored a garbage/missing timeout; `version 4294967296` accepted; no arity cap.
   Now one parser mirrors C: `nonutf8_keeps_compact`, `timeout_noncanonical_rejected`,
   `garbage_timeout_rejected`, `version_above_uint_rejected`, `arity_capped`, and the
   general `rust_c_agree`.
2. Writes ignore the per-query TIMEOUT (`graph_core.rs:911-914`): with TIMEOUT_MAX
   = 100000, `UNWIND range(1,300000) … CREATE … TIMEOUT 1` creates 300000 nodes on Rust;
   C times out and creates none (`write_timeout_divergence`).
3. **FIXED by #3021 (`30b4fb7dc`, issue #3020)** — formerly GRAPH.CONFIG:
   `VKEY_MAX_ENTITY_COUNT -5` accepted (then read as `as u64`), `CMD_INFO 1/TRUE`
   accepted, `JS_HEAP_SIZE 5` accepted (C: ≥ 1 MB), `GET x extra` accepted, non-ASCII
   names case-fold (`tımeout` → TIMEOUT via `to_uppercase`), ASYNC_DELETE reports 0
   (C 1), error text "Unknown configuration field 'X'". Historical counterexamples:
   `pre3021_vkey_negative_accepted`, `pre3021_cmdinfo_extra_spellings`,
   `pre3021_jsheap_tiny_accepted`. Now: `fixed3021_counterexamples`, `vkey_ok_nonneg`,
   `jsheap_ok_ge`, `cmdinfo_ok_iff`, `get_extra_wrong_arity`, `set_pairs_complete`,
   `dotless_i_not_folded`, `asyncDelete_default_matches_C`, `getOne_unknown_text`.
4. GRAPH.BULK header naming a property twice (`p,q,p`): Rust stores both span entries
   (`properties(n)` = `{p: 1, p: 3, q: 2}`, `n.p` = 3); C keeps one (`n.p` = 1).
5. Verbose format: top-level point `point({latitude: 1.000000, …})` (C: no space after
   `:`), NaN in lists `NaN` (C `nan`), a node deleted in the same query keeps its labels
   (C `[]`), and `CREATE (a:A) DELETE a` reports no `Nodes created/deleted` stats.
6. Minor: several error texts differ (WRONGTYPE, empty key, count mismatch; `GRAPH.MEMORY …
   SAMPLES -1` Rust `ERR SAMPLES must be a non-negative integer`, C without the `ERR `).
   **Fixed by #3053** (`fe619ac5f`, live-checked on 19010/19011): `SAMPLES 0` accepted
   (`memArgs_samples`); `GRAPH.CONSTRAINT` refuses `LABEL`/`EDGE` (`cons_label_refused`,
   `cons_edge_refused`) and `+1`/`01` counts (`cons_count_noncanonical`, `string2ll`), as C.

Also seen / root cause for known issues: `Slot.len = n as u16` (`attribute_store.rs:598`,
`:609`) is why 65536 properties vanish (#2539; 65537 keeps one, `slot_len_truncates`);
GRAPH.RESTORE `""` SIGSEGV (#2537); createNodeIndex arity (#2540); Labels-added (#2655).
C-side bugs found on the way: C crashes on malformed bulk tokens; `L:L` counts twice.

## Wave 4 additions (source cited from `origin/main`)

| here | there |
| --- | --- |
| `BufferedIO.*`, `BufferedReader`, `RoundTrip` | `src/serializers/buffered_io.rs` (all 55 fns) |
| `Serial`, `SerialSchema`, `SerialEntry`, `SerialProof` | `src/serializers/mod.rs` (header/schema encode+decode) |
| `Resp`, `RespCompact`, `RespVerbose` | `src/reply.rs` (call streams, postponed lengths, RESP2 bytes) |
| `Telemetry`, `TelemetryReg`, `TelemetryLoop` | `src/telemetry.rs` |
| `ModuleInit` | `src/module_init.rs`, `src/lib.rs` config table, `src/config.rs` |
| `QuerySession` | `src/query_session.rs` |
| `GraphCore` | `src/graph_core.rs` |
| `Commands`, `Commands2` | `src/commands/*.rs` |
| `SlowLog` | `src/slow_log.rs` |

Headline theorems:
* `writeAll_flatten` — buffered writes = concatenation of the logical writes for every
  buffer size (flush points invisible); `writer_layout`/`roundtrip` — any chunk source
  honouring the load contract reads back exactly the written values, so `rdb_roundtrip`
  (RDB string buffers), `pipe_roundtrip` (GRAPH.COPY framing) and `vec_roundtrip`
  (GRAPH.RESTORE payload) all hold; i64 two's complement and f64/f32 bit patterns exact.
* `decH_encH`, `decSchema_encSchema` — header and schema decode∘encode = identity /
  explicit normal form (`normSchema`).
* `compact_emits`, `verbose_emits`, `stats_emits`, `compact_reply`, `verbose_reply` — every
  `RedisModule_Reply*` call stream (incl. postponed-length arrays) denotes exactly the spec
  tree; `compact_reply` lands on `Compact.encReply`, which `decReply_enc` decodes;
  `bytes_of_emits` gives the RESP2 bytes; `postponed_wrong_len` shows a wrong count breaks.
* Telemetry: `streamName_key_name`/`streamName_inj`/`stream_same_slot`, registry placement and
  transitions, `WaitingEntry` guard laws, `drain_conserves`, `capDeferred_suffix`,
  `retain_sound`/`pass2Step_sound` (re-keying), `group_keeps_order` (per-graph arrival order
  survives the sort), `iter_replica_never_writes`.
* `load_missing`/`load_extra` — load-time config table vs C's documented list;
  `maxInfo_clamped`; `cGraphName_key`; `rename_moves`; `loopExit_safe`; `escalate_order`,
  `writer_holds_gil`, `release_order`; `consArgs_ok`, `cons_label_refused`, `cons_count_noncanonical`, `memArgs_samples`; `copy_ok_iff`/`exited_iff`;
  `memTotal_sum`; `sync_is_yield_without_yields`.

New confirmed findings (live, Rust `target/release` vs C `bin/…/falkordb.so`, ports 18820-3):
8. **Crash**: `src/slow_log.rs:49` (`entry_hash`, also `truncate` `:31`) slices the query at
   byte 2048; a query > 2048 bytes, slower than 10 ms, with a multi-byte char across byte
   2048 panics and the server exits (`SlowLog.entry_hash_panics`; repro scratch
   `slowlog_utf8.py`). C answers normally.
9. Index on an attribute literally named `range:x` (or `vector:x`) is rebuilt on `x` after
   RDB reload (`serializers/mod.rs:642-646`, `SerialSchema.field_prefix_lost`): Rust
   `db.indexes()` → `[x]`, plan loses the index scan; C keeps `[range:x]`.
10. Module-load args: `CMD_INFO YES` → Rust 0 / C 1; `CMD_INFO garbage` → Rust loads with 0,
    C refuses to load; `cache_size 10` → Rust ignores (25), C 10; `ASYNC_DELETE yes` → Rust
    ignores (0), C 1 (`ModuleInit.bool_yes_case`, `table_name_case`, `load_missing`; cause:
    redismodule-rs `find_config_value` is case-sensitive and bools are `== "yes"`).
11. `GRAPH.BULK` success reply says `relations created`, C `edges created`
    (`Commands.bulk_reply_differs`).
Also seen: #2537's root cause is general — any truncated GRAPH.RESTORE payload makes
`BufferedReader::from_slice` call `RedisModule_LoadStringBuffer(NULL)` (`from_slice_past_end`).
Model correction: compact vector elements use `RedisModule_ReplyWithDouble` (`Env.vfmt`), not
`%.15g` (same as C).

## Re-target to 2c874022a (#2846, #3161)
`MvccGraph::commit` validates; the callers here changed: `graph_effect` folds the
commit into the apply result (`effectCmd` gains `valid`; `effect_refused_resyncs`),
`attempt_settle`/`settle_constraint` report `WriteAbort::Invalid` as permanent
(`attempt_invalid`, `settle_permanent`), `graph_bulk_insert` validates before
publishing index documents (`bulk_invalid_discards`, `bulk_publish_after_validate`),
`commit_and_replicate` panics on a refusal (`commitStepsV_refused`); `WriteAbort`
gains `WriteSlotBusy`/`Invalid` (`writeAbortMsg_invalid`). The bulk token loops lost
their `IdSpace` parameter only (`processTokens` unchanged).

## Not covered

AXIOMATISED only: RedisModule_* wrappers, libc (`pipe/fork/_exit/waitpid/pthread_atfork`),
OS clock and fd I/O. Concurrency is modelled per operation / single-thread event logs (the
flusher, write-queue election and GIL nesting are proven as protocols, not under an
interleaving semantics). Floats/number formatting abstract. `serializers/{encoder,decoder}/mod.rs` are proved in `proofs/graph_persist` (`Persist.*`);
`divergence_guard.rs` in `proofs/replication`.

## Wave 5: `src/allocator.rs` (`Allocator`)

`step`/`run` model the three thread-local cells and the `GlobalAlloc` hooks (`RedisAlloc`
is an input). `query_accounting` — after `reset_counter(); enable_tracking();` the counters
report exactly the bytes the query's own traffic allocated/freed (below `2^64`) and
`net_thread_usage` their saturating difference; `null_alloc_uncounted`, `disabled_frozen`
(logging under `disable_tracking` is not counted), `other_thread_invisible` (the
`QUERY_MEM_CAPACITY` check sees only the calling thread), `fresh_thread_counts` (the flag
starts `true`).
-/
