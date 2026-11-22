import RedisLayer.Allocator
import RedisLayer.Args
import RedisLayer.QueryArgsSpec
import RedisLayer.Config
import RedisLayer.LoadArgs
import RedisLayer.Compact
import RedisLayer.RoQuery
import RedisLayer.Bulk
import RedisLayer.RoundTrip
import RedisLayer.SerialProof
import RedisLayer.SerialCLayout
import RedisLayer.RespVerbose
import RedisLayer.TelemetryLoop
import RedisLayer.TelemetryFreed
import RedisLayer.ModuleInit
import RedisLayer.ConfigRewrite
import RedisLayer.QuerySession
import RedisLayer.GraphCore
import RedisLayer.Commands2
import RedisLayer.SlowLog
import RedisLayer.MultiDb
import RedisLayer.KeyName
import RedisLayer.ErrorText
import RedisLayer.RelLookup
import RedisLayer.EnvBuild

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
| `QueryArgs.parseQueryFlags`/`scanFlags`/`upToNul` | `parse_query_flags`/`parse_schema_hash`/`up_to_nul`, `src/commands/query_args.rs` (#3757 `SCHEMA_HASH`) |
| `QueryArgs.queryFront`/`roQueryFront`/`profileFront`/`explainFront` | `query.rs`, `ro_query.rs`, `profile.rs`, `explain.rs` |
| `Args.rustParseI64`     | `str::parse::<i64>` (GRAPH.CONFIG) |
| `Args.string2ll`        | Redis `string2ll` (`RedisModule_StringToLongLong`, `parse_integer`) |
| `QueryArgs.cReadFlags`  | C `_read_flags` + `_validate_command_arity` (`cmd_dispatcher.c`) |
| `Args.rustTimeout`      | `compute_effective_timeout`, `src/graph_core.rs` |
| `Config.validate/cross/apply1/rustSet` | `config_cmd.rs` (#3816 `WRITE_LOCK_POLL_INTERVAL`) |
| `Config.dispatch`, `upA` | `graph_config` arity / sub-command / `to_ascii_uppercase`, `config_cmd.rs` |
| `Config.cSet`           | C `_Config_set` (`cmd_config.c`) |
| `Compact.encPair/encV/encReply` | `reply_compact_value` / `reply_query` (`ReplyFormat::Compact`), `src/reply.rs` |
| `Compact.dec/decReply`  | falkordb-py `query_result.py` `parse_scalar`, `__parse_records` |
| `Ro.plan`               | parser `write` flag (`graph/src/parser/cypher.rs`) + Commit insertion (`planner/mod.rs`) |
| `Ro.isWritePlan/roQuery/runRead` | `execute_query`, `graph_core.rs`; guards `runtime/eval.rs` |
| `Bulk.go/attach`        | `read_property`, `src/commands/bulk_insert.rs` |
| `Bulk.cstr`             | `read_cstring` |
| `Bulk.records/readN`    | `process_node_token` record loop |
| `Bulk.PropColumns.new/readRecord`, `keepFirst` | `PropColumns::new`/`read_record` |
| `Bulk.parseCount`       | `parse_count` |
| `Bulk.getOrCreate/slotLen` | `attribute_store.rs` (`get_or_create`) (`len: n as u16`) |

## Proven (headline theorems)

* `rust_c_agree_except_schema` (QueryArgsAgree) — for **every** argument vector, under the
  GRAPH.CONFIG invariant, unless Rust refuses a `SCHEMA_HASH` value or a `SCHEMA_HASH`
  without `--compact` (#3757; `rust_c_agree_except_schemaParse_refuted`),
  `parse_query_flags` and C's `_read_flags` accept the same vectors, fail with the
  corresponding error, and read the same compact flag, version and timeout;
  `rust_ok_c_ok`. COUNTEREXAMPLE (#3757): `rust_c_agree_refuted` /
  `rust_c_same_accept_refuted` — `SCHEMA_HASH abc` is refused by Rust, skipped by C. `parse_ok_iff` (QueryArgsSpec) — the parser accepts
  exactly 3..8 arguments whose tail is in `Grammar` and sends `SCHEMA_HASH` only with
  `--compact` (`schema_hash_requires_compact`); `parse_total`; `parse_documented`
  — every documented command line is accepted with its documented meaning;
  `four_commands_share` — QUERY, RO_QUERY, PROFILE, EXPLAIN reject the same vectors and
  read the same flags; `parse_timeout_nonneg` — accepted timeouts are in `0..=TIMEOUT_MAX`.
* `string2ll_dec`, `rustParseI64_dec` — every canonical decimal round-trips.
* `rustTimeout_le_max`, `rustTimeout_pos`, `rustTimeout_exceeds`,
  `rustTimeout_write_pq`, `rustTimeout_rw_uniform`, `rustTimeout_write_ignores_pq` — the
  effective timeout never exceeds TIMEOUT_MAX (under the config invariant), is positive
  when set, applies to writes exactly as to reads once `timeout_rw` is set, and
  ignores the per-query value for writes only under the deprecated TIMEOUT alone.
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

## Open divergences from C (reproduced live: `repro_live.py`)

5. Verbose format: a node deleted in the same query keeps
   its labels (C `[]`, intended), and `CREATE (a:A) DELETE a` reports no `Nodes
   created/deleted` stats.

Also seen / root cause for known issues: `Slot.len = n as u16` (`attribute_store.rs`) is why 65536 properties vanish (#2539; 65537 keeps one, `slot_len_truncates`);
GRAPH.BULK can no longer reach it (the resolved span has at most
65,535 entries, proofs/pending_commit `Content.normResolved_fits`), a malformed RDB still can;
GRAPH.RESTORE `""` SIGSEGV (#2537); createNodeIndex arity (#2540); Labels-added (#2655).
C-side bugs found on the way: C crashes on malformed bulk tokens; `L:L` counts twice.

## Wave 4 additions (source cited from `origin/main`)

| here | there |
| --- | --- |
| `BufferedIO.*`, `BufferedReader`, `RoundTrip` | `src/serializers/buffered_io.rs` (all 55 fns) |
| `Serial`, `SerialSchema`, `SerialEntry`, `SerialProof` | `src/serializers/mod.rs` (header/schema encode+decode; `SerialSchema.decField` = `decode_index_field`, `encField`/`encPh` = field loop) |
| `SerialCLayout` | C's multi-type index-field entry vs `decode_index_field` (the pre-#3514 behaviour is kept as a historical counterexample) |
| `Resp`, `RespCompact`, `RespVerbose` | `src/reply.rs` (call streams, postponed lengths, RESP2 bytes) |
| `Telemetry`, `TelemetryReg`, `TelemetryLoop`, `TelemetryFreed` | `src/telemetry.rs`; `TelemetryFreed`: the #3844 persistence strip (`strip_stream`, the freed-graph queue, `redis_type.rs` `strip_telemetry_streams`) |
| `ModuleInit` | `src/module_init.rs`, `src/lib.rs` config table, `src/config.rs` |
| `ConfigRewrite` | `CONFIG REWRITE` of the `lib.rs` native configs vs. what `GRAPH.CONFIG SET` / load arguments store |
| `QuerySession` | `src/query_session.rs` |
| `GraphCore` | `src/graph_core.rs` |
| `Commands`, `Commands2` | `src/commands/*.rs` |
| `SlowLog` | `src/slow_log.rs` |
| `RelLookup` | relationship lookups of `src/reply.rs` (#3447) |
| `EnvBuild` | `sync_deepbind_environ` (`module_init.rs`) + `graph::env_snapshot::leak_deep_copy` (`env_snapshot.rs`) and `RedisBuildHost` (`graph_core.rs`, #3585): `copy_read_back`, `copy_fresh`, `sync_unchanged`, `sync_installs`, `publish_holds`, `publish_log` |

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
  transitions, `WaitingEntry` guard laws, `drain_conserves`,
  `retain_sound`/`pass2Step_sound` (re-keying), `group_keeps_order` (per-graph arrival order
  survives the sort), `iterSteps_delete_before_write`; since #3844 telemetry is node-local:
  `GraphCore.sync_unserved_silent` (replicated / AOF-replayed queries are not logged),
  `freed_stream_stripped`/`stripAll_covers` (no stream reaches a snapshot); the old replica
  rule is kept as history (`Pre3844.enqueue_replica`, `Pre3844.iter_replica_never_writes`).
* `LoadArgs.rust_c_same_accept` — module-load arguments accepted exactly as C's
  `Config_Init` (modulo leading whitespace and Rust-only names, both confirmed live; for
  arguments without embedded NUL, since C reads C strings; given `FsOk`, i.e. `File::create`
  agrees with `access(W_OK)`);
  `error_applies_nothing`; `cross_eq`; historical `ModuleInit.pre3328_*`;
  `Bulk.deep_refused` (#3339 depth cap); `QuerySession.checkFork_spec`, `upgrade_stale`, `current_kept`, `ofCommit_ne`,
  `GraphCore.executeWrite_first` (#3290); `GraphCore.admit_bounded` (#3014); `cGraphName_key`; `rename_moves`; `loopExit_safe`; `escalate_order`,
  `writer_holds_gil`, `release_order`; #3816 sliced write-lock wait and writer gate: `pollLoop_spec`,
  `acquire_gate_open`, `escalate_refused_reader`, `upgrade_refused_rechecks` (historical
  `Pre3816.pre3816_upgrade_twice_skips_checks`), `readPastGate_ends_reading`, `Config.wlpi_ok_nonneg`; `consArgs_ok`, `cons_label_refused`, `cons_count_noncanonical`, `memArgs_samples`; `copy_ok_iff`/`exited_iff`;
  `memTotal_sum`; `sync_is_yield_without_yields`.

Also seen: #2537's root cause is general — any truncated GRAPH.RESTORE payload makes
`BufferedReader::from_slice` call `RedisModule_LoadStringBuffer(NULL)` (`from_slice_past_end`).
Model correction: compact vector elements use `RedisModule_ReplyWithDouble` (`Env.vfmt`), not
`%.15g` (same as C).

## `Bulk.PropColumns`
`Bulk.PropColumns.new` (`new_spec`: column `i` holds `ids[i]` and the
index of that attribute among the distinct ones; `slot_inj`: two columns share a slot iff
they name the same attribute), `Bulk.PropColumns.readRecord` (`readRecord_enc`: whatever
`filled` the previous record left, one encoded record reads `keepFirst [] (ids.zip vs)` and
leaves exactly the following bytes). `keepFirst_nodup` (at most one entry per attribute),
`keepFirst_find` (the kept value is the first non-null in header order),
`keepFirst_sublist` (header order kept), `keepFirst_plain_of_nodup`/`readRecord_plain` (no
repeated name → the plain per-column read), `repeated_header_null_first`.

## Verbose-format floats and points
`Resp.wf` = `write_f` (`reply.rs`, NaN as `nan`) in the float and
vector leaves of `fmtS`; `Resp.pointStr` (top-level point) without a space after the
colons. Against C's `%f` (hypothesis `CPf`): `wf_c`, `fmtS_floats_c`, `fmtS_vec_c`,
`pointStr_c`, `verbose_point_c`, `fixed3043_nan_in_list`. Glibc prints a NaN with the sign bit set as `-nan`, which `write_f`
does not reproduce (`CPf.nan_text` is stated for `nan`).

## `CONFIG REWRITE` ranges
`ModuleInit.table` gives `VKEY_MAX_ENTITY_COUNT` the range `0..i64::MAX`; `ConfigRewrite.vkey_rewrite_reloads_config`
/ `vkey_rewrite_reloads_load` — any value `GRAPH.CONFIG SET` or a module-load argument
accepts is written by `CONFIG REWRITE` as a `graph.VKEY_MAX_ENTITY_COUNT` line that the next
start accepts and reads back exactly.
**Open:** `rewrite_reload_counterexamples` — `CACHE_SIZE 1001`,
`THREAD_COUNT 1025`, `NODE_CREATION_BUFFER 1073741825` (load arguments) and
`JS_HEAP_SIZE 4294967297` (load arguments and `GRAPH.CONFIG SET`) are accepted, but the
rewritten line is outside the table's range (not live-checked).

## Not covered

AXIOMATISED only: RedisModule_* wrappers, libc (`pipe/fork/_exit/waitpid/pthread_atfork`),
OS clock and fd I/O. Concurrency is modelled per operation / single-thread event logs (the
flusher, write-queue election and GIL nesting are proven as protocols, not under an
interleaving semantics). Floats/number formatting abstract. `serializers/{encoder,decoder}/mod.rs` are proved in `proofs/graph_persist` (`Persist.*`);
`divergence_guard.rs` in `proofs/replication`.
The server's own error text when `open_key_writable(..)?` fails (a refusal to open a key) is not modelled.

## Wave 5: `src/allocator.rs` (`Allocator`)

`step`/`run` model the three thread-local cells and the `GlobalAlloc` hooks (`RedisAlloc`
is an input). `query_accounting` — after `reset_counter(); enable_tracking();` the counters
report exactly the bytes the query's own traffic allocated/freed (below `2^64`) and
`net_thread_usage` their saturating difference; `null_alloc_uncounted`, `disabled_frozen`
(logging under `disable_tracking` is not counted), `other_thread_invisible` (the
`QUERY_MEM_CAPACITY` check sees only the calling thread), `fresh_thread_counts` (the flag
starts `true`).

## Open bugs the model missed (live-checked on main 9d3869aa8 vs C)
* #2782 `MultiDb`: keyspace keyed by `(db, key)` over the name-only `GRAPH_REGISTRY`.
  `bug2782_same_name_other_db_aborts` (db0, db1, db0 → ok, ok, aborted, as the writer check aborts; C ok×3), property
  `KeyspaceRegistered` / `WritesSucceed`; `existing_ok` shows the invariant suffices.
* #2653 `KeyName`: key names as bytes with a concrete `from_utf8_lossy`.
  `bug2653_nonutf8_key_mangled` (`\xc3(bin` → `\xef\xbf\xbd(bin`, `\xff\xfe` → two U+FFFD),
  property `ReplicatedKeyIsStored` (C satisfies it: `c_ok`; ASCII keys: `ascii_kept`).

## `ErrorText`
`Context::str_as_legal_resp_string` (the step every error
reply and simple-string reply goes through before `RedisModule_ReplyWithError` /
`…SimpleString`) maps the UTF-8 *bytes* (`\r \n NUL → space`) instead of `c as u8` per char.
`legalResp` (main) — `legalResp_safe` (no CR/LF/NUL, so `from_vec_unchecked`
is sound and the `-…\r\n` frame of `Resp.respT` holds), `legalResp_length`,
`legalResp_high_bytes` (every byte ≥ 0x80 kept in place: error text is byte-preserving for
non-ASCII), `legalResp_clean`, `legalResp_cjk`, `legalResp_latin1`, `bytesOf_toUTF8`.
`Resp.Call.err s` / `respT (.err s)` take `s` to be that already-scrubbed text.

## #3447 `RelLookup` (compact-reply part, **Open**)
The reply's type id of a deleted edge
is a *lookup* by name (`reply.rs`), which `RespCompact.encC` takes as given.
`commitCascade` registers only surviving edge types (`pending.rs`);
`bug3447_compact_cancelled_new_type_panics` (compact `none` = panic, verbose renders;
`RegisteredTypes`, `ReplyTotal` fail); `replyTotal_of_registered` — after commit,
`RegisteredTypes` makes both encoders total; `s3447_survivor_ok`. The compact encoder
still reads the committed graph for a pending edge's type; `GRAPH.RECORD` renders verbose only.

-/
#print axioms RedisLayer.Serial.decField_encFieldC
#print axioms RedisLayer.Serial.decField_encField
#print axioms RedisLayer.Serial.encPh_c_on_iff
