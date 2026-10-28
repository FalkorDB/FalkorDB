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
| `Args.rustTimeout`      | `compute_effective_timeout`, `src/graph_core.rs:886-921` |
| `Config.validate/cross/apply1/rustSet` | `config_cmd.rs:114-202`, `:210-242`, `:251-291`, `:342-390` (as of #3021; `validatePre3021` historical) |
| `Config.dispatch`, `upA` | `graph_config` arity / sub-command / `to_ascii_uppercase`, `config_cmd.rs:308-343` (#3021) |
| `Config.cSet`           | C `_Config_set` (`cmd_config.c`) |
| `Compact.encPair/encV/encReply` | `reply_compact_value` / `reply_result::<true>`, `src/reply.rs:145-355`, `:632-679` |
| `Compact.dec/decReply`  | falkordb-py `query_result.py` `parse_scalar`, `__parse_records` |
| `Ro.plan`               | parser `write` flag (`graph/src/parser/cypher.rs:780-910`) + Commit insertion (`planner/mod.rs:2416`, `:2749`) |
| `Ro.isWritePlan/roQuery/runRead` | `execute_query`, `graph_core.rs:597-608`; guards `runtime/eval.rs:936,956` |
| `Bulk.go/attach`        | `read_property`, `src/commands/bulk_insert.rs:95-176` |
| `Bulk.cstr`             | `read_cstring`, `:31-45` |
| `Bulk.records/readN`    | `process_node_token` record loop, `:419-440` |
| `Bulk.PropColumns.new/readRecord`, `keepFirst` | `PropColumns::new`/`read_record`, `:178-235` (#3026, `fb3d516d0`) |
| `Bulk.parseCount`       | `parse_count`, `:647-666` |
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
  `rustTimeout_write_pq`, `rustTimeout_rw_uniform`, `rustTimeout_write_ignores_pq` — the
  effective timeout never exceeds TIMEOUT_MAX (under the config invariant), is positive
  when set, applies to writes exactly as to reads once `timeout_rw` is set (#3000), and
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

## Confirmed divergences from C (all reproduced live: `repro_live.py`)

1. **FIXED by #3010 (`557f18868`, issue #3009)** — formerly: a non-UTF-8 flag ended
   the Rust flag loop (`--compact` dropped); `TIMEOUT -5/+10/010` accepted; RO_QUERY and
   PROFILE ignored a garbage/missing timeout; `version 4294967296` accepted; no arity cap.
   Now one parser mirrors C: `nonutf8_keeps_compact`, `timeout_noncanonical_rejected`,
   `garbage_timeout_rejected`, `version_above_uint_rejected`, `arity_capped`, and the
   general `rust_c_agree`.
2. **FIXED by #3000 (`2e8172d76`, issue #2998)** — formerly writes ignored the per-query
   TIMEOUT: with TIMEOUT_MAX = 100000, `UNWIND range(1,300000) … CREATE … TIMEOUT 1`
   created 300000 nodes on Rust; C times out and creates none
   (`pre3000_write_timeout_divergence`). Now `timeout_rw` (`graph_core.rs:895`) applies it:
   `write_timeout_agrees`, `Args.rustTimeout_write_pq`, `rustTimeout_rw_uniform`.
3. **FIXED by #3021 (`30b4fb7dc`, issue #3020)** — formerly GRAPH.CONFIG:
   `VKEY_MAX_ENTITY_COUNT -5` accepted (then read as `as u64`), `CMD_INFO 1/TRUE`
   accepted, `JS_HEAP_SIZE 5` accepted (C: ≥ 1 MB), `GET x extra` accepted, non-ASCII
   names case-fold (`tımeout` → TIMEOUT via `to_uppercase`), ASYNC_DELETE reports 0
   (C 1), error text "Unknown configuration field 'X'". Historical counterexamples:
   `pre3021_vkey_negative_accepted`, `pre3021_cmdinfo_extra_spellings`,
   `pre3021_jsheap_tiny_accepted`. Now: `fixed3021_counterexamples`, `vkey_ok_nonneg`,
   `jsheap_ok_ge`, `cmdinfo_ok_iff`, `get_extra_wrong_arity`, `set_pairs_complete`,
   `dotless_i_not_folded`, `asyncDelete_default_matches_C`, `getOne_unknown_text`.
4. GRAPH.BULK header naming a property twice (`p,q,p`): Rust stored both span entries
   (`properties(n)` = `{p: 1, p: 3, q: 2}`, `n.p` = 3); C keeps one (`n.p` = 1).
   **Fixed by #3063 (`0f45a8685`)**: `import_attrs_resolved` sorts stably, dedups by id
   (first column wins, as C) and drops `ATTRIBUTE_ID_NONE` — proven in proofs/pending_commit
   (`Content.normResolved_get`, `resolved_duplicate_fixed`). Since #3026 (`fb3d516d0`,
   issue #3024) the parse itself keeps one entry per attribute, the first non-null in
   header order: `Bulk.readRecord_enc`, `keepFirst_nodup`, `keepFirst_find`,
   `keepFirst_plainRead` (agrees with what #3063 kept); history
   `Bulk.pre3026_two_entries_for_one_attr`.
5. Verbose format: top-level point `point({latitude: 1.000000, …})` (C: no space after
   `:`) and NaN in lists `NaN` (C `nan`) — **FIXED by #3043 (`7cde92c1d`, closes #3042)**:
   `Resp.pointStr_c`, `verbose_point_c`, `wf_c`, `fmtS_floats_c`, `fmtS_vec_c`; history
   `pre3043_point_space`, `pre3043_nan_in_list`. Still: a node deleted in the same query keeps
   its labels (C `[]`, intended), and `CREATE (a:A) DELETE a` reports no `Nodes
   created/deleted` stats.
6. Minor: several error texts differ (WRONGTYPE, empty key, count mismatch; `GRAPH.MEMORY …
   SAMPLES -1` Rust `ERR SAMPLES must be a non-negative integer`, C without the `ERR `).
   **Fixed by #3053** (`fe619ac5f`, live-checked on 19010/19011): `SAMPLES 0` accepted
   (`memArgs_samples`); `GRAPH.CONSTRAINT` refuses `LABEL`/`EDGE` (`cons_label_refused`,
   `cons_edge_refused`) and `+1`/`01` counts (`cons_count_noncanonical`, `string2ll`), as C.

Also seen / root cause for known issues: `Slot.len = n as u16` (`attribute_store.rs:598`,
`:609`) is why 65536 properties vanished (#2539; 65537 keeps one, `slot_len_truncates`) —
no longer reachable from GRAPH.BULK since #3063 (`0f45a8685`): the resolved span has at most
65,535 entries (proofs/pending_commit `Content.normResolved_fits`);
GRAPH.RESTORE `""` SIGSEGV (#2537); createNodeIndex arity (#2540); Labels-added (#2655).
C-side bugs found on the way: C crashes on malformed bulk tokens; `L:L` counts twice.

## Wave 4 additions (source cited from `origin/main`)

| here | there |
| --- | --- |
| `BufferedIO.*`, `BufferedReader`, `RoundTrip` | `src/serializers/buffered_io.rs` (all 55 fns) |
| `Serial`, `SerialSchema`, `SerialEntry`, `SerialProof` | `src/serializers/mod.rs` (header/schema encode+decode; `SerialSchema.decField` = `decode_index_field` `:637-725`, `encField`/`encPh` = field loop `:383-427`) |
| `SerialCLayout` | C's multi-type index-field entry vs `decode_index_field` (#3484, fixed by #3514 `a0ba12950`); `pre3514_*` history |
| `Resp`, `RespCompact`, `RespVerbose` | `src/reply.rs` (call streams, postponed lengths, RESP2 bytes) |
| `Telemetry`, `TelemetryReg`, `TelemetryLoop` | `src/telemetry.rs` |
| `ModuleInit` | `src/module_init.rs`, `src/lib.rs` config table, `src/config.rs` |
| `ConfigRewrite` | `CONFIG REWRITE` of the `lib.rs:97-116` native configs vs. what `GRAPH.CONFIG SET` / load arguments store (#3419) |
| `QuerySession` | `src/query_session.rs` |
| `GraphCore` | `src/graph_core.rs` |
| `Commands`, `Commands2` | `src/commands/*.rs` |
| `SlowLog` | `src/slow_log.rs` |
| `RelLookup` | relationship lookups of `src/reply.rs:252-293`, `:472-492` (#3447, #3449; line pins re-targeted to 5e475e6c9) |

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
* `LoadArgs.rust_c_same_accept` (#3328) — module-load arguments accepted exactly as C's
  `Config_Init` (modulo leading whitespace and Rust-only names, both confirmed live);
  `error_applies_nothing`; `cross_eq`; historical `ModuleInit.pre3328_*`;
  `Bulk.deep_refused` (#3339 depth cap); `QuerySession.claimFork_spec`, `upgrade_stale`,
  `GraphCore.executeWrite_first` (#3290); `GraphCore.admit_bounded` (#3014); `cGraphName_key`; `rename_moves`; `loopExit_safe`; `escalate_order`,
  `writer_holds_gil`, `release_order`; `consArgs_ok`, `cons_label_refused`, `cons_count_noncanonical`, `memArgs_samples`; `copy_ok_iff`/`exited_iff`;
  `memTotal_sum`; `sync_is_yield_without_yields`.

New confirmed findings (live, Rust `target/release` vs C `bin/…/falkordb.so`, ports 18820-3):
8. **Crash — FIXED by #3234 (`3660a5fc3`)**: `src/slow_log.rs` (`entry_hash`, `truncate`)
   sliced the query at byte 2048; a query > 2048 bytes, slower than 10 ms, with a multi-byte
   char across byte 2048 panicked and the server exited (`SlowLog.pre3234_entry_hash_panics`,
   `pre3234_truncate_panics`; repro scratch `slowlog_utf8.py`). Now both cut at
   `prefix` (`:32`, `floor_char_boundary(2048)`): `SlowLog.hashInput_total`,
   `truncate_total` (no panic for any string), `prefix_le`/`prefix_isPrefix`/
   `prefix_boundary`/`prefix_longest` (longest char-boundary prefix ≤ 2048 bytes).
9. **FIXED by #3252 (`cfe6e7945`)**: an index on an attribute literally named `range:x` (or
   `vector:x`) was rebuilt on `x` after RDB reload (cab3cf192 `serializers/mod.rs:642-646`,
   `SerialSchema.pre3252_field_prefix_lost`): Rust `db.indexes()` → `[x]`, C `[range:x]`.
   Now `decode_index_field` keeps the name verbatim (`field_name_kept`; `decField_encField`
   round-trips every name).
9b. **FIXED by #3514 (`a0ba12950`, closes #3484)**: a C-written RDB entry carries every index type
   of an attribute in one type word; `decode_index_field` kept one, so RANGE was lost next to a
   FULLTEXT or VECTOR index (`SerialCLayout.pre3514_range_fulltext_loses_range`,
   `pre3514_range_vector_loses_range`). Main decodes every set type (`SerialSchema.decField`;
   `SerialCLayout.decField_encFieldC`: every C entry decodes to exactly its types;
   `decField_encField`: Rust's own entries unchanged; `decField_empty_refused`,
   `decField_unknown_bit_refused`: no type / unknown bit refused as C does). Same PR, phonetic:
   C's `"no"` (off) read as a code, so every C-loaded fulltext index turned phonetic ON
   (`pre3514_phonetic_no_kept`), and we wrote an empty string for off, which C reads as ON
   (`pre3514_encPh_off_read_on_by_c`). Now `"no"` reads as off and off is written `"no"`
   (`decField_c_phonetic_no_off`, `encPh_c_on_iff`; `index_layer` `Meta.TextOpts.phoneticEnabled`).
10. Module-load args: `CMD_INFO YES` → Rust 0 / C 1; `CMD_INFO garbage` → Rust loads with 0,
    C refuses to load; `cache_size 10` → Rust ignores (25), C 10; `ASYNC_DELETE yes` → Rust
    ignores (0), C 1 (FIXED by #3328: `ModuleInit.pre3328_bool_yes_case`, `pre3328_table_name_case`, `pre3328_load_missing`; now `LoadArgs.fixed3328_counterexamples`; cause:
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

## Re-target to fb3d516d0 (#3090, #3026; previously 73077bb4a)
#3090 touches only `matrix.rs` (proofs/graphblas_wrappers). #3026 adds `PropColumns`
to `bulk_insert.rs`: `Bulk.PropColumns.new` (`new_spec`: column `i` holds `ids[i]` and the
index of that attribute among the distinct ones; `slot_inj`: two columns share a slot iff
they name the same attribute), `Bulk.PropColumns.readRecord` (`readRecord_enc`: whatever
`filled` the previous record left, one encoded record reads `keepFirst [] (ids.zip vs)` and
leaves exactly the following bytes). `keepFirst_nodup` (at most one entry per attribute),
`keepFirst_find` (the kept value is the first non-null in header order),
`keepFirst_sublist` (header order kept), `keepFirst_plain_of_nodup`/`readRecord_plain` (no
repeated name → the plain per-column read), `repeated_header_null_first`; history
`pre3026_two_entries_for_one_attr`.

## Re-target to 8621212fc (#3043, #3419; previously 289c201fe)
Of the main commits since 289c201fe only `src/reply.rs` (#3043) and `src/lib.rs` (#3419)
touch `src/`. #3043: `Resp.wf` = `write_f` (`reply.rs:36`, NaN as `nan`) in the float and
vector leaves of `fmtS`; `Resp.pointStr` (top-level point) without the space after the
colons. Against C's `%f` (hypothesis `CPf`): `wf_c`, `fmtS_floats_c`, `fmtS_vec_c`,
`pointStr_c`, `verbose_point_c`, `fixed3043_nan_in_list`; history `pre3043_point_space`,
`pre3043_nan_in_list`. Glibc prints a NaN with the sign bit set as `-nan`, which `write_f`
does not reproduce (`CPf.nan_text` is stated for `nan`). #3419: `ModuleInit.table` gives
`VKEY_MAX_ENTITY_COUNT` the range `0..i64::MAX`; `ConfigRewrite.vkey_rewrite_reloads_config`
/ `vkey_rewrite_reloads_load` — any value `GRAPH.CONFIG SET` or a module-load argument
accepts is written by `CONFIG REWRITE` as a `graph.VKEY_MAX_ENTITY_COUNT` line that the next
start accepts and reads back exactly; history `pre3419_rewrite_refused` (#3413).
**Same class, open:** `rewrite_reload_counterexamples` — `CACHE_SIZE 1001`,
`THREAD_COUNT 1025`, `NODE_CREATION_BUFFER 1073741825` (load arguments) and
`JS_HEAP_SIZE 4294967297` (load arguments and `GRAPH.CONFIG SET`) are accepted, but the
rewritten line is outside the table's range (not live-checked).

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

## Gap closed 2026-10-20 (bugs the model missed; both **Open**, live-checked on main 9d3869aa8 vs C)
* #2782 `MultiDb`: keyspace keyed by `(db, key)` over the name-only `GRAPH_REGISTRY`.
  `bug2782_same_name_other_db_aborts` (db0, db1, db0 → ok, ok, aborted; C ok×3), property
  `KeyspaceRegistered` / `WritesSucceed`; `existing_ok` shows the invariant suffices.
* #2653 `KeyName`: key names as bytes with a concrete `from_utf8_lossy`.
  `bug2653_nonutf8_key_mangled` (`\xc3(bin` → `\xef\xbf\xbd(bin`, `\xff\xfe` → two U+FFFD),
  property `ReplicatedKeyIsStored` (C satisfies it: `c_ok`; ASCII keys: `ascii_kept`).

## Re-target to b84473a7f (#3428; previously b6567cb97)
#3428 (closes #3185) touches no `src/` file: it moves `redis-module` to the fork branch
`fix/resp-string-bytes`, where `Context::str_as_legal_resp_string` (the step every error
reply and simple-string reply goes through before `RedisModule_ReplyWithError` /
`…SimpleString`) maps the UTF-8 *bytes* (`\r \n NUL → space`) instead of `c as u8` per char.
`ErrorText`: `legalResp` (main) — `legalResp_safe` (no CR/LF/NUL, so `from_vec_unchecked`
is sound and the `-…\r\n` frame of `Resp.respT` holds), `legalResp_length`,
`legalResp_high_bytes` (every byte ≥ 0x80 kept in place: error text is byte-preserving for
non-ASCII), `legalResp_clean`, `legalResp_cjk`, `legalResp_latin1`, `bytesOf_toUTF8`;
history `pre3428_legalResp`, `pre3428_cjk_panics` (`一` → NUL → `CString::new(..).unwrap()`
panic), `pre3428_latin1_garbled` (`é` → lone 0xE9). `Resp.Call.err s` / `respT (.err s)`
take `s` to be that already-scrubbed text.

## Gap closed 2026-10-22 (re-targeted to 5e475e6c9: #3442 and #3449 now **Fixed**)
* #3442 `SlowLog` (fixed by #3456, closes #3442): on b84473a7f `add`'s `some _` arm
  (`slow_log.rs:120-132`) kept the stale `minLat` (recompute at `:172-177` on b84473a7f). Main's update arm
  falls through to the recompute (`slow_log.rs:121-139`, `:162-169`): `SlowLog.add` is that code,
  `pre3442_add` the old one. History `pre3442_update_breaks_min_cache` (property `MinCache`),
  `pre3442_faster_evicts_slower` (the issue's trace: Q11 at 25 ms evicted Q2 at 30 ms; property
  `SlowestKept`). On main: `add_minCache` (preserved for every input),
  `fixed3442_update_keeps_min_cache`, `fixed3442_trace` (Q2 kept, Q11 rejected), and
  `add_entries_pre3442` (the fix changes only the cached minimum). `add_repeat_max` now assumes
  one entry per hash (`add_nodup`).
* #3447 `RelLookup` (compact-reply part, still **Open**): the reply's type id of a deleted edge
  is a *lookup* by name (`reply.rs:260`), which `RespCompact.encC` takes as given.
  `commitCascade` registers only surviving edge types (`pending.rs:872-873`, `:1242-1251`);
  `bug3447_compact_cancelled_new_type_panics` (compact `none` = panic, verbose renders;
  `RegisteredTypes`, `ReplyTotal` fail); `replyTotal_of_registered` — after commit,
  `RegisteredTypes` makes both encoders total; `s3447_survivor_ok`.
* #3449 `RelLookup` (fixed by #3462, closes #3449): `recordDelete` and `s3449abort` (pending
  edge rendered after `abandon_write`, `record.rs:106-132`). On b84473a7f a top-level `DELETE r`
  was not snapshotted (`delete.rs:69-81,99`) and the verbose reply asked the committed graph for
  a pending edge's type. #3462 snapshots when `runtime.inspect || snapshot_required`
  (`delete.rs:99-101`, `snapshot`) and resolves the verbose type through
  `Runtime::get_relationship_type` (deleted -> pending -> graph, `runtime.rs:1902-1914`,
  `verboseRel`). History `pre3449_record_renders_dead_rel` (`pre3449_snapshot`,
  `pre3449_verboseRel`). On main: `fixed3449_record_snapshot` (`RecordedLive` holds and the
  row renders from its snapshot), `snapshot_inspect`, `fixed3449_record_abort`,
  `verboseRelFixed_total` (the verbose encoder is total on every known id). The compact encoder
  still reads the committed graph for a pending edge's type; `GRAPH.RECORD` renders verbose only.
-/
#print axioms RedisLayer.Serial.pre3514_range_fulltext_loses_range
#print axioms RedisLayer.Serial.decField_encFieldC
#print axioms RedisLayer.Serial.decField_encField
#print axioms RedisLayer.Serial.encPh_c_on_iff
