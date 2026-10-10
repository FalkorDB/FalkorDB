/-
# graph_persist: RDB save/load round-trip and constraint enforcement

Lean model of FalkorDB-rs persistence (v19 RDB codec, multi-key virtual-key
layout, deleted-id bitmap, attribute-id filter) and of UNIQUE/MANDATORY
constraint enforcement, checked against the C engine on live servers
(Rust :18420 `target/release/libfalkordb.dylib` @2363723ac, C :18421
`bin/macos-arm64v8-release/falkordb.so`). Zero sorry/admit/axiom.
Coverage per Rust fn: `COVERAGE.tsv`. Repro: `graph/tests/lean_graph_persist.rs`.

## Modules ↔ Rust

| Lean | Rust |
| --- | --- |
| `Codec.Tok`, `Val.encode/decode`, `decOne` | `BufferedWriter/Reader` tagged words; `impl Encode/Decode<19> for Value` (value.rs:1898,1984) |
| `Codec.nulTerm/stripNul` | `null_terminated`/`strip_null_terminator` (serializers/mod.rs:244,252), `save_string_nul`/`strip_trailing_nul` (redis_type.rs:241,246) |
| `Layout.chunkType`, `covered` | `build_multi_key_payloads` entity loop (encoder/mod.rs:95-153) + `encode_with_range` offsets (attribute_store.rs:1409) |
| `Layout.roaringRT` | `Encode/Decode<19> for RoaringTreemap` (serialization.rs:172-241) |
| `Layout.decodeSpanFilter` | `AttributeStore::decode_with_count` filter (attribute_store.rs:1460) |
| `Constraint.rustFrag`, `uniqueCollideRust` | `Graph::build_composite_key` `format!("{v:?}")` (graph.rs:4395), `check_node_constraint`/`check_edge_constraint` (pending.rs:1355,1469) |
| `Constraint.uniqueCollideC` | C `EnforceUniqueEntity` (observed behaviour) |
| `Constraint.hasDup`, `rustConstraintOK` | `validate_unique_constraint` via `create_constraint` (graph.rs:4339,3859) |
| `Constraint.rustEnforceKeyBuilds` | nested scan in `check_node_constraint` (pending.rs:1393-1418) |

## Proven (headline)

* `decOne_encode` — every scalar Value (null/bool/int/float bits/string incl.
  interned/point/datetime/date/time/duration) decodes back to itself with the
  suffix untouched; `encode_injective_scalar`; `stripNul_nulTerm`; `tags_distinct`.
  Lists/vectors: `#guard_msgs` round-trip checks on nested samples (MODELLED).
* `chunkType_covers`, `chunkType_count` — the per-key entity slices of a
  multi-key save tile `[0,total)` exactly once, in order.
* `roaring_roundtrip`, `encoded_len_mult8`; `decode_filter_sound`,
  `oob_id_dropped`, `good_attr_kept` — load never binds an out-of-dictionary or
  NULL attribute, and keeps every good one.
* `rust_C_unique_disagree`, `disagreement_two_sided`,
  `constraint_creation_diverges`, `mandatory_agrees`, `enforcement_is_quadratic`.

## Live round-trip results (DEBUG RELOAD, SAVE+restart, and cross-engine
Rust→C / C→Rust, each with and without VKEY_MAX_ENTITY_COUNT=2/3)

Identical graph dumps (ids, labels, all property types incl. nested lists,
point, temporal, vecf32, NaN, inf, -0.0, i64 extremes, multi-edges, self-loops,
deleted node/edge ids, range/fulltext/vector/edge indexes with options,
operational constraints) on every path. Also OK: RENAME then reload (single
and multi-key), brace-bearing graph names, 5 multi-key graphs in one RDB,
GRAPH.COPY then reload. The only before/after difference — FAILED constraints
vanish on reload — is identical in C (both write only operational ones).

## CONFIRMED bugs (Rust vs C on live servers)

1. **UNIQUE key is type-tagged, C's is value-equal** (graph.rs:4403
   `format!("{v:?}")`). Under `UNIQUE NODE L (v)`: `CREATE (:L {v:1})` then
   `{v:1.0}` / `{v:true}`, and `{v:0.0}` then `{v:-0.0}` — Rust admits, C rejects
   ("unique constraint violation"). Conversely duplicate lists, points, dates,
   vecf32 — Rust rejects, C admits. Same via SET (`SET n.v = 1.0`) and edges.
   Repro: `cargo test -p graph --test lean_graph_persist` —
   `unique_key_int_eq_float_like_c`, `unique_key_neg_zero_eq_zero_like_c`,
   `unique_key_bool_eq_int_like_c` FAIL (keys `Int(1)|` vs `Float(1.0)|`).
2. **Constraint created over already-violating data comes up OPERATIONAL**
   (consequence of 1, graph.rs:4339): graph with `(:L {v:1}),(:L {v:1.0})`,
   `(:M {v:0.0}),(:M {v:-0.0})`, `(:K {v:1}),(:K {v:true})` + range indexes +
   `GRAPH.CONSTRAINT CREATE g UNIQUE NODE <X> PROPERTIES 1 v` →
   `db.constraints()` Rust: all OPERATIONAL; C: all FAILED.
3. **UNIQUE enforcement is Θ(n²) per write query** (pending.rs:1393-1418 rebuilds
   a `seen` map over the whole label for every affected node).
   `UNWIND range(1,n) AS x CREATE (:L {v:x})` under UNIQUE: n=2000 Rust 9.2 s vs
   C 0.047 s; n=8000 Rust 212 s vs C 2.4 s (machine loaded; ratio, not absolute,
   is the point). Fix: build the label's key set once per constraint per commit,
   or probe the supporting range index like C.

## Unconfirmed / notes

* After reload, reused ids come out ascending (Rust 2,5,9) where C reuses
  LIFO (9,5,2); after deleting every node, Rust's next id is 0, C's is 19 — id
  choice is not a documented contract, recorded only.
* Rust keeps an extra `telemetry{<graph>}` key per graph (not RDB-related).
* C (reference) crashes the server on `CREATE (:L {v:0.0/0.0})` under a UNIQUE
  constraint (`EnforceUniqueEntity`); Rust does not — C bug, not ours.
* Not attempted: hostile/malformed RDB payloads.

## Wave 5: `serializers/{encoder,decoder}/mod.rs` and `redis_type.rs` (written at `3fec7d7c9`; line numbers now `2c874022a`)

| Lean | Rust |
| --- | --- |
| `Persist.Reader` — `Out`, `RT`, `RT_bind`, `RT_rep`, `readU`, `decV` | `BufferedReader` words; `Value::decode` (value.rs:1984) as an `Out` reader |
| `Persist.Store` — `encRoar`/`decRoar`, `encEnt`/`decEnt`, `applyEnt`, `fold_ents` | `RoaringTreemap` codec (serialization.rs:172-241), `AttributeStore::encode_with_range`/`decode_with_count` (attribute_store.rs:1409-1497) |
| `Persist.Payload` — `encDir`/`decDir`, `encIdx`/`decIdx`, `buildPayloads`, `encodeGraph`, `decPayload`, `applyPR`, `loadFromReader` | `encode_graph`, `build_payloads`, `Graph::encode_payload` (graph.rs:4716), `load_graph_from_reader` |
| `Persist.MultiKey` — `fillKey`, `keysFrom`, `buildMulti` | `build_multi_key_payloads` (encoder/mod.rs:95-189) |
| `Persist.KeyRT`, `Fold`, `Pending`, `MultiLoad*` — `rdbKey`, `stepKey`, `runKeys`, `fold_char` | `rdb_load_graph` (decoder/mod.rs:44-174), `decode_payloads_into_pending`, `finalize_pending_graph`, `DECODE_STATE` |
| `Persist.Indexes` — `rebuild` | `rebuild_indexes` (decoder/mod.rs:345-400) |
| `RedisType.AuxData`, `Uuid`, `VKeys`, `Load` | `redis_type.rs`: aux save/load, `on_persistence`, `pre_fork_prepare`, `uuid_v4`, virtual keys, `graph_rdb_save`, `graph_rdb_load`, `finalize_pending_graphs`, `install_graph`, `graphmeta_*` |

Sub-codecs enter as the laws of `Persist.Codecs` (hypotheses, not axioms): header/schema
(`redis_layer` `decH_encH`/`decSchema_encSchema`), GraphBLAS matrix/tensor (trusted FFI).

Headline theorems:
* `load_encode` — `load_graph_from_reader(encode_graph(G))` = the `Graph::restore` arguments
  of `G` (counts, deleted-id sets, both attribute stores, all matrices, schema, indexes,
  constraints), **and** every word-boundary truncation reads past the end
  (`truncated_payload_eof`, `empty_payload_eof`).
* `multi_key_tiles`, `multi_key_bound`, `multi_key_all_placed`, `multi_key_matrices` — the
  multi-key layout tiles each entity kind once, in order, ≤ `vkey_max` per key, matrices on
  key 0 only.
* `multi_load` — the `K ≥ 2` keys of a graph, loaded in **any** order, each read back
  (`key_rt`), finalize exactly at the last key (`run_keys`, `run_prefix_pending`) to the
  single-key restore of the graph (`multi_acc`), and the meta keys are exactly the virtual
  keys (`multi_meta`).
* `rebuild_calls`, `rebuild_entity`, `rebuild_meta_first` — index replay.
* `save_main`, `save_vkey`, `save_single`, `createdKeys_spec`, `deleteVKeys_spec` — SAVE
  writes the graph's key + `K - 1` virtual keys with slices `0..K`, `key_count = K`, and
  deletes exactly those virtual keys afterwards; `onPersistence_spec` (BGSAVE never splits).
* `load_main_last`, `load_main_first` — whatever the position of the graph's own key in the
  load order, the registered `Arc` and the key's value coincide and hold the finished graph;
  no virtual key is registered; placeholders are dropped.
* `aux_roundtrip` — UDF libraries round-trip through the aux fields (one NUL stripped).
* `uuid_injective`, `uuid_same_clock_distinct`, `uuid_collision_iff` — virtual-key names are
  unique iff `t₁ ⊕ s₁ ≠ t₂ ⊕ s₂`; `uuid_collision_example` (counters 2,3 at clocks 4,5 ns
  collide), so uniqueness rests on the clock (not reproduced live; theoretical).

Confirmed live (Rust release build of origin/main vs C `bin/macos-arm64v8-release/falkordb.so`,
ports 18910/18911, `CACHE_SIZE 1`):
4. **RDB-loaded graphs ignore `CACHE_SIZE`** (`redis_type.rs:66` `DEFAULT_CACHE_SIZE = 25`,
   passed at :97/:856; `cache_size_ignored`). Re-checked at 2c874022a: **still present**
   (#3161 only removed the vkey placeholder that also used it; after `DEBUG RELOAD`
   Rust `Cached execution: 1`, C `0`). Repro: `CREATE (:A)`; run q1, q2, q1 →
   "Cached execution: 0" on both; `DEBUG RELOAD`; q1, q2, q1 → Rust `1`, C `0`.
5. **#2537 generalised**: `GRAPH.RESTORE k <payload>` crashes Rust (SIGSEGV in
   `RM_LoadStringBuffer` via `load_string_buffer`) for any payload cut at a record boundary,
   e.g. one complete name record `\x00` + `le64(2)` + `g\0`; a cut *inside* a record returns
   `-BufferedReader: need 8 bytes …` instead (`readBytes` errs). C crashes on both.
   Re-checked at 2c874022a: **still present** (Rust SIGSEGV in `RM_LoadStringBuffer` via
   `BufferedReader::ensure_available`; C also dies).

## Re-target to 2c874022a (#3161 virtual keys as `graphmeta`, #2846 IdSpace)
* `save_key_slice` (redis_type.rs:203) is `saveKeySlice`; `graph_rdb_save` (:178) is
  `rdbSave = saveKeySlice.getD whole` (same `save_main`/`save_vkey`/`save_single`);
  `graphmeta_rdb_save` (:874) is `metaSave = saveKeySlice` — it now writes the slice
  (`metaSave_of_slice`, `metaSave_rdbSave`) and nothing for a leftover key
  (`metaSave_stale`); `create_virtual_keys` types virtual keys `graphmeta` (`vkeyType`).
  Live (Rust 2c874022a, `VKEY_MAX_ENTITY_COUNT 10`, 100 nodes → 11 keys): Rust
  `DEBUG RELOAD` keeps `count 100, sum 5050`; the same RDB loaded by C keeps `g` in
  `GRAPH.LIST` with 100 nodes (the #3160 symptom is gone).
* `Graph::restore` now builds `IdSpace::restored(count, deleted)`; caps unchanged
  (proved in proofs/graph_queries `restore_ids`). `MvccGraph` rows here stay NOT
  COVERED/MODELLED; the validated `commit` is proved in proofs/search_concurrency
  (`refused_commit_releases_slot`) and proofs/versioned_matrix (`refused_commit_rolls_back`).

## Gaps

GraphBLAS matrix/tensor (de)serialisation and RediSearch index creation are trusted
(hypotheses / AXIOMATISED); the codec's list/vector value arms are checked by evaluation,
not proof (`decV` covers scalars); `key_holds_graph`/`delete_key` and the `RedisModule_*`
calls are FFI; MvccGraph concurrency is not modelled; floats are bit images.
-/
import GraphPersist.Codec
import GraphPersist.Constraint
import GraphPersist.Layout
import GraphPersist.Persist
import GraphPersist.RedisType
