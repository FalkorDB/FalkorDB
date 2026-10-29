# graph_persist: saving a graph to RDB and loading it back, and constraint enforcement

This project proves that FalkorDB's RDB persistence gives back the graph it saved, and compares UNIQUE/MANDATORY constraint enforcement with C. It covers:

- **the v19 RDB codec**: property values, the deleted-id bitmaps, the attribute stores, the payload directory, the whole graph payload;
- **the multi-key layout**: a large graph is saved across several Redis keys (the graph's own key plus *virtual keys*), and Redis may load them back in any order;
- **`src/redis_type.rs`**: save, load, aux fields (UDF libraries), virtual-key names and their cleanup;
- **constraints**: how the UNIQUE key is built, and where Rust and C disagree.

`SAVE`/`BGSAVE`, restart, `DEBUG RELOAD`, `GRAPH.COPY`, replica full sync and moving an RDB between the Rust and C engines all depend on this. A bug here loses data or changes it silently on restart.

The full Lean↔Rust mapping is in the header of [`GraphPersist.lean`](GraphPersist.lean) and in the table at the top of each file. This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. Does loading a saved graph give back the same graph?**

Yes, for a graph saved in one key: `load_encode`. Decoding the payload that `encode_graph` writes gives exactly the arguments `Graph::restore` needs to rebuild that graph:

- node and edge counts and the deleted-id sets;
- both attribute stores;
- every matrix;
- the schema, indexes and constraints.

It holds under any destination key name, which is what `RENAME` and `GRAPH.COPY` need.

The leaves are proven separately:

- `decOne_encode`: every scalar property value comes back as itself, with the bytes after it untouched. That covers null, bool, int, float (bit for bit), string (including interned strings), point, datetime, date, time and duration;
- `roaring_roundtrip`: the deleted-id bitmap round-trips;
- `stripNul_nulTerm`: NUL-terminated strings round-trip.

**2. What if the RDB is cut short?**

The decoder notices instead of inventing data. `load_encode` also proves that cutting the payload at *any* word boundary makes the decoder ask for one more word, rather than returning a smaller graph (`truncated_payload_eof`, `empty_payload_eof`).

That is the decoder's own logic. The live server still crashes on one kind of truncated `GRAPH.RESTORE`, inside the Redis module API (#2537, see [Bugs found](#bugs-found)).

**3. Does a graph split over several keys load correctly, whatever order Redis loads the keys in?**

Yes: `multi_load`. When a graph has more entities than `VKEY_MAX_ENTITY_COUNT`, `build_multi_key_payloads` spreads them over K ≥ 2 keys. On load:

- the keys can arrive in any order (any permutation of `0..K`), and each one reads back (`key_rt`);
- nothing is finalized before the last key arrives, and the last one finalizes the full graph (`run_keys`);
- the finalized graph equals the single-key restore of the same graph (`multi_acc`);
- the keys recorded as virtual keys are exactly the keys not named after the graph (`multi_meta`).

The layout itself is proven too:

- `multi_key_tiles`: every entity of every kind is in exactly one key, in order;
- `multi_key_bound`: no key holds more than `vkey_max` entities;
- `multi_key_all_placed`: the expected number of keys holds everything;
- `multi_key_matrices`: the matrices go on key 0 and only there.

**4. Does load ever bind a property to the wrong attribute?**

No. `decode_filter_sound`: on load, `decode_with_count` drops any attribute whose id is outside the attribute dictionary (`oob_id_dropped`) or whose value is `Null`, and keeps every good one (`good_attr_kept`).

**5. Does SAVE write exactly the right keys, and clean up after itself?**

Yes:

- `save_main`, `save_vkey`, `save_single`: SAVE writes the graph's own key plus K − 1 virtual keys holding slices `0..K`, with `key_count = K`;
- `deleteVKeys_spec`: the virtual keys are deleted afterwards;
- `onPersistence_spec`: BGSAVE never splits a graph;
- `metaSave_stale`: since #3161 virtual keys have the `graphmeta` type, and a leftover one writes nothing;
- `load_main_first`, `load_main_last`: whatever position the graph's own key has in the load order, the registered graph and the key's value end up as the same finished graph.

**6. Are virtual-key names unique?**

- **Before #3286, only as long as the clock moved.** `uuid_v4` mixed the wall clock with a counter, `a = nanos ^ seq`, and printed only `a`. Two calls gave the same name exactly when `t₁ ⊕ s₁ = t₂ ⊕ s₂` (counterexample removed; in 17ccc5412); counters 2 and 3 at clocks 4 ns and 5 ns collided, and the second virtual key replaced the first one's slice.
- **On main, yes, within a process.** `uuid_fields` keeps the clock in the first three fields and `seq × odd constant + c` in the last two. `counter_field_bijective` proves the last field is a bijection of `seq` modulo 2^48, and `uuid_counter_distinct` that two calls whose counters differ mod 2^48 never share a name, whatever the clock reads. `vkeyName_uuid_counter` lifts this to the full key names, and `names_vname_inj` discharges the `vname_inj` part of the naming assumption the SAVE theorems take (`VKeys.Names`).

**7. Do UNIQUE constraints agree with C?**

No; see [Bugs found](#bugs-found). Rust builds the UNIQUE key from `format!("{v:?}")`, which includes the value's *type*. C compares *values*. Under `UNIQUE NODE L (v)`:

```cypher
CREATE (:L {v: 1})
CREATE (:L {v: 1.0})   // Rust: accepted. C: unique constraint violation
```

The same happens for `0.0` vs `-0.0` and `1` vs `true`. The disagreement goes both ways (`disagreement_two_sided`): for duplicate lists, points, dates and vectors, Rust rejects and C accepts. MANDATORY constraints agree with C (`mandatory_agrees`).

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `load_encode` | Loading a single-key payload gives exactly the `Graph::restore` arguments of the saved graph, and every truncation is detected | `Persist/RoundTrip.lean` |
| `multi_load` | A graph saved across K ≥ 2 keys, loaded in any key order, finalizes once, at the last key, to the same graph | `Persist/MultiLoad3.lean` |
| `multi_key_tiles`, `multi_key_bound`, `multi_key_all_placed`, `multi_key_matrices` | The multi-key layout puts each entity in exactly one key, in order, within the size limit, with matrices on key 0 only | `Persist/MultiKey.lean` |
| `decOne_encode` | Every scalar property value round-trips through the v19 codec | `Codec.lean` |
| `roaring_roundtrip`, `encoded_len_mult8` | The deleted-id bitmap round-trips; its encoding is 8 bytes per id | `Layout.lean` |
| `decode_filter_sound` | Load never binds an out-of-dictionary or `Null` attribute, and keeps every good one | `Layout.lean` |
| `rebuild_calls`, `rebuild_entity`, `rebuild_meta_first` | Indexes are replayed after load with the right fields, entity type and full-text options | `Persist/Indexes.lean` |
| `save_main`, `save_vkey`, `deleteVKeys_spec`, `onPersistence_spec` | SAVE writes the graph key plus K − 1 virtual keys and deletes them afterwards; BGSAVE never splits | `RedisType/VKeys.lean`, `RedisType/AuxData.lean` |
| `load_main_first`, `load_main_last` | Whatever the load order, the registered graph is the finished graph | `RedisType/Load.lean` |
| `aux_roundtrip` | UDF libraries round-trip through the RDB aux fields | `RedisType/AuxData.lean` |
| `uuid_counter_distinct`, `counter_field_bijective` | Virtual-key names differ whenever their counters differ mod 2^48, whatever the clock (#3286) | `RedisType/Uuid.lean` |
| `countCheck_in_bounds` | The deleted-id decoder's checks keep every 8-byte read in bounds, whatever `count` the payload claims (#2996) | `Layout.lean` |
| `rust_C_unique_disagree`, `constraint_creation_diverges` | Rust's UNIQUE key and C's disagree; creating a constraint over violating data diverges | `Constraint.lean` |

### How `load_encode` is proved

1. **Model the stream.** An RDB payload is a sequence of tagged words (`Tok`: unsigned, signed, double, buffer), as `BufferedWriter`/`BufferedReader` write and read them. A decoder returns one of three outcomes (`Out`): a value and the rest of the stream, an error, or "ran out of words".
2. **Define round trip with truncation built in.** `RT d toks a` says: decoder `d` reads `toks ++ rest` as `a` and leaves `rest`, **and** on every strict prefix of `toks` it runs out of words rather than erring or returning something. `RT_bind` and `RT_seq` prove this composes: if each field round-trips, so does a sequence of fields. Proving it once for the combinators gives the truncation result for the whole graph for free.
3. **Prove the leaves.** Scalar values (`decV`), the deleted-id bitmap (`encRoar`/`decRoar`), the attribute-store spans (`encEnt`/`decEnt`), the payload directory and the index list each get an `RT` lemma.
4. **Fold the payloads.** The decoder accumulates payloads into locals before calling `Graph::restore`. `fold_char` describes the result after *any* list of payloads by which ids and matrices it carries, not by order. That is what lets `multi_load` accept keys in any order.
5. **Hypotheses.** The header and schema codecs, and the GraphBLAS matrix and tensor codecs, enter as the laws of `Codecs`, not as axioms. The header and schema laws are proven in [`../redis_layer`](../redis_layer) (`decH_encH`, `decSchema_encSchema`), but the projects are not linked. The GraphBLAS ones are trusted FFI.

## Bugs found

All confirmed against live Rust and C servers (see [`../FINDINGS.md`](../FINDINGS.md)). The Rust repro tests are in `graph/tests/lean_graph_persist.rs`, which is not committed because some of its tests fail on purpose.

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| 1 | `rust_C_unique_disagree`, `disagreement_two_sided` | The UNIQUE key is type-tagged (`format!("{v:?}")` in `build_composite_key`). `{v:1}` and `{v:1.0}`, `{v:0.0}` and `{v:-0.0}`, `{v:1}` and `{v:true}` are all admitted (C rejects); duplicate lists, points, dates and vectors are rejected (C admits). Same through `SET` and on edges | **Open** (#2774) |
| 2 | `constraint_creation_diverges` | A consequence of 1: creating a UNIQUE constraint over data that already violates it under C's rules reports OPERATIONAL (C: FAILED) | **Open** |
| 3 | `enforcement_is_quadratic` | UNIQUE enforcement rebuilt a key map over the whole label for every affected node: `UNWIND range(1,8000) … CREATE` took 212 s (C 2.4 s) | Fixed by #3334 (closes #3330), modelled in [`../pending_commit`](../pending_commit) (`uniqueViolated_spec`). This project's theorem still describes the old scan |
| 4 | removed (17ccc5412) | RDB-loaded graphs ignored `CACHE_SIZE` and used 25 (W5-rdb-1) | Fixed by #3262 (`cache_size_on_load`) |
| 5 | none (the crash is inside the Redis module API) | `GRAPH.RESTORE` with a payload cut at a record boundary SIGSEGVs in `RM_LoadStringBuffer`; a cut inside a record errors cleanly. C crashes on both | **Open** (#2537; external PR #3167) |
| 6 | removed (17ccc5412) | Two virtual-key names could collide if the clock and counter lined up; a collision lost a virtual key's slice | Fixed by #3286 (`bf11a2774`): `uuid_counter_distinct`, `uuid_regression_pair_distinct` |
| 7 | removed (17ccc5412) | A payload claiming `2^61` deleted ids passed the length check with an empty buffer (`count * 8` wrapped) and indexed past it | Fixed by #2996 (`d99861124`, closes #2994): `countCheck_in_bounds`. #2996 does not touch the reader, so bug 5 is unaffected |

Recorded but not bugs:

- After reload, Rust reuses deleted ids in ascending order and C in LIFO order. Id choice is not a documented contract.
- C crashes on `CREATE (:L {v:0.0/0.0})` under a UNIQUE constraint; Rust does not. A C bug.
- Live round trips (`DEBUG RELOAD`, SAVE and restart, Rust→C and C→Rust, each with `VKEY_MAX_ENTITY_COUNT` 2 and 3) gave identical graph dumps. The one difference before and after reload, FAILED constraints disappearing, is identical in C.

## Files

| File | What's in it |
|---|---|
| `GraphPersist.lean` | Root: the report header (Lean↔Rust tables, theorems, bugs, live results, gaps) |
| `Codec.lean` | The v19 value codec over tagged words; `decOne_encode`, `stripNul_nulTerm`, `tags_distinct` |
| `Layout.lean` | The first model of the multi-key entity split (`chunkType_covers`), the deleted-id bitmap, the load-side attribute filter |
| `Constraint.lean` | The UNIQUE key in Rust and C, constraint creation, enforcement cost, MANDATORY |
| `Persist.lean` | Imports the `Persist/` modules; its header describes the encoder/decoder model |
| `Persist/Reader.lean` | Reader outcomes `Out`, the round-trip law `RT` and its combinators, words and scalar values |
| `Persist/Store.lean` | Deleted-id bitmaps and attribute-store spans |
| `Persist/Payload.lean` | The payload directory, the graph `G`, the sub-codec laws `Codecs`, `encodeGraph`, `loadFromReader` |
| `Persist/RoundTrip.lean` | `load_encode` and the truncation results |
| `Persist/Fold.lean` | Folding payloads in any order (`fold_char`) |
| `Persist/MultiKey.lean` | `build_multi_key_payloads` loop for loop; the layout theorems |
| `Persist/KeyRT.lean` | One key of a multi-key graph reads back its slices (`key_rt`) |
| `Persist/Pending.lean` | The `DECODE_STATE` machine for one multi-key graph (`run_keys`) |
| `Persist/MultiLoad.lean`, `MultiLoad2.lean`, `MultiLoad3.lean` | The parts of `multi_load` |
| `Persist/Indexes.lean` | `rebuild_indexes` |
| `RedisType.lean` | Imports the `RedisType/` modules |
| `RedisType/AuxData.lean` | Aux fields (UDF libraries), persistence events, the fork hook, `graphmeta` stubs |
| `RedisType/Uuid.lean` | `uuid_v4` and virtual-key names |
| `RedisType/VKeys.lean` | Virtual keys on save, `graph_rdb_save`, the stale-key sweeps |
| `RedisType/Load.lean` | `graph_rdb_load`, placeholders, `finalize_pending_graphs`, `CACHE_SIZE` on load |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

The project has no Lake dependencies.

## Coverage

`COVERAGE.tsv` has 70 rows: `redis_type.rs` (27), `graphblas/serialization.rs` (17), the encoder and decoder (7 each), `mvcc_graph.rs` (7), `graph.rs` (5).

| Bucket | Rows |
|---|---|
| PROVEN | 47 |
| MODELLED | 2 |
| NOT COVERED | 10 |
| AXIOMATISED | 11 |

- **AXIOMATISED:** the eight `BufferedWriter`/`BufferedReader` word methods (`write_unsigned` … `read_buffer`), taken as inverse to each other; `delete_key` and `key_holds_graph` (Redis module API); `rebuild_derived_matrices` (GraphBLAS).
- **MODELLED:** `from_u64` (the payload-directory state tags, exercised by the live round trips) and `MvccGraph::from_graph` (the RDB load path, checked live).
- **NOT COVERED:** four trait declarations or `unimplemented!` defaults in `serialization.rs`, and six `MvccGraph` functions (`new`, `read`, `write`, `commit`, `rollback`, `drop`) whose concurrency is not modelled here. The validated `commit` is proven in `search_concurrency` and `versioned_matrix`; `bash proofs/coverage.sh` counts each function once, in its best bucket.

## What the model does not cover

- **GraphBLAS matrix and tensor (de)serialisation** and RediSearch index creation are trusted (hypotheses or AXIOMATISED).
- **List and vector property values** are checked by evaluating round trips on nested samples (`#guard_msgs`), not proven; the decoder proof (`decV`) covers scalars. The nesting cap added by #3339 is proven in `value_math` (`stored_roundtrip`).
- **Floats** are bit patterns; no IEEE reasoning is needed or done.
- **`MvccGraph` concurrency** is not modelled.
- **Hostile or malformed RDB payloads** beyond truncation were not attempted.
- **Header and schema codecs** are hypotheses here, proven in `redis_layer`.
- **The UNIQUE enforcement path after #3334** (index lookup) is modelled in `pending_commit`, not here.

## Build and check

```bash
cd proofs/graph_persist && lake build    # this project
bash proofs/lean_ci.sh graph_persist     # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                  # repo-wide coverage report
```

**When you change `src/serializers/encoder/mod.rs`, `src/serializers/decoder/mod.rs`, `src/redis_type.rs`, `graphblas/serialization.rs`, `AttributeStore::encode_with_range`/`decode_with_count`, the value codec in `value.rs`, or `build_composite_key`:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match.
3. Re-run the build.

A new field in the payload needs an `RT` lemma for it; `load_encode` will not build until it has one. If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR.
