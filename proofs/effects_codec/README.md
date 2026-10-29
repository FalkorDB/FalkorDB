# effects_codec: the byte format a master sends its replicas

When a write query commits, the master does not send the query to its replicas. It sends an *effects payload*: a byte buffer listing what changed (nodes created, attributes set, edges deleted, labels added, indexes created, and so on). The replica decodes it and applies it. This project proves that the byte codec for that payload, `graph/src/effects/**` (format version 3), is correct:

- every record the master encodes decodes on the replica as the same record;
- every value (strings, floats, lists, temporal values, vectors, points) round-trips bit for bit;
- the decoder never panics and never reads past the buffer, whatever bytes it is given;
- compression is transparent: the replica opens exactly the records the master built.

Replication, AOF, and client `GRAPH.EFFECT` all go through this codec. A codec bug means a replica that silently differs from its master, or one that refuses a payload and has to resync from scratch.

What happens *before* encoding (which records a query emits) and *after* decoding (how the replica applies them to the graph) is proven in [`../effects_emit_apply`](../effects_emit_apply). This project is only about bytes.

The full Lean↔Rust mapping is in the header of [`FalkorEffectsCodec.lean`](FalkorEffectsCodec.lean) and in the header table of each file under `FalkorEffectsCodec/`. This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. Does every record the master writes read back as itself on the replica?**

The payload is a list of records. There are fourteen opcodes (`Opcode` in `v3/mod.rs`): `UPDATE_NODE`, `UPDATE_EDGE`, `CREATE_NODE`, `CREATE_EDGE`, `DELETE_NODE`, `DELETE_EDGE`, `SET_LABELS`, `REMOVE_LABELS`, `ADD_SCHEMA` (a new label or relationship type), `ADD_ATTRIBUTE`, `CREATE_INDEX`, `DROP_INDEX`, `CREATE_CONSTRAINT` and `DROP_CONSTRAINT`.

- **Before #2916, no.** `Record::encode` happily wrote two shapes that `read_record` cannot read: a batch with zero ids, and a row block whose length is not `count × attrs`. A `CREATE_NODE` with one id, one attribute and two values was written fine; the reader took one value, then parsed the spare value as the next opcode.
- **On main, yes.** `readRecord_encRecord_wf`: for every one of the fourteen variants, if the encoder returns bytes, the reader returns exactly that record and leaves whatever follows untouched. The only conditions left are per value: integers fit their width and strings are valid UTF-8. Both hold for anything the engine stores.

**2. Can a hostile or corrupt payload crash the replica?**

No, for the value decoder. `decodeV_safe` says that for *any* bytes, `Value::decode`:

- never panics (every `panic!` and every out-of-bounds slice in Rust is modelled as an error `panic`, and the theorem says that error never happens; this includes #3267's `owed -= 1`, because `owed` is always the number of children the open frames still need);
- never reads past the input (on success, the input is the bytes consumed followed by the rest);
- terminates (the Lean model needs a step budget, and `loop_fuel_enough` proves the budget never runs out: every turn reads a 4-byte tag);
- reserves memory linearly (`decode_reserves_linear`): the list capacities one decode asks for sum to at most a quarter of the input's length.

- **Before #3267, not the last one.** Each list's claimed width was checked against the remaining bytes alone, so every level of `[[[…` could claim the whole remainder. `hostile 10` (80 bytes) reserved 90 elements; 1 MB reserved about 256 GB, and the Rust test measured 2.3 GB of RSS from a client-sent `GRAPH.EFFECT`.
- **On main, yes.** `read_one` checks the claim together with the siblings its ancestors still owe (`guard_count(n + owed)`). `capTrace_linear` proves the bound for every input, and `hostile_refused_second_level` shows the crafted input is refused at its second level, as the Rust test asserts.

**3. Do floats survive exactly, including NaN and `-0.0`?**

Yes. On this wire a float *is* its 64-bit pattern, so "round-trips" means bit for bit: NaN payloads, the sign of a NaN, and `-0.0` all come back unchanged. The model deliberately does not use Lean's `Float`, because Lean's runtime canonicalises NaN and Rust does not. Three `#guard` checks in `Value.lean` decode a signalling NaN with payload 1, a negative quiet NaN, and `-0.0`.

**4. Is compression transparent?**

Yes, under one stated assumption about zstd. `openPayload_maybeCompress`: whatever `maybe_compress` produces, `open_payload` returns the original record bytes, for payloads of any length.

- **Before #3278, only below 4 GiB.** Both frame lengths were written `as u32`. A payload of 4 GiB of records or more that zstd shrank was framed with a wrapped length, and no zstd can make such a frame open to its records (counterexample removed; in 17ccc5412). The replica would refuse it and loop on forced resyncs.
- **On main, yes.** `maybe_compress` leaves such a payload uncompressed, and `frame_prefix` refuses a length that does not fit (`framePrefix_none`). `fmtReplicate_transparent` lifts this to `EffectsFormat::replicate`: what the replica receives opens to exactly the records the master built. `maybeCompress_shrinks`: a payload is only ever compressed when that makes it smaller.

zstd and crc32 are outside Rust, so the proof does not model them. It takes their contract as the hypothesis `CodecOK`: decompressing what zstd compressed gives the input back, and crc32 returns a 32-bit number.

**5. Does a payload of several records come back as the same list, in order?**

Yes: `records_concat`. If each record's bytes read back as that record whatever follows (question 1), then the concatenation reads back as the list, in order, with nothing left over and nothing skipped. This is what `Payload::records` and `Records::next` do. Those two rows are **CONDITIONAL** (see [Coverage](#coverage)).

**6. Does every index option survive the trip?**

All but one. `bSim_lossy`: the similarity function of a vector index is normalised by the wire. The master writes a code (1 for `ip`, 2 for `cosine`, 0 for anything else) and the replica decodes it back to lowercase `ip`, `cosine`, or `euclidean`. So `COSINE` arrives as `cosine`, and any name the encoder does not recognise arrives as `euclidean`. This matters only for a name the index layer refuses anyway: `index/mod.rs` builds a vector index only for `euclidean`, `ip` or `cosine`, and errors on anything else.

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `readRecord_encRecord_wf` | Every record the encoder writes, of all fourteen kinds, reads back as itself whatever follows it, with no hypothesis about its shape | `RecShape.lean` |
| `readRecord_encRecord` | The per-variant round trip it is built on, under the explicit well-formedness condition `RecOk` | `RecRT.lean` |
| `encRecord_ok_shape` | A record the encoder accepts has at least one id, `count × attrs` row values, and endpoint columns as long as the id column | `RecShape.lean` |
| `encRecord_empty_refused`, `encRecord_rowShape_refused` | …and a record without that shape is refused, with the error that names why | `RecShape.lean` |
| `decodeV_encV` | `decode(encode(v)) = v` for every value, bit for bit, and the decoder consumes exactly the encoding | `Value.lean` |
| `decodeV_safe` | On any bytes at all, the value decoder never panics, never over-reads, and always terminates | `Value.lean` |
| `capTrace_linear`, `decode_reserves_linear` | #3267: the capacities the decoder reserves sum to at most a quarter of the input | `Value.lean` |
| `encV_total` | Every value the engine accepts as a property encodes; the encoder's `panic!` arm (maps, nodes, edges, paths) is unreachable because those are refused where the value is stored | `Value.lean` |
| `openPayload_maybeCompress`, `fmtReplicate_transparent` | Compression is invisible: the replica opens exactly the records the master built, at any length (assuming `CodecOK`) | `Payload.lean`, `Format.lean` |
| `records_concat` | A payload built from encoded records yields those records, in order | `Payload.lean` |
| `Rd.take_refines`, `Rd.takeN_refines` | The real cursor (`Reader { buf, pos }`) does exactly what the simpler "unread suffix" model does, so every theorem above holds of the Rust reader | `Cursor.lean` |
| `describe_spec` | The diagnostic dump of a payload (`EffectsPayload::describe`, logged by `src/divergence_guard.rs`) reports the true record count, at most 16 record lines, and one error line exactly when decoding stopped | `Format.lean` |
| `bSim_lossy` | The vector similarity name is normalised by the wire (see question 6) | `RecOpts.lean` |

### How `readRecord_encRecord_wf` is proved

1. **Model the bytes.** `Bytes` is a list of `u8`. A reader action is a function from the unread bytes to either an error or a result plus the bytes left. `Cursor.lean` then models the real `Reader { buf, pos }` field for field and proves each method does the same thing as the suffix model (`Rd.take_refines`, `Rd.guardCount_refines`, `Rd.bits_refines`, `Rd.takeN_refines`).
2. **Model the values.** `Val` has one constructor per `Value` variant the codec handles. Floats are `BitVec 64`, `i64` is its two's-complement bits. `encV` is `EffectEncode<3> for Value`. `decodeV` is `Value::decode`, including its explicit stack of list frames (`Frame`, `deliver`, `loop`), because the Rust decoder is iterative, not recursive.
3. **Prove blocks once, compose them.** A `Blk α` bundles a writer, a reader, and a proof that the reader inverts the writer whatever bytes follow. `Blk.seq` composes two blocks the way a Rust `encode` writes one field after another, including when the second depends on the first (the id list's length comes from the header count), and its round trip is proven once. Labels, attribute ids, rows, options, index fields and constraint properties are all blocks.
4. **Model the records.** `Record` has all fourteen variants. `encRecord` is `Record::encode`, `readRecord` is `read_record`, `writeHeader` is `write_header`, `checkRowShape` is `check_row_shape`. `readRecord_encRecord` is then a case split on the variant, each arm built from the block lemmas.
5. **Remove the shape hypothesis.** `readRecord_encRecord` needs `RecOk`, which includes "at least one id" and "rows = count × attrs". `encRecord_ok_shape` proves the encoder now checks both, so `readRecord_encRecord_wf` needs only `RecWF`: integers fit their width and strings are valid UTF-8.
6. **The id lists are abstract.** Node and edge ids travel as an `IdList`. Here that codec is a parameter, `IdCodec`, that comes with its own round-trip law. That law is proven in [`../id_list`](../id_list), but the two projects are not linked, which is why the two payload-level rows are CONDITIONAL.

## Bugs found

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| 1 | the wave-1 `CREATE_NODE` counterexample, now the `#guard`s after `encRec_refuses_shape` in `Records.lean` | The v3 encoder wrote records its own reader rejects: a batch of zero ids, and a row block that is not `count × attrs`. The spare value was parsed as the next opcode | Fixed by #2916 (closes #2914); now `encRecord_ok_shape`, `readRecord_encRecord_wf` |
| 2 | removed (17ccc5412) | Nested lists could each claim the whole remaining buffer, so the decoder's reservations grew with the square of the payload (2.3 GB RSS from 1 MB) | Fixed by #3267 (`eaa65eeba`); now `capTrace_linear`, `hostile_refused_second_level` |
| 3 | removed (17ccc5412) | A payload of 4 GiB of records or more was compressed with a wrapped length that no replica can open | Fixed by #3278 (`dcbd445d1`, issue #3276); now `openPayload_maybeCompress` with no length hypothesis, `framePrefix_none` |

Two findings are modelled but not bugs on main:

- **Vector similarity normalisation** (`bSim_lossy`): an unknown name becomes `euclidean` on the replica. Harmless, because the index layer refuses unknown names.
- **Deep nesting** (`nest_decodes`): a value `d` levels deep costs only `8d + 4` bytes and decodes. The Rust decoder itself is iterative and does not overflow the stack; dropping the value is recursive (see below). The quadratic *width* reservation that used to come with it is bug 2 above.

## Files

| File | What's in it |
|---|---|
| `FalkorEffectsCodec.lean` | Root: the report header, imports, `#print axioms` for the headline theorems |
| `Prim.lean` | `Bytes`, decode errors `DErr`, little-endian integers, `take`, `guardCount`, strings; "never panics" and "only consumes" combinators |
| `Cursor.lean` | The real `Reader { buf, pos }` (`Rd`) and `EffectWrite`; every method refines the suffix model |
| `Value.lean` | `Val`, the type tags, `encV`, the iterative `decodeV` with #3267's `owed`; `decodeV_encV`, `decodeV_safe`, `encV_total`, `capTrace_linear`; the pre-#3267 counterexamples (removed; in 17ccc5412) history |
| `Records.lean` | The original model of a subset of records (`Rec`, `encRec`, `readRec`, `readAll`), the abstract `IdCodec`, `readAll_concat`, the refused-shape `#guard`s |
| `RecBlocks.lean` | `Blk` and `Blk.seq`: blocks for integers, strings, label sets, attribute ids, ids, rows, optional fields, counted lists |
| `RecArms.lean` | Entity, constraint and status tags; `indexTypeOf`; attribute and schema references; index schemas (the encoder sorts them by id) |
| `RecOpts.lean` | `IndexFieldOptions`: booleans, the similarity function (`bSim_lossy`), stopwords, vector options |
| `RecFull.lean` | `Record` with all fourteen variants, `writeHeader`, `checkRowShape`, `encRecord`, `readRecord` |
| `RecRT.lean` | `readRecord_encRecord` for every variant, `writeHeader_spec`, `checkEndpoints_ok`, `checkIndex_ok`, `relType_rt` |
| `RecShape.lean` | #2916: `encRecord_ok_shape`, the refusal theorems, `readRecord_empty`, `readRecord_encRecord_wf` |
| `Payload.lean` | `openPayload`, `maybeCompress`, `framePrefix`, the `Codec`/`CodecOK` hypothesis, `readAllRecords`, `records_concat`; the pre-#3278 counterexamples (removed; in 17ccc5412) history |
| `Format.lean` | `format.rs`: `fmtReplicate_transparent`, `describe_spec`, `build_ok`, `payload.rs`/`announce.rs` glue, `LocalName` |
| `ModRs.lean` | `effects/mod.rs`: version dispatch (`describeTop_spec`, `applyTop_spec`), `EffectsBuffer`; `v3/mod.rs` tags and `seal` |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

The project has no Lake dependencies.

## Coverage

`COVERAGE.tsv` has 141 rows, covering `graph/src/effects/**`:

| Bucket | Rows |
|---|---|
| PROVEN | 116 |
| CONDITIONAL | 2 |
| AXIOMATISED | 1 |
| NOT COVERED | 22 |

- **CONDITIONAL:** `Payload::records` and `Records::next` rest on `records_concat`, which assumes the `IdCodec` round trip. That round trip is proven in `id_list` but not linked here.
- **AXIOMATISED:** `ReplicationSink::replicate` is implemented in `src/` by `RedisModule_Replicate` (Redis FFI). `format.rs`'s `replicate` is proven up to the sink (`fmtReplicate_transparent`).
- **NOT COVERED:** all 22 are test-only: `v3/staging.rs` and `v3/test_aux.rs` are `#[cfg(test)]` modules and are not compiled into the engine.

## What the model does not cover

- **zstd and crc32** are the hypothesis `CodecOK`, not proven.
- **`IdList`** is the abstract `IdCodec`; its round trip is proven in `id_list`, not here.
- **Applying the records to the graph**, and which records a query emits, are parameters here (`fmtApply`, `fmtBuildIndex`, `fmtBuildConstraint`). They are proven in `effects_emit_apply`.
- **Stack depth on drop.** The decoder is iterative, but `Value` is recursive and its `Drop` is not. The model proves arbitrarily deep values decode (`nest_decodes`); it says nothing about the stack used to drop them.
- **Allocation.** Memory is not modelled byte for byte; what is proven is the sum of the `with_capacity` requests (`capTrace_linear`), not the allocator's behaviour.

## Build and check

```bash
cd proofs/effects_codec && lake build    # this project
bash proofs/lean_ci.sh effects_codec     # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                  # repo-wide coverage report
```

**When you change `graph/src/effects/reader.rs`, `writer.rs`, `v3/records.rs`, `v3/value.rs`, `v3/blocks.rs`, `v3/format.rs` or `effects/mod.rs`:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match.
3. Re-run the build.

Adding a record kind or a value tag means a new arm in `Record`/`readRecord` or `Val`/`readOne`, and the round-trip theorems will not build until the new arm is proven. If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR.
