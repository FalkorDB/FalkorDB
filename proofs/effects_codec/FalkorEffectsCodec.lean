/-
# Effects v3 byte codec (`graph/src/effects/**`) — wave 4

Modules (each file's header has its Lean ↔ Rust table):
* `Prim`, `Value`, `Records` — primitives, `Value` codec, the original CREATE_NODE/ADD_ATTRIBUTE model.
* `Cursor` — the real `Reader { buf, pos }` and `EffectWrite`: every method refines the suffix model
  (`Rd.take_refines`, `Rd.guardCount_refines`, `Rd.bits_refines`, `Rd.takeN_refines` = chunked read equals
  per-element reads), `vec_*` laws, `schemaId_roundtrip`.
* `RecBlocks` — `Blk` (writer, reader, roundtrip-under-`ok`) and `Blk.seq` (dependent composition, proven
  once); blocks for ints, strings, `LabelSet`, `AttrIds`, ids, rows, `put_opt`/`take_opt`, counted lists.
* `RecArms`, `RecOpts` — tags, `index_type_of`, `IndexFields`, `IndexSchemas` (encoder sorts by id),
  constraint props, `IndexFieldOptions` (`bSim_lossy`: the similarity name is normalised — also seen).
* `RecShape` — **#2916 (`2c874022a`)**: the encoder now refuses exactly the shapes `read_record` rejects
  (`encRecord_ok_shape`, `encRecord_rowShape_refused`, `encRecord_empty_refused`, `readRecord_empty`), so
  `readRecord_encRecord_wf`: every record the encoder writes reads back with no shape hypothesis. The wave-1
  counterexample (a spare row read as the next opcode, `Records.lean`) is now a refusal `#guard`.
* `RecFull`, `RecRT` — **all fifteen `Record` constructors** (fourteen opcodes: `ADD_SCHEMA` has separate label and relationship-type constructors), `write_header`, `check_row_shape`, `Record::encode`, `read_record`;
  `readRecord_encRecord`: every record round-trips whatever follows it (UPDATE_EDGE, ADD_SCHEMA, index and
  constraint DDL included); `checkEndpoints_ok`, `checkIndex_ok`, `writeHeader_spec`, `relType_rt`.
* `Payload` — `open_payload`, `maybe_compress`, `frame_prefix`, `Records::next`: `openPayload_maybeCompress`
  (compression is transparent for records of any length since #3278, zstd/crc32 as the `CodecOK` hypothesis),
  `maybeCompress_shrinks`, `framePrefix_none`, `records_concat`.
* `Format`, `ModRs` — `format.rs` (`describe_spec`, `build_ok`, `fmtReplicate_transparent`), `effects/mod.rs`
  (`describeTop_spec`, `applyTop_spec`, `EffectsBuffer`), `v3/mod.rs` tags/`seal`, `payload.rs`, `announce.rs`,
  `LocalName`.

Re-targeted to origin/main `1d15bf713` (from `0f45a8685`):
* **#3267 (`eaa65eeba`)** — `Value::decode` bounds a list's width against the whole stack (`owed`).
  `decodeV`/`readOne`/`loop` now carry `owed`; `decodeV_encV` and `decodeV_safe` re-proved (the
  `owed -= 1` never underflows: `owed = owedOf st`), new `capTrace_linear`/`decode_reserves_linear`
  (reservations ≤ remaining / 4), `hostile_refused_second_level`.
* **#3278 (`dcbd445d1`, #3276)** — `maybe_compress` refuses a frame whose lengths overflow `u32`.
  `openPayload_maybeCompress`, `fmtReplicate_transparent`, `seal_transparent` lose their `< 2^32`
  hypothesis; new `framePrefix_*`.

Gaps: zstd/crc32 are hypotheses (`CodecOK`).
`staging.rs`/`test_aux.rs` are `#[cfg(test)]` modules (NOT COVERED, TEST-ONLY).
Bugs: #2914 (encoder wrote records its reader rejects) — **fixed by #2916 (`2c874022a`)**, now theorems;
quadratic list reservation — **fixed by #3267**; ≥ 4 GiB compressed frame — **fixed by #3278**.

## Re-pinned to 5e475e6c9 (#3436)
No model change. `graph/src/effects/mod.rs` and `v3/format.rs` build entry points take `&Graph`
instead of `&AtomicRefCell<Graph>` (a read query borrows its graph once, #3436); every
`COVERAGE.tsv` row is re-pinned (mod.rs -1 line from the removed import, format.rs -2).

## PR #3610 (durations carry months, days, seconds; effects wire v4, v3 still read)
No counterexample. Model changes (Lean file:line ↔ Rust file:line):
| Lean | Rust | change |
|---|---|---|
| `Value.lean:129` `encV` | `v3/value.rs:38` encode | `Val.duration m d s` → `T_DURATION · i64 m · i64 d · i64 s` (28 bytes ≥ `MIN_VALUE_BYTES`, so `encV_len`, `capTrace_linear`, `decode_reserves_linear` keep their statements) |
| `Value.lean:201` `durNewOk` | `temporal.rs:515` `Dur::new` | months/days fit `i32`, the legacy sum fits `i64` |
| `Value.lean:237` `durationFromLegacy` | `temporal.rs:719` | decomposition, `duration_to_legacy` check, fallbacks; `durationFromLegacy_some` (never fails) |
| `Value.lean:293` `readOne` (`lg`) | `v3/value.rs:243` read_one (`legacy_duration`) | v3: one `i64` via `durationFromLegacy`; v4: three `i64`, `Dur::new` refusal = `BadValueType(T_DURATION)` |
| `Value.lean:369/372/374` `decodeV`/`decodeV3`/`decodeV4` | `v3/value.rs:197/148/155` | `decode_value(r, legacy)` and its two impls |
| `RecFull.lean:152` `readRecord … lg` | `v3/records.rs:619` `read_record::<VERSION>` | `lg = (VERSION = 3)`, threaded through `rRows`/`bRows`/bodies/`RecOk` |
| `Payload.lean:47` `openPayload` | `v3/records.rs:1308` | accepts 4 or 3, returns `(bytes, version)` |
| `Payload.lean:241` `payloadRecords` | `v3/records.rs:1255/1282` | `Records::next` decodes in the payload's version |
| `Payload.lean` `newBuffer` | `v3/mod.rs:486` | `[4, 0]` |
| `Format.lean:64` `describe` | `v3/format.rs:45` | lines carry the header's version |
| `ModRs.lean:29/57` `describeTop`/`applyTop` | `effects/mod.rs:248/294` | version 4 or 3 |
New theorems: `decodeV4_encV`, `decodeV3_encV`, `encV_duration`, `readOne_v3_duration` (v3 reads exactly 8
bytes; the duration's legacy offset is the stored one), `decodeV3_duration`, `readOne_v4_duration`,
`openPayload_versions` (3, 4 accepted; anything else `UnsupportedVersion`), `payloadRecords_maybeCompress`,
`describe_unreadable`, `describeTop_versions`. Re-proved on the new model: `decodeV_encV`, `decodeV_safe`,
`capTrace_linear`, `readOne_*`, `readRecord_encRecord(_wf)`, `records_concat`, `openPayload_*`,
`describe_spec`, `applyTop_spec`. `WF utf8 true (.duration …)` is `False`: a v3 reader never sees a
v4 duration (no encoder writes v3 any more). `v3/mod.rs` rows re-pinned +1.
-/
import FalkorEffectsCodec.Prim
import FalkorEffectsCodec.Value
import FalkorEffectsCodec.Records
import FalkorEffectsCodec.Cursor
import FalkorEffectsCodec.ModRs
import FalkorEffectsCodec.RecShape

open FalkorCodec in
#print axioms decodeV_encV
open FalkorCodec in
#print axioms decodeV_safe
open FalkorCodec in
#print axioms readRec_createNode
open FalkorCodec in
#print axioms readAll_concat
open FalkorCodec in
#print axioms readRecord_encRecord_wf
open FalkorCodec in
#print axioms capTrace_linear
open FalkorCodec in
#print axioms openPayload_maybeCompress
open FalkorCodec in
#print axioms readOne_v3_duration
open FalkorCodec in
#print axioms readOne_v4_duration
open FalkorCodec in
#print axioms decodeV3_duration
open FalkorCodec in
#print axioms decodeV4_encV
open FalkorCodec in
#print axioms openPayload_versions
open FalkorCodec in
#print axioms payloadRecords_maybeCompress
open FalkorCodec in
#print axioms describe_spec
open FalkorCodec in
#print axioms applyTop_spec
open FalkorCodec in
#print axioms records_concat
