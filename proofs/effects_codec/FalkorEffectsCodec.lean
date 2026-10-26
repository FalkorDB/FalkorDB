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
* `RecFull`, `RecRT` — **all fourteen `Record` variants**, `write_header`, `check_row_shape`, `Record::encode`, `read_record`;
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
  (reservations ≤ remaining / 4), `hostile_refused_second_level`. History: `pre3267_readOne`,
  `pre3267_capTrace`, `pre3267_hostile_quadratic` (90 elements from 80 bytes).
* **#3278 (`dcbd445d1`, #3276)** — `maybe_compress` refuses a frame whose lengths overflow `u32`.
  `openPayload_maybeCompress`, `fmtReplicate_transparent`, `seal_transparent` lose their `< 2^32`
  hypothesis; new `framePrefix_*`. History: `pre3278_maybeCompress`, `pre3278_openPayload_maybeCompress`,
  `pre3278_wrapped_frame_unreadable`.

Gaps: zstd/crc32 are hypotheses (`CodecOK`).
`staging.rs`/`test_aux.rs` are `#[cfg(test)]` modules (NOT COVERED, TEST-ONLY).
Bugs: #2914 (encoder wrote records its reader rejects) — **fixed by #2916 (`2c874022a`)**, now theorems;
quadratic list reservation — **fixed by #3267**; ≥ 4 GiB compressed frame — **fixed by #3278**.

## Re-pinned to 5e475e6c9 (#3436)
No model change. `graph/src/effects/mod.rs` and `v3/format.rs` build entry points take `&Graph`
instead of `&AtomicRefCell<Graph>` (a read query borrows its graph once, #3436); every
`COVERAGE.tsv` row is re-pinned (mod.rs -1 line from the removed import, format.rs -2).
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
#print axioms pre3278_wrapped_frame_unreadable
