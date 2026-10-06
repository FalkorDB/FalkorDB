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
* `Payload` — `open_payload`, `maybe_compress`, `Records::next`: `openPayload_maybeCompress` (compression is
  transparent for < 4 GiB of records, zstd/crc32 as the `CodecOK` hypothesis), `maybeCompress_shrinks`,
  `records_concat`.
* `Format`, `ModRs` — `format.rs` (`describe_spec`, `build_ok`, `fmtReplicate_transparent`), `effects/mod.rs`
  (`describeTop_spec`, `applyTop_spec`, `EffectsBuffer`), `v3/mod.rs` tags/`seal`, `payload.rs`, `announce.rs`,
  `LocalName`.

Gaps: `maybe_compress` writes `plain_len` `as u32`; a payload of ≥ 4 GiB of records would be framed with a
wrapped length (unreachable in practice; not reproduced). zstd/crc32 are hypotheses (`CodecOK`).
`staging.rs`/`test_aux.rs` are `#[cfg(test)]` modules (NOT COVERED, TEST-ONLY).
Bugs: #2914 (encoder wrote records its reader rejects) — **fixed by #2916 (`2c874022a`)**, now theorems.
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
