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
* `RecShape` — the encoder refuses exactly the shapes `read_record` rejects
  (`encRecord_ok_shape`, `encRecord_rowShape_refused`, `encRecord_empty_refused`, `readRecord_empty`), so
  `readRecord_encRecord_wf`: every record the encoder writes reads back with no shape hypothesis. The wave-1
  case (a spare row read as the next opcode, `Records.lean`) is a refusal `#guard`.
* `RecFull`, `RecRT` — **all fifteen `Record` constructors** (fourteen opcodes: `ADD_SCHEMA` has separate label and relationship-type constructors), `write_header`, `check_row_shape`, `Record::encode`, `read_record`;
  `readRecord_encRecord`: every record round-trips whatever follows it (UPDATE_EDGE, ADD_SCHEMA, index and
  constraint DDL included); `checkEndpoints_ok`, `checkIndex_ok`, `writeHeader_spec`, `relType_rt`.
* `Payload` — `open_payload`, `maybe_compress`, `frame_prefix`, `Records::next`: `openPayload_maybeCompress`
  (compression is transparent for records of any length, zstd/crc32 as the `CodecOK` hypothesis),
  `maybeCompress_shrinks`, `framePrefix_none`, `records_concat`.
* `Format`, `ModRs` — `format.rs` (`describe_spec`, `build_ok`, `fmtReplicate_transparent`), `effects/mod.rs`
  (`describeTop_spec`, `applyTop_spec`, `EffectsBuffer`), `v3/mod.rs` tags/`seal`, `payload.rs`, `announce.rs`,
  `LocalName`.

Gaps: zstd/crc32 are hypotheses (`CodecOK`).
`staging.rs`/`test_aux.rs` are `#[cfg(test)]` modules (`mod staging;`/`mod test_aux;` in `v3/mod.rs`): test helpers, not engine code, so they carry no COVERAGE rows (`coverage.sh` excludes test fns from its denominators).
Bugs: no open bugs. Fixed ones are in this file's git history.

## `IdLink`
* `IdLink` discharges the last hypothesis: `idListCodec ρ hρ : IdCodec` is built from `proofs/id_list`
  (`IdListWire.encodeIds`/`readIds`, a Lake path dependency), so `records_concat_idList` (`Payload::records` /
  `Records::next` over the real IdList codec) is PROVEN. The only
  assumption left is `RoaringOK ρ`, the `roaring` crate's serialize/deserialize contract (an external library,
  as in id_list). Also: `encodeIds_bytes` (every byte the encoder writes is a byte), `idDec_roundtrip`, `idDec_good`.
-/
import FalkorEffectsCodec.Prim
import FalkorEffectsCodec.Value
import FalkorEffectsCodec.Records
import FalkorEffectsCodec.Cursor
import FalkorEffectsCodec.ModRs
import FalkorEffectsCodec.RecShape
import FalkorEffectsCodec.IdLink

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
#print axioms records_concat_idList
