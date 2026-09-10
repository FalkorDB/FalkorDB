/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "effects_bytes.h"
#include "effects_v3_id_list.h"

#include <stdint.h>

// v3 wire writers for the shared blocks
//
// Everything here writes LITTLE-ENDIAN explicitly, byte at a time, rather than
// copying a struct or an integer's storage. v2 writes its records through
// packed structs, which is correct only while both engines run little-endian
// and is a silent wrong answer the day one does not. The widths here are also
// chosen per value rather than fixed, so there is no native type to copy from
// in the first place.

// the segment header's field positions come from effects_v3.h

// write an unsigned value at a fixed width, little-endian
void EffectsV3_WriteUint
(
	EffectsBytes *out,  // sink
	uint64_t v,         // value
	uint8_t width       // 1, 2, 4 or 8 bytes
);

// write one segment: its header byte, then its payload
//
// THE WIDTHS COME FROM THE SEGMENT, not from its values. A freshly built
// segment was given the narrowest width that holds each value when the builder
// converted it; a decoded one carries whatever width its peer chose. Writing
// what the struct says is what makes a decode-then-encode round trip reproduce
// the peer's bytes rather than this engine's arithmetic - a peer may
// legitimately use a wider field than it needs, and narrowing it on the way
// back out is a different buffer for the same ids.
//
// a Range and a Repeat write two values at the widths their header declares; an
// Ascending writes a u32 blob length then the portable roaring serialization,
// and its width fields are unused and written zero
void EffectsV3_EncodeSegment
(
	const EffectsV3Segment *s,  // segment to write
	EffectsBytes *out           // sink
);

// write an IdList: a u32 segment count, then the segments
//
// both counts are on the wire - the segment count here, and every segment's own
// length - even though the record's id count could imply them. A segment list
// has to be well-formed on its own rather than only inside the record carrying
// it: inferring either makes a truncated list indistinguishable from a complete
// one, and lets a wrong record count be absorbed silently by the final segment,
// binding rows to the wrong entities instead of failing
void EffectsV3_EncodeIdList
(
	const EffectsV3IdList *l,  // list to write
	EffectsBytes *out          // sink
);

// write one record: its opcode, then whatever that opcode carries
//
// The shape comes FIRST, before the ids, for every batchable record without
// exception - a record is self-describing before its rows. AttrValues is the
// one part that follows the IdList, because it is per row rather than per
// record, which is why the attribute ids and their values are two blocks and
// not one.
//
//   1 UPDATE_NODE     count · LabelSet · AttrIds · IdList · AttrValues
//   2 UPDATE_EDGE     count · RelType  · AttrIds · IdList · AttrValues
//   3 CREATE_NODE     count · LabelSet · AttrIds · IdList · AttrValues
//   4 CREATE_EDGE     count · RelType  · AttrIds · IdList · src · dst · AttrValues
//   5 DELETE_NODE     count · LabelSet · IdList
//   6 DELETE_EDGE     count · RelType  · IdList · src · dst
// 7·8 SET/REMOVE_LABELS   count · LabelSet · IdList
//   9 ADD_SCHEMA      SchemaType · id · name        - no count, inherently one
//  10 ADD_ATTRIBUTE   attr_id · name                - no count, inherently one
//
// DELETE_NODE's LabelSet is not decoration: they are the labels the node
// actually held, captured as it was deleted, and a replica needs them to clear
// the right label-scoped index documents. `MATCH (n:A) DELETE n` over an
// (:A:B) node must clear :B's indexes too, and the pattern cannot say so.
void EffectsV3_EncodeRecord
(
	const EffectsV3Record *r,  // record to write
	EffectsBytes *out          // sink
);

// write a record whose values are ALREADY ENCODED
//
// The grouping accumulator encodes each row's values as the row arrives, so by
// emission time it holds bytes rather than SIValues - the values it was handed
// belonged to the caller and are long gone. Re-encoding is not an option, and
// keeping every SIValue alive until the query ends would mean owning a copy of
// every property written.
//
// 'values' supplies the AttrValues block verbatim; the record's own 'values'
// and 'n_values' are ignored. Everything before that block is written exactly
// as EffectsV3_EncodeRecord writes it.
void EffectsV3_EncodeRecordWithRawValues
(
	const EffectsV3Record *r,   // record to write
	const EffectsBytes *values, // the AttrValues block, already encoded
	EffectsBytes *out           // sink
);
