/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "effects.h"

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

//------------------------------------------------------------------------------
// effects v3 - the decode/apply seam
//------------------------------------------------------------------------------
//
// v2 fuses decoding and applying: Effects_Apply takes a buffer and a
// GraphContext and walks one into the other. That makes three things impossible
// which v3 needs:
//
//   * a round-trip test - there is no intermediate value to hand back to the
//     encoder, so decode(encode(x)) == x cannot be written
//   * a cheap truncation corpus - every prefix would need a live graph, and a
//     failure could not be attributed to "the reader refused these bytes"
//     rather than "apply found divergence"
//   * an encoder reachable without a graph - EffectsBuffer_Add*Effect takes
//     live entities
//
// v3 therefore splits them, mirroring what the Rust side does:
//
//     encode:   records --EffectsV3_Encode-->  bytes
//     decode:   bytes   --EffectsV3_Decode-->  records
//     apply:    records --EffectsV3_Apply-->   Graph
//
// `EffectsV3Records` is the pivot both directions turn on. It is a plain value:
// it holds no GraphContext, no Graph, no live Node or Edge, and it can be
// decoded, compared, re-encoded and freed with no graph in scope.
//
//------------------------------------------------------------------------------
// OWNERSHIP - this header is the shared contract
//------------------------------------------------------------------------------
//
//   * the entry points and the status enum below are fixed by the organizer
//     and are stable; build against them
//   * the concrete `EffectsV3Records` / record struct definitions are published
//     by THE READER, which has to transcribe the record layouts from
//     docs/effects-v3.md in order to decode them at all. THE WRITER consumes
//     that definition as its input contract - reading a struct definition is
//     not reading the other half's implementation
//   * THE REFEREE builds against this header only, and does not need the record
//     fields: a round trip passes records through without inspecting them
//
// Two rules the type must preserve, both load-bearing:
//
//   * IDS ARE NEVER EXPANDED AT DECODE TIME. An IdList is held as its segments.
//     One valid segment describes four billion ids in seven bytes, and
//     GRAPH.EFFECT is applied inline on the main thread with no timeout, so a
//     decoder whose cost is unbounded by its input is a denial of service.
//     Expanding is EffectsV3_Apply's decision, made where the graph is in scope
//     and the memory is inherent to the write.
//   * DECODING AND RE-ENCODING IS BYTE-IDENTICAL. The segmentation the peer
//     chose is preserved rather than recomputed; re-pushing the ids would re-run
//     the collapse rule and could group them differently.
//
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// the record model - published by THE READER, consumed by THE WRITER
//------------------------------------------------------------------------------
//
// TRANSCRIBED FROM THE BYTES, NOT FROM THE RECORD TABLE. The spec described
// record layout twice and the two disagreed: the INVARIANT put the shape before
// the rows, while the "Changed in v3" table put the IdList first for records 3
// through 8 and omitted the LabelSet from DELETE_NODE entirely.
//
// The invariant is right and the table was an earlier draft. Established by
// decoding all 25 fixtures in .handover/fixtures/ against their .json - under
// the table's ordering, none of the id-carrying records decode at all - then
// ruled on by the Rust side, which owns the format. SIX of the eight batchable
// records were documented backwards, not the two found first: DELETE_EDGE and
// SET/REMOVE_LABELS too. Spec corrected at e3e227f77 on feat/effects-v3, which
// is also where it now lives.
//
// So every batchable record is:
//
//     u32 opcode
//     u32 count
//     <shape>      LabelSet (node-shaped) | RelType (edge-shaped)
//     <attr ids>   AttrSet header, only on records that carry values
//     IdList       the entity ids, positionally bound to the rows
//     IdList x2    src then dst, only on CREATE_EDGE / DELETE_EDGE
//     <values>     count * n_attrs SIValues, row-major
//
//   record            shape        attr ids  ids  src/dst  values
//   1  UPDATE_NODE    LabelSet     yes       yes  -        yes
//   2  UPDATE_EDGE    RelType      yes       yes  -        yes
//   3  CREATE_NODE    LabelSet     yes       yes  -        yes
//   4  CREATE_EDGE    RelType      yes       yes  yes      yes
//   5  DELETE_NODE    LabelSet     -         yes  -        -
//   6  DELETE_EDGE    RelType      -         yes  yes      -
//   7  SET_LABELS     LabelSet     -         yes  -        -
//   8  REMOVE_LABELS  LabelSet     -         yes  -        -
//   9  ADD_SCHEMA     inherently singular - no count, no ids
//   10 ADD_ATTRIBUTE  inherently singular - no count, no ids
//
// DELETE_NODE carrying a LabelSet is not a transcription slip: the fixture
// rec_delete_node.hex holds labels [0,1,2] ahead of its IdList, and the record
// does not decode without them. A replica deleting a node needs its labels to
// maintain the label matrices and indexes, so this reads as deliberate.

// width code -> width in bytes: 0 -> 1, 1 -> 2, 2 -> 4, 3 -> 8
#define EFFECTS_V3_WIDTH_BYTES(code) (1 << (code))

// segment header bit fields
//
//   bits 0-1  kind
//   bits 2-3  value width code
//   bits 4-5  count width code
//   bit  6    descending
//   bit  7    reserved - MUST be rejected
//
// bit 6 became 'descending' at 08dca4a1e, after the first cut of the fixture
// corpus. It is covered now, by encoder-generated fixtures rather than by
// hand-patched headers, and the corpus confirms it two ways:
//
//   * dir_ascending / dir_descending - 19 bytes each, differing in exactly two:
//     the header (0x00 / 0x40) and the base (200 / 207, lowest vs highest).
//     A decoder that ignored bit 6 would read the descending file as the
//     ascending one and return the ids REVERSED rather than malformed - then
//     re-encode what it thought it read, so a round trip would still pass.
//     Only comparing the two files catches that.
//   * dir_descending_bitmap / collapse_above - a roaring blob is a SET and has
//     no direction, so the two carry byte-identical 66-byte blobs and bit 6 is
//     the only thing that distinguishes them.
#define EFFECTS_V3_SEG_KIND_MASK    0x03  // bits 0-1
#define EFFECTS_V3_SEG_VWIDTH_SHIFT 2     // bits 2-3
#define EFFECTS_V3_SEG_CWIDTH_SHIFT 4     // bits 4-5
#define EFFECTS_V3_SEG_DESCENDING   0x40  // bit 6
#define EFFECTS_V3_SEG_RESERVED     0x80  // bit 7 - MUST be zero

typedef enum {
	EFFECTS_V3_SEG_RANGE     = 0,  // base, len  - steps by one
	EFFECTS_V3_SEG_ASCENDING = 1,  // roaring64 blob - several ranges collapsed
	EFFECTS_V3_SEG_REPEAT    = 2,  // id, count  - does not move
} EffectsV3SegmentKind;

// one segment of an IdList
//
// the observed width codes are RETAINED rather than recomputed on re-encode.
// The spec has the encoder pick the narrowest width that holds each value, so
// recomputing would usually agree - but "usually" is not byte-identical, and a
// round trip that recomputes is testing our own arithmetic rather than the
// peer's bytes. A record built fresh (not decoded) must set these to the
// narrowest holding width; EFFECTS_V3_SEG_ASCENDING ignores both.
typedef struct {
	EffectsV3SegmentKind kind;
	uint8_t value_width;  // header bits 2-3: Range base, Repeat id
	uint8_t count_width;  // header bits 4-5: Range len,  Repeat count

	// header bit 6: the same payload read the other way
	//
	// a descending Range's 'base' is its FIRST id and therefore its HIGHEST:
	// it describes base, base-1, ... base-(len-1). An Ascending segment's blob
	// is a set, so the ascending and descending forms of the same ids differ in
	// exactly this bit and nowhere else.
	//
	// A REPEAT HAS NO DIRECTION. It holds one id 'count' times, so both
	// readings are the same sequence and the bit carries no information - which
	// means a peer that set it meant something this build does not know. The
	// decoder REJECTS it there rather than ignoring it.
	bool descending;
	union {
		struct {
			uint64_t base;  // first id
			uint64_t len;   // how many, ascending by one
		} range;
		struct {
			uint64_t id;     // the id, held once
			uint64_t count;  // how many times it repeats
		} repeat;
		struct {
			unsigned char *blob;   // serialized roaring64, owned
			uint32_t       n;      // blob length, as the u32 on the wire
			uint64_t       cardinality;  // ids described; checked at decode
		} ascending;
	};
} EffectsV3Segment;

// an ordered, duplicate-preserving list of entity ids, held AS SEGMENTS
//
// never expanded at decode time: one valid segment describes four billion ids
// in seven bytes. EffectsV3_Apply expands, where the graph is in scope.
typedef struct {
	EffectsV3Segment *segments;  // owned, 'n' entries
	uint32_t          n;         // segment count, as the u32 on the wire
} EffectsV3IdList;

// a single decoded record
//
// 'opcode' selects which of the remaining fields carry meaning - see the table
// above. A field a record does not use is zeroed: NULL pointers, zero counts.
typedef struct {
	EffectType opcode;
	uint32_t   count;  // entities in this record; 0 for records 9 and 10

	// the shape, hoisted once per record
	LabelID    *labels;       // node-shaped records, owned
	uint16_t    n_labels;     // as the u16 on the wire - n, NOT count
	RelationID  relation_id;  // edge-shaped records

	// the AttrSet header - ids stated once, values row-major below
	AttributeID *attr_ids;  // owned
	uint16_t     n_attrs;   // as the u16 on the wire

	// the rows
	EffectsV3IdList ids;  // positionally bound to 'values'
	EffectsV3IdList src;  // CREATE_EDGE / DELETE_EDGE only
	EffectsV3IdList dst;  // CREATE_EDGE / DELETE_EDGE only

	// count * n_attrs values, row-major: row k is values[k * n_attrs ..]
	// and belongs to the k-th id in 'ids' AS WRITTEN
	//
	// T_NULL in a slot means REMOVE THIS ATTRIBUTE - it is not padding and must
	// not be filtered out, or every property removal becomes a no-op
	SIValue *values;    // owned; each freed with SIValue_Free
	uint64_t n_values;  // count * n_attrs

	// records 9 and 10 only
	SchemaType  schema_type;  // ADD_SCHEMA
	int         schema_id;    // ADD_SCHEMA - LabelID or RelationID
	AttributeID attr_id;      // ADD_ATTRIBUTE
	char       *name;         // owned, NUL terminated
} EffectsV3Record;

// a decoded payload: the header, then the records in apply order
//
// defined here rather than left opaque because the writer encodes from it and
// the reader decodes into it - the definition IS the contract between them
struct EffectsV3Records {
	uint8_t version;  // the version byte the payload declared
	uint8_t flags;    // the flags byte; bit 0 = compressed

	EffectsV3Record *records;  // owned, 'n' entries, in apply order
	uint32_t         n;        // records decoded
};

// opaque to everything except the reader (which defines it above) and the
// writer (which consumes it)
typedef struct EffectsV3Records EffectsV3Records;

// why a buffer was refused
//
// separated rather than collapsed to a bool so a test can pin WHICH check
// fired - notably that a segment header setting the reserved bit is rejected
// as malformed rather than silently masked off
//
// UNIMPLEMENTED is deliberately distinct from MALFORMED. A record this build
// has not implemented yet is not a corrupt one, and reporting it as corrupt
// sends an operator hunting a wire problem that does not exist. It is still a
// refusal - the buffer is rejected and the caller treats it as divergence,
// because silently skipping a record it cannot apply is data loss. It exists
// so the log names the real cause while records 11-14 are outstanding, and it
// becomes unreachable once they land.
typedef enum {
	EFFECTS_V3_OK = 0,             // decoded cleanly
	EFFECTS_V3_TRUNCATED,          // ran out of bytes mid-field
	EFFECTS_V3_MALFORMED,          // well-sized but invalid: reserved bits set,
	                               // unknown opcode, bad width code, a count
	                               // that outruns the payload, a cardinality
	                               // that disagrees with the record
	EFFECTS_V3_UNSUPPORTED_VERSION,// version byte above what this build reads
	EFFECTS_V3_UNSUPPORTED_FLAGS,  // a flag bit outside the mask we understand,
	                               // e.g. compression before zstd is vendored
	EFFECTS_V3_UNIMPLEMENTED,      // a well-formed record this build does not
	                               // implement yet
} EffectsV3Status;

// human readable form of a status, for logs and test failures
const char *EffectsV3Status_ToString
(
	EffectsV3Status status
);

//------------------------------------------------------------------------------
// the seam
//------------------------------------------------------------------------------

// decode a v3 payload into records
//
// takes NO GraphContext: this is a pure byte-to-value transformation, and that
// is what makes it testable against a fixture corpus and cheap to fuzz
//
// on EFFECTS_V3_OK the caller owns '*records' and must free it with
// EffectsV3_RecordsFree; on anything else '*records' is set to NULL and
// nothing is left allocated
EffectsV3Status EffectsV3_Decode
(
	const char *buff,             // encoded payload, including the version byte
	size_t n,                     // size of buff
	EffectsV3Records **records    // [output] decoded records
);

// encode records back into a payload
//
// takes NO GraphContext, for the same reason. Re-encoding what
// EffectsV3_Decode produced MUST reproduce the input bytes exactly
//
// returns false only on an internal failure; records that decoded cleanly
// always re-encode
bool EffectsV3_Encode
(
	const EffectsV3Records *records,  // records to encode
	EffectsBuffer *eb                 // buffer to write into
);

// apply decoded records to a graph
//
// this is where ids are expanded and where divergence is detected, so it is
// where a GraphContext is required. Returns false if the records reference
// graph state that does not exist locally - the caller treats that as
// divergence, exactly as the v2 path does
bool EffectsV3_Apply
(
	GraphContext *gc,                 // graph to operate on
	const EffectsV3Records *records   // records to apply
);

// free records returned by EffectsV3_Decode
void EffectsV3_RecordsFree
(
	EffectsV3Records *records
);

//------------------------------------------------------------------------------
// readiness - what is actually linkable
//------------------------------------------------------------------------------
//
// The entry points above are declared before they are defined, so a test that
// links one that does not exist yet breaks the build for everyone. These flags
// say what is safe to call.
//
// Deliberately TWO flags with one owner each, rather than one shared flag.
// Decode and free are the reader's; encode is the writer's. A single
// EFFECTS_V3_CODEC_READY could only be defined honestly by whichever of the two
// landed second, and defining it on decode alone would link a round-trip test
// against an undefined EffectsV3_Encode - the exact failure the flag exists to
// prevent. Same conflation EFFECTS_VERSION had before it was split into a read
// ceiling and an emit version.
//
// A truncation corpus needs decode alone, so it can run a PR earlier than a
// round trip.

// EffectsV3_ENCODE_READY is defined by the writer when EffectsV3_Encode lands

#if defined(EFFECTS_V3_DECODE_READY) && defined(EFFECTS_V3_ENCODE_READY)
// both directions are linkable, so a round trip can be built
#define EFFECTS_V3_CODEC_READY 1
#endif
