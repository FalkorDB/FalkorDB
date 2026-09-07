/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "effects.h"

#include <stdio.h>
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

// opaque to everything except the reader (which defines it) and the writer
// (which consumes it)
typedef struct EffectsV3Records EffectsV3Records;

// why a buffer was refused
//
// separated rather than collapsed to a bool so a test can pin WHICH check
// fired - notably that a segment header with its reserved bits 6-7 set is
// rejected as malformed rather than silently masked off
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
