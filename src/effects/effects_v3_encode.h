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
