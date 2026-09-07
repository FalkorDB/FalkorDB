/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../util/roaring.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

// builds an IdList's segments as the ids arrive
//
// An IdList is an ordered, duplicate-preserving list of entity ids, written as
// a count of segments followed by the segments. There is no plain form, no
// dictionary, and no discriminator for the list as a whole - the segments ARE
// the encoding, decided as the ids are pushed rather than rediscovered later.
// So a bulk create or a delete-by-label is one Range from the first push to
// the last and never allocates beyond it.
//
// The segmentation is NORMATIVE. Two engines must reach the same segments for
// the same ids, or the same write produces different bytes, so every decision
// here is fixed by the format rather than chosen for this implementation:
//
//   * the collapse rule is evaluated on every new segment against the run's own
//     shape, using closed-form arithmetic (effects_v3_run_cost.h)
//   * a bitmap is built with one add_range_closed per contributing range, never
//     id by id - the same set serializes to different bytes depending on how it
//     was built, so construction path is part of the format
//   * run_optimize is called once, unconditionally. On the mandated
//     construction path it is a NORMALIZATION rather than a size win: it leaves
//     most shapes untouched and can make one larger. Skipping it produces
//     identical bytes on most sets and different bytes on a few, which is the
//     worst available failure - it passes casual testing and diverges a replica
//     on a minority of writes
//   * a Repeat never carries a direction. One id however many times reads the
//     same both ways, so the bit would carry no information and a decoder
//     rejects it there rather than ignoring it

// which of the three wire kinds a segment is
//
// direction is a separate flag rather than more kinds, because a descending
// segment is the same payload read the other way: a Range gains a
// first-and-highest base instead of a first-and-lowest one, and a bitmap holds
// a set, which has no direction at all
typedef enum {
	EFFECTS_V3_SEG_RANGE     = 0,  // consecutive ids, stepping by one
	EFFECTS_V3_SEG_BITMAP    = 1,  // ids with gaps, as a roaring bitmap
	EFFECTS_V3_SEG_REPEAT    = 2,  // one id, several times
} EffectsV3SegKind;

// one segment under construction
//
// BOTH EXTREMES ARE CACHED on a bitmap segment rather than one being derived
// from the other and the length. Deriving assumes the ids are gapless, which is
// exactly what a bitmap segment is not - the derived extreme is then wrong by
// however much the gaps total, and a wrong extreme both misroutes the next
// push and, if it ever reaches a range insertion, fabricates ids the set does
// not hold
typedef struct {
	EffectsV3SegKind kind;
	bool descending;  // header bit 6; always false for a Repeat
	union {
		struct {
			uint64_t base;  // first id: the lowest ascending, the highest descending
			uint32_t len;   // how many, never 0
		} range;
		struct {
			uint64_t id;     // the id, held once
			uint32_t count;  // how many times it repeats
		} repeat;
		struct {
			roaring64_bitmap_t *bitmap;  // owned
			uint32_t len;                // ids described
			uint64_t min;                // exact, not derived
			uint64_t max;                // exact, not derived
		} bitmap;
	};
} EffectsV3Seg;

// the builder
typedef struct EffectsV3IdListBuilder EffectsV3IdListBuilder;

// create a builder
EffectsV3IdListBuilder *EffectsV3IdListBuilder_New(void);

// append one id
//
// ids arrive in whatever order the write produced them: duplicates and steps
// backwards are the ordinary case, not an error. Edge endpoints are nothing but
// repeats, and a scan walking nodes downward writes a strictly descending
// column
void EffectsV3IdListBuilder_Push
(
	EffectsV3IdListBuilder *b,  // builder
	uint64_t id                 // id to append
);

// how many ids have been pushed
uint64_t EffectsV3IdListBuilder_Len
(
	const EffectsV3IdListBuilder *b  // builder
);

// how many segments they became
uint32_t EffectsV3IdListBuilder_SegmentCount
(
	const EffectsV3IdListBuilder *b  // builder
);

// borrow a segment for inspection; valid until the next push
const EffectsV3Seg *EffectsV3IdListBuilder_Segment
(
	const EffectsV3IdListBuilder *b,  // builder
	uint32_t i                        // segment index
);

// free the builder and every bitmap it owns
void EffectsV3IdListBuilder_Free
(
	EffectsV3IdListBuilder *b  // builder
);

//------------------------------------------------------------------------------
// segment geometry - shared with the encoder
//------------------------------------------------------------------------------

// the lowest id a segment carries
uint64_t EffectsV3Seg_Min
(
	const EffectsV3Seg *s  // segment
);

// the highest id a segment carries
uint64_t EffectsV3Seg_Max
(
	const EffectsV3Seg *s  // segment
);

// how many ids a segment carries
uint32_t EffectsV3Seg_Len
(
	const EffectsV3Seg *s  // segment
);

// bytes a segment costs on the wire, without writing it
//
// exact rather than an estimate, so the collapse rule compares like with like.
// A descending Range is charged from its actual base - the run's HIGHEST id -
// which can need a wider value field than the ascending form over the same
// ids, so charging the set's minimum would weigh the wrong encoding
size_t EffectsV3Seg_EncodedLen
(
	const EffectsV3Seg *s  // segment
);

// the narrowest of 1, 2, 4, 8 bytes that holds v
uint8_t EffectsV3_WidthFor
(
	uint64_t v  // value to size
);

// the wire's width code, 0..3, for a width of 1, 2, 4 or 8 bytes
uint8_t EffectsV3_WidthCode
(
	uint8_t width  // width in bytes
);
