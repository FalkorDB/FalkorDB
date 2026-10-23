/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "effects_v3.h"
#include "../util/roaring.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

// builds an IdList's segments as the ids arrive
//
// An IdList is an ordered, duplicate-preserving list of entity ids, written as a
// count of segments followed by the segments. There is no plain form and no
// dictionary - the segments ARE the encoding, decided as the ids are pushed
// rather than rediscovered later, so a bulk create is one Range from the first
// push to the last and never allocates beyond it.
//
// The segmentation is NORMATIVE: two engines must reach the same segments for
// the same ids, or the same write produces different bytes. So every decision
// here is fixed by the format rather than chosen for this implementation:
//
//   * the collapse rule is evaluated on every new segment against the run's own
//     shape, using closed-form arithmetic (effects_v3_run_cost.h)
//   * a bitmap is built with one add_range_closed per contributing range, never
//     id by id - the same set serializes to different bytes depending on how it
//     was built, so the construction path is part of the format
//   * run_optimize is called once, unconditionally. On that path it is a
//     NORMALIZATION rather than a size win, and skipping it produces identical
//     bytes on most sets and different bytes on a few - the worst available
//     failure, since it passes casual testing and diverges a replica on a
//     minority of writes
//   * a Repeat never carries a direction. One id however many times reads the
//     same both ways, so a decoder rejects the bit rather than ignoring it

// the kind enum and the header-bit macros come from effects_v3.h, the shared
// contract: a segment kind that differs between the encoder and the decoder is
// precisely the disagreement the corpus exists to catch.

// one segment under construction
//
// A SEPARATE TYPE from the contract's EffectsV3IdListSegment (effects_v3.h),
// which is a DECODED record, deliberately: collapsing the two forces one of two
// regressions. Either the decoded form carries a live roaring bitmap, which
// means deserialising every segment at decode - and the contract's invariant is
// that an IdList is never expanded there, one valid segment being four billion
// ids in seven bytes - or this form serialises on every push, which is the trial
// build the closed-form cost model exists to avoid.
//
// Hence the fields a decoded segment has no use for: BOTH EXTREMES ARE CACHED on
// a bitmap segment, because the builder routes the next push on them. Deriving
// one assumes the ids are gapless, which is exactly what a bitmap segment is not
// - the derived extreme is then wrong by however much the gaps total, misrouting
// the next push and, if it reaches a range insertion, fabricating ids the set
// does not hold.
typedef struct {
	// direction is in the kind, not a flag beside it: the contract's closed set
	// makes a descending Repeat unrepresentable rather than merely forbidden by
	// an assert
	EffectsV3IdListSegmentKind kind;
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
// repeats, and a scan walking nodes downward writes a strictly descending column
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

// convert what the builder holds into the shared wire representation
//
// Where a freshly built list acquires its width codes: each value is given the
// narrowest width that holds it, and each bitmap is serialized to a blob. A
// DECODED list already carries the widths its peer chose, and those are
// preserved rather than recomputed - a peer may legitimately write a value wider
// than it needs, and a round trip that narrows it produces different bytes from
// the ones it read. The two paths meet at the same struct, so the encoder never
// has to know which one it is writing.
//
// the caller owns the result and must free it with
// EffectsV3IdListBuilder_FreeIdList
EffectsV3IdList EffectsV3IdListBuilder_ToIdList
(
	const EffectsV3IdListBuilder *b  // builder
);

// free an IdList produced by EffectsV3IdListBuilder_ToIdList
void EffectsV3IdListBuilder_FreeIdList
(
	EffectsV3IdList *l  // list to free
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
