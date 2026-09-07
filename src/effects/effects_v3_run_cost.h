/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// the collapse rule's arithmetic - ported from Run/RunCost in
// graph/src/effects/v3/id_list.rs, which the format spec names as the source
//
// An IdList encoder tracks the ascending (or descending) run of segments it is
// currently building and collapses it into a single Ascending segment when a
// roaring bitmap would be strictly cheaper than the ranges it replaces:
//
//     collapse when  range_bytes >= 32  and  5 + bitmap_bytes < range_bytes
//
// Three properties of that rule matter more than its constants, and all three
// are why this file exists rather than a call to a size function:
//
//   * IT IS EVALUATED ON EVERY NEW SEGMENT, against the run's own shape - never
//     as a counter over the whole list, and never as a pass at encode time.
//     Twenty ascending segments weighed at sixteen and the same twenty weighed
//     together can disagree, so a comparison made at an arbitrary point makes
//     the bytes depend on when the encoder chose to look rather than on the ids.
//
//   * bitmap_bytes IS ARITHMETIC, NOT A TRIAL BUILD. Roaring's serialized size
//     is a closed-form function of how many ids there are, how many maximal runs
//     they form, and how they spread across 65,536-wide buckets - and a segment
//     list already knows all three. Nothing is ever built in order to be
//     measured and discarded; a bitmap is constructed once, after it has already
//     won. roaring64_bitmap_portable_size_in_bytes cannot answer this, because
//     it needs the bitmap that is being avoided.
//
//   * A DECLINE RETIRES NOTHING. The comparison is not monotone - crossing a
//     bucket boundary adds a container and steps the bitmap's size up - so a run
//     that loses now can win later and is re-weighed on the next segment.
//
// Every operation here is O(1). A run's ids advance in one direction, so an
// added range can only extend the newest bucket or open one after it, which
// means every bucket behind the cursor has its cost frozen for good and
// collapses into an accumulator. Only the open bucket and the open bitmap keep
// their parts.
//
// The constants are roaring's serialized layout, not a public API. That is safe
// to depend on for two reasons: the bytes actually written come from
// roaring64_bitmap_portable_serialize whatever they are, so a library drift
// makes the encoder pick a marginally worse segmentation rather than a wrong
// one - and both engines still agree, because agreement comes from running the
// same arithmetic, not from matching the library. And the twelve-shape table in
// tests/unit/ fails the moment they diverge, which is what a version bump
// should do rather than drift quietly.

// the run currently being built, and what it would cost encoded either way
//
// exposed rather than opaque so the cost table can assert the arithmetic
// directly against the sizes the Rust formula predicts
typedef struct {
	// what the run's segments cost as ranges - the comparison's other side
	//
	// counts only CLOSED segments: the open one can still grow, and charging it
	// would make the answer depend on where in the sequence it was asked
	size_t range_bytes;

	// whether any id has been folded in; distinguishes an empty run from one
	// whose open bucket happens to start at zero
	bool started;

	// bytes of every bitmap that is finished - the run has moved past its 2^32
	// slice, so nothing can change it again
	size_t closed_bitmaps;

	// the 2^32 slice being filled: id >> 32
	uint64_t bitmap_key;

	// buckets in it that are finished, their total body bytes, and whether any
	// chose a run store - which is what selects the header flavour
	size_t closed_buckets;
	size_t closed_body;
	bool   closed_has_run;

	// the bucket still being filled: id >> 16, and what has landed in it
	//
	// kept apart from the frozen totals because it is the one thing an
	// extending range can still change, and because whether IT chooses a run
	// store can flip either way as it grows
	uint64_t bucket;
	uint64_t bucket_ids;
	uint32_t bucket_runs;
} EffectsV3Run;

// discard everything tallied and begin a new run
void EffectsV3Run_Restart
(
	EffectsV3Run *run  // run to reset
);

// fold one consecutive range of ids into the bitmap side of the tally
//
// split at each 65,536 bucket boundary it crosses, because a container covers a
// fixed slice of the id SPACE rather than a stretch of data: a range spanning
// three buckets is three runs, one in each, not one run. That, and only that,
// is why this splits
void EffectsV3Run_AddRange
(
	EffectsV3Run *run,  // run to fold into
	uint64_t base,      // lowest id in the range
	uint64_t len        // number of consecutive ids; must be > 0
);

// charge a segment that has stopped growing to the range side of the tally
void EffectsV3Run_AddRangeBytes
(
	EffectsV3Run *run,  // run to charge
	size_t n            // encoded length of the closed segment
);

// the bytes roaring would report for this run's ids, in closed form
//
// the id SET is what is measured, which is the same whichever direction the run
// runs, so this is direction-independent
size_t EffectsV3Run_BitmapBytes
(
	const EffectsV3Run *run  // run to size
);

// whether a bitmap has already won, so one is worth building
//
// both sides are exact; nothing speculative is constructed to answer it
bool EffectsV3Run_PrefersBitmap
(
	const EffectsV3Run *run  // run to weigh
);
