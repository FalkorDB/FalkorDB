/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// the collapse rule's arithmetic, ported from Run/RunCost in
// graph/src/effects/v3/id_list.rs, which the spec names as the source
//
//     collapse when  range_bytes >= 32  and  5 + bitmap_bytes < range_bytes
//
// Evaluated on every new segment against the run's own shape: weighed at an
// arbitrary point instead, the bytes would depend on when the encoder looked
// rather than on the ids. A decline retires nothing - crossing a bucket
// boundary steps the bitmap up, so a run that loses now can win later.
//
// bitmap_bytes is closed form, never a trial build. The constants are roaring's
// serialized layout rather than a public API; drift costs a worse segmentation,
// never a wrong one, and the shape table in tests/unit/ fails if it happens.

// the run currently being built, and what it would cost encoded either way
//
// exposed rather than opaque so the cost table can assert the arithmetic
// against the sizes the Rust formula predicts
typedef struct {
	// what the run's CLOSED segments cost as ranges. The open one can still
	// grow, so charging it would make the answer depend on when it was asked
	size_t range_bytes;

	// distinguishes an empty run from one whose open bucket starts at zero
	bool started;

	size_t   closed_bitmaps;  // 2^32 slices the run has moved past
	uint64_t bitmap_key;      // the slice being filled: id >> 32

	// finished buckets in it; closed_has_run selects the header flavour
	size_t closed_buckets;
	size_t closed_body;
	bool   closed_has_run;

	// the bucket still being filled - separate because an extending range can
	// still change it, and whether IT chooses a run store can flip as it grows
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
// split at each 65,536 boundary it crosses: a container covers a fixed slice of
// the id SPACE, so a range spanning three buckets is three runs, not one
void EffectsV3Run_AddRange
(
	EffectsV3Run *run,  // run to fold into
	uint64_t base,      // lowest id in the range
	uint64_t len        // number of consecutive ids; must be > 0
);

// charge a segment that has stopped growing to the range side of the tally
//
// ONLY WHEN A SEGMENT IS SUPERSEDED, never for the one being extended. This
// module has no notion of segments and cannot enforce it, so breaking the
// contract shifts the threshold by one segment and stays self-consistent -
// invisible without collapseThreshold in test_effects_v3_run_cost.c
void EffectsV3Run_AddRangeBytes
(
	EffectsV3Run *run,  // run to charge
	size_t n            // encoded length of the closed segment
);

// the bytes roaring would report for this run's ids, in closed form
//
// measures the id SET, so it is direction-independent
size_t EffectsV3Run_BitmapBytes
(
	const EffectsV3Run *run  // run to size
);

// whether a bitmap has already won, so one is worth building
bool EffectsV3Run_PrefersBitmap
(
	const EffectsV3Run *run  // run to weigh
);
