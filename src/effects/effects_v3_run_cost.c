/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "effects_v3_run_cost.h"

#include <string.h>

//------------------------------------------------------------------------------
// roaring's serialized layout, as the pieces this arithmetic needs
//------------------------------------------------------------------------------
//
// Layout: <magic> [<container count>] <per-container descriptor> x n
//         [<per-container offset> x n] <container bodies>
// see https://github.com/RoaringBitmap/RoaringFormatSpec
//
// A BUCKET IS NOT A RANGE. It is a fixed 65,536-wide slice of the id SPACE - a
// partition of the addresses, not of the data. Roaring splits ids by their high
// bits and gives each bucket the cheapest of three stores for whatever landed
// there: an array at 2 bytes per id, a fixed 8,192-byte bitset, or runs at 4
// bytes per interval. So a "run" is one of the three encodings INSIDE a bucket,
// and that is where our segments land - one Range is one run, except where it
// straddles a bucket boundary, which makes it a run in each bucket it touches.
//
// Ids are 64-bit, so there are two levels: a treemap is keyed by id >> 32, each
// value a bitmap holding the buckets of that slice. Each bitmap is charged one
// header, so buckets are grouped by bucket >> 16 to size them.

// an array container: one u16 per value
#define ARRAY_ELEMENT_BYTES 2

// a bitset container: a fixed 1024 x u64, whatever it holds
#define BITSET_CONTAINER_BYTES 8192

// a run container: a u16 count of intervals, then each interval
#define RUN_COUNT_BYTES 2

// one interval: u16 start and u16 length
#define RUN_INTERVAL_BYTES 4

// the leading magic number, which selects the container layout that follows
// (roaring and the format spec call it the cookie: 12346 introduces the layout
// without run containers, 12347 the one with them)
#define MAGIC_BYTES 4

// the u32 container count, present only in the no-run-container layout - the
// other packs it into the magic number's high bits
#define CONTAINER_COUNT_BYTES 4

// one container's descriptor: its u16 key and its u16 cardinality
#define CONTAINER_DESC_BYTES 4

// one container's u32 offset into the body
#define CONTAINER_OFFSET_BYTES 4

// below this many containers the run-container layout omits the offset table
#define OFFSET_TABLE_MIN_CONTAINERS 4

// a treemap's own prefix: a u64 count of the bitmaps in it
#define TREEMAP_COUNT_BYTES 8

// each bitmap in a treemap is preceded by its u32 key
#define BITMAP_KEY_BYTES 4

// an Ascending segment's own overhead on top of the blob: its header byte and
// the u32 length prefix
#define ASCENDING_SEGMENT_OVERHEAD (1 + 4)

// the floor below which there is nothing to weigh
//
// the smallest treemap roaring can serialize is 27 bytes - a run container, the
// cheapest of the three - so a run costing less than that cannot lose to one.
// 32 rather than 27 because the arithmetic is only worth running once a bitmap
// is plausibly competitive, not the instant it becomes possible
//
// (30 is the ARRAY container's floor. Both were measured on this tree's
// CRoaring while checking parity; the constant is unchanged, the reason the
// spec gave for it was wrong)
#define ROARING_FLOOR_BYTES 32

// how many ids share one bucket
#define BUCKET_WIDTH_BITS 16

// what one bucket's body costs, and whether it chose a run store
//
// array up to 4,096 ids and bitset past it - which is exactly a min, because
// two bytes times 4,096 is the bitset's fixed size. A TIE GOES TO THE RUN
// STORE: roaring's optimize leaves a run container alone unless strictly
// beaten, and ours are built by range so they start as runs
static size_t _bucket_body
(
	uint64_t ids,   // ids that landed in the bucket
	uint32_t runs,  // maximal runs they form
	bool *is_run    // [output] whether a run store was chosen
) {
	size_t sparse = (size_t)ids * ARRAY_ELEMENT_BYTES;
	size_t plain  = (sparse < BITSET_CONTAINER_BYTES)
		? sparse
		: BITSET_CONTAINER_BYTES;
	size_t as_run = RUN_COUNT_BYTES + RUN_INTERVAL_BYTES * (size_t)runs;

	if(as_run <= plain) {
		*is_run = true;
		return as_run;
	}

	*is_run = false;
	return plain;
}

// one bitmap's header, given how many buckets it holds and whether any of them
// is a run container
static size_t _bitmap_header
(
	size_t buckets,  // buckets in the bitmap
	bool has_run     // whether any of them chose a run store
) {
	if(has_run) {
		// the run-container layout adds a bitset marking which buckets are
		// runs, and only carries the offset table once there are enough
		size_t run_flags = (buckets + 7) / 8;

		if(buckets >= OFFSET_TABLE_MIN_CONTAINERS) {
			return MAGIC_BYTES
				+ (CONTAINER_DESC_BYTES + CONTAINER_OFFSET_BYTES) * buckets
				+ run_flags;
		}

		return MAGIC_BYTES + CONTAINER_DESC_BYTES * buckets + run_flags;
	}

	return MAGIC_BYTES
		+ CONTAINER_COUNT_BYTES
		+ (CONTAINER_DESC_BYTES + CONTAINER_OFFSET_BYTES) * buckets;
}

// fold the open bucket into its bitmap's frozen totals
static void _freeze_bucket
(
	EffectsV3Run *run  // run to fold within
) {
	bool is_run = false;
	size_t body = _bucket_body(run->bucket_ids, run->bucket_runs, &is_run);

	run->closed_body     += body;
	run->closed_has_run  |= is_run;
	run->closed_buckets  += 1;
}

// fold the open bitmap into the frozen total
static void _freeze_bitmap
(
	EffectsV3Run *run  // run to fold within
) {
	run->closed_bitmaps += BITMAP_KEY_BYTES
		+ _bitmap_header(run->closed_buckets, run->closed_has_run)
		+ run->closed_body;

	run->closed_buckets = 0;
	run->closed_body    = 0;
	run->closed_has_run = false;
}

void EffectsV3Run_Restart
(
	EffectsV3Run *run  // run to reset
) {
	memset(run, 0, sizeof(*run));
}

void EffectsV3Run_AddRangeBytes
(
	EffectsV3Run *run,  // run to charge
	size_t n            // encoded length of the closed segment
) {
	run->range_bytes += n;
}

void EffectsV3Run_AddRange
(
	EffectsV3Run *run,  // run to fold into
	uint64_t base,      // lowest id in the range
	uint64_t len        // number of consecutive ids; must be > 0
) {
	if(len == 0) {
		return;
	}

	uint64_t end = base + len - 1;
	uint64_t lo  = base;

	while(true) {
		uint64_t bucket = lo >> BUCKET_WIDTH_BITS;

		// the last id this bucket can hold. Computed without overflowing on
		// the topmost bucket, where (bucket + 1) << 16 would wrap to zero
		uint64_t bucket_end = (bucket == (UINT64_MAX >> BUCKET_WIDTH_BITS))
			? UINT64_MAX
			: (((bucket + 1) << BUCKET_WIDTH_BITS) - 1);

		uint64_t piece_end = (end < bucket_end) ? end : bucket_end;
		uint64_t ids       = piece_end - lo + 1;

		if(!run->started) {
			run->started     = true;
			run->bitmap_key  = bucket >> BUCKET_WIDTH_BITS;
			run->bucket      = bucket;
			run->bucket_ids  = ids;
			run->bucket_runs = 1;
		} else if(bucket == run->bucket) {
			// still the same bucket: one more run in it
			run->bucket_ids  += ids;
			run->bucket_runs += 1;
		} else {
			// moved on, so the open bucket is now frozen and can be folded into
			// its bitmap's totals - and if the bitmap changed too, that bitmap
			// is frozen as well
			_freeze_bucket(run);

			uint64_t bitmap_key = bucket >> BUCKET_WIDTH_BITS;
			if(bitmap_key != run->bitmap_key) {
				_freeze_bitmap(run);
				run->bitmap_key = bitmap_key;
			}

			run->bucket      = bucket;
			run->bucket_ids  = ids;
			run->bucket_runs = 1;
		}

		// guard the increment as well: piece_end == UINT64_MAX would wrap
		if(piece_end >= end) {
			break;
		}
		lo = piece_end + 1;
	}
}

size_t EffectsV3Run_BitmapBytes
(
	const EffectsV3Run *run  // run to size
) {
	if(!run->started) {
		return TREEMAP_COUNT_BYTES;
	}

	// close out the open bucket and the open bitmap arithmetically, without
	// committing either
	bool is_run = false;
	size_t body = _bucket_body(run->bucket_ids, run->bucket_runs, &is_run);

	return TREEMAP_COUNT_BYTES
		+ run->closed_bitmaps
		+ BITMAP_KEY_BYTES
		+ _bitmap_header(run->closed_buckets + 1,
				run->closed_has_run || is_run)
		+ run->closed_body
		+ body;
}

bool EffectsV3Run_PrefersBitmap
(
	const EffectsV3Run *run  // run to weigh
) {
	// below roaring's own floor there is nothing to weigh
	return run->range_bytes >= ROARING_FLOOR_BYTES
		&& ASCENDING_SEGMENT_OVERHEAD + EffectsV3Run_BitmapBytes(run)
			< run->range_bytes;
}
