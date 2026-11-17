/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "src/effects/effects_v3_id_cursor.h"
#include "src/util/rmalloc.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>

void setup() {
	Alloc_Reset();
}
#define TEST_INIT setup();
#include "acutest.h"

//------------------------------------------------------------------------------
// helpers
//------------------------------------------------------------------------------

// collect a whole walk, so a test states the SEQUENCE rather than poking at
// one step of it
static uint32_t _WalkAscending
(
	EffectsV3IdCursor *cur,
	uint64_t *out,
	uint32_t cap
) {
	uint32_t n = 0;
	uint64_t id;

	cur->base.reset(&cur->base);
	while(n < cap && cur->base.next(&cur->base, &id)) out[n++] = id;
	return n;
}

static uint32_t _WalkWire
(
	EffectsV3IdCursor *cur,
	uint64_t *out,
	uint32_t cap
) {
	uint32_t n = 0;
	uint64_t id;

	EffectsV3IdCursor_RewindWire(cur);
	while(n < cap && EffectsV3IdCursor_NextWire(cur, &id)) out[n++] = id;
	return n;
}

static bool _Same(const uint64_t *a, const uint64_t *b, uint32_t n) {
	for(uint32_t i = 0; i < n; i++) if(a[i] != b[i]) return false;
	return true;
}

//------------------------------------------------------------------------------
// tests
//------------------------------------------------------------------------------

void test_cursor_sorts_segments_the_wire_left_unsorted() {
	// wire order is 10,11,12 then 0,1,2 - the ids a primary wrote in that
	// order. Claiming needs them sorted ACROSS segments, which is the one
	// thing a per-segment walk cannot do.
	EffectsV3IdListSegment segs[2] = {
		{ .kind = EFFECTS_V3_SEG_RANGE_ASCENDING,
		  .range_ascending = { .base = 10, .len = 3 } },
		{ .kind = EFFECTS_V3_SEG_RANGE_ASCENDING,
		  .range_ascending = { .base = 0,  .len = 3 } },
	};
	EffectsV3IdList list = { .segments = segs, .n = 2 };

	EffectsV3IdCursor cur;
	TEST_ASSERT(EffectsV3IdCursor_Init(&cur, &list, 6) == true);

	uint64_t got[6];
	const uint64_t want_asc[6]  = {0, 1, 2, 10, 11, 12};
	const uint64_t want_wire[6] = {10, 11, 12, 0, 1, 2};

	TEST_ASSERT(_WalkAscending(&cur, got, 6) == 6);
	TEST_ASSERT(_Same(got, want_asc, 6));

	TEST_ASSERT(_WalkWire(&cur, got, 6) == 6);
	TEST_ASSERT(_Same(got, want_wire, 6));

	EffectsV3IdCursor_Free(&cur);
}

void test_cursor_interleaves_two_segments() {
	// the two runs overlap in range, so a merge is the only thing that
	// produces ascending order - concatenating either way does not
	EffectsV3IdListSegment segs[2] = {
		{ .kind = EFFECTS_V3_SEG_RANGE_ASCENDING,
		  .range_ascending = { .base = 0, .len = 3 } },   // 0,1,2
		{ .kind = EFFECTS_V3_SEG_REPEAT,
		  .repeat = { .id = 1, .count = 1 } },            // 1
	};
	EffectsV3IdList list = { .segments = segs, .n = 2 };

	EffectsV3IdCursor cur;
	TEST_ASSERT(EffectsV3IdCursor_Init(&cur, &list, 4) == true);

	uint64_t got[4];
	const uint64_t want[4] = {0, 1, 1, 2};

	TEST_ASSERT(_WalkAscending(&cur, got, 4) == 4);
	TEST_ASSERT(_Same(got, want, 4));

	EffectsV3IdCursor_Free(&cur);
}

void test_cursor_reads_a_descending_range_from_the_other_end() {
	// a descending range's base is its HIGHEST id. Ascending must start at the
	// far end, not at the base - taking the base would be off by len - 1 and
	// would claim ids the record never named.
	EffectsV3IdListSegment segs[1] = {
		{ .kind = EFFECTS_V3_SEG_RANGE_DESCENDING,
		  .range_descending = { .base = 5, .len = 3 } },  // wire: 5,4,3
	};
	EffectsV3IdList list = { .segments = segs, .n = 1 };

	EffectsV3IdCursor cur;
	TEST_ASSERT(EffectsV3IdCursor_Init(&cur, &list, 3) == true);

	uint64_t got[3];
	const uint64_t want_asc[3]  = {3, 4, 5};
	const uint64_t want_wire[3] = {5, 4, 3};

	TEST_ASSERT(_WalkAscending(&cur, got, 3) == 3);
	TEST_ASSERT(_Same(got, want_asc, 3));

	TEST_ASSERT(_WalkWire(&cur, got, 3) == 3);
	TEST_ASSERT(_Same(got, want_wire, 3));

	EffectsV3IdCursor_Free(&cur);
}

void test_cursor_repeats_an_id_as_many_times_as_stated() {
	// a Repeat is how an endpoint list says "all from one node". The walk
	// yields it every time; refusing the duplicate is the claim's job, and a
	// walk that silently collapsed it would hide the refusal.
	EffectsV3IdListSegment segs[1] = {
		{ .kind = EFFECTS_V3_SEG_REPEAT, .repeat = { .id = 7, .count = 3 } },
	};
	EffectsV3IdList list = { .segments = segs, .n = 1 };

	EffectsV3IdCursor cur;
	TEST_ASSERT(EffectsV3IdCursor_Init(&cur, &list, 3) == true);

	uint64_t got[3];
	const uint64_t want[3] = {7, 7, 7};

	TEST_ASSERT(_WalkAscending(&cur, got, 3) == 3);
	TEST_ASSERT(_Same(got, want, 3));
	TEST_ASSERT(_WalkWire(&cur, got, 3) == 3);
	TEST_ASSERT(_Same(got, want, 3));

	EffectsV3IdCursor_Free(&cur);
}

// build a serialized roaring blob holding 'n' ids
static uint32_t _Blob(const uint64_t *ids, uint32_t n, char *buf, size_t cap) {
	roaring64_bitmap_t *b = roaring64_bitmap_create();
	for(uint32_t i = 0; i < n; i++) roaring64_bitmap_add(b, ids[i]);
	roaring64_bitmap_run_optimize(b);

	const size_t sz = roaring64_bitmap_portable_size_in_bytes(b);
	TEST_ASSERT(sz <= cap);
	const size_t w = roaring64_bitmap_portable_serialize(b, buf);
	roaring64_bitmap_free(b);
	return (uint32_t)w;
}

void test_cursor_walks_a_set_both_ways() {
	const uint64_t members[4] = {3, 9, 40, 41};
	char buf[512];
	const uint32_t n = _Blob(members, 4, buf, sizeof(buf));

	// the SAME blob read two ways: the two Set kinds differ in one header bit
	// and nowhere else
	EffectsV3IdListSegment desc = {
		.kind = EFFECTS_V3_SEG_SET_DESCENDING,
		.set_descending = { .blob = (unsigned char *)buf, .n = n,
							.cardinality = 4 },
	};
	EffectsV3IdList list = { .segments = &desc, .n = 1 };

	EffectsV3IdCursor cur;
	TEST_ASSERT(EffectsV3IdCursor_Init(&cur, &list, 4) == true);

	uint64_t got[4];
	const uint64_t want_asc[4]  = {3, 9, 40, 41};
	const uint64_t want_wire[4] = {41, 40, 9, 3};

	// claiming sorts it regardless of the direction the record wrote it in
	TEST_ASSERT(_WalkAscending(&cur, got, 4) == 4);
	TEST_ASSERT(_Same(got, want_asc, 4));

	// applying reads it the way the record wrote it, because row k binds to
	// the k-th id AS WRITTEN
	TEST_ASSERT(_WalkWire(&cur, got, 4) == 4);
	TEST_ASSERT(_Same(got, want_wire, 4));

	EffectsV3IdCursor_Free(&cur);
}

void test_cursor_refuses_a_set_that_is_not_the_size_decode_recorded() {
	const uint64_t members[3] = {1, 2, 3};
	char buf[512];
	const uint32_t n = _Blob(members, 3, buf, sizeof(buf));

	// the blob holds three ids and the segment says four. The walk bounds
	// itself on that number, so left unchecked it would read an exhausted
	// iterator - and it would do so DURING the apply, after the ids were
	// claimed, where there is nothing left to refuse.
	EffectsV3IdListSegment seg = {
		.kind = EFFECTS_V3_SEG_SET_ASCENDING,
		.set_ascending = { .blob = (unsigned char *)buf, .n = n,
						   .cardinality = 4 },
	};
	EffectsV3IdList list = { .segments = &seg, .n = 1 };

	EffectsV3IdCursor cur;
	TEST_ASSERT(EffectsV3IdCursor_Init(&cur, &list, 4) == false);
	EffectsV3IdCursor_Free(&cur);

	// and the same blob, correctly described, opens - so the refusal above is
	// the cardinality and not the blob
	seg.set_ascending.cardinality = 3;
	TEST_ASSERT(EffectsV3IdCursor_Init(&cur, &list, 3) == true);
	EffectsV3IdCursor_Free(&cur);
}

void test_cursor_refuses_segments_that_do_not_total_the_count() {
	EffectsV3IdListSegment segs[1] = {
		{ .kind = EFFECTS_V3_SEG_RANGE_ASCENDING,
		  .range_ascending = { .base = 0, .len = 3 } },
	};
	EffectsV3IdList list = { .segments = segs, .n = 1 };

	EffectsV3IdCursor cur;

	// three ids described, four claimed. Caught here rather than by the
	// claim's own tally, so the apply never starts.
	TEST_ASSERT(EffectsV3IdCursor_Init(&cur, &list, 4) == false);
	EffectsV3IdCursor_Free(&cur);

	TEST_ASSERT(EffectsV3IdCursor_Init(&cur, &list, 3) == true);
	EffectsV3IdCursor_Free(&cur);
}

void test_cursor_walks_identically_every_time() {
	// the claim decides on the first walk and acts on the second. A cursor
	// that disagreed with itself would claim a different set from the one it
	// was cleared for, so sameness is the contract, not an optimisation.
	const uint64_t members[3] = {2, 60000, 70000};
	char buf[512];
	const uint32_t n = _Blob(members, 3, buf, sizeof(buf));

	EffectsV3IdListSegment segs[2] = {
		{ .kind = EFFECTS_V3_SEG_SET_DESCENDING,
		  .set_descending = { .blob = (unsigned char *)buf, .n = n,
							  .cardinality = 3 } },
		{ .kind = EFFECTS_V3_SEG_RANGE_DESCENDING,
		  .range_descending = { .base = 4, .len = 2 } },
	};
	EffectsV3IdList list = { .segments = segs, .n = 2 };

	EffectsV3IdCursor cur;
	TEST_ASSERT(EffectsV3IdCursor_Init(&cur, &list, 5) == true);

	uint64_t first[5], second[5], third[5];
	TEST_ASSERT(_WalkAscending(&cur, first,  5) == 5);
	TEST_ASSERT(_WalkAscending(&cur, second, 5) == 5);
	TEST_ASSERT(_Same(first, second, 5));

	// and a wire walk in between must not disturb the ascending one, since
	// the two share the same bitmaps
	uint64_t wire[5];
	TEST_ASSERT(_WalkWire(&cur, wire, 5) == 5);
	TEST_ASSERT(_WalkAscending(&cur, third, 5) == 5);
	TEST_ASSERT(_Same(first, third, 5));

	const uint64_t want_asc[5] = {2, 3, 4, 60000, 70000};
	TEST_ASSERT(_Same(first, want_asc, 5));

	EffectsV3IdCursor_Free(&cur);
}

void test_cursor_steps_over_empty_segments() {
	// decode only bounds a range when its length is non-zero, so a
	// zero-length segment is representable and simply holds nothing
	EffectsV3IdListSegment segs[3] = {
		{ .kind = EFFECTS_V3_SEG_RANGE_ASCENDING,
		  .range_ascending = { .base = 100, .len = 0 } },
		{ .kind = EFFECTS_V3_SEG_RANGE_ASCENDING,
		  .range_ascending = { .base = 1, .len = 2 } },
		{ .kind = EFFECTS_V3_SEG_REPEAT,
		  .repeat = { .id = 50, .count = 0 } },
	};
	EffectsV3IdList list = { .segments = segs, .n = 3 };

	EffectsV3IdCursor cur;
	TEST_ASSERT(EffectsV3IdCursor_Init(&cur, &list, 2) == true);

	uint64_t got[2];
	const uint64_t want[2] = {1, 2};

	TEST_ASSERT(_WalkAscending(&cur, got, 2) == 2);
	TEST_ASSERT(_Same(got, want, 2));
	TEST_ASSERT(_WalkWire(&cur, got, 2) == 2);
	TEST_ASSERT(_Same(got, want, 2));

	EffectsV3IdCursor_Free(&cur);
}

TEST_LIST = {
	{"cursor_sorts_segments_the_wire_left_unsorted", test_cursor_sorts_segments_the_wire_left_unsorted},
	{"cursor_interleaves_two_segments", test_cursor_interleaves_two_segments},
	{"cursor_reads_a_descending_range_from_the_other_end", test_cursor_reads_a_descending_range_from_the_other_end},
	{"cursor_repeats_an_id_as_many_times_as_stated", test_cursor_repeats_an_id_as_many_times_as_stated},
	{"cursor_walks_a_set_both_ways", test_cursor_walks_a_set_both_ways},
	{"cursor_refuses_a_set_that_is_not_the_size_decode_recorded", test_cursor_refuses_a_set_that_is_not_the_size_decode_recorded},
	{"cursor_refuses_segments_that_do_not_total_the_count", test_cursor_refuses_segments_that_do_not_total_the_count},
	{"cursor_walks_identically_every_time", test_cursor_walks_identically_every_time},
	{"cursor_steps_over_empty_segments", test_cursor_steps_over_empty_segments},
	{NULL, NULL}
};
