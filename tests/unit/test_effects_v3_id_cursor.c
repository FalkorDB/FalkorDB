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

//------------------------------------------------------------------------------
// distinctness
//------------------------------------------------------------------------------
//
// A REPEAT HAS THREE PLACES TO HIDE, and there is one test per place, because
// a check that found only the obvious one would pass a smaller set of these.
// The control below is the test that keeps the whole group honest: a check
// that simply refused everything would pass all three negatives.

void test_distinct_accepts_a_sparse_interleaved_list() {
	// THE CONTROL. Unsorted across segments, sparse, and every id different -
	// the shape correct grouping produces. Refusing this would put a replica
	// into a resync loop on legitimate data.
	EffectsV3IdListSegment segs[3] = {
		{ .kind = EFFECTS_V3_SEG_RANGE_ASCENDING,
		  .range_ascending = { .base = 10, .len = 3 } },
		{ .kind = EFFECTS_V3_SEG_RANGE_DESCENDING,
		  .range_descending = { .base = 2, .len = 3 } },
		{ .kind = EFFECTS_V3_SEG_REPEAT,
		  .repeat = { .id = 100, .count = 1 } },
	};
	EffectsV3IdList list = { .segments = segs, .n = 3 };

	// 10,11,12 + 2,1,0 + 100
	TEST_ASSERT(EffectsV3IdList_AreDistinct(&list, 7) == true);
}

void test_distinct_refuses_a_repeat_segment() {
	// hiding place 1: one segment naming one id twice. This is the shape a
	// malformed DELETE_NODE takes, and the one that deletes a slot twice and
	// still reports success.
	EffectsV3IdListSegment seg = {
		.kind = EFFECTS_V3_SEG_REPEAT,
		.repeat = { .id = 7, .count = 2 },
	};
	EffectsV3IdList list = { .segments = &seg, .n = 1 };

	TEST_ASSERT_(EffectsV3IdList_AreDistinct(&list, 2) == false,
			"a Repeat of 2 names one id twice and must not pass as two ids");

	// and the same segment naming it ONCE is fine - so the refusal above is
	// the repetition and not the kind
	seg.repeat.count = 1;
	TEST_ASSERT(EffectsV3IdList_AreDistinct(&list, 1) == true);
}

void test_distinct_refuses_two_segments_that_overlap() {
	// hiding place 2: each segment is individually fine and the pair is not.
	// Nothing inside a segment can see this.
	EffectsV3IdListSegment segs[2] = {
		{ .kind = EFFECTS_V3_SEG_RANGE_ASCENDING,
		  .range_ascending = { .base = 4, .len = 3 } },   // 4,5,6
		{ .kind = EFFECTS_V3_SEG_RANGE_ASCENDING,
		  .range_ascending = { .base = 6, .len = 3 } },   // 6,7,8
	};
	EffectsV3IdList list = { .segments = segs, .n = 2 };

	TEST_ASSERT_(EffectsV3IdList_AreDistinct(&list, 6) == false,
			"6 appears in both ranges, so six positions name five ids");

	// moved apart by one, the same two shapes are legitimate
	segs[1].range_ascending.base = 7;
	TEST_ASSERT(EffectsV3IdList_AreDistinct(&list, 6) == true);
}

void test_distinct_refuses_a_bitmap_that_overlaps_a_range() {
	// hiding place 3: across two DIFFERENT kinds, where comparing bounds is
	// least able to help - a bitmap's bounds can overlap a range while its
	// members do not, so bounds alone would refuse good data here and
	// members are the only exact answer
	const uint64_t members[3] = {20, 30, 40};
	char buf[512];
	const uint32_t n = _Blob(members, 3, buf, sizeof(buf));

	EffectsV3IdListSegment segs[2] = {
		{ .kind = EFFECTS_V3_SEG_SET_ASCENDING,
		  .set_ascending = { .blob = (unsigned char *)buf, .n = n,
							 .cardinality = 3 } },
		{ .kind = EFFECTS_V3_SEG_RANGE_ASCENDING,
		  .range_ascending = { .base = 30, .len = 1 } },
	};
	EffectsV3IdList list = { .segments = segs, .n = 2 };

	TEST_ASSERT_(EffectsV3IdList_AreDistinct(&list, 4) == false,
			"30 is in the bitmap and in the range");

	// a range INSIDE the bitmap's bounds but not among its members is
	// legitimate, and must be accepted - this is the case bounds would get
	// wrong
	segs[1].range_ascending.base = 25;
	TEST_ASSERT_(EffectsV3IdList_AreDistinct(&list, 4) == true,
			"25 lies between 20 and 40 but is not a member");
}

void test_distinct_refuses_a_count_the_list_does_not_hold() {
	// the same comparison catches a list that is short of its stated count,
	// which is how a repeat reaches this at all
	EffectsV3IdListSegment seg = {
		.kind = EFFECTS_V3_SEG_RANGE_ASCENDING,
		.range_ascending = { .base = 0, .len = 3 },
	};
	EffectsV3IdList list = { .segments = &seg, .n = 1 };

	TEST_ASSERT(EffectsV3IdList_AreDistinct(&list, 3) == true);
	TEST_ASSERT(EffectsV3IdList_AreDistinct(&list, 4) == false);
	TEST_ASSERT(EffectsV3IdList_AreDistinct(&list, 2) == false);
}

void test_distinct_reads_a_descending_range_from_its_highest_id() {
	// this helper has its OWN copy of the span arithmetic, so the direction is
	// pinned here as well as on the walk. The two segments are chosen so that
	// reading the descending base as the range's LOWEST id - the natural
	// mistake - makes them overlap: 9,10,11 against 10,11,12.
	EffectsV3IdListSegment segs[2] = {
		{ .kind = EFFECTS_V3_SEG_RANGE_ASCENDING,
		  .range_ascending = { .base = 10, .len = 3 } },   // 10,11,12
		{ .kind = EFFECTS_V3_SEG_RANGE_DESCENDING,
		  .range_descending = { .base = 9, .len = 3 } },   // 9,8,7
	};
	EffectsV3IdList list = { .segments = segs, .n = 2 };

	TEST_ASSERT_(EffectsV3IdList_AreDistinct(&list, 6) == true,
			"7,8,9 and 10,11,12 are six different ids; reading the descending "
			"base as the lowest would overlap them and refuse this");
}

void test_distinct_refuses_a_repeat_a_short_count_would_have_hidden() {
	// THE CASE CARDINALITY ALONE GETS WRONG. Three different ids, four
	// positions, 12 named twice - and a count of 3 makes the member count
	// agree. Only comparing the POSITIONS as well catches it.
	EffectsV3IdListSegment segs[2] = {
		{ .kind = EFFECTS_V3_SEG_RANGE_ASCENDING,
		  .range_ascending = { .base = 10, .len = 2 } },  // 10,11
		{ .kind = EFFECTS_V3_SEG_REPEAT,
		  .repeat = { .id = 12, .count = 2 } },           // 12,12
	};
	EffectsV3IdList list = { .segments = segs, .n = 2 };

	TEST_ASSERT_(EffectsV3IdList_AreDistinct(&list, 3) == false,
			"the segments name four ids and hold three; a count of 3 makes "
			"the members agree while 12 is still named twice");

	// the honest count for the same segments is still refused, now because
	// the members fall short - so the test above is the positions check and
	// not an accident of the member check
	TEST_ASSERT(EffectsV3IdList_AreDistinct(&list, 4) == false);
}

void test_distinct_counts_an_empty_segment_as_nothing() {
	// zero-length segments are accepted by decode; each guard here is what
	// stops a length of 0 becoming a span of (0 - 1)
	EffectsV3IdListSegment segs[4] = {
		{ .kind = EFFECTS_V3_SEG_RANGE_ASCENDING,
		  .range_ascending = { .base = 5, .len = 0 } },
		{ .kind = EFFECTS_V3_SEG_RANGE_DESCENDING,
		  .range_descending = { .base = 5, .len = 0 } },
		{ .kind = EFFECTS_V3_SEG_REPEAT,
		  .repeat = { .id = 5, .count = 0 } },
		{ .kind = EFFECTS_V3_SEG_RANGE_ASCENDING,
		  .range_ascending = { .base = 7, .len = 2 } },
	};
	EffectsV3IdList list = { .segments = segs, .n = 4 };

	TEST_ASSERT_(EffectsV3IdList_AreDistinct(&list, 2) == true,
			"only 7,8 are named; the three empty segments contribute nothing");
	TEST_ASSERT(EffectsV3IdList_AreDistinct(&list, 3) == false);
}

void test_distinct_refuses_a_segment_kind_it_does_not_know() {
	EffectsV3IdListSegment seg = { .kind = (EffectsV3IdListSegmentKind)99 };
	EffectsV3IdList list = { .segments = &seg, .n = 1 };

	// AT COUNT 0, which is the only count that asks the question. An unknown
	// kind contributes no positions and no members, so at any other count the
	// two comparisons refuse the record whether or not the kind was noticed -
	// and a test that passes either way pins nothing.
	TEST_ASSERT_(EffectsV3IdList_AreDistinct(&list, 0) == false,
			"a kind this build cannot read is refused, not counted as empty");

	// the control: a list that genuinely holds nothing IS distinct at 0
	EffectsV3IdList empty = { .segments = NULL, .n = 0 };
	TEST_ASSERT(EffectsV3IdList_AreDistinct(&empty, 0) == true);
}

void test_cursor_refuses_a_segment_naming_the_invalid_id() {
	// AN ENDPOINT COLUMN NEVER REACHES A CLAIM. _ApplyCreateEdge walks src and
	// dst with the wire walk and hands each id to Graph_HasNode, so the
	// cursor is the only thing between the wire and that call - and the
	// sentinel is no entity's id.
	EffectsV3IdListSegment rep = {
		.kind = EFFECTS_V3_SEG_REPEAT,
		.repeat = { .id = UINT64_MAX, .count = 1 },
	};
	EffectsV3IdList l1 = { .segments = &rep, .n = 1 };

	EffectsV3IdCursor cur;
	TEST_ASSERT_(EffectsV3IdCursor_Init(&cur, &l1, 1) == false,
			"a Repeat naming INVALID_ENTITY_ID must be refused");
	EffectsV3IdCursor_Free(&cur);

	// a descending range whose BASE is the sentinel - its highest id, so the
	// arm that reads the other end would miss it
	EffectsV3IdListSegment desc = {
		.kind = EFFECTS_V3_SEG_RANGE_DESCENDING,
		.range_descending = { .base = UINT64_MAX, .len = 2 },
	};
	EffectsV3IdList l2 = { .segments = &desc, .n = 1 };
	TEST_ASSERT(EffectsV3IdCursor_Init(&cur, &l2, 2) == false);
	EffectsV3IdCursor_Free(&cur);

	// and an ascending range REACHING it from below
	EffectsV3IdListSegment asc = {
		.kind = EFFECTS_V3_SEG_RANGE_ASCENDING,
		.range_ascending = { .base = UINT64_MAX - 1, .len = 2 },
	};
	EffectsV3IdList l3 = { .segments = &asc, .n = 1 };
	TEST_ASSERT(EffectsV3IdCursor_Init(&cur, &l3, 2) == false);
	EffectsV3IdCursor_Free(&cur);

	// THE CONTROL: one id below the sentinel is an ordinary id
	EffectsV3IdListSegment ok = {
		.kind = EFFECTS_V3_SEG_RANGE_ASCENDING,
		.range_ascending = { .base = UINT64_MAX - 2, .len = 2 },
	};
	EffectsV3IdList l4 = { .segments = &ok, .n = 1 };
	TEST_ASSERT_(EffectsV3IdCursor_Init(&cur, &l4, 2) == true,
			"ids below the sentinel are ordinary and must still walk");
	EffectsV3IdCursor_Free(&cur);
}

void test_distinct_falls_back_when_spans_overlap() {
	// THE CASE THAT DECIDES THE SHAPE OF THIS CHECK.
	//
	// Sorted by lowest id: {0,50}, [10..20], {50,60}. Comparing only ADJACENT
	// pairs asks ({0,50}, [10..20]) - spans overlap, no shared id, clean - and
	// then ([10..20], {50,60}) - spans apart, clean - and never asks
	// ({0,50}, {50,60}), which share 50.
	//
	// Adjacent-only is sound when OVERLAPPING SPANS ARE THEMSELVES the
	// refusal, because sorted by start an overlap always shows up between
	// neighbours. It stops being sound as soon as an overlapping pair can come
	// back clean, which is exactly what makes the bitmap case above correct.
	char b1[512], b2[512];
	const uint64_t lo[2] = {0, 50};
	const uint64_t hi[2] = {50, 60};
	const uint32_t n1 = _Blob(lo, 2, b1, sizeof(b1));
	const uint32_t n2 = _Blob(hi, 2, b2, sizeof(b2));

	EffectsV3IdListSegment segs[3] = {
		{ .kind = EFFECTS_V3_SEG_SET_ASCENDING,
		  .set_ascending = { .blob = (unsigned char *)b1, .n = n1, .cardinality = 2 } },
		{ .kind = EFFECTS_V3_SEG_RANGE_ASCENDING,
		  .range_ascending = { .base = 10, .len = 11 } },
		{ .kind = EFFECTS_V3_SEG_SET_ASCENDING,
		  .set_ascending = { .blob = (unsigned char *)b2, .n = n2, .cardinality = 2 } },
	};
	EffectsV3IdList list = { .segments = segs, .n = 3 };

	TEST_ASSERT_(EffectsV3IdList_AreDistinct(&list, 15) == false,
			"50 is in the first and last segments, and a scan comparing only "
			"neighbours never puts them side by side - which is why an "
			"overlap is handed to the union rather than settled in place");
}

void test_distinct_pins_the_descending_span_against_an_overlap() {
	// THE DIRECTION HAS TO BE PINNED WHERE IT COSTS SOMETHING TO GET WRONG.
	// A descending range read from the wrong end is caught by the test above
	// only if the mistake WIDENS the span - a widened span overlaps, hands the
	// list to the union, and the union's own arithmetic answers correctly.
	// A NARROWED span is the dangerous one: it hides a real overlap, the fast
	// path sees nothing touching, and the list is called distinct.
	//
	// 7,8,9 descending against 7,8 ascending: five positions, three ids.
	EffectsV3IdListSegment segs[2] = {
		{ .kind = EFFECTS_V3_SEG_RANGE_DESCENDING,
		  .range_descending = { .base = 9, .len = 3 } },   // 9,8,7
		{ .kind = EFFECTS_V3_SEG_RANGE_ASCENDING,
		  .range_ascending = { .base = 7, .len = 2 } },    // 7,8
	};
	EffectsV3IdList list = { .segments = segs, .n = 2 };

	TEST_ASSERT_(EffectsV3IdList_AreDistinct(&list, 5) == false,
			"reading the descending base as anything but the HIGHEST id "
			"narrows the span to [9,9], which misses the overlap at 7 and 8");

	// and the same two shapes, moved apart, are legitimate
	segs[1].range_ascending.base = 20;
	TEST_ASSERT(EffectsV3IdList_AreDistinct(&list, 5) == true);
}

void test_distinct_refuses_a_set_whose_blob_disagrees_with_its_count() {
	// THE CHECK THAT LICENSES THE FAST PATH FOR A SET. Spans alone say a
	// bitmap occupies [min, max]; only this says it holds 'card' different
	// ids in there. Without it a blob of three passes as five.
	const uint64_t members[3] = {1, 2, 3};
	char buf[512];
	const uint32_t n = _Blob(members, 3, buf, sizeof(buf));

	EffectsV3IdListSegment seg = {
		.kind = EFFECTS_V3_SEG_SET_ASCENDING,
		.set_ascending = { .blob = (unsigned char *)buf, .n = n,
						   .cardinality = 5 },
	};
	EffectsV3IdList list = { .segments = &seg, .n = 1 };

	TEST_ASSERT_(EffectsV3IdList_AreDistinct(&list, 5) == false,
			"the blob holds three ids and the segment claims five");

	// corrected, the same blob is fine
	seg.set_ascending.cardinality = 3;
	TEST_ASSERT(EffectsV3IdList_AreDistinct(&list, 3) == true);
}

void test_distinct_opens_a_set_that_claims_to_be_empty() {
	// A Range or a Repeat stating zero names nothing and there is nothing left
	// to ask. A SET states its size beside a blob that has its own, and if the
	// segment is stepped over as empty that disagreement is never found - the
	// blob's two ids would travel unexamined.
	const uint64_t members[2] = {100, 101};
	char buf[512];
	const uint32_t n = _Blob(members, 2, buf, sizeof(buf));

	EffectsV3IdListSegment segs[2] = {
		{ .kind = EFFECTS_V3_SEG_SET_ASCENDING,
		  .set_ascending = { .blob = (unsigned char *)buf, .n = n,
							 .cardinality = 0 } },
		{ .kind = EFFECTS_V3_SEG_RANGE_ASCENDING,
		  .range_ascending = { .base = 1, .len = 3 } },
	};
	EffectsV3IdList list = { .segments = segs, .n = 2 };

	TEST_ASSERT_(EffectsV3IdList_AreDistinct(&list, 3) == false,
			"a set claiming nothing while its blob holds two ids is refused, "
			"not stepped over");

	// a set that IS empty, and says so, names nothing and is stepped over
	const uint32_t e = _Blob(NULL, 0, buf, sizeof(buf));
	segs[0].set_ascending.n = e;
	TEST_ASSERT(EffectsV3IdList_AreDistinct(&list, 3) == true);
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
	{"distinct_accepts_a_sparse_interleaved_list", test_distinct_accepts_a_sparse_interleaved_list},
	{"distinct_refuses_a_repeat_segment", test_distinct_refuses_a_repeat_segment},
	{"distinct_refuses_two_segments_that_overlap", test_distinct_refuses_two_segments_that_overlap},
	{"distinct_refuses_a_bitmap_that_overlaps_a_range", test_distinct_refuses_a_bitmap_that_overlaps_a_range},
	{"distinct_refuses_a_count_the_list_does_not_hold", test_distinct_refuses_a_count_the_list_does_not_hold},
	{"distinct_reads_a_descending_range_from_its_highest_id", test_distinct_reads_a_descending_range_from_its_highest_id},
	{"distinct_refuses_a_repeat_a_short_count_would_have_hidden", test_distinct_refuses_a_repeat_a_short_count_would_have_hidden},
	{"distinct_counts_an_empty_segment_as_nothing", test_distinct_counts_an_empty_segment_as_nothing},
	{"distinct_refuses_a_segment_kind_it_does_not_know", test_distinct_refuses_a_segment_kind_it_does_not_know},
	{"cursor_refuses_a_segment_naming_the_invalid_id", test_cursor_refuses_a_segment_naming_the_invalid_id},
	{"distinct_falls_back_when_spans_overlap", test_distinct_falls_back_when_spans_overlap},
	{"distinct_pins_the_descending_span_against_an_overlap", test_distinct_pins_the_descending_span_against_an_overlap},
	{"distinct_refuses_a_set_whose_blob_disagrees_with_its_count", test_distinct_refuses_a_set_whose_blob_disagrees_with_its_count},
	{"distinct_opens_a_set_that_claims_to_be_empty", test_distinct_opens_a_set_that_claims_to_be_empty},
	{NULL, NULL}
};
