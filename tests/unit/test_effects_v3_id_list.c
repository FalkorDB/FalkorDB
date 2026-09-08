/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// The IdList segment builder: which segments a sequence of ids becomes.
//
// This is normative, not an implementation choice. Two engines must reach the
// same segments for the same ids or the same write produces different bytes,
// so every assertion here is about the format rather than about this code being
// self-consistent.
//
// What the twelve cost shapes and the threshold test do NOT cover, and this
// does: the extension arms, the two lone-id rewrites, which segments end a run,
// and the interaction between a collapse and the pushes that follow it.

#include "src/effects/effects_v3_id_list.h"
#include "src/util/rmalloc.h"

// the builder allocates through the module allocator, which is a set of
// function pointers left NULL outside a loaded module
void setup() {
	Alloc_Reset();
}

#define TEST_INIT setup();
#include "acutest.h"

#include <stdio.h>
#include <stdlib.h>

// push a sequence and return the builder
static EffectsV3IdListBuilder *_build(const uint64_t *ids, size_t n) {
	EffectsV3IdListBuilder *b = EffectsV3IdListBuilder_New();
	for(size_t i = 0; i < n; i++) {
		EffectsV3IdListBuilder_Push(b, ids[i]);
	}
	return b;
}

#define BUILD(b, ...)                                              \
	uint64_t _ids[] = { __VA_ARGS__ };                             \
	EffectsV3IdListBuilder *b =                                    \
		_build(_ids, sizeof(_ids) / sizeof(_ids[0]))

// the ids a builder's segments describe, in order, so a test can assert on the
// sequence rather than on the segmentation where the sequence is what matters
static uint64_t *_expand(const EffectsV3IdListBuilder *b, uint64_t *out_n) {
	uint32_t n_segs = EffectsV3IdListBuilder_SegmentCount(b);
	uint64_t total  = EffectsV3IdListBuilder_Len(b);
	uint64_t *out   = malloc(sizeof(uint64_t) * (total > 0 ? total : 1));
	uint64_t k      = 0;

	for(uint32_t i = 0; i < n_segs; i++) {
		const EffectsV3Seg *s = EffectsV3IdListBuilder_Segment(b, i);
		uint32_t len = EffectsV3Seg_Len(s);

		if(s->kind == EFFECTS_V3_SEG_REPEAT) {
			for(uint32_t j = 0; j < len && k < total; j++) {
				out[k++] = s->repeat.id;
			}
		} else if(s->kind == EFFECTS_V3_SEG_RANGE) {
			for(uint32_t j = 0; j < len && k < total; j++) {
				out[k++] = s->descending
					? s->range.base - j
					: s->range.base + j;
			}
		} else {
			// a bitmap is a set read in the header's direction
			uint64_t card = roaring64_bitmap_get_cardinality(s->bitmap.bitmap);
			uint64_t *buf = malloc(sizeof(uint64_t) * (card > 0 ? card : 1));
			roaring64_bitmap_to_uint64_array(s->bitmap.bitmap, buf);
			for(uint64_t j = 0; j < card && k < total; j++) {
				out[k++] = s->descending ? buf[card - 1 - j] : buf[j];
			}
			free(buf);
		}
	}

	*out_n = k;
	return out;
}

static void _assert_ids(EffectsV3IdListBuilder *b, const uint64_t *want,
		size_t n_want, const char *what) {
	uint64_t got_n = 0;
	uint64_t *got  = _expand(b, &got_n);

	TEST_ASSERT_(got_n == n_want,
			"%s: describes %llu ids, expected %zu", what,
			(unsigned long long)got_n, n_want);

	for(size_t i = 0; i < n_want && i < got_n; i++) {
		TEST_ASSERT_(got[i] == want[i],
				"%s: id %zu is %llu, expected %llu", what, i,
				(unsigned long long)got[i], (unsigned long long)want[i]);
	}
	free(got);
}

// consecutive ids are one Range from the first push to the last
//
// this is the shape every bulk create and every delete-by-label takes, and the
// reason there is no plain form: it must never allocate a segment per id
void test_effectsV3IdList_ascendingRunIsOneSegment(void) {
	BUILD(b, 5, 6, 7, 8, 9, 10);

	TEST_ASSERT_(EffectsV3IdListBuilder_SegmentCount(b) == 1,
			"6 consecutive ids should be 1 segment, got %u",
			EffectsV3IdListBuilder_SegmentCount(b));

	const EffectsV3Seg *s = EffectsV3IdListBuilder_Segment(b, 0);
	TEST_ASSERT(s->kind == EFFECTS_V3_SEG_RANGE);
	TEST_ASSERT(!s->descending);
	TEST_ASSERT_(s->range.base == 5 && s->range.len == 6,
			"expected Range{base:5,len:6}, got Range{base:%llu,len:%u}",
			(unsigned long long)s->range.base, s->range.len);

	EffectsV3IdListBuilder_Free(b);
}

// a downward sequence is one DESCENDING Range, whose base is its highest id
//
// this is what an endpoint column written by a downward scan looks like.
// Without it the same ids are one segment each, which is the regression Repeat
// exists to prevent in the other direction
void test_effectsV3IdList_descendingRunIsOneSegment(void) {
	BUILD(b, 10, 9, 8, 7, 6, 5);

	TEST_ASSERT_(EffectsV3IdListBuilder_SegmentCount(b) == 1,
			"6 descending ids should be 1 segment, got %u",
			EffectsV3IdListBuilder_SegmentCount(b));

	const EffectsV3Seg *s = EffectsV3IdListBuilder_Segment(b, 0);
	TEST_ASSERT(s->kind == EFFECTS_V3_SEG_RANGE);
	TEST_ASSERT_(s->descending, "a downward run must set the descending flag");
	TEST_ASSERT_(s->range.base == 10 && s->range.len == 6,
			"a descending range's base is its FIRST and HIGHEST id: expected "
			"base 10 len 6, got base %llu len %u",
			(unsigned long long)s->range.base, s->range.len);

	// and the geometry helpers must read it the right way round
	TEST_ASSERT_(EffectsV3Seg_Min(s) == 5 && EffectsV3Seg_Max(s) == 10,
			"expected min 5 max 10, got min %llu max %llu",
			(unsigned long long)EffectsV3Seg_Min(s),
			(unsigned long long)EffectsV3Seg_Max(s));

	uint64_t want[] = { 10, 9, 8, 7, 6, 5 };
	_assert_ids(b, want, 6, "descending run");

	EffectsV3IdListBuilder_Free(b);
}

// a repeated id is one Repeat, and the flag is never set on it
//
// every edge out of a supernode carries the same source, so this is a whole
// endpoint column. 10,000 of them was 10,000 segments before Repeat existed
void test_effectsV3IdList_repeatIsOneSegment(void) {
	BUILD(b, 7, 7, 7, 7, 7);

	TEST_ASSERT_(EffectsV3IdListBuilder_SegmentCount(b) == 1,
			"5 copies of one id should be 1 segment, got %u",
			EffectsV3IdListBuilder_SegmentCount(b));

	const EffectsV3Seg *s = EffectsV3IdListBuilder_Segment(b, 0);
	TEST_ASSERT(s->kind == EFFECTS_V3_SEG_REPEAT);
	TEST_ASSERT_(!s->descending,
			"a Repeat has no direction and must never set the flag - a decoder "
			"rejects it there rather than ignoring it");
	TEST_ASSERT_(s->repeat.id == 7 && s->repeat.count == 5,
			"expected Repeat{id:7,count:5}, got Repeat{id:%llu,count:%u}",
			(unsigned long long)s->repeat.id, s->repeat.count);

	EffectsV3IdListBuilder_Free(b);
}

// a lone id has no direction; the id after it decides
void test_effectsV3IdList_loneIdTakesItsDirectionFromTheNext(void) {
	{
		BUILD(b, 5, 6);
		const EffectsV3Seg *s = EffectsV3IdListBuilder_Segment(b, 0);
		TEST_ASSERT_(EffectsV3IdListBuilder_SegmentCount(b) == 1 &&
				!s->descending && s->range.len == 2,
				"5 then 6 should be one ascending Range of 2");
		EffectsV3IdListBuilder_Free(b);
	}
	{
		BUILD(b, 5, 4);
		const EffectsV3Seg *s = EffectsV3IdListBuilder_Segment(b, 0);
		TEST_ASSERT_(EffectsV3IdListBuilder_SegmentCount(b) == 1 &&
				s->descending && s->range.base == 5 && s->range.len == 2,
				"5 then 4 should REWRITE the lone range descending, not open a "
				"second segment");
		EffectsV3IdListBuilder_Free(b);
	}
	{
		BUILD(b, 5, 5);
		const EffectsV3Seg *s = EffectsV3IdListBuilder_Segment(b, 0);
		TEST_ASSERT_(EffectsV3IdListBuilder_SegmentCount(b) == 1 &&
				s->kind == EFFECTS_V3_SEG_REPEAT && s->repeat.count == 2,
				"5 then 5 should rewrite the lone range as a Repeat");
		EffectsV3IdListBuilder_Free(b);
	}
}

// a reversal ends the run rather than continuing it
//
// a bitmap holds one order, so a step against the run's own direction settles
// everything before it
void test_effectsV3IdList_reversalEndsTheRun(void) {
	// up, then a step down: two segments, and the ids stay in push order
	BUILD(b, 1, 2, 3, 2);

	TEST_ASSERT_(EffectsV3IdListBuilder_SegmentCount(b) == 2,
			"an ascending run then a reversal should be 2 segments, got %u",
			EffectsV3IdListBuilder_SegmentCount(b));

	uint64_t want[] = { 1, 2, 3, 2 };
	_assert_ids(b, want, 4, "reversal");

	EffectsV3IdListBuilder_Free(b);
}

// duplicates and order survive, because rows are bound to ids positionally
//
// row k belongs to the k-th id AS WRITTEN, so anything that sorts or
// deduplicates this list lands every later row on the wrong entity
void test_effectsV3IdList_orderAndDuplicatesSurvive(void) {
	BUILD(b, 9, 3, 3, 7, 1, 9);

	uint64_t want[] = { 9, 3, 3, 7, 1, 9 };
	_assert_ids(b, want, 6, "unordered with duplicates");

	TEST_ASSERT_(EffectsV3IdListBuilder_Len(b) == 6,
			"6 ids pushed, builder reports %llu",
			(unsigned long long)EffectsV3IdListBuilder_Len(b));

	EffectsV3IdListBuilder_Free(b);
}

// a long gapped ascending sequence collapses to one bitmap, and the ids survive
void test_effectsV3IdList_gappedRunCollapses(void) {
	uint64_t ids[40];
	for(int i = 0; i < 40; i++) {
		ids[i] = 100 + 2 * (uint64_t)i;
	}

	EffectsV3IdListBuilder *b = _build(ids, 40);

	TEST_ASSERT_(EffectsV3IdListBuilder_SegmentCount(b) < 40,
			"40 gapped ids should have collapsed, still %u segments",
			EffectsV3IdListBuilder_SegmentCount(b));

	bool saw_bitmap = false;
	uint32_t n = EffectsV3IdListBuilder_SegmentCount(b);
	for(uint32_t i = 0; i < n; i++) {
		if(EffectsV3IdListBuilder_Segment(b, i)->kind ==
				EFFECTS_V3_SEG_ASCENDING) {
			saw_bitmap = true;
		}
	}
	TEST_ASSERT_(saw_bitmap, "expected a bitmap segment among %u", n);

	_assert_ids(b, ids, 40, "collapsed gapped run");

	EffectsV3IdListBuilder_Free(b);
}

// A REPEAT IS NEVER INSIDE THE RUN.
//
// A Repeat cannot become a bitmap: a bitmap holds a value once, a Repeat holds
// it count times. If the run still began at the Repeat, a later collapse would
// fold it in - keeping ONE copy of the id while counting all of them - and the
// segment's cardinality would then disagree with the ids it describes. That
// fails the decoder's cardinality check and refuses the buffer, so the replica
// resyncs and fails again identically.
//
// The sequence: a repeat, then a long gapped ascending run behind it, long
// enough that the run collapses.
void test_effectsV3IdList_repeatIsNeverCollapsed(void) {
	uint64_t ids[42];
	ids[0] = 100;
	ids[1] = 100;             // -> Repeat{100, 2}
	for(int i = 0; i < 40; i++) {
		ids[2 + i] = 200 + 2 * (uint64_t)i;   // a gapped run that will collapse
	}

	EffectsV3IdListBuilder *b = _build(ids, 42);

	// the Repeat must still be a Repeat holding both copies
	const EffectsV3Seg *first = EffectsV3IdListBuilder_Segment(b, 0);
	TEST_ASSERT_(first->kind == EFFECTS_V3_SEG_REPEAT,
			"the first segment should still be a Repeat, got kind %d",
			(int)first->kind);
	TEST_ASSERT_(first->repeat.count == 2,
			"the Repeat should still hold 2 copies, holds %u",
			first->repeat.count);

	// and every id must survive, in order, both copies of 100 included
	_assert_ids(b, ids, 42, "repeat ahead of a collapsing run");

	EffectsV3IdListBuilder_Free(b);
}

// a collapsed bitmap is never re-collapsed
//
// inserting a bitmap's span into a second bitmap adds every id in that span the
// set does not hold. Caching exact extremes removes a fabricated minimum; it
// does not remove this, which is why the run must start past the collapse
void test_effectsV3IdList_bitmapIsNeverRecollapsed(void) {
	// collapse a gapped ascending run, then push ids that keep going down from
	// below the bitmap's true minimum, which is what drives a second collapse
	uint64_t ids[80];
	size_t k = 0;
	for(int i = 0; i < 40; i++) {
		ids[k++] = 1000 + 2 * (uint64_t)i;   // collapses
	}
	for(int i = 0; i < 40; i++) {
		ids[k++] = 999 - 2 * (uint64_t)i;    // below it, descending
	}

	EffectsV3IdListBuilder *b = _build(ids, k);

	// whatever the segmentation, the ids it describes must be exactly what was
	// pushed - no id dropped, none invented
	_assert_ids(b, ids, k, "collapse then descend below it");

	EffectsV3IdListBuilder_Free(b);
}

// the extension arms ask "is this the id one past the end?", which has no
// answer at either end of the id space
//
// Unsigned wraparound is DEFINED in C rather than a trap, which makes this
// worse than undefined behaviour: base + len for base = UINT64_MAX, len = 1 is
// 0, so pushing id 0 after the highest id silently extends the range. The
// result claims an id that does not exist and reports a max below its min. The
// descending arm asks the mirror question and needs the mirror guard.
void test_effectsV3IdList_extensionAtTheEndsOfTheIdSpace(void) {
	{
		// the highest id, then the lowest: two unrelated ids, two segments
		BUILD(b, UINT64_MAX, 0);

		TEST_ASSERT_(EffectsV3IdListBuilder_SegmentCount(b) == 2,
				"UINT64_MAX then 0 must be 2 segments, got %u - id 0 is not "
				"one past the top of the id space",
				EffectsV3IdListBuilder_SegmentCount(b));

		const EffectsV3Seg *s = EffectsV3IdListBuilder_Segment(b, 0);
		TEST_ASSERT_(EffectsV3Seg_Min(s) <= EffectsV3Seg_Max(s),
				"segment 0 reports min %llu above max %llu",
				(unsigned long long)EffectsV3Seg_Min(s),
				(unsigned long long)EffectsV3Seg_Max(s));

		uint64_t want[] = { UINT64_MAX, 0 };
		_assert_ids(b, want, 2, "top then bottom");
		EffectsV3IdListBuilder_Free(b);
	}
	{
		// the mirror: the lowest id, then the highest. A descending run cannot
		// step below zero, so this is two segments too
		BUILD(b, 0, UINT64_MAX);

		TEST_ASSERT_(EffectsV3IdListBuilder_SegmentCount(b) == 2,
				"0 then UINT64_MAX must be 2 segments, got %u",
				EffectsV3IdListBuilder_SegmentCount(b));

		uint64_t want[] = { 0, UINT64_MAX };
		_assert_ids(b, want, 2, "bottom then top");
		EffectsV3IdListBuilder_Free(b);
	}
	{
		// and a genuine run at the top still extends: these ARE consecutive
		BUILD(b, UINT64_MAX - 2, UINT64_MAX - 1, UINT64_MAX);

		TEST_ASSERT_(EffectsV3IdListBuilder_SegmentCount(b) == 1,
				"the three highest ids are consecutive and must be 1 segment, "
				"got %u - the guard must not refuse a legal run",
				EffectsV3IdListBuilder_SegmentCount(b));

		uint64_t want[] = { UINT64_MAX - 2, UINT64_MAX - 1, UINT64_MAX };
		_assert_ids(b, want, 3, "run reaching the top");
		EffectsV3IdListBuilder_Free(b);
	}
	{
		// the descending mirror, reaching exactly zero
		BUILD(b, 2, 1, 0);

		TEST_ASSERT_(EffectsV3IdListBuilder_SegmentCount(b) == 1,
				"2,1,0 is one descending range reaching zero exactly, got %u",
				EffectsV3IdListBuilder_SegmentCount(b));

		uint64_t want[] = { 2, 1, 0 };
		_assert_ids(b, want, 3, "descending run reaching zero");
		EffectsV3IdListBuilder_Free(b);
	}
}

TEST_LIST = {
	{ "EffectsV3IdList:ascendingRunIsOneSegment",
		test_effectsV3IdList_ascendingRunIsOneSegment },
	{ "EffectsV3IdList:descendingRunIsOneSegment",
		test_effectsV3IdList_descendingRunIsOneSegment },
	{ "EffectsV3IdList:repeatIsOneSegment",
		test_effectsV3IdList_repeatIsOneSegment },
	{ "EffectsV3IdList:loneIdTakesItsDirectionFromTheNext",
		test_effectsV3IdList_loneIdTakesItsDirectionFromTheNext },
	{ "EffectsV3IdList:reversalEndsTheRun",
		test_effectsV3IdList_reversalEndsTheRun },
	{ "EffectsV3IdList:orderAndDuplicatesSurvive",
		test_effectsV3IdList_orderAndDuplicatesSurvive },
	{ "EffectsV3IdList:gappedRunCollapses",
		test_effectsV3IdList_gappedRunCollapses },
	{ "EffectsV3IdList:repeatIsNeverCollapsed",
		test_effectsV3IdList_repeatIsNeverCollapsed },
	{ "EffectsV3IdList:bitmapIsNeverRecollapsed",
		test_effectsV3IdList_bitmapIsNeverRecollapsed },
	{ "EffectsV3IdList:extensionAtTheEndsOfTheIdSpace",
		test_effectsV3IdList_extensionAtTheEndsOfTheIdSpace },
	{ NULL, NULL }
};
