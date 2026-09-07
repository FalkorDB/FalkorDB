/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// Pins C's ported collapse arithmetic against the sizes roaring actually
// produces.
//
// The collapse rule decides whether to emit an Ascending segment by ARITHMETIC:
// `range_bytes >= 32 AND 5 + bitmap_bytes < range_bytes`, where bitmap_bytes is
// computed in closed form rather than by building a trial bitmap. If that
// arithmetic is wrong, an encoder collapses when it should not, or fails to when
// it should, and two engines emit different bytes for the same write.
//
// The chain this closes:
//
//     Rust formula == crate       predicted_matches_roaring, Rust side
//     crate        == CRoaring    test_effects_v3_roaring.c, twelve shapes
//     C formula    == CRoaring    HERE, on those same twelve shapes
//
// so the C formula predicts what the Rust formula predicts. The shapes and
// their expected byte counts are taken from that table, which took them from
// the pinned crate (=0.11.5); they are asserted here against BOTH the recorded
// number and what this tree's CRoaring 4.5.1 actually serializes, because a
// formula pinned against only one of those can be right about the crate and
// wrong about the library it will really size against.
//
// Shapes were chosen to hit array, bitset and run containers, the
// four-container offset-table threshold, and both the 2^16 and 2^32 boundaries.
//
// Built with one add_range_closed per range, matching the crate's one
// insert_range per range, because rule 3 makes construction path normative.

#include "src/effects/effects_v3_run_cost.h"
#include "src/util/roaring.h"

#include "acutest.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_RANGES 8

typedef struct {
	const char         *name;                  // shape name
	int                 n_ranges;              // number of inclusive ranges
	unsigned long long  ranges[MAX_RANGES][2]; // [lo, hi] inclusive
	size_t              n_bytes;               // serialized size, from the crate
} CostCase;

static const CostCase COST_CASES[] = {
	{ "one_range",               1, { {0, 9} },                                    27 },
	{ "one_long_range",          1, { {0, 99999} },                                37 },
	{ "two_runs",                2, { {0, 4999}, {100000, 104999} },               37 },
	{ "many_runs_one_container", 5, { {0,0}, {2,2}, {4,4}, {6,6}, {8,8} },          38 },
	{ "gap_of_one_array",        7, { {0,0}, {2,2}, {4,4}, {6,6}, {8,8},
	                                  {10,10}, {12,12} },                          42 },
	{ "dense_past_array_limit",  1, { {0, 4999} },                                 27 },
	{ "crosses_2p16",            1, { {65000, 65999} },                            37 },
	{ "crosses_2p32",            1, { {4294967000ULL, 4294967999ULL} },            46 },
	{ "four_containers_run",     4, { {0,2}, {65536,65538}, {131072,131074},
	                                  {196608,196610} },                           73 },
	{ "five_containers",         5, { {0,2}, {65536,65538}, {131072,131074},
	                                  {196608,196610}, {262144,262146} },          87 },
	{ "two_2p32_entries",        2, { {0,2}, {4294967296ULL, 4294967298ULL} },     46 },
	{ "wide_one_per_container",  6, { {0,0}, {65536,65536}, {131072,131072},
	                                  {196608,196608}, {262144,262144},
	                                  {327680,327680} },                           80 },
};

#define N_COST_CASES ((int)(sizeof(COST_CASES) / sizeof(COST_CASES[0])))

// fold a shape's ranges into a run tally, in ascending order, exactly as an
// encoder would as the ids arrived
static void _fold(const CostCase *c, EffectsV3Run *run) {
	EffectsV3Run_Restart(run);
	for(int r = 0; r < c->n_ranges; r++) {
		uint64_t lo = c->ranges[r][0];
		uint64_t hi = c->ranges[r][1];
		EffectsV3Run_AddRange(run, lo, hi - lo + 1);
	}
}

// what this tree's CRoaring charges for a shape, built by range
static size_t _croaring_bytes(const CostCase *c) {
	roaring64_bitmap_t *b = roaring64_bitmap_create();
	for(int r = 0; r < c->n_ranges; r++) {
		roaring64_bitmap_add_range_closed(b, c->ranges[r][0], c->ranges[r][1]);
	}
	// rule 2: called once, unconditionally. On the mandated add_range path it
	// is a NORMALIZATION rather than a size win -- it leaves ten shapes in
	// eleven untouched and makes one bigger -- so it must never be made
	// conditional on an expected saving
	roaring64_bitmap_run_optimize(b);

	size_t n = roaring64_bitmap_portable_size_in_bytes(b);
	roaring64_bitmap_free(b);
	return n;
}

// the ported formula predicts the recorded size, and CRoaring agrees with both
void test_effectsV3RunCost_predictsRoaringSize(void) {
	for(int i = 0; i < N_COST_CASES; i++) {
		const CostCase *c = COST_CASES + i;
		TEST_CASE(c->name);

		EffectsV3Run run;
		_fold(c, &run);

		size_t predicted = EffectsV3Run_BitmapBytes(&run);
		size_t croaring  = _croaring_bytes(c);

		TEST_ASSERT_(predicted == c->n_bytes,
				"%s: formula predicted %zu bytes, the Rust crate produces %zu -- "
				"the two engines would disagree on whether to collapse this run",
				c->name, predicted, c->n_bytes);

		TEST_ASSERT_(croaring == c->n_bytes,
				"%s: this tree's CRoaring serializes %zu bytes, the crate %zu -- "
				"the formula is pinned to the wrong library",
				c->name, croaring, c->n_bytes);
	}
}

// an empty run costs a treemap's bare prefix
//
// worth pinning because it is the one case with no open bucket, so it takes a
// different arm, and because a run is weighed before anything is folded in
void test_effectsV3RunCost_emptyRun(void) {
	EffectsV3Run run;
	EffectsV3Run_Restart(&run);

	TEST_ASSERT_(EffectsV3Run_BitmapBytes(&run) == 8,
			"an empty run should cost the 8-byte treemap prefix, got %zu",
			EffectsV3Run_BitmapBytes(&run));

	TEST_ASSERT_(!EffectsV3Run_PrefersBitmap(&run),
			"an empty run must never prefer a bitmap");
}

// the floor: below 32 range-bytes nothing is weighed, however cheap the bitmap
//
// this is the constant that stops a two-id run from being compared against a
// 27-byte bitmap it cannot beat
void test_effectsV3RunCost_floorSuppressesComparison(void) {
	EffectsV3Run run;
	EffectsV3Run_Restart(&run);
	EffectsV3Run_AddRange(&run, 0, 10);

	// a bitmap would cost 27 here, so without the floor a 31-byte run of ranges
	// would already look like a loss at 5 + 27 = 32 > 31 -- it does not, but the
	// floor is what makes the answer independent of that arithmetic
	EffectsV3Run_AddRangeBytes(&run, 31);
	TEST_ASSERT_(!EffectsV3Run_PrefersBitmap(&run),
			"31 range-bytes is below the 32 floor and must not be weighed");

	// at the floor the comparison runs: 5 + 27 = 32, which is not < 32
	EffectsV3Run_Restart(&run);
	EffectsV3Run_AddRange(&run, 0, 10);
	EffectsV3Run_AddRangeBytes(&run, 32);
	TEST_ASSERT_(!EffectsV3Run_PrefersBitmap(&run),
			"5 + 27 == 32 is not strictly less than 32, so it must not collapse");

	// one byte more of ranges and the bitmap wins
	EffectsV3Run_Restart(&run);
	EffectsV3Run_AddRange(&run, 0, 10);
	EffectsV3Run_AddRangeBytes(&run, 33);
	TEST_ASSERT_(EffectsV3Run_PrefersBitmap(&run),
			"5 + 27 == 32 < 33, so this run should collapse");
}

// a decline retires nothing: the comparison is not monotone, so a run that
// loses now can win later and must be re-weighed on the next segment
void test_effectsV3RunCost_declineIsNotFinal(void) {
	EffectsV3Run run;
	EffectsV3Run_Restart(&run);
	EffectsV3Run_AddRange(&run, 0, 3);
	EffectsV3Run_AddRangeBytes(&run, 20);

	TEST_ASSERT_(!EffectsV3Run_PrefersBitmap(&run),
			"20 range-bytes is below the floor");

	// the same run, more ranges charged: now it wins
	EffectsV3Run_AddRangeBytes(&run, 20);
	TEST_ASSERT_(EffectsV3Run_PrefersBitmap(&run),
			"40 range-bytes against a 27-byte bitmap should collapse");
}

// the id SET is what is sized, so folding the same ids as one range or as
// several adjacent ones changes the RUN COUNT and therefore the cost
//
// this is the property that makes add_range's bucket splitting necessary: a
// range spanning three buckets is three runs, not one
void test_effectsV3RunCost_bucketSplitCountsRuns(void) {
	// one range wholly inside a bucket: one run
	EffectsV3Run one;
	EffectsV3Run_Restart(&one);
	EffectsV3Run_AddRange(&one, 0, 10);

	// the same number of ids straddling a bucket boundary: two runs, in two
	// buckets, so two containers and a bigger bitmap
	EffectsV3Run split;
	EffectsV3Run_Restart(&split);
	EffectsV3Run_AddRange(&split, 65531, 10);

	TEST_ASSERT_(EffectsV3Run_BitmapBytes(&split) >
			EffectsV3Run_BitmapBytes(&one),
			"a range crossing a bucket boundary must cost more than one that "
			"does not: got %zu straddling vs %zu inside",
			EffectsV3Run_BitmapBytes(&split), EffectsV3Run_BitmapBytes(&one));
}

// the collapse THRESHOLD, in both directions
//
// The twelve shapes above pin the sizes the formula predicts. They cannot catch
// an off-by-one in what is IN SCOPE when the rule is evaluated, because that is
// invisible to every input except the two straddling the threshold - which is
// exactly the mistake this test exists for.
//
// ONLY SEGMENTS THAT HAVE STOPPED GROWING COUNT. On pushing the nth id the nth
// segment is still open and can still be extended, so it is excluded from
// range_bytes and from the candidate bitmap alike. Including it makes the
// decision depend on when the encoder happened to look rather than on the ids,
// which is the whole reason the rule is phrased over closed segments.
//
// Singletons at 100, 102, ... : each closed one is a lone Range costing
// 1 + width(base) + width(len) = 3 bytes. Charging the n-1 closed segments puts
// the threshold at 35 - the run collapses as the 35th id arrives, weighing 34
// frozen segments. Charging all n instead moves it to 34, and the two engines
// then emit different buffers for the same write.
void test_effectsV3RunCost_collapseThreshold(void) {
	struct { int n; bool collapses; } expect[] = {
		{ 33, false },  // 96 range-bytes against a 92-byte bitmap: 5 + 92 == 97
		{ 34, false },  // 99 against 94: 5 + 94 == 99, not strictly less than 99
		{ 35, true  },  // 102 against 96: 5 + 96 == 101 < 102
		{ 36, true  },
	};

	for(size_t e = 0; e < sizeof(expect) / sizeof(expect[0]); e++) {
		int n = expect[e].n;
		TEST_CASE_("%d singletons", n);

		EffectsV3Run run;
		EffectsV3Run_Restart(&run);

		// the n-1 segments that have stopped growing; the nth is still open
		for(int i = 0; i < n - 1; i++) {
			EffectsV3Run_AddRange(&run, 100 + 2 * (uint64_t)i, 1);
			EffectsV3Run_AddRangeBytes(&run, 3);
		}

		bool got = EffectsV3Run_PrefersBitmap(&run);
		TEST_ASSERT_(got == expect[e].collapses,
				"%d singletons: %zu range-bytes against a %zu-byte bitmap "
				"should%s collapse, but %s",
				n, run.range_bytes, EffectsV3Run_BitmapBytes(&run),
				expect[e].collapses ? "" : " not",
				got ? "it did" : "it did not");
	}
}

// a range reaching the very top of the id space terminates
//
// AddRange walks bucket by bucket and advances with piece_end + 1, which wraps
// to zero once piece_end is UINT64_MAX. That wrap is defined on unsigned in C,
// so it is not a fault - it is a valid loop index, and a loop condition tested
// after advancing would restart at the bottom of the id space and hang. This
// test is here because a hang is the one failure a size assertion cannot show.
void test_effectsV3RunCost_topOfIdSpaceTerminates(void) {
	EffectsV3Run run;

	// the last two ids in existence
	EffectsV3Run_Restart(&run);
	EffectsV3Run_AddRange(&run, UINT64_MAX - 1, 2);
	TEST_ASSERT_(EffectsV3Run_BitmapBytes(&run) > 0,
			"a range at the top of the id space should size, not hang");

	// a single id at the very top
	EffectsV3Run_Restart(&run);
	EffectsV3Run_AddRange(&run, UINT64_MAX, 1);
	TEST_ASSERT_(EffectsV3Run_BitmapBytes(&run) > 0,
			"the topmost single id should size");

	// one that ends exactly on a bucket boundary, so piece_end == end on the
	// last iteration rather than short of it
	EffectsV3Run_Restart(&run);
	EffectsV3Run_AddRange(&run, 65530, 6);
	TEST_ASSERT_(EffectsV3Run_BitmapBytes(&run) > 0,
			"a range ending exactly on a bucket boundary should size");
}

TEST_LIST = {
	{ "EffectsV3RunCost:predictsRoaringSize",
		test_effectsV3RunCost_predictsRoaringSize },
	{ "EffectsV3RunCost:emptyRun",
		test_effectsV3RunCost_emptyRun },
	{ "EffectsV3RunCost:floorSuppressesComparison",
		test_effectsV3RunCost_floorSuppressesComparison },
	{ "EffectsV3RunCost:declineIsNotFinal",
		test_effectsV3RunCost_declineIsNotFinal },
	{ "EffectsV3RunCost:bucketSplitCountsRuns",
		test_effectsV3RunCost_bucketSplitCountsRuns },
	{ "EffectsV3RunCost:collapseThreshold",
		test_effectsV3RunCost_collapseThreshold },
	{ "EffectsV3RunCost:topOfIdSpaceTerminates",
		test_effectsV3RunCost_topOfIdSpaceTerminates },
	{ NULL, NULL }
};
