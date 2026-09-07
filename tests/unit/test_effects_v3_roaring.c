/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

// Pins C's vendored CRoaring against the Rust `roaring` crate's bytes.
//
// An effects v3 Ascending segment carries a roaring64 bitmap verbatim, so the
// two engines agree on that segment only if their two roaring implementations
// serialize the same set to the same bytes. That was measured once, by hand, on
// 2026-09-03 (harness in .handover/roaring-parity): identical on all 11 shapes,
// on both construction paths. A measurement that lives in a scratch directory
// rots the moment CRoaring is bumped and nobody notices, which is why it is
// here instead.
//
// The expected bytes below are the RUST CRATE's output, captured from that
// harness. Nothing in this file runs Rust: the crate's bytes are the constants
// and CRoaring is the thing under test. So a CRoaring bump that changes the
// wire fails here, which is the whole point -- C vendors CRoaring 4.5.1 at
// src/util/roaring.{c,h} and the crate is version-pinned exactly in
// graph/Cargo.toml for the same reason.
//
// WHAT THIS DOES NOT COVER. Only the byte comparison is platform-independent
// and runs everywhere. The ASAN runs over the truncation corpus are CI-on-Linux
// (`make unit-tests SAN=address`); on darwin the unit targets link the objects
// rather than the shared library and the sanitizer story differs. Nothing here
// needs a sanitizer to be meaningful.

#include "src/util/roaring.h"

#include "acutest.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_RANGES 64

typedef struct {
	const char         *name;                 // shape name
	int                 n_ranges;             // number of inclusive ranges
	unsigned long long  ranges[MAX_RANGES][2];// [lo, hi] inclusive
	size_t              n_range_path;         // bytes, built by add_range
	const char         *hex_range_path;       // those bytes, from the crate
	size_t              n_byid_path;          // bytes, built id by id
	const char         *hex_byid_path;        // those bytes, from the crate
} RoaringCase;

static const RoaringCase ROARING_CASES[] = {
	{
		"single_id", 1,
		{ {7ULL, 7ULL} },
		30, "0100000000000000000000003a3000000100000000000000100000000700",
		30, "0100000000000000000000003a3000000100000000000000100000000700",
	},
	{
		"one_small_range", 1,
		{ {0ULL, 2ULL} },
		27, "0100000000000000000000003b3000000100000200010000000200",
		34, "0100000000000000000000003a300000010000000000020010000000000001000200",
	},
	{
		"four_3id_buckets", 4,
		{ {0ULL, 2ULL}, {65536ULL, 65538ULL}, {131072ULL, 131074ULL}, {196608ULL, 196610ULL} },
		73, "0100000000000000000000003b3003000f00000200010002000200020003000200250000002b0000003100000037000000010000000200010000000200010000000200010000000200",
		76, "0100000000000000000000003a3000000400000000000200010002000200020003000200280000002e000000340000003a000000000001000200000001000200000001000200000001000200",
	},
	{
		"dense_10k", 1,
		{ {0ULL, 9999ULL} },
		27, "0100000000000000000000003b3000000100000f27010000000f27",
		27, "0100000000000000000000003b3000000100000f27010000000f27",
	},
	{
		"two_ranges_one_bucket", 2,
		{ {10ULL, 20ULL}, {5000ULL, 5100ULL} },
		31, "0100000000000000000000003b3000000100006f0002000a000a0088136400",
		31, "0100000000000000000000003b3000000100006f0002000a000a0088136400",
	},
	{
		"many_pairs_one_bucket", 8,
		{ {0ULL, 1ULL}, {100ULL, 101ULL}, {200ULL, 201ULL}, {300ULL, 301ULL}, {400ULL, 401ULL}, {500ULL, 501ULL}, {600ULL, 601ULL}, {700ULL, 701ULL} },
		60, "0100000000000000000000003a3000000100000000000f00100000000000010064006500c800c9002c012d0190019101f401f50158025902bc02bd02",
		60, "0100000000000000000000003a3000000100000000000f00100000000000010064006500c800c9002c012d0190019101f401f50158025902bc02bd02",
	},
	{
		"crosses_bucket_bdry", 1,
		{ {65530ULL, 65540ULL} },
		37, "0100000000000000000000003b3001000300000500010004000100faff0500010000000400",
		37, "0100000000000000000000003b3001000300000500010004000100faff0500010000000400",
	},
	{
		"sparse_singletons", 6,
		{ {0ULL, 0ULL}, {70000ULL, 70000ULL}, {140000ULL, 140000ULL}, {210000ULL, 210000ULL}, {280000ULL, 280000ULL}, {350000ULL, 350000ULL} },
		80, "0100000000000000000000003a30000006000000000000000100000002000000030000000400000005000000380000003a0000003c0000003e000000400000004200000000007011e0225034c0453057",
		80, "0100000000000000000000003a30000006000000000000000100000002000000030000000400000005000000380000003a0000003c0000003e000000400000004200000000007011e0225034c0453057",
	},
	{
		"wide_sparse_32b", 3,
		{ {0ULL, 9ULL}, {4294967296ULL, 4294967305ULL}, {8589934592ULL, 8589934601ULL} },
		65, "0300000000000000000000003b3000000100000900010000000900010000003b3000000100000900010000000900020000003b3000000100000900010000000900",
		65, "0300000000000000000000003b3000000100000900010000000900010000003b3000000100000900010000000900020000003b3000000100000900010000000900",
	},
	{
		"runs_16", 16,
		{ {0ULL, 5ULL}, {10ULL, 15ULL}, {20ULL, 25ULL}, {30ULL, 35ULL}, {40ULL, 45ULL}, {50ULL, 55ULL}, {60ULL, 65ULL}, {70ULL, 75ULL}, {80ULL, 85ULL}, {90ULL, 95ULL}, {100ULL, 105ULL}, {110ULL, 115ULL}, {120ULL, 125ULL}, {130ULL, 135ULL}, {140ULL, 145ULL}, {150ULL, 155ULL} },
		87, "0100000000000000000000003b3000000100005f001000000005000a000500140005001e00050028000500320005003c00050046000500500005005a000500640005006e00050078000500820005008c00050096000500",
		87, "0100000000000000000000003b3000000100005f001000000005000a000500140005001e00050028000500320005003c00050046000500500005005a000500640005006e00050078000500820005008c00050096000500",
	},
	{
		"big_bitset_bucket", 40,
		{ {0ULL, 500ULL}, {1000ULL, 1500ULL}, {2000ULL, 2500ULL}, {3000ULL, 3500ULL}, {4000ULL, 4500ULL}, {5000ULL, 5500ULL}, {6000ULL, 6500ULL}, {7000ULL, 7500ULL}, {8000ULL, 8500ULL}, {9000ULL, 9500ULL}, {10000ULL, 10500ULL}, {11000ULL, 11500ULL}, {12000ULL, 12500ULL}, {13000ULL, 13500ULL}, {14000ULL, 14500ULL}, {15000ULL, 15500ULL}, {16000ULL, 16500ULL}, {17000ULL, 17500ULL}, {18000ULL, 18500ULL}, {19000ULL, 19500ULL}, {20000ULL, 20500ULL}, {21000ULL, 21500ULL}, {22000ULL, 22500ULL}, {23000ULL, 23500ULL}, {24000ULL, 24500ULL}, {25000ULL, 25500ULL}, {26000ULL, 26500ULL}, {27000ULL, 27500ULL}, {28000ULL, 28500ULL}, {29000ULL, 29500ULL}, {30000ULL, 30500ULL}, {31000ULL, 31500ULL}, {32000ULL, 32500ULL}, {33000ULL, 33500ULL}, {34000ULL, 34500ULL}, {35000ULL, 35500ULL}, {36000ULL, 36500ULL}, {37000ULL, 37500ULL}, {38000ULL, 38500ULL}, {39000ULL, 39500ULL} },
		183, "0100000000000000000000003b300000010000474e28000000f401e803f401d007f401b80bf401a00ff4018813f4017017f401581bf401401ff4012823f4011027f401f82af401e02ef401c832f401b036f401983af401803ef4016842f4015046f401384af401204ef4010852f401f055f401d859f401c05df401a861f4019065f4017869f401606df4014871f4013075f4011879f401007df401e880f401d084f401b888f401a08cf4018890f4017094f4015898f401",
		183, "0100000000000000000000003b300000010000474e28000000f401e803f401d007f401b80bf401a00ff4018813f4017017f401581bf401401ff4012823f4011027f401f82af401e02ef401c832f401b036f401983af401803ef4016842f4015046f401384af401204ef4010852f401f055f401d859f401c05df401a861f4019065f4017869f401606df4014871f4013075f4011879f401007df401e880f401d084f401b888f401a08cf4018890f4017094f4015898f401",
	},
};

#define N_ROARING_CASES ((int)(sizeof(ROARING_CASES) / sizeof(ROARING_CASES[0])))

//------------------------------------------------------------------------------
// helpers
//------------------------------------------------------------------------------

// how a bitmap is built is itself normative, so both paths are spelled out
typedef enum {
	BUILD_ADD_RANGE,  // one add_range_closed per contributing range
	BUILD_ID_BY_ID,   // every id added individually
} BuildPath;

static roaring64_bitmap_t *_build
(
	const RoaringCase *c,  // shape to build
	BuildPath path,        // how to build it
	bool optimize          // whether to run_optimize before returning
) {
	roaring64_bitmap_t *b = roaring64_bitmap_create();

	for(int i = 0; i < c->n_ranges; i++) {
		if(path == BUILD_ADD_RANGE) {
			roaring64_bitmap_add_range_closed(b, c->ranges[i][0], c->ranges[i][1]);
		} else {
			for(unsigned long long id = c->ranges[i][0];
					id <= c->ranges[i][1]; id++) {
				roaring64_bitmap_add(b, id);
			}
		}
	}

	if(optimize) roaring64_bitmap_run_optimize(b);

	return b;
}

// serialize into a caller-freed hex string; *n_bytes gets the byte count
static char *_serialize_hex
(
	const roaring64_bitmap_t *b,  // bitmap to serialize
	size_t *n_bytes               // [output] serialized size in bytes
) {
	size_t sz  = roaring64_bitmap_portable_size_in_bytes(b);
	char  *buf = (char*)malloc(sz);
	size_t w   = roaring64_bitmap_portable_serialize(b, buf);

	char *hex = (char*)malloc(w * 2 + 1);
	for(size_t i = 0; i < w; i++) {
		snprintf(hex + i * 2, 3, "%02x", (unsigned char)buf[i]);
	}
	hex[w * 2] = '\0';

	free(buf);
	*n_bytes = w;

	return hex;
}

// assert one shape, built one way, serializes to the crate's bytes
static void _assert_matches
(
	const RoaringCase *c,     // shape under test
	BuildPath path,           // how to build it
	size_t want_n,            // expected byte count
	const char *want_hex      // expected bytes, from the Rust crate
) {
	roaring64_bitmap_t *b = _build(c, path, true);

	size_t got_n = 0;
	char  *got   = _serialize_hex(b, &got_n);

	TEST_ASSERT_(got_n == want_n,
			"%s (%s): CRoaring wrote %zu bytes, the Rust crate wrote %zu",
			c->name, path == BUILD_ADD_RANGE ? "add_range" : "id-by-id",
			got_n, want_n);

	TEST_ASSERT_(strcmp(got, want_hex) == 0,
			"%s (%s): bytes diverged from the Rust crate\n"
			"  CRoaring: %s\n"
			"  crate:    %s",
			c->name, path == BUILD_ADD_RANGE ? "add_range" : "id-by-id",
			got, want_hex);

	free(got);
	roaring64_bitmap_free(b);
}

//------------------------------------------------------------------------------
// the two construction paths, against the crate
//------------------------------------------------------------------------------

// the path a v3 encoder MUST use: one add_range_closed per contributing range
void test_effectsV3Roaring_addRangePath(void) {
	for(int i = 0; i < N_ROARING_CASES; i++) {
		const RoaringCase *c = ROARING_CASES + i;
		TEST_CASE(c->name);
		_assert_matches(c, BUILD_ADD_RANGE, c->n_range_path, c->hex_range_path);
	}
}

// the path it must NOT use, pinned so that the difference stays visible
void test_effectsV3Roaring_idByIdPath(void) {
	for(int i = 0; i < N_ROARING_CASES; i++) {
		const RoaringCase *c = ROARING_CASES + i;
		TEST_CASE(c->name);
		_assert_matches(c, BUILD_ID_BY_ID, c->n_byid_path, c->hex_byid_path);
	}
}

//------------------------------------------------------------------------------
// the normative rules these bytes exist to justify
//------------------------------------------------------------------------------

// How a bitmap is BUILT changes its bytes, so the encoder's choice of
// add_range_closed is normative and not a style preference.
//
// roaring's optimize() is path dependent: a container reached from an array
// store converts to runs only on a strict win, one reached from a run store
// stays runs unless strictly beaten -- and a run-flavoured bitmap carries a
// different header. This asserts the difference is still real, on the shapes
// where it was measured.
//
// If a CRoaring bump ever makes the two paths converge, this test fails and
// that is the correct outcome: the encoder's constraint would then rest on
// something no longer true, and somebody should decide that deliberately
// rather than discover it from a replica whose bitmaps do not match.
void test_effectsV3Roaring_constructionPathIsNormative(void) {
	// measured 2026-09-03 and reproduced here; the magnitudes are the evidence
	// cited in the handover brief for rule 3
	struct { const char *name; size_t range_path; size_t byid_path; } expect[] = {
		{ "one_small_range",  27, 34 },
		{ "four_3id_buckets", 73, 76 },
	};

	int checked = 0;

	for(size_t e = 0; e < sizeof(expect) / sizeof(expect[0]); e++) {
		for(int i = 0; i < N_ROARING_CASES; i++) {
			const RoaringCase *c = ROARING_CASES + i;
			if(strcmp(c->name, expect[e].name) != 0) continue;

			TEST_CASE(c->name);

			// build both ways HERE rather than comparing the recorded numbers
			// to each other: this has to fail when the library changes, and a
			// table compared against itself never would
			roaring64_bitmap_t *by_range = _build(c, BUILD_ADD_RANGE, true);
			roaring64_bitmap_t *by_id    = _build(c, BUILD_ID_BY_ID,  true);

			size_t n_range = 0, n_id = 0;
			char  *h_range = _serialize_hex(by_range, &n_range);
			char  *h_id    = _serialize_hex(by_id,    &n_id);

			TEST_ASSERT_(strcmp(h_range, h_id) != 0,
					"%s: both construction paths now produce the same %zu "
					"bytes -- rule 3 rests on a difference that no longer "
					"exists in this CRoaring", c->name, n_range);

			TEST_ASSERT_(n_range == expect[e].range_path &&
						 n_id    == expect[e].byid_path,
					"%s: measured %zu/%zu bytes (add_range/id-by-id), the "
					"handover brief cites %zu/%zu",
					c->name, n_range, n_id,
					expect[e].range_path, expect[e].byid_path);

			free(h_range);
			free(h_id);
			roaring64_bitmap_free(by_range);
			roaring64_bitmap_free(by_id);

			checked++;
		}
	}

	TEST_ASSERT_(checked == 2, "expected 2 path-dependent shapes, checked %d",
			checked);
}

// optimize() is necessary, and separately, not sufficient.
//
// An unoptimized bitmap serializes differently, which is rule 2. This asserts
// it on the shapes where run encoding wins, so that an encoder which forgets
// the call cannot pass by accident.
void test_effectsV3Roaring_optimizeIsNecessary(void) {
	int differed = 0;

	for(int i = 0; i < N_ROARING_CASES; i++) {
		const RoaringCase *c = ROARING_CASES + i;
		TEST_CASE(c->name);

		roaring64_bitmap_t *opt = _build(c, BUILD_ADD_RANGE, true);
		roaring64_bitmap_t *raw = _build(c, BUILD_ADD_RANGE, false);

		size_t n_opt = 0, n_raw = 0;
		char  *h_opt = _serialize_hex(opt, &n_opt);
		char  *h_raw = _serialize_hex(raw, &n_raw);

		// the optimized form is what the crate's bytes were captured from
		TEST_ASSERT_(strcmp(h_opt, c->hex_range_path) == 0,
				"%s: optimized bytes do not match the crate", c->name);

		if(strcmp(h_opt, h_raw) != 0) differed++;

		free(h_opt);
		free(h_raw);
		roaring64_bitmap_free(opt);
		roaring64_bitmap_free(raw);
	}

	TEST_ASSERT_(differed > 0,
			"run_optimize changed no shape's bytes; rule 2 would be vacuous "
			"and the corpus needs a shape where run encoding wins");

	TEST_MSG("run_optimize changed the bytes of %d of %d shapes",
			differed, N_ROARING_CASES);
}

// The collapse rule's constant 32 is only meaningful if a bitmap can actually
// be smaller than that, so the measured floor is worth pinning. The handover
// brief records 27 bytes measured, against the 30 the spec claims -- 30 is the
// array-container minimum, and a run container beats it.
void test_effectsV3Roaring_measuredFloor(void) {
	size_t      min      = (size_t)-1;
	const char *min_name = NULL;

	// measured from the library, not read back out of the table above
	for(int i = 0; i < N_ROARING_CASES; i++) {
		roaring64_bitmap_t *b = _build(ROARING_CASES + i, BUILD_ADD_RANGE, true);

		size_t n = 0;
		char  *h = _serialize_hex(b, &n);

		if(n < min) {
			min      = n;
			min_name = ROARING_CASES[i].name;
		}

		free(h);
		roaring64_bitmap_free(b);
	}

	TEST_ASSERT_(min == 27,
			"smallest serialized bitmap is %zu bytes (%s); the brief records 27",
			min, min_name);

	// this is what makes `range_bytes >= 32` a live threshold rather than one
	// no bitmap could ever undercut
	TEST_ASSERT_(min < 32,
			"no bitmap is smaller than the collapse rule's 32-byte threshold");
}

TEST_LIST = {
	{ "EffectsV3Roaring.addRangePath",   test_effectsV3Roaring_addRangePath   },
	{ "EffectsV3Roaring.idByIdPath",     test_effectsV3Roaring_idByIdPath     },
	{ "EffectsV3Roaring.pathIsNormative",
	  test_effectsV3Roaring_constructionPathIsNormative                       },
	{ "EffectsV3Roaring.optimizeNeeded", test_effectsV3Roaring_optimizeIsNecessary },
	{ "EffectsV3Roaring.measuredFloor",  test_effectsV3Roaring_measuredFloor  },
	{ NULL, NULL }
};
