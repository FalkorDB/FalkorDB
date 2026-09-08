/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// The segment encoder, checked against the conformance fixtures.
//
// Every test before this one was self-consistent: the builder was asserted
// against my own reading of the format, and the cost model against a table of
// sizes. This is the first check where the expected bytes came from somewhere
// else - the Rust encoder, via the fixture corpus - so it is the first thing
// that can catch both halves of my own work being wrong in the same direction.
//
// The fixtures are NOT ground truth. They are generated from the Rust encoder,
// so if it is wrong about the format they enshrine the error faithfully. Ground
// truth is C's own source for the primitives it reuses, and docs/effects-v3.md
// for the v3 blocks. A disagreement is settled against those and the fixture
// regenerated - not by editing the expectation here.
//
// The IdList bytes are extracted from each fixture rather than the whole
// payload being rebuilt: these cases are DELETE_NODE records, so the record
// framing around the list belongs to a later test, and asserting on the slice
// this code actually produces keeps a failure pointing at the encoder rather
// than at whatever else the record carries.

#include "src/effects/effects_v3_encode.h"
#include "src/effects/effects_v3_id_list.h"
#include "src/effects/effects_bytes.h"
#include "src/util/rmalloc.h"

void setup() {
	Alloc_Reset();
}

#define TEST_INIT setup();
#include "acutest.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// a DELETE_NODE fixture's bytes ahead of the IdList:
//   u8 version, u8 flags, u32 opcode, u32 count, u16 n_labels
// with n_labels 0 in every case here, so the list starts at a fixed offset
#define DELETE_NODE_PREFIX 12

typedef struct {
	const char *name;      // fixture case
	const char *hex;       // the whole payload, as the fixture holds it
	uint64_t    ids[16];   // the ids it describes, in order
	size_t      n_ids;
} SegCase;

// From .handover/fixtures/. Kept as the literal payload so a mismatch can be
// read against the file it came from.
static const SegCase SEG_CASES[] = {
	{
		"seg_range",
		"03000500000008000000000001000000006408",
		{ 100, 101, 102, 103, 104, 105, 106, 107 }, 8,
	},
	{
		"seg_range_single",
		"03000500000001000000000001000000000701",
		{ 7 }, 1,
	},
	{
		"seg_repeat",
		"03000500000006000000000001000000022a06",
		{ 42, 42, 42, 42, 42, 42 }, 6,
	},
	{
		"seg_mixed",
		"03000500000008000000000003000000000a04026303000501",
		{ 10, 11, 12, 13, 99, 99, 99, 5 }, 8,
	},
};

#define N_SEG_CASES ((int)(sizeof(SEG_CASES) / sizeof(SEG_CASES[0])))

static unsigned char *_unhex(const char *hex, size_t *n) {
	size_t len = strlen(hex) / 2;
	unsigned char *out = malloc(len);
	for(size_t i = 0; i < len; i++) {
		unsigned v = 0;
		sscanf(hex + 2 * i, "%2x", &v);
		out[i] = (unsigned char)v;
	}
	*n = len;
	return out;
}

static char *_hex(const unsigned char *b, size_t n) {
	char *out = malloc(2 * n + 1);
	for(size_t i = 0; i < n; i++) {
		sprintf(out + 2 * i, "%02x", b[i]);
	}
	out[2 * n] = '\0';
	return out;
}

// build the ids, encode the list, and compare against the fixture's slice
void test_effectsV3Encode_matchesFixtureSegments(void) {
	for(int i = 0; i < N_SEG_CASES; i++) {
		const SegCase *c = SEG_CASES + i;
		TEST_CASE(c->name);

		EffectsV3IdListBuilder *b = EffectsV3IdListBuilder_New();
		for(size_t k = 0; k < c->n_ids; k++) {
			EffectsV3IdListBuilder_Push(b, c->ids[k]);
		}

		EffectsV3IdList list = EffectsV3IdListBuilder_ToIdList(b);
		EffectsBytes *out = EffectsBytes_New(256);
		EffectsV3_EncodeIdList(&list, out);

		size_t got_n = EffectsBytes_Len(out);
		unsigned char *got = malloc(got_n);
		EffectsBytes_CopyInto(out, got);

		size_t want_all_n = 0;
		unsigned char *want_all = _unhex(c->hex, &want_all_n);
		const unsigned char *want = want_all + DELETE_NODE_PREFIX;
		size_t want_n = want_all_n - DELETE_NODE_PREFIX;

		char *got_s  = _hex(got, got_n);
		char *want_s = _hex(want, want_n);

		TEST_ASSERT_(got_n == want_n && memcmp(got, want, want_n) == 0,
				"%s: encoded IdList\n      got %s\n  expected %s",
				c->name, got_s, want_s);

		free(got_s);
		free(want_s);
		free(want_all);
		free(got);
		EffectsBytes_Free(out);
		EffectsV3IdListBuilder_FreeIdList(&list);
		EffectsV3IdListBuilder_Free(b);
	}
}

// THE COLLAPSE BOUNDARY, from the corpus rather than from my own arithmetic.
//
// These two cases are the same shape one id apart - ids 0, 1024, 2048, ... -
// and the corpus says 18 of them stay as 18 Range segments while 19 become one
// bitmap. So they pin the whole chain end to end against bytes I did not
// produce: the cost model's prediction, the collapse decision, the
// segmentation, and the encoding.
//
// This is a stronger check than the synthetic threshold test in
// test_effects_v3_run_cost.c, which asserts the decision against my own
// reading of the rule. Here the boundary itself came from the other engine.
//
// They also carry the only NON-ZERO width codes in the corpus: a base of 1024
// needs two bytes, so collapse_below's segments are header 0x04 - value width
// code 1 in bits 2-3. Every seg_* case has one-byte ids, so without these the
// fixtures cannot tell a correct width-code placement from a swapped one.
void test_effectsV3Encode_collapseBoundaryFromCorpus(void) {
	struct {
		const char *name;
		size_t      n_ids;   // ids are 1024 * i
		const char *hex;
	} cases[] = {
		{
			"collapse_below",  18,
			"03000500000012000000000012000000000001040004010400080104000c0104"
			"001001040014010400180104001c0104002001040024010400280104002c0104"
			"003001040034010400380104003c010400400104004401",
		},
		{
			"collapse_above",  19,
			"0300050000001300000000000100000001420000000100000000000000000000"
			"003a300000010000000000120010000000000000040008000c00100014001800"
			"1c002000240028002c003000340038003c004000440048",
		},
	};

	for(size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
		TEST_CASE(cases[i].name);

		EffectsV3IdListBuilder *b = EffectsV3IdListBuilder_New();
		for(size_t k = 0; k < cases[i].n_ids; k++) {
			EffectsV3IdListBuilder_Push(b, 1024 * (uint64_t)k);
		}

		EffectsV3IdList list = EffectsV3IdListBuilder_ToIdList(b);
		EffectsBytes *out = EffectsBytes_New(512);
		EffectsV3_EncodeIdList(&list, out);

		size_t got_n = EffectsBytes_Len(out);
		unsigned char *got = malloc(got_n);
		EffectsBytes_CopyInto(out, got);

		size_t want_all_n = 0;
		unsigned char *want_all = _unhex(cases[i].hex, &want_all_n);
		const unsigned char *want = want_all + DELETE_NODE_PREFIX;
		size_t want_n = want_all_n - DELETE_NODE_PREFIX;

		char *got_s  = _hex(got, got_n);
		char *want_s = _hex(want, want_n);

		TEST_ASSERT_(got_n == want_n && memcmp(got, want, want_n) == 0,
				"%s (%zu ids): the corpus and this encoder disagree\n"
				"      got %s\n  expected %s",
				cases[i].name, cases[i].n_ids, got_s, want_s);

		free(got_s);
		free(want_s);
		free(want_all);
		free(got);
		EffectsBytes_Free(out);
		EffectsV3IdListBuilder_FreeIdList(&list);
		EffectsV3IdListBuilder_Free(b);
	}
}

// DIRECTION AND WIDTH, against the corpus.
//
// These eight cases exist because the original corpus could not distinguish a
// correct header layout from several wrong ones: every seg_* case has one-byte
// ids, so both width fields are zero, and swapping the two width shifts left
// all of them passing. Now codes 2 and 3 appear in the value field and codes 1
// and 2 in the count field, which nothing touched before at all.
//
// The direction pair is an encoder test rather than a decoder one.
// dir_ascending and dir_descending are the same eight ids in opposite orders
// and their payloads differ in exactly two bytes: the header (0x00 -> 0x40) and
// the base (200 -> 207, lowest vs highest). That is the builder's descending
// rules expressed as bytes rather than as my reading of them.
//
// Each case also asserts the header carries the field its NAME claims, with
// dir_ascending's cleared bit as the control. Without that a case named
// value_width_8_bytes that quietly encoded as width 0 would sit in the table
// looking like coverage - the same defect as four passing seg_* cases proving
// nothing about widths.
void test_effectsV3Encode_directionAndWidthFromCorpus(void) {
	struct {
		const char *name;
		uint64_t    first;        // ids are first + step * k
		int64_t     step;
		size_t      n;
		uint8_t     want_header;  // the field this case is named for
		const char *hex;
	} cases[] = {
		{ "dir_ascending",          200, 1, 8, 0x00,
		  "0300050000000800000000000100000000c808" },
		{ "dir_descending",         207, -1, 8, 0x40,
		  "0300050000000800000000000100000040cf08" },
		{ "dir_descending_to_zero",   7, -1, 8, 0x40,
		  "03000500000008000000000001000000400708" },
		{ "value_width_4_bytes",  65536, 1, 8, 0x08,
		  "03000500000008000000000001000000080000010008" },
		{ "value_width_8_bytes", 4294967296ULL, 1, 8, 0x0c,
		  "030005000000080000000000010000000c000000000100000008" },
		{ "count_width_2_bytes",      0, 1, 300, 0x10,
		  "0300050000002c01000000000100000010002c01" },
		{ "count_width_4_bytes",      0, 1, 70000, 0x20,
		  "03000500000070110100000001000000200070110100" },
		{ "dir_descending_bitmap", 18432, -1024, 19, 0x41,
		  "0300050000001300000000000100000041420000000100000000000000000000"
		  "003a300000010000000000120010000000000000040008000c00100014001800"
		  "1c002000240028002c003000340038003c004000440048" },
	};

	for(size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
		TEST_CASE(cases[i].name);

		EffectsV3IdListBuilder *b = EffectsV3IdListBuilder_New();
		for(size_t k = 0; k < cases[i].n; k++) {
			EffectsV3IdListBuilder_Push(b,
					cases[i].first + (uint64_t)(cases[i].step * (int64_t)k));
		}

		EffectsV3IdList list = EffectsV3IdListBuilder_ToIdList(b);
		EffectsBytes *out = EffectsBytes_New(1024);
		EffectsV3_EncodeIdList(&list, out);

		size_t got_n = EffectsBytes_Len(out);
		unsigned char *got = malloc(got_n);
		EffectsBytes_CopyInto(out, got);

		size_t want_all_n = 0;
		unsigned char *want_all = _unhex(cases[i].hex, &want_all_n);
		const unsigned char *want = want_all + DELETE_NODE_PREFIX;
		size_t want_n = want_all_n - DELETE_NODE_PREFIX;

		char *got_s  = _hex(got, got_n);
		char *want_s = _hex(want, want_n);

		TEST_ASSERT_(got_n == want_n && memcmp(got, want, want_n) == 0,
				"%s: the corpus and this encoder disagree\n"
				"      got %s\n  expected %s",
				cases[i].name, got_s, want_s);

		// the case carries the field it is named for; got[4] is the first
		// segment's header, after the u32 segment count
		TEST_ASSERT_(got_n > 4 && got[4] == cases[i].want_header,
				"%s: header 0x%02x, expected 0x%02x - a case that does not "
				"carry the field it is named for is not coverage",
				cases[i].name, got_n > 4 ? got[4] : 0, cases[i].want_header);

		free(got_s);
		free(want_s);
		free(want_all);
		free(got);
		EffectsBytes_Free(out);
		EffectsV3IdListBuilder_FreeIdList(&list);
		EffectsV3IdListBuilder_Free(b);
	}
}

// a set has no direction, so the two bitmap forms differ in the header alone
//
// the same ids ascending and descending must produce byte-identical blobs, with
// only bit 6 distinguishing them. If the bitmap were built in push order rather
// than over the set, the two would diverge in the payload
void test_effectsV3Encode_bitmapDirectionIsOnlyTheHeaderBit(void) {
	unsigned char *bytes[2];
	size_t lens[2];

	for(int d = 0; d < 2; d++) {
		EffectsV3IdListBuilder *b = EffectsV3IdListBuilder_New();
		for(size_t k = 0; k < 19; k++) {
			// ascending 0, 1024, ... then the same set descending
			uint64_t id = d ? (18432 - 1024 * (uint64_t)k) : (1024 * (uint64_t)k);
			EffectsV3IdListBuilder_Push(b, id);
		}

		EffectsV3IdList list = EffectsV3IdListBuilder_ToIdList(b);
		EffectsBytes *out = EffectsBytes_New(512);
		EffectsV3_EncodeIdList(&list, out);
		lens[d] = EffectsBytes_Len(out);
		bytes[d] = malloc(lens[d]);
		EffectsBytes_CopyInto(out, bytes[d]);

		EffectsBytes_Free(out);
		EffectsV3IdListBuilder_FreeIdList(&list);
		EffectsV3IdListBuilder_Free(b);
	}

	TEST_ASSERT_(lens[0] == lens[1],
			"the two directions encode to %zu and %zu bytes; a set has no "
			"direction, so only the header bit may differ",
			lens[0], lens[1]);

	TEST_ASSERT_(bytes[0][4] == 0x01 && bytes[1][4] == 0x41,
			"expected headers 0x01 ascending and 0x41 descending, got 0x%02x "
			"and 0x%02x", bytes[0][4], bytes[1][4]);

	// everything except that one header byte must match
	bytes[1][4] = bytes[0][4];
	TEST_ASSERT_(memcmp(bytes[0], bytes[1], lens[0]) == 0,
			"the blobs differ beyond the header bit - the bitmap is being "
			"built in push order rather than over the id set");

	free(bytes[0]);
	free(bytes[1]);
}

// the header byte's fields, pinned individually
//
// A width or a shift that is wrong by one still produces a plausible byte, and
// the fixtures above only cover the widths their ids happen to need - all of
// which are one byte. These reach the wider codes.
void test_effectsV3Encode_headerFields(void) {
	struct {
		const char *what;
		uint64_t    ids[4];
		size_t      n_ids;
		uint8_t     want_header;
	} cases[] = {
		// kind 0, both widths 1 byte
		{ "range, 1-byte base and len",   { 5, 6 },       2, 0x00 },
		// base needs 2 bytes -> value width code 1, in bits 2-3
		{ "range, 2-byte base",           { 300, 301 },   2, 0x04 },
		// base needs 4 bytes -> value width code 2
		{ "range, 4-byte base",           { 70000, 70001 }, 2, 0x08 },
		// a descending pair: kind 0, bit 6 set
		{ "descending range",             { 6, 5 },       2, 0x40 },
		// kind 2, no direction ever
		{ "repeat",                       { 9, 9 },       2, 0x02 },
	};

	for(size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
		TEST_CASE(cases[i].what);

		EffectsV3IdListBuilder *b = EffectsV3IdListBuilder_New();
		for(size_t k = 0; k < cases[i].n_ids; k++) {
			EffectsV3IdListBuilder_Push(b, cases[i].ids[k]);
		}

		EffectsV3IdList list = EffectsV3IdListBuilder_ToIdList(b);
		EffectsBytes *out = EffectsBytes_New(64);
		EffectsV3_EncodeIdList(&list, out);

		size_t n = EffectsBytes_Len(out);
		unsigned char *got = malloc(n);
		EffectsBytes_CopyInto(out, got);

		// [0..4) is the u32 segment count; the header byte follows it
		TEST_ASSERT_(got[4] == cases[i].want_header,
				"%s: header byte 0x%02x, expected 0x%02x",
				cases[i].what, got[4], cases[i].want_header);

		TEST_ASSERT_((got[4] & EFFECTS_V3_SEG_RESERVED) == 0,
				"%s: bit 7 is reserved and must be zero", cases[i].what);

		free(got);
		EffectsBytes_Free(out);
		EffectsV3IdListBuilder_FreeIdList(&list);
		EffectsV3IdListBuilder_Free(b);
	}
}

// a width WIDER than the value needs is written back as it stands
//
// This is the re-encode path, and it is the reason the shared segment struct
// carries the widths as fields rather than deriving them. A freshly built
// segment gets the narrowest width that holds each value; a DECODED one
// carries whatever its peer chose, and a peer may legitimately write a value
// wider than it needs. Narrowing on the way back out produces different bytes
// for the same ids, which fails a round trip and looks like a decoder bug.
//
// Built here by hand rather than through the builder, because the builder
// cannot produce this - which is exactly why it needs its own test.
void test_effectsV3Encode_observedWidthsArePreserved(void) {
	// base 5 and len 3 both fit in one byte, but say four
	EffectsV3Segment s = {
		.kind        = EFFECTS_V3_SEG_RANGE,
		.descending  = false,
		.value_width = 4,
		.count_width = 4,
		.range       = { .base = 5, .len = 3 },
	};
	EffectsV3IdList l = { .segments = &s, .n = 1 };

	EffectsBytes *out = EffectsBytes_New(64);
	EffectsV3_EncodeIdList(&l, out);

	size_t n = EffectsBytes_Len(out);
	unsigned char *got = malloc(n);
	EffectsBytes_CopyInto(out, got);

	// u32 count, header, then two 4-byte values
	TEST_ASSERT_(n == 4 + 1 + 4 + 4,
			"expected 13 bytes for a segment declaring 4-byte fields, got %zu "
			"- the widths were recomputed rather than preserved", n);

	// value width code 2 in bits 2-3, count width code 2 in bits 4-5
	TEST_ASSERT_(got[4] == 0x28,
			"expected header 0x28 (both widths code 2), got 0x%02x", got[4]);

	unsigned char want[] = { 0x01,0x00,0x00,0x00, 0x28,
	                         0x05,0x00,0x00,0x00, 0x03,0x00,0x00,0x00 };
	TEST_ASSERT_(memcmp(got, want, sizeof(want)) == 0,
			"the declared widths must be written verbatim");

	free(got);
	EffectsBytes_Free(out);
}

// values are written little-endian regardless of the host
//
// v2 writes its records by copying packed structs, which is right only while
// every engine is little-endian. These widths are chosen per value, so there is
// no struct to copy and the byte order has to be stated
void test_effectsV3Encode_littleEndian(void) {
	EffectsBytes *out = EffectsBytes_New(64);
	EffectsV3_WriteUint(out, 0x0102030405060708ULL, 8);

	unsigned char got[8];
	EffectsBytes_CopyInto(out, got);

	unsigned char want[8] = { 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01 };
	TEST_ASSERT_(memcmp(got, want, 8) == 0,
			"expected little-endian 0807060504030201, got "
			"%02x%02x%02x%02x%02x%02x%02x%02x",
			got[0], got[1], got[2], got[3], got[4], got[5], got[6], got[7]);

	EffectsBytes_Free(out);
}

// the segment count is stated, and so is every segment's own length
//
// either could be inferred from the record's id count. Both are on the wire so
// a segment list is well-formed on its own: inferring makes a truncated list
// indistinguishable from a complete one
void test_effectsV3Encode_countsAreStated(void) {
	{
		EffectsV3IdListBuilder *b = EffectsV3IdListBuilder_New();
		uint64_t ids[] = { 10, 11, 12, 13, 99, 99, 99, 5 };
		for(size_t i = 0; i < 8; i++) {
			EffectsV3IdListBuilder_Push(b, ids[i]);
		}

		EffectsV3IdList list = EffectsV3IdListBuilder_ToIdList(b);
		EffectsBytes *out = EffectsBytes_New(64);
		EffectsV3_EncodeIdList(&list, out);

		size_t n = EffectsBytes_Len(out);
		unsigned char *got = malloc(n);
		EffectsBytes_CopyInto(out, got);

		uint32_t n_segs = (uint32_t)got[0] | ((uint32_t)got[1] << 8)
			| ((uint32_t)got[2] << 16) | ((uint32_t)got[3] << 24);

		TEST_ASSERT_(n_segs == 3,
				"expected a stated segment count of 3, read %u", n_segs);
		TEST_ASSERT_(n_segs == EffectsV3IdListBuilder_SegmentCount(b),
				"the stated count must equal the segments written");

		free(got);
		EffectsBytes_Free(out);
		EffectsV3IdListBuilder_FreeIdList(&list);
		EffectsV3IdListBuilder_Free(b);
	}
}

TEST_LIST = {
	{ "EffectsV3Encode:matchesFixtureSegments",
		test_effectsV3Encode_matchesFixtureSegments },
	{ "EffectsV3Encode:collapseBoundaryFromCorpus",
		test_effectsV3Encode_collapseBoundaryFromCorpus },
	{ "EffectsV3Encode:directionAndWidthFromCorpus",
		test_effectsV3Encode_directionAndWidthFromCorpus },
	{ "EffectsV3Encode:bitmapDirectionIsOnlyTheHeaderBit",
		test_effectsV3Encode_bitmapDirectionIsOnlyTheHeaderBit },
	{ "EffectsV3Encode:headerFields",
		test_effectsV3Encode_headerFields },
	{ "EffectsV3Encode:observedWidthsArePreserved",
		test_effectsV3Encode_observedWidthsArePreserved },
	{ "EffectsV3Encode:littleEndian",
		test_effectsV3Encode_littleEndian },
	{ "EffectsV3Encode:countsAreStated",
		test_effectsV3Encode_countsAreStated },
	{ NULL, NULL }
};
