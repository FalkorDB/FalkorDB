/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "src/util/rmalloc.h"
#include "src/util/crc32.h"
#include "src/effects/effects_compress.h"

#include <string.h>
#include <stdlib.h>

void setup() {
	Alloc_Reset();
}

#define TEST_INIT setup();
#include "acutest.h"

// build a v3 payload: two byte header followed by 'n' bytes of records
static char *_payload
(
	size_t n,           // record bytes
	bool compressible,  // repetitive, or pseudo-random
	size_t *len         // [output] total payload length
) {
	char *p = rm_malloc(2 + n);
	p[0] = 3;   // version
	p[1] = 0;   // flags

	// deterministic either way - a test that compresses random bytes must be
	// reproducible, so this is a fixed LCG rather than rand()
	uint32_t s = 0x12345678;
	for(size_t i = 0; i < n; i++) {
		s = s * 1103515245u + 12345u;
		p[2 + i] = compressible ? (char)('A' + (i % 7)) : (char)(s >> 24);
	}

	*len = 2 + n;
	return p;
}

//------------------------------------------------------------------------------
// CRC-32
//------------------------------------------------------------------------------

// the published check values for CRC-32/ISO-HDLC
//
// This exists so a wrong polynomial or a missed reflection fails here, locally,
// rather than as a checksum mismatch between a C replica and a Rust primary -
// where the symptom is a refused payload and a resync loop, and the cause is
// four layers away from it.
void test_crc32_known_answers() {
	TEST_ASSERT(CRC32("123456789", 9) == 0xCBF43926U);
	TEST_ASSERT(CRC32("", 0)           == 0x00000000U);
	TEST_ASSERT(CRC32("a", 1)          == 0xE8B7BE43U);
}

//------------------------------------------------------------------------------
// the worth-it boundary
//------------------------------------------------------------------------------

// a frame must be MORE than twelve bytes smaller than the records it replaces
//
// Twelve, because a compressed payload carries two u32 lengths and a u32
// checksum that an uncompressed one does not. An earlier reading of the format
// had this at eight, which silently turns a payload that saves nine bytes into
// a payload that grew.
void test_worth_it_boundary() {
	// one byte better than break-even is taken
	TEST_ASSERT(EffectsV3_CompressionWorthIt(87, 100) == true);

	// exactly break-even is refused: the compressed form is the same size and
	// strictly worse, because it costs both ends a zstd pass
	TEST_ASSERT(EffectsV3_CompressionWorthIt(88, 100) == false);

	// and anything worse
	TEST_ASSERT(EffectsV3_CompressionWorthIt(89, 100) == false);

	// a nine byte saving is not enough - this is the case that a prefix of 8
	// would wrongly accept
	TEST_ASSERT(EffectsV3_CompressionWorthIt(91, 100) == false);

	TEST_ASSERT(EFFECTS_V3_COMPRESSED_PREFIX == 12);
}

//------------------------------------------------------------------------------
// round trip
//------------------------------------------------------------------------------

void test_round_trip() {
	size_t len;
	char *p = _payload(4000, true, &len);

	size_t  orig_len = len;
	char   *orig     = rm_malloc(len);
	memcpy(orig, p, len);

	TEST_ASSERT(EffectsV3_MaybeCompress(&p, &len, 64) == true);
	TEST_ASSERT(len < orig_len);
	TEST_ASSERT((unsigned char)p[0] == 3);                                // version kept
	TEST_ASSERT(((unsigned char)p[1] & EFFECTS_V3_FLAG_COMPRESSED) != 0);  // flag set

	char                   *plain     = NULL;
	size_t                  plain_len = 0;
	EffectsV3CompressFault  fault;

	EffectsV3Status st = EffectsV3_OpenCompressed(p + 2, len - 2, &plain,
			&plain_len, &fault);

	TEST_ASSERT(st == EFFECTS_V3_OK);
	TEST_ASSERT(plain_len == orig_len - 2);
	TEST_ASSERT(plain != NULL);
	TEST_ASSERT(memcmp(plain, orig + 2, plain_len) == 0);

	rm_free(plain);
	rm_free(p);
	rm_free(orig);
}

// compressing twice produces a payload nothing can read: the second pass would
// swallow the first one's lengths and checksum as if they were records, and a
// reader inflates once
//
// Deliberately a refusal rather than an ASSERT - ASSERT compiles to nothing in
// release builds, which is exactly where this would matter, and it would make
// the property untestable in the builds these tests run in
void test_double_compression_refused() {
	size_t len;
	char *p = _payload(4000, true, &len);

	TEST_ASSERT(EffectsV3_MaybeCompress(&p, &len, 64) == true);

	char   *before     = p;
	size_t  before_len = len;

	TEST_ASSERT(EffectsV3_MaybeCompress(&p, &len, 64) == false);
	TEST_ASSERT(p == before);           // buffer not reallocated
	TEST_ASSERT(len == before_len);     // length untouched

	rm_free(p);

	// The assertions above do NOT on their own prove the guard exists, and
	// this is worth understanding before trusting them: an already-compressed
	// payload is incompressible, so with the guard deleted the second call is
	// refused anyway - by the worth-it test, one check further down. Removing
	// the guard entirely left the test above green.
	//
	// So pin the guard by itself: a payload whose flag is set but whose body
	// is still perfectly compressible. Only the flag can refuse this one.
	size_t  n = 4000;
	char   *q = _payload(n, true, &len);
	q[1] |= EFFECTS_V3_FLAG_COMPRESSED;

	before     = q;
	before_len = len;

	TEST_ASSERT(EffectsV3_MaybeCompress(&q, &len, 64) == false);
	TEST_ASSERT(q == before);
	TEST_ASSERT(len == before_len);

	rm_free(q);
}

//------------------------------------------------------------------------------
// when NOT to compress
//------------------------------------------------------------------------------

// EFFECTS_COMPRESSION is a byte threshold, not a boolean, and 0 disables it
void test_threshold_is_a_byte_count() {
	size_t len;
	char *p = _payload(4000, true, &len);

	char   *before     = p;
	size_t  before_len = len;

	// 0 disables compression entirely - the default
	TEST_ASSERT(EffectsV3_MaybeCompress(&p, &len, 0) == false);
	TEST_ASSERT(p == before && len == before_len);

	// the threshold is measured against the RECORD STREAM, not the whole
	// payload: 4000 record bytes sit under a threshold of 4001
	TEST_ASSERT(EffectsV3_MaybeCompress(&p, &len, 4001) == false);
	TEST_ASSERT(p == before && len == before_len);

	// and exactly at the threshold it compresses
	TEST_ASSERT(EffectsV3_MaybeCompress(&p, &len, 4000) == true);

	rm_free(p);
}

// zstd inflates an already-minimal buffer, so the smaller form has to win
void test_incompressible_stays_uncompressed() {
	size_t len;
	char *p = _payload(2000, false, &len);

	char   *before     = p;
	size_t  before_len = len;

	TEST_ASSERT(EffectsV3_MaybeCompress(&p, &len, 1) == false);
	TEST_ASSERT(p == before && len == before_len);

	rm_free(p);
}

//------------------------------------------------------------------------------
// refusals
//------------------------------------------------------------------------------

void test_refusals() {
	size_t len;
	char *p = _payload(4000, true, &len);
	TEST_ASSERT(EffectsV3_MaybeCompress(&p, &len, 64) == true);

	char                   *plain = NULL;
	size_t                  pl;
	EffectsV3CompressFault  f;
	EffectsV3Status         st;

	// ends inside the twelve byte prefix
	st = EffectsV3_OpenCompressed(p + 2, 11, &plain, &pl, &f);
	TEST_ASSERT(st == EFFECTS_V3_TRUNCATED);
	TEST_ASSERT(f == EFFECTS_V3_COMPRESS_SHORT_PREFIX);
	TEST_ASSERT(plain == NULL);

	// compressed_length outruns the bytes available
	st = EffectsV3_OpenCompressed(p + 2, len - 3, &plain, &pl, &f);
	TEST_ASSERT(st == EFFECTS_V3_TRUNCATED);
	TEST_ASSERT(f == EFFECTS_V3_COMPRESS_SHORT_FRAME);
	TEST_ASSERT(plain == NULL);

	// trailing bytes are an error, not tolerated padding: the header and the
	// payload disagree about where the frame ends, so nothing after it can be
	// trusted either. This is what compressed_length buys
	char *pad = rm_malloc(len + 1);
	memcpy(pad, p, len);
	pad[len] = 0x7F;
	st = EffectsV3_OpenCompressed(pad + 2, len - 2 + 1, &plain, &pl, &f);
	TEST_ASSERT(st == EFFECTS_V3_MALFORMED);
	TEST_ASSERT(f == EFFECTS_V3_COMPRESS_TRAILING);
	TEST_ASSERT(plain == NULL);
	rm_free(pad);

	// a corrupt checksum
	char *bad = rm_malloc(len);
	memcpy(bad, p, len);
	bad[2 + 8] ^= 0xFF;
	st = EffectsV3_OpenCompressed(bad + 2, len - 2, &plain, &pl, &f);
	TEST_ASSERT(st == EFFECTS_V3_MALFORMED);
	TEST_ASSERT(f == EFFECTS_V3_COMPRESS_CHECKSUM);
	TEST_ASSERT(plain == NULL);
	rm_free(bad);

	// a corrupt frame body
	char *bf = rm_malloc(len);
	memcpy(bf, p, len);
	bf[len - 5] ^= 0xFF;
	bf[len - 9] ^= 0xFF;
	st = EffectsV3_OpenCompressed(bf + 2, len - 2, &plain, &pl, &f);
	TEST_ASSERT(st == EFFECTS_V3_MALFORMED);
	TEST_ASSERT(plain == NULL);
	rm_free(bf);

	rm_free(p);
}

//------------------------------------------------------------------------------
// the ceiling
//------------------------------------------------------------------------------

// uncompressed_length is the decompress allocation CEILING, and it is applied
// by passing it as zstd's destination capacity
//
// READ THE SECOND ASSERTION BEFORE CHANGING THIS TEST. Asserting only that the
// payload is refused does NOT test the ceiling: with the ceiling removed, the
// frame still expands - eight megabytes of it - and the refusal then comes from
// the after-the-fact length check instead. The outcome is identical and the
// security property is gone. That is not hypothetical; this test passed against
// a deliberately broken ceiling until it checked the fault code.
//
// zstd's ratio on repetitive input is unbounded in practice, so a decoder that
// grows to fit whatever a frame expands to can be made to allocate gigabytes
// from a payload of a few hundred bytes. GRAPH.EFFECT runs inline on the main
// thread.
void test_declared_length_is_an_allocation_ceiling() {
	size_t  n     = 8u << 20;         // 8 MiB of zeros
	size_t  len   = 2 + n;
	char   *zeros = rm_calloc(1, len);
	zeros[0] = 3;
	zeros[1] = 0;

	TEST_ASSERT(EffectsV3_MaybeCompress(&zeros, &len, 64) == true);
	TEST_ASSERT(len < 4096);          // a few hundred bytes

	// claim the frame expands to only 100 bytes
	char *lie = rm_malloc(len);
	memcpy(lie, zeros, len);
	lie[2] = 100; lie[3] = 0; lie[4] = 0; lie[5] = 0;

	char                   *plain = NULL;
	size_t                  pl    = 0;
	EffectsV3CompressFault  f;

	EffectsV3Status st = EffectsV3_OpenCompressed(lie + 2, len - 2, &plain,
			&pl, &f);

	TEST_ASSERT(st == EFFECTS_V3_MALFORMED);
	TEST_ASSERT(plain == NULL);

	// THE MECHANISM: zstd refused for lack of room, which can only happen if
	// the declared length was the destination capacity
	TEST_ASSERT(f == EFFECTS_V3_COMPRESS_BAD_FRAME);

	rm_free(lie);
	rm_free(zeros);
}

// a frame that expands to LESS than declared satisfies the capacity and is
// still a disagreement between header and payload
//
// The pair matters: this one and the ceiling above fail on different inputs,
// which is why the implementation checks both rather than picking one
void test_short_expansion_is_still_a_mismatch() {
	size_t len;
	char *p = _payload(4000, true, &len);
	TEST_ASSERT(EffectsV3_MaybeCompress(&p, &len, 64) == true);

	// claim it expands to one byte MORE than it really does
	uint32_t declared = (uint32_t)p[2] | ((uint32_t)(unsigned char)p[3] << 8)
	                  | ((uint32_t)(unsigned char)p[4] << 16)
	                  | ((uint32_t)(unsigned char)p[5] << 24);
	declared += 1;
	p[2] = (char)( declared        & 0xFF);
	p[3] = (char)((declared >> 8)  & 0xFF);
	p[4] = (char)((declared >> 16) & 0xFF);
	p[5] = (char)((declared >> 24) & 0xFF);

	char                   *plain = NULL;
	size_t                  pl    = 0;
	EffectsV3CompressFault  f;

	EffectsV3Status st = EffectsV3_OpenCompressed(p + 2, len - 2, &plain,
			&pl, &f);

	TEST_ASSERT(st == EFFECTS_V3_MALFORMED);
	TEST_ASSERT(f == EFFECTS_V3_COMPRESS_LENGTH_MISMATCH);
	TEST_ASSERT(plain == NULL);

	rm_free(p);
}

TEST_LIST = {
	{ "crc32_known_answers",                   test_crc32_known_answers},
	{ "worth_it_boundary",                     test_worth_it_boundary},
	{ "round_trip",                            test_round_trip},
	{ "double_compression_refused",            test_double_compression_refused},
	{ "threshold_is_a_byte_count",             test_threshold_is_a_byte_count},
	{ "incompressible_stays_uncompressed",     test_incompressible_stays_uncompressed},
	{ "refusals",                              test_refusals},
	{ "declared_length_is_an_allocation_ceiling", test_declared_length_is_an_allocation_ceiling},
	{ "short_expansion_is_still_a_mismatch",   test_short_expansion_is_still_a_mismatch},
	{ NULL, NULL }
};
