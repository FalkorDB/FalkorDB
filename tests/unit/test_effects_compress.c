/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "src/util/rmalloc.h"
#include "src/util/crc32.h"
#include "src/effects/effects_compress.h"
#include "src/effects/effects.h"
#include "src/effects/effects_v3.h"
#include "src/configuration/config.h"

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

//------------------------------------------------------------------------------
// the wire bytes
//------------------------------------------------------------------------------

// the three header fields are LITTLE ENDIAN, pinned as bytes
//
// This is here because a round trip cannot detect a byte-order bug: swap the
// store and the load together and every round-trip test still passes, because
// the error is symmetric. It only shows up against the other engine, as a
// refused payload with a plausible-looking length in the log.
//
// Rust writes these with to_le_bytes, so the wire is defined as little endian
// rather than as native. On every target either engine builds for that is the
// same bytes a memcpy would produce, which is exactly why an accidental swap
// would go unnoticed locally.
void test_header_fields_are_little_endian() {
	size_t  records_len = 4000;                  // 0x00000FA0
	size_t  len;
	char   *p = _payload(records_len, true, &len);

	char *records = rm_malloc(records_len);
	memcpy(records, p + 2, records_len);
	uint32_t expect_crc = CRC32(records, records_len);

	TEST_ASSERT(EffectsV3_MaybeCompress(&p, &len, 64) == true);

	const unsigned char *b = (const unsigned char *)p;

	// uncompressed_length == 4000 == 0x00000FA0, low byte first
	TEST_ASSERT(b[2] == 0xA0);
	TEST_ASSERT(b[3] == 0x0F);
	TEST_ASSERT(b[4] == 0x00);
	TEST_ASSERT(b[5] == 0x00);

	// compressed_length is whatever is left after header and prefix
	size_t frame_len = len - EFFECTS_V3_HEADER_LEN - EFFECTS_V3_COMPRESSED_PREFIX;
	TEST_ASSERT(b[6] == (unsigned char)( frame_len        & 0xFF));
	TEST_ASSERT(b[7] == (unsigned char)((frame_len >> 8)  & 0xFF));
	TEST_ASSERT(b[8] == (unsigned char)((frame_len >> 16) & 0xFF));
	TEST_ASSERT(b[9] == (unsigned char)((frame_len >> 24) & 0xFF));

	// checksum of the PLAINTEXT, not the frame
	TEST_ASSERT(b[10] == (unsigned char)( expect_crc        & 0xFF));
	TEST_ASSERT(b[11] == (unsigned char)((expect_crc >> 8)  & 0xFF));
	TEST_ASSERT(b[12] == (unsigned char)((expect_crc >> 16) & 0xFF));
	TEST_ASSERT(b[13] == (unsigned char)((expect_crc >> 24) & 0xFF));

	// and the checksum really is over the plaintext: the frame's own bytes
	// hash to something else, so a reader that checksummed the frame would
	// disagree with this
	TEST_ASSERT(CRC32(p + EFFECTS_V3_HEADER_LEN + EFFECTS_V3_COMPRESSED_PREFIX,
				frame_len) != expect_crc);

	rm_free(records);
	rm_free(p);
}

// a v2 payload is never compressed, whatever the threshold says
//
// v2's header is ONE byte, not two, so a v2 payload treated as compressible
// would have its first record byte overwritten as a flags byte. The guard
// reads the payload's own version rather than trusting a caller, which is what
// lets this hook attach while EFFECTS_VERSION_EMIT is still 2 - today every
// payload reaching it is a v2 one.
void test_pre_v3_payload_is_never_compressed() {
	size_t  len;
	char   *p = _payload(4000, true, &len);

	for(int version = 0; version < 3; version++) {
		p[0] = (char)version;

		char   *before     = p;
		size_t  before_len = len;
		char    first_byte = p[1];

		TEST_ASSERT(EffectsV3_MaybeCompress(&p, &len, 1) == false);
		TEST_ASSERT(p == before);
		TEST_ASSERT(len == before_len);
		TEST_ASSERT(p[1] == first_byte);   // nothing overwritten
	}

	// and v3 with the same bytes and threshold does compress, so the refusals
	// above are the version and not something else
	p[0] = 3;
	p[1] = 0;
	TEST_ASSERT(EffectsV3_MaybeCompress(&p, &len, 1) == true);

	rm_free(p);
}

//------------------------------------------------------------------------------
// robustness
//------------------------------------------------------------------------------

// every field in the compressed header comes off the wire, so sweep them
//
// This is not a search for a specific bug; it is the test that makes the six
// refusal paths actually execute, which is what a sanitizer run needs. Each of
// them frees the decompression buffer on its way out, and a missed or repeated
// free there is invisible to a test that only checks the returned status.
//
// Two invariants hold for EVERY input, valid or not:
//
//   * the status is OK or one of the refusals - never anything else, and never
//     a crash. A truncated or corrupt payload arrives over a replication link
//     and is applied inline on the main thread.
//   * on any refusal '*plain' is NULL. A codec that returned a buffer
//     alongside a failure would leak it on every corrupt payload, and the
//     caller has no reason to look.
//
// When a mutation happens to produce a still-valid payload, the plaintext is
// checked rather than merely freed - a decoder that accepted corruption and
// returned the wrong bytes would otherwise pass this.
static void _check_refusal_is_clean
(
	const char *body,
	size_t body_len,
	const char *expect_plain,
	size_t expect_plain_len
) {
	char                   *plain = NULL;
	size_t                  pl    = 0;
	EffectsV3CompressFault  f     = EFFECTS_V3_COMPRESS_OK;

	EffectsV3Status st = EffectsV3_OpenCompressed(body, body_len, &plain, &pl, &f);

	TEST_ASSERT(st == EFFECTS_V3_OK       ||
	            st == EFFECTS_V3_TRUNCATED ||
	            st == EFFECTS_V3_MALFORMED);

	if(st == EFFECTS_V3_OK) {
		// a mutation can land on a still-valid payload; if it decoded, it must
		// have decoded correctly
		TEST_ASSERT(plain != NULL);
		TEST_ASSERT(pl == expect_plain_len);
		TEST_ASSERT(memcmp(plain, expect_plain, pl) == 0);
		rm_free(plain);
	} else {
		TEST_ASSERT(plain == NULL);
		TEST_ASSERT(pl == 0);
	}
}

void test_truncation_and_corruption_sweep() {
	size_t  records_len = 4000;
	size_t  len;
	char   *p = _payload(records_len, true, &len);

	char *records = rm_malloc(records_len);
	memcpy(records, p + 2, records_len);

	TEST_ASSERT(EffectsV3_MaybeCompress(&p, &len, 64) == true);

	const char *body     = p + EFFECTS_V3_HEADER_LEN;
	size_t      body_len = len - EFFECTS_V3_HEADER_LEN;

	// every truncation, including zero bytes
	for(size_t take = 0; take <= body_len; take++) {
		_check_refusal_is_clean(body, take, records, records_len);
	}

	// every byte of the body, flipped four ways. The twelve byte prefix is the
	// part that matters most - those are the lengths that drive the allocation
	// and the checksum that gates it - but the frame bytes are swept too, so
	// zstd's own rejection path frees correctly as well.
	static const unsigned char PATTERNS[] = { 0x00, 0xFF, 0x01, 0x80 };
	char *scratch = rm_malloc(body_len);

	for(size_t off = 0; off < body_len; off++) {
		for(size_t k = 0; k < sizeof(PATTERNS); k++) {
			memcpy(scratch, body, body_len);
			if(PATTERNS[k] == 0x00 || PATTERNS[k] == 0xFF) {
				scratch[off] = (char)PATTERNS[k];
			} else {
				scratch[off] = (char)((unsigned char)scratch[off] ^ PATTERNS[k]);
			}
			_check_refusal_is_clean(scratch, body_len, records, records_len);
		}
	}

	rm_free(scratch);
	rm_free(records);
	rm_free(p);
}

//------------------------------------------------------------------------------
// who owns the compressed bit
//------------------------------------------------------------------------------

// re-encoding a payload that ARRIVED compressed must not claim compressed
//
// The decoder carries the payload's flags byte onto the records it produces,
// and EffectsBuffer_TakeBody writes those flags back out on a re-encode, so a
// round trip reproduces the header it decoded. That is right for every bit
// except this one: the compressed bit describes THE BODY AS WRITTEN, and the
// body a re-encode writes is the plaintext the decoder inflated. Inheriting
// the bit emits a header claiming compressed over plaintext, and
// EffectsV3_MaybeCompress then correctly declines to re-compress because the
// bit is already set - which is what turns it from wasteful into corrupt. A
// reader would try to inflate plaintext and refuse the payload.
//
// THIS STATE IS UNREACHABLE TODAY - EffectsV3_Encode has no callers - so it is
// constructed by hand here rather than reached through a decode. Without that,
// the guard would be unfalsifiable until the conformance round trip becomes
// its first caller, and then it would be someone else's confusing failure.
//
// Compression is left OFF for this test so nothing can set the bit
// legitimately: the only way it can appear is by being inherited.
void test_reencode_does_not_inherit_the_compressed_flag() {
	Config_Option_set(Config_EFFECTS_VERSION, "3", NULL);
	Config_Option_set(Config_EFFECTS_COMPRESSION, "0", NULL);

	// exactly what a decode of a compressed payload leaves behind
	EffectsV3Records recs;
	recs.version = 3;
	recs.flags   = EFFECTS_V3_FLAG_COMPRESSED;
	recs.records = NULL;
	recs.n       = 0;

	EffectsBuffer *eb = EffectsBuffer_New();
	TEST_ASSERT(EffectsV3_Encode(&recs, eb) == true);

	size_t          len = 0;
	unsigned char  *out = EffectsBuffer_Buffer(eb, &len);

	TEST_ASSERT(out != NULL);
	TEST_ASSERT(len >= EFFECTS_V3_HEADER_LEN);
	TEST_ASSERT(out[0] == 3);                                        // version kept
	TEST_ASSERT((out[1] & EFFECTS_V3_FLAG_COMPRESSED) == 0);         // bit NOT inherited

	rm_free(out);
	EffectsBuffer_Free(eb);
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
	{ "header_fields_are_little_endian",       test_header_fields_are_little_endian},
	{ "pre_v3_payload_is_never_compressed",    test_pre_v3_payload_is_never_compressed},
	{ "truncation_and_corruption_sweep",       test_truncation_and_corruption_sweep},
	{ "reencode_does_not_inherit_the_compressed_flag", test_reencode_does_not_inherit_the_compressed_flag},
	{ NULL, NULL }
};
