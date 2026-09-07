/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

// The round-trip and truncation harnesses for effects v3.
//
// These are the tests that judge the codec rather than the corpus. They are
// written against src/effects/effects_v3.h and nothing else: EffectsV3Records
// is opaque here, because a round trip passes records through without ever
// inspecting a field. That is deliberate -- it means this harness cannot be
// quietly bent to agree with either half's idea of the record layout.
//
// GATING: the seam is declared on the wire-primitives branch but EffectsV3_Decode
// / _Encode / _RecordsFree are not defined yet, so linking this in now would
// break `make unit-tests` for everyone. The bodies are therefore compiled only
// when EFFECTS_V3_CODEC_READY is defined, which the reader's decode PR turns on
// (the intended home for the define is effects_v3.h, alongside the contract it
// describes). Until then these report as skips, not as passes.

#include "tests/unit/effects_v3_corpus.h"

#ifdef EFFECTS_V3_CODEC_READY
#include "src/effects/effects_v3.h"
#endif

#include "acutest.h"

#ifndef EFFECTS_V3_CODEC_READY

static void _skip(const char *what) {
	TEST_MSG("skipped: %s needs EffectsV3_Decode, which is declared in "
			"src/effects/effects_v3.h but not yet defined. Define "
			"EFFECTS_V3_CODEC_READY once the reader's decode lands.", what);
	TEST_CASE("pending the reader's decode");
	TEST_ASSERT(1);
}

void test_effectsV3_roundTrip(void)   { _skip("the round-trip harness");   }
void test_effectsV3_truncation(void)  { _skip("the truncation corpus");   }
void test_effectsV3_rejections(void)  { _skip("the rejection cases");     }

#else

//------------------------------------------------------------------------------
// round trip
//------------------------------------------------------------------------------

// decode a fixture, re-encode it, and require the original bytes back
//
// This is the whole point of a shared corpus. Each engine round-tripping
// against bytes it generated itself proves only self-consistency: a decoder can
// read every field one width narrow, write it back one width narrow, and pass
// all of its own tests. That is the failure that segfaulted C in
// AttributeSet_Update when it was handed a Rust buffer.
static void _round_trip(const EffectsV3CorpusEntry *e) {
	EffectsV3Fixture f = EffectsV3Corpus_Load(e->name);
	TEST_ASSERT_(f.buf != NULL, "%s: %s", e->name, f.err);
	if(f.buf == NULL) return;

	EffectsV3Records *records = NULL;
	EffectsV3Status   st      = EffectsV3_Decode((const char*)f.buf, f.len,
			&records);

	TEST_ASSERT_(st == EFFECTS_V3_OK, "%s: decode refused a corpus fixture: %s",
			e->name, EffectsV3Status_ToString(st));

	if(st == EFFECTS_V3_OK) {
		TEST_ASSERT_(records != NULL,
				"%s: decode returned OK with no records", e->name);

		EffectsBuffer *eb = EffectsBuffer_New();
		bool ok = EffectsV3_Encode(records, eb);
		TEST_ASSERT_(ok, "%s: encode failed on records that decoded cleanly",
				e->name);

		if(ok) {
			size_t n = 0;
			unsigned char *out = EffectsBuffer_Buffer(eb, &n);

			TEST_ASSERT_(n == f.len, "%s: re-encoded to %zu bytes, corpus has %zu",
					e->name, n, f.len);

			if(n == f.len) {
				// report the first divergent byte rather than just "differs" --
				// the offset is what identifies the field that was read wrong
				size_t at = 0;
				while(at < n && out[at] == f.buf[at]) at++;
				TEST_ASSERT_(at == n,
						"%s: re-encoded bytes differ at offset %zu "
						"(got 0x%02x, corpus has 0x%02x)",
						e->name, at, out[at], f.buf[at]);
			}

			rm_free(out);
		}

		EffectsBuffer_Free(eb);
		EffectsV3_RecordsFree(records);
	}

	EffectsV3Corpus_Free(&f);
}

void test_effectsV3_roundTrip(void) {
	for(size_t i = 0; i < EFFECTS_V3_CORPUS_COUNT; i++) {
		TEST_CASE(EFFECTS_V3_CORPUS[i].name);
		_round_trip(EFFECTS_V3_CORPUS + i);
	}
}

//------------------------------------------------------------------------------
// truncation
//------------------------------------------------------------------------------

// every proper prefix of every fixture, handed to the decoder
//
// Run this under a sanitizer -- `make unit-tests SAN=address` -- because the
// assertion that matters is the one the test cannot make itself: that no prefix
// causes a read past the end of the buffer. On plain master these reads go
// through fread_assert, whose ASSERT compiles to nothing in release, so a short
// read leaves the destination holding stack garbage and execution continues
// into it. v3 reads far more wire-derived counts than v2 (a segment count and
// two width codes per segment, a record count, an AttrSet width), so the
// exposure is larger, not smaller.
//
// WHAT A PREFIX IS ALLOWED TO DO. A v3 payload is `version · flags · record*`
// with no payload-level record count -- records are read until the bytes run
// out (confirmed against the corpus: seg_range goes straight from the flags
// byte to a u32 opcode, and payload_multi_record carries four records with no
// count preceding them). So a prefix cut exactly at a record boundary is a
// perfectly valid shorter payload and MUST decode cleanly. The invariant is
// therefore not "every prefix is rejected" -- that would be wrong, and a
// decoder could satisfy it by refusing everything. It is:
//
//   * the decoder terminates and returns a status, for every prefix
//   * a prefix it accepts must itself round-trip to those same bytes
//   * a prefix it rejects leaves nothing allocated and returns TRUNCATED or
//     MALFORMED, never OK-with-NULL
//
// The accepted-prefix case is the strong half: it means a decoder cannot pass
// by being permissive, because anything it accepts it must also reproduce.
void test_effectsV3_truncation(void) {
	size_t prefixes = 0;
	size_t accepted = 0;

	for(size_t i = 0; i < EFFECTS_V3_CORPUS_COUNT; i++) {
		const EffectsV3CorpusEntry *e = EFFECTS_V3_CORPUS + i;

		EffectsV3Fixture f = EffectsV3Corpus_Load(e->name);
		TEST_CASE(e->name);
		TEST_ASSERT_(f.buf != NULL, "%s: %s", e->name, f.err);
		if(f.buf == NULL) continue;

		for(size_t len = 0; len < f.len; len++) {
			EffectsV3Records *records = NULL;
			EffectsV3Status   st      =
				EffectsV3_Decode((const char*)f.buf, len, &records);

			prefixes++;

			if(st == EFFECTS_V3_OK) {
				accepted++;

				TEST_ASSERT_(records != NULL,
						"%s[..%zu]: OK with no records", e->name, len);

				// an accepted prefix must reproduce itself
				EffectsBuffer *eb = EffectsBuffer_New();
				if(EffectsV3_Encode(records, eb)) {
					size_t n = 0;
					unsigned char *out = EffectsBuffer_Buffer(eb, &n);
					TEST_ASSERT_(n == len && memcmp(out, f.buf, len) == 0,
							"%s[..%zu]: accepted as a valid payload but does "
							"not round-trip (re-encoded %zu bytes)",
							e->name, len, n);
					rm_free(out);
				} else {
					TEST_ASSERT_(false,
							"%s[..%zu]: accepted by decode, refused by encode",
							e->name, len);
				}
				EffectsBuffer_Free(eb);
				EffectsV3_RecordsFree(records);
			} else {
				TEST_ASSERT_(records == NULL,
						"%s[..%zu]: rejected as %s but left records allocated",
						e->name, len, EffectsV3Status_ToString(st));

				TEST_ASSERT_(st == EFFECTS_V3_TRUNCATED ||
							 st == EFFECTS_V3_MALFORMED,
						"%s[..%zu]: rejected as %s; a prefix of a valid v3 "
						"payload is either truncated or malformed",
						e->name, len, EffectsV3Status_ToString(st));
			}
		}

		EffectsV3Corpus_Free(&f);
	}

	TEST_MSG("%zu prefixes decoded, %zu accepted as valid shorter payloads",
			prefixes, accepted);
	TEST_ASSERT(prefixes > 0);
}

//------------------------------------------------------------------------------
// rejections that must be distinguishable
//------------------------------------------------------------------------------

// the status enum exists so a test can pin WHICH check fired, not merely that
// the buffer was refused. these mutate a known-good fixture one field at a time
void test_effectsV3_rejections(void) {
	// seg_range is the smallest fixture and its layout is pinned by the corpus
	// hash, so these offsets cannot drift silently:
	//
	//   [0]      version                          03
	//   [1]      flags                            00
	//   [2..5]   u32 opcode                       05 = DELETE_NODE
	//   [6..9]   u32 count                        08
	//   [10..11] u16 label count                  00      (LabelID is int, so
	//   [12..15] u32 segment count                01       a label is 4 bytes;
	//   [16]     segment header                   00       node.h:12)
	//   [17]     u8 range base                    64 = 100
	//   [18]     u8 range length                  08
	//
	// widths taken from C's own definitions: EffectsBuffer_AddCreateNodeEffect
	// writes label count as `ushort` and labels as sizeof(LabelID)
	// (effects.c:462,488), EntityID is GrB_Index (graph_entity.h:23).
	const size_t OFF_VERSION = 0;
	const size_t OFF_FLAGS   = 1;
	const size_t OFF_SEG_HDR = 16;

	struct {
		const char     *what;      // what is being mutated
		size_t          off;       // byte to change
		unsigned char   val;       // value to write
		EffectsV3Status expect;    // required status
	} cases[] = {
		{ "a version above what this build reads",
		  OFF_VERSION, 0x04, EFFECTS_V3_UNSUPPORTED_VERSION },

		{ "a version below v3 reaching the v3 decoder",
		  OFF_VERSION, 0x02, EFFECTS_V3_UNSUPPORTED_VERSION },

		{ "a flag bit outside the mask we understand",
		  OFF_FLAGS, 0x80, EFFECTS_V3_UNSUPPORTED_FLAGS },

		// the spec makes bits 6-7 of a segment header reserved and says they
		// MUST be rejected if set. a decoder that masks them off instead will
		// decode this buffer happily, which is why this asserts MALFORMED and
		// not merely "not OK"
		{ "reserved bits 6-7 of a segment header set",
		  OFF_SEG_HDR, 0xC0, EFFECTS_V3_MALFORMED },

		{ "reserved bit 6 alone",
		  OFF_SEG_HDR, 0x40, EFFECTS_V3_MALFORMED },

		{ "reserved bit 7 alone",
		  OFF_SEG_HDR, 0x80, EFFECTS_V3_MALFORMED },
	};

	for(size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
		TEST_CASE(cases[i].what);

		EffectsV3Fixture f = EffectsV3Corpus_Load("seg_range");
		TEST_ASSERT_(f.buf != NULL, "%s", f.err);
		if(f.buf == NULL) continue;

		TEST_ASSERT(f.len > cases[i].off);

		// a mutation that does not change the byte proves nothing
		TEST_ASSERT_(f.buf[cases[i].off] != cases[i].val,
				"offset %zu already holds 0x%02x", cases[i].off, cases[i].val);

		f.buf[cases[i].off] = cases[i].val;

		EffectsV3Records *records = NULL;
		EffectsV3Status   st      = EffectsV3_Decode((const char*)f.buf, f.len,
				&records);

		TEST_ASSERT_(st == cases[i].expect,
				"%s: got %s, expected %s", cases[i].what,
				EffectsV3Status_ToString(st),
				EffectsV3Status_ToString(cases[i].expect));

		TEST_ASSERT_(records == NULL, "%s: rejected but left records allocated",
				cases[i].what);

		EffectsV3Corpus_Free(&f);
	}
}

#endif  // EFFECTS_V3_CODEC_READY

// acutest has no notion of a skipped test, so the gated state says so in the
// test's NAME. Without this the run prints three [ OK ] lines and reads as
// three passing conformance tests, which is the opposite of true.
#ifdef EFFECTS_V3_CODEC_READY
#define V3_TEST(name) "EffectsV3." name
#else
#define V3_TEST(name) "EffectsV3." name " (SKIPPED: no codec yet)"
#endif

TEST_LIST = {
	{ V3_TEST("roundTrip"),   test_effectsV3_roundTrip  },
	{ V3_TEST("truncation"),  test_effectsV3_truncation },
	{ V3_TEST("rejections"),  test_effectsV3_rejections },
	{ NULL, NULL }
};
