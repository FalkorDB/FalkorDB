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
// GATING. The seam is declared on the wire-primitives branch, but linking a
// harness against an undefined entry point would break `make unit-tests` for
// everyone, so each test compiles only once the half it needs exists:
//
//   EFFECTS_V3_DECODE_READY   the reader's decode PR defines it
//   EFFECTS_V3_ENCODE_READY   the writer's encode PR defines it
//
// Both live in effects_v3.h alongside the contract they describe, so they flip
// exactly when the implementations do. One flag with two owners could not work:
// defining it on decode alone would link the round trip against an undefined
// EffectsV3_Encode, which is the failure this gate exists to prevent.
//
// The truncation corpus and the rejection cases need only decode, so they land
// a PR earlier than the round trip. Truncation's one encode-dependent clause --
// that an ACCEPTED prefix re-encodes to itself -- is gated separately, and the
// test says so when it is running without that half.

#include "tests/unit/effects_v3_corpus.h"

// Unconditionally, and BEFORE the flags are tested: effects_v3.h is where
// EFFECTS_V3_DECODE_READY and EFFECTS_V3_ENCODE_READY are defined, so gating
// this include on one of them would mean the include never happens and every
// test here stayed skipped forever -- including after the codec landed, which
// is the one outcome the gate must not produce.
#include "src/effects/effects_v3.h"

#if defined(EFFECTS_V3_DECODE_READY) && defined(EFFECTS_V3_ENCODE_READY)
#define EFFECTS_V3_CODEC_READY 1
#endif

// The decoder allocates through rm_malloc, which calls the RedisModule_Alloc
// function pointer -- NULL until Alloc_Reset() points it at malloc. Without
// this every call into the codec segfaults on its first allocation, which is
// exactly what happened the first time this ran against the real decoder.
// Every other suite in tests/unit does the same; mine did not.
#include "src/util/rmalloc.h"
#include "src/globals.h"
#include "src/util/thpool/pool.h"

static void setup(void) {
	Alloc_Reset();

	// The decoder reaches Globals_Get_StringPool() through SI_InternStringVal
	// for any interned string, and StringPool_rent guards a NULL pool with
	// ASSERT -- which compiles to nothing in release, so it dereferences NULL
	// instead. The server calls Globals_Init() at module load; a unit test has
	// to do it itself. Once per process: Globals_Init asserts it is called
	// once, and that assert is also empty in release.
	// Globals_Init sizes its CommandCtx table from ThreadPool_ThreadCount(),
	// which dereferences the pool, so the pool has to exist first. That this
	// is the minimum to decode a byte buffer is worth noting: effects_v3.h
	// calls decode "a pure byte-to-value transformation", and it is pure of
	// any GRAPH, but not of module-global state.
	static bool globals_ready = false;
	if(!globals_ready) {
		ThreadPool_Init();
		ThreadPool_CreatePool(1, 64);
		Globals_Init();
		globals_ready = true;
	}
}

#define TEST_INIT setup();
#include "acutest.h"

// a macro rather than a function so that an unused skip cannot warn
#define V3_SKIP(what, needs)                                                  \
	do {                                                                      \
		TEST_CASE("pending " needs);                                          \
		TEST_MSG("skipped: %s needs %s, declared in src/effects/effects_v3.h " \
				"but not yet defined", what, needs);                          \
		TEST_ASSERT(1);                                                       \
	} while(0)

#ifndef EFFECTS_V3_DECODE_READY

void test_effectsV3_roundTrip(void)  { V3_SKIP("the round-trip harness", "EffectsV3_Decode"); }
void test_effectsV3_truncation(void) { V3_SKIP("the truncation corpus",  "EffectsV3_Decode"); }
void test_effectsV3_rejections(void) { V3_SKIP("the rejection cases",    "EffectsV3_Decode"); }
void test_effectsV3_handBuiltRejections(void) { V3_SKIP("the hand-built rejections", "EffectsV3_Decode"); }
void test_effectsV3_decodesEveryFixture(void) { V3_SKIP("decoding every fixture", "EffectsV3_Decode"); }

#else

//------------------------------------------------------------------------------
// every fixture decodes
//------------------------------------------------------------------------------

// The gap this closes: until encode lands the round trip does not compile, and
// it was the ONLY test that decoded a corpus payload at its full length. So a
// record could stop decoding entirely and this suite would stay green --
// truncation feeds the decoder proper prefixes only, and the rejection cases
// feed it deliberately-broken ones. Nothing asserted the corpus decodes.
//
// It also replaces a list. Records 11-14 used to report
// EFFECTS_V3_UNIMPLEMENTED, and were named in a list that was REQUIRED to
// report it so that the day they decoded, the test would fail and say to remove
// them. They decode now -- EFFECTS_V3_UNIMPLEMENTED appears nowhere in
// effects_v3_decode.c -- so the list is gone and every case is held to the same
// bar. That is the ratchet finishing, not being switched off: a future
// unimplemented record needs a new list rather than a revived one.
void test_effectsV3_decodesEveryFixture(void) {
	for(size_t i = 0; i < EFFECTS_V3_CORPUS_COUNT; i++) {
		const EffectsV3CorpusEntry *e = EFFECTS_V3_CORPUS + i;

		EffectsV3Fixture f = EffectsV3Corpus_Load(e->name);
		TEST_CASE(e->name);
		TEST_CHECK_(f.buf != NULL, "%s: %s", e->name, f.err);
		if(f.buf == NULL) continue;

		EffectsV3Records *records = NULL;
		EffectsV3Status   st      = EffectsV3_Decode((const char*)f.buf, f.len,
				&records);

		TEST_CHECK_(st == EFFECTS_V3_OK,
				"%s: a corpus fixture failed to decode: %s. Every case here is "
				"a payload a conforming encoder produced, so a refusal is this "
				"decoder's defect and not the corpus's",
				e->name, EffectsV3Status_ToString(st));

		if(st == EFFECTS_V3_OK) {
			TEST_CHECK_(records != NULL, "%s: decoded OK with no records",
					e->name);
			EffectsV3_RecordsFree(records);
		} else {
			TEST_CHECK_(records == NULL,
					"%s: refused as %s but left records allocated",
					e->name, EffectsV3Status_ToString(st));
		}

		EffectsV3Corpus_Free(&f);
	}
}

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
#ifdef EFFECTS_V3_CODEC_READY

//------------------------------------------------------------------------------
// cases C decodes correctly and cannot re-encode
//------------------------------------------------------------------------------

// A SECOND LIST, on a different axis from the one that just retired. That list
// meant "the decoder does not know this record". This one means "the decoder
// knows it perfectly and the encoder cannot reproduce it". Collapsing them into
// one would make "why is this still here" unanswerable, because they shrink for
// different reasons: the first as C implements records, this one as C's core
// types gain capability.
//
// The case: value_string_interior_nul carries TWO strings with interior NULs,
// and the whole payload is 59 bytes where C re-encodes 56. Measured rather than
// relayed -- the figure here was 2 until the assertion fired and made me look:
//
//   row 0  "a\0b"  wire length 4, bytes 61 00 62 00
//                  strlen is 1, so C emits 2 bytes       -2
//   row 1  "\0"    wire length 2, bytes 00 00
//                  strlen is 0, so C emits 1 byte        -1
//
// 59 - 3 = 56. The second value is the one every summary of this bug omits: a
// string that is nothing but a NUL still loses a byte, so a fix that only
// handled embedded NULs in the middle of a string would leave this case short.
//
// C decodes both correctly, then re-encodes through EffectsBuffer_WriteString,
// which takes its length from strlen. Not a bug in the effects code:
//
//   * SIValue holds a bare char * with no length (value.h), so giving the
//     encoder a length to use is a core-type change, not an effects one
//   * C cannot produce an interior NUL in the first place -- it does not
//     implement \uXXXX -- so no C-originated write reaches this shape
//   * EffectsBuffer_WriteString is shared with v2 across seven call sites, so
//     editing it moves shipped v2 bytes, and versioning it means a second
//     SIValue codec, which is how the Rust side acquired a string-pool bug
//
// The live consequence, measured on a running C module rather than reasoned
// about: the payload is ACCEPTED, `n.a = 'a'` returns TRUE for a value that was
// "a\0b", and size(n.a) is 1. No refusal, no divergence guard, silent. That is
// with Dvir as a core-type question. This assertion documents it; it does not
// fix it.
//
// ASSERTED RATHER THAN SKIPPED, which is the point. A skip records that C does
// not round-trip this. Asserting the expected short re-encode records WHAT C
// DOES INSTEAD, and fails the day that changes -- so if SIValue ever carries a
// length, this test says to delete the entry rather than silently passing.
typedef struct {
	const char *name;      // case name
	size_t      encodes_to;// bytes C's encoder actually produces
	const char *why;       // why it cannot match, for the failure message
} EffectsV3EncodeDivergentCase;

static const EffectsV3EncodeDivergentCase EFFECTS_V3_ENCODE_DIVERGENT_CASES[] = {
	{ "value_string_interior_nul", 56,
	  "EffectsBuffer_WriteString takes its length from strlen and SIValue has "
	  "no length to take instead. Two values are affected: \"a\\0b\" loses 2 "
	  "bytes and \"\\0\" loses 1, so 59 becomes 56" },
};

#define EFFECTS_V3_ENCODE_DIVERGENT_COUNT              \
	(sizeof(EFFECTS_V3_ENCODE_DIVERGENT_CASES) /       \
	 sizeof(EFFECTS_V3_ENCODE_DIVERGENT_CASES[0]))

// returns the expected short length, or 0 if the case must round-trip exactly
static size_t _expected_short_encode
(
	const char *name  // case name
) {
	for(size_t i = 0; i < EFFECTS_V3_ENCODE_DIVERGENT_COUNT; i++) {
		if(strcmp(EFFECTS_V3_ENCODE_DIVERGENT_CASES[i].name, name) == 0) {
			return EFFECTS_V3_ENCODE_DIVERGENT_CASES[i].encodes_to;
		}
	}

	return 0;
}

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

			size_t expect_short = _expected_short_encode(e->name);

			if(expect_short != 0) {
				// a known divergence: assert exactly what C does, so the day
				// it stops doing it this fails and says to remove the entry
				TEST_ASSERT_(n == expect_short,
						"%s: re-encoded to %zu bytes; this case is listed as "
						"diverging at %zu. If C now reproduces the corpus's "
						"%zu bytes, delete it from "
						"EFFECTS_V3_ENCODE_DIVERGENT_CASES so it is held to "
						"the round trip like every other case. %s",
						e->name, n, expect_short, f.len,
						EFFECTS_V3_ENCODE_DIVERGENT_CASES[0].why);

				rm_free(out);
				EffectsBuffer_Free(eb);
				EffectsV3_RecordsFree(records);
				EffectsV3Corpus_Free(&f);
				return;
			}

			// length before bytes, so a truncating encoder cannot pass on a
			// prefix match
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

#else

void test_effectsV3_roundTrip(void) {
	V3_SKIP("the round-trip harness", "EffectsV3_Encode");
}

#endif  // EFFECTS_V3_CODEC_READY

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
// Per-prefix checks use TEST_CHECK_, not TEST_ASSERT_. acutest's TEST_ASSERT_
// longjmps out of the whole test on the first failure, which for a sweep of
// ~1,700 prefixes means one bad prefix hides every later one -- and it is how a
// missing string-pool init masqueraded as a single crashing prefix in this very
// test. A sweep has to report all of its failures to be a sweep.
void test_effectsV3_truncation(void) {
	size_t prefixes = 0;
	size_t accepted = 0;

	for(size_t i = 0; i < EFFECTS_V3_CORPUS_COUNT; i++) {
		const EffectsV3CorpusEntry *e = EFFECTS_V3_CORPUS + i;

		EffectsV3Fixture f = EffectsV3Corpus_Load(e->name);
		TEST_CASE(e->name);
		TEST_CHECK_(f.buf != NULL, "%s: %s", e->name, f.err);
		if(f.buf == NULL) continue;

			for(size_t len = 0; len < f.len; len++) {
			EffectsV3Records *records = NULL;
			EffectsV3Status   st      =
				EffectsV3_Decode((const char*)f.buf, len, &records);

			prefixes++;

			if(st == EFFECTS_V3_OK) {
				accepted++;

				TEST_CHECK_(records != NULL,
						"%s[..%zu]: OK with no records", e->name, len);

#ifdef EFFECTS_V3_ENCODE_READY
				// an accepted prefix must reproduce itself -- the clause that
				// stops a permissive decoder from passing, and the only part of
				// this test that needs the encoder
				EffectsBuffer *eb = EffectsBuffer_New();
				if(EffectsV3_Encode(records, eb)) {
					size_t n = 0;
					unsigned char *out = EffectsBuffer_Buffer(eb, &n);
					TEST_CHECK_(n == len && memcmp(out, f.buf, len) == 0,
							"%s[..%zu]: accepted as a valid payload but does "
							"not round-trip (re-encoded %zu bytes)",
							e->name, len, n);
					rm_free(out);
				} else {
					TEST_CHECK_(false,
							"%s[..%zu]: accepted by decode, refused by encode",
							e->name, len);
				}
				EffectsBuffer_Free(eb);
#endif
				EffectsV3_RecordsFree(records);
			} else {
				TEST_CHECK_(records == NULL,
						"%s[..%zu]: rejected as %s but left records allocated",
						e->name, len, EffectsV3Status_ToString(st));

				// UNIMPLEMENTED is legitimate here: a prefix long enough to
				// carry a complete records 11-14 opcode is refused for that
				// reason and not for its length
				TEST_CHECK_(st == EFFECTS_V3_TRUNCATED ||
							 st == EFFECTS_V3_MALFORMED ||
							 st == EFFECTS_V3_UNIMPLEMENTED,
						"%s[..%zu]: rejected as %s; a prefix of a valid v3 "
						"payload is truncated, malformed, or a record this "
						"build does not implement",
						e->name, len, EffectsV3Status_ToString(st));
			}
		}

		EffectsV3Corpus_Free(&f);
	}

	TEST_MSG("%zu prefixes decoded, %zu accepted as valid shorter payloads%s",
			prefixes, accepted,
#ifdef EFFECTS_V3_ENCODE_READY
			" and re-encoded"
#else
			" (re-encode check off: no encoder yet)"
#endif
			);
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

	// seg_repeat has the same shape, so its header is at the same offset:
	// 2 preamble + 4 opcode + 4 count + 2 label count + 4 segment count.
	//
	// ONLY BIT 7 IS RESERVED. Bit 6 is `descending`, added in 08dca4a1e after
	// this corpus was cut (graph/src/effects/v3/id_list.rs:518-519,
	// SEG_DESCENDING = 0b0100_0000, SEG_RESERVED = 0b1000_0000). A descending
	// Range reads its base as the first and HIGHEST id, and Ascending's blob is
	// a set, so the two directions differ in exactly that bit. Setting bit 6 on
	// a Range or an Ascending is therefore legal and must NOT be asserted as
	// malformed -- there are no descending fixtures yet, so this corpus cannot
	// exercise the accepting side of that rule at all.
	//
	// Repeat is the exception: one id however many times has no direction, so
	// the bit is rejected there rather than ignored (id_list.rs:721,
	// `SEG_KIND_REPEAT if descending => Err(...)`). That one needs no fixture,
	// which is why it is pinned here.
	struct {
		const char     *fixture;   // case to mutate
		const char     *what;      // what is being mutated
		size_t          off;       // byte to change
		unsigned char   val;       // value to write
		EffectsV3Status expect;    // required status
	} cases[] = {
		{ "seg_range", "a version above what this build reads",
		  OFF_VERSION, 0x04, EFFECTS_V3_UNSUPPORTED_VERSION },

		{ "seg_range", "a version below v3 reaching the v3 decoder",
		  OFF_VERSION, 0x02, EFFECTS_V3_UNSUPPORTED_VERSION },

		{ "seg_range", "a flag bit outside the mask we understand",
		  OFF_FLAGS, 0x80, EFFECTS_V3_UNSUPPORTED_FLAGS },

		// bit 7 is reserved and MUST be rejected if set. a decoder that masks
		// it off instead decodes these happily, which is why this asserts
		// MALFORMED and not merely "not OK"
		{ "seg_range", "reserved bit 7 set on a Range header",
		  OFF_SEG_HDR, 0x80, EFFECTS_V3_MALFORMED },

		{ "seg_repeat", "reserved bit 7 set on a Repeat header",
		  OFF_SEG_HDR, 0x82, EFFECTS_V3_MALFORMED },

		// the descending bit on a kind that has no direction
		{ "seg_repeat", "the descending bit on a Repeat",
		  OFF_SEG_HDR, 0x42, EFFECTS_V3_MALFORMED },
	};

	for(size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
		TEST_CASE(cases[i].what);

		EffectsV3Fixture f = EffectsV3Corpus_Load(cases[i].fixture);
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

//------------------------------------------------------------------------------
// hand-built payloads: shapes no fixture should carry
//------------------------------------------------------------------------------

// A deliberately-invalid payload needs no fixture. Generating one would put a
// shape into the corpus that no conforming encoder produces, and the corpus is
// the record of what the engines AGREE on -- so these are built here, byte by
// byte, the way the mutation cases above are.
//
// ZERO-COUNT RECORDS ARE ILLEGAL, and this is the one case where both engines
// accept today what they have agreed to refuse. Measured rather than assumed:
//
//   * Rust's read_ids decodes n_segments segments and then asserts len ==
//     count, so n_segments = 0 with count = 0 never calls Segment::decode and
//     returns an empty list
//   * C's _ReadIdList mirrors it exactly -- effects_v3_decode.c:304-306
//     returns OK when the segment count is zero and nothing is owed
//
// The ruling puts the check at the record header where count is read, NOT in
// the id-list path: count governs every block, so rejecting it there removes
// the zero-length parse path through AttrIds, AttrValues and LabelSet from the
// surface both engines must agree on, instead of requiring agreement on each.
// In C that is effects_v3_decode.c:680, immediately after
// `_ReadU32 (stream, &rec->count)` and before the shape is read at :686.
//
// Rust has since landed its half as an `EmptyRecord` error, so this is no
// longer C asserting a rule alone -- the invariant is true on one engine and
// ruled on both. Still gated, because C's check is not in yet and a red build
// helps nobody: the reader's PR defines EFFECTS_V3_ZERO_COUNT_REJECTED in
// effects_v3.h, exactly as the readiness flags work, and this goes live with it.
void test_effectsV3_handBuiltRejections(void) {
	//--------------------------------------------------------------------------
	// UNGATED: the legal empty forms must keep decoding
	//--------------------------------------------------------------------------
	//
	// WHY THIS HALF IS NOT GATED, since it is the only ungated case in a file
	// full of gated ones and would otherwise read as an oversight.
	//
	// Everything else here tests behaviour that does not exist yet, so it is
	// gated until it does. This tests behaviour that must SURVIVE work that has
	// not happened yet. Those are opposite requirements and they take opposite
	// defaults: a guard against a wrong fix is worthless if it only runs once
	// the fix is right.
	//
	// Zero-empty is legal under some records and illegal under others, and the
	// blocks are read by SHARED helpers with one call site each:
	//
	//   _ReadLabelSet  <- every node-shaped record (decode.c:688):
	//                     CREATE_NODE, UPDATE_NODE, DELETE_NODE legal-empty;
	//                     REMOVE_LABELS illegal; SET_LABELS held
	//   _ReadAttrIds   <- every record with values (decode.c:700):
	//                     CREATE_NODE, CREATE_EDGE legal-empty;
	//                     UPDATE_NODE, UPDATE_EDGE illegal
	//
	// Both sets are SPLIT by the rulings, 3-against-2 and 2-against-2. So the
	// obvious place to reject an empty block -- where n is read, inside the
	// helper -- is wrong in both cases: it would refuse CREATE (:Person) and
	// CREATE (). The check has to sit at the call site, keyed on the opcode.
	//
	// The label half of that hazard is already covered: seg_* and collapse_*
	// are DELETE_NODE records with zero labels, so a blanket rejection in
	// _ReadLabelSet fails 18 existing cases immediately. The ATTRIBUTE half is
	// not covered by anything, because no fixture in the corpus has an empty
	// attribute set -- that is the gap that let the decoder refuse
	// CREATE (:Person) in the first place, and create_node_no_attrs is still
	// owed from the generator. Until it lands, this is the only thing standing
	// between a blanket _ReadAttrIds rejection and a green build.
	{
		// version . flags . CREATE_NODE . count=1 . 1 label . 0 attrs . 1 id
		//
		// Zero attributes means zero value rows -- count x attrs_per_row is 0 --
		// so the payload ends after the id list and no SIValue encoding is
		// involved. Layout verified against rec_create_node.hex field by field,
		// and the same construction with one attribute reproduces
		// seg_range_single byte for byte.
		const unsigned char create_no_attrs[] = {
			0x03, 0x00,                          // v3, uncompressed
			0x03, 0x00, 0x00, 0x00,              // CREATE_NODE
			0x01, 0x00, 0x00, 0x00,              // count = 1
			0x01, 0x00,                          // one label...
			0x00, 0x00, 0x00, 0x00,              // ...label 0
			0x00, 0x00,                          // NO attribute ids
			0x01, 0x00, 0x00, 0x00,              // one segment
			0x00, 0x05, 0x01,                    // Range, base 5, len 1
		};

		EffectsV3Records *r = NULL;
		EffectsV3Status   st = EffectsV3_Decode((const char*)create_no_attrs,
				sizeof(create_no_attrs), &r);

		TEST_CASE("CREATE (:Person) -- a create with no attributes");
		TEST_ASSERT_(st == EFFECTS_V3_OK,
				"a create with an empty attribute set was refused as %s. This "
				"is the commonest statement in the language and both engines "
				"emit it; if an empty-block check was just added, it belongs "
				"at the call site keyed on the opcode, not inside "
				"_ReadAttrIds, which every create also goes through",
				EffectsV3Status_ToString(st));

		if(st == EFFECTS_V3_OK) EffectsV3_RecordsFree(r);
	}

	//--------------------------------------------------------------------------
	// GATED: empty blocks that are illegal under THIS record
	//--------------------------------------------------------------------------
	//
	// Ruled invalid by the Rust side with an emitter citation each, so both
	// engines are agreed. Two of the five ruled entries are here; the two
	// constraint ones are records 13-14, which C reports UNIMPLEMENTED, so a
	// rejection case there cannot tell "refused for empty props" from "refused
	// because the record is not decoded". UPDATE_EDGE.attr_ids is absent
	// because its RelType block layout is not verified against any fixture and
	// a hand-built case may not guess one. SET_LABELS.labels is held by Rust
	// and CREATE_INDEX.fields is unruled -- neither is built here, deliberately.
	//
	// The over-rejection side needs nothing new: rec_remove_labels carries one
	// label and rec_update_node carries two attribute ids, so a check that
	// refused the shape rather than the empty block fails those fixtures.
#ifndef EFFECTS_V3_EMPTY_BLOCKS_REJECTED
	V3_SKIP("the empty-block rejections", "EFFECTS_V3_EMPTY_BLOCKS_REJECTED");
#else
	{
		// REMOVE_LABELS removing no labels states nothing.
		// Layout verified against rec_remove_labels.hex, which consumes all 23
		// of its bytes with no values block.
		const unsigned char remove_no_labels[] = {
			0x03, 0x00,
			0x08, 0x00, 0x00, 0x00,              // REMOVE_LABELS
			0x01, 0x00, 0x00, 0x00,              // count = 1
			0x00, 0x00,                          // NO labels
			0x01, 0x00, 0x00, 0x00,              // one segment
			0x00, 0x05, 0x01,                    // Range, base 5, len 1
		};

		EffectsV3Records *r = NULL;
		EffectsV3Status st = EffectsV3_Decode((const char*)remove_no_labels,
				sizeof(remove_no_labels), &r);

		TEST_CASE("REMOVE_LABELS with no labels");
		TEST_ASSERT_(st == EFFECTS_V3_MALFORMED,
				"decoded as %s. The check belongs at the _ReadLabelSet CALL "
				"SITE keyed on the opcode, not inside the helper -- "
				"CREATE_NODE, UPDATE_NODE and DELETE_NODE share it and are "
				"legally empty", EffectsV3Status_ToString(st));
		TEST_ASSERT_(r == NULL, "refused but left records allocated");
	}

	{
		// An UPDATE_NODE that updates nothing. One label, so the attribute set
		// is the only empty block and a failure cannot be blamed on the other.
		// Layout verified against rec_update_node.hex.
		const unsigned char update_no_attrs[] = {
			0x03, 0x00,
			0x01, 0x00, 0x00, 0x00,              // UPDATE_NODE
			0x01, 0x00, 0x00, 0x00,              // count = 1
			0x01, 0x00,                          // one label...
			0x01, 0x00, 0x00, 0x00,              // ...label 1
			0x00, 0x00,                          // NO attribute ids
			0x01, 0x00, 0x00, 0x00,              // one segment
			0x00, 0x05, 0x01,                    // Range, base 5, len 1
		};

		EffectsV3Records *r = NULL;
		EffectsV3Status st = EffectsV3_Decode((const char*)update_no_attrs,
				sizeof(update_no_attrs), &r);

		TEST_CASE("UPDATE_NODE with no attribute ids");
		TEST_ASSERT_(st == EFFECTS_V3_MALFORMED,
				"decoded as %s. Note the empty-attrs fix made n_values == 0 "
				"legal for every record that has values, so this is now "
				"accepted and must be refused again for UPDATE only -- keyed "
				"on the opcode, since CREATE shares the path",
				EffectsV3Status_ToString(st));
		TEST_ASSERT_(r == NULL, "refused but left records allocated");
	}
#endif

	//--------------------------------------------------------------------------
	// GATED: shapes both engines have agreed to refuse
	//--------------------------------------------------------------------------
#ifndef EFFECTS_V3_ZERO_COUNT_REJECTED
	V3_SKIP("the zero-count rejection", "EFFECTS_V3_ZERO_COUNT_REJECTED");
#else
	// version . flags . opcode . count . u16 label count . u32 segment count
	//   03      00       05         00        0000              00000000
	// A DELETE_NODE that is well-formed in every field except the one that
	// makes it meaningless: it names no entities, so it states nothing, and
	// applying nothing silently would slip past the divergence report.
	const unsigned char zero_count[] = {
		0x03, 0x00,                                      // v3, uncompressed
		0x05, 0x00, 0x00, 0x00,                          // DELETE_NODE
		0x00, 0x00, 0x00, 0x00,                          // count = 0
		0x00, 0x00,                                      // no labels
		0x00, 0x00, 0x00, 0x00,                          // no segments
	};

	EffectsV3Records *records = NULL;
	EffectsV3Status   st      = EffectsV3_Decode((const char*)zero_count,
			sizeof(zero_count), &records);

	TEST_CASE("a record covering zero entities");
	TEST_ASSERT_(st == EFFECTS_V3_MALFORMED,
			"a zero-count record decoded as %s; it is well-sized and invalid, "
			"which is what MALFORMED means",
			EffectsV3Status_ToString(st));

	TEST_ASSERT_(records == NULL,
			"a zero-count record was refused but left records allocated");

	// and the same shape with a non-zero count must still be accepted, so the
	// check cannot be satisfied by refusing the shape rather than the count
	unsigned char one_id[] = {
		0x03, 0x00,
		0x05, 0x00, 0x00, 0x00,
		0x01, 0x00, 0x00, 0x00,                          // count = 1
		0x00, 0x00,                                      // no labels
		0x01, 0x00, 0x00, 0x00,                          // one segment
		0x00,                                            // Range, widths 0
		0x07,                                            // base 7
		0x01,                                            // len 1
	};

	EffectsV3Records *ok_records = NULL;
	EffectsV3Status   ok_st      = EffectsV3_Decode((const char*)one_id,
			sizeof(one_id), &ok_records);

	TEST_CASE("the same shape with one entity still decodes");
	TEST_ASSERT_(ok_st == EFFECTS_V3_OK,
			"a one-id record decoded as %s; the zero-count check must reject "
			"the count, not the shape", EffectsV3Status_ToString(ok_st));

	if(ok_st == EFFECTS_V3_OK) EffectsV3_RecordsFree(ok_records);
#endif
}

#endif  // EFFECTS_V3_DECODE_READY

// acutest has no notion of a skipped test, so the gated state says so in the
// test's NAME. Without this the run prints [ OK ] lines that read as passing
// conformance tests, which is the opposite of true. The two halves are named
// separately because they land in different PRs, and truncation is marked
// PARTIAL rather than skipped in the window where decode exists and encode does
// not: it really is testing something then, just not its strongest clause.
#ifdef EFFECTS_V3_DECODE_READY
#define V3_DEC_SUFFIX ""
#else
#define V3_DEC_SUFFIX " (SKIPPED: no decode yet)"
#endif

// the hand-built cases carry a second gate of their own, so the name has to
// say which one is holding them back -- a bare [ OK ] on a test that skipped
// internally is the misleading green this naming exists to prevent
#if !defined(EFFECTS_V3_DECODE_READY)
#define V3_HB_SUFFIX " (SKIPPED: no decode yet)"
#elif !defined(EFFECTS_V3_ZERO_COUNT_REJECTED) && \
      !defined(EFFECTS_V3_EMPTY_BLOCKS_REJECTED)
#define V3_HB_SUFFIX " (PARTIAL: no empty-block or zero-count checks yet)"
#elif !defined(EFFECTS_V3_ZERO_COUNT_REJECTED)
#define V3_HB_SUFFIX " (PARTIAL: zero-count check not implemented)"
#elif !defined(EFFECTS_V3_EMPTY_BLOCKS_REJECTED)
#define V3_HB_SUFFIX " (PARTIAL: empty-block checks not implemented)"
#else
#define V3_HB_SUFFIX ""
#endif

#if !defined(EFFECTS_V3_DECODE_READY)
#define V3_RT_SUFFIX    " (SKIPPED: no decode yet)"
#define V3_TRUNC_SUFFIX " (SKIPPED: no decode yet)"
#elif !defined(EFFECTS_V3_ENCODE_READY)
#define V3_RT_SUFFIX    " (SKIPPED: no encode yet)"
#define V3_TRUNC_SUFFIX " (PARTIAL: no re-encode check)"
#else
#define V3_RT_SUFFIX    ""
#define V3_TRUNC_SUFFIX ""
#endif

TEST_LIST = {
	{ "EffectsV3.decodesAll" V3_DEC_SUFFIX,   test_effectsV3_decodesEveryFixture },
	{ "EffectsV3.roundTrip"  V3_RT_SUFFIX,    test_effectsV3_roundTrip  },
	{ "EffectsV3.truncation" V3_TRUNC_SUFFIX, test_effectsV3_truncation },
	{ "EffectsV3.rejections" V3_DEC_SUFFIX,   test_effectsV3_rejections },
	{ "EffectsV3.handBuilt"  V3_HB_SUFFIX,    test_effectsV3_handBuiltRejections },
	{ NULL, NULL }
};
