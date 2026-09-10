/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

// The effects v3 checks that need no fixtures, and the one that notices when
// the fixtures are missing.
//
// WHY THIS FILE EXISTS AT ALL. The conformance corpus and its fixture-driven
// harness live on an auxiliary branch, mounted into this tree rather than
// committed to it. That arrangement has one failure mode, and it is silent:
//
//     tests/unit/CMakeLists.txt:2   file(GLOB TEST_SOURCES ... test_*.c)
//
// The glob runs at CONFIGURE time. With the aux branch unmounted, the harness's
// sources are not there, so no target is generated -- and a target that does not
// exist does not fail, does not skip, and does not appear. The corpus is not
// "unverified", it is absent, and nothing in the build says so. That is exactly
// how the other engine reached 22 hours with no corpus verification at all: the
// consumer left the tree and no signal fired.
//
// So something has to stay in-tree whose job is to notice. This is it. It also
// carries the checks that genuinely need no fixtures, so it is a real test
// rather than a tripwire with nothing behind it -- hand-built payloads are C
// source, not corpus artifacts, and they were always the half of the harness
// that could stand alone.

#include "src/effects/effects_v3.h"

#include "src/util/rmalloc.h"
#include "src/globals.h"
#include "src/util/thpool/pool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static void setup(void) {
	Alloc_Reset();

	// The decoder interns strings, which reaches Globals_Get_StringPool(), and
	// StringPool_rent guards a NULL pool with ASSERT -- empty in release, so it
	// dereferences NULL instead. Globals_Init sizes its table from
	// ThreadPool_ThreadCount(), so the pool has to exist first even though
	// nothing here runs concurrently.
	static bool ready = false;
	if(!ready) {
		ThreadPool_Init();
		ThreadPool_CreatePool(1, 64);
		Globals_Init();
		ready = true;
	}
}

#define TEST_INIT setup();
#include "acutest.h"

//------------------------------------------------------------------------------
// is the conformance corpus mounted?
//------------------------------------------------------------------------------

#ifndef EFFECTS_V3_FIXTURES_DIR
#define EFFECTS_V3_FIXTURES_DIR "tests/fixtures/effects_v3"
#endif

// THE MOUNT HAS TWO HALVES AND ONLY ONE OF THEM IS THE FIXTURES.
//
//   fixtures   reachable through EFFECTS_V3_FIXTURES_DIR, which any path
//              satisfies -- a runtime question
//   sources    present in tests/unit/ BEFORE cmake configures, because that is
//              when the glob runs -- a COMPILE-TIME question the env var cannot
//              speak to at all
//
// An earlier version of this file tested only the first and reported "MOUNTED"
// on the strength of it. That is a false green in the exact state a two-part
// mount makes easy to reach: someone runs the worktree add, does not copy the
// harness sources or does not reconfigure, and is told everything is fine while
// the harness does not exist. EFFECTS_V3_REQUIRE_CORPUS did not catch it
// either, because the fixture opens.
//
// __has_include answers the same question the glob answers, at the moment the
// glob answers it, which is the only way to detect a target that was never
// generated from inside a target that was.
#if defined(__has_include)
#  if __has_include("tests/unit/effects_v3_corpus.h")
#    define EFFECTS_V3_HARNESS_PRESENT 1
#  endif
#endif

#define MOUNT_CMD \
	"git worktree add tests/fixtures/effects_v3 origin/effects-v3-corpus"

static const char *_corpus_dir(void) {
	const char *d = getenv("EFFECTS_V3_FIXTURES_DIR");
	return (d != NULL && *d != '\0') ? d : EFFECTS_V3_FIXTURES_DIR;
}

// Reported by opening a fixture rather than by stat'ing the directory: an empty
// directory, a half-finished checkout and a mount pointing somewhere wrong all
// look identical to a directory test, and all three are the states worth
// distinguishing from "mounted".
static bool _corpus_present(char *why, size_t n) {
	char path[1024];
	snprintf(path, sizeof(path), "%s/seg_range.hex", _corpus_dir());

	FILE *f = fopen(path, "rb");
	if(f == NULL) {
		snprintf(why, n, "cannot open %s", path);
		return false;
	}

	// and it has to contain something: a zero-length file is what an
	// interrupted checkout leaves behind
	fseek(f, 0, SEEK_END);
	long len = ftell(f);
	fclose(f);

	if(len <= 0) {
		snprintf(why, n, "%s is empty", path);
		return false;
	}

	return true;
}

// THE TRIPWIRE.
//
// Passing when the corpus is absent is deliberate: an unmounted tree is the
// normal state for anyone not working on the wire format, and a red build for
// them would be noise that gets suppressed, which ends with nobody reading it.
// So absence is LOUD rather than fatal -- printed unconditionally to stdout,
// not through TEST_MSG, which acutest shows only on failure or with --verbose
// and would therefore be invisible in exactly the case that matters.
//
// CI IS THE OTHER HALF. A job that means to run the corpus sets
// EFFECTS_V3_REQUIRE_CORPUS=1, and then absence is a hard failure. Without
// that, a CI job whose mount step silently failed would produce a green run
// with no harness in it -- which is the same silence this file exists to
// break, one level up. The variable is the difference between "not mounted
// here" and "was supposed to be mounted and is not".
void test_effectsV3Wire_corpusMountStatus(void) {
	char why[1024] = {0};
	bool fixtures  = _corpus_present(why, sizeof(why));
	bool required  = getenv("EFFECTS_V3_REQUIRE_CORPUS") != NULL;

#ifdef EFFECTS_V3_HARNESS_PRESENT
	const bool harness = true;
#else
	const bool harness = false;
#endif

	if(fixtures && harness) {
		printf("\n  effects v3 conformance corpus: MOUNTED, harness compiled "
				"in (%s)\n", _corpus_dir());
		TEST_ASSERT(1);
		return;
	}

	// The dangerous middle. Fixtures are reachable, so every runtime check
	// says yes, and the harness still is not in this binary -- almost always
	// because cmake was configured before the sources arrived.
	if(fixtures && !harness) {
		printf("\n"
			"  ==========================================================\n"
			"   effects v3 corpus: FIXTURES PRESENT, HARNESS ABSENT\n"
			"   fixtures at %s\n"
			"\n"
			"   tests/unit/effects_v3_corpus.h did not exist when this file\n"
			"   was compiled, so the fixture-driven tests -- corpus\n"
			"   integrity, the byte round trip, the truncation sweep -- were\n"
			"   never built. They are not failing; there is no target.\n"
			"\n"
			"   The fixtures being reachable is why nothing else notices.\n"
			"   Usually this means cmake configured before the aux branch\n"
			"   sources were copied into tests/unit/. RECONFIGURE:\n"
			"     cmake <build-dir> && make unit-tests\n"
			"  ==========================================================\n",
			_corpus_dir());
	} else if(!fixtures && harness) {
		printf("\n"
			"  ==========================================================\n"
			"   effects v3 corpus: HARNESS COMPILED IN, FIXTURES MISSING\n"
			"   %s\n"
			"   The fixture-driven tests exist and will each report that\n"
			"   they cannot open their case. Point EFFECTS_V3_FIXTURES_DIR\n"
			"   at the corpus, or:\n"
			"     %s\n"
			"  ==========================================================\n",
			why, MOUNT_CMD);
	} else {
		printf("\n"
			"  ==========================================================\n"
			"   effects v3 conformance corpus: NOT MOUNTED\n"
			"   %s\n"
			"\n"
			"   WHAT IT IS, since you may not have met it. 39 byte fixtures\n"
			"   that C and Rust must both encode identically -- the shared\n"
			"   statement of the v3 wire format, on the branch named below.\n"
			"   Its tests (integrity, a byte round trip, a ~4,500-prefix\n"
			"   truncation sweep) are NOT FAILING: they do not exist in this\n"
			"   build. tests/unit/CMakeLists.txt globs test_*.c at configure\n"
			"   time and their sources are not in this tree.\n"
			"\n"
			"   WHY THAT MATTERS IF YOU DID NOT ASK. Those fixtures go stale\n"
			"   against a moving wire and nothing notices: the manifest hashes\n"
			"   them against a table shipped beside them, so a stale mirror is\n"
			"   self-consistent. It has happened -- one fixture went 96 bytes\n"
			"   to 69 while every hash still matched. Only a codec decoding\n"
			"   them catches that, which needs corpus and engine in one place.\n"
			"\n"
			"   NO CI JOB RUNS THIS, so this message is the only notice anyone\n"
			"   gets. If you are changing src/effects/, please mount it:\n"
			"     %s\n"
			"     cp -r tests/fixtures/effects_v3/c/tests/unit/. tests/unit/\n"
			"     cmake <build-dir> && make unit-tests   # reconfigure\n"
			"\n"
			"   The checks below need no fixtures and did run.\n"
			"  ==========================================================\n",
			why, MOUNT_CMD);
	}

	// Required means BOTH halves. A job that means to run the corpus and got
	// only the fixtures has produced a run its result does not describe.
	TEST_ASSERT_(!required,
			"EFFECTS_V3_REQUIRE_CORPUS is set and the corpus harness is not "
			"fully mounted (fixtures=%s, harness=%s). This run was supposed to "
			"include the fixture-driven tests and does not, so its result says "
			"nothing about the wire format. %s",
			fixtures ? "yes" : "no", harness ? "yes" : "no",
			harness ? MOUNT_CMD : "mount the branch, copy its tests/unit "
			                      "sources, and reconfigure");
}

//------------------------------------------------------------------------------
// payloads built here, needing no corpus
//------------------------------------------------------------------------------

// A deliberately-invalid payload should never be a fixture. Generating one
// would put a shape into the corpus that no conforming encoder produces, and
// the corpus records what the two engines AGREE on. So these are built byte by
// byte, which is also what lets them stay in-tree.
//
// Layout verified against the corpus rather than assumed, while it was still
// committed here: the same construction with one attribute reproduced
// seg_range_single byte for byte, and the CREATE_NODE header parses
// field-for-field against rec_create_node. Record order is
// `u32 opcode · u32 count · LabelSet · AttrIds · IdList · AttrValues`.

#ifndef EFFECTS_V3_DECODE_READY

void test_effectsV3Wire_handBuilt(void) {
	printf("\n  hand-built payloads: skipped, EffectsV3_Decode not defined\n");
	TEST_ASSERT(1);
}

#else

void test_effectsV3Wire_handBuilt(void) {
	//--------------------------------------------------------------------------
	// the legal empty forms must keep decoding
	//--------------------------------------------------------------------------
	//
	// UNGATED, and that is the point. Everything else here waits for behaviour
	// that does not exist yet; this tests behaviour that must SURVIVE work that
	// has not happened. Opposite requirements, opposite defaults -- a guard
	// against a wrong fix is worthless if it only runs once the fix is right.
	//
	// The hazard it guards: zero-empty is legal under some records and illegal
	// under others, and both blocks are read by SHARED helpers with one call
	// site each. _ReadLabelSet serves every node-shaped record, where
	// CREATE/UPDATE/DELETE_NODE are legally empty and REMOVE_LABELS is not.
	// _ReadAttrIds serves every record with values, where CREATE_NODE and
	// CREATE_EDGE are legally empty and the UPDATEs are not. Both sets are
	// split by the rulings, so the obvious place to reject an empty block --
	// inside the helper, where n is read -- is wrong in both cases and would
	// refuse `CREATE (:Person)`.
	{
		// version . flags . CREATE_NODE . count=1 . 1 label . 0 attrs . 1 id
		//
		// Zero attributes means zero value rows, so the payload ends after the
		// id list and no SIValue encoding is involved.
		const unsigned char create_no_attrs[] = {
			0x03, 0x00,
			0x03, 0x00, 0x00, 0x00,              // CREATE_NODE
			0x01, 0x00, 0x00, 0x00,              // count = 1
			0x01, 0x00,                          // one label...
			0x00, 0x00, 0x00, 0x00,              // ...label 0
			0x00, 0x00,                          // NO attribute ids
			0x01, 0x00, 0x00, 0x00,              // one segment
			0x00, 0x05, 0x01,                    // Range, base 5, len 1
		};

		EffectsV3Records *r = NULL;
		EffectsV3Status st = EffectsV3_Decode((const char*)create_no_attrs,
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
	// empty blocks that are illegal under THIS record
	//--------------------------------------------------------------------------
	//
	// Ruled invalid with an emitter citation on each, so both engines agree.
	// Two of the five ruled entries are here; the two constraint ones are
	// records 13-14, and UPDATE_EDGE is absent because its RelType block layout
	// is verified against no fixture and a hand-built case may not guess one.
#ifndef EFFECTS_V3_EMPTY_BLOCKS_REJECTED
	TEST_CASE("pending EFFECTS_V3_EMPTY_BLOCKS_REJECTED");
	printf("  empty-block rejections: skipped, check not implemented\n");
#else
	{
		const unsigned char remove_no_labels[] = {
			0x03, 0x00,
			0x08, 0x00, 0x00, 0x00,              // REMOVE_LABELS
			0x01, 0x00, 0x00, 0x00,              // count = 1
			0x00, 0x00,                          // NO labels
			0x01, 0x00, 0x00, 0x00,
			0x00, 0x05, 0x01,
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
		// one label, so the attribute set is the only empty block and a failure
		// cannot be blamed on the other
		const unsigned char update_no_attrs[] = {
			0x03, 0x00,
			0x01, 0x00, 0x00, 0x00,              // UPDATE_NODE
			0x01, 0x00, 0x00, 0x00,
			0x01, 0x00,
			0x01, 0x00, 0x00, 0x00,              // label 1
			0x00, 0x00,                          // NO attribute ids
			0x01, 0x00, 0x00, 0x00,
			0x00, 0x05, 0x01,
		};

		EffectsV3Records *r = NULL;
		EffectsV3Status st = EffectsV3_Decode((const char*)update_no_attrs,
				sizeof(update_no_attrs), &r);

		TEST_CASE("UPDATE_NODE with no attribute ids");
		TEST_ASSERT_(st == EFFECTS_V3_MALFORMED,
				"decoded as %s. The empty-attrs fix made n_values == 0 legal "
				"for every record with values, so this must be refused again "
				"for UPDATE only, keyed on the opcode",
				EffectsV3Status_ToString(st));
		TEST_ASSERT_(r == NULL, "refused but left records allocated");
	}
#endif

	//--------------------------------------------------------------------------
	// a record covering zero entities
	//--------------------------------------------------------------------------
	//
	// Ruled illegal after both engines were found accepting it while both
	// rejected the same idea one level down at the segment. The check belongs
	// at the record header where count is read, not in the id-list path: count
	// governs every block, so rejecting it there removes the zero-length parse
	// path through AttrIds, AttrValues and LabelSet from the surface both
	// engines must agree on, rather than requiring agreement on each.
#ifndef EFFECTS_V3_ZERO_COUNT_REJECTED
	TEST_CASE("pending EFFECTS_V3_ZERO_COUNT_REJECTED");
	printf("  zero-count rejection: skipped, check not implemented\n");
#else
	{
		const unsigned char zero_count[] = {
			0x03, 0x00,
			0x05, 0x00, 0x00, 0x00,              // DELETE_NODE
			0x00, 0x00, 0x00, 0x00,              // count = 0
			0x00, 0x00,                          // no labels
			0x00, 0x00, 0x00, 0x00,              // no segments
		};

		EffectsV3Records *r = NULL;
		EffectsV3Status st = EffectsV3_Decode((const char*)zero_count,
				sizeof(zero_count), &r);

		TEST_CASE("a record covering zero entities");
		TEST_ASSERT_(st == EFFECTS_V3_MALFORMED,
				"a zero-count record decoded as %s; it is well-sized and "
				"invalid, which is what MALFORMED means",
				EffectsV3Status_ToString(st));
		TEST_ASSERT_(r == NULL, "refused but left records allocated");
	}

	{
		// the same shape with one entity must still decode, so the check
		// cannot be satisfied by refusing the shape rather than the count
		const unsigned char one_id[] = {
			0x03, 0x00,
			0x05, 0x00, 0x00, 0x00,
			0x01, 0x00, 0x00, 0x00,              // count = 1
			0x00, 0x00,
			0x01, 0x00, 0x00, 0x00,
			0x00, 0x07, 0x01,
		};

		EffectsV3Records *r = NULL;
		EffectsV3Status st = EffectsV3_Decode((const char*)one_id,
				sizeof(one_id), &r);

		TEST_CASE("the same shape with one entity still decodes");
		TEST_ASSERT_(st == EFFECTS_V3_OK,
				"a one-id record decoded as %s; the zero-count check must "
				"reject the count, not the shape",
				EffectsV3Status_ToString(st));

		if(st == EFFECTS_V3_OK) EffectsV3_RecordsFree(r);
	}
#endif
}

#endif  // EFFECTS_V3_DECODE_READY

TEST_LIST = {
	{ "EffectsV3Wire.corpusMountStatus", test_effectsV3Wire_corpusMountStatus },
	{ "EffectsV3Wire.handBuilt",         test_effectsV3Wire_handBuilt         },
	{ NULL, NULL }
};
