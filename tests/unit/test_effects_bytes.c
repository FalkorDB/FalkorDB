/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// The block-list byte sequence every v3 encoder writes into.
//
// It had no suite of its own: what covered it was the encoder tests, which only
// ever build a sequence and read it back once. That leaves the two operations a
// reused buffer depends on - Clear and Append - asserted by nothing, and
// EffectsBytes_Len now answers from a running count rather than a walk, so a
// count that drifts from the bytes is no longer impossible by construction.
//
// Block sizes here are deliberately tiny, so every case crosses block
// boundaries. The production sizes are 1024 (a group's values) and larger.

#include "src/effects/effects_bytes.h"
#include "src/util/rmalloc.h"

void setup() {
	Alloc_Reset();
}

#define TEST_INIT setup();
#include "acutest.h"

#include <stdlib.h>
#include <string.h>

// read the whole sequence back as a NUL terminated string
//
// ASSERTS THE COUPLING, because every caller below would otherwise hide it:
// they ask Len, size a buffer from it, and compare the text. A Len that
// OVER-reports then reads as a short string and a Len that UNDER-reports
// overflows the allocation - one looks like a content failure and the other
// does not fail here at all. The slack lets CopyInto run past a bad Len
// without corrupting the heap, and the cursor it returns is the real length
static char *_drain(const EffectsBytes *bs) {
	size_t n = EffectsBytes_Len(bs);
	char *out = malloc(n + 64);
	unsigned char *end = EffectsBytes_CopyInto(bs, (unsigned char *)out);
	size_t copied = (size_t)(end - (unsigned char *)out);
	TEST_ASSERT_(copied == n,
			"EffectsBytes_Len says %zu, CopyInto wrote %zu", n, copied);
	out[n] = '\0';
	return out;
}

// the length is what was written, across however many blocks that took
void test_effectsBytes_lenCountsEveryBlock(void) {
	EffectsBytes *bs = EffectsBytes_New(4);
	TEST_ASSERT(EffectsBytes_Len(bs) == 0);
	TEST_ASSERT(EffectsBytes_IsEmpty(bs));

	// 26 bytes through 4-byte blocks: seven blocks, and a write that spans
	EffectsBytes_Write(bs, "abcdefghijklmnopqrstuvwxyz", 26);

	TEST_ASSERT(EffectsBytes_Len(bs) == 26);
	TEST_ASSERT(!EffectsBytes_IsEmpty(bs));

	char *got = _drain(bs);
	TEST_ASSERT_(strcmp(got, "abcdefghijklmnopqrstuvwxyz") == 0,
			"read back \"%s\"", got);
	free(got);

	EffectsBytes_Free(bs);
}

// a single write larger than a whole block still lands intact
void test_effectsBytes_writeLargerThanABlock(void) {
	EffectsBytes *bs = EffectsBytes_New(2);

	unsigned char big[1000];
	for(size_t i = 0; i < sizeof(big); i++) big[i] = (unsigned char)(i & 0xff);
	EffectsBytes_Write(bs, big, sizeof(big));

	TEST_ASSERT(EffectsBytes_Len(bs) == sizeof(big));

	unsigned char *got = malloc(sizeof(big));
	unsigned char *end = EffectsBytes_CopyInto(bs, got);
	TEST_ASSERT_(end == got + sizeof(big),
			"CopyInto advanced %zd, expected %zu", end - got, sizeof(big));
	TEST_ASSERT(memcmp(got, big, sizeof(big)) == 0);

	free(got);
	EffectsBytes_Free(bs);
}

// CLEAR TAKES THE LENGTH WITH IT
//
// EffectsBuffer_Reset clears and reuses the records sequence (effects.c:355),
// and EffectsBuffer_Buffer sizes its output allocation from EffectsBytes_Len
// and then reports that size as the payload length (effects.c:390-402). A
// length that survived a Clear would have the next payload claim bytes that
// were never written.
void test_effectsBytes_clearResetsTheLength(void) {
	EffectsBytes *bs = EffectsBytes_New(4);

	EffectsBytes_Write(bs, "first payload", 13);
	TEST_ASSERT(EffectsBytes_Len(bs) == 13);

	EffectsBytes_Clear(bs);

	TEST_ASSERT_(EffectsBytes_Len(bs) == 0,
			"after Clear the length is %zu, expected 0",
			EffectsBytes_Len(bs));
	TEST_ASSERT(EffectsBytes_IsEmpty(bs));

	// and the reused sequence reports only what it holds now
	EffectsBytes_Write(bs, "second", 6);
	TEST_ASSERT_(EffectsBytes_Len(bs) == 6,
			"after reuse the length is %zu, expected 6",
			EffectsBytes_Len(bs));

	char *got = _drain(bs);
	TEST_ASSERT_(strcmp(got, "second") == 0, "read back \"%s\"", got);
	free(got);

	EffectsBytes_Free(bs);
}

// clearing an empty sequence is not a special case
void test_effectsBytes_clearOnAnEmptySequence(void) {
	EffectsBytes *bs = EffectsBytes_New(8);

	EffectsBytes_Clear(bs);
	TEST_ASSERT(EffectsBytes_Len(bs) == 0);
	TEST_ASSERT(EffectsBytes_IsEmpty(bs));

	EffectsBytes_Write(bs, "x", 1);
	TEST_ASSERT(EffectsBytes_Len(bs) == 1);

	EffectsBytes_Free(bs);
}

// APPEND CONCATENATES AND LEAVES THE SOURCE ALONE
//
// This is how a group's values block joins its rows, so the source is read
// again after the append in the encoder's own flow.
void test_effectsBytes_appendConcatenates(void) {
	EffectsBytes *dst = EffectsBytes_New(4);
	EffectsBytes *src = EffectsBytes_New(4);

	EffectsBytes_Write(dst, "head:", 5);
	// spans several of src's 4-byte blocks
	EffectsBytes_Write(src, "the appended part", 17);

	EffectsBytes_Append(dst, src);

	TEST_ASSERT_(EffectsBytes_Len(dst) == 5 + 17,
			"dst is %zu bytes, expected 22", EffectsBytes_Len(dst));

	char *got = _drain(dst);
	TEST_ASSERT_(strcmp(got, "head:the appended part") == 0,
			"read back \"%s\"", got);
	free(got);

	// the source is untouched and still readable
	TEST_ASSERT_(EffectsBytes_Len(src) == 17,
			"src is %zu bytes after being appended, expected 17",
			EffectsBytes_Len(src));
	char *s = _drain(src);
	TEST_ASSERT_(strcmp(s, "the appended part") == 0, "src reads \"%s\"", s);
	free(s);

	EffectsBytes_Free(dst);
	EffectsBytes_Free(src);
}

// appending an empty sequence changes nothing
void test_effectsBytes_appendEmpty(void) {
	EffectsBytes *dst = EffectsBytes_New(16);
	EffectsBytes *src = EffectsBytes_New(16);

	EffectsBytes_Write(dst, "unchanged", 9);
	EffectsBytes_Append(dst, src);

	TEST_ASSERT(EffectsBytes_Len(dst) == 9);
	char *got = _drain(dst);
	TEST_ASSERT_(strcmp(got, "unchanged") == 0, "read back \"%s\"", got);
	free(got);

	EffectsBytes_Free(dst);
	EffectsBytes_Free(src);
}

// CopyInto returns the advanced cursor, so sequences thread
void test_effectsBytes_copyIntoThreads(void) {
	EffectsBytes *a = EffectsBytes_New(4);
	EffectsBytes *b = EffectsBytes_New(4);

	EffectsBytes_Write(a, "aaaa", 4);
	EffectsBytes_Write(b, "bbbbbb", 6);

	unsigned char buf[11];
	unsigned char *p = EffectsBytes_CopyInto(a, buf);
	p = EffectsBytes_CopyInto(b, p);
	*p = '\0';

	TEST_ASSERT_(p == buf + 10, "threaded cursor advanced %zd, expected 10",
			p - buf);
	TEST_ASSERT_(strcmp((char *)buf, "aaaabbbbbb") == 0,
			"read back \"%s\"", (char *)buf);

	EffectsBytes_Free(a);
	EffectsBytes_Free(b);
}

TEST_LIST = {
	{ "EffectsBytes:lenCountsEveryBlock",
			test_effectsBytes_lenCountsEveryBlock },
	{ "EffectsBytes:writeLargerThanABlock",
			test_effectsBytes_writeLargerThanABlock },
	{ "EffectsBytes:clearResetsTheLength",
			test_effectsBytes_clearResetsTheLength },
	{ "EffectsBytes:clearOnAnEmptySequence",
			test_effectsBytes_clearOnAnEmptySequence },
	{ "EffectsBytes:appendConcatenates",
			test_effectsBytes_appendConcatenates },
	{ "EffectsBytes:appendEmpty",       test_effectsBytes_appendEmpty },
	{ "EffectsBytes:copyIntoThreads",   test_effectsBytes_copyIntoThreads },
	{ NULL, NULL }
};
