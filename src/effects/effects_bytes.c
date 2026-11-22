/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "effects_bytes.h"
#include "../util/rmalloc.h"

#include <string.h>

// determine how many bytes have been written to a block
#define BLOCK_USED_SPACE(b) ((size_t)((b)->offset - (b)->buffer))

// determine a block's remaining capacity
#define BLOCK_AVAILABLE_SPACE(b) ((b)->cap - BLOCK_USED_SPACE(b))

// linked list of blocks
struct EffectsBytesBlock {
	size_t cap;                      // block capacity
	unsigned char *offset;           // write position within buffer
	struct EffectsBytesBlock *next;  // next block
	unsigned char buffer[];          // the bytes themselves
};

struct EffectsBytes {
	size_t block_size;               // size of each block
	size_t len;                      // bytes appended, over every block

	struct EffectsBytesBlock *head;  // first block

	// ALWAYS THE LAST BLOCK, never an interior one: AddBlock only ever links
	// onto it, and Clear frees everything behind head and points this back at
	// head. Held rather than walked to so an append is O(1)
	struct EffectsBytesBlock *tail;
};

// create a new block
static struct EffectsBytesBlock *EffectsBytesBlock_New
(
	size_t n  // size of block
) {
	struct EffectsBytesBlock *b =
		rm_malloc(sizeof(struct EffectsBytesBlock) + n);

	b->cap    = n;
	b->next   = NULL;
	b->offset = b->buffer;

	return b;
}

static void EffectsBytesBlock_Free
(
	struct EffectsBytesBlock *b  // block to free
) {
	ASSERT(b != NULL);

	rm_free(b);
}

// append a new block and make it the tail
static void EffectsBytes_AddBlock
(
	EffectsBytes *bs  // byte sequence
) {
	struct EffectsBytesBlock *b = EffectsBytesBlock_New(bs->block_size);

	bs->tail->next = b;
	bs->tail       = b;
}

// write n bytes from ptr into block
// returns the number of bytes actually written, which is fewer than n when the
// block fills up
static size_t EffectsBytesBlock_Write
(
	struct EffectsBytesBlock *b,  // block to write to
	const unsigned char *ptr,     // data to write
	size_t n                      // number of bytes to write
) {
	ASSERT(n   > 0);
	ASSERT(b   != NULL);
	ASSERT(ptr != NULL);

	// determine number of bytes we can write
	size_t available = BLOCK_AVAILABLE_SPACE(b);
	if(n > available) {
		n = available;
	}

	if(n == 0) {
		return 0;
	}

	memcpy(b->offset, ptr, n);
	b->offset += n;

	return n;
}

EffectsBytes *EffectsBytes_New
(
	size_t block_size  // size of each block
) {
	ASSERT(block_size > 0);

	EffectsBytes *bs = rm_malloc(sizeof(EffectsBytes));

	bs->block_size = block_size;
	bs->len        = 0;
	bs->head       = EffectsBytesBlock_New(block_size);
	bs->tail       = bs->head;

	return bs;
}

void EffectsBytes_Write
(
	EffectsBytes *bs,  // byte sequence
	const void *ptr,   // data to append
	size_t n           // number of bytes to append
) {
	ASSERT(n   > 0);
	ASSERT(bs  != NULL);
	ASSERT(ptr != NULL);

	const unsigned char *src = ptr;

	bs->len += n;

	while(n > 0) {
		size_t written = EffectsBytesBlock_Write(bs->tail, src, n);

		if(written == 0) {
			// tail block is full, add another
			EffectsBytes_AddBlock(bs);
			continue;
		}

		src += written;
		n   -= written;
	}
}

void EffectsBytes_Append
(
	EffectsBytes *bs,       // byte sequence to append to
	const EffectsBytes *src  // byte sequence to append
) {
	ASSERT(bs  != NULL);
	ASSERT(src != NULL);
	// RELEASE_ASSERT for the reason EffectsV3_WriteUint's width check is one:
	// ASSERT compiles out of a release build, and self-append does not fail
	// cleanly - Append would keep finding blocks that Write is still adding,
	// allocating until the host gives up. No caller can do it today, but that
	// is a fact about the call sites
	RELEASE_ASSERT(bs != src);

	for(const struct EffectsBytesBlock *b = src->head; b != NULL; b = b->next) {
		size_t n = BLOCK_USED_SPACE(b);
		if(n > 0) {
			EffectsBytes_Write(bs, b->buffer, n);
		}
	}
}

size_t EffectsBytes_Len
(
	const EffectsBytes *bs  // byte sequence
) {
	ASSERT(bs != NULL);

	// RUNNING COUNT, not a walk. Callers ask per record - the encoder sizes
	// every group's values block before writing it - and the blocks a long
	// payload holds are the thing being counted, so a walk here is the one
	// part of encoding that gets slower the more there is to encode
	return bs->len;
}

bool EffectsBytes_IsEmpty
(
	const EffectsBytes *bs  // byte sequence
) {
	ASSERT(bs != NULL);

	return bs->len == 0;
}

unsigned char *EffectsBytes_CopyInto
(
	const EffectsBytes *bs,  // byte sequence
	unsigned char *dst       // destination
) {
	ASSERT(bs  != NULL);
	ASSERT(dst != NULL);

	for(const struct EffectsBytesBlock *b = bs->head; b != NULL; b = b->next) {
		size_t n = BLOCK_USED_SPACE(b);
		if(n > 0) {
			memcpy(dst, b->buffer, n);
			dst += n;
		}
	}

	return dst;
}

void EffectsBytes_Clear
(
	EffectsBytes *bs  // byte sequence
) {
	ASSERT(bs != NULL);

	// free every block but the first
	struct EffectsBytesBlock *b = bs->head->next;
	while(b != NULL) {
		struct EffectsBytesBlock *next = b->next;
		EffectsBytesBlock_Free(b);
		b = next;
	}

	bs->head->next   = NULL;
	bs->head->offset = bs->head->buffer;
	bs->tail         = bs->head;
	bs->len          = 0;
}

void EffectsBytes_Free
(
	EffectsBytes *bs  // byte sequence
) {
	if(bs == NULL) {
		return;
	}

	struct EffectsBytesBlock *b = bs->head;
	while(b != NULL) {
		struct EffectsBytesBlock *next = b->next;
		EffectsBytesBlock_Free(b);
		b = next;
	}

	rm_free(bs);
}
