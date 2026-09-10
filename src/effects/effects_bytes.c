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
	size_t block_size;                  // size of each block
	struct EffectsBytesBlock *head;     // first block
	struct EffectsBytesBlock *current;  // block currently being written into
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

// append a new block and make it current
static void EffectsBytes_AddBlock
(
	EffectsBytes *bs  // byte sequence
) {
	struct EffectsBytesBlock *b = EffectsBytesBlock_New(bs->block_size);

	bs->current->next = b;
	bs->current       = b;
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
	bs->head       = EffectsBytesBlock_New(block_size);
	bs->current    = bs->head;

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

	while(n > 0) {
		size_t written = EffectsBytesBlock_Write(bs->current, src, n);

		if(written == 0) {
			// current block is full, add another
			EffectsBytes_AddBlock(bs);
			continue;
		}

		src += written;
		n   -= written;
	}
}

size_t EffectsBytes_Len
(
	const EffectsBytes *bs  // byte sequence
) {
	ASSERT(bs != NULL);

	size_t l = 0;

	for(const struct EffectsBytesBlock *b = bs->head; b != NULL; b = b->next) {
		l += BLOCK_USED_SPACE(b);
	}

	return l;
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
	bs->current      = bs->head;
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
