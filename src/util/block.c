/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "block.h"
#include "RG.h"
#include "rmalloc.h"

Block *Block_New(uint itemSize, uint capacity) {
	ASSERT(itemSize > 0);
	Block *block = rm_calloc(1, sizeof(Block) + (capacity * itemSize));
	block->itemSize = itemSize;
	return block;
}

void Block_Free(Block *block) {
	ASSERT(block != NULL);
	rm_free(block);
}

