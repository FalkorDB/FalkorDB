/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "block.h"
#include "RG.h"
#include "rmalloc.h"

Block *Block_New
(
	uint itemSize,  // item size
	uint capacity   // number of items
) {
	ASSERT (itemSize > 0) ;

	size_t n = sizeof (Block) + (capacity * itemSize) ;
	Block *block = rm_calloc (1, n) ;

	block->itemSize = itemSize ;

	return block ;
}

void Block_Free
(
	Block *block
) {
	ASSERT (block != NULL) ;
	rm_free (block) ;
}

