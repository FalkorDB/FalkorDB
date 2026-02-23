/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stdlib.h>
#include <sys/types.h>

// forward declaration: the struct body is in block.c
typedef struct Block Block ;

// create a new block
Block *Block_New
(
	uint32_t itemSize,  // item byte size
	uint32_t capacity   // number of items
);

// link blocks
void Block_Link
(
	Block *prev,  // prev block
	Block *next   // next block
) ;

// get a pointer to the ith item
unsigned char *Block_GetItem
(
	Block *block,  // block
	uint32_t i     // item position
) ;

// get next block
Block *Block_Next
(
	const Block *block  // block
);

// reset the ith item header
void Block_ResetHeader
(
	Block *block,  // block
	uint32_t i     // item's position
);

//------------------------------------------------------------------------------
// status checks
//------------------------------------------------------------------------------

// check if the ith item is marked as deleted
bool Block_IsItemDeleted
(
	const Block *block,  // block
	uint32_t i           // item's position
) ;

// check if the ith item is marked as offloaded
bool Block_IsItemOffloaded
(
	const Block *block,  // block
	uint32_t i           // item's position
);

// checks if block contains non offloaded, non deleted items
// return false if offloaded + deleted = block capacity
bool Block_HasActiveItems
(
	const Block *block  // block
);

//------------------------------------------------------------------------------
// state transitions
//------------------------------------------------------------------------------

// marks the ith item as deleted
void Block_MarkItemDeleted
(
	Block *block,  // block
	uint32_t i     // item position within block
);

// marks the ith item as active, counterpart of Block_MarkItemDeleted
void Block_MarkItemActive
(
	Block *block,  // block
	uint32_t i     // item position within block
);

// marks the ith item as offloaded
void Block_MarkItemOffload
(
	Block *block,  // block
	uint32_t i     // item position within block
);

// marks the ith item as loaded, counterpart of Block_MarkItemOffload
void Block_MarkItemLoaded
(
	Block *block,  // block
	uint32_t i     // item position within block
);

// free block
void Block_Free
(
	Block *block  // block to free
);

