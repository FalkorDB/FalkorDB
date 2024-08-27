/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../block.h"
#include <stdlib.h>
#include <stdint.h>

// Number of items in a block. Should always be a power of 2.
#define POOL_BLOCK_CAP 256

// the ObjectPool is a container structure for holding arbitrary items
// of a uniform type in order to reduce the number of alloc/free calls
// and improve locality of reference deletions are not thread-safe
typedef struct {
	uint64_t itemCount;         // number of items stored in ObjectPool
	uint64_t itemCap;           // number of items ObjectPool can hold
	uint blockCount;            // number of blocks in ObjectPool
	uint itemSize;              // size of a single item in bytes
	Block **blocks;             // array of blocks
	uint64_t *deletedIdx;       // array of free indices
	void (*destructor)(void *); // function pointer to a clean-up function of an item
} ObjectPool;

// create a new ObjectPool
// itemCap - number of items ObjectPool can hold before resizing
// itemSize - item size in bytes
// fp - destructor routine for freeing items
ObjectPool *ObjectPool_New
(
	uint64_t itemCap,
	uint itemSize,
	void (*fp)(void *)
);

// allocate a new item within the given pool and return a pointer to it
void *ObjectPool_NewItem
(
	ObjectPool *pool
);

// removes item from pool
void ObjectPool_DeleteItem
(
	ObjectPool *pool,
	void *item
);

// free pool
void ObjectPool_Free
(
	ObjectPool *pool
);

