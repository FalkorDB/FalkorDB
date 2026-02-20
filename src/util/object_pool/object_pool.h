/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../block.h"
#include <stdlib.h>
#include <stdint.h>

// number of items in a block, should always be a power of 2
#define POOL_BLOCK_CAP 256

// the ObjectPool is a container structure for holding arbitrary items
// of a uniform type in order to reduce the number of alloc/free calls
// and improve locality of reference deletions are not thread-safe
typedef struct {
	int64_t itemCount;           // number of items stored in ObjectPool
	int64_t itemCap;             // number of items ObjectPool can hold
	uint32_t blockCount;         // number of blocks in ObjectPool
	uint32_t itemSize;           // size of a single item in bytes
	Block **blocks;              // array of blocks
	int64_t *deletedIdx;         // array of free indices
	void (*destructor)(void *);  // function pointer to a clean-up function of an item
} ObjectPool;

// create a new ObjectPool
ObjectPool *ObjectPool_New
(
	uint64_t itemCap,   // number of items ObjectPool can hold before resizing
	uint itemSize,      // item size in bytes
	void (*fp)(void *)  // destructor routine for freeing items
);

// allocate a new item from the pool and return a pointer to it
void *ObjectPool_NewItem
(
	ObjectPool *pool  // pool
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

