/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "../block.h"
#include <stdlib.h>
#include <stdint.h>

// Number of items in a block. Should always be a power of 2.
#define POOL_BLOCK_CAP 256

// typedef void (*fpDestructor)(void *);

/* The ObjectPool is a container structure for holding arbitrary items of a uniform type
 * in order to reduce the number of alloc/free calls and improve locality of reference.
 * Deletions are not thread-safe. */
typedef struct {
	uint64_t itemCount;         // Number of items stored in ObjectPool.
	uint64_t itemCap;           // Number of items ObjectPool can hold.
	uint blockCount;            // Number of blocks in ObjectPool.
	uint itemSize;              // Size of a single item in bytes.
	Block **blocks;             // Array of blocks.
	uint64_t *deletedIdx;       // Array of free indices.
	void (*destructor)(void *); // Function pointer to a clean-up function of an item.
} ObjectPool;

// Create a new ObjectPool
// itemCap - number of items ObjectPool can hold before resizing.
// itemSize - item size in bytes.
// fp - destructor routine for freeing items.
ObjectPool *ObjectPool_New(uint64_t itemCap, uint itemSize, void (*fp)(void *));

// Allocate a new item within the given pool and return a pointer to it.
void *ObjectPool_NewItem(ObjectPool *pool);

// Removes item from pool.
void ObjectPool_DeleteItem(ObjectPool *pool, void *item);

// Free pool.
void ObjectPool_Free(ObjectPool *pool);

