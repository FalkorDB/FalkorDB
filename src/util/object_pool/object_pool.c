/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "object_pool.h"
#include "../arr.h"
#include "../rmalloc.h"

#include <math.h>
#include <string.h>

// computes the number of blocks required to accommodate n items
#define ITEM_COUNT_TO_BLOCK_COUNT(n) \
	ceil((double)n / POOL_BLOCK_CAP)

// computes block index from item index
#define ITEM_INDEX_TO_BLOCK_INDEX(idx) \
	(idx / POOL_BLOCK_CAP)

// computes item position within a block
#define ITEM_POSITION_WITHIN_BLOCK(idx) \
	(idx % POOL_BLOCK_CAP)

// retrieves block in which item with index resides
#define GET_ITEM_BLOCK(pool, idx) \
	pool->blocks[ITEM_INDEX_TO_BLOCK_INDEX(idx)]

// each allocated item has an ID header that is equivalent to
// its index in the ObjectPool, this ID is held in the bytes immediately
// preceding the item in the Block, and is only used internally
typedef uint64_t ObjectID;
#define HEADER_SIZE sizeof(ObjectID)

// given an item, retrieve its ID
// as this is a direct memory access, it can be used as either
// a getter or a setter
// ITEM_ID(x) = 5 will assign 5 to the item x, while
// objectID num = ITEM_ID(x) will write x's ID to the variable num
#define ITEM_ID(item) *((ObjectID*)((item) - sizeof(ObjectID)))

// given an item header, retrieve the item data
#define ITEM_FROM_HEADER(header) ((header) + sizeof(ObjectID))

// add blocks to the pool
static void _ObjectPool_AddBlocks
(
	ObjectPool *pool,    // object pool
	uint32_t blockCount  // number of blocks to add
) {
	ASSERT (pool != NULL) ;
	ASSERT (blockCount > 0) ;

	uint32_t prevBlockCount = pool->blockCount ;
	pool->blockCount += blockCount ;

	pool->blocks =
		rm_realloc (pool->blocks, sizeof (Block *) * pool->blockCount) ;

	for(uint32_t i = prevBlockCount; i < pool->blockCount; i++) {
		pool->blocks[i] = Block_New (pool->itemSize, POOL_BLOCK_CAP) ;
		if (i > 0) {
			Block_Link (pool->blocks[i - 1], pool->blocks[i]) ;
		}
	}
	pool->itemCap = pool->blockCount * POOL_BLOCK_CAP ;
}

// clear a deleted item and recycle it to the caller
static void *_ObjectPool_ReuseItem
(
	ObjectPool *pool
) {
	ASSERT (pool != NULL) ;

	ObjectID idx = array_pop (pool->deletedIdx) ;

	pool->itemCount++ ;

	Block *block = GET_ITEM_BLOCK (pool, idx) ;
	uint32_t pos = ITEM_POSITION_WITHIN_BLOCK (idx) ;

	// retrieve a pointer to the item's header
	unsigned char *header = Block_GetItem    (block, pos) ;
	unsigned char *item   = ITEM_FROM_HEADER (header) ;

	// the item ID should not change on reuse
	ASSERT (*((ObjectID*)header) == idx) ;

	// zero-set the item being returned
	memset (item, 0, pool->itemSize - HEADER_SIZE) ;

	return item ;
}

// create a new ObjectPool
ObjectPool *ObjectPool_New
(
	uint64_t itemCap,   // number of items ObjectPool can hold before resizing
	uint itemSize,      // item size in bytes
	void (*fp)(void *)  // destructor routine for freeing items
) {
	ObjectPool *pool = rm_calloc (1, sizeof (ObjectPool)) ;

	pool->itemSize   = itemSize + HEADER_SIZE ;  // accommodate the header
	pool->deletedIdx = array_new (int64_t, 128) ;
	pool->destructor = fp ;

	_ObjectPool_AddBlocks (pool, ITEM_COUNT_TO_BLOCK_COUNT (itemCap)) ;

	return pool ;
}

// allocate a new item from the pool and return a pointer to it
void *ObjectPool_NewItem
(
	ObjectPool *pool  // pool
) {
	ASSERT (pool != NULL) ;

	// reuse a deleted item if one is available
	if (array_len (pool->deletedIdx) > 0) {
		return _ObjectPool_ReuseItem (pool) ;
	}

	// make sure we have room for a new item
	if (pool->itemCount >= pool->itemCap) {
		// double the capacity of the pool
		_ObjectPool_AddBlocks (pool, ITEM_COUNT_TO_BLOCK_COUNT (pool->itemCap)) ;
	}

	// get the index of the new allocation
	ObjectID idx = pool->itemCount ;
	pool->itemCount++ ;

	Block *block = GET_ITEM_BLOCK (pool, idx) ;
	uint pos = ITEM_POSITION_WITHIN_BLOCK (idx) ;

	// retrieve a pointer to the item's header
	unsigned char *header = Block_GetItem    (block, pos) ;
	unsigned char *item   = ITEM_FROM_HEADER (header) ;
	ITEM_ID (item) = idx ; // set the item ID

	return item ;
}

void ObjectPool_DeleteItem
(
	ObjectPool *pool,
	void *item
) {
	ASSERT (pool != NULL) ;
	ASSERT (item != NULL) ;

	// get item ID
	ObjectID idx = ITEM_ID (item) ;
	ASSERT (idx < pool->itemCap) ;

	// call item destructor
	if (pool->destructor) {
		pool->destructor (item) ;
	}

	// add ID to deleted list
	array_append (pool->deletedIdx, idx) ;
	pool->itemCount-- ;
	ASSERT (pool->itemCount >= 0) ;
}

void ObjectPool_Free
(
	ObjectPool *pool
) {
	ASSERT (pool != NULL) ;

	for (uint32_t i = 0; i < pool->blockCount; i++) {
		Block_Free (pool->blocks[i]) ;
	}

	rm_free (pool->blocks) ;
	array_free (pool->deletedIdx) ;
	rm_free (pool) ;
}

