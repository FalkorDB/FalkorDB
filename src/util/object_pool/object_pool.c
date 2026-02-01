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

// Given an item header, retrieve the item data.
#define ITEM_FROM_HEADER(header) ((header) + sizeof(ObjectID))

static void _ObjectPool_AddBlocks
(
	ObjectPool *pool,
	uint blockCount
) {
	ASSERT(pool && blockCount > 0);

	uint prevBlockCount = pool->blockCount;
	pool->blockCount += blockCount;
	if(!pool->blocks) {
		pool->blocks = rm_malloc(sizeof(Block *) * pool->blockCount);
	} else {
		pool->blocks = rm_realloc(pool->blocks,
				sizeof(Block *) * pool->blockCount);
	}

	uint i;
	for(i = prevBlockCount; i < pool->blockCount; i++) {
		pool->blocks[i] = Block_New(pool->itemSize, POOL_BLOCK_CAP);
		if(i > 0) pool->blocks[i - 1]->next = pool->blocks[i];
	}
	pool->blocks[i - 1]->next = NULL;

	pool->itemCap = pool->blockCount * POOL_BLOCK_CAP;
}

// clear a deleted item and recycle it to the caller
static void *_ObjectPool_ReuseItem
(
	ObjectPool *pool
) {
	ObjectID idx = array_pop(pool->deletedIdx);

	pool->itemCount++;

	Block *block = GET_ITEM_BLOCK(pool, idx);
	uint pos = ITEM_POSITION_WITHIN_BLOCK(idx);

	// retrieve a pointer to the item's header
	unsigned char *item_header = block->data + (pos * block->itemSize);
	unsigned char *item = ITEM_FROM_HEADER(item_header);
	ASSERT(ITEM_ID(item) == idx); // the item ID should not change on reuse

	// zero-set the item being returned
	memset(item, 0, block->itemSize - HEADER_SIZE);

	return item;
}

ObjectPool *ObjectPool_New
(
	uint64_t itemCap,
	uint itemSize,
	void (*fp)(void *)
) {
	ObjectPool *pool = rm_malloc(sizeof(ObjectPool));

	pool->itemCount  = 0;
	pool->itemSize   = itemSize + HEADER_SIZE;  // accommodate the header
	pool->blockCount = 0;
	pool->blocks     = NULL;
	pool->deletedIdx = array_new(uint64_t, 128);
	pool->destructor = fp;

	_ObjectPool_AddBlocks(pool, ITEM_COUNT_TO_BLOCK_COUNT(itemCap));

	return pool;
}

// borrow `n` records from `pool`
void ObjectPool_NewItems
(
	void **records,    // [input/output] records to return
	ObjectPool *pool,  // object pool
	uint32_t n         // number of records to borrow
) {
	ASSERT (n       >  0) ;
	ASSERT (pool    != NULL) ;
	ASSERT (records != NULL) ;
	ASSERT (array_len(records) >= n) ;

	//--------------------------------------------------------------------------
	// recycle items
	//--------------------------------------------------------------------------

	// number of deleted items
	uint32_t n_deleted = array_len (pool->deletedIdx) ;
	uint32_t m = MIN (n, n_deleted) ;

	for (uint32_t i = 0; i < m; i++) {
		records[i] = _ObjectPool_ReuseItem (pool) ;
	}

	//--------------------------------------------------------------------------
	// borrow remaining
	//--------------------------------------------------------------------------

	// borrow remaining
	uint32_t remaining = n - m ;
	if (remaining == 0) {
		return ;
	}

	// check capacity and grow if needed
	uint32_t available = pool->itemCap - pool->itemCount;
	if (unlikely (remaining > available)) {
		_ObjectPool_AddBlocks (pool,
				ITEM_COUNT_TO_BLOCK_COUNT (remaining - available)) ;
	}

	for (uint32_t i = 0; i < remaining; i++) {
		// get the index of the new allocation
		ObjectID idx = pool->itemCount;
		pool->itemCount++;

		Block *block = GET_ITEM_BLOCK (pool, idx) ;
		uint pos = ITEM_POSITION_WITHIN_BLOCK (idx) ;

		// retrieve a pointer to the item's header
		unsigned char *item_header = block->data + (pos * block->itemSize);
		unsigned char *item = ITEM_FROM_HEADER(item_header);

		ITEM_ID(item) = idx; // set the item ID
		records[m + i] = item ;
	}
}

void *ObjectPool_NewItem
(
	ObjectPool *pool
) {
	// reuse a deleted item if one is available
	if(array_len(pool->deletedIdx)) return _ObjectPool_ReuseItem(pool);

	// make sure we have room for a new item
	if(pool->itemCount >= pool->itemCap) {
		// Double the capacity of the pool.
		_ObjectPool_AddBlocks(pool, ITEM_COUNT_TO_BLOCK_COUNT(pool->itemCap));
	}

	// get the index of the new allocation
	ObjectID idx = pool->itemCount;
	pool->itemCount++;

	Block *block = GET_ITEM_BLOCK(pool, idx);
	uint pos = ITEM_POSITION_WITHIN_BLOCK(idx);

	// retrieve a pointer to the item's header
	unsigned char *item_header = block->data + (pos * block->itemSize);
	unsigned char *item = ITEM_FROM_HEADER(item_header);
	ITEM_ID(item) = idx; // set the item ID

	return item;
}

void ObjectPool_DeleteItem
(
	ObjectPool *pool,
	void *item
) {
	ASSERT (pool != NULL) ;

	// get item ID
	ObjectID idx = ITEM_ID (item) ;

	// call item destructor
	if (pool->destructor) {
		pool->destructor (item) ;
	}

	// add ID to deleted list
	array_append (pool->deletedIdx, idx) ;
	pool->itemCount-- ;
}

void ObjectPool_Free
(
	ObjectPool *pool
) {
	for (uint i = 0; i < pool->blockCount; i++) {
		Block_Free (pool->blocks[i]) ;
	}

	rm_free (pool->blocks) ;
	array_free (pool->deletedIdx) ;
	rm_free (pool) ;
}

