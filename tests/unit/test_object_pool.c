/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "mock_log.h"
#include "src/util/arr.h"
#include "src/util/rmalloc.h"
#include "src/util/block_struct.h"
#include "src/util/object_pool/object_pool.h"

#include <stdio.h>

void setup() {
	Alloc_Reset () ;
	Logging_Reset () ;
}

#define TEST_INIT setup();
#include "acutest.h"

void test_objectPoolNew() {
	// Create a new ObjectPool capable of holding at least 1024 integer items.
	uint item_size = sizeof(uint);
	ObjectPool *object_pool = ObjectPool_New(1024, item_size, NULL);

	TEST_ASSERT(object_pool->itemCount == 0);     // No items have been added.
	TEST_ASSERT(object_pool->itemCap >= 1024);

	// The size of items in the pool is greater to accommodate an internal header ID.
	uint header_size = sizeof(uint64_t);
	TEST_ASSERT(object_pool->itemSize == item_size + header_size);
	TEST_ASSERT(object_pool->blockCount >= 1024 / POOL_BLOCK_CAP);

	// Verify that blocks are properly chained together.
	for(uint i = 0; i < object_pool->blockCount; i++) {
		Block *block = object_pool->blocks[i];
		TEST_ASSERT(block->itemSize == object_pool->itemSize);
		TEST_ASSERT(block->data != NULL);
		if(i > 0) {
			TEST_ASSERT(object_pool->blocks[i - 1]->next == object_pool->blocks[i]);
		}
	}

	ObjectPool_Free(object_pool);
}

void test_objectPoolAddItem() {
	ObjectPool *object_pool = ObjectPool_New(256, sizeof(uint), NULL);
	uint item_count = 128;
	uint final_item_count = item_count *= 16;
	uint *item_pointers[final_item_count];
	// Add items to pool.
	for(uint i = 0; i < item_count; i++) {
		uint *item = (uint *)ObjectPool_NewItem(object_pool);
		*item = i;
		item_pointers[i] = item;
	}

	// Add enough items to cause the ObjectPool to reallocate.
	uint prev_count = object_pool->itemCount;
	item_count = final_item_count;

	// Add new items.
	for(uint i = prev_count; i < item_count; i++) {
		uint *item = (uint *)ObjectPool_NewItem(object_pool);
		*item = i;
		item_pointers[i] = item;
	}

	// Validate that no items have been modified.
	for(uint i = 0; i < item_count; i++) {
		TEST_ASSERT(*item_pointers[i] == i);
	}
	ObjectPool_Free(object_pool);
}

void test_objectPoolRemoveItem() {
	ObjectPool *object_pool = ObjectPool_New(1024, sizeof(uint), NULL);
	uint item_count = 32;

	uint *item_pointers[item_count];
	// Set items.
	for(uint i = 0; i < item_count; i++) {
		uint *item = (uint *)ObjectPool_NewItem(object_pool);
		*item = i;
		item_pointers[i] = item;
	}

	// Validate that no items have been modified.
	for(uint i = 0; i < item_count; i++) {
		TEST_ASSERT(*item_pointers[i] == i);
	}

	// Delete an entry from the pool and perform validations..
	uint delete_idx = 10;
	ObjectPool_DeleteItem(object_pool, item_pointers[delete_idx]);
	TEST_ASSERT(object_pool->itemCount == item_count - 1);
	TEST_ASSERT(array_len(object_pool->deletedIdx) == 1);
	// Index 10 should be added to ObjectPool deletedIdx array.
	TEST_ASSERT(object_pool->deletedIdx[0] == 10);

	// Add a new item and verify that the deleted entry is reused.
	uint *new_item = (uint *)ObjectPool_NewItem(object_pool);
	TEST_ASSERT(object_pool->itemCount == item_count);
	TEST_ASSERT(array_len(object_pool->deletedIdx) == 0);
	// Verify that the new entry is zeroed.
	TEST_ASSERT(*new_item == 0);

	// cleanup
	ObjectPool_Free(object_pool);
}

TEST_LIST = {
	{"objectPoolNew", test_objectPoolNew},
	{"objectPoolAddItem", test_objectPoolAddItem},
	{"objectPoolRemoveItem", test_objectPoolRemoveItem},
	{NULL, NULL}
};

