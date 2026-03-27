/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "src/util/arr.h"
#include "src/util/datablock/datablock.h"
#include "src/util/datablock/oo_datablock.h"
#include "src/util/rmalloc.h"
#include "tests/utils/mock_log.h"

#include <stdio.h>
#include <string.h>

void setup() {
	Alloc_Reset();
	Logging_Reset();
}
#define TEST_INIT setup();
#include "acutest.h"

#define DATABLOCK_BLOCK_CAP 16384

void test_dataBlockNew() {
	// create a new data block, which can hold at least 1024 items
	// each item is an integer.
	size_t itemSize = sizeof(int);
	DataBlock *dataBlock = DataBlock_New(DATABLOCK_BLOCK_CAP, 1024, itemSize, NULL);

	TEST_ASSERT(dataBlock->itemCount == 0);     // No items were added.
	TEST_ASSERT(dataBlock->itemCap >= 1024);
	TEST_ASSERT(dataBlock->itemSize == itemSize + ITEM_HEADER_SIZE);
	TEST_ASSERT(dataBlock->blockCount >= 1024 / DATABLOCK_BLOCK_CAP);

	for(int i = 0; i < dataBlock->blockCount; i++) {
		Block *block = dataBlock->blocks[i];
		TEST_ASSERT(block->itemSize == dataBlock->itemSize);
		TEST_ASSERT(block->data != NULL);
		if(i > 0) {
			TEST_ASSERT(dataBlock->blocks[i - 1]->next == dataBlock->blocks[i]);
		}
	}

	DataBlock_Free(dataBlock);
}

void test_dataBlockAddItem() {
	DataBlock *dataBlock = DataBlock_New(DATABLOCK_BLOCK_CAP, 1024, sizeof(int), NULL);
	size_t itemCount = 512;
	DataBlock_Accommodate(dataBlock, itemCount);
	TEST_ASSERT(dataBlock->itemCap >= itemCount);

	// Set items.
	for(int i = 0 ; i < itemCount; i++) {
		int *value = (int *)DataBlock_AllocateItem(dataBlock, NULL);
		*value = i;
	}

	// Read items.
	for(int i = 0 ; i < itemCount; i++) {
		int *value = (int *)DataBlock_GetItem(dataBlock, i);
		TEST_ASSERT(*value == i);
	}

	// Add enough item to cause datablock to re-allocate.
	size_t prevItemCount = dataBlock->itemCount;
	itemCount *= 64;
	DataBlock_Accommodate(dataBlock, itemCount);
	TEST_ASSERT(dataBlock->itemCap >= prevItemCount + itemCount);

	// Set items.
	for(int i = 0 ; i < itemCount; i++) {
		int *value = (int *)DataBlock_AllocateItem(dataBlock, NULL);
		*value = prevItemCount + i;
	}

	// Read items.
	for(int i = 0 ; i < dataBlock->itemCount; i++) {
		int *value = (int *)DataBlock_GetItem(dataBlock, i);
		TEST_ASSERT(*value == i);
	}

	DataBlock_Free(dataBlock);
}

void test_dataBlockScan() {
	DataBlock *dataBlock = DataBlock_New(DATABLOCK_BLOCK_CAP, 1024, sizeof(int), NULL);
	size_t itemCount = 2048;
	DataBlock_Accommodate(dataBlock, itemCount);

	// Set items.
	for(int i = 0 ; i < itemCount; i++) {
		int *item = (int *)DataBlock_AllocateItem(dataBlock, NULL);
		*item = i;
	}

	// Scan through items.
	int count = 0;		// items iterated so far
	int *item = NULL;	// current iterated item
	uint64_t idx = 0;	// iterated item index

	DataBlockIterator *it = DataBlock_Scan(dataBlock);
	while((item = (int *)DataBlockIterator_Next(it, &idx))) {
		TEST_ASSERT(count == idx);
		TEST_ASSERT(*item == count);
		count++;
	}
	TEST_ASSERT(count == itemCount);

	item = (int *)DataBlockIterator_Next(it, NULL);
	TEST_ASSERT(item == NULL);

	DataBlockIterator_Reset(it);

	// Re-scan through items.
	count = 0;
	item = NULL;
	while((item = (int *)DataBlockIterator_Next(it, &idx))) {
		TEST_ASSERT(count == idx);
		TEST_ASSERT(*item == count);
		count++;
	}
	TEST_ASSERT(count == itemCount);

	item = (int *)DataBlockIterator_Next(it, NULL);
	TEST_ASSERT(item == NULL);

	DataBlock_Free(dataBlock);
	DataBlockIterator_Free(it);
}

void test_dataBlockRemoveItem() {
	DataBlock *dataBlock = DataBlock_New(DATABLOCK_BLOCK_CAP, 1024, sizeof(int), NULL);
	uint itemCount = 32;
	DataBlock_Accommodate(dataBlock, itemCount);

	// Set items.
	for(int i = 0 ; i < itemCount; i++) {
		int *item = (int *)DataBlock_AllocateItem(dataBlock, NULL);
		*item = i;
	}

	// Validate item at position 0.
	int *item = (int *)DataBlock_GetItem(dataBlock, 0);
	TEST_ASSERT(item != NULL);
	TEST_ASSERT(*item == 0);

	// Remove item at position 0 and perform validations
	// Index 0 should be added to datablock deletedIdx array.
	DataBlock_DeleteItem(dataBlock, 0);
	TEST_ASSERT(dataBlock->itemCount == itemCount - 1);
	TEST_ASSERT(array_len(dataBlock->deletedIdx) == 1);
	DataBlockItemHeader *header = (DataBlockItemHeader *)dataBlock->blocks[0]->data;
	TEST_ASSERT(IS_ITEM_DELETED(header));

	// Try to get item from deleted cell.
	item = (int *)DataBlock_GetItem(dataBlock, 0);
	TEST_ASSERT(item == NULL);

	// Iterate over datablock, deleted item should be skipped.
	DataBlockIterator *it = DataBlock_Scan(dataBlock);
	uint counter = 0;
	while(DataBlockIterator_Next(it, NULL)) counter++;
	TEST_ASSERT(counter == itemCount - 1);
	DataBlockIterator_Free(it);

	// There's no harm in deleting a deleted item.
	DataBlock_DeleteItem(dataBlock, 0);

	// Add a new item, expecting deleted cell to be reused.
	int *newItem = (int *)DataBlock_AllocateItem(dataBlock, NULL);
	TEST_ASSERT(dataBlock->itemCount == itemCount);
	TEST_ASSERT(array_len(dataBlock->deletedIdx) == 0);
	TEST_ASSERT((void *)newItem == (void *)((dataBlock->blocks[0]->data) + ITEM_HEADER_SIZE));

	it = DataBlock_Scan(dataBlock);
	counter = 0;
	while(DataBlockIterator_Next(it, NULL)) counter++;
	TEST_ASSERT(counter == itemCount);
	DataBlockIterator_Free(it);

	// Cleanup.
	DataBlock_Free(dataBlock);
}

void test_dataBlockOutOfOrderBuilding() {
	// This test checks for a fragmented, data block out of order re-construction.
	DataBlock *dataBlock = DataBlock_New(DATABLOCK_BLOCK_CAP, 1, sizeof(int), NULL);
	int insert_arr1[4] = {8, 2, 3, 6};
	int delete_arr[2] = {4, 7};
	int insert_arr2[4] = {9, 1, 5, 0};
	int expected[8] = {0, 1, 2, 3, 5, 6, 8, 9};

	// Insert the first array.
	for(int i = 0; i < 4; i++) {
		int *item = (int *)DataBlock_AllocateItemOutOfOrder(dataBlock, insert_arr1[i]);
		*item = insert_arr1[i];
	}
	TEST_ASSERT(4 == dataBlock->itemCount);
	TEST_ASSERT(0 == array_len(dataBlock->deletedIdx));

	// Mark deleted values.
	for(int i = 0; i < 2; i++) {
		DataBlock_MarkAsDeletedOutOfOrder(dataBlock, delete_arr[i]);
	}

	TEST_ASSERT(4 == dataBlock->itemCount);
	TEST_ASSERT(2 == array_len(dataBlock->deletedIdx));

	// Add another array
	for(int i = 0; i < 4; i++) {
		int *item = (int *)DataBlock_AllocateItemOutOfOrder(dataBlock, insert_arr2[i]);
		*item = insert_arr2[i];
	}

	TEST_ASSERT(8 == dataBlock->itemCount);
	TEST_ASSERT(2 == array_len(dataBlock->deletedIdx));

	// Validate
	DataBlockIterator *it = DataBlock_Scan(dataBlock);
	for(int i = 0; i < 8; i++) {
		int *item = (int *)DataBlockIterator_Next(it, NULL);
		TEST_ASSERT(*item == expected[i]);
	}
	TEST_ASSERT(!DataBlockIterator_Next(it, NULL));

	TEST_ASSERT(dataBlock->deletedIdx[0] == 4 || dataBlock->deletedIdx[0] == 7);
	TEST_ASSERT(dataBlock->deletedIdx[1] == 4 || dataBlock->deletedIdx[1] == 7);

	DataBlock_Free(dataBlock);
	DataBlockIterator_Free(it);
}

TEST_LIST = {
	{"dataBlockNew", test_dataBlockNew},
	{"dataBlockAddItem", test_dataBlockAddItem },
	{"dataBlockScan", test_dataBlockScan},
	{"dataBlockRemoveItem", test_dataBlockRemoveItem},
	{"dataBlockOutOfOrderBuilding", test_dataBlockOutOfOrderBuilding},
	{NULL, NULL}
};

