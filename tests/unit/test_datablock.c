/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "db.h"
#include "mock_log.h"
#include "src/util/arr.h"
#include "src/util/rmalloc.h"
#include "src/util/block_struct.h"
#include "src/util/datablock/datablock.h"
#include "src/util/datablock/oo_datablock.h"

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

void setup() {
	Alloc_Reset () ;
	Logging_Reset () ;
}
#define TEST_INIT setup();
#include "acutest.h"

#define DATABLOCK_BLOCK_CAP 16384

void test_dataBlockNew() {
	// create a new data block, which can hold at least 1024 items
	// each item is an integer.
	size_t itemSize = sizeof (int) ;
	DataBlock *dataBlock = DataBlock_New (DATABLOCK_BLOCK_CAP, 1024, itemSize, NULL) ;

	TEST_ASSERT (dataBlock->itemCount == 0) ;     // no items were added
	TEST_ASSERT (dataBlock->itemCap >= 1024) ;
	TEST_ASSERT (dataBlock->itemSize == itemSize) ;
	TEST_ASSERT (dataBlock->blockCount >= 1024 / DATABLOCK_BLOCK_CAP) ;
	TEST_ASSERT (DataBlock_HasStorage (dataBlock) == false) ;

	for(int i = 0; i < dataBlock->blockCount; i++) {
		Block *block = dataBlock->blocks[i];
		TEST_ASSERT(block->itemSize == dataBlock->itemSize);
		TEST_ASSERT(block->data != NULL);
		if(i > 0) {
			TEST_ASSERT(dataBlock->blocks[i - 1]->next == dataBlock->blocks[i]);
		}
	}

	DataBlock_Free (&dataBlock) ;
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

	DataBlock_Free (&dataBlock) ;
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

	DataBlock_Free (&dataBlock) ;
	DataBlockIterator_Free (&it) ;
}

void test_dataBlockRemoveItem(void) {
	DataBlock *dataBlock =
		DataBlock_New (DATABLOCK_BLOCK_CAP, 1024, sizeof(int), NULL) ;
	uint itemCount = 32 ;
	DataBlock_Accommodate (dataBlock, itemCount) ;

	// set items
	for (int i = 0; i < itemCount; i++) {
		int *item = (int *)DataBlock_AllocateItem (dataBlock, NULL) ;
		*item = i ;
	}

	// validate item at position 0
	int *item = (int *)DataBlock_GetItem (dataBlock, 0) ;
	TEST_ASSERT (item != NULL) ;
	TEST_ASSERT (*item == 0) ;

	// remove item at position 0 and perform validations
	// Index 0 should be added to datablock deletedIdx array.
	DataBlock_DeleteItem (dataBlock, 0) ;
	TEST_ASSERT (dataBlock->itemCount == itemCount - 1) ;
	TEST_ASSERT (array_len (dataBlock->deletedIdx) == 1) ;
	TEST_ASSERT (DataBlock_ItemIsDeleted (dataBlock, 0)) ;

	// Try to get item from deleted cell.
	item = (int *)DataBlock_GetItem(dataBlock, 0);
	TEST_ASSERT(item == NULL);

	// Iterate over datablock, deleted item should be skipped.
	DataBlockIterator *it = DataBlock_Scan(dataBlock);
	uint counter = 0;
	while(DataBlockIterator_Next(it, NULL)) counter++;
	TEST_ASSERT(counter == itemCount - 1);
	DataBlockIterator_Free (&it) ;

	// There's no harm in deleting a deleted item.
	DataBlock_DeleteItem(dataBlock, 0);

	// add a new item, expecting deleted cell to be reused
	int *newItem = (int *)DataBlock_AllocateItem (dataBlock, NULL) ;
	TEST_ASSERT (dataBlock->itemCount == itemCount) ;
	TEST_ASSERT (array_len(dataBlock->deletedIdx) == 0) ;
	TEST_ASSERT ((void *)newItem == (void *)((dataBlock->blocks[0]->elements))) ;

	it = DataBlock_Scan(dataBlock);
	counter = 0;
	while(DataBlockIterator_Next(it, NULL)) counter++;
	TEST_ASSERT(counter == itemCount);
	DataBlockIterator_Free (&it) ;

	// cleanup
	DataBlock_Free (&dataBlock) ;
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

	DataBlock_Free (&dataBlock) ;
	DataBlockIterator_Free (&it) ;
}

void test_dataBlockItemOffload (void) {
	// test datablock item offloading and loading

	// create a new datablock
	size_t n = 16 ;
	DataBlock *dataBlock =
		DataBlock_New (DATABLOCK_BLOCK_CAP, n, sizeof(uint64_t), NULL) ;
	TEST_ASSERT (dataBlock != NULL) ;

	//--------------------------------------------------------------------------
	// set datablock disk storage
	//--------------------------------------------------------------------------

	// initialize tidesdb
	tidesdb_t *db = NULL ;

	// tidesdb config
	tidesdb_config_t config = tidesdb_default_config () ;
	config.db_path   = "./tidesdb_test_datablock" ;
	config.log_level = TDB_LOG_ERROR ;

	// delete test database folder if it exists
	struct stat st ;
	if (stat (config.db_path, &st) == 0) {
		// folder exists, remove it recursively
		char cmd[256] ;
		snprintf (cmd, sizeof (cmd), "rm -rf %s", config.db_path) ;
		int rm_res = system (cmd) ;
		TEST_ASSERT (rm_res == 0) ;
	}

	int res = tidesdb_open (&config, &db) ;
	TEST_ASSERT (res == 0) ;

	tidesdb_column_family_config_t cf_config =
		tidesdb_default_column_family_config () ;
	res = tidesdb_create_column_family (db, "col", &cf_config) ;
	TEST_ASSERT (res == 0) ;

	tidesdb_column_family_t *cf = tidesdb_get_column_family (db, "col") ;
	DataBlock_SetStorage (dataBlock, cf) ;

	//--------------------------------------------------------------------------
	// load items
	//--------------------------------------------------------------------------

	uint64_t  ids   [n] ;
	uint64_t *items [n] ;

	for (int i = 0; i < n; i++) {
		uint64_t *item = (uint64_t*) DataBlock_AllocateItem (dataBlock, ids + i) ;
		*item = i ;
		items[i] = item ;
	}

	TEST_ASSERT (n == DataBlock_ItemCount (dataBlock)) ;

	// get offloaded status
	bool offloaded[n] ;
	DataBlock_IsOffloaded (offloaded, dataBlock, ids, n) ;

	// all items should be loaded
	for (int i = 0; i < n; i++) {
		TEST_ASSERT (offloaded[i] == false) ;
	}

	//--------------------------------------------------------------------------
	// offload items to disk
	//--------------------------------------------------------------------------

	tidesdb_txn_t *txn = NULL ;
	res = tidesdb_txn_begin (db, &txn) ;
	TEST_ASSERT (res == 0) ;

	size_t n_indices = 0 ;
	uint64_t indices[n/2] ;
	for (int i = 0; i < n; i+=2) {
		indices[n_indices++] = i ;

		uint64_t *item = DataBlock_GetItem (dataBlock, i) ;
		TEST_ASSERT (*item == i) ;

		res = tidesdb_txn_put (txn, cf, (const uint8_t *)(&i),
				sizeof (int), (const uint8_t *)item, sizeof (uint64_t), -1) ;
		TEST_ASSERT (res == 0) ;
	}

	res = tidesdb_txn_commit (txn);
	TEST_ASSERT (res == 0) ;

	tidesdb_txn_free (txn) ;
	TEST_ASSERT (res == 0) ;

	DataBlock_MarkOffloaded (dataBlock, indices, n_indices) ;

	//--------------------------------------------------------------------------
	// verify items status
	//--------------------------------------------------------------------------

	DataBlock_IsOffloaded (offloaded, dataBlock, ids, n) ;

	// all even items should be offloaded
	// odd items should be loaded
	for (int i = 0; i < n; i++) {
		if (i % 2 == 0) {
			TEST_ASSERT (offloaded[i] == true) ;
		} else {
			TEST_ASSERT (offloaded[i] == false) ;
		}
	}
	TEST_ASSERT (n == DataBlock_ItemCount (dataBlock)) ;

	//--------------------------------------------------------------------------
	// clean up
	//--------------------------------------------------------------------------

	DataBlock_Free (&dataBlock) ;

	res = tidesdb_drop_column_family (db, "col") ;
	TEST_ASSERT (res == 0) ;

	res = tidesdb_close (db) ;
	TEST_ASSERT (res == 0) ;
}

TEST_LIST = {
	{"dataBlockNew", test_dataBlockNew},
	{"dataBlockAddItem", test_dataBlockAddItem },
	{"dataBlockScan", test_dataBlockScan},
	{"dataBlockRemoveItem", test_dataBlockRemoveItem},
	{"dataBlockOutOfOrderBuilding", test_dataBlockOutOfOrderBuilding},
	{"dataBlockItemOffload", test_dataBlockItemOffload},
	{NULL, NULL}
};

