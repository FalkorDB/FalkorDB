/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "src/util/arr.h"
#include "src/util/datablock/datablock.h"
#include "src/util/datablock/oo_datablock.h"
#include "src/util/rmalloc.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>

void setup() {
	Alloc_Reset();
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
	TEST_ASSERT(arr_len(dataBlock->deletedIdx) == 1);
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
	TEST_ASSERT(arr_len(dataBlock->deletedIdx) == 0);
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
	TEST_ASSERT(0 == arr_len(dataBlock->deletedIdx));

	// Mark deleted values.
	for(int i = 0; i < 2; i++) {
		DataBlock_MarkAsDeletedOutOfOrder(dataBlock, delete_arr[i]);
	}

	TEST_ASSERT(4 == dataBlock->itemCount);
	TEST_ASSERT(2 == arr_len(dataBlock->deletedIdx));

	// Add another array
	for(int i = 0; i < 4; i++) {
		int *item = (int *)DataBlock_AllocateItemOutOfOrder(dataBlock, insert_arr2[i]);
		*item = insert_arr2[i];
	}

	TEST_ASSERT(8 == dataBlock->itemCount);
	TEST_ASSERT(2 == arr_len(dataBlock->deletedIdx));

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


//------------------------------------------------------------------------------
// claiming ids a peer stated
//------------------------------------------------------------------------------

// a cursor over a plain array, for driving DataBlock_ClaimIds from a test
//
// The real one walks decoded wire segments; this one exists so the claim can
// be tested on its own, including with a cursor that LIES - yields fewer ids
// than it states, or yields them out of order - which the wire cursor cannot
// be made to do.
typedef struct {
	IdCursor        base;
	const uint64_t *ids;
	uint64_t        n;   // how many the array actually holds
	uint64_t        at;
} ArrayCursor;

static void _ArrayCursor_Reset(IdCursor *self) {
	((ArrayCursor *)self)->at = 0;
}

static bool _ArrayCursor_Next(IdCursor *self, uint64_t *id) {
	ArrayCursor *c = (ArrayCursor *)self;
	if(c->at >= c->n) return false;
	*id = c->ids[c->at++];
	return true;
}

// 'stated' is what the cursor claims to hold, which a test can make disagree
// with what it yields
static void _ArrayCursor_Init(ArrayCursor *c, const uint64_t *ids, uint64_t n,
		uint64_t stated) {
	c->base.reset = _ArrayCursor_Reset;
	c->base.next  = _ArrayCursor_Next;
	c->base.count = stated;
	c->ids        = ids;
	c->n          = n;
	c->at         = 0;
}

// a refusal must leave the block exactly as it was - not only unclaimed, but
// un-grown, since growing is the damage an unbounded id would do
typedef struct {
	uint64_t itemCount;
	uint64_t itemCap;
	uint     deleted;
} BlockState;

static BlockState _Snapshot(const DataBlock *db) {
	BlockState st = { db->itemCount, db->itemCap, arr_len(db->deletedIdx) };
	return st;
}

static void _AssertUnchanged(const DataBlock *db, BlockState before) {
	TEST_ASSERT(db->itemCount        == before.itemCount);
	TEST_ASSERT(db->itemCap          == before.itemCap);
	TEST_ASSERT(arr_len(db->deletedIdx) == before.deleted);
}

void test_dataBlockClaimIds() {
	DataBlock *db = DataBlock_New(DATABLOCK_BLOCK_CAP, 16, sizeof(int), NULL);

	const uint64_t ids[] = {0, 1, 2};
	ArrayCursor cur;
	_ArrayCursor_Init(&cur, ids, 3, 3);

	TEST_ASSERT(DataBlock_ClaimIds(db, &cur.base) == true);
	TEST_ASSERT(db->itemCount == 3);
	TEST_ASSERT(DataBlock_DeletedItemsCount(db) == 0);

	// the claimed slots are live and addressable
	for(int i = 0; i < 3; i++) {
		int *v = (int *)DataBlock_ClaimedItem(db, ids[i]);
		TEST_ASSERT(v != NULL);
		*v = i * 10;
	}
	for(int i = 0; i < 3; i++) {
		TEST_ASSERT(*(int *)DataBlock_GetItem(db, ids[i]) == i * 10);
	}

	DataBlock_Free(db);
}

void test_dataBlockClaimTakesIdsOffTheFreeList() {
	DataBlock *db = DataBlock_New(DATABLOCK_BLOCK_CAP, 16, sizeof(int), NULL);

	for(int i = 0; i < 5; i++) {
		uint64_t idx;
		DataBlock_AllocateItem(db, &idx);
	}
	DataBlock_DeleteItem(db, 1);
	DataBlock_DeleteItem(db, 3);
	TEST_ASSERT(DataBlock_DeletedItemsCount(db) == 2);
	TEST_ASSERT(db->itemCount == 3);

	// reuse is the ORDINARY case for a replica, and the free list has to be
	// maintained or the next allocation hands the same id out twice
	const uint64_t ids[] = {1, 3};
	ArrayCursor cur;
	_ArrayCursor_Init(&cur, ids, 2, 2);

	TEST_ASSERT(DataBlock_ClaimIds(db, &cur.base) == true);
	TEST_ASSERT(DataBlock_DeletedItemsCount(db) == 0);
	TEST_ASSERT(db->itemCount == 5);

	// nothing is handed out twice: the next ordinary allocation must be a
	// fresh id, not one this batch just claimed
	uint64_t idx;
	DataBlock_AllocateItem(db, &idx);
	TEST_ASSERT(idx != 1 && idx != 3);

	DataBlock_Free(db);
}

void test_dataBlockClaimRefusesALiveId() {
	DataBlock *db = DataBlock_New(DATABLOCK_BLOCK_CAP, 16, sizeof(int), NULL);

	for(int i = 0; i < 4; i++) {
		uint64_t idx;
		DataBlock_AllocateItem(db, &idx);
	}
	DataBlock_DeleteItem(db, 2);

	// id 0 is live - that is divergence, not reuse
	const uint64_t ids[] = {0, 2};
	ArrayCursor cur;
	_ArrayCursor_Init(&cur, ids, 2, 2);

	BlockState before = _Snapshot(db);
	TEST_ASSERT(DataBlock_ClaimIds(db, &cur.base) == false);
	_AssertUnchanged(db, before);

	// and id 2, which WAS claimable, is still on the free list
	TEST_ASSERT(DataBlock_DeletedItemsCount(db) == 1);
	TEST_ASSERT(DataBlock_GetItem(db, 2) == NULL);  // still deleted

	DataBlock_Free(db);
}

void test_dataBlockClaimRefusesARepeatedId() {
	DataBlock *db = DataBlock_New(DATABLOCK_BLOCK_CAP, 16, sizeof(int), NULL);

	// naming one id twice would put two entities on one slot
	const uint64_t ids[] = {0, 1, 1};
	ArrayCursor cur;
	_ArrayCursor_Init(&cur, ids, 3, 3);

	BlockState before = _Snapshot(db);
	TEST_ASSERT(DataBlock_ClaimIds(db, &cur.base) == false);
	_AssertUnchanged(db, before);

	DataBlock_Free(db);
}

void test_dataBlockClaimRefusesIdsOutOfOrder() {
	DataBlock *db = DataBlock_New(DATABLOCK_BLOCK_CAP, 16, sizeof(int), NULL);

	// the contract is ascending: out of order, the duplicate check and the
	// gap fill below are both wrong, so it is refused rather than sorted
	const uint64_t ids[] = {2, 0, 1};
	ArrayCursor cur;
	_ArrayCursor_Init(&cur, ids, 3, 3);

	BlockState before = _Snapshot(db);
	TEST_ASSERT(DataBlock_ClaimIds(db, &cur.base) == false);
	_AssertUnchanged(db, before);

	DataBlock_Free(db);
}

void test_dataBlockClaimAcceptsTheSparseRecordsGroupingProduces() {
	DataBlock *db = DataBlock_New(DATABLOCK_BLOCK_CAP, 16, sizeof(int), NULL);

	// A RECORD'S IDS ARE NOT DENSE, and no bound taken from the record's own
	// count can accept them. CREATE (:A), (:B), (:A) allocates 0,1,2 and
	// travels as TWO records, because v3 sends one per shape: ':A' holding
	// {0, 2} and ':B' holding {1}. An earlier version of this function
	// refused the first of them on the grounds that a record creating two
	// entities could not reach id 2 - which is true of a whole payload and
	// false of one record.
	const uint64_t a_ids[] = {0, 2};
	ArrayCursor a;
	_ArrayCursor_Init(&a, a_ids, 2, 2);
	TEST_ASSERT(DataBlock_ClaimIds(db, &a.base) == true);

	// slot 1 is skipped, so it is FREE rather than lost - which is precisely
	// what lets the second record have it
	TEST_ASSERT(db->itemCount == 2);
	TEST_ASSERT(DataBlock_DeletedItemsCount(db) == 1);
	TEST_ASSERT(db->deletedIdx[0] == 1);

	const uint64_t b_ids[] = {1};
	ArrayCursor b;
	_ArrayCursor_Init(&b, b_ids, 1, 1);
	TEST_ASSERT(DataBlock_ClaimIds(db, &b.base) == true);

	// and the pair lands exactly where the primary is: three live, none free
	TEST_ASSERT(db->itemCount == 3);
	TEST_ASSERT(DataBlock_DeletedItemsCount(db) == 0);
	for(int i = 0; i < 3; i++) TEST_ASSERT(DataBlock_GetItem(db, i) != NULL);

	DataBlock_Free(db);
}

void test_dataBlockClaimAcceptsAWholeInterleavedShapeGroup() {
	DataBlock *db = DataBlock_New(DATABLOCK_BLOCK_CAP, 16, sizeof(int), NULL);

	// the same thing at the scale an edge create produces: four of
	// CREATE (:A)-[:R]->(:B) allocates 0..7, and the ':A' record holds every
	// even id while ':B' holds every odd one
	const uint64_t evens[] = {0, 2, 4, 6};
	ArrayCursor a;
	_ArrayCursor_Init(&a, evens, 4, 4);
	TEST_ASSERT(DataBlock_ClaimIds(db, &a.base) == true);
	TEST_ASSERT(DataBlock_DeletedItemsCount(db) == 3);  // 1, 3, 5

	const uint64_t odds[] = {1, 3, 5, 7};
	ArrayCursor b;
	_ArrayCursor_Init(&b, odds, 4, 4);
	TEST_ASSERT(DataBlock_ClaimIds(db, &b.base) == true);

	TEST_ASSERT(db->itemCount == 8);
	TEST_ASSERT(DataBlock_DeletedItemsCount(db) == 0);
	for(int i = 0; i < 8; i++) TEST_ASSERT(DataBlock_GetItem(db, i) != NULL);

	DataBlock_Free(db);
}

void test_dataBlockClaimRefusesTheInvalidId() {
	DataBlock *db = DataBlock_New(DATABLOCK_BLOCK_CAP, 16, sizeof(int), NULL);

	// THE ONE ID THAT IS REFUSED, and not as a guess about what the peer could
	// have reached: UINT64_MAX is INVALID_ENTITY_ID, so no entity carries it,
	// and it is where DataBlock_Ensure's (1 + idx) wraps to zero and the write
	// lands off the end of the block
	const uint64_t ids[] = {UINT64_MAX};
	ArrayCursor cur;
	_ArrayCursor_Init(&cur, ids, 1, 1);

	BlockState before = _Snapshot(db);
	TEST_ASSERT(DataBlock_ClaimIds(db, &cur.base) == false);
	_AssertUnchanged(db, before);

	DataBlock_Free(db);
}

void test_dataBlockClaimFreesTheIdsItSkips() {
	DataBlock *db = DataBlock_New(DATABLOCK_BLOCK_CAP, 16, sizeof(int), NULL);

	for(int i = 0; i < 4; i++) {
		uint64_t idx;
		DataBlock_AllocateItem(db, &idx);
	}
	DataBlock_DeleteItem(db, 1);
	// live: 0, 2, 3   free: 1   high water: 4

	// one id off the free list and one past the high-water mark, which leaves
	// slot 4 behind. A SKIPPED SLOT MUST BECOME FREE: leaked, it would make
	// this replica's own allocation diverge the first time it allocates.
	const uint64_t ids[] = {1, 5};
	ArrayCursor cur;
	_ArrayCursor_Init(&cur, ids, 2, 2);

	TEST_ASSERT(DataBlock_ClaimIds(db, &cur.base) == true);
	TEST_ASSERT(db->itemCount == 5);

	// 1 was taken off the list, 4 was put on it
	TEST_ASSERT(DataBlock_DeletedItemsCount(db) == 1);
	TEST_ASSERT(db->deletedIdx[0] == 4);
	TEST_ASSERT(DataBlock_GetItem(db, 4) == NULL);  // marked deleted, not lost

	// so the next ordinary allocation reuses it rather than losing it
	uint64_t idx;
	DataBlock_AllocateItem(db, &idx);
	TEST_ASSERT(idx == 4);

	DataBlock_Free(db);
}

void test_dataBlockClaimRefusesAWalkThatDisagreesWithItsCount() {
	DataBlock *db = DataBlock_New(DATABLOCK_BLOCK_CAP, 16, sizeof(int), NULL);

	// states three, yields two. Left unchecked the caller would write three
	// rows into two claimed slots and report success.
	const uint64_t ids[] = {0, 1};
	ArrayCursor cur;
	_ArrayCursor_Init(&cur, ids, 2, 3);

	BlockState before = _Snapshot(db);
	TEST_ASSERT(DataBlock_ClaimIds(db, &cur.base) == false);
	_AssertUnchanged(db, before);

	// and the other way round: states two, yields three
	ArrayCursor over;
	const uint64_t more[] = {0, 1, 2};
	_ArrayCursor_Init(&over, more, 3, 2);

	before = _Snapshot(db);
	TEST_ASSERT(DataBlock_ClaimIds(db, &over.base) == false);
	_AssertUnchanged(db, before);

	DataBlock_Free(db);
}

TEST_LIST = {
	{"dataBlockNew", test_dataBlockNew},
	{"dataBlockAddItem", test_dataBlockAddItem },
	{"dataBlockScan", test_dataBlockScan},
	{"dataBlockRemoveItem", test_dataBlockRemoveItem},
	{"dataBlockOutOfOrderBuilding", test_dataBlockOutOfOrderBuilding},
	{"dataBlockClaimIds", test_dataBlockClaimIds},
	{"dataBlockClaimTakesIdsOffTheFreeList", test_dataBlockClaimTakesIdsOffTheFreeList},
	{"dataBlockClaimRefusesALiveId", test_dataBlockClaimRefusesALiveId},
	{"dataBlockClaimRefusesARepeatedId", test_dataBlockClaimRefusesARepeatedId},
	{"dataBlockClaimRefusesIdsOutOfOrder", test_dataBlockClaimRefusesIdsOutOfOrder},
	{"dataBlockClaimAcceptsTheSparseRecordsGroupingProduces", test_dataBlockClaimAcceptsTheSparseRecordsGroupingProduces},
	{"dataBlockClaimAcceptsAWholeInterleavedShapeGroup", test_dataBlockClaimAcceptsAWholeInterleavedShapeGroup},
	{"dataBlockClaimRefusesTheInvalidId", test_dataBlockClaimRefusesTheInvalidId},
	{"dataBlockClaimFreesTheIdsItSkips", test_dataBlockClaimFreesTheIdsItSkips},
	{"dataBlockClaimRefusesAWalkThatDisagreesWithItsCount", test_dataBlockClaimRefusesAWalkThatDisagreesWithItsCount},
	{NULL, NULL}
};

