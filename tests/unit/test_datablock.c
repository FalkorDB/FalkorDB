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
	// The write that freed the cell has to end first: a write is never handed
	// back an id it freed itself.
	DataBlock_ReleaseFreedIds(dataBlock);
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

	// so the id is not lost - but it is not handed out again until the write
	// that freed it is over, exactly like any other id that write freed. Until
	// then an allocation goes past it
	uint64_t idx;
	DataBlock_AllocateItem(db, &idx);
	TEST_ASSERT(idx == 6);                              // first never handed out
	TEST_ASSERT(DataBlock_DeletedItemsCount(db) == 1);  // 4 is still on the list

	// and once the write is over it is reused rather than lost
	DataBlock_ReleaseFreedIds(db);
	DataBlock_AllocateItem(db, &idx);
	TEST_ASSERT(idx == 4);

	DataBlock_Free(db);
}

// A WRITE NEVER REUSES AN ID IT FREED ITSELF.
//
// Freeing an id and allocating in the same write used to hand that id straight
// back, which is what lands a create and a delete of one id in a single effects
// payload - and no replica can apply that pair, because the create arrives
// while the entity the delete names is still live there
void test_dataBlockHoldsBackIdsFreedSinceTheLastRelease() {
	DataBlock *db = DataBlock_New(DATABLOCK_BLOCK_CAP, 16, sizeof(int), NULL);

	uint64_t idx;
	for(int i = 0; i < 3; i++) {
		DataBlock_AllocateItem(db, &idx);
	}
	// live: 0, 1, 2

	DataBlock_DeleteItem(db, 1);

	// THE FREED ID IS STILL ON THE FREE LIST - it is held back, not dropped
	TEST_ASSERT(DataBlock_DeletedItemsCount(db) == 1);

	// yet this write gets a fresh id rather than the one it just freed, and
	// that id is one no entity holds: 2 is live, so counting live items alone
	// would have named it
	DataBlock_AllocateItem(db, &idx);
	TEST_ASSERT(idx == 3);
	TEST_ASSERT(DataBlock_GetItem(db, 2) != NULL);

	// the next write reuses it, which is the whole point of holding rather
	// than dropping
	DataBlock_ReleaseFreedIds(db);
	DataBlock_AllocateItem(db, &idx);
	TEST_ASSERT(idx == 1);
	TEST_ASSERT(DataBlock_DeletedItemsCount(db) == 0);

	DataBlock_Free(db);
}

// the free list still hands out the most recently freed id first
//
// Holding ids back moves entries around inside the list, and the order it hands
// them out in is observable - it is which id a new entity gets
void test_dataBlockReleasedIdsComeBackMostRecentlyFreedFirst() {
	DataBlock *db = DataBlock_New(DATABLOCK_BLOCK_CAP, 16, sizeof(int), NULL);

	uint64_t idx;
	for(int i = 0; i < 5; i++) {
		DataBlock_AllocateItem(db, &idx);
	}

	DataBlock_DeleteItem(db, 1);
	DataBlock_DeleteItem(db, 3);
	DataBlock_ReleaseFreedIds(db);

	DataBlock_AllocateItem(db, &idx);
	TEST_ASSERT(idx == 3);
	DataBlock_AllocateItem(db, &idx);
	TEST_ASSERT(idx == 1);

	DataBlock_Free(db);
}

// deleting and allocating, interleaved, never names a live slot twice
//
// The hole left by a reused id is closed by moving the last entry of the free
// list into it. Walking one delete and one allocation at a time is what catches
// a move that puts a held-back id within reach, or loses a reusable one
void test_dataBlockInterleavedDeleteAndAllocateKeepIdsDistinct() {
	DataBlock *db = DataBlock_New(DATABLOCK_BLOCK_CAP, 16, sizeof(int), NULL);

	const uint64_t n = 64;
	uint64_t idx;

	for(uint64_t i = 0; i < n; i++) {
		DataBlock_AllocateItem(db, &idx);
	}

	// free every third id, allocating after each one, all inside one write
	uint64_t handed[64];
	uint64_t handed_n = 0;

	for(uint64_t i = 0; i < n; i += 3) {
		DataBlock_DeleteItem(db, i);
		DataBlock_AllocateItem(db, &idx);

		// not an id this write freed
		TEST_ASSERT(idx % 3 != 0 || idx >= n);

		// and not an id already handed out
		for(uint64_t j = 0; j < handed_n; j++) {
			TEST_ASSERT(handed[j] != idx);
		}
		handed[handed_n++] = idx;
	}

	// every freed id is still on the list, and the next write gets all of them
	TEST_ASSERT(DataBlock_DeletedItemsCount(db) == handed_n);

	DataBlock_ReleaseFreedIds(db);
	for(uint64_t i = 0; i < handed_n; i++) {
		DataBlock_AllocateItem(db, &idx);
		TEST_ASSERT(idx % 3 == 0 && idx < n);
	}
	TEST_ASSERT(DataBlock_DeletedItemsCount(db) == 0);

	DataBlock_Free(db);
}

// the reserved index is the one allocation actually gives
//
// Reserving predicts an id without taking it. A prediction made from the whole
// free list rather than the reusable part of it would name an id this write
// cannot have
void test_dataBlockReservedIdxAgreesWithAllocation() {
	DataBlock *db = DataBlock_New(DATABLOCK_BLOCK_CAP, 16, sizeof(int), NULL);

	uint64_t idx;
	for(int i = 0; i < 4; i++) {
		DataBlock_AllocateItem(db, &idx);
	}

	DataBlock_DeleteItem(db, 2);   // held back: this write freed it

	const uint64_t predicted = DataBlock_GetReservedIdx(db, 0);
	DataBlock_AllocateItem(db, &idx);
	TEST_ASSERT(predicted == idx);
	TEST_ASSERT(idx != 2);

	// and after the write ends the prediction follows the reuse
	DataBlock_ReleaseFreedIds(db);
	TEST_ASSERT(DataBlock_GetReservedIdx(db, 0) == 2);

	DataBlock_Free(db);
}

// "room for k items" counts the ids this write cannot reuse
//
// DataBlock_Accommodate grows the block, and the graph sizes its matrices from
// the capacity that leaves, then commits with the matrix sync policy set to
// NOP so they never grow again. Counting an index this write freed as a free
// slot leaves the block - and every matrix with it - too small, and the nodes
// created past the matrix dimension are labelled into nothing: gone from a
// label scan while the statistics still count them
void test_dataBlockAccommodateCountsHeldBackIds() {
	DataBlock *db = DataBlock_New(4, 4, sizeof(int), NULL);

	uint64_t idx;
	for(int i = 0; i < 4; i++) {
		DataBlock_AllocateItem(db, &idx);
	}
	TEST_ASSERT(db->itemCap == 4);

	// this write frees all four, so they are on the list and out of its reach
	for(uint64_t i = 0; i < 4; i++) {
		DataBlock_DeleteItem(db, i);
	}
	TEST_ASSERT(DataBlock_DeletedItemsCount(db) == 4);

	// room for four more. The live count is 0 against a capacity of 4, so
	// counting free slots finds room already there - and is wrong, because
	// this write cannot have any of those four slots
	DataBlock_Accommodate(db, 4);

	const uint64_t cap = db->itemCap;
	TEST_ASSERT(cap > DataBlock_GetReservedIdx(db, 3));

	// and that capacity covers all four allocations, with no further growth -
	// which is the part the caller relies on, having already sized its
	// matrices from it
	for(int i = 0; i < 4; i++) {
		DataBlock_AllocateItem(db, &idx);
		TEST_ASSERT(idx < cap);
	}
	TEST_ASSERT(db->itemCap == cap);

	DataBlock_Free(db);
}

// the room counted with some ids reusable and some held back
//
// The boundary between DataBlock_GetReservedIdx's two branches is where an
// under-allocation of exactly one block hides: the run of allocations crosses
// out of the reusable front into fresh ids, and only the fresh ones need room
void test_dataBlockAccommodateCountsAMixOfReusableAndFresh() {
	DataBlock *db = DataBlock_New(4, 4, sizeof(int), NULL);

	uint64_t idx;
	for(int i = 0; i < 4; i++) {
		DataBlock_AllocateItem(db, &idx);
	}

	// two ids freed by a write that has ended, so they are reusable
	DataBlock_DeleteItem(db, 0);
	DataBlock_DeleteItem(db, 2);
	DataBlock_ReleaseFreedIds(db);

	// and two freed by the write running now, which are not
	DataBlock_DeleteItem(db, 1);
	DataBlock_DeleteItem(db, 3);

	TEST_ASSERT(DataBlock_DeletedItemsCount(db) == 4);

	// four allocations: the two reusable ones, then two fresh ids that start
	// above the whole free list
	DataBlock_Accommodate(db, 4);

	const uint64_t cap = db->itemCap;

	uint64_t handed[4];
	for(int i = 0; i < 4; i++) {
		DataBlock_AllocateItem(db, handed + i);
		TEST_ASSERT(handed[i] < cap);
	}

	// no further growth was needed - which is what the caller relies on,
	// having already sized its matrices from this capacity
	TEST_ASSERT(db->itemCap == cap);

	TEST_ASSERT(handed[2] == 4);
	TEST_ASSERT(handed[3] == 5);

	DataBlock_Free(db);
}

// a decoder's deleted ids are reusable the moment it records them
//
// DataBlock_MarkAsDeletedOutOfOrder rebuilds a free list an earlier run already
// froze. Holding those ids back would mean a loaded graph could not reuse any
// of them until something else wrote to it first
void test_dataBlockDecodedDeletedIdsAreReusableAtOnce() {
	DataBlock *db = DataBlock_New(DATABLOCK_BLOCK_CAP, 16, sizeof(int), NULL);

	DataBlock_AllocateItemOutOfOrder(db, 0);
	DataBlock_AllocateItemOutOfOrder(db, 2);
	DataBlock_MarkAsDeletedOutOfOrder(db, 1);

	uint64_t idx;
	DataBlock_AllocateItem(db, &idx);
	TEST_ASSERT(idx == 1);

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
	{"dataBlockHoldsBackIdsFreedSinceTheLastRelease", test_dataBlockHoldsBackIdsFreedSinceTheLastRelease},
	{"dataBlockReleasedIdsComeBackMostRecentlyFreedFirst", test_dataBlockReleasedIdsComeBackMostRecentlyFreedFirst},
	{"dataBlockInterleavedDeleteAndAllocateKeepIdsDistinct", test_dataBlockInterleavedDeleteAndAllocateKeepIdsDistinct},
	{"dataBlockReservedIdxAgreesWithAllocation", test_dataBlockReservedIdxAgreesWithAllocation},
	{"dataBlockAccommodateCountsHeldBackIds", test_dataBlockAccommodateCountsHeldBackIds},
	{"dataBlockAccommodateCountsAMixOfReusableAndFresh", test_dataBlockAccommodateCountsAMixOfReusableAndFresh},
	{"dataBlockDecodedDeletedIdsAreReusableAtOnce", test_dataBlockDecodedDeletedIdsAreReusableAtOnce},

	{NULL, NULL}
};

