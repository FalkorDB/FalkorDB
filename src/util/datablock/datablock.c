/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "datablock.h"
#include "datablock_iterator.h"
#include "../arr.h"
#include "../rmalloc.h"
#include <math.h>
#include <stdbool.h>

// computes the number of blocks required to accommodate n items.
#define ITEM_COUNT_TO_BLOCK_COUNT(n, cap) \
    ceil((double)n / cap)

// computes block index from item index.
#define ITEM_INDEX_TO_BLOCK_INDEX(idx, cap) \
    (idx / cap)

// computes item position within a block.
#define ITEM_POSITION_WITHIN_BLOCK(idx, cap) \
    (idx % cap)

// retrieves block in which item with index resides.
#define GET_ITEM_BLOCK(dataBlock, idx) \
    dataBlock->blocks[ITEM_INDEX_TO_BLOCK_INDEX(idx, dataBlock->blockCap)]

static void _DataBlock_AddBlocks
(
	DataBlock *dataBlock,
	uint blockCount
) {
	ASSERT (dataBlock != NULL) ;
	ASSERT (blockCount > 0) ;

	uint prevBlockCount = dataBlock->blockCount ;
	dataBlock->blockCount += blockCount ;
	if (!dataBlock->blocks) {
		dataBlock->blocks =
			rm_malloc (sizeof (Block *) * dataBlock->blockCount) ;
	} else {
		dataBlock->blocks =
			rm_realloc (dataBlock->blocks, sizeof (Block *) * dataBlock->blockCount) ;
	}

	uint i;
	for(i = prevBlockCount; i < dataBlock->blockCount; i++) {
		dataBlock->blocks[i] = Block_New(dataBlock->itemSize, dataBlock->blockCap);
		if(i > 0) dataBlock->blocks[i - 1]->next = dataBlock->blocks[i];
	}
	dataBlock->blocks[i - 1]->next = NULL;

	dataBlock->itemCap = dataBlock->blockCount * dataBlock->blockCap;
}

// Checks to see if idx is within global array bounds
// array bounds are between 0 and itemCount + #deleted indices
// e.g. [3, 7, 2, D, 1, D, 5] where itemCount = 5 and #deleted indices is 2
// and so it is valid to query the array with idx 6.
static inline bool _DataBlock_IndexOutOfBounds
(
	const DataBlock *dataBlock,
	uint64_t idx
) {
	return (idx >= (dataBlock->itemCount + arr_len(dataBlock->deletedIdx)));
}

DataBlockItemHeader *DataBlock_GetItemHeader
(
	const DataBlock *dataBlock,
	uint64_t idx
) {
	Block *block = GET_ITEM_BLOCK(dataBlock, idx);
	idx = ITEM_POSITION_WITHIN_BLOCK(idx, dataBlock->blockCap);
	return (DataBlockItemHeader *)block->data + (idx * block->itemSize);
}

//------------------------------------------------------------------------------
// DataBlock API implementation
//------------------------------------------------------------------------------

DataBlock *DataBlock_New
(
	uint64_t blockCap,
	uint64_t itemCap,
	uint itemSize,
	fpDestructor fp
) {
	DataBlock *dataBlock = rm_malloc(sizeof(DataBlock));
	dataBlock->blocks     = NULL;
	dataBlock->itemSize   = itemSize + ITEM_HEADER_SIZE;
	dataBlock->itemCount  = 0;
	dataBlock->blockCount = 0;
	dataBlock->blockCap   = blockCap;
	dataBlock->deletedIdx = arr_new(uint64_t, 128);
	dataBlock->destructor = fp;

	_DataBlock_AddBlocks(dataBlock,
			ITEM_COUNT_TO_BLOCK_COUNT(itemCap, dataBlock->blockCap));

	return dataBlock;
}

uint64_t DataBlock_ItemCount(const DataBlock *dataBlock) {
	return dataBlock->itemCount;
}

// returns datablock item size
uint DataBlock_itemSize
(
	const DataBlock *dataBlock  // datablock
) {
	ASSERT(dataBlock != NULL);

	return dataBlock->itemSize;
}

DataBlockIterator *DataBlock_Scan(const DataBlock *dataBlock) {
	ASSERT(dataBlock != NULL);
	Block *startBlock = dataBlock->blocks[0];

	// Deleted items are skipped, we're about to perform
	// array_len(dataBlock->deletedIdx) skips during out scan.
	int64_t endPos = dataBlock->itemCount + arr_len(dataBlock->deletedIdx);
	return DataBlockIterator_New(startBlock, dataBlock->blockCap, endPos);
}

DataBlockIterator *DataBlock_FullScan(const DataBlock *dataBlock) {
	ASSERT(dataBlock != NULL);
	Block *startBlock = dataBlock->blocks[0];

	int64_t endPos = dataBlock->blockCount * dataBlock->blockCap;
	return DataBlockIterator_New(startBlock, dataBlock->blockCap, endPos);
}

// Make sure datablock can accommodate at least k items.
void DataBlock_Accommodate
(
	DataBlock *dataBlock,
	int64_t k
) {
	// compute number of free slots
	int64_t freeSlotsCount  = dataBlock->itemCap - dataBlock->itemCount ;
	int64_t additionalItems = k - freeSlotsCount ;

	if (additionalItems > 0) {
		int64_t additionalBlocks =
			ITEM_COUNT_TO_BLOCK_COUNT (additionalItems, dataBlock->blockCap) ;
		_DataBlock_AddBlocks (dataBlock, additionalBlocks) ;
	}
}

void DataBlock_Ensure(DataBlock *dataBlock, uint64_t idx) {
	ASSERT(dataBlock != NULL);

	// datablock[idx] exists
	if(dataBlock->itemCap > idx) return;

	// make sure datablock cap > 'idx'
	int64_t additionalItems = (1 + idx) - dataBlock->itemCap;
	int64_t additionalBlocks =
		ITEM_COUNT_TO_BLOCK_COUNT(additionalItems, dataBlock->blockCap);
	_DataBlock_AddBlocks(dataBlock, additionalBlocks);

	ASSERT(dataBlock->itemCap > idx);
}

void *DataBlock_GetItem
(
	const DataBlock *dataBlock,
	uint64_t idx
) {
	ASSERT (dataBlock != NULL) ;

	// return NULL if idx is out of bounds
	if (_DataBlock_IndexOutOfBounds (dataBlock, idx))
	{
		return NULL ;
	}

	DataBlockItemHeader *item_header = DataBlock_GetItemHeader (dataBlock, idx) ;

	// incase item is marked as deleted, return NULL.
	if (IS_ITEM_DELETED (item_header))
	{
		return NULL ;
	}

	return ITEM_DATA (item_header) ;
}

uint64_t DataBlock_GetReservedIdx
(
	const DataBlock *dataBlock,
	uint64_t n
) {
	ASSERT (dataBlock != NULL) ;

	uint deleted = DataBlock_DeletedItemsCount (dataBlock) ;
	if (n < deleted) {
		return dataBlock->deletedIdx[deleted - n - 1] ;
	} 

	return DataBlock_ItemCount (dataBlock) + n ;
}

void *DataBlock_AllocateItem
(
	DataBlock *dataBlock,
	uint64_t *idx
) {
	// make sure we've got room for items
	if (unlikely (dataBlock->itemCount >= dataBlock->itemCap)) {
		// allocate an additional block
		_DataBlock_AddBlocks (dataBlock, 1) ;
	}

	ASSERT (dataBlock->itemCap > dataBlock->itemCount) ;

	// get index into which to store item,
	// prefer reusing free indicies
	uint pos = dataBlock->itemCount ;
	if (arr_len (dataBlock->deletedIdx) > 0) {
		pos = arr_pop (dataBlock->deletedIdx) ;

		// trim array if number of free entries is greater than 20%
		if (unlikely (
			(float)arr_len (dataBlock->deletedIdx) /
			(float)arr_cap (dataBlock->deletedIdx) <= 0.8)
		) {
			dataBlock->deletedIdx = arr_trimm_cap (dataBlock->deletedIdx,
					arr_len (dataBlock->deletedIdx)) ;
		}
	}

	dataBlock->itemCount++ ;

	if (idx) {
		*idx = pos;
	}

	DataBlockItemHeader *header = DataBlock_GetItemHeader (dataBlock, pos) ;
	MARK_HEADER_AS_NOT_DELETED (header) ;

	return ITEM_DATA (header) ;
}

void DataBlock_DeleteItem(DataBlock *dataBlock, uint64_t idx) {
	ASSERT(dataBlock != NULL);
	ASSERT(!_DataBlock_IndexOutOfBounds(dataBlock, idx));

	// Return if item already deleted.
	DataBlockItemHeader *item_header = DataBlock_GetItemHeader(dataBlock, idx);
	if(IS_ITEM_DELETED(item_header)) return;

	// Call item destructor.
	if(dataBlock->destructor) {
		unsigned char *item = ITEM_DATA(item_header);
		dataBlock->destructor(item);
	}

	MARK_HEADER_AS_DELETED(item_header);

	arr_append(dataBlock->deletedIdx, idx);
	dataBlock->itemCount--;
}

uint DataBlock_DeletedItemsCount(const DataBlock *dataBlock) {
	return arr_len(dataBlock->deletedIdx);
}

inline bool DataBlock_ItemIsDeleted(void *item) {
	DataBlockItemHeader *header = GET_ITEM_HEADER(item);
	return IS_ITEM_DELETED(header);
}

// returns datablock's deleted indices array
const uint64_t *DataBlock_DeletedItems
(
	const DataBlock *dataBlock
) {
	ASSERT(dataBlock != NULL);

	return (const uint64_t *) dataBlock->deletedIdx;
}

//------------------------------------------------------------------------------
// Out of order functionality
//------------------------------------------------------------------------------

// compare two ids; COMPARES rather than subtracts, because the difference of
// two uint64 does not fit an int
static int _DataBlock_IdCmp
(
	const void *a,
	const void *b
) {
	const uint64_t x = *(const uint64_t*)a ;
	const uint64_t y = *(const uint64_t*)b ;
	return (x > y) - (x < y) ;
}

// claim a BATCH of specific indices for live allocation
//
// This exists because DataBlock_AllocateItemOutOfOrder below carries a
// precondition the effects path cannot meet.
//
// That one marks the header and bumps itemCount and never touches 'deletedIdx',
// so it REQUIRES that 'idx' is not on the free list. Its callers are the RDB
// decoders, where that holds by construction: the file stores live entities and
// deleted ids as two separate lists, the deleted list is restored through
// DataBlock_MarkAsDeletedOutOfOrder (which does append), and the live list
// through AllocateItemOutOfOrder. The two id sets are disjoint, so there is
// never anything to remove. It is correct there and stays in use.
//
// Effects apply violates that precondition head-on: the ordinary case for a
// replica is a create naming an id that IS on the free list, because that is
// what reuse means. Used there, the id would stay on the list and the next
// allocation would hand it out a second time, putting two entities on one id.
//
// Used by effects apply, where the id is not this node's to choose. C's
// allocator reuses the most recently freed id and Rust's reuses the smallest,
// so after any delete-then-create cycle the two disagree about which id a new
// entity gets, and the replica must take the id the primary states.
//
// A BATCH RATHER THAN ONE AT A TIME, and that is the whole point of the
// signature. Claiming ids one by one means finding each on the free list,
// which is a flat array, so it is a scan per id. Claiming ascending ids out of
// an ascending list finds entry j at roughly position j, and the positions sum
// rather than cancel: measured at 0.7 ms for 1,000 ids and 237.8 ms for
// 64,000, growing 3.89x per doubling against the 2x of linear, with the scan
// 93% of the record at the top end. This path had already produced two
// quadratics - the per-edge GB_wait and the UPDATE_EDGE inner scan - and a
// third was not worth shipping.
//
// So the free list is walked ONCE per call instead: sort the claimed ids, then
// keep the entries that are not among them. O(k log n + n log n) for a free
// list of k and a batch of n, against O(n*k).
//
// Three cases, and the caller only needs to distinguish the last:
//
//   the id is on the free list   claim it, and drop it from the list
//   the id is past high water    extend, and every id SKIPPED OVER becomes
//                                free rather than lost - a leaked id makes
//                                this replica's own allocation diverge the
//                                moment it allocates anything itself, which
//                                is invisible until promotion
//   the id is live               genuine divergence
//
// NOTHING IS CLAIMED when this returns false, so the caller is free to refuse
// the payload without unwinding.
//
// returns false if any id is already live, or if the batch names one twice
bool DataBlock_AllocateItemsAtIdx
(
	DataBlock *dataBlock,
	const uint64_t *ids,  // ids to claim
	uint32_t n,           // how many
	void **items          // out: 'n' item pointers, caller allocated
) {
	ASSERT (dataBlock != NULL) ;
	ASSERT (n == 0 || (ids != NULL && items != NULL)) ;

	if (n == 0) {
		return true ;
	}

	// every id below this is either live or on the free list; every id at or
	// above it has never been handed out. Taken BEFORE anything is changed.
	const uint64_t high_water =
		dataBlock->itemCount + (uint64_t)arr_len (dataBlock->deletedIdx) ;

	// make room, and reject a live id before touching anything
	for (uint32_t i = 0 ; i < n ; i++) {
		DataBlock_Ensure (dataBlock, ids[i]) ;

		if (ids[i] < high_water) {
			DataBlockItemHeader *h =
				DataBlock_GetItemHeader (dataBlock, ids[i]) ;
			if (!IS_ITEM_DELETED (h)) {
				return false ;
			}
		}
	}

	uint64_t *sorted = rm_malloc (sizeof (uint64_t) * n) ;
	memcpy (sorted, ids, sizeof (uint64_t) * n) ;
	qsort (sorted, n, sizeof (uint64_t), _DataBlock_IdCmp) ;

	// one batch naming an id twice would claim it twice and leave the second
	// entity on top of the first
	for (uint32_t i = 1 ; i < n ; i++) {
		if (sorted[i] == sorted[i - 1]) {
			rm_free (sorted) ;
			return false ;
		}
	}

	// ONE PASS over the free list, keeping what was not claimed
	//
	// The list is a SET stored in an array and popped from the back, so its
	// order carries no meaning and compacting in place preserves it. A replica
	// never allocates while it is a replica, so the order is only observable
	// after promotion, and then only as which free id is reused first.
	const uint32_t k = arr_len (dataBlock->deletedIdx) ;
	uint32_t w = 0 ;
	for (uint32_t i = 0 ; i < k ; i++) {
		const uint64_t id = dataBlock->deletedIdx[i] ;
		if (bsearch (&id, sorted, n, sizeof (uint64_t),
					_DataBlock_IdCmp) == NULL) {
			dataBlock->deletedIdx[w++] = id ;
		}
	}
	for (uint32_t i = w ; i < k ; i++) {
		arr_pop (dataBlock->deletedIdx) ;
	}

	// ids past the high-water mark, in ascending order so each gap is walked
	// once. In practice this runs zero times, because a primary's ids are
	// dense - it is here for when they are not.
	uint64_t next = high_water ;
	for (uint32_t i = 0 ; i < n ; i++) {
		if (sorted[i] < high_water) {
			continue ;
		}
		for (uint64_t j = next ; j < sorted[i] ; j++) {
			DataBlockItemHeader *skipped =
				DataBlock_GetItemHeader (dataBlock, j) ;
			MARK_HEADER_AS_DELETED (skipped) ;
			arr_append (dataBlock->deletedIdx, j) ;
		}
		next = sorted[i] + 1 ;
	}

	rm_free (sorted) ;

	for (uint32_t i = 0 ; i < n ; i++) {
		DataBlockItemHeader *h = DataBlock_GetItemHeader (dataBlock, ids[i]) ;
		MARK_HEADER_AS_NOT_DELETED (h) ;
		items[i] = ITEM_DATA (h) ;
	}
	dataBlock->itemCount += n ;

	return true ;
}

void *DataBlock_AllocateItemOutOfOrder
(
	DataBlock *dataBlock,
	uint64_t idx
) {
	// Check if idx<=data block's current capacity. If needed, allocate additional blocks.
	DataBlock_Ensure(dataBlock, idx);
	DataBlockItemHeader *item_header = DataBlock_GetItemHeader(dataBlock, idx);
	MARK_HEADER_AS_NOT_DELETED(item_header);
	dataBlock->itemCount++;
	return ITEM_DATA(item_header);
}

void DataBlock_MarkAsDeletedOutOfOrder
(
	DataBlock *dataBlock,
	uint64_t idx
) {
	// check if idx<=data block's current capacity
	// if needed, allocate additional blocks
	DataBlock_Ensure(dataBlock, idx);
	DataBlockItemHeader *item_header = DataBlock_GetItemHeader(dataBlock, idx);

	// delete
	MARK_HEADER_AS_DELETED(item_header);
	arr_append(dataBlock->deletedIdx, idx);
}

size_t DataBlock_memoryUsage
(
	const DataBlock *dataBlock
) {
	ASSERT(dataBlock != NULL);

	// datablock size = deleted index array size +
	//                  (number of blocks * block size)
	size_t data_size = sizeof(DataBlock) ;
	data_size += arr_bytesize(dataBlock->deletedIdx);
	data_size += RedisModule_MallocSize(dataBlock->blocks) ;
	data_size += dataBlock->blockCount
		* (sizeof (Block) + dataBlock->itemSize * dataBlock->blockCap);
	return data_size;
}

void DataBlock_Free(DataBlock *dataBlock) {
	for(uint i = 0; i < dataBlock->blockCount; i++) Block_Free(dataBlock->blocks[i]);

	rm_free(dataBlock->blocks);
	arr_free(dataBlock->deletedIdx);
	rm_free(dataBlock);
}

