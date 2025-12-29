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

// computes the number of blocks required to accommodate n items
#define ITEM_COUNT_TO_BLOCK_COUNT(n) \
    ceil((double)n / dataBlock->blockCap)

// computes block index from item index
#define ITEM_INDEX_TO_BLOCK_INDEX(idx) \
    ((idx) / dataBlock->blockCap)

// retrieves block in which item with index resides
#define GET_ITEM_BLOCK(idx) \
    dataBlock->blocks[ITEM_INDEX_TO_BLOCK_INDEX(idx)]

// computes item position within a block
#define ITEM_POSITION_WITHIN_BLOCK(idx) \
    ((idx) % dataBlock->blockCap)

// sets the deleted bit in the header to 1.
#define MARK_ITEM_AS_DELETED(idx) \
	((unsigned char*)(GET_ITEM_BLOCK(idx)->data))[ITEM_POSITION_WITHIN_BLOCK(idx) * dataBlock->itemSize] |= _8BIT_MSB_MASK

// sets the deleted bit in the item's header to 0
#define MARK_ITEM_AS_NOT_DELETED(idx) \
	((unsigned char*)(GET_ITEM_BLOCK(idx)->data))[ITEM_POSITION_WITHIN_BLOCK(idx) * dataBlock->itemSize] &= _8BIT_MSB_MASK_CMP

// returns a pointer to the ith item
#define GET_ITEM(idx) \
	((unsigned char*)(GET_ITEM_BLOCK(idx)->data)) + (ITEM_POSITION_WITHIN_BLOCK(idx) * dataBlock->itemSize)

// add new blocks to datablock
static void _DataBlock_AddBlocks
(
	DataBlock *dataBlock,  // datablock
	uint blockCount        // number of new blocks to add
) {
	ASSERT (blockCount > 0) ;
	ASSERT (dataBlock  != NULL) ;

	uint prevBlockCount = dataBlock->blockCount ;
	dataBlock->blockCount += blockCount ;

	dataBlock->blocks =
		rm_realloc (dataBlock->blocks, sizeof (Block *) * dataBlock->blockCount) ;

	uint i ;
	for (i = prevBlockCount; i < dataBlock->blockCount; i++) {
		dataBlock->blocks[i] =
			Block_New (dataBlock->itemSize, dataBlock->blockCap) ;
		if (i > 0) {
			dataBlock->blocks[i - 1]->next = dataBlock->blocks[i] ;
		}
	}

	dataBlock->blocks[i - 1]->next = NULL ;
	dataBlock->itemCap = dataBlock->blockCount * dataBlock->blockCap ;
}

// checks to see if idx is within global array bounds
// array bounds are between 0 and itemCount + #deleted indices
// e.g. [3, 7, 2, D, 1, D, 5] where itemCount = 5 and #deleted indices is 2
// and so it is valid to query the array with idx 6
static inline bool _DataBlock_IndexOutOfBounds
(
	const DataBlock *dataBlock,
	uint64_t idx
) {
	return (idx >= (dataBlock->itemCount + array_len (dataBlock->deletedIdx))) ;
}

//------------------------------------------------------------------------------
// DataBlock API implementation
//------------------------------------------------------------------------------

// create a new DataBlock
// itemCap - number of items datablock can hold before resizing.
// itemSize - item size in bytes.
// fp - destructor routine for freeing items.
DataBlock *DataBlock_New
(
	uint64_t blockCap,  // block size
	uint64_t itemCap,   // initial item cap
	uint itemSize,      // size of item in bytes
	fpDestructor fp     // item destructor
) {
	DataBlock *dataBlock = rm_malloc (sizeof (DataBlock)) ;

	dataBlock->blocks     = NULL ;
	dataBlock->itemSize   = itemSize ;
	dataBlock->itemCount  = 0 ;
	dataBlock->blockCount = 0 ;
	dataBlock->blockCap   = blockCap ;
	dataBlock->deletedIdx = array_new (uint64_t, 128) ;
	dataBlock->destructor = fp ;

	_DataBlock_AddBlocks (dataBlock,
			ITEM_COUNT_TO_BLOCK_COUNT (itemCap)) ;

	return dataBlock ;
}

uint64_t DataBlock_ItemCount
(
	const DataBlock *dataBlock
) {
	return dataBlock->itemCount;
}

// make sure datablock can accommodate at least k items.
void DataBlock_Accommodate
(
	DataBlock *dataBlock,  // datablock
	int64_t k              // number of items required
) {
	// compute number of free slots
	int64_t freeSlotsCount  = dataBlock->itemCap - dataBlock->itemCount ;
	int64_t additionalItems = k - freeSlotsCount ;

	if (additionalItems > 0) {
		int64_t additionalBlocks =
			ITEM_COUNT_TO_BLOCK_COUNT (additionalItems) ;
		_DataBlock_AddBlocks (dataBlock, additionalBlocks) ;
	}
}

// ensure datablock capacity >= `n`
void DataBlock_Ensure
(
	DataBlock *dataBlock,  // datablock
	uint64_t n             // minumum capacity
) {
	ASSERT (dataBlock != NULL) ;

	// datablock[n] exists
	if (dataBlock->itemCap > n) {
		return ;
	}

	// make sure datablock cap > 'n'
	int64_t additionalItems = (1 + n) - dataBlock->itemCap ;
	int64_t additionalBlocks =
		ITEM_COUNT_TO_BLOCK_COUNT (additionalItems) ;
	_DataBlock_AddBlocks (dataBlock, additionalBlocks) ;

	ASSERT (dataBlock->itemCap > n) ;
}

// returns an iterator which scans entire datablock.
DataBlockIterator *DataBlock_Scan
(
	const DataBlock *dataBlock  // datablock
) {
	ASSERT (dataBlock != NULL) ;
	Block *startBlock = dataBlock->blocks[0] ;

	// deleted items are skipped, we're about to perform
	// array_len(dataBlock->deletedIdx) skips during out scan
	int64_t endPos = dataBlock->itemCount + array_len (dataBlock->deletedIdx) ;
	return DataBlockIterator_New (startBlock, dataBlock->blockCap, endPos) ;
}

// returns an iterator which scans entire out of order datablock
DataBlockIterator *DataBlock_FullScan
(
	const DataBlock *dataBlock  // datablock
) {
	ASSERT (dataBlock != NULL) ;
	Block *startBlock = dataBlock->blocks[0] ;

	int64_t endPos = dataBlock->blockCount * dataBlock->blockCap;
	return DataBlockIterator_New (startBlock, dataBlock->blockCap, endPos) ;
}

// get item at position idx
void *DataBlock_GetItem
(
	const DataBlock *dataBlock,  // datablock
	uint64_t idx                 // item's index
) {
	ASSERT (dataBlock != NULL) ;

	// return NULL if idx is out of bounds
	if (unlikely (_DataBlock_IndexOutOfBounds (dataBlock, idx))) {
		return NULL ;
	}

	void *item = GET_ITEM (idx) ;

	// incase item is marked as deleted, return NULL
	if (IS_ITEM_DELETED (item)) {
		return NULL ;
	}

	return item ;
}

// get reserved item id after 'n' items
uint64_t DataBlock_GetReservedIdx
(
	const DataBlock *dataBlock,  // datablock
	uint64_t n                   // number of reserved items
) {
	ASSERT (dataBlock != NULL) ;

	uint deleted = DataBlock_DeletedItemsCount (dataBlock) ;
	if (n < deleted) {
		return dataBlock->deletedIdx[deleted - n - 1] ;
	} 

	return DataBlock_ItemCount (dataBlock) + n ;
}

// allocate a new item within given dataBlock,
// if idx is not NULL, idx will contain item position
// return a pointer to the newly allocated item.
void *DataBlock_AllocateItem
(
	DataBlock *dataBlock,  // datablock
	uint64_t *idx          // [optional] item's index
) {
	// make sure there's room for item
	if (unlikely (dataBlock->itemCount >= dataBlock->itemCap)) {
		// allocate an additional block
		_DataBlock_AddBlocks (dataBlock, 1) ;
	}

	ASSERT (dataBlock->itemCap > dataBlock->itemCount) ;

	// get index into which to store item,
	// prefer reusing free indicies
	uint pos = dataBlock->itemCount ;
	if (array_len (dataBlock->deletedIdx) > 0) {
		pos = array_pop (dataBlock->deletedIdx) ;

		// trim array if number of free entries is greater than 20%
		if (unlikely (
			array_cap (dataBlock->deletedIdx) > 128 &&
			(float)array_len (dataBlock->deletedIdx) /
			(float)array_cap (dataBlock->deletedIdx) <= 0.8)
		) {
			dataBlock->deletedIdx = array_trimm_cap (dataBlock->deletedIdx,
					array_len (dataBlock->deletedIdx)) ;
		}

		MARK_ITEM_AS_NOT_DELETED (pos) ;
	}

	dataBlock->itemCount++ ;

	if (idx) {
		*idx = pos;
	}

	return GET_ITEM (pos) ;
}

// try to get n consecutive items, this function operates on a best effort
// bases, it's not guarantee that it will be able to provide n items
// the actual number of returned items is reported back via `actual`
void *DataBlock_AllocateItems
(
	DataBlock *dataBlock,  // datablock
	uint32_t n,            // number of requested items
	uint32_t *actual       // number of returned items
) {
	ASSERT (n         >  0) ;
	ASSERT (actual    != NULL) ;
	ASSERT (dataBlock != NULL) ;

	// this function can not operate on a datablock which contains
	// deleted entries
	ASSERT (DataBlock_DeletedItemsCount(dataBlock) == 0) ;

	// ensure physical space exists for at least itemCount + n items
    DataBlock_Ensure (dataBlock, dataBlock->itemCount + n);

	// locate the current block and local index
	uint64_t global_idx = dataBlock->itemCount ;
	uint32_t local_idx = ITEM_POSITION_WITHIN_BLOCK (global_idx) ;

	Block *block = GET_ITEM_BLOCK (global_idx) ;

	// calculate how much we can give from THIS block (Contiguous Constraint)
	uint32_t available = dataBlock->blockCap - local_idx ; // number available items
	ASSERT (available > 0) ;

	*actual = MIN (available, n) ;

	// advance the global counter
	dataBlock->itemCount += *actual ;

	return block->data + (local_idx * dataBlock->itemSize) ;
}

// removes item at position idx
void DataBlock_DeleteItem
(
	DataBlock *dataBlock,  // datablock
	uint64_t idx           // item position
) {
	ASSERT (dataBlock != NULL) ;
	ASSERT (!_DataBlock_IndexOutOfBounds (dataBlock, idx)) ;

	// return if item already deleted
	void *item = GET_ITEM (idx) ;
	if (unlikely (IS_ITEM_DELETED (item))) {
		return ;
	}

	// call item destructor
	if (dataBlock->destructor != NULL) {
		dataBlock->destructor (item) ;
	}

	MARK_ITEM_AS_DELETED (idx) ;

	array_append (dataBlock->deletedIdx, idx) ;
	dataBlock->itemCount-- ;
}

// returns the number of deleted items
uint DataBlock_DeletedItemsCount
(
	const DataBlock *dataBlock  // datablock
) {
	return array_len (dataBlock->deletedIdx) ;
}

// returns true if the given item has been deleted
bool DataBlock_ItemIsDeleted
(
	void *item
) {
	ASSERT (item != NULL) ;
	return IS_ITEM_DELETED (item) ;
}

// returns datablock's deleted indices array
const uint64_t *DataBlock_DeletedItems
(
	const DataBlock *dataBlock
) {
	ASSERT (dataBlock != NULL) ;

	return (const uint64_t *) dataBlock->deletedIdx ;
}

size_t DataBlock_memoryUsage
(
	const DataBlock *dataBlock  // datablock
) {
	ASSERT (dataBlock != NULL) ;

	// datablock size = deleted index array size +
	//                  (number of blocks * block size)
	return array_len (dataBlock->deletedIdx) * sizeof (uint64_t) +
		dataBlock->blockCount * (dataBlock->itemSize * dataBlock->blockCap) ;
}

//------------------------------------------------------------------------------
// Out of order functionality
//------------------------------------------------------------------------------

void *DataBlock_AllocateItemOutOfOrder
(
	DataBlock *dataBlock,
	uint64_t idx
) {
	// check if idx<=data block's current capacity
	// if needed, allocate additional blocks
	DataBlock_Ensure (dataBlock, idx) ;
	MARK_ITEM_AS_NOT_DELETED (idx) ;
	dataBlock->itemCount++ ;
	return GET_ITEM (idx) ;
}

void DataBlock_MarkAsDeletedOutOfOrder
(
	DataBlock *dataBlock,
	uint64_t idx
) {
	// check if idx<=data block's current capacity
	// if needed, allocate additional blocks
	DataBlock_Ensure(dataBlock, idx);

	// delete
	MARK_ITEM_AS_DELETED (idx) ;
	array_append (dataBlock->deletedIdx, idx) ;
}

// free datablock
void DataBlock_Free
(
	DataBlock *dataBlock  // datablock
) {
	for (uint i = 0; i < dataBlock->blockCount; i++) {
		Block_Free (dataBlock->blocks[i]) ;
	}

	rm_free (dataBlock->blocks) ;
	array_free (dataBlock->deletedIdx) ;
	rm_free (dataBlock) ;
}

