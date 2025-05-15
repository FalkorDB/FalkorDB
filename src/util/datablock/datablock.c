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
	ASSERT(dataBlock);
	ASSERT(blockCount > 0);

	uint prevBlockCount = dataBlock->blockCount;
	dataBlock->blockCount += blockCount;
	if(!dataBlock->blocks)
		dataBlock->blocks = rm_malloc(sizeof(Block *) * dataBlock->blockCount);
	else
		dataBlock->blocks = rm_realloc(dataBlock->blocks, sizeof(Block *) * dataBlock->blockCount);

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
	return (idx >= (dataBlock->itemCount + array_len(dataBlock->deletedIdx)));
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
	dataBlock->blocks      =  NULL;
	dataBlock->itemSize    =  itemSize + ITEM_HEADER_SIZE;
	dataBlock->itemCount   =  0;
	dataBlock->blockCount  =  0;
	dataBlock->blockCap    =  blockCap;
	dataBlock->deletedIdx  =  array_new(uint64_t, 128);
	dataBlock->destructor  =  fp;

	_DataBlock_AddBlocks(dataBlock,
			ITEM_COUNT_TO_BLOCK_COUNT(itemCap, dataBlock->blockCap));

	return dataBlock;
}

uint64_t DataBlock_ItemCount(const DataBlock *dataBlock) {
	return dataBlock->itemCount;
}

DataBlockIterator *DataBlock_Scan(const DataBlock *dataBlock) {
	ASSERT(dataBlock != NULL);
	Block *startBlock = dataBlock->blocks[0];

	// Deleted items are skipped, we're about to perform
	// array_len(dataBlock->deletedIdx) skips during out scan.
	int64_t endPos = dataBlock->itemCount + array_len(dataBlock->deletedIdx);
	return DataBlockIterator_New(startBlock, dataBlock->blockCap, endPos);
}

DataBlockIterator *DataBlock_FullScan(const DataBlock *dataBlock) {
	ASSERT(dataBlock != NULL);
	Block *startBlock = dataBlock->blocks[0];

	int64_t endPos = dataBlock->blockCount * dataBlock->blockCap;
	return DataBlockIterator_New(startBlock, dataBlock->blockCap, endPos);
}

// Make sure datablock can accommodate at least k items.
void DataBlock_Accommodate(DataBlock *dataBlock, int64_t k) {
	// Compute number of free slots.
	int64_t freeSlotsCount = dataBlock->itemCap - dataBlock->itemCount;
	int64_t additionalItems = k - freeSlotsCount;

	if(additionalItems > 0) {
		int64_t additionalBlocks =
			ITEM_COUNT_TO_BLOCK_COUNT(additionalItems, dataBlock->blockCap);
		_DataBlock_AddBlocks(dataBlock, additionalBlocks);
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

void *DataBlock_GetItem(const DataBlock *dataBlock, uint64_t idx) {
	ASSERT(dataBlock != NULL);

	// return NULL if idx is out of bounds
	if(_DataBlock_IndexOutOfBounds(dataBlock, idx)) return NULL;

	DataBlockItemHeader *item_header = DataBlock_GetItemHeader(dataBlock, idx);

	// Incase item is marked as deleted, return NULL.
	if(IS_ITEM_DELETED(item_header)) return NULL;

	return ITEM_DATA(item_header);
}

uint64_t DataBlock_GetReservedIdx(const DataBlock *dataBlock, uint64_t n) {
	ASSERT(dataBlock != NULL);

	uint deleted = DataBlock_DeletedItemsCount(dataBlock);
	if(n < deleted) {
		return dataBlock->deletedIdx[deleted - n - 1];
	} 
	
	return DataBlock_ItemCount(dataBlock) + n;
}

void *DataBlock_AllocateItem(DataBlock *dataBlock, uint64_t *idx) {
	// make sure we've got room for items
	if(dataBlock->itemCount >= dataBlock->itemCap) {
		// allocate an additional block
		_DataBlock_AddBlocks(dataBlock, 1);
	}
	ASSERT(dataBlock->itemCap > dataBlock->itemCount);

	// get index into which to store item,
	// prefer reusing free indicies
	uint pos = dataBlock->itemCount;
	if(array_len(dataBlock->deletedIdx) > 0) {
		pos = array_pop(dataBlock->deletedIdx);
	}
	dataBlock->itemCount++;

	if(idx) *idx = pos;

	DataBlockItemHeader *item_header = DataBlock_GetItemHeader(dataBlock, pos);
	MARK_HEADER_AS_NOT_DELETED(item_header);

	return ITEM_DATA(item_header);
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

	array_append(dataBlock->deletedIdx, idx);
	dataBlock->itemCount--;
}

uint DataBlock_DeletedItemsCount(const DataBlock *dataBlock) {
	return array_len(dataBlock->deletedIdx);
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
	array_append(dataBlock->deletedIdx, idx);
}

size_t DataBlock_memoryUsage
(
	const DataBlock *dataBlock
) {
	ASSERT(dataBlock != NULL);

	// datablock size = deleted index array size +
	//                  (number of blocks * block size)
	return array_len(dataBlock->deletedIdx) * sizeof(uint64_t) +
		dataBlock->blockCount * (dataBlock->itemSize * dataBlock->blockCap);
}

void DataBlock_Free(DataBlock *dataBlock) {
	for(uint i = 0; i < dataBlock->blockCount; i++) Block_Free(dataBlock->blocks[i]);

	rm_free(dataBlock->blocks);
	array_free(dataBlock->deletedIdx);
	rm_free(dataBlock);
}

