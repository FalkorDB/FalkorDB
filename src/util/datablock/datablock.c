/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../arr.h"
#include "datablock.h"
#include "../rmalloc.h"
#include "../../storage/storage.h"

#include <math.h>
#include <stdbool.h>

// computes the number of blocks required to accommodate n items.
#define ITEM_COUNT_TO_BLOCK_COUNT(n, cap) \
    ceil((double)n / cap)

// computes block index from item index.
#define ITEM_INDEX_TO_BLOCK_INDEX(idx, cap) \
    (idx / cap)

// translate item global index to block local index
#define GLOBAL_TO_LOCAL_IDX(idx) \
    (idx % dataBlock->blockCap)

// retrieves block in which item with index resides.
#define GET_ITEM_BLOCK(idx) \
    dataBlock->blocks[ITEM_INDEX_TO_BLOCK_INDEX(idx, dataBlock->blockCap)]

// checks to see if idx is within global array bounds
// array bounds are between 0 and itemCount + #deleted indices
// e.g. [3, 7, 2, D, 1, D, 5] where itemCount = 5 and #deleted indices is 2
// and so it is valid to query the array with idx 6.
static inline bool _DataBlock_IndexOutOfBounds
(
	const DataBlock *dataBlock,
	uint64_t idx
) {
	return (idx >= (dataBlock->itemCount + array_len (dataBlock->deletedIdx))) ;
}

// add additional blocks to the datablock
static void _DataBlock_AddBlocks
(
	DataBlock *dataBlock,  // datablock
	uint blockCount        // number of blocks to add
) {
	ASSERT (dataBlock) ;
	ASSERT (blockCount > 0) ;

	uint prevBlockCount = dataBlock->blockCount ;
	dataBlock->blockCount += blockCount ;

	//--------------------------------------------------------------------------
	// allocate blocks array
	//--------------------------------------------------------------------------

	size_t n = sizeof (Block *) * dataBlock->blockCount ;

	if (dataBlock->blocks == NULL) {
		dataBlock->blocks = rm_malloc (n) ;
	} else {
		dataBlock->blocks = rm_realloc (dataBlock->blocks, n) ;
	}

	//--------------------------------------------------------------------------
	// allocate blocks
	//--------------------------------------------------------------------------

	uint i = prevBlockCount ;
	for (; i < dataBlock->blockCount; i++) {
		dataBlock->blocks[i] =
			Block_New (dataBlock->itemSize, dataBlock->blockCap) ;

		// link blocks
		if (i > 0) {
			Block_Link (dataBlock->blocks[i-1], dataBlock->blocks[i]) ;
		}
	}

	dataBlock->itemCap = dataBlock->blockCount * dataBlock->blockCap ;
}

// returns a pointer to an item within the DataBlock
static inline void *DataBlock_GetItemPrt
(
	const DataBlock *dataBlock,  // the DataBlock containing the item
	uint64_t idx                 // the global index of the item
) {
	return (void *) Block_GetItem (GET_ITEM_BLOCK (idx),
			GLOBAL_TO_LOCAL_IDX (idx)) ;
}

//------------------------------------------------------------------------------
// DataBlock API implementation
//------------------------------------------------------------------------------

// create a new DataBlock
DataBlock *DataBlock_New
(
	uint64_t blockCap,  // block capacity
	uint64_t itemCap,   // total number of items
	uint itemSize,      // item byte size
	fpDestructor fp     // [optional] item destructor
) {
	DataBlock *dataBlock  = rm_calloc (1, sizeof (DataBlock)) ;

	dataBlock->blocks     = NULL ;
	dataBlock->itemSize   = itemSize ;
	dataBlock->itemCount  = 0 ;
	dataBlock->blockCount = 0 ;
	dataBlock->blockCap   = blockCap ;
	dataBlock->deletedIdx = array_new (uint64_t, 128) ;
	dataBlock->destructor = fp ;

	_DataBlock_AddBlocks (dataBlock,
			ITEM_COUNT_TO_BLOCK_COUNT (itemCap, dataBlock->blockCap)) ;

	return dataBlock;
}

// set datablock disk storage
void DataBlock_SetStorage
(
	DataBlock *dataBlock,        // datablock
	tidesdb_column_family_t *cf  // tidesdb storage
) {
	ASSERT (dataBlock     != NULL) ;
	ASSERT (dataBlock->cf == NULL) ;

	if (cf != NULL) {
		// strict requirement for atomic pointer safety
		ASSERT (dataBlock->itemSize % 8  == 0) ;
		dataBlock->cf = cf ;
	}
}

// checks if datablock has a tidesdb column
bool DataBlock_HasStorage
(
	const DataBlock *dataBlock  // datablock
) {
	ASSERT (dataBlock != NULL) ;

	return (dataBlock->cf != NULL) ;
}

// returns number of items stored in datablock
uint64_t DataBlock_ItemCount
(
	const DataBlock *dataBlock  // datablock
) {
	return dataBlock->itemCount ;
}

// returns datablock item size
uint DataBlock_itemSize
(
	const DataBlock *dataBlock  // datablock
) {
	ASSERT (dataBlock != NULL) ;

	return dataBlock->itemSize ;
}

// returns an iterator which scans entire datablock.
DataBlockIterator *DataBlock_Scan
(
	const DataBlock *dataBlock  // datablock to scan
) {
	ASSERT (dataBlock != NULL) ;
	Block *startBlock = dataBlock->blocks[0] ;

	// deleted items are skipped, we're about to perform
	// array_len(dataBlock->deletedIdx) skips during out scan
	int64_t endPos = dataBlock->itemCount + array_len (dataBlock->deletedIdx) ;
	return DataBlockIterator_New (dataBlock, startBlock, dataBlock->blockCap,
			endPos) ;
}

// returns an iterator which scans entire out of order datablock
DataBlockIterator *DataBlock_FullScan
(
	const DataBlock *dataBlock  // datablock to scan
) {
	ASSERT (dataBlock != NULL);
	Block *startBlock = dataBlock->blocks[0] ;

	int64_t endPos = dataBlock->blockCount * dataBlock->blockCap ;
	return DataBlockIterator_New (dataBlock, startBlock, dataBlock->blockCap,
			endPos) ;
}

// make sure datablock can accommodate at least k items
void DataBlock_Accommodate
(
	DataBlock *dataBlock,
	int64_t k
) {
	// compute number of free slots
	int64_t freeSlotsCount = dataBlock->itemCap - dataBlock->itemCount ;
	int64_t additionalItems = k - freeSlotsCount ;

	if (additionalItems > 0) {
		int64_t additionalBlocks =
			ITEM_COUNT_TO_BLOCK_COUNT (additionalItems, dataBlock->blockCap) ;
		_DataBlock_AddBlocks (dataBlock, additionalBlocks) ;
	}
}

// ensure datablock capacity >= 'idx'
void DataBlock_Ensure
(
	DataBlock *dataBlock,
	uint64_t idx
) {
	ASSERT (dataBlock != NULL) ;

	// datablock[idx] exists
	if (dataBlock->itemCap > idx) {
		return ;
	}

	// make sure datablock cap > 'idx'
	int64_t additionalItems = (1 + idx) - dataBlock->itemCap ;
	int64_t additionalBlocks =
		ITEM_COUNT_TO_BLOCK_COUNT (additionalItems, dataBlock->blockCap) ;
	_DataBlock_AddBlocks (dataBlock, additionalBlocks) ;

	ASSERT (dataBlock->itemCap > idx) ;
}

// loads item from storage back into the datablock
// this function is thread safe
// multiple threads can load the same item concurrently and only one will set
// the item
static void *_DataBlock_LoadItem
(
	const DataBlock *dataBlock,   // datablock
	uint64_t idx,                 // the local index of the item
	bool delete                   // delete item from storage
) {
	ASSERT (dataBlock     != NULL) ;
	ASSERT (dataBlock->cf != NULL) ;

	tidesdb_column_family_t *cf = dataBlock->cf ;

	//--------------------------------------------------------------------------
	// load item from storage
	//--------------------------------------------------------------------------

	void *item = NULL ;
	int loaded = Storage_load (cf, &item, NULL, &idx, 1) ;

	// failed to load item from storage, return NULL
	if (loaded != 0) {
		return NULL ;
	}

	// we expect the slot to currently be NULL (not yet loaded)
	void **slot = DataBlock_GetItemPrt (dataBlock, idx) ;
	void *expected = NULL;

	// if *slot == expected (NULL), sets *slot = item and returns true
	// if *slot != expected, updates 'expected' with the current value of *slot 
	// and returns false

	bool success = __atomic_compare_exchange_n (
		slot,              // target: the 8-byte pointer address
        &expected,         // what we think is there (NULL)
        item,              // what we want to put there
        false,             // 'false' for strong CAS (better for this logic)
        __ATOMIC_SEQ_CST,  // success memory order: sequentially consistent
        __ATOMIC_RELAXED   // failure memory order
    );

	if (!success) {
		// race lost: another thread loaded this item while we were 
		// fetching it from TidesDB
		ASSERT (expected != NULL) ;

		rm_free (item) ;   // free redundant item
		item = expected ;  // use the pointer the other thread successfully set
	} else {
		// race won: we are responsible for updating the metadata
		Block_MarkItemActive (GET_ITEM_BLOCK (idx), GLOBAL_TO_LOCAL_IDX (idx)) ;

		if (delete) {
			// delete item from storage
			Storage_deleteAttributes (cf, &idx, 1) ;
		}
	}

	return (void*)slot ;
}

// retrieves a pointer to the data of a specific item within the DataBlock
// return A pointer to the item's data, or NULL if:
// 1. the index is out of bounds
// 2. the item is marked as deleted
void *DataBlock_GetItem
(
	const DataBlock *dataBlock,  // datablock
	uint64_t idx                 // the local index of the item
) {
	ASSERT (dataBlock != NULL) ;

	// return NULL if idx is out of bounds
	if (_DataBlock_IndexOutOfBounds (dataBlock, idx)) {
		return NULL ;
	}

	Block *block = GET_ITEM_BLOCK (idx) ;
	uint32_t local_idx = GLOBAL_TO_LOCAL_IDX (idx) ;

	// incase item is marked as deleted, return NULL
	if (unlikely (Block_IsItemDeleted (block, local_idx))) {
		return NULL ;
	}

	//  load item if offloaded
	if (Block_IsItemOffloaded (block, local_idx)) {
		return _DataBlock_LoadItem (dataBlock, idx, false) ;
	}

	return DataBlock_GetItemPrt (dataBlock, idx) ;
}

// get reserved item id after 'n' items
uint64_t DataBlock_GetReservedIdx
(
	const DataBlock *dataBlock,  // datablock
	uint64_t n                   // number of items already reserved
) {
	ASSERT (dataBlock != NULL) ;

	uint deleted = DataBlock_DeletedItemsCount (dataBlock) ;
	if (n < deleted) {
		return dataBlock->deletedIdx[deleted - n - 1] ;
	} 

	return DataBlock_ItemCount (dataBlock) + n ;
}

// allocate a new item within given dataBlock
// if idx is not NULL, idx will contain item position
// return a pointer to the newly allocated item
void *DataBlock_AllocateItem
(
	DataBlock *dataBlock,  // datablock
	uint64_t *idx          // [optional] item position
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
	if (array_len (dataBlock->deletedIdx) > 0) {
		pos = array_pop (dataBlock->deletedIdx) ;

		// trim array if number of free entries is greater than 20%
		if (unlikely (
			(float)array_len (dataBlock->deletedIdx) /
			(float)array_cap (dataBlock->deletedIdx) <= 0.8)
		) {
			dataBlock->deletedIdx = array_trimm_cap (dataBlock->deletedIdx,
					array_len (dataBlock->deletedIdx)) ;
		}

		Block_MarkItemActive (GET_ITEM_BLOCK (pos), GLOBAL_TO_LOCAL_IDX (pos)) ;
	}

	dataBlock->itemCount++ ;

	if (idx) {
		*idx = pos;
	}

	return DataBlock_GetItemPrt (dataBlock, pos) ;
}

// removes item at position idx
void DataBlock_DeleteItem
(
	DataBlock *dataBlock,  // datablock from which to delete item
	uint64_t idx           // item index
) {
	ASSERT (dataBlock != NULL) ;
	ASSERT (!_DataBlock_IndexOutOfBounds (dataBlock, idx)) ;

	Block *block = GET_ITEM_BLOCK (idx) ;
	uint32_t i = GLOBAL_TO_LOCAL_IDX (idx) ;

	// return if item already deleted
	if (Block_IsItemDeleted (block, i)) {
		return ;
	}

	void *item = NULL ;
	if (Block_IsItemOffloaded (block,  i)) {
		item = _DataBlock_LoadItem (dataBlock, idx, true) ;
	} else {
		item = DataBlock_GetItemPrt (dataBlock, idx) ;
	}

	// call item destructor
	if (dataBlock->destructor && item != NULL) {
		dataBlock->destructor (item) ;
	}

	Block_MarkItemDeleted (block, i) ;

	// add item index to delete list
	array_append (dataBlock->deletedIdx, idx) ;
	dataBlock->itemCount-- ;
}

// returns the number of deleted items
uint DataBlock_DeletedItemsCount
(
	const DataBlock *dataBlock  // datablock to query
) {
	ASSERT (dataBlock != NULL) ;
	return array_len (dataBlock->deletedIdx) ;
}

// returns true if the ith item is deleted
bool DataBlock_ItemIsDeleted
(
	const DataBlock *dataBlock,  // datablock
	uint64_t i                   // item index
) {
	ASSERT (dataBlock != NULL) ;
	ASSERT (!_DataBlock_IndexOutOfBounds (dataBlock, i)) ;

	return Block_IsItemDeleted (GET_ITEM_BLOCK (i), GLOBAL_TO_LOCAL_IDX (i)) ;
}

// returns datablock's deleted indices array
const uint64_t *DataBlock_DeletedItems
(
	const DataBlock *dataBlock
) {
	ASSERT (dataBlock != NULL) ;

	return (const uint64_t *) dataBlock->deletedIdx ;
}

// marks specified items within a DataBlock as offloaded to external storage
void DataBlock_MarkOffloaded (
	DataBlock *dataBlock,     // datablock
	const uint64_t *indices,  // array of indices to be marked
	size_t n_indices          // number of elements in the indices array
) {
	// validations
	ASSERT (indices       != NULL) ;
	ASSERT (dataBlock     != NULL) ;
	ASSERT (dataBlock->cf != NULL) ;

	//--------------------------------------------------------------------------
	// process entries
	//--------------------------------------------------------------------------

	for (size_t i = 0 ; i < n_indices ; i++) {
		uint64_t idx = indices[i] ;

		// item index must be valid
		ASSERT (!_DataBlock_IndexOutOfBounds (dataBlock, idx)) ;

		Block_MarkItemOffload (GET_ITEM_BLOCK (idx),
				GLOBAL_TO_LOCAL_IDX (idx)) ;
	}
}

// checks the offloaded status of each queried item
void DataBlock_IsOffloaded (
	bool *res,                // [output] datablock[indices[i]] is offloaded?
	DataBlock *dataBlock,     // datablock
	const uint64_t *indices,  // array of indices to be marked
	size_t n_indices          // number of elements in the indices array
) {
	ASSERT (res       != NULL) ;
	ASSERT (indices   != NULL) ;
	ASSERT (dataBlock != NULL) ;
	ASSERT (n_indices > 0) ;

	for (size_t i = 0; i < n_indices; i++) {
		int32_t idx = indices[i] ;
		ASSERT (!_DataBlock_IndexOutOfBounds (dataBlock, idx)) ;

		Block *b = GET_ITEM_BLOCK (idx) ;
		ASSERT (b != NULL) ;

		res[i] = Block_IsItemOffloaded (b, GLOBAL_TO_LOCAL_IDX (idx)) ;
	}
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

	// reset header
	Block_ResetHeader (GET_ITEM_BLOCK (idx), GLOBAL_TO_LOCAL_IDX (idx)) ;

	dataBlock->itemCount++ ;
	return DataBlock_GetItemPrt (dataBlock, idx) ;
}

void DataBlock_MarkAsDeletedOutOfOrder
(
	DataBlock *dataBlock,
	uint64_t idx
) {
	// check if idx <= data block's current capacity
	// if needed, allocate additional blocks
	DataBlock_Ensure (dataBlock, idx) ;

	// delete
	Block_MarkItemDeleted (GET_ITEM_BLOCK (idx), GLOBAL_TO_LOCAL_IDX (idx)) ;
	array_append (dataBlock->deletedIdx, idx) ;
}

size_t DataBlock_memoryUsage
(
	const DataBlock *dataBlock
) {
	ASSERT(dataBlock != NULL);

	// datablock size = deleted index array size +
	//                  (number of blocks * block size)
	return array_len (dataBlock->deletedIdx) * sizeof(uint64_t) +
		dataBlock->blockCount * (dataBlock->itemSize * dataBlock->blockCap) ;
}

// free datablock
void DataBlock_Free
(
	DataBlock **dataBlock  // datablock to free
) {
	ASSERT (dataBlock != NULL && *dataBlock != NULL) ;
	DataBlock *_dataBlock = *dataBlock ;

	// free blocks
	for (uint i = 0; i < _dataBlock->blockCount; i++) {
		Block_Free (_dataBlock->blocks[i]) ;
	}

	if (_dataBlock->cf != NULL) {
		// TODO: free tidesdb column family
		//if (tidesdb_drop_column_family (db, "my_cf") != 0) {
		//	return -1;
		//}
	}

	rm_free (_dataBlock->blocks) ;
	array_free (_dataBlock->deletedIdx) ;
	rm_free (_dataBlock) ;

	*dataBlock = NULL ;
}

