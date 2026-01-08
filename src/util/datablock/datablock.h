/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stdlib.h>
#include <stdbool.h>
#include <pthread.h>
#include "../block.h"
#include "./datablock_iterator.h"

// checks if item is marked as deleted
#define IS_ITEM_DELETED(item) \
	((*(uintptr_t*)(item)) & MSB_MASK)

typedef void (*fpDestructor)(void *);

// the DataBlock is a container structure for holding arbitrary items
// of a uniform type in order to reduce the number of alloc/free calls
// and improve locality of reference
// a DataBlockIterator can be used to traverse a range within the block
typedef struct {
	uint64_t itemCount;       // number of items stored in datablock
	uint64_t itemCap;         // number of items datablock can hold
	uint64_t blockCap;        // number of items a single block can hold
	uint blockCount;          // number of blocks in datablock
	uint itemSize;            // size of a single item in bytes
	Block **blocks;           // array of blocks
	uint64_t *deletedIdx;     // array of free indicies
	fpDestructor destructor;  // function pointer to a clean-up function of an item
} DataBlock;

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
);

// returns number of items stored
uint64_t DataBlock_ItemCount
(
	const DataBlock *dataBlock  // datablock
);

// make sure datablock can accommodate at least k items.
void DataBlock_Accommodate
(
	DataBlock *dataBlock,  // datablock
	int64_t k              // number of items required
);

// ensure datablock capacity >= `n`
void DataBlock_Ensure
(
	DataBlock *dataBlock,  // datablock
	uint64_t n             // minumum capacity
);

// returns an iterator which scans entire datablock.
DataBlockIterator *DataBlock_Scan
(
	const DataBlock *dataBlock  // datablock
);

// returns an iterator which scans entire out of order datablock
DataBlockIterator *DataBlock_FullScan
(
	const DataBlock *dataBlock  // datablock
);

// get item at position idx
void *DataBlock_GetItem
(
	const DataBlock *dataBlock,  // datablock
	uint64_t idx                 // item's index
);

// get reserved item id after 'n' items
uint64_t DataBlock_GetReservedIdx
(
	const DataBlock *dataBlock,  // datablock
	uint64_t n                   // number of reserved items
);

// reserve an additional `n` IDs ontop of the already `k` reserved
void DataBlock_ReservedIDs
(
	uint64_t *ids,               // [output]
	const DataBlock *dataBlock,  // datablock
	uint64_t k,                  // number of already reserved ids
	uint64_t n                   // number of IDs to reserve
) ;

// allocate a new item within given dataBlock,
// if idx is not NULL, idx will contain item position
// return a pointer to the newly allocated item.
void *DataBlock_AllocateItem
(
	DataBlock *dataBlock,  // datablock
	uint64_t *idx          // [optional] item's index
);

// try to get n consecutive items, this function operates on a best effort
// bases, it's not guarantee that it will be able to provide n items
// the actual number of returned items is reported back via `actual`
void *DataBlock_AllocateItems
(
	DataBlock *dataBlock,  // datablock
	uint32_t n,            // number of requested items
	uint32_t *actual       // number of returned items
);

// removes item at position idx
void DataBlock_DeleteItem
(
	DataBlock *dataBlock,  // datablock
	uint64_t idx           // item position
);

// returns the number of deleted items
uint DataBlock_DeletedItemsCount
(
	const DataBlock *dataBlock  // datablock
);

// returns true if the given item has been deleted
bool DataBlock_ItemIsDeleted
(
	void *item
);

// returns datablock's deleted indices array
const uint64_t *DataBlock_DeletedItems
(
	const DataBlock *dataBlock  // datablock
);

// returns to amount of memory consumed by the datablock
size_t DataBlock_memoryUsage
(
	const DataBlock *dataBlock
);

// free datablock
void DataBlock_Free
(
	DataBlock *dataBlock  // datablock
) ;

