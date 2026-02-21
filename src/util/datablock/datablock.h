/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "db.h"
#include "../block.h"

#include <stdlib.h>
#include <stdbool.h>
#include <pthread.h>

typedef void (*fpDestructor)(void *);

// the DataBlock is a container structure for holding arbitrary items of a uniform type
// in order to reduce the number of alloc/free calls and improve locality of reference.
// Item deletions are thread-safe, and a DataBlockIterator can be used to traverse a
// range within the block
typedef struct {
	uint64_t itemCount;           // number of items stored in datablock
	uint64_t itemCap;             // number of items datablock can hold
	uint64_t blockCap;            // number of items a single block can hold
	uint blockCount;              // number of blocks in datablock
	uint itemSize;                // size of a single item in bytes
	Block **blocks;               // array of blocks
	uint64_t *deletedIdx;         // array of free indicies
	fpDestructor destructor;      // function pointer to a clean-up function of an item
	tidesdb_column_family_t *cf;  // [optional] disk storage
} DataBlock;

// datablock iterator iterates over items within a datablock
typedef struct {
	const DataBlock *datablock;   // datablock being iterated
	Block *start_block;           // first block accessed by iterator
	Block *current_block;         // current block
	uint64_t block_pos;           // position within a block
	uint64_t block_cap;           // max number of items in block
	uint64_t current_pos;         // iterator current position
	uint64_t end_pos;             // iterator won't pass end position
} DataBlockIterator;

// create a new DataBlock
DataBlock *DataBlock_New
(
	uint64_t blockCap,  // block capacity
	uint64_t itemCap,   // total number of items
	uint itemSize,      // item byte size
	fpDestructor fp     // [optional] item destructor
);

// set datablock disk storage
void DataBlock_SetStorage
(
	DataBlock *dataBlock,        // datablock
	tidesdb_column_family_t *cf  // tidesdb storage
);

// checks if datablock has a tidesdb column
bool DataBlock_HasStorage
(
	const DataBlock *dataBlock  // datablock
) ;

// returns number of items stored in datablock
uint64_t DataBlock_ItemCount
(
	const DataBlock *dataBlock  // datablock
);

// returns datablock item size
uint DataBlock_itemSize
(
	const DataBlock *dataBlock  // datablock
);

// returns an iterator which scans entire datablock.
DataBlockIterator *DataBlock_Scan
(
	const DataBlock *dataBlock  // datablock to scan
);

// returns an iterator which scans entire out of order datablock
DataBlockIterator *DataBlock_FullScan
(
	const DataBlock *dataBlock  // datablock to scan
);

// make sure datablock can accommodate at least k items
void DataBlock_Accommodate
(
	DataBlock *dataBlock,
	int64_t k
);

// ensure datablock capacity >= 'idx'
void DataBlock_Ensure
(
	DataBlock *dataBlock,
	uint64_t idx
);

// get item at position idx
void *DataBlock_GetItem
(
	const DataBlock *dataBlock,
	uint64_t idx
);

// get reserved item id after 'n' items
uint64_t DataBlock_GetReservedIdx
(
	const DataBlock *dataBlock,  // datablock
	uint64_t n                   // number of items already reserved
);

// allocate a new item within given dataBlock
// if idx is not NULL, idx will contain item position
// return a pointer to the newly allocated item
void *DataBlock_AllocateItem
(
	DataBlock *dataBlock,  // datablock
	uint64_t *idx          // [optional] item position
);

// removes item at position idx
void DataBlock_DeleteItem
(
	DataBlock *dataBlock,  // datablock from which to delete item
	uint64_t idx           // item index
);

// returns the number of deleted items
uint DataBlock_DeletedItemsCount
(
	const DataBlock *dataBlock  // datablock to query
) ;

// returns true if the ith item is deleted
bool DataBlock_ItemIsDeleted
(
	const DataBlock *dataBlock,  // datablock
	uint64_t i                   // item index
);

// returns datablock's deleted indices array
const uint64_t *DataBlock_DeletedItems
(
	const DataBlock *dataBlock
);

// marks specified items within a DataBlock as offloaded to external storage
void DataBlock_MarkOffloaded (
	DataBlock *dataBlock,     // datablock
	const uint64_t *indices,  // array of indices to be marked
	size_t n_indices          // number of elements in the indices array
);

// checks the offloaded status of each queried item
void DataBlock_IsOffloaded (
	bool *res,                // [output] datablock[indices[i]] is offloaded?
	DataBlock *dataBlock,     // datablock
	const uint64_t *indices,  // array of indices to be marked
	size_t n_indices          // number of elements in the indices array
);

// returns to amount of memory consumed by the datablock
size_t DataBlock_memoryUsage
(
	const DataBlock *dataBlock
);

// free datablock
void DataBlock_Free
(
	DataBlock **dataBlock  // datablock to free
) ;

//------------------------------------------------------------------------------
// datablock iterator
//------------------------------------------------------------------------------

// creates a new datablock iterator
DataBlockIterator *DataBlockIterator_New
(
	const DataBlock *datablock,  // datablock being iterated
	Block *block,                // block from which iteration begins
	uint64_t block_cap,          // max number of items in block
	uint64_t end_pos	         // iteration stops here
);

// get iterator's global position
uint64_t DataBlockIterator_Position
(
	const DataBlockIterator *it
);

// checks if iterator is depleted
bool DataBlockIterator_Depleted
(
	const DataBlockIterator *it  // iterator
);

// returns the next item, unless we've reached the end
// in which case NULL is returned
// if `id` is provided and an item is located
// `id` will be set to the returned item index
void *DataBlockIterator_Next
(
	DataBlockIterator *it,  // iterator
	uint64_t *id            // item position
);

// returns the next item, unless we've reached the end
// in which case NULL is returned
// if `id` is provided and an item is located
// `id` will be set to the returned item index
void *DataBlockIterator_NextSkipOffloaded
(
	DataBlockIterator *it,  // iterator
	uint64_t *id            // item position
);

// reset iterator to original position
void DataBlockIterator_Reset
(
	DataBlockIterator *it  // iterator
);

// seek iterator to index
void DataBlockIterator_Seek
(
	DataBlockIterator *it,  // iterator
	uint64_t idx            // index to seek to
);

// free iterator
void DataBlockIterator_Free
(
	DataBlockIterator **it  // iterator
);

