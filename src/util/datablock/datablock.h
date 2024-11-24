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

typedef void (*fpDestructor)(void *);

// returns the item header size
#define ITEM_HEADER_SIZE 1

// DataBlock item is stored as ||header|data||
// this macro retrive the data pointer out of the header pointer
#define ITEM_DATA(header) ((void *)((header) + ITEM_HEADER_SIZE))

// DataBlock item is stored as ||header|data||
// this macro retrive the header pointer out of the data pointer
#define GET_ITEM_HEADER(item) ((item) - ITEM_HEADER_SIZE)

// sets the deleted bit in the header to 1
#define MARK_HEADER_AS_DELETED(header) ((header)->deleted |= 1)

// sets the deleted bit in the header to 0
#define MARK_HEADER_AS_NOT_DELETED(header) ((header)->deleted &= 0)

// checks if the deleted bit in the header is 1 or not
#define IS_ITEM_DELETED(header) ((header)->deleted & 1)

// the DataBlock is a container structure for holding arbitrary items of a uniform type
// in order to reduce the number of alloc/free calls and improve locality of reference.
// item deletions are thread-safe, and a DataBlockIterator can be used to traverse a
// range within the block. */
typedef struct {
	uint64_t itemCount;         // number of items stored in datablock
	uint64_t itemCap;           // number of items datablock can hold
	uint64_t blockCap;          // number of items a single block can hold
	uint blockCount;            // number of blocks in datablock
	uint itemSize;              // size of a single item in bytes
	Block **blocks;             // array of blocks
	uint64_t *deletedIdx;       // array of free indicies
	fpDestructor destructor;    // function pointer to a clean-up function of an item
} DataBlock;

// this struct is for data block item header data
// TODO: Consider using pragma pack/pop for tight memory/word alignment
typedef struct {
	unsigned char deleted: 1;  // a bit indicate if the current item is deleted or not
} DataBlockItemHeader;

// create a new DataBlock
// itemCap - number of items datablock can hold before resizing
// itemSize - item size in bytes
// fp - destructor routine for freeing items
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
	const DataBlock *dataBlock
);

// returns number of items datablock can hold
uint64_t DataBlock_ItemCap
(
	const DataBlock *dataBlock
);

// returns the deleted item ids array
uint64_t *DataBlock_DeletedItems
(
	DataBlock *dataBlock
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

// returns an iterator which scans entire datablock
DataBlockIterator *DataBlock_Scan
(
	const DataBlock *dataBlock
);

// returns an iterator which scans entire out of order datablock
DataBlockIterator *DataBlock_FullScan
(
	const DataBlock *dataBlock
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
	const DataBlock *dataBlock,
	uint64_t n
);

// allocate a new item within given dataBlock
// if idx is not NULL idx will contain item position
// return a pointer to the newly allocated item
void *DataBlock_AllocateItem
(
	DataBlock *dataBlock,
	uint64_t *idx
);

// removes item at position idx
void DataBlock_DeleteItem
(
	DataBlock *dataBlock,
	uint64_t idx
);

// returns the number of deleted items
uint DataBlock_DeletedItemsCount
(
	const DataBlock *dataBlock
);

// returns true if the given item has been deleted
bool DataBlock_ItemIsDeleted
(
	void *item
);

// free block
void DataBlock_Free
(
	DataBlock *block
);

