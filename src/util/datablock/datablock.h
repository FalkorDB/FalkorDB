/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stdlib.h>
#include <stdbool.h>
#include <pthread.h>
#include "db.h"
#include "../block.h"
#include "./datablock_iterator.h"

typedef void (*fpDestructor)(void *);

// Returns the item header size.
#define ITEM_HEADER_SIZE 1

// DataBlock item is stored as ||header|data||. This macro retrive the data pointer out of the header pointer.
#define ITEM_DATA(header) ((void *)((header) + ITEM_HEADER_SIZE))

// DataBlock item is stored as ||header|data||. This macro retrive the header pointer out of the data pointer.
#define GET_ITEM_HEADER(item) ((item) - ITEM_HEADER_SIZE)

// Sets the deleted bit in the header to 1.
#define MARK_HEADER_AS_DELETED(header) ((header)->deleted |= 1)

// Sets the deleted bit in the header to 0.
#define MARK_HEADER_AS_NOT_DELETED(header) ((header)->deleted &= 0)

// Checks if the deleted bit in the header is 1 or not.
#define IS_ITEM_DELETED(header) ((header)->deleted & 1)

//------------------------------------------------------------------------------
// offloaded macros
//------------------------------------------------------------------------------

// sets the offloaded bit in the header to 1
#define MARK_HEADER_AS_OFFLOADED(header) ((header)->offloaded |= 1)

// sets the offloaded bit in the header to 0
#define MARK_HEADER_AS_NOT_OFFLOADED(header) ((header)->offloaded &= 0)

// checks if the offloaded bit in the header is 1 or not
#define IS_ITEM_OFFLOADED(header) ((header)->offloaded & 1)

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

// this struct is for data block item header data
typedef struct {
	unsigned char deleted   : 1; // 1 if the item is deleted
	unsigned char offloaded : 1; // 1 if the item is offloaded to disk
    unsigned char reserved  : 6; // explicitly padding to fill the byte
} DataBlockItemHeader;

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

// returns number of items stored
uint64_t DataBlock_ItemCount(const DataBlock *dataBlock);

// returns datablock item size
uint DataBlock_itemSize
(
	const DataBlock *dataBlock  // datablock
);

// Make sure datablock can accommodate at least k items.
void DataBlock_Accommodate(DataBlock *dataBlock, int64_t k);

// ensure datablock capacity >= 'idx'
void DataBlock_Ensure(DataBlock *dataBlock, uint64_t idx);

// Returns an iterator which scans entire datablock.
DataBlockIterator *DataBlock_Scan(const DataBlock *dataBlock);

// Returns an iterator which scans entire out of order datablock.
DataBlockIterator *DataBlock_FullScan(const DataBlock *dataBlock);

// get item at position idx
void *DataBlock_GetItem
(
	const DataBlock *dataBlock,
	uint64_t idx
);

// get reserved item id after 'n' items
uint64_t DataBlock_GetReservedIdx(const DataBlock *dataBlock, uint64_t n);

// Allocate a new item within given dataBlock,
// if idx is not NULL, idx will contain item position
// return a pointer to the newly allocated item.
void *DataBlock_AllocateItem(DataBlock *dataBlock, uint64_t *idx);

// Removes item at position idx.
void DataBlock_DeleteItem(DataBlock *dataBlock, uint64_t idx);

// Returns the number of deleted items.
uint DataBlock_DeletedItemsCount(const DataBlock *dataBlock);

// Returns true if the given item has been deleted.
bool DataBlock_ItemIsDeleted(void *item);

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

// returns to amount of memory consumed by the datablock
size_t DataBlock_memoryUsage
(
	const DataBlock *dataBlock
);

// free block
void DataBlock_Free
(
	DataBlock *block
) ;

