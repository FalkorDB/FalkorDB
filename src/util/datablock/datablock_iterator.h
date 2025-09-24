/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stdint.h>
#include "../block.h"

/* Datablock iterator iterates over items within a datablock. */

typedef struct {
	Block *_start_block;			// first block accessed by iterator
	Block *_current_block;			// current block
	uint64_t _block_pos;			// position within a block
	uint64_t _block_cap;            // max number of items in block
	uint64_t _current_pos;			// iterator current position
	uint64_t _end_pos;				// iterator won't pass end position
} DataBlockIterator;

// creates a new datablock iterator
DataBlockIterator *DataBlockIterator_New
(
	Block *block,        // block from which iteration begins
	uint64_t block_cap,  // max number of items in block
	uint64_t end_pos	 // iteration stops here
);

#define DataBlockIterator_Position(iter) (iter)->_current_pos

// Returns the next item, unless we've reached the end
// in which case NULL is returned.
// if `id` is provided and an item is located
// `id` will be set to the returned item index
void *DataBlockIterator_Next(DataBlockIterator *iter, uint64_t *id);

// Reset iterator to original position.
void DataBlockIterator_Reset(DataBlockIterator *iter);

// Free iterator.
void DataBlockIterator_Free(DataBlockIterator *iter);

