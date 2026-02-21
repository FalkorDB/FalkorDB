/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "datablock.h"
#include "../rmalloc.h"

// creates a new datablock iterator
DataBlockIterator *DataBlockIterator_New
(
	const DataBlock *datablock,  // datablock being iterated
	Block *block,                // block from which iteration begins
	uint64_t block_cap,          // max number of items in block
	uint64_t end_pos             // iteration stops here
) {
	ASSERT (block     != NULL) ;
	ASSERT (datablock != NULL) ;

	DataBlockIterator *it = rm_malloc (sizeof (DataBlockIterator)) ;

	it->datablock     = datablock ;
	it->end_pos       = end_pos ;
	it->block_pos     = 0 ;
	it->block_cap     = block_cap ;
	it->current_pos   = 0 ;
	it->start_block   = block ;
	it->current_block = block ;

	return it ;
}

inline uint64_t DataBlockIterator_Position
(
	const DataBlockIterator *it
) {
	ASSERT (it != NULL) ;

	return it->current_pos ;
}

// checks if iterator is depleted
bool DataBlockIterator_Depleted
(
	const DataBlockIterator *it  // iterator
) {
	ASSERT (it != NULL) ;

	return (it->current_pos == it->end_pos) ;
}

// returns the next item, unless we've reached the end
// in which case NULL is returned
// if `id` is provided and an item is located
// `id` will be set to the returned item index
void *DataBlockIterator_Next
(
	DataBlockIterator *it,  // iterator
	uint64_t *id            // item position
) {
	ASSERT (it != NULL) ;

	// set default
	void *item = NULL ;

	// have we reached the end of our iterator?
	while (it->current_pos < it->end_pos && it->current_block != NULL) {
		// advance to next position
		it->block_pos   += 1 ;
		it->current_pos += 1 ;

		// advance to next block if current block consumed
		if (it->block_pos == it->block_cap) {
			it->block_pos = 0 ;
			it->current_block = Block_Next (it->current_block) ;
		}

		item = DataBlock_GetItem (it->datablock, it->current_pos - 1) ;
		if (item == NULL) {
			continue ;
		}

		if (id != NULL) {
			*id = it->current_pos - 1 ;
		}

		break ;
	}

	return item ;
}

// returns the next item, unless we've reached the end
// in which case NULL is returned
// if `id` is provided and an item is located
// `id` will be set to the returned item index
void *DataBlockIterator_NextSkipOffloaded
(
	DataBlockIterator *it,  // iterator
	uint64_t *id            // item position
) {
	ASSERT (it != NULL) ;

	// set default
	void *item = NULL ;

	// have we reached the end of our iterator?
	while (it->current_pos < it->end_pos && it->current_block != NULL) {
		// get item at current position
		Block *block = it->current_block ;

		// advance to next block if current block consumed or empty
		if (it->block_pos == it->block_cap ||
			!Block_HasActiveItems (it->current_block)) {

			it->block_pos = 0 ;

			// search for a valid block
			do {
				it->current_block = Block_Next (it->current_block) ;
			} while (it->current_block != NULL &&
					 !Block_HasActiveItems (it->current_block)) ;
		}

		// skip if either offloaded or deleted
		bool skip = Block_IsItemDeleted   (block, it->current_pos) ||
					Block_IsItemOffloaded (block, it->current_pos) ;

		// advance to next position
		it->block_pos   += 1 ;
		it->current_pos += 1 ;

		if (skip) {
			continue ;
		}

		item = DataBlock_GetItem (it->datablock, it->current_pos - 1) ;
		ASSERT (item != NULL) ;

		if (id != NULL) {
			*id = it->current_pos - 1 ;
		}

		break ;
	}

	return item ;
}

// reset iterator to original position
void DataBlockIterator_Reset
(
	DataBlockIterator *it  // iterator
) {
	ASSERT (it != NULL) ;

	it->block_pos     = 0 ;
	it->current_pos   = 0 ;
	it->current_block = it->start_block ;
}

// seek iterator to index
void DataBlockIterator_Seek
(
	DataBlockIterator *it,  // iterator
	uint64_t idx            // index to seek to
) {
	ASSERT (it != NULL) ;
	ASSERT (idx <= it->end_pos) ;

	// reset iterator
	it->block_pos     = 0 ;
	it->current_block = it->start_block ;

	//--------------------------------------------------------------------------
	// seek to idx
	//--------------------------------------------------------------------------

	// absolute position
	it->current_pos = idx ;

	// set current block
	int skipped_blocks = idx / it->block_cap ;
	it->current_block = it->datablock->blocks [skipped_blocks] ;

	// set offset within current block
	it->block_pos = (idx % it->block_cap) ;
}

// free iterator
void DataBlockIterator_Free
(
	DataBlockIterator **it  // iterator
) {
	ASSERT (it != NULL && *it != NULL) ;
	rm_free (*it) ;
	*it = NULL ;
}

