/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "datablock_iterator.h"
#include "RG.h"
#include "datablock.h"
#include "../rmalloc.h"
#include <stdio.h>
#include <stdbool.h>

// creates a new datablock iterator
DataBlockIterator *DataBlockIterator_New
(
	Block *block,        // block from which iteration begins
	uint64_t block_cap,  // max number of items in block
	uint64_t end_pos     // iteration stops here
) {
	ASSERT (block) ;

	DataBlockIterator *iter = rm_malloc (sizeof (DataBlockIterator)) ;

	iter->_end_pos       = end_pos ;
	iter->_block_pos     = 0 ;
	iter->_block_cap     = block_cap ;
	iter->_current_pos   = 0 ;
	iter->_start_block   = block ;
	iter->_current_block = block ;

	return iter ;
}

// returns the next item, unless we've reached the end
// in which case NULL is returned
// if `id` is provided and an item is located
// `id` will be set to the returned item index
void *DataBlockIterator_Next
(
	DataBlockIterator *iter,  // iterator
	uint64_t *id              // item position
) {
	ASSERT(iter != NULL);

	// set default
	void                 *item         =  NULL;
	DataBlockItemHeader  *item_header  =  NULL;

	// have we reached the end of our iterator?
	while(iter->_current_pos < iter->_end_pos && iter->_current_block != NULL) {
		// get item at current position
		Block *block = iter->_current_block;
		item_header = (DataBlockItemHeader *)block->data + (iter->_block_pos * block->itemSize);

		// advance to next position
		iter->_block_pos += 1;
		iter->_current_pos += 1;

		// advance to next block if current block consumed
		if(iter->_block_pos == iter->_block_cap) {
			iter->_block_pos = 0;
			iter->_current_block = iter->_current_block->next;
		}

		if(!IS_ITEM_DELETED(item_header)) {
			item = ITEM_DATA(item_header);
			if(id) *id = iter->_current_pos - 1;
			break;
		}
	}

	return item;
}

// reset iterator to original position
void DataBlockIterator_Reset
(
	DataBlockIterator *iter  // iterator
) {
	ASSERT (iter != NULL) ;

	iter->_block_pos     = 0 ;
	iter->_current_pos   = 0 ;
	iter->_current_block = iter->_start_block ;
}

// seek iterator to index
void DataBlockIterator_Seek
(
	DataBlockIterator *it,  // iterator
	uint64_t idx            // index to seek to
) {
	ASSERT (it != NULL) ;
	ASSERT (idx <= it->_end_pos) ;

	// reset iterator
	it->_block_pos     = 0 ;
	it->_current_block = it->_start_block ;

	//--------------------------------------------------------------------------
	// seek to idx
	//--------------------------------------------------------------------------

	// absolute position
	it->_current_pos = idx ;

	// set current block
	int skipped_blocks = idx / it->_block_cap ;
	for (int i = 0; i < skipped_blocks ; i++) {
		it->_current_block = it->_current_block->next ;
		ASSERT (it->_current_block != NULL) ;
	}

	// set offset within current block
	it->_block_pos = (idx % it->_block_cap) ;
}

// free iterator
void DataBlockIterator_Free
(
	DataBlockIterator *iter  // iterator
) {
	ASSERT (iter != NULL) ;
	rm_free (iter) ;
}

