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

DataBlockIterator *DataBlockIterator_New
(
	Block *block,
	uint64_t block_cap,
	uint64_t end_pos
) {
	ASSERT(block);

	DataBlockIterator *iter = rm_malloc(sizeof(DataBlockIterator));

	iter->_start_block    =  block;
	iter->_current_block  =  block;
	iter->_block_pos      =  0;
	iter->_block_cap      =  block_cap;
	iter->_current_pos    =  0;
	iter->_end_pos        =  end_pos;
	return iter;
}

void *DataBlockIterator_Next
(
	DataBlockIterator *iter,
	uint64_t *id
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

void DataBlockIterator_Reset
(
	DataBlockIterator *iter
) {
	ASSERT(iter != NULL);
	iter->_block_pos      =  0;
	iter->_current_pos    =  0;
	iter->_current_block  =  iter->_start_block;
}

void DataBlockIterator_Free
(
	DataBlockIterator *iter
) {
	ASSERT(iter != NULL);
	rm_free(iter);
}

