/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "block.h"
#include "rmalloc.h"

// item deleted bit
#define ITEM_DELETED_BIT (1 << 0)
#define MASK_DELETED_BIT ~(ITEM_DELETED_BIT)

// item offloaded bit
#define ITEM_OFFLOADED_BIT (1 << 1)
#define MASK_OFFLOADED_BIT ~(ITEM_OFFLOADED_BIT)

// checks if the deleted bit in the header is 1 or not
#define IS_ITEM_DELETED(header) ((*(header)) & ITEM_DELETED_BIT)

// sets the deleted bit in the header to 0
#define MARK_HEADER_AS_ACTIVE(header) ((*(header)) &= MASK_DELETED_BIT)

// sets the deleted bit in the header to 1
#define MARK_HEADER_AS_DELETED(header) ((*(header)) |= ITEM_DELETED_BIT)

// checks if the offloaded bit in the header is 1 or not
#define IS_ITEM_OFFLOADED(header) ((*(header)) & ITEM_OFFLOADED_BIT)

// sets the offloaded bit in the header to 1
#define MARK_HEADER_AS_OFFLOADED(header) ((*(header)) |= ITEM_OFFLOADED_BIT)

// sets the offloaded bit in the header to 0
#define MARK_HEADER_AS_LOADED(header) ((*(header)) &= MASK_OFFLOADED_BIT)

// include block structure
#include "block_struct.h"

// create a new block
Block *Block_New
(
	uint32_t itemSize,  // item byte size
	uint32_t capacity   // number of items
) {
	ASSERT (capacity > 0);
	ASSERT (itemSize > 0) ;

	// calculate offsets
    // headers take 1 byte each
	// we must pad this section to 8-byte alignment
	size_t headersSectionSize = ALIGN8 (capacity) ;
	size_t itemsSectionSize = (size_t)capacity * itemSize ;

	// total size: struct header + metadata + elements
	size_t totalSize = sizeof (Block) + headersSectionSize + itemsSectionSize ;

	// allocation
    // using rm_calloc to ensure all headers start as '0'
	// (not deleted, not offloaded)
	Block *block = rm_calloc (1, totalSize) ;

	// initialization
	block->cap      = capacity ;
    block->itemSize = itemSize ;

	// set the convenience pointer
	// block->data points to the start of the header section
	// block->elements points to the start of the items section
    block->elements = block->data + headersSectionSize ;

    // final pedantic check: the elements pointer must be 8-byte aligned
    ASSERT (((uintptr_t)block->elements % 8) == 0) ;

    return block;
}

// link blocks
void Block_Link
(
	Block *prev,  // prev block
	Block *next   // next block
) {
	ASSERT (prev != NULL) ;
	ASSERT (next != NULL) ;
	ASSERT (prev != next) ;
	ASSERT (prev->next == NULL) ;

	prev->next = next ;
}

// get a pointer to the ith item
inline unsigned char *Block_GetItem
(
	Block *block,  // block
	uint32_t i     // item position
) {
	ASSERT (block != NULL) ;
	ASSERT (i < block->cap) ;

	return block->elements + (i * block->itemSize) ;
}

// get next block
Block *Block_Next
(
	const Block *block  // block
) {
	ASSERT (block != NULL) ;

	return block->next ;
}

// reset the ith item header
void Block_ResetHeader
(
	Block *block,  // block
	uint32_t i     // item's position
) {
	ASSERT (block != NULL) ;
	ASSERT (i < block->cap) ;

	unsigned char *header = block->data + i ;

	if (IS_ITEM_DELETED (header)) {
		block->deleted_count-- ;
		ASSERT (block->deleted_count >= 0) ;
	}

	if (IS_ITEM_OFFLOADED (header)) {
		block->offloaded_count-- ;
		ASSERT (block->offloaded_count >= 0) ;
	}

	// clear bits
	block->data[i] = 0 ;
}

//------------------------------------------------------------------------------
// status checks
//------------------------------------------------------------------------------

// check if the ith item is marked as deleted
inline bool Block_IsItemDeleted
(
	const Block *block,  // block
	uint32_t i           // item's position
) {
	// validations
	ASSERT (block != NULL) ;
	ASSERT (i < block->cap) ;

	const unsigned char *header = block->data + i ;

	return IS_ITEM_DELETED (header) ;
}

// check if the ith item is marked as offloaded
inline bool Block_IsItemOffloaded
(
	const Block *block,  // block
	uint32_t i           // item's position
) {
	// validations
	ASSERT (block != NULL) ;
	ASSERT (i < block->cap) ;

	const unsigned char *header = block->data + i ;

	return IS_ITEM_OFFLOADED (header) ;
}

// checks if block contains non offloaded, non deleted items
// return false if offloaded + deleted = block capacity
bool Block_HasActiveItems
(
	const Block *block  // block
) {
	ASSERT (block != NULL) ;

	return (block->offloaded_count + block->deleted_count < block->cap) ;
}

//------------------------------------------------------------------------------
// state transitions
//------------------------------------------------------------------------------

// marks the ith item as deleted
void Block_MarkItemDeleted
(
	Block *block,  // block
	uint32_t i     // item position within block
) {
	// validations
	ASSERT (block != NULL) ;
	ASSERT (i < block->cap) ;

	// get the item's header
	unsigned char *header = block->data + i ;

	// item shouldn't be deleted
	ASSERT (!IS_ITEM_DELETED (header)) ;

	// mark item as offloaded and increase the offloaded count
	MARK_HEADER_AS_DELETED (header) ;
	block->deleted_count++ ;

	// validate counts
	ASSERT (block->offloaded_count + block->deleted_count <= block->cap) ;
}

// marks the ith item as active
void Block_MarkItemActive
(
	Block *block,  // block
	uint32_t i     // item position within block
) {
	// validations
	ASSERT (block != NULL) ;
	ASSERT (i < block->cap) ;

	// get the item's header
	unsigned char *header = block->data + i ;

	// item should be deleted
	ASSERT (IS_ITEM_DELETED (header)) ;

	// mark item as acrive and decrement the deleted count
	MARK_HEADER_AS_ACTIVE (header) ;
	block->deleted_count-- ;

	// validate counts
	ASSERT (block->deleted_count >= 0) ;
}

// marks the ith item as offloaded
void Block_MarkItemOffload
(
	Block *block,  // block
	uint32_t i     // item position within block
) {
	// validations
	ASSERT (block != NULL) ;
	ASSERT (i < block->cap) ;

	// get the item's header
	unsigned char *header = block->data + i ;

	// item shouldn't be deleted nor offloaded
	ASSERT (!IS_ITEM_DELETED   (header)) ;
	ASSERT (!IS_ITEM_OFFLOADED (header)) ;

	// mark item as offloaded and increase the offloaded count
	MARK_HEADER_AS_OFFLOADED (header) ;
	block->offloaded_count++ ;

	// validate counts
	ASSERT (block->offloaded_count + block->deleted_count <= block->cap) ;

	// TODO: not sure if really required
	// nullify item
	memset (block->elements + (i * block->itemSize), 0, block->itemSize) ;
}

// marks the ith item as loaded
void Block_MarkItemLoaded
(
	Block *block,  // block
	uint32_t i     // item position within block
) {
	// validations
	ASSERT (block != NULL) ;
	ASSERT (i < block->cap) ;

	// get the item's header
	unsigned char *header = block->data + i ;

	// item should be marked offloaded and non deleted
	ASSERT (!IS_ITEM_DELETED  (header)) ;
	ASSERT (IS_ITEM_OFFLOADED (header)) ;

	// mark item as offloaded and increase the offloaded count
	MARK_HEADER_AS_LOADED (header) ;
	block->offloaded_count-- ;

	// validate counts
	ASSERT (block->offloaded_count >= 0) ;
}

// free block
void Block_Free
(
	Block *block  // block to free
) {
	ASSERT (block != NULL) ;
	rm_free (block) ;
}

