/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "RG.h"
#include "rmalloc.h"
#include "circular_buffer.h"
#include "graph/graphcontext.h"

#include <stdatomic.h>

// circular buffer structure
// the buffer is of fixed size
// items are removed by order of insertion, similar to a queue
struct _CircularBuffer {
	char *read;                   // read data from here
	_Atomic uint64_t write;       // write offset into data
	size_t item_size;             // item size in bytes
	_Atomic uint64_t item_count;  // current number of items in buffer
	uint64_t item_cap;            // max number of items held by buffer
	char *end_marker;             // marks the end of the buffer
	char data[];                  // data
};

CircularBuffer CircularBuffer_New
(
	size_t item_size,  // size of item in bytes
	uint cap           // max number of items in buffer
) {
	CircularBuffer cb = rm_calloc(1, sizeof(_CircularBuffer) + item_size * cap);

	cb->read       = cb->data;                      // initial read position
	cb->write      = ATOMIC_VAR_INIT(0);            // write offset into data
	cb->item_cap   = cap;                           // buffer capacity
	cb->item_size  = item_size;                     // item size
	cb->item_count = ATOMIC_VAR_INIT(0);            // no items in buffer
	cb->end_marker = cb->data + (item_size * cap);  // end of data marker

	return cb;
}

// returns number of items in buffer
uint64_t CircularBuffer_ItemCount
(
	CircularBuffer cb  // buffer to inspect
) {
	ASSERT(cb != NULL);

	return cb->item_count;
}

// returns buffer capacity
uint64_t CircularBuffer_Cap
(
	CircularBuffer cb // buffer
) {
	ASSERT(cb != NULL);

	return cb->item_cap;
}

uint CircularBuffer_ItemSize
(
	const CircularBuffer cb  // buffer
) {
	return cb->item_size;
}

// return true if buffer is empty
inline bool CircularBuffer_Empty
(
	const CircularBuffer cb  // buffer
) {
	ASSERT(cb != NULL);

	return cb->item_count == 0;
}

// returns true if buffer is full
inline bool CircularBuffer_Full
(
	const CircularBuffer cb  // buffer
) {
	ASSERT(cb != NULL);

	return cb->item_count == cb->item_cap;
}

// sets the read pointer to the oldest item in buffer
// assuming the buffer looks like this:
//
// [., ., ., A, B, C, ., ., .]
//                    ^
//                    W
//
// CircularBuffer_ResetReader will set 'read' to A
//
// [., ., ., A, B, C, ., ., .]
//           ^        ^
//           R        W
//
void CircularBuffer_ResetReader
(
	CircularBuffer cb  // circular buffer
) {
	// compensate for circularity
	uint64_t write = cb->write;

	// compute newest item index, e.g. newest item is at index k
	uint idx = write / cb->item_size;

	// compute offset to oldest item
	// oldest item is n elements before newest item
	//
	// example:
	//
	// [C, ., ., ., ., ., ., A, B]
	//
	// idx = 1, item_count = 3
	// offset is 1 - 3 = -2
	//
	// [C, ., ., ., ., ., ., A, B]
	//     ^                 ^
	//     W                 R

	int offset = idx - cb->item_count;
	offset *= cb->item_size;

	if(offset >= 0) {
		// offset is positive, read from beginning of buffer
		cb->read = cb->data + offset;
	} else {
		// offset is negative, read from end of buffer
		cb->read = cb->end_marker + offset;
	}
}

// adds an item to buffer
// returns 1 on success, 0 otherwise
// this function is thread-safe and lock-free
int CircularBuffer_Add
(
	CircularBuffer cb,   // buffer to populate
	void *item           // item to add
) {
	ASSERT(cb   != NULL);
	ASSERT(item != NULL);

	// atomic update buffer item count
	// do not add item if buffer is full
	uint64_t item_count = atomic_fetch_add(&cb->item_count, 1);
	if(unlikely(item_count >= cb->item_cap)) {
		cb->item_count = cb->item_cap;
		return 0;
	}

	// determine current and next write position
	uint64_t offset = atomic_fetch_add(&cb->write, cb->item_size);

	// check for buffer overflow
	if(unlikely(cb->data + offset >= cb->end_marker)) {
		// write need to circle back
		// [., ., ., ., ., ., A, B, C]
		//                           ^  ^
		//                           W0 W1
		uint64_t overflow = offset + cb->item_size;

		// adjust offset
		// [., ., ., ., ., ., A, B, C]
		//  ^  ^
		//  W0 W1
		offset -= cb->item_size * cb->item_cap;

		// update write position
		// multiple threads "competing" to update write position
		// only the thread with the largest offset will succeed
		// for the above example, W1 will succeed
		//
		// [., ., ., ., ., ., A, B, C]
		//        ^
		//        W
		atomic_compare_exchange_weak(&cb->write, &overflow, offset + cb->item_size);
	}

	// copy item into buffer
	memcpy(cb->data + offset, item, cb->item_size);

	// report success
	return 1;
}

// reserve a slot within buffer
// returns a pointer to a 'item size' slot within the buffer
// this function is thread-safe and lock-free
void *CircularBuffer_Reserve
(
	CircularBuffer cb  // buffer to populate
) {
	ASSERT(cb != NULL);

	// atomic update buffer item count
	// an item will be overwritten if buffer is full
	uint64_t item_count = atomic_fetch_add(&cb->item_count, 1);
	if(unlikely(item_count >= cb->item_cap)) {
		cb->item_count = cb->item_cap;
	}

	// determine current and next write position
	uint64_t offset = atomic_fetch_add(&cb->write, cb->item_size);

	// check for buffer overflow
	if(unlikely(cb->data + offset >= cb->end_marker)) {
		// write need to circle back
		// [., ., ., ., ., ., A, B, C]
		//                           ^  ^
		//                           W0 W1
		uint64_t overflow = offset + cb->item_size;

		// adjust offset
		// [., ., ., ., ., ., A, B, C]
		//  ^  ^
		//  W0 W1
		offset -= cb->item_size * cb->item_cap;

		// update write position
		// multiple threads "competing" to update write position
		// only the thread with the largest offset will succeed
		// for the above example, W1 will succeed
		//
		// [., ., ., ., ., ., A, B, C]
		//        ^
		//        W
		atomic_compare_exchange_weak(&cb->write, &overflow, offset + cb->item_size);
	}

	// return slot pointer
	return cb->data + offset;
}

// read oldest item from buffer
// note: this function is not thread-safe
void *CircularBuffer_Read
(
	CircularBuffer cb,  // buffer to read item from
	void *item          // [optional] pointer populated with removed item
) {
	ASSERT(cb != NULL);

	// make sure there's data to return
	if(unlikely(CircularBuffer_Empty(cb))) {
		return NULL;
	}

	void *read = cb->read;

	// update buffer item count
	cb->item_count--;

	// copy item from buffer to output
	if(item != NULL) {
		memcpy(item, cb->read, cb->item_size);
	}

	// advance read position
	// circle back if read reached the end of the buffer
	cb->read += cb->item_size;
	if(unlikely(cb->read >= cb->end_marker)) {
		cb->read = cb->data;
	}

	// return original read position
	return read;
}

// free buffer (does not free its elements if its free callback is NULL)
void CircularBuffer_Free
(
	CircularBuffer cb  // buffer to free
) {
	ASSERT(cb != NULL);

	rm_free(cb);
}

