/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */


#include "RG.h"
#include "rmalloc.h"
#include "circular_buffer.h"

#include <pthread.h>

// circular buffer structure
// the buffer is of fixed size
// items are removed by order of insertion, similar to a queue
struct _CircularBuffer {
	void *read;            // read data from here
	void *write;           // write data here
	size_t item_size;      // item size in bytes
	uint32_t item_cap;     // max number of items held by buffer
	uint32_t item_count;   // number of items in buffer
	pthread_mutex_t lock;  // mutex
	void *end_marker;      // marks the end of the buffer
	char data[];           // data
};

static void _CircularBuffer_Lock
(
	CircularBuffer cb
) {
	int res = pthread_mutex_lock (&cb->lock) ;
	ASSERT (res == 0) ;
}

static void _CircularBuffer_Unlock
(
	CircularBuffer cb
) {
	int res = pthread_mutex_unlock (&cb->lock) ;
	ASSERT (res == 0) ;
}

// create a new circular buffer
CircularBuffer CircularBuffer_New
(
	size_t item_size,  // size of item in bytes
	uint32_t cap       // max number of items in buffer
) {
	CircularBuffer cb = rm_calloc(1, sizeof(_CircularBuffer) + item_size * cap);

	cb->read       = cb->data ;
	cb->write      = cb->data ;
	cb->item_size  = item_size ;
	cb->item_cap   = cap ;
	cb->item_count = 0 ;
	cb->end_marker = cb->data + (item_size * cap) ;  // end of data marker

	// initialize circular buffer lock
	int res = pthread_mutex_init (&cb->lock, NULL) ;
	ASSERT (res == 0) ;

	return cb;
}

// returns number of items in buffer
uint32_t CircularBuffer_ItemCount
(
	CircularBuffer cb  // buffer to inspect
) {
	ASSERT (cb != NULL) ;

	return cb->item_count ;
}

// returns buffer capacity
uint32_t CircularBuffer_Cap
(
	CircularBuffer cb // buffer
) {
	ASSERT (cb != NULL) ;

	return cb->item_cap ;
}

size_t CircularBuffer_ItemSize
(
	const CircularBuffer cb  // buffer
) {
	ASSERT (cb != NULL) ;

	return cb->item_size ;
}

// return true if buffer is empty
inline bool CircularBuffer_Empty
(
	const CircularBuffer cb  // buffer
) {
	ASSERT (cb != NULL) ;
	return (CircularBuffer_ItemCount (cb) == 0) ;
}

// returns true if buffer is full
inline bool CircularBuffer_Full
(
	const CircularBuffer cb  // buffer
) {
	ASSERT (cb != NULL) ;

	return (CircularBuffer_ItemCount (cb) == cb->item_cap) ;
}

// adds an item to buffer
// returns true on success, false otherwise
// this function is thread-safe
bool CircularBuffer_Add
(
	CircularBuffer cb,  // buffer to populate
	void *item          // item to add
) {
	ASSERT (cb   != NULL) ;
	ASSERT (item != NULL) ;

	_CircularBuffer_Lock (cb) ;

	// check if buffer is full
	if (cb->item_count == cb->item_cap) {
		_CircularBuffer_Unlock (cb) ;
		return false ;
	}

	//--------------------------------------------------------------------------
	// determine item's position within the buffer
	//--------------------------------------------------------------------------

	ASSERT (cb->write <= cb->end_marker) ;

	// handle wrap around
	if (unlikely (cb->write == cb->end_marker)) {
		cb->write = cb->data ;
	}

	// copy item into buffer
	memcpy(cb->write, item, cb->item_size);

	// update write position
	cb->write += cb->item_size ;

	// increase item count
	cb->item_count++ ;

	_CircularBuffer_Unlock (cb) ;

	// report success
	return true;
}

// read oldest item from buffer
// this function is thread-safe
bool CircularBuffer_Read
(
	CircularBuffer cb,  // buffer to read item from
	void *item          // pointer populated with removed item
) {
	ASSERT (cb   != NULL) ;
	ASSERT (item != NULL) ;

	_CircularBuffer_Lock (cb) ;

	// make sure there's data to return
	if (unlikely(cb->item_count == 0)) {
		_CircularBuffer_Unlock (cb) ;
		return false ;
	}

	// update buffer item count
	cb->item_count-- ;

	// copy item from buffer to output
	memcpy (item, cb->read, cb->item_size) ;

	// advance read position
	// wrap around if read reached the end of the buffer
	cb->read += cb->item_size ;

	ASSERT (cb->read <= cb->end_marker) ;
	if (unlikely (cb->read == cb->end_marker)) {
		cb->read = cb->data ;
	}

	_CircularBuffer_Unlock (cb) ;

	return true;
}

// free buffer (does not free its elements if its free callback is NULL)
void CircularBuffer_Free
(
	CircularBuffer cb,                 // buffer to free
	CircularBuffer_ItemFreeCB free_cb  // [optional]
) {
	ASSERT (cb != NULL) ;

	// invoke free callback for each item
	if (free_cb != NULL) {
		for (uint32_t i = 0; i < cb->item_count; i++) {
			// wrap around
			if (cb->read == cb->end_marker) {
				cb->read = cb->data ;
			}

			void *item = cb->read ;
			free_cb (item) ;
			cb->read += cb->item_size ;
		}
	}

	// release circular buffer lock
	int res = pthread_mutex_destroy (&cb->lock) ;
	ASSERT (res == 0) ;

	rm_free (cb) ;
}

