/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include <stdbool.h>
#include <sys/types.h>

// forward declaration
typedef struct _CircularBuffer _CircularBuffer;
typedef _CircularBuffer* CircularBuffer;

CircularBuffer CircularBuffer_New
(
	size_t item_size,  // size of item in bytes
	uint cap           // max number of items in buffer
);

// returns number of items in buffer
uint64_t CircularBuffer_ItemCount
(
	CircularBuffer cb  // buffer
);

// returns buffer capacity
uint64_t CircularBuffer_Cap
(
	CircularBuffer cb // buffer
);

uint CircularBuffer_ItemSize
(
	const CircularBuffer cb  // buffer
);

// return true if buffer is empty
bool CircularBuffer_Empty
(
	const CircularBuffer cb  // buffer to inspect
);

// returns true if buffer is full
bool CircularBuffer_Full
(
	const CircularBuffer cb  // buffer to inspect
);

// adds an item to buffer
// returns 1 on success, 0 otherwise
int CircularBuffer_Add
(
	CircularBuffer cb,  // buffer to populate
	void *item          // item to add
);

// reserve a slot within buffer
// returns a pointer to a 'item size' slot within the buffer
// this function is thread-safe and lock-free
void *CircularBuffer_Reserve
(
	CircularBuffer cb  // buffer to populate
);

// read oldest item from buffer
void *CircularBuffer_Read
(
	CircularBuffer cb,  // buffer to read item from
	void *item          // [optional] pointer populated with removed item
);

// sets the read pointer to the beginning of the buffer
void CircularBuffer_ResetReader
(
	CircularBuffer cb  // circular buffer
);

// free buffer (does not free its elements if its free callback is NULL)
void CircularBuffer_Free
(
	CircularBuffer cb  // buffer to free
);

