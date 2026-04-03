/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stdbool.h>
#include <sys/types.h>

// forward declaration
typedef struct _CircularBuffer _CircularBuffer;
typedef _CircularBuffer* CircularBuffer;

// item free callback
typedef void (*CircularBuffer_ItemFreeCB)(void *);

// create a new circular buffer
CircularBuffer CircularBuffer_New
(
	size_t item_size,  // size of item in bytes
	uint32_t cap       // max number of items in buffer
);

// returns number of items in buffer
uint32_t CircularBuffer_ItemCount
(
	CircularBuffer cb  // buffer
);

// returns buffer capacity
uint32_t CircularBuffer_Cap
(
	CircularBuffer cb // buffer
);

size_t CircularBuffer_ItemSize
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
// returns true on success, false otherwise
// this function is thread-safe
bool CircularBuffer_Add
(
	CircularBuffer cb,  // buffer to populate
	void *item          // item to add
);

// read oldest item from buffer
// this function is thread-safe
bool CircularBuffer_Read
(
	CircularBuffer cb,  // buffer to read item from
	void *item          // pointer populated with removed item
);

// free buffer (does not free its elements if its free callback is NULL)
void CircularBuffer_Free
(
	CircularBuffer cb,                 // buffer to free
	CircularBuffer_ItemFreeCB free_cb  // [optional]
);

