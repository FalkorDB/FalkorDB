#pragma once
#include <stdlib.h>

typedef struct heap_s heap_t;

// heap element compare function signature
typedef int (*heap_cmp)(const void *, const void *, void *udata);

// create new heap and initialise it
heap_t *Heap_new
(
	heap_cmp cmp,  // cmp callback used to get an item's priority
	void *udata    // udata user data passed through to cmp callback
);

 // initialise heap
void Heap_init
(
	heap_t *hp,        // heap to initialise
	heap_cmp cmp,      // cmp Callback used to get an item's priority
	void *udata,       // udata User data passed through to cmp callback
	unsigned int size  // size Initial size of the heap's array
);

// add item, ensures that the data structure can hold the item
// NOTE: realloc() possibly called
// return 0 on success; -1 on failure
int Heap_offer
(
	heap_t **hp_ptr,  // pointer to the heap, changed when heap is enlarged
	void *item        // item The item to be added
);

// add item
// an error will occur if there isn't enough space for this item
// return 0 on success; -1 on error
int Heap_offerx
(
	heap_t *hp,  // item The item to be added
	void *item
);

// remove the item with the top priority
// return top item
void *Heap_poll
(
	heap_t *hp
);

// return top item of the heap
void *Heap_peek
(
	const heap_t *hp
);

// clear all items
// note: does not free items
// only use if item memory is managed outside of heap
void Heap_clear
(
	heap_t *hp
);

// return number of items in heap
int Heap_count
(
	const heap_t *hp
);

// return size of array
int Heap_size
(
	const heap_t *hp
);

// return number of bytes needed for a heap of this size
size_t Heap_sizeof
(
	unsigned int size
);

// remove item
// return item to be removed; NULL if item does not exist
void *Heap_remove_item
(
	heap_t *hp,
	const void *item  // the item that is to be removed
);

// test membership of item
// return 1 if the heap contains this item; otherwise 0
int Heap_contains_item
(
	const heap_t *hp,
	const void *item // the item to test
);

// prints heap
void Heap_print
(
	const heap_t *hp  // heap to print
);

// free heap
void Heap_free
(
	heap_t *hp  // heap to free
);

