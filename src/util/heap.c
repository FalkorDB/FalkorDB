#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <string.h>

#include "heap.h"

#define DEFAULT_CAPACITY 13

struct heap_s
{
	unsigned int size;   // size of array
	unsigned int count;  // items within heap
	void *udata;         // user data
	heap_cmp cmp;        // item compare function
	void *array[];       // heap elements
};

// compute left child index of given parent
static int __child_left
(
	const int idx
) {
	return idx * 2 + 1;
}

// compute right child index of given parent
static int __child_right
(
	const int idx
) {
	return idx * 2 + 2;
}

// compute parent index for given child
static int __parent
(
	const int idx
) {
	return (idx - 1) / 2;
}

static void __swap
(
	heap_t *hp,
	const int i1,
	const int i2
) {
	void *tmp = hp->array[i1];

	hp->array[i1] = hp->array[i2];
	hp->array[i2] = tmp;
}

static int __pushup
(
	heap_t *hp,
	unsigned int idx
) {
	// 0 is the root node
	while (0 != idx)
	{
		int parent = __parent(idx);

		// we are smaller than the parent
		if(hp->cmp(hp->array[idx], hp->array[parent], hp->udata) < 0) {
			return -1;
		} else {
			__swap(hp, idx, parent);
		}

		idx = parent;
	}

	return idx;
}

static void __pushdown
(
	heap_t *hp,
	unsigned int idx
) {
	while (1) {
		unsigned int childl, childr, child;

		// get left and right child indices
		childl = __child_left(idx);
		childr = __child_right(idx);

		// see if right child index is within bounds
		if(childr >= hp->count) {
			// see if left child index is within bounds
			if(childl >= hp->count) {
				// we've reached a leaf node
				// can't push down any further
				return;
			}

			// only left child should be considered
			child = childl;
		}
		// find smallest child
		else if(hp->cmp(hp->array[childl], hp->array[childr], hp->udata) > 0) {
			child = childl;
		} else {
			child = childr;
		}

		// idx is smaller than child
		if(hp->cmp(hp->array[idx], hp->array[child], hp->udata) < 0) {
			__swap(hp, idx, child);
			idx = child;
		} else {
			// parent is smaller than its children we can stop
			return;
		}
	}
}

// return item's index on the heap's array; otherwise -1
static int __item_get_idx
(
	const heap_t *hp,
	const void *item
) {
	unsigned int idx;

	for(idx = 0; idx < hp->count; idx++) {
		if(hp->array[idx] == item) {
			return idx;
		}
	}

	return -1;
}

static void __Heap_offerx
(
	heap_t *hp,
	void *item
) {
	hp->array[hp->count] = item;

	// ensure heap properties
	__pushup(hp, hp->count++);
}

// return a new heap on success; NULL otherwise
static heap_t* __ensurecapacity
(
	heap_t *hp
) {
	if(hp->count < hp->size) {
		return hp;
	}

	hp->size *= 2;

	return realloc(hp, Heap_sizeof(hp->size));
}

// create new heap and initialise it
heap_t *Heap_new
(
	heap_cmp cmp,  // cmp callback used to get an item's priority
	void *udata    // udata user data passed through to cmp callback
) {
	heap_t *hp = malloc(Heap_sizeof(DEFAULT_CAPACITY));

	if(!hp) {
		return NULL;
	}

	Heap_init(hp, cmp, udata, DEFAULT_CAPACITY);

	return hp;
}

 // initialise heap
void Heap_init
(
	heap_t *hp,        // heap to initialise
	heap_cmp cmp,      // cmp Callback used to get an item's priority
	void *udata,       // udata User data passed through to cmp callback
	unsigned int size  // size Initial size of the heap's array
) {
	hp->cmp   = cmp;
	hp->size  = size;
	hp->count = 0;
	hp->udata = udata;
}

// add item, ensures that the data structure can hold the item
// NOTE: realloc() possibly called
// return 0 on success; -1 on failure
int Heap_offer
(
	heap_t **hp,  // pointer to the heap, changed when heap is enlarged
	void *item    // item The item to be added
) {
	if(NULL == (*hp = __ensurecapacity(*hp))) {
		return -1;
	}

	__Heap_offerx(*hp, item);
	return 0;
}

// add item
// an error will occur if there isn't enough space for this item
// return 0 on success; -1 on error
int Heap_offerx
(
	heap_t *hp,  // item The item to be added
	void *item
) {
	if(hp->count == hp->size) {
		return -1;
	}

	__Heap_offerx(hp, item);

	return 0;
}

// remove the item with the top priority
// return top item
void *Heap_poll
(
	heap_t *hp
) {
	if(0 == Heap_count(hp)) {
		return NULL;
	}

	void *item = hp->array[0];

	hp->array[0] = hp->array[hp->count - 1];
	hp->count--;

	if(hp->count > 1) {
		__pushdown(hp, 0);
	}

	return item;
}

// return top item of the heap
void *Heap_peek
(
	const heap_t *hp
) {
	if(0 == Heap_count(hp)) {
		return NULL;
	}

	return hp->array[0];
}

// clear all items
// note: does not free items
// only use if item memory is managed outside of heap
void Heap_clear
(
	heap_t *hp
) {
	hp->count = 0;
}

// return number of items in heap
int Heap_count
(
	const heap_t *hp
) {
	return hp->count;
}

// return size of array
int Heap_size
(
	const heap_t *hp
) {
	return hp->size;
}

// return number of bytes needed for a heap of this size
size_t Heap_sizeof
(
	unsigned int size
) {
	return sizeof(heap_t) + size * sizeof(void *);
}

// remove item
// return item to be removed; NULL if item does not exist
void *Heap_remove_item
(
	heap_t *hp,
	const void *item  // the item that is to be removed
) {
	int idx = __item_get_idx(hp, item);

	if(idx == -1) {
		return NULL;
	}

	// swap the item we found with the last item on the heap
	void *ret_item = hp->array[idx];
	hp->array[idx] = hp->array[hp->count - 1];
	hp->array[hp->count - 1] = NULL;

	hp->count -= 1;
	if(idx < hp->count) {
		if(hp->cmp(hp->array[idx], ret_item, hp->udata) < 0) {
			// replacement > removed
			// ensure heap property
			__pushdown(hp, idx);
		} else {
			// replacement <= removed
			// ensure heap property
			__pushup(hp, idx);
		}
	}

	return ret_item;
}

// test membership of item
// return 1 if the heap contains this item; otherwise 0
int Heap_contains_item
(
	const heap_t *hp,
	const void *item
) {
	return __item_get_idx(hp, item) != -1;
}

static void _Heap_print
(
	const heap_t *hp,
	int idx,
	int level
) {
	int val = *(int*)hp->array[idx];
	for(int i = 0; i < level; i++) {
		putchar('\t');
	}
	printf("%d\n", val);

	int l = __child_left(idx);
	int r = __child_right(idx);

	if(l < Heap_count(hp)) {
		_Heap_print(hp, l, level+1);
	}
	
	if(r < Heap_count(hp)) {
		_Heap_print(hp, r, level+1);
	}
}

// prints heap
void Heap_print
(
	const heap_t *hp
) {
	if(Heap_count(hp) == 0) {
		return;
	}
	_Heap_print(hp, 0, 0);
}

// free heap
void Heap_free
(
	heap_t *hp
) {
	free(hp);
}
