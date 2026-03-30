/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "acutest.h"
#include "../../src/util/heap.h"
#include <time.h>
#include <stdlib.h>

#define arr_alloc_fn malloc
#define arr_realloc_fn realloc
#define arr_free_fn free
#include "../../src/util/arr.h"

// find min value in array
static int find_min
(
	int *arr
) {
	int min = arr[0];
	int n = arr_len(arr);
	for(int i = 1; i < n; i++) {
		if(min > arr[i]) {
			min = arr[i];
		}
	}

	return min;
}

// remove value 'v' from array
static void remove_value
(
	int *arr,
	int v
) {
	int n = arr_len(arr);
	for(int i = 0; i < n; i++) {
		if(arr[i] == v) {
			arr_del_fast(arr, i);
			break;
		}
	}
}

// heap compare function
static int cmp
(
	const void *a,
	const void *b,
	void *ud
) {
	int _a = *(int*)a;
	int _b = *(int*)b;

	return _b - _a;
}

// quick sort compare function
static int qsort_cmp
(
	const void *a,
	const void *b
) {
	int _a = *(int*)a;
	int _b = *(int*)b;

	return _a - _b;
}

static void test_heapCreate(void) {
	void *elem;
	// create a new heap
	heap_t *heap = Heap_new(cmp, NULL);

	TEST_ASSERT(Heap_count(heap) == 0);  // expecting heap to be empty

	//--------------------------------------------------------------------------
	// try to peek top of heap
	//--------------------------------------------------------------------------

	elem = Heap_peek(heap);
	TEST_ASSERT(elem == NULL);  // expecting NULL for empty heap

	//--------------------------------------------------------------------------
	// try to get element from heap
	//--------------------------------------------------------------------------

	elem = Heap_poll(heap);
	TEST_ASSERT(elem == NULL);  // expecting NULL for empty heap

	//--------------------------------------------------------------------------
	// try to locate a none existing element
	//--------------------------------------------------------------------------

	int x = 2;
	TEST_ASSERT(Heap_contains_item(heap, &x) == 0);

	//--------------------------------------------------------------------------
	// try to remove a none existing element
	//--------------------------------------------------------------------------

	elem = Heap_remove_item(heap, &x);
	TEST_ASSERT(elem == NULL);

	// free heap
	Heap_free(heap);
}

static void test_heapPopulate(void) {
	// create a new heap
	heap_t *heap = Heap_new(cmp, NULL);

	int n = 10;
	int elements[n];

	//--------------------------------------------------------------------------
	// populate heap
	//--------------------------------------------------------------------------

	// insert elements 0..9
	for(int i = 0; i < n; i++) {
		elements[i] = i;
		TEST_ASSERT(Heap_offer(&heap, elements + i) == 0);
	}

	// validate number of elements in heap
	TEST_ASSERT(Heap_count(heap) == n);

	// validate expected elements are indeed in heap
	for(int i = 0; i < n; i++) {
		TEST_ASSERT(Heap_contains_item(heap, elements + i) == 1);
	}

	//--------------------------------------------------------------------------
	// empty heap
	//--------------------------------------------------------------------------

	// empty heap by polling until empty
	int *elem;
	for(int i = 0; i < n; i++) {
		elem = (int*)Heap_peek(heap);
		TEST_ASSERT(*elem == i);

		elem = Heap_poll(heap);
		TEST_ASSERT(*elem == i);
	}

	//--------------------------------------------------------------------------
	// validate heap is empty
	//--------------------------------------------------------------------------

	TEST_ASSERT(Heap_count(heap) == 0);

	// free heap
	Heap_free(heap);
}

static void test_heapPopulateDup(void) {
	// create a new heap
	heap_t *heap = Heap_new(cmp, NULL);

	int e = 7;     // duplicated value
	int n = 1000;  // number of duplicates

	//--------------------------------------------------------------------------
	// populate heap
	//--------------------------------------------------------------------------

	// insert duplicated elements
	for(int i = 0; i < n; i++) {
		TEST_ASSERT(Heap_offer(&heap, &e) == 0);
	}

	// validate number of elements in heap
	TEST_ASSERT(Heap_count(heap) == n);

	// validate expected elements are indeed in heap
	for(int i = 0; i < n; i++) {
		TEST_ASSERT(Heap_contains_item(heap, &e) == 1);
	}

	//--------------------------------------------------------------------------
	// empty heap
	//--------------------------------------------------------------------------

	// empty heap by polling until empty
	int *elem;
	for(int i = 0; i < n; i++) {
		elem = (int*)Heap_peek(heap);
		TEST_ASSERT(*elem == e);

		elem = Heap_poll(heap);
		TEST_ASSERT(*elem == e);
	}

	//--------------------------------------------------------------------------
	// validate heap is empty
	//--------------------------------------------------------------------------

	TEST_ASSERT(Heap_count(heap) == 0);

	// free heap
	Heap_free(heap);
}

static void test_heapPopulateRand(void) {
	// seed random
	srand(time(NULL));

	// create a new heap
	heap_t *heap = Heap_new(cmp, NULL);

	int n = 100;
	int elements[n];
	int sorted_elements[n];

	//--------------------------------------------------------------------------
	// generate n random numbers within the range 0..999
	//--------------------------------------------------------------------------

	// insert elements 0..n
	for(int i = 0; i < n; i++) {
		elements[i] = rand() % 1000;
		sorted_elements[i] = elements[i];
		TEST_ASSERT(Heap_offer(&heap, elements + i) == 0);
	}

	// validate number of elements in heap
	TEST_ASSERT(Heap_count(heap) == n);

	// validate expected elements are indeed in heap
	for(int i = 0; i < n; i++) {
		TEST_ASSERT(Heap_contains_item(heap, elements + i) == 1);
	}

	//--------------------------------------------------------------------------
	// sort elements
	//--------------------------------------------------------------------------

	qsort(sorted_elements, n, sizeof(int), qsort_cmp);

	//--------------------------------------------------------------------------
	// validate heap poll ordering
	//--------------------------------------------------------------------------

	int *elem;
	for(int i = 0; i < n; i++) {
		elem = (int*)Heap_peek(heap);
		TEST_ASSERT(*elem == sorted_elements[i]);

		elem = Heap_poll(heap);
		TEST_ASSERT(*elem == sorted_elements[i]);
	}

	//--------------------------------------------------------------------------
	// validate heap is empty
	//--------------------------------------------------------------------------

	TEST_ASSERT(Heap_count(heap) == 0);

	// free heap
	Heap_free(heap);
}

static void test_heapRemoveElement(void) {
	// create a new heap
	heap_t *heap = Heap_new(cmp, NULL);

	//--------------------------------------------------------------------------
	// populate heap
	//--------------------------------------------------------------------------

	int n = 10;
	int elements[n];

	// insert elements 0..9
	for(int i = 0; i < n; i++) {
		elements[i] = i;
		TEST_ASSERT(Heap_offer(&heap, elements + i) == 0);
	}

	// validate number of elements in heap
	TEST_ASSERT(Heap_count(heap) == n);

	//--------------------------------------------------------------------------
	// remove heap head
	//--------------------------------------------------------------------------

	int elem;
	elem = *(int*)Heap_remove_item(heap, elements + 0);

	TEST_ASSERT(elem == elements[0]);

	// validate new head
	elem = *(int*)Heap_peek(heap);
	TEST_ASSERT(elem == 1);

	//--------------------------------------------------------------------------
	// remove 2 from heap
	//--------------------------------------------------------------------------

	elem = *(int*)Heap_remove_item(heap, elements + 2);
	TEST_ASSERT(elem == elements[2]);

	// remove heap head
	elem = *(int*)Heap_poll(heap);
	TEST_ASSERT(elem == 1);

	// validate new head
	elem = *(int*)Heap_peek(heap);
	TEST_ASSERT(elem == 3);

	//--------------------------------------------------------------------------
	// remove last element
	//--------------------------------------------------------------------------

	elem = *(int*)Heap_remove_item(heap, elements + 9);
	TEST_ASSERT(elem == elements[9]);

	// validate new head
	elem = *(int*)Heap_peek(heap);
	TEST_ASSERT(elem == 3);

	//--------------------------------------------------------------------------
	// remove none existing element
	//--------------------------------------------------------------------------

	int x;
	void *res = Heap_remove_item(heap, &x);
	TEST_ASSERT(res == NULL);

	// free heap
	Heap_free(heap);
}

static void test_heapFuzz(void) {
	// seed random
	srand(time(NULL));

	// create a new heap
	heap_t *heap = Heap_new(cmp, NULL);

	int n                = 500;
	int *elements        = arr_new(int, n);  // all elements ever introduced
	int *synced_elements = arr_new(int, n);  // elements synced with heap

	// perform random operations:
	// 1. introduce a new random element
	// 2. remove a random element
	// 3. remove head
	// 
	// after each operation perform validation against head of heap
	for(int i = 0; i < 500; i++) {
		int idx;              // index of element to remove
		int *top;             // head of heap
		int elem;             // element to add
		int op = rand() % 3;  // operation to perform

		switch(op) {
			case 0:
				// introduce a new random element
				elem = rand() % 100;
				arr_append(elements, elem);
				arr_append(synced_elements, elem);
				Heap_offer(&heap, elements + (arr_len(elements)-1));
				break;
			case 1:
				// remove random element
				if(Heap_count(heap) > 0) {
					// remove a random element
					idx = rand() % arr_len(elements);

					if(Heap_remove_item(heap, elements + idx) != NULL) {
						remove_value(synced_elements, elements[idx]);
					}
				}
				break;
			case 2:
				// pop heap head
				top = (int*)Heap_poll(heap);
				if(top != NULL) {
					remove_value(synced_elements, *top);
				}
				break;
			default:
				break;
		}

		//----------------------------------------------------------------------
		// validate head of heap contains the lowest value in elements array
		//----------------------------------------------------------------------
		int heap_count = Heap_count(heap);
		int arr_count  = arr_len(synced_elements);
		TEST_ASSERT(heap_count == arr_count);
		if(heap_count > 0) {
			top = (int*)Heap_peek(heap);
			int min = find_min(synced_elements);
			TEST_ASSERT(*top == min);
		}
	}

	// clean up
	Heap_free(heap);
	arr_free(elements);
	arr_free(synced_elements);
}

TEST_LIST = {
	{"heapCreate", test_heapCreate},
	{"heapPopulate", test_heapPopulate},
	{"heapPopulateDup", test_heapPopulateDup},
	{"heapPopulateRand", test_heapPopulateRand},
	{"heapRemoveElement", test_heapRemoveElement},
	{"heapFuzz", test_heapFuzz},
	{NULL, NULL}
};

