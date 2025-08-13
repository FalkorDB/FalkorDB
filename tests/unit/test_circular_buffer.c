/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "src/util/rmalloc.h"
#include "src/util/arr.h"
#include "src/util/circular_buffer.h"

void setup() {
	Alloc_Reset();
}

#define TEST_INIT setup();
#include "acutest.h"

void test_CircularBufferInit(void) {
	CircularBuffer buff = CircularBuffer_New(sizeof(int), 16);

	// a new circular buffer should be empty
	TEST_ASSERT(CircularBuffer_Empty(buff) == true);

	// item count of an empty circular buffer should be 0
	TEST_ASSERT(CircularBuffer_ItemCount(buff) == 0);

	// buffer should have available slots in it i.e. not full
	TEST_ASSERT(CircularBuffer_Full(buff) == false);

	// clean up
	CircularBuffer_Free(buff, NULL);
}

void test_CircularBufferPopulation(void) {
	int n;
	int cap = 16;
	CircularBuffer buff = CircularBuffer_New(sizeof(int), cap);

	// remove item from an empty buffer should report failure
	TEST_ASSERT(CircularBuffer_Read(buff, &n) == false);

	//--------------------------------------------------------------------------
	// fill buffer
	//--------------------------------------------------------------------------

	for(int i = 0; i < cap; i++) {
		// make sure item was added
		TEST_ASSERT(CircularBuffer_Add(buff, &i) == true);

		// validate buffer's item count
		TEST_ASSERT(CircularBuffer_ItemCount(buff) == i+1);
	}

	TEST_ASSERT(CircularBuffer_Full(buff) == true);

	// forcefully try to overflow buffer
	for(int i = 0; i < 10; i++) {
		TEST_ASSERT(CircularBuffer_Add(buff, &n) == false);
	}

	//--------------------------------------------------------------------------
	// empty buffer
	//--------------------------------------------------------------------------

	for(int i = 0; i < cap; i++) {
		// get item from buffer
		TEST_ASSERT(CircularBuffer_Read(buff, &n) == true);

		// validate item's value
		TEST_ASSERT(n == i);
	}

	TEST_ASSERT(CircularBuffer_Empty(buff) == true);

	// forcefully try to read an item from an empty buffer
	for(int i = 0; i < 10; i++) {
		TEST_ASSERT(CircularBuffer_Read(buff, &n) == false);
	}

	// clean up
	CircularBuffer_Free(buff, NULL);
}

void test_CircularBuffer_Circularity(void) {
	int n;
	int cap = 16;
	CircularBuffer buff = CircularBuffer_New(sizeof(int), cap);

	//--------------------------------------------------------------------------
	// fill buffer
	//--------------------------------------------------------------------------
	for(int i = 0; i < cap; i++) {
		// make sure item was added
		TEST_ASSERT(CircularBuffer_Add(buff, &i) == true);
	}
	TEST_ASSERT(CircularBuffer_Full(buff) == true);

	// try to overflow buffer
	TEST_ASSERT(CircularBuffer_Add(buff, &n) == false);

	// removing an item should make space in the buffer
	TEST_ASSERT(CircularBuffer_Read(buff, &n) != false);
	TEST_ASSERT(CircularBuffer_Add(buff, &n) == true);

	//--------------------------------------------------------------------------
	// clear buffer
	//--------------------------------------------------------------------------

	while(CircularBuffer_Empty(buff) == false) {
		CircularBuffer_Read(buff, &n);
	}

	// add/remove elements cycling through the buffer multiple times
	for(int i = 0; i < cap * 4; i++) {
		TEST_ASSERT(CircularBuffer_Add(buff, &i) == true);
		TEST_ASSERT(CircularBuffer_Read(buff, &n) == true);
		TEST_ASSERT(n == i);
	}
	TEST_ASSERT(CircularBuffer_Empty(buff) == true);

	// clean up
	CircularBuffer_Free(buff, NULL);
}

void test_CircularBuffer_free(void) {
	//--------------------------------------------------------------------------
	// fill a buffer of size 16 with int *
	//--------------------------------------------------------------------------

	uint cap = 16;
	CircularBuffer buff = CircularBuffer_New(sizeof(int64_t *), cap);
	for(int i = 0; i < cap; i++) {
		int64_t *j = malloc(sizeof(int64_t));
		CircularBuffer_Add(buff, (void*)&j);
	}

	//--------------------------------------------------------------------------
	// free the buffer
	//--------------------------------------------------------------------------

	for(int i = 0; i < cap; i++) {
		int64_t *item;
		CircularBuffer_Read(buff, &item);
		free(item);
	}

	CircularBuffer_Free(buff, NULL);
}

void _assert_val_cb
(
	const void *item,
	void *user_data
) {
	int n = array_pop((int *)user_data);
	TEST_ASSERT(n == *(int *)item);
}

TEST_LIST = {
	{"CircularBuffer_Init", test_CircularBufferInit},
	{"CircularBuffer_Population", test_CircularBufferPopulation},
	{"CircularBuffer_Circularity", test_CircularBuffer_Circularity},
	{"CircularBuffer_Free", test_CircularBuffer_free},
	{NULL, NULL}
};

