/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "src/util/rmalloc.h"
#include "src/util/arr.h"

void setup() {
	Alloc_Reset();
}

#define TEST_INIT setup();
#include "acutest.h"

int int_identity
(
	int x
) {
	return x;
}

int int_cmp
(
	const void *a,
	const void *b
) {
	return *(int*)a - *(int*)b;
}

int str_cmp
(
	const void *a,
	const void *b
) {
	return strcmp(*(const char**)a, *(const char**)b);
}

void test_arrCloneWithCB(void) {
	int *arr = array_new(int, 10);
	for(int i = 0; i < 10; i++) {
		array_append(arr, i);
	}

	int *arr_clone;
	array_clone_with_cb(arr_clone, arr, int_identity);

	TEST_ASSERT(array_len(arr) == array_len(arr_clone));

	for(int i = 0; i < 10; i++) {
		TEST_ASSERT(arr[i] == arr_clone[i]);
	}

	array_free(arr);
	array_free(arr_clone);
}

void test_arrDedupe(void) {
	// test array dedupe

	//--------------------------------------------------------------------------
	// test dedupe empty array
	//--------------------------------------------------------------------------

	int *arr = array_new(int, 0);
	array_dedupe(arr, int_cmp);

	TEST_ASSERT(array_len(arr) == 0);

	array_free(arr);

	//--------------------------------------------------------------------------
	// test dedupe int array
	//--------------------------------------------------------------------------

	// create array without any duplicates
	arr = array_new(int, 10);

	for(int i = 0; i < 10; i++) array_append(arr, i);

	array_dedupe(arr, int_cmp);
	TEST_ASSERT(array_len(arr) == 10);

	array_free(arr);

	// create array with 10 duplicates
	arr = array_new(int, 10);

	for(int i = 0; i < 10; i++) array_append(arr, 2);

	array_dedupe(arr, int_cmp);
	TEST_ASSERT(array_len(arr) == 1);
	TEST_ASSERT(arr[0] == 2);

	array_free(arr);

	// dedupe [1, 2, 3, 4, 5, 5, 4, 3, 2, 1]
	arr = array_new(int, 10);
	for(int i = 0; i < 5; i++) array_append(arr, i + 1);
	for(int i = 0; i < 5; i++) array_append(arr, 5 - i);

	array_dedupe(arr, int_cmp);
	TEST_ASSERT(array_len(arr) == 5);
	int sum = 0;
	for(int i = 0; i < 5; i++) sum += arr[i];
	TEST_ASSERT(sum == 15);

	//--------------------------------------------------------------------------
	// test dedupe string array
	//--------------------------------------------------------------------------

	// create array without any duplicates
	char **str_arr = array_new(char*, 10);

	for(int i = 0; i < 10; i++) {
		char *s;
		asprintf(&s, "str%d", i);
		array_append(str_arr, s);
	}

	array_dedupe(str_arr, str_cmp);
	TEST_ASSERT(array_len(str_arr) == 10);

	array_free_cb(str_arr, free);

	// create array with 10 duplicates

	str_arr = array_new(char*, 10);

	for(int i = 0; i < 10; i++) {
		char *s;
		asprintf(&s, "hello");
		array_append(str_arr, s);
	}

	array_dedupe(str_arr, str_cmp);
	TEST_ASSERT(array_len(str_arr) == 1);
	TEST_ASSERT(strcmp(str_arr[0], "hello") == 0);

	array_free_cb(str_arr, free);
}

TEST_LIST = {
	{ "arrCloneWithCB", test_arrCloneWithCB},
	{ "arrDedupe", test_arrDedupe},
	{ NULL, NULL }
};

