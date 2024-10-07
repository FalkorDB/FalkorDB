/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "src/value.h"
#include "src/util/arr.h"
#include "src/util/rmalloc.h"
#include "src/datatypes/map.h"
#include "src/datatypes/array.h"

void setup() {
	Alloc_Reset();
}
#define TEST_INIT setup();
#include "acutest.h"

void test_empty_map() {
	SIValue map = Map_New(2);

	//--------------------------------------------------------------------------
	// map should be empty
	//--------------------------------------------------------------------------

	TEST_ASSERT(0 == Map_KeyCount(map));

	SIValue keys = Map_Keys(map);
	TEST_ASSERT(0 == SIArray_Length(keys));
	SIArray_Free(keys);

	//--------------------------------------------------------------------------
	// try getting a key
	//--------------------------------------------------------------------------

	SIValue key = SI_ConstStringVal("k");
	SIValue v;
	TEST_ASSERT(!Map_Contains(map, key));
	TEST_ASSERT(!Map_Get(map, key, &v));

	//--------------------------------------------------------------------------
	// try removing a none existing key
	//--------------------------------------------------------------------------

	Map_Remove(map, key);

	// clean up
	Map_Free(map);
}

void test_map_add() {
	SIValue map = Map_New(2);

	SIValue k0        = SI_ConstStringVal("key0");
	SIValue k1        = SI_ConstStringVal("key1");
	SIValue k2        = SI_ConstStringVal("key2");
	SIValue v0        = SI_LongVal(0);
	SIValue v1        = SI_DoubleVal(1);
	SIValue v2        = SI_ConstStringVal("");
	SIValue keys[3]   = {k0, k1, k2};
	SIValue values[3] = {v0, v1, v2};

	// add elements to map
	for(int i = 0; i < 3; i++) {
		Map_Add(&map, keys[i], values[i]);
	}

	// remove none existing key
	SIValue none_existing = SI_ConstStringVal("none_existing");
	Map_Remove(map, none_existing);

	// try to get none existing keys
	SIValue v = SI_NullVal();
	TEST_ASSERT(!Map_Get(map, none_existing, &v));

	// expecting 3 keys
	TEST_ASSERT(Map_KeyCount(map) == 3);

	// verify map content
	for(int i = 0; i < 3; i++) {
		SIValue v = SI_NullVal();

		TEST_ASSERT(Map_Contains(map, keys[i]));

		Map_Get(map, keys[i], &v);
		TEST_ASSERT(SI_TYPE(v) == SI_TYPE(values[i]));
	}

	// verify keys
	SIValue stored_keys = Map_Keys(map);
	TEST_ASSERT(SIArray_Length(stored_keys) == 3);
	for(int i = 0; i < 3; i++) {
		SIValue key = SIArray_Get(stored_keys, i);
		TEST_ASSERT(strcmp(keys[i].stringval, key.stringval) == 0);
	}

	// clean up
	SIArray_Free(stored_keys);
	Map_Free(map);
}

void test_map_add_path() {
	SIValue map = Map_New(0);

	SIValue t       = SI_NullVal();
	SIValue A       = SI_ConstStringVal("A");
	SIValue B       = SI_ConstStringVal("B");
	SIValue C       = SI_ConstStringVal("C");
	SIValue v       = SI_LongVal(0);
	SIValue path[4] = {A, B, C, A};

	//--------------------------------------------------------------------------
	// set map['A']['B']['C'] = 0
	//--------------------------------------------------------------------------

	TEST_ASSERT(Map_AddPath(&map, path, 3, v) == true);

	// get map['A']['B']['C']
	TEST_ASSERT(Map_GetPath(map, path, 3, &t) == true);
	TEST_ASSERT(SIValue_Compare(t, v, NULL) == 0);

	// get map['A']
	TEST_ASSERT(Map_Get(map, A, &t) == true);
	TEST_ASSERT(SI_TYPE(t) == T_MAP);
	TEST_ASSERT(Map_KeyCount(t) == 1);

	// get map['A']['B']
	TEST_ASSERT(Map_Get(t, B, &t) == true);
	TEST_ASSERT(SI_TYPE(t) == T_MAP);
	TEST_ASSERT(Map_KeyCount(t) == 1);

	// get map['A']['B']['C']
	TEST_ASSERT(Map_Get(t, C, &t) == true);
	TEST_ASSERT(SIValue_Compare(t, v, NULL) == 0);

	//--------------------------------------------------------------------------
	// set map['A']['C'] = 1
	//--------------------------------------------------------------------------

	path[1] = C;
	v = SI_LongVal(1);
	TEST_ASSERT(Map_AddPath(&map, path, 2, v) == true);

	// get map['A']['C']
	TEST_ASSERT(Map_GetPath(map, path, 2, &t) == true);
	TEST_ASSERT(SIValue_Compare(t, v, NULL) == 0);


	//--------------------------------------------------------------------------
	// set map['C'] = 2
	//--------------------------------------------------------------------------

	path[0] = C;
	v = SI_LongVal(2);
	TEST_ASSERT(Map_AddPath(&map, path, 1, v) == true);

	// get map['C']
	TEST_ASSERT(Map_GetPath(map, path, 1, &t) == true);
	TEST_ASSERT(SIValue_Compare(t, v, NULL) == 0);

	//--------------------------------------------------------------------------
	// set map['A']['B']['C']['A'] = 3
	//--------------------------------------------------------------------------

	// set should fail!
	path[0] = A;
	path[1] = B;
	path[2] = C;
	path[3] = A;
	v = SI_LongVal(3);
	TEST_ASSERT(Map_AddPath(&map, path, 4, v) == false);

	// get map['A']['B']['C']['A']
	TEST_ASSERT(Map_GetPath(map, path, 4, &t) == false);

	//--------------------------------------------------------------------------
	// update same element
	//--------------------------------------------------------------------------

	// set map['A']['B']['C'] = v
	TEST_ASSERT(Map_AddPath(&map, path, 3, v) == true);

	// reset map['A']['B']['C'] = v
	TEST_ASSERT(Map_AddPath(&map, path, 3, v) == false);

	// assign a different value to map['A']['B']['C'] = v
	v = SI_LongVal(-3);
	TEST_ASSERT(Map_AddPath(&map, path, 3, v) == true);

	// clean up
	Map_Free(map);
}

void test_map_remove() {
	SIValue map = Map_New(2);

	SIValue  k0  =  SI_ConstStringVal("key0");
	SIValue  k1  =  SI_ConstStringVal("key1");
	SIValue  k2  =  SI_ConstStringVal("key2");
	SIValue  v0  =  SI_LongVal(0);
	SIValue  v1  =  SI_DoubleVal(1);
	SIValue  v2  =  SI_ConstStringVal("");
	SIValue keys[3] = {k0, k1, k2};
	SIValue values[3] = {v0, v1, v2};

	// add elements to map
	for(int i = 0; i < 3; i++) {
		Map_Add(&map, keys[i], values[i]);
	}

	// remove none existing key
	SIValue none_existing = SI_ConstStringVal("none_existing");
	Map_Remove(map, none_existing);

	for(int i = 0; i < 3; i++) {
		SIValue v = SI_NullVal();

		// remove key
		TEST_ASSERT(Map_Contains(map, keys[i]));
		Map_Remove(map, keys[i]);

		// try removing removed key
		Map_Remove(map, keys[i]);

		// validate key count
		TEST_ASSERT(3 - i - 1 == Map_KeyCount(map));

		// get removed key
		TEST_ASSERT(!Map_Contains(map, keys[i]));
		TEST_ASSERT(!Map_Get(map, keys[0], &v));
	}

	SIValue stored_keys = Map_Keys(map);
	TEST_ASSERT(0 == SIArray_Length(stored_keys));

	// clean up
	SIArray_Free(stored_keys);
	Map_Free(map);
}

void test_map_remove_path() {
	SIValue map = Map_New(0);

	SIValue A       = SI_ConstStringVal("A");
	SIValue B       = SI_ConstStringVal("B");
	SIValue C       = SI_ConstStringVal("C");
	SIValue t       = SI_NullVal();
	SIValue v       = SI_LongVal(0);
	SIValue path[3] = {A, B, C};

	//--------------------------------------------------------------------------
	// remove non existing path
	//--------------------------------------------------------------------------

	// try to delete map['A']['B']['C']
	TEST_ASSERT(Map_RemovePath(map, path, 3) == false);

	// create map['A']['B']['C'] = 0
	TEST_ASSERT(Map_AddPath(&map, path, 3, v) == true);

	// try to delete map['A']['B']['A']
	path[2] = A;
	TEST_ASSERT(Map_RemovePath(map, path, 3) == false);

	// try to delete map['C']
	TEST_ASSERT(Map_RemovePath(map, &C, 1) == false);

	//--------------------------------------------------------------------------
	// remove existing path
	//--------------------------------------------------------------------------

	// delete map['A']['B']
	TEST_ASSERT(Map_RemovePath(map, path, 2) == true);
	// repeate same deletion
	TEST_ASSERT(Map_RemovePath(map, path, 2) == false);

	// make sure map['A'] exists
	TEST_ASSERT(Map_Get(map, A, &t) == true);
	TEST_ASSERT(SI_TYPE(t) == T_MAP);

	// make sure map['A']['B'] doesn't exists
	TEST_ASSERT(Map_GetPath(map, path, 2, &t) == false);

	// remove root element map['A']
	TEST_ASSERT(Map_RemovePath(map, path, 1) == true);
	TEST_ASSERT(Map_KeyCount(map) == 0);

	// clean up
	Map_Free(map);
}

void test_map_tostring() {
	SIValue  k;
	SIValue  v;
	SIValue map = Map_New(3);
	SIValue inner_map = Map_New(3);

	k = SI_ConstStringVal("inner_key");
	v = SI_NullVal();
	Map_Add(&inner_map, k, v);

	k = SI_ConstStringVal("key0");
	v = SI_LongVal(0);
	Map_Add(&map, k, v);

	k = SI_ConstStringVal("key1");
	v = SI_DoubleVal(1);
	Map_Add(&map, k, v);

	k = SI_ConstStringVal("key2");
	v = SI_ConstStringVal("val0");
	Map_Add(&map, k, v);

	k = SI_ConstStringVal("key3");
	Map_Add(&map, k, inner_map);

	size_t buf_len = 256;
	size_t bytes_written = 0;
	char *buf = (char *)rm_malloc(sizeof(char) * 256);
	Map_ToString(map, &buf, &buf_len, &bytes_written);

	TEST_ASSERT(strcmp(buf, "{key0: 0, key1: 1.000000, key2: val0, key3: {inner_key: NULL}}") == 0);

	rm_free(buf);
	SIValue_Free(map);
	SIValue_Free(inner_map);
}

TEST_LIST = {
	{"empty_map", test_empty_map},
	{"map_Add", test_map_add},
	{"map_AddPath", test_map_add_path},
	{"map_Remove", test_map_remove},
	{"map_RemovePath", test_map_remove_path},
	{"map_ToString", test_map_tostring},
	{NULL, NULL}
};
