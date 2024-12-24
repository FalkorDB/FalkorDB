/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

/* A map is a set of key/value pairs
 * where key is a string and value can be anyone of the following types:
 * map, array, node, edge, path, date, string, bool, numeric and NULL
 *
 * this implimantaion of map, uses SIValue for the stored values
 * and keeps the keys sorted for quick lookup.
 *
 * the underline structure of map is an array of key/value pairs
 * [ (key/value), (key/value), ... (key/value) ] */

#include "../value.h"

typedef struct Pair {
	SIValue key;  // key associated with value
	SIValue val;  // value stored under key
} Pair;

typedef Pair *Map;

// retrieves value under key, map[key]
// where key is const string 
// return true and set 'value' if key is in map
// otherwise return false
#define MAP_GET(map, key, value) Map_Get(map, SI_ConstStringVal(key), &value)

// create a new map
SIValue Map_New
(
	uint capacity     // map initial capacity
);

// create a map from keys and values arrays
// keys and values are both of length n
SIValue Map_FromArrays
(
	const SIValue *keys,    // keys
	const SIValue *values,  // values
	uint n                  // arrays length
);

// clones map
SIValue Map_Clone
(
	SIValue map  // map to clone
);

// adds key/value to map
void Map_Add
(
	SIValue *map,  // map to add element to
	SIValue key,   // key under which value is added
	SIValue value  // value to add under key
);

// removes key from map
void Map_Remove
(
	SIValue map,  // map to remove key from
	SIValue key   // key to remove
);

// clears map
void Map_Clear
(
	SIValue map  // map to clear
);

// retrieves value under key, map[key]
// return true and set 'value' if key is in map
// otherwise return false
bool Map_Get
(
	SIValue map,    // map to get value from
	SIValue key,    // key to lookup value
	SIValue *value  // [output] value to retrieve
);

// checks if 'key' is in map
bool Map_Contains
(
	SIValue map,  // map to query
	SIValue key   // key to look-up
);

// return number of keys in map
uint Map_KeyCount
(
	SIValue map  // map to count number of keys in
);

// return an SIArray of all keys in map
// caller should free returned array with SIArray_Free
SIValue Map_Keys
(
	SIValue map  // map to extract keys from
);

// populate 'key' and 'value' pointers with
// the map contents at the indicated index
void Map_GetIdx
(
	const SIValue map,
	uint idx,
	SIValue *key,
	SIValue *value
);

// compare two maps
// if map lengths are not equal, the map with the greater length is
// considered greater
//
// {a:1, b:2} > {Z:100}
//
// if both maps have the same length they are sorted and comparision is done
// on a key by key basis:
//
// if the key sets are not equal, key names are compared lexicographically

// otherwise compare the values for that key and, if they are
// inequal, return the inequality
//
// {a:1, b:3} > {a:1, b:2} as 3 > 2
// {a:1, c:1} > {b:1, a:1} as 'c' > 'b'
int Map_Compare
(
	SIValue mapA,
	SIValue mapB,
	int *disjointOrNull
);

// compute hash code for map
XXH64_hash_t Map_HashCode
(
	SIValue map
);

// populate 'buf' with string representation of map
void Map_ToString
(
	SIValue map,          // map to get string representation from
	char **buf,           // buffer to populate
	size_t *bufferLen,    // size of buffer
	size_t *bytesWritten  // length of string
);

// free map
void Map_Free
(
	SIValue map  // map to free
);

