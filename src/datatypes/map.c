/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "map.h"
#include "array.h"
#include "../util/arr.h"
#include "../util/rmalloc.h"
#include "../util/strutil.h"

#include <stdlib.h>

static inline int _key_cmp
(
	const Pair *a,
	const Pair *b
) {
	return strcmp(a->key.stringval, b->key.stringval);
}

static inline Pair Pair_New
(
	SIValue key,
	SIValue val
) {
	ASSERT(SI_TYPE(key) & T_STRING);
	return (Pair) {
		.key = SI_CloneValue(key), .val = SI_CloneValue(val)
	};
}

static void Pair_Free
(
	Pair p
) {
	SIValue_Free(p.key);
	SIValue_Free(p.val);
}

static int Map_KeyIdx
(
	SIValue map,
	SIValue key
) {
	ASSERT(SI_TYPE(map) & T_MAP);
	ASSERT(SI_TYPE(key) & T_STRING);

	Map m = map.map;
	uint n = array_len(m);

	// search for key in map
	for(uint i = 0; i < n; i++) {
		Pair pair = m[i];
		if(strcmp(pair.key.stringval, key.stringval) == 0) {
			return i;
		}
	}

	// key not in map
	return -1;
}

// create a new map
SIValue Map_New
(
	uint capacity
) {
	SIValue map;
	map.map = array_new(Pair, capacity);
	map.type = T_MAP;
	map.allocation = M_SELF;
	return map;
}

// create a map from keys and values arrays
// keys and values are both of length n
SIValue Map_FromArrays
(
	const SIValue *keys,    // keys
	const SIValue *values,  // values
	uint n                  // arrays length
) {
	ASSERT(keys   != NULL);
	ASSERT(values != NULL);

	SIValue map = Map_New(n);

	for(uint i = 0; i < n; i++) {
		array_append(map.map, Pair_New(keys[i], values[i]));
	}

	return map;
}

// clone map
SIValue Map_Clone
(
	SIValue map  // map to clone
) {
	ASSERT(SI_TYPE(map) & T_MAP);

	uint key_count = Map_KeyCount(map);
	SIValue clone = Map_New(key_count);

	for(uint i = 0; i < key_count; i++) {
		Pair p = map.map[i];
		Map_Add(&clone, p.key, p.val);
	}

	return clone;
}

// adds key/value to map
void Map_Add
(
	SIValue *map,
	SIValue key,
	SIValue value
) {
	ASSERT(SI_TYPE(*map) & T_MAP);
	ASSERT(SI_TYPE(key) & T_STRING);

	// remove key if already existed
	Map_Remove(*map, key);

	// create a new pair
	Pair pair = Pair_New(key, value);

	// add pair to the end of map
	array_append(map->map, pair);
}

// removes key from map
void Map_Remove
(
	SIValue map,
	SIValue key
) {
	ASSERT(SI_TYPE(map) & T_MAP);
	ASSERT(SI_TYPE(key) & T_STRING);

	Map m = map.map;

	// search for key in map
	int idx = Map_KeyIdx(map, key);

	// key missing from map
	if(idx == -1) return;

	// override removed key with last pair
	Pair_Free(m[idx]);
	array_del_fast(m, idx);
}

// clears map
void Map_Clear
(
	SIValue map  // map to clear
) {
	ASSERT(SI_TYPE(map) & T_MAP);

	Map m  = map.map;
	uint n = array_len(m);

	for(uint i = 0; i < n; i++) {
		Pair_Free(m[i]);
	}

	array_clear(m);
}

// retrieves value under key, map[key]
// sets 'value' to NULL if key isn't in map
bool Map_Get
(
	SIValue map,
	SIValue key,
	SIValue *value
) {
	ASSERT(SI_TYPE(map) & T_MAP);
	ASSERT(SI_TYPE(key) & T_STRING);
	ASSERT(value != NULL);

	int idx = Map_KeyIdx(map, key);

	// key isn't in map, set 'value' to NULL and return
	if(idx == -1) {
		*value = SI_NullVal();
		return false;
	} else {
		*value = SI_ShareValue(map.map[idx].val);
		return true;
	}
}

void Map_GetIdx
(
	const SIValue map,
	uint idx,
	SIValue *key,
	SIValue *value
) {
	ASSERT(SI_TYPE(map) & T_MAP);
	ASSERT(idx < Map_KeyCount(map));
	ASSERT(key != NULL);
	ASSERT(value != NULL);

	Pair p = map.map[idx];

	*key = SI_ShareValue(p.key);
	*value = SI_ShareValue(p.val);
}

// checks if 'key' is in map
bool Map_Contains
(
	SIValue map,
	SIValue key
) {
	ASSERT(SI_TYPE(map) & T_MAP);
	ASSERT(SI_TYPE(key) & T_STRING);

	return (Map_KeyIdx(map, key) != -1);
}

uint Map_KeyCount
(
	SIValue map
) {
	ASSERT(SI_TYPE(map) & T_MAP);
	return array_len(map.map);
}

SIValue Map_Keys
(
	SIValue map
) {
	ASSERT(SI_TYPE(map) & T_MAP);

	uint key_count = Map_KeyCount(map);
	SIValue keys = SIArray_New(key_count);

	for(uint i = 0; i < key_count; i++) {
		Pair p = map.map[i];
		SIArray_Append(&keys, p.key);
	}

	return keys;
}

int Map_Compare
(
	SIValue mapA,
	SIValue mapB,
	int *disjointOrNull
) {
	int   order        =  0;
	Map   A            =  mapA.map;
	Map   B            =  mapB.map;
	uint  key_count    =  Map_KeyCount(mapA);
	uint  A_key_count  =  Map_KeyCount(mapA);
	uint  B_key_count  =  Map_KeyCount(mapB);

	if(A_key_count != B_key_count) {
		if(A_key_count > B_key_count) return 1;
		else return -1;
	}

	// sort both maps
	qsort(A, A_key_count, sizeof(Pair),
			(int(*)(const void*, const void*))_key_cmp);
	qsort(B, B_key_count, sizeof(Pair),
			(int(*)(const void*, const void*))_key_cmp);

	// element-wise key comparison
	for(uint i = 0; i < key_count; i++) {
		// if the maps contain different keys, order in favor
		// of the first lexicographically greater key
		order = SIValue_Compare(A[i].key, B[i].key, NULL);
		if(order != 0) return order;
	}

	// element-wise value comparison
	for(uint i = 0; i < key_count; i++) {
		// key lookup succeeded; compare values
		order = SIValue_Compare(A[i].val, B[i].val, disjointOrNull);
		if(disjointOrNull && (*disjointOrNull == COMPARED_NULL ||
							  *disjointOrNull == DISJOINT)) {
			return 0;
		}

		if(order != 0) return order;
	}

	// maps are equal
	return 0;
}

// assumption: a and b are maps
SIValue Map_Merge
(
	const SIValue a,
	const SIValue b
) {
	SIValue result = Map_Clone(a);

	// Merge b into result
	uint bLen = Map_KeyCount(b);
	for(uint i = 0; i < bLen; i++) {
		SIValue key, value;
		Map_GetIdx(b, i, &key, &value);
		Map_Add(&result, key, value);
	}

	return result;
}

// this method referenced by Java ArrayList.hashCode() method, which takes
// into account the hashing of nested values
XXH64_hash_t Map_HashCode
(
	SIValue map
) {
	// sort the map by key, so that {a:1, b:1} and {b:1, a:1}
	// have the same hash value
	uint key_count = Map_KeyCount(map);
	qsort(map.map, key_count, sizeof(Pair),
			(int(*)(const void*, const void*))_key_cmp);

	SIType t = T_MAP;
	XXH64_hash_t hashCode = XXH64(&t, sizeof(t), 0);

	for(uint i = 0; i < key_count; i++) {
		Pair p = map.map[i];
		hashCode = 31 * hashCode + SIValue_HashCode(p.key);
		hashCode = 31 * hashCode + SIValue_HashCode(p.val);
	}

	return hashCode;
}

void Map_ToString
(
	SIValue map,          // map to get string representation from
	char **buf,           // buffer to populate
	size_t *bufferLen,    // size of buffer
	size_t *bytesWritten  // length of string
) {
	ASSERT(SI_TYPE(map) & T_MAP);
	ASSERT(buf != NULL);
	ASSERT(bufferLen != NULL);
	ASSERT(bytesWritten != NULL);

	// resize buffer if buffer length is less than 64
	if(*bufferLen - *bytesWritten < 64) str_ExtendBuffer(buf, bufferLen, 64);

	uint key_count = Map_KeyCount(map);

	// "{" marks the beginning of a map
	*bytesWritten += snprintf(*buf + *bytesWritten, *bufferLen, "{");

	for(uint i = 0; i < key_count; i ++) {
		Pair p = map.map[i];
		// write the next key/value pair
		SIValue_ToString(p.key, buf, bufferLen, bytesWritten);
		if(*bufferLen - *bytesWritten < 64) str_ExtendBuffer(buf, bufferLen, 64);
		*bytesWritten += snprintf(*buf + *bytesWritten, *bufferLen, ": ");
		SIValue_ToString(p.val, buf, bufferLen, bytesWritten);
		// if this is not the last element, add ", "
		if(i != key_count - 1) {
			if(*bufferLen - *bytesWritten < 64) str_ExtendBuffer(buf, bufferLen, 64);
			*bytesWritten += snprintf(*buf + *bytesWritten, *bufferLen, ", ");
		}
	}

	// make sure there's enough space for "}"
	if(*bufferLen - *bytesWritten < 2) str_ExtendBuffer(buf, bufferLen, 2);

	// "}" marks the end of a map
	*bytesWritten += snprintf(*buf + *bytesWritten, *bufferLen, "}");
}

// free map
void Map_Free
(
	SIValue map
) {
	ASSERT(SI_TYPE(map) & T_MAP);

	uint l = Map_KeyCount(map);

	// free stored pairs
	for(uint i = 0; i < l; i++) {
		Pair p = map.map[i];
		Pair_Free(p);
	}

	array_free(map.map);
}

