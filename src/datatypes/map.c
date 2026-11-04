/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "map.h"
#include "array.h"
#include "../util/arr.h"
#include "errors/errors.h"
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

// sorts pairs by key
// a map whose keys are already in order is left untouched: equality tests
// run once per join candidate on values cached by the hash join, and must
// neither re-sort nor write to them
static void _sort_pairs
(
	Pair *pairs,
	uint n
) {
	for(uint i = 1; i < n; i++) {
		if(_key_cmp(pairs + i - 1, pairs + i) > 0) {
			qsort(pairs, n, sizeof(Pair),
					(int(*)(const void*, const void*))_key_cmp);
			return;
		}
	}
}

static inline Pair Pair_New
(
	SIValue key,
	SIValue val
) {
	ASSERT(SI_TYPE(key) & T_STRING);
	ASSERT(SI_ALLOCATION(&key) != M_VOLATILE);
	ASSERT(SI_ALLOCATION(&val) != M_VOLATILE);

	return (Pair) { .key = key, .val = val };
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

	Map  m = map.map;
	uint n = arr_len(m);

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

static int Map_KeyIdxCaseInsensitive
(
	SIValue map,
	SIValue key
) {
	ASSERT(SI_TYPE(map) & T_MAP);
	ASSERT(SI_TYPE(key) & T_STRING);

	Map  m = map.map;
	uint n = arr_len(m);

	// search for key in map
	for(uint i = 0; i < n; i++) {
		Pair pair = m[i];
		if(strcasecmp(pair.key.stringval, key.stringval) == 0) {
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

	map.map        = arr_new(Pair, capacity);
	map.type       = T_MAP;
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
		arr_append(map.map, Pair_New(SI_CloneValue(keys[i]),
					SI_CloneValue(values[i])));
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
	SIValue clone  = Map_New(key_count);

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
	ASSERT(SI_TYPE(key)  & T_STRING);

	// remove key if already existed
	Map_Remove(*map, key);

	// create a new pair
	Pair pair = Pair_New(SI_CloneValue(key), SI_CloneValue(value));

	// add pair to the end of map
	arr_append(map->map, pair);
}

// adds key/value to map
// both key and value aren't cloned
void Map_AddNoClone
(
	SIValue *map,  // map to add element to
	SIValue key,   // key under which value is added
	SIValue value  // value to add under key
) {
	ASSERT(SI_TYPE(*map) & T_MAP);
	ASSERT(SI_TYPE(key)  & T_STRING);

	// remove key if already existed
	Map_Remove(*map, key);

	// create a new pair
	Pair pair = Pair_New(key, value);

	// add pair to the end of map
	arr_append(map->map, pair);
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
	arr_del_fast(m, idx);
}

// clears map
void Map_Clear
(
	SIValue map  // map to clear
) {
	ASSERT(SI_TYPE(map) & T_MAP);

	Map m  = map.map;
	uint n = arr_len(m);

	for(uint i = 0; i < n; i++) {
		Pair_Free(m[i]);
	}

	arr_clear(m);
}

// retrieves value under key, map[key]
// return true and set 'value' if key is in map
// otherwise return false
bool Map_Get
(
	SIValue map,    // map to get value from
	SIValue key,    // key to lookup value
	SIValue *value  // [output] value to retrieve
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

// retrieves value under lower(key), map[lower(key)]
// sets 'value' to NULL if key isn't in map
bool Map_GetCaseInsensitive
(
	SIValue map,    // map
	SIValue key,    // key to access
	SIValue *value  // [output] map[lower(key)]
) {
	ASSERT(SI_TYPE(map) & T_MAP);
	ASSERT(SI_TYPE(key) & T_STRING);
	ASSERT(value != NULL);

	int idx = Map_KeyIdxCaseInsensitive(map, key);

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
	ASSERT (key   != NULL) ;
	ASSERT (value != NULL) ;
	ASSERT (SI_TYPE (map) & T_MAP) ;
	ASSERT (idx < Map_KeyCount (map)) ;

	Pair p = map.map [idx] ;

	*key   = SI_ShareValue (p.key) ;
	*value = SI_ShareValue (p.val) ;
}

// creates a map from its binary representation
// this is the reverse of the map-writing side of SIValue_ToBinary /
// EffectsBuffer_WriteSIValue, mirroring SIArray_FromBinary
SIValue Map_FromBinary
(
	FILE *stream  // stream containing binary representation of a map
) {
	// read number of keys
	uint32_t n ;
	fread_assert (&n, sizeof (uint32_t), stream) ;

	SIValue map = Map_New (n) ;

	for (uint32_t i = 0; i < n; i++) {
		SIValue key = SIValue_FromBinary (stream) ;
		SIValue val = SIValue_FromBinary (stream) ;
		Map_AddNoClone (&map, key, val) ;
	}

	return map ;
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
	return arr_len(map.map);
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
	_sort_pairs(A, A_key_count);
	_sort_pairs(B, B_key_count);

	// element-wise key comparison
	for(uint i = 0; i < key_count; i++) {
		// if the maps contain different keys, order in favor
		// of the first lexicographically greater key
		order = SIValue_Compare(A[i].key, B[i].key, NULL);
		if(order != 0) return order;
	}

	// element-wise value comparison
	//
	// the returned order is the first non-zero value comparison
	// it must not depend on whether the caller asked for 'disjointOrNull':
	// sort and binary search pass NULL while relational operators pass a
	// pointer and both must agree on the order of the same two maps
	//
	// 'disjointOrNull' is taken from the first pair of values which is either
	// not equal or compared as null, NaN or disjoint types, so '<' follows the
	// scalar rules for that pair: null or disjoint types give null and NaN
	// gives false
	// this pair does not always decide the returned order, a leading null pair
	// sets the flag and a later pair can still decide the order
	// equality is not decided here, see Map_Equals
	bool decided = false;  // the flag was taken from a pair

	if(disjointOrNull) *disjointOrNull = 0;

	for(uint i = 0; i < key_count; i++) {
		// key lookup succeeded; compare values
		int flag = 0;
		order = SIValue_Compare(A[i].val, B[i].val, &flag);

		if(!decided && (order != 0 || flag != 0)) {
			decided = true;
			if(disjointOrNull) *disjointOrNull = flag;
		}

		if(order != 0) break;
	}

	return order;
}

SIValueEquality Map_Equals
(
	SIValue mapA,
	SIValue mapB
) {
	Map   A          =  mapA.map;
	Map   B          =  mapB.map;
	uint  key_count  =  Map_KeyCount(mapA);

	if(key_count != Map_KeyCount(mapB)) return SI_EQ_FALSE;

	// sort both maps
	_sort_pairs(A, key_count);
	_sort_pairs(B, key_count);

	// maps with different keys are not equal
	for(uint i = 0; i < key_count; i++) {
		if(SIValue_Compare(A[i].key, B[i].key, NULL) != 0) return SI_EQ_FALSE;
	}

	// an unequal pair of values decides, wherever it is
	// otherwise a null pair makes the result unknown
	bool compared_null = false;
	for(uint i = 0; i < key_count; i++) {
		SIValueEquality eq = SIValue_Equals(A[i].val, B[i].val);
		if(eq == SI_EQ_FALSE) return SI_EQ_FALSE;
		if(eq == SI_EQ_NULL)  compared_null = true;
	}

	return compared_null ? SI_EQ_NULL : SI_EQ_TRUE;
}

// merge two maps
// in case of key collision, the value from 'b' is used
SIValue Map_Merge
(
	const SIValue a,
	const SIValue b
) {
	// in case both operands aren't maps
	if(! (SI_TYPE(a) & T_MAP && SI_TYPE(b) & T_MAP)) {
		// raise an error
		ErrorCtx_RaiseRuntimeException(EMSG_MERGE_MAP_ERROR);
		return SI_NullVal();
	}

	SIValue result = Map_Clone(a);

	// merge b into result
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
	_sort_pairs(map.map, key_count);

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

// defrag map
// returns true if map memory been relocated
bool Map_Defrag
(
	SIValue *map,              // map to defrag
	RedisModuleDefragCtx *ctx  // redis defrag context
) {
	ASSERT (map != NULL) ;
	ASSERT (ctx != NULL) ;

	// pointer to map's elements
	void *p     = arr_hdr (map->map) ;  // array header points to allocation
	void *moved = RedisModule_DefragAlloc (ctx, p) ;

	// update map if p been relocated
	if (moved != NULL) {
		// assign back array's buffer
		map->map = (struct Pair *) (((arr_hdr_t *)moved)->buf) ;
		return true ;
	}

	return false ;
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

	arr_free(map.map);
}

