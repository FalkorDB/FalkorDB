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
	return strcmp(a->key, b->key);
}

static inline Pair Pair_New
(
	const char *key,
	SIValue val
) {
	return (Pair) {
		.key = rm_strdup(key), .val = SI_CloneValue(val)
	};
}

static void Pair_Free
(
	Pair *p
) {
	rm_free(p->key);
	SIValue_Free(p->val);
}

static int Map_KeyIdx
(
	SIValue map,
	const char *key
) {
	ASSERT(SI_TYPE(map) == T_MAP);
	ASSERT(key          != NULL);

	Map m = map.map;
	uint n = array_len(m);

	// search for key in map
	for(uint i = 0; i < n; i++) {
		Pair *p = m + i;
		if(strcmp(p->key, key) == 0) {
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
// map takes ownership over keys and values elements
// the function will nullify each element within the arrays
SIValue Map_FromArrays
(
	char **keys,      // keys
	SIValue *values,  // values
	uint n            // arrays length
) {
	ASSERT(keys   != NULL);
	ASSERT(values != NULL);

	SIValue map = Map_New(n);

	for(uint i = 0; i < n; i++) {
		Pair p = {
			.key = keys[i],
			.val = values[i]
		};

		array_append(map.map, p);

		keys[i]   = NULL;
		values[i] = SI_NullVal();
	}

	return map;
}

// clone map
SIValue Map_Clone
(
	SIValue map  // map to clone
) {
	ASSERT(SI_TYPE(map) == T_MAP);

	uint key_count = Map_KeyCount(map);
	SIValue clone = Map_New(key_count);

	for(uint i = 0; i < key_count; i++) {
		Pair *p = map.map + i;
		Map_Add(&clone, p->key, p->val);
	}

	return clone;
}

// adds key/value to map
void Map_Add
(
	SIValue *map,     // map to add element to
	const char *key,  // key under which value is added
	SIValue value     // value to add under key
) {
	ASSERT(map           != NULL);
	ASSERT(key           != NULL);
	ASSERT(SI_TYPE(*map) == T_MAP);

	// remove key if already existed
	Map_Remove(*map, key);

	// create a new pair
	// add pair to the end of map
	array_append(map->map, Pair_New(key, value));
}

// adds value under path to map
// returns true is value was added
// false otherwise
//
// example: M[a][b][c] = 8
bool Map_AddPath
(
	SIValue *map,       // map to add element to
	const char **path,  // path under which value is added
	uint8_t n,          // path's length
	SIValue value       // value to add under key
) {
	ASSERT(map  != NULL);
	ASSERT(path != NULL);
	ASSERT(n    > 0)
	ASSERT(SI_TYPE(*map) == T_MAP);

	// add keys along the path
	// e.g. map[a][b][c] = 8
	// making sure 'a' and 'b' are nested maps
	for(uint8_t i = 0; i < n-1; i++) {
		const char *key = path[i];  // current element on path

		// see if key exists
		int idx = Map_KeyIdx(*map, key);

		//----------------------------------------------------------------------
		// missing key
		//----------------------------------------------------------------------

		if(idx == -1) {
			// add nested maps: path[i..i-1]
			for(uint8_t j = i; j < n-1; j++) {
				key = path[j];

				// create a new nested map
				Pair pair = Pair_New(key, SI_NullVal());  // avoid value cloning
				pair.val = Map_New(1);

				// add pair to the end of map
				array_append(map->map, pair);

				// update map
				uint last_key_idx = Map_KeyCount(*map) -1;
				map = &(map->map[last_key_idx].val);
			}

			// entire path had been created
			break;
		}

		//----------------------------------------------------------------------
		// key exists
		//----------------------------------------------------------------------

		// make sure key's value is a map, if it isn't fail
		map = &(map->map[idx].val);
		if(SI_TYPE(*map) != T_MAP) {
			// can't continue following path, reached a none map type
			// e.g. m['a']['b']['c'] = 8
			// where m['a']['b'] is a string
			return false;
		}
	}

	// map contains entire path, add value
	const char *key = path[n-1];
	int idx = Map_KeyIdx(*map, key);

	if(idx != -1) {
		// compare current value to new value, only update if current != new
		SIValue curr = map->map[idx].val;
		if(unlikely(SIValue_Compare(curr, value, NULL) == 0)) {
			// values are the same, do not update
			return false;
		}

		// values are different, perform update
		SIValue_Free(curr);
		map->map[idx].val = SI_CloneValue(value);
	} else {
		// map doesn't contains key, add it
		Pair pair = Pair_New(key, value);
		array_append(map->map, pair);
	}

	return true;
}

// removes key from map
bool Map_Remove
(
	SIValue map,     // map to remove key from
	const char *key  // key to remove
) {
	ASSERT(SI_TYPE(map) == T_MAP);
	ASSERT(key          != NULL);

	Map m = map.map;

	// search for key in map
	int idx = Map_KeyIdx(map, key);

	// key missing from map
	if(idx == -1) return false;

	// override removed key with last pair
	Pair_Free(m + idx);
	array_del_fast(m, idx);

	return true;
}

// removes key from map
// del M['a']['b']['c']
// deletes key 'c' from the path M['a']['b']
bool Map_RemovePath
(
	SIValue map,        // map to remove key from
	const char **path,  // key to remove
	uint8_t n           // path's length
) {
	ASSERT(SI_TYPE(map) == T_MAP);

	Map m = map.map;

	// follow path
	for(uint8_t i = 0; i < n-1; i++) {
		const char *key = path[i];  // current element on path

		// see if key exists
		int idx = Map_KeyIdx(map, key);

		//----------------------------------------------------------------------
		// missing key
		//----------------------------------------------------------------------

		if(idx == -1) {
			// can't proceed on path, nothing to delete
			return false;
		}

		//----------------------------------------------------------------------
		// key exists
		//----------------------------------------------------------------------

		// make sure key's value is a map, if it isn't return
		map = m[idx].val;
		if(SI_TYPE(map) != T_MAP) {
			// can't proceed on path, reached a none map type
			return false;
		}
	}

	// delete key
	return Map_Remove(map, path[n-1]);
}

// clears map
void Map_Clear
(
	SIValue map  // map to clear
) {
	ASSERT(SI_TYPE(map) == T_MAP);

	Map m  = map.map;
	uint n = array_len(m);

	for(uint i = 0; i < n; i++) {
		Pair_Free(m + i);
	}

	array_clear(m);
}

// retrieves value under key, map[key]
// return true and set 'value' if key is in map
// otherwise return false
bool Map_Get
(
	SIValue map,      // map to get value from
	const char *key,  // key to lookup value
	SIValue *value    // [output] value to retrieve
) {
	ASSERT(key          != NULL);
	ASSERT(value        != NULL);
	ASSERT(SI_TYPE(map) == T_MAP);

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

// retrieves value under path
// return true and set 'value' if path is in map
// otherwise return false
bool Map_GetPath
(
	SIValue map,        // map to get value from
	const char **path,  // path to lookup
	uint8_t n,          // path's length
	SIValue *value      // [output] value to retrieve
) {
	ASSERT(n            > 0);
	ASSERT(path         != NULL);
	ASSERT(value        != NULL);
	ASSERT(SI_TYPE(map) == T_MAP);

	//--------------------------------------------------------------------------
	// advance on path
	//--------------------------------------------------------------------------

	// as long as current path element exists and element is a map
	uint8_t i = 0;
	while(i < n-1 && Map_Get(map, path[i], &map) && SI_TYPE(map) == T_MAP) i++;

	// did we reach last path element?
	if(i != n-1) return false;

	return Map_Get(map, path[i], value);
}

// checks if 'key' is in map
bool Map_Contains
(
	SIValue map,     // map to query
	const char *key  // key to look-up
) {
	ASSERT(key          != NULL);
	ASSERT(SI_TYPE(map) == T_MAP);

	return (Map_KeyIdx(map, key) != -1);
}

// check if map contains a key with type 't'
bool Map_ContainsType
(
	SIValue map,  // map to scan
	SIType t      // type to match
) {
	ASSERT(SI_TYPE(map) == T_MAP);

	uint n = Map_KeyCount(map);
	for(uint i = 0; i < n; i++) {
		Pair   *p = map.map + i;
		SIValue v = p->val;

		if(SI_TYPE(v) & t) return true;

		// recursively check nested containers
		if(SI_TYPE(v) == T_ARRAY) {
			if(SIArray_ContainsType(v, t) == true) return true;
		} else if(SI_TYPE(v) == T_MAP) {
			if(Map_ContainsType(v, t) == true) return true;
		}
	}

	return false;
}

// return number of keys in map
uint Map_KeyCount
(
	SIValue map  // map to count number of keys in
) {
	ASSERT(SI_TYPE(map) == T_MAP);
	return array_len(map.map);
}

// return an array of all keys in map
// caller should free returned array rm_free
const char **Map_Keys
(
	SIValue map  // map to extract keys from
) {
	ASSERT(SI_TYPE(map) == T_MAP);

	uint key_count = Map_KeyCount(map);
	const char **keys = rm_malloc(sizeof(char *) * key_count);

	for(uint i = 0; i < key_count; i++) {
		Pair *p = map.map + i;
		keys[i] = p->key;
	}

	return keys;
}

// populate 'key' and 'value' pointers with
// the map contents at the indicated index
void Map_GetIdx
(
	const SIValue map,  // map
	uint idx,           // key position
	const char **key,   // key
	SIValue *value      // map[key]
) {
	ASSERT(idx          < Map_KeyCount(map));
	ASSERT(key          != NULL);
	ASSERT(value        != NULL);
	ASSERT(SI_TYPE(map) == T_MAP);

	Pair *p = map.map + idx;

	*key   = p->key;
	*value = SI_ShareValue(p->val);
}

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
		order = strcmp(A[i].key, B[i].key);
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
		Pair *p = map.map + i;
		SIValue key = SI_ConstStringVal(p->key);

		hashCode = 31 * hashCode + SIValue_HashCode(key);
		hashCode = 31 * hashCode + SIValue_HashCode(p->val);
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
	ASSERT(buf          != NULL);
	ASSERT(bufferLen    != NULL);
	ASSERT(bytesWritten != NULL);
	ASSERT(SI_TYPE(map) == T_MAP);

	// resize buffer if buffer length is less than 64
	if(*bufferLen - *bytesWritten < 64) str_ExtendBuffer(buf, bufferLen, 64);

	uint key_count = Map_KeyCount(map);

	// "{" marks the beginning of a map
	*bytesWritten += snprintf(*buf + *bytesWritten, *bufferLen, "{");

	for(uint i = 0; i < key_count; i ++) {
		Pair *p = map.map + i;
		// write the next key/value pair
		SIValue key = SI_ConstStringVal(p->key);
		SIValue_ToString(key, buf, bufferLen, bytesWritten);
		if(*bufferLen - *bytesWritten < 64) str_ExtendBuffer(buf, bufferLen, 64);
		*bytesWritten += snprintf(*buf + *bytesWritten, *bufferLen, ": ");
		SIValue_ToString(p->val, buf, bufferLen, bytesWritten);
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

// encode map to binary stream
void Map_ToBinary
(
	SerializerIO stream,  // binary stream
	SIValue *map          // map to encode
) {
	ASSERT(stream != NULL);
	ASSERT(SI_TYPE(*map) == T_MAP);

	// write number of keys
	uint32_t n = Map_KeyCount(*map);
	SerializerIO_WriteUnsigned(stream, n);

	// write individual key value pairs
	for(uint32_t i = 0; i < n; i++) {
		Pair *p = map->map + i;

		// write key
		SerializerIO_WriteBuffer(stream, p->key, strlen(p->key));

		// write value
		SIValue_ToBinary(stream, &p->val);
	}
}

// read a map from binary stream
SIValue Map_FromBinary
(
	SerializerIO stream  // binary stream
) {
	// format:
	// key count
	// key:value

	ASSERT(stream != NULL);

	// read number of keys in map
	uint32_t n = SerializerIO_ReadUnsigned(stream);

	SIValue map = Map_New(n);

	for(uint32_t i = 0; i < n; i++) {
		// read string from stream
		char *key = SerializerIO_ReadBuffer(stream, NULL);

		Pair p = {
			.key = key,
			.val = SIValue_FromBinary(stream)
		};

		array_append(map.map, p);
	}

	return map;
}

// free map
void Map_Free
(
	SIValue map
) {
	ASSERT(SI_TYPE(map) == T_MAP);

	uint l = Map_KeyCount(map);

	// free stored pairs
	for(uint i = 0; i < l; i++) Pair_Free(map.map + i);

	array_free(map.map);
}

