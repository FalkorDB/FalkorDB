/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "string_pool.h"
#include "deps/xxHash/xxhash.h"

#include <stdatomic.h>

// key hash function
static uint64_t hashFunc
(
	const void *key
) {
	const char *str = (const char *)key;
	return XXH64(key, strlen(key), 0);
}

// key compare function
int keyCompare
(
	dict *d,
	const void *key1,
	const void *key2
) {
	return (strcmp(key1, key2) == 0);
}

// key free function
static void keyDestructor
(
	dict *d,
	void *obj
) {
	rm_free(obj);
}

// key metadata byte size
static size_t metadataBytes
(
	dict *d
) {
	return sizeof(uint32_t);
}

// StringPool hashtable callbacks
static const dictType _type = {
    hashFunc,
	NULL,
    NULL,
    keyCompare,
    keyDestructor,
    NULL,
    NULL,
    metadataBytes,
    NULL,
    NULL
};

// create a new StringPool
StringPool StringPool_create(void) {
	StringPool pool = HashTableCreate(&_type);
	return pool;
}

// add a string to the pool
// incase the string is already stored in the pool
// its reference count is increased
// returns a pointer to the stored string
char *StringPool_add
(
	StringPool pool,  // string pool
	const char *str   // string to add
) {
	// validate arguments
	ASSERT(str  != NULL);	
	ASSERT(pool != NULL);

	char *ret;
	dictEntry *existing;
	dictEntry *de = HashTableAddRaw(pool, (void*)str, &existing);

	if(de != NULL) {
		// new string
		HashTableSetKey(pool, de, rm_strdup(str));
	} else {
		// string already in pool
		de = existing;
	}

	// increase string reference count atomically
	atomic_uint *count = (atomic_uint*) HashTableEntryMetadata(de);
	atomic_fetch_add_explicit(count, 1, memory_order_relaxed);

	ret = (char*)HashTableGetKey(de);
	return ret;
}

// add string to pool in case it doesn't already exists
// the string isn't cloned
char *StringPoll_addNoClone
(
	StringPool pool,  // string pool
	char *str         // string to add
) {
	// validate arguments
	ASSERT(str  != NULL);	
	ASSERT(pool != NULL);

	char *ret;
	dictEntry *existing;
	dictEntry *de = HashTableAddRaw(pool, (void*)str, &existing);

	if(de == NULL) {
		// str already in pool
		de = existing;
	}

	// increase string reference count atomically
	atomic_uint *count = (atomic_uint*) HashTableEntryMetadata(de);
	atomic_fetch_add_explicit(count, 1, memory_order_relaxed);

	ret = (char*)HashTableGetKey(de);
	return ret;
}

// remove string from pool
// the string will be free only when its reference count drops to 0
// returns true if the string was freed
void StringPool_remove
(
	StringPool pool,  // string pool
	char *str         // string to remove
) {
	// validate arguments
	ASSERT(str  != NULL);	
	ASSERT(pool != NULL);

	dictEntry *de = HashTableFind(pool, (void*)str);
	if(unlikely(de == NULL)) {
		return;
	}

	// get reference count atomically
	atomic_uint *count = (atomic_uint*)HashTableEntryMetadata(de);

	// decrease reference count atomically
	uint32_t prev_count = atomic_fetch_sub_explicit(count, 1,
			memory_order_acquire);

	// free entry if reference count reached 0
	if(unlikely(prev_count == 1)) {
		int res = HashTableDelete(pool, (const void *)str);
		ASSERT(res == DICT_OK);
	}
}

// free pool
void StringPool_free
(
	StringPool *pool  // string pool
) {
	ASSERT(pool);

	HashTableRelease(*pool);
	*pool = NULL;
}

