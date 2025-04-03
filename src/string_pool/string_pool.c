/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "string_pool.h"
#include "deps/xxHash/xxhash.h"

#include <string.h>
#include <pthread.h>

pthread_key_t _tlsStringPool; // thread local storage string-pool access flag

// StringPool structure
struct OpaqueStringPool {
	dict *ht;                  // hashtable
	uint64_t total_ref_count;  // number of references
};

// key hash function
static uint64_t hashFunc
(
	const void *key
) {
	const char *str = (const char *)key;
	return XXH3_64bits(key, strlen(key));
}

// key compare function
int keyCompare
(
	dict *d,
	const void *key1,
	const void *key2
) {
	return (strcmp((const char*)key1, (const char*)key2) == 0);
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

// grant access to string-pool via TLS key
// if a thread has this key set, access to the string pool is granted
// otherwise the TLS key is NULL and access is denied
void StringPool_grantAccessViaTLS
(
	void* unused
) {
	ASSERT(unused == NULL);

	int res = pthread_setspecific(_tlsStringPool, (const void*)1);
	ASSERT(res == 0);
}

// create a new StringPool
StringPool StringPool_create(void) {
	// create thread local storage string-pool access flag
	int res = pthread_key_create(&_tlsStringPool, NULL);
	ASSERT(res == 0);

	StringPool pool = rm_malloc(sizeof(struct OpaqueStringPool));

	pool->ht = HashTableCreate(&_type);
	pool->total_ref_count = 0;

	return pool;
}

// add a string to the pool
// incase the string is already stored in the pool
// its reference count is increased
// returns a pointer to the stored string
char *StringPool_rent
(
	StringPool pool,  // string pool
	const char *str   // string to add
) {
	// validate arguments
	ASSERT(str  != NULL);	
	ASSERT(pool != NULL);

	char *ret;            // returned string
	dictEntry *existing;  // existing dict entry

	dict *ht = pool->ht;
	dictEntry *de = HashTableAddRaw(ht, (void*)str, &existing);
	if(de != NULL) {
		// new string
		HashTableSetKey(ht, de, rm_strdup(str));
	} else {
		de = existing;
	}

	// increase string reference count
	uint32_t *count = (uint32_t*)HashTableEntryMetadata(de);
	*count = *count + 1;

	// increase total reference count
	pool->total_ref_count++;

	ret = (char*)HashTableGetKey(de);
	return ret;
}

// add string to pool in case it doesn't already exists
// the string isn't cloned
char *StringPool_rentNoClone
(
	StringPool pool,  // string pool
	char *str         // string to add
) {
	// validate arguments
	ASSERT(str  != NULL);	
	ASSERT(pool != NULL);

	char *ret;            // returned string
	dictEntry *existing;  // existing dict entry

	dict *ht = pool->ht;
	dictEntry *de = HashTableAddRaw(ht, (void*)str, &existing);
	if(de == NULL) {
		de = existing;
	}

	// increase string reference count
	uint32_t *count = (uint32_t*) HashTableEntryMetadata(de);
	*count = *count + 1;

	// increase total reference count
	pool->total_ref_count++;

	ret = (char*)HashTableGetKey(de);
	return ret;
}

// remove string from pool
// the string will be free only when its reference count drops to 0
// returns true if the string was freed
void StringPool_return
(
	StringPool pool,  // string pool
	char *str         // string to remove
) {
	// validate arguments
	ASSERT(str  != NULL);	
	ASSERT(pool != NULL);

	dict *ht = pool->ht;
	dictEntry *de = HashTableFind(ht, str);

	if(unlikely(de == NULL)) {
		// str is missing from pool
		return;
	}

	// get reference count
	uint32_t *count = (uint32_t*)HashTableEntryMetadata(de);

	// decrease reference count
	*count = *count -1;

	// decrease total reference count
	pool->total_ref_count--;

	// free entry if reference count reached 0
	if(unlikely(*count == 0)) {
		int res = HashTableDelete(ht, (const void *)str);
		ASSERT(res == DICT_OK);
	}
}

// get string pool statistics
StringPoolStats StringPool_stats
(
	const StringPool pool  // string pool
) {
	StringPoolStats stats;  // statistics object to populate

	uint64_t n_entries   = 0;
	double avg_ref_count = 0;

	if(pool != NULL) {
		n_entries = HashTableElemCount(pool->ht);

		if(n_entries > 0) {
			avg_ref_count = (double)pool->total_ref_count / n_entries;
		}
	}

	stats.n_entries     = n_entries;
	stats.avg_ref_count = avg_ref_count;

	return stats;
}

// free pool
void StringPool_free
(
	StringPool *pool  // string pool
) {
	ASSERT(pool != NULL && *pool != NULL);

	StringPool p = *pool;

	// free hashtable
	HashTableRelease(p->ht);

	rm_free(*pool);

	*pool = NULL;
}

