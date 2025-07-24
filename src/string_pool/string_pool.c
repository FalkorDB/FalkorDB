/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../globals.h"
#include "string_pool.h"
#include "deps/xxHash/xxhash.h"

#include <string.h>
#include <pthread.h>

//------------------------------------------------------------------------------
// internal structures & definitions
//------------------------------------------------------------------------------

struct OpaqueStringPool {
	dict *ht;                  // hash table mapping string -> reference count
	pthread_rwlock_t rwlock;   // read-write lock to protect hash table
	uint64_t total_ref_count;  // total number of active references
};

// hashtable callbacks

// key hash function
static uint64_t _hashFunc
(
	const void *key
) {
	const char *str = (const char *)key;
	return XXH3_64bits(key, strlen(key));
}

// key compare function
int _keyCompare
(
	dict *d,
	const void *key1,
	const void *key2
) {
	return strcmp((const char*)key1, (const char*)key2) == 0;
}

// key free function
static void _keyDestructor
(
	dict *d,
	void *key
) {
	rm_free(key);
}

// key metadata byte size
static size_t _metadataBytes
(
	dict *d
) {
	return sizeof(uint32_t);
}

// StringPool hashtable callbacks
static const dictType _dictType = {
	.hashFunction           = _hashFunc,
	.keyDup                 = NULL,
	.valDup                 = NULL,
	.keyCompare             = _keyCompare,
	.keyDestructor          = _keyDestructor,
	.valDestructor          = NULL,
	.expandAllowed          = NULL,
	.dictEntryMetadataBytes = _metadataBytes,
	.dictMetadataBytes      = NULL,
	.afterReplaceEntry      = NULL
};

//------------------------------------------------------------------------------
// public API
//------------------------------------------------------------------------------

// create a new StringPool
StringPool StringPool_create(void) {
	StringPool pool = rm_malloc(sizeof(struct OpaqueStringPool));
	ASSERT(pool != NULL);

	pool->ht = HashTableCreate(&_dictType);
	pool->total_ref_count = 0;

	// initialize read-write lock with writer preference if supported

	pthread_rwlockattr_t attr;
	int res = pthread_rwlockattr_init(&attr);
	ASSERT(res == 0);

#ifdef PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP
	int pref = PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP;
	res = pthread_rwlockattr_setkind_np(&attr, pref);
	ASSERT(res == 0);
#endif

	res = pthread_rwlock_init(&pool->rwlock, &attr);
	ASSERT(res == 0);

	res = pthread_rwlockattr_destroy(&attr);
	ASSERT(res == 0);

	return pool;
}

// add a string to the pool
// incase the string is already stored in the pool
// its reference count is increased
// returns a pointer to the stored string
#ifdef DEBUG_STRINGPOOL
char *StringPool_rent
(
	const char *str,   // string to add
	const char *file,  // source file calling this function
	int line           // source file line calling this function
)
#else
char *StringPool_rent
(
	const char *str
)
#endif
{
	// validate arguments
	ASSERT(str != NULL);

	StringPool pool = Globals_Get_StringPool();
	ASSERT(pool != NULL);

	#ifdef DEBUG_STRINGPOOL
	RedisModule_Log(NULL, "debug", "StringPool_rent: \"%s\" from %s:%d\n",
			str, file, line);
	#endif

	dictEntry *de;
	dict *ht = pool->ht;

	// first, try to find the string under a read lock
	pthread_rwlock_rdlock(&pool->rwlock);

	de = HashTableFind(ht, (void*)str);

	if(de != NULL) {
        // found existing entry - increment reference count atomically
        uint32_t *count = (uint32_t*)HashTableEntryMetadata(de);
        __atomic_fetch_add(count, 1, __ATOMIC_RELAXED);
        __atomic_fetch_add(&pool->total_ref_count, 1, __ATOMIC_RELAXED);

        pthread_rwlock_unlock(&pool->rwlock);

        char *stored_str = (char*)HashTableGetKey(de);
        return stored_str;
    }
    pthread_rwlock_unlock(&pool->rwlock);

	// entry not found, insert under write lock
	pthread_rwlock_wrlock(&pool->rwlock);

	// double-check: another thread might have inserted while we waited
	dictEntry *existing = NULL;  // existing dict entry
	de = HashTableAddRaw(ht, (void*)str, &existing);

	uint32_t *count = NULL;
	if(de != NULL) {
		// new entry: duplicate key and insert
		HashTableSetKey(ht, de, rm_strdup(str));
		// set initial ref-count to 1
		count = (uint32_t*)HashTableEntryMetadata(de);
		*count = 1;
	} else {
		// another thread inserted it before us
		de = existing;
		count = (uint32_t*)HashTableEntryMetadata(de);
		__atomic_fetch_add(count, 1, __ATOMIC_RELAXED);
	}

	// release write lock
	pthread_rwlock_unlock(&pool->rwlock);

	// update total reference count
	__atomic_fetch_add(&pool->total_ref_count, 1, __ATOMIC_RELAXED);

	char *stored_str = (char*)HashTableGetKey(de);
	return stored_str;
}

// remove string from pool
// the string will be free only when its reference count drops to 0
// returns true if the string was freed
#ifdef DEBUG_STRINGPOOL
void StringPool_return
(
	char *str,         // string to remove
	const char *file,  // source file calling this function
	int line           // source file line calling this function
)
#else
void StringPool_return
(
	char *str          // string to remove
)
#endif
{
	// validate arguments
	ASSERT(str != NULL);

	StringPool pool = Globals_Get_StringPool();
	ASSERT(pool != NULL);

	#ifdef DEBUG_STRINGPOOL
	RedisModule_Log(NULL, "debug", "StringPool_return: \"%s\" from %s:%d\n",
			str, file, line);
	#endif

	dictEntry *de;
	dict *ht = pool->ht;

	// access entry under read lock
    pthread_rwlock_rdlock(&pool->rwlock);

	de = HashTableFind(ht, str);
	ASSERT(de != NULL);

    pthread_rwlock_unlock(&pool->rwlock);

	// decrease reference count
	uint32_t *count = (uint32_t*)HashTableEntryMetadata(de);
	uint32_t old_count = __atomic_fetch_sub(count, 1, __ATOMIC_RELAXED);

	// if this was the last reference, delete the entry
	if(old_count == 1) {
		// acquire write lock for potential deletion
		pthread_rwlock_wrlock(&pool->rwlock);

		// re-check under write lock in case another thread modified the count
		de = HashTableFind(ht, str);
		if(de != NULL) {
			count = (uint32_t*)HashTableEntryMetadata(de);
            if(*count == 0) {
                // safe to delete
                int res = HashTableDelete(ht, (const void *)str);
                ASSERT(res == DICT_OK);
            }
			// if count != 0, another thread incremented it, so don't delete
		}
		// if de == NULL, another thread already deleted it

		// release write lock
		pthread_rwlock_unlock(&pool->rwlock);
	}

	// update total reference count
	__atomic_fetch_sub(&pool->total_ref_count, 1, __ATOMIC_RELAXED);
}

// get string pool statistics
StringPoolStats StringPool_stats
(
	const StringPool pool  // string pool
) {
	ASSERT(pool != NULL);

	uint64_t n_entries     = 0;
	uint64_t total_refs    = 0;
	double   avg_ref_count = 0;

	// get hash table entry count under read lock
	pthread_rwlock_rdlock(&pool->rwlock);
	n_entries = HashTableElemCount(pool->ht);
	pthread_rwlock_unlock(&pool->rwlock);

	if(n_entries > 0) {
		// read total_ref_count atomically to avoid stale values
		total_refs = __atomic_load_n(&pool->total_ref_count, __ATOMIC_RELAXED);
        avg_ref_count = (double)total_refs / n_entries;
    }

	StringPoolStats stats = {0};  // statistics object to populate
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

	int res = pthread_rwlock_destroy(&p->rwlock);
	ASSERT(res == 0);

	rm_free(*pool);
	*pool = NULL;
}

