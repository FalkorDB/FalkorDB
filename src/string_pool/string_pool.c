/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "string_pool.h"
#include "deps/xxHash/xxhash.h"

#include <string.h>
#include <pthread.h>

// StringPool structure
struct OpaqueStringPool {
	dict *ht;                  // hashtable
	pthread_rwlock_t rwlock;   // read-write lock scoped to this specific graph
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

// create a new StringPool
StringPool StringPool_create(void) {
	StringPool pool = rm_malloc(sizeof(struct OpaqueStringPool));

	pool->ht = HashTableCreate(&_type);
	pool->total_ref_count = 0;

	// create a read write lock which favors writes
	// specify prefer write in lock creation attributes
	int res = 0;

	pthread_rwlockattr_t attr;
	res = pthread_rwlockattr_init(&attr);
	ASSERT(res == 0);

#if !defined(__APPLE__) && !defined(__FreeBSD__)
	int pref = PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP;
	res = pthread_rwlockattr_setkind_np(&attr, pref);
	ASSERT(res == 0);
#endif

	res = pthread_rwlock_init(&pool->rwlock, &attr);
	ASSERT(res == 0) ;

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
	uint32_t *count;
	dictEntry *de;
	dict *ht = pool->ht;
	dictEntry *existing;  // existing dict entry

	// first attempt: try to find existing entry with read lock
    pthread_rwlock_rdlock(&pool->rwlock);

	de = HashTableFind(ht, (void*)str);

	if(de != NULL) {
        // found existing entry - increment reference count atomically
        uint32_t *count = (uint32_t*)HashTableEntryMetadata(de);
        __atomic_fetch_add(count, 1, __ATOMIC_SEQ_CST);
        __atomic_fetch_add(&pool->total_ref_count, 1, __ATOMIC_SEQ_CST);

        pthread_rwlock_unlock(&pool->rwlock);
        ret = (char*)HashTableGetKey(de);
        return ret;
    }
    pthread_rwlock_unlock(&pool->rwlock);

	// entry not found - need write lock for insertion
	pthread_rwlock_wrlock(&pool->rwlock);

	// double-check: another thread might have inserted while we waited
	de = HashTableAddRaw(ht, (void*)str, &existing);

	if(de != NULL) {
		// new string
		HashTableSetKey(ht, de, rm_strdup(str));
	} else {
		de = existing;
	}

	count = (uint32_t*)HashTableEntryMetadata(de);
	__atomic_fetch_add(count, 1, __ATOMIC_SEQ_CST);

	// update total reference count
	__atomic_fetch_add(&pool->total_ref_count, 1, __ATOMIC_SEQ_CST);

	// release write lock
	pthread_rwlock_unlock(&pool->rwlock);

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

	dictEntry *de;
	dict *ht = pool->ht;
    uint32_t old_count;

	// hold read lock while accessing the entry
    pthread_rwlock_rdlock(&pool->rwlock);

	de = HashTableFind(ht, str);
	ASSERT(de != NULL);

    pthread_rwlock_unlock(&pool->rwlock);

	// decrease reference count
	uint32_t *count = (uint32_t*)HashTableEntryMetadata(de);
	old_count = __atomic_fetch_sub(count, 1, __ATOMIC_SEQ_CST);

	if(old_count == 1) {
		// acquire write lock for potential deletion
		pthread_rwlock_wrlock(&pool->rwlock);

		// double-check: find entry again and verify count is still 0
		de = HashTableFind(ht, str);
		if(de != NULL) {
			count = (uint32_t*)HashTableEntryMetadata(de);
            if(*count == 0) {
                // Safe to delete
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
	__atomic_fetch_sub(&pool->total_ref_count, 1, __ATOMIC_SEQ_CST);
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

	int res = pthread_rwlock_destroy(&p->rwlock);
	ASSERT(res == 0);

	rm_free(*pool);

	*pool = NULL;
}

