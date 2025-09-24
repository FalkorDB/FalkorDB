/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "cache.h"
#include "RG.h"
#include "../rmalloc.h"
#include "cache_array.h"
#include <pthread.h>

static CacheEntry *_CacheEvictLRU(Cache *cache) {
	CacheEntry *entry = CacheArray_FindMinLRU(cache->arr, cache->cap);
	// Remove evicted element from the rax.
	raxRemove(cache->lookup, (unsigned  char *)entry->key,
	  strlen(entry->key), NULL);
	CacheArray_CleanEntry(entry, cache->free_item);

	return entry;
}

static bool _Cache_SetValue(Cache *cache, const char *key, void *value,
  		size_t key_len) {
	ASSERT(key != NULL);
	ASSERT(cache != NULL);

	/* in case that another working thread had already inserted the item to the
	 * cache, no need to re-insert it */
	CacheEntry *entry = raxFind(cache->lookup, (unsigned char *)key, key_len);
	if(entry != raxNotFound) {
		return false;
	}

	// key is not in cache! test to see if cache is full?
	if(cache->size == cache->cap) {
		/* the cache is full, evict the least-recently-used element
		 * and reuse its space for the new element */
		entry = _CacheEvictLRU(cache);
	} else {
		// the array has space left in it, use the next available entry
		entry = cache->arr + cache->size++;
	}

	// populate the entry
	char *k = rm_strdup(key);
	cache->counter++;
	CacheArray_PopulateEntry(cache->counter, entry, k, value);


	// Add the new entry to the rax.
	raxInsert(cache->lookup, (unsigned char *)key, key_len, entry, NULL);

	return true;
}

Cache *Cache_New(uint cap, CacheEntryFreeFunc freeFunc, CacheEntryCopyFunc copyFunc) {
	ASSERT(cap > 0);
	ASSERT(copyFunc != NULL);

	Cache *cache     = rm_malloc(sizeof(Cache));
	cache->cap       = cap;
	cache->size      = 0;
	cache->lookup    = raxNew();       // Instantiate key entry mapping.
	cache->counter   = 0;             // Initialize counter to zero.
	cache->copy_item = copyFunc;
	cache->free_item = freeFunc;
	cache->arr = rm_calloc(cap, sizeof(CacheEntry)); // Array of cached values.

	// Initialize the read-write lock to protect access to the cache.
	int res = pthread_rwlock_init(&cache->_cache_rwlock, NULL);
	UNUSED(res);
	ASSERT(res == 0);

	return cache;
}

void *Cache_GetValue(Cache *cache, const char *key) {
	void *item = NULL;

	ASSERT(cache != NULL);

	int res = pthread_rwlock_rdlock(&cache->_cache_rwlock);
	UNUSED(res);
	ASSERT(res == 0);

	size_t key_len = strlen(key);
	CacheEntry *entry = raxFind(cache->lookup, (unsigned char *)key, key_len);

	if(entry == raxNotFound) goto cleanup;

	// element is now the most recently used; update its LRU
	// note that multiple threads can be here simultaneously
	cache->counter++;
	entry->LRU = cache->counter;

	// return a copy of element
	item = cache->copy_item(entry->value);

cleanup:
	res = pthread_rwlock_unlock(&cache->_cache_rwlock);
	ASSERT(res == 0);
	return item;
}

void Cache_SetValue(Cache *cache, const char *key, void *value) {
	ASSERT(key != NULL);
	ASSERT(cache != NULL);

	size_t key_len = strlen(key);

	// Acquire WRITE lock
	int res = pthread_rwlock_wrlock(&cache->_cache_rwlock);
	UNUSED(res);
	ASSERT(res == 0);

	// Insert the value to the cache.
	_Cache_SetValue(cache, key, value, key_len);

	res = pthread_rwlock_unlock(&cache->_cache_rwlock);
	ASSERT(res == 0);
}

void *Cache_SetGetValue(Cache *cache, const char *key, void *value) {
	ASSERT(key != NULL);
	ASSERT(cache != NULL);

	size_t key_len = strlen(key);
	void *value_to_return = value;

	// acquire WRITE lock
	int res = pthread_rwlock_wrlock(&cache->_cache_rwlock);
	UNUSED(res);
	ASSERT(res == 0);

	// return true if value was added, false if value already in cache
	if(_Cache_SetValue(cache, key, value, key_len)) {
		// return a copy of original value
		value_to_return = cache->copy_item(value);
	}

	res = pthread_rwlock_unlock(&cache->_cache_rwlock);
	ASSERT(res == 0);

	return value_to_return;
}

void Cache_Free(Cache *cache) {
	ASSERT(cache != NULL);

	// free cache entries
	for(size_t i = 0; i < cache->size; i++) {
		CacheEntry *entry = cache->arr + i;
		rm_free(entry->key);
		cache->free_item(entry->value);
	}

	rm_free(cache->arr);
	raxFree(cache->lookup);

	int res = pthread_rwlock_destroy(&cache->_cache_rwlock);
	UNUSED(res);
	ASSERT(res == 0);

	rm_free(cache);
}

