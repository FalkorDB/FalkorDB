/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "cache_array.h"
#include "rax.h"

/**
 * @brief Key-value cache, uses LRU policy for eviction.
 * Assumes owership over stored objects.
 */
typedef struct Cache {
	uint cap;                          // Cache capacity.
	uint size;                         // Cache current size.
	long long counter;                 // Atomic counter for number of reads.
	rax *lookup;                       // Mapping between keys to entries, for fast lookups.
	CacheEntry *arr;                   // Array of cache elements.
	CacheEntryFreeFunc free_item;      // Callback function that free cached value.
	CacheEntryCopyFunc copy_item;      // Callback function that copies cached value.
	pthread_rwlock_t _cache_rwlock;    // Read-write lock to protect access to the cache.
} Cache;

/**
 * @brief  Initialize a cache.
 * @param  size: Number of entries.
 * @param  freeFUnc: callback for freeing the stored values.
 * @param  copyFunc: callback for copying cached items before returning it to the caller.
 * @retval cache pointer - Initialized empty cache.
 */
Cache *Cache_New(uint size, CacheEntryFreeFunc freeFunc, CacheEntryCopyFunc copyFunc);

/**
 * @brief  Returns a copy of value if it is cached, NULL otherwise.
 * @param  *cache: cache pointer.
 * @param  *key: Key to look for.
 * @retval  pointer with the cached answer, NULL if the key isn't cached.
 */
void *Cache_GetValue(Cache *cache, const char *key);

/**
 * @brief  Stores value under key within the cache.
 * @note   In case the cache is full, this operation causes a cache eviction.
 * @param  *cache: cache pointer.
 * @param  *key: Key for associating with value.
 * @param  *value: pointer with the relevant value.
 */
void Cache_SetValue(Cache *cache, const char *key, void *item);

/**
 * @brief  Stores value under key within the cache, and return a copy of that value.
 * @note   In case the cache is full, this operation causes a cache eviction.
 * @param  *cache: cache pointer.
 * @param  *key: Key for associating with value.
 * @param  *value: pointer with the relevant value.
 * @retval A copy of the given value if a new item was added to the cache, the original value if it was already exist.
 */
void *Cache_SetGetValue(Cache *cache, const char *key, void *value);

/**
 * @brief  Destroys the cache and free all stored items.
 * @param  *cache: cache pointer
 */
void Cache_Free(Cache *cache);

