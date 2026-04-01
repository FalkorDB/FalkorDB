/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/types.h>

//------------------------------------------------------------------------------
// function pointers
//------------------------------------------------------------------------------

// cache entry free function
typedef void (*CacheEntryFreeFunc)(void *);

// cache entry duplicate function
typedef void *(*CacheEntryCopyFunc)(void *);

/**
 * @brief  A struct for an entry in cache array with a key and value.
 */
typedef struct CacheEntry_t {
	char *key;      // Entry key.
	void *value;    // Entry stored value.
	long long LRU;  // Indicates the time when the entry was last recently used.
} CacheEntry;


// Returns a pointer to the entry in the cache array with the minimum LRU.
CacheEntry *CacheArray_FindMinLRU(CacheEntry *cache_arr, uint cap);

// Assign new values to the fields of a cache entry.
CacheEntry *CacheArray_PopulateEntry(long long counter, CacheEntry *entry, char *key,
  			void *value);

// Free the fields of a cache entry to prepare it for reuse.
void CacheArray_CleanEntry(CacheEntry *entry, CacheEntryFreeFunc free_entry);

