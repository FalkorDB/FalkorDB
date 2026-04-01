/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "cache_array.h"
#include "../rmalloc.h"
#include "../../RG.h"

CacheEntry *CacheArray_FindMinLRU(CacheEntry *cache_arr, uint cap) {
	ASSERT(cache_arr != NULL);

	CacheEntry *min_LRU_entry = cache_arr;
	long long min_LRU = atomic_load(&min_LRU_entry->LRU);

	for(size_t i = 1; i < cap; i++) {
		CacheEntry *current_entry = cache_arr + i;
		long long current_LRU = atomic_load(&current_entry->LRU);
		if(current_LRU < min_LRU) {
			min_LRU_entry = current_entry;
			min_LRU = current_LRU;
		}
	}

	return min_LRU_entry;
}

CacheEntry *CacheArray_PopulateEntry(long long counter, CacheEntry *entry, char *key,
  									void *value) {

	entry->key   = key;
	entry->value = value;
	atomic_store(&entry->LRU, counter);

	return entry;
}

void CacheArray_CleanEntry(CacheEntry *entry, CacheEntryFreeFunc free_entry) {
	ASSERT(entry != NULL);
	ASSERT(free_entry != NULL);

	if(entry->key != NULL) {
		rm_free(entry->key);
		entry->key = NULL;
	}

	if(entry->value != NULL) {
		free_entry(entry->value);
		entry->value = NULL;
	}

	atomic_store(&entry->LRU, 0);
}

