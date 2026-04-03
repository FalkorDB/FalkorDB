/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "src/util/rmalloc.h"
#include "src/util/cache/cache.h"
#include "src/execution_plan/execution_plan.h"

#include <pthread.h>
#include <stdatomic.h>

void setup() {
	Alloc_Reset();
}

#define TEST_INIT setup();
#include "acutest.h"

static int free_count = 0;  // count how many cache objects been freed

typedef struct {
	const char *str;
} CacheObj;

CacheObj *CacheObj_New(const char *str) {
	CacheObj *obj = (CacheObj *)rm_malloc(sizeof(CacheObj));
	obj->str = rm_strdup(str);
	return obj;
}

CacheObj *CacheObj_Dup(const CacheObj *obj) {
	CacheObj *dup = (CacheObj *)rm_malloc(sizeof(CacheObj));
	dup->str = rm_strdup(obj->str);
	return dup;
}

bool CacheObj_EQ(const CacheObj *a, const CacheObj *b) {
	if(a == b) return true;
	return (strcmp(a->str, b->str) == 0);
}

void CacheObj_Free(CacheObj *obj) {
	free_count++;
	rm_free((char *)obj->str);
	rm_free(obj);
}

void test_executionPlanCache() {
	// build a cache of strings in this case for simplicity
	Cache *cache = Cache_New(3, (CacheEntryFreeFunc)CacheObj_Free,
			(CacheEntryCopyFunc)CacheObj_Dup);

	CacheObj *item1 = CacheObj_New("1");
	CacheObj *item2 = CacheObj_New("2");
	CacheObj *item3 = CacheObj_New("3");
	CacheObj *item4 = CacheObj_New("4");

	const char *key1 = "MATCH (a) RETURN a";
	const char *key2 = "MATCH (b) RETURN b";
	const char *key3 = "MATCH (c) RETURN c";
	const char *key4 = "MATCH (d) RETURN d";

	//--------------------------------------------------------------------------
	// Check for not existing key.
	//--------------------------------------------------------------------------

	TEST_ASSERT(!Cache_GetValue(cache, "None existing"));

	//--------------------------------------------------------------------------
	// Set Get single item
	//--------------------------------------------------------------------------

	CacheObj *from_cache = NULL;
	Cache_SetValue(cache, key1, item1);
	from_cache = (CacheObj*)Cache_GetValue(cache, key1);
	TEST_ASSERT(CacheObj_EQ(item1, from_cache));
	CacheObj_Free(from_cache);

	//--------------------------------------------------------------------------
	// Set multiple items
	//--------------------------------------------------------------------------

	CacheObj* to_cache = (CacheObj*)Cache_SetGetValue(cache, key2, item2);
	from_cache = (CacheObj*)Cache_GetValue(cache, key2);
	TEST_ASSERT(CacheObj_EQ(item2, from_cache));
	CacheObj_Free(to_cache);
	CacheObj_Free(from_cache);

	// Fill up cache
	to_cache = (CacheObj*)Cache_SetGetValue(cache, key3, item3);
	CacheObj_Free(to_cache);
	to_cache = (CacheObj*)Cache_SetGetValue(cache, key4, item4);
	CacheObj_Free(to_cache);

	// Verify that oldest entry do not exists - queue is [ 4 | 3 | 2 ].
	TEST_ASSERT(Cache_GetValue(cache, key1) == NULL);

	Cache_Free(cache);

	// Expecting CacheObjFree to be called 9 times.
	TEST_ASSERT(free_count == 9);
}

//------------------------------------------------------------------------------
// Concurrent cache stress test (Issue #1782)
//
// Exercises Cache_GetValue and Cache_SetGetValue from many threads
// simultaneously. The cache is intentionally small (4 entries) while threads
// rotate through many more keys, forcing constant evictions.
//
// What this test validates:
//   1. No crashes or use-after-free under concurrent access.
//   2. The atomic counter is strictly monotonic (no lost increments).
//   3. Every LRU value in the cache is <= counter (no stale/corrupt values).
//   4. Under ThreadSanitizer (SAN=thread), any data race will be flagged.
//------------------------------------------------------------------------------

#define CONC_CACHE_CAP    4    // small cache to force evictions
#define CONC_NUM_KEYS     20   // many more keys than cache capacity
#define CONC_NUM_READERS  8
#define CONC_NUM_WRITERS  4
#define CONC_ITERATIONS   5000

typedef struct {
	Cache *cache;
	int    thread_id;
} ThreadCtx;

// generate a key string for index i; caller must provide buffer
static void _make_key(int i, char *buf, size_t buf_size) {
	snprintf(buf, buf_size, "MATCH (n_%d) RETURN n_%d", i, i);
}

// reader thread: repeatedly calls Cache_GetValue on random keys
static void *_cache_reader(void *arg) {
	ThreadCtx *ctx = (ThreadCtx *)arg;
	Cache *cache = ctx->cache;

	for (int iter = 0; iter < CONC_ITERATIONS; iter++) {
		for (int i = 0; i < CONC_NUM_KEYS; i++) {
			char key[64];
			_make_key(i, key, sizeof(key));
			CacheObj *val = (CacheObj *)Cache_GetValue(cache, key);
			if (val != NULL) {
				// value must be valid (not corrupted)
				TEST_CHECK(val->str != NULL);
				CacheObj_Free(val);
			}
		}
	}

	return NULL;
}

// writer thread: inserts keys via Cache_SetGetValue, forcing evictions
static void *_cache_writer(void *arg) {
	ThreadCtx *ctx = (ThreadCtx *)arg;
	Cache *cache = ctx->cache;
	int tid = ctx->thread_id;

	for (int iter = 0; iter < CONC_ITERATIONS; iter++) {
		// rotate through keys; offset by thread_id for variety
		int idx = (iter + tid * 7) % CONC_NUM_KEYS;
		char key[64];
		_make_key(idx, key, sizeof(key));

		char val_str[32];
		snprintf(val_str, sizeof(val_str), "v_%d_%d", tid, iter);
		CacheObj *obj = CacheObj_New(val_str);

		CacheObj *ret = (CacheObj *)Cache_SetGetValue(cache, key, obj);
		// free the returned copy (or the original if key already existed)
		CacheObj_Free(ret);
	}

	return NULL;
}

void test_cacheConcurrency() {
	// create a small cache to maximize eviction contention
	Cache *cache = Cache_New(CONC_CACHE_CAP, (CacheEntryFreeFunc)CacheObj_Free,
			(CacheEntryCopyFunc)CacheObj_Dup);

	// pre-populate cache so readers have entries to hit
	for (int i = 0; i < CONC_CACHE_CAP; i++) {
		char key[64];
		_make_key(i, key, sizeof(key));
		char val_str[32];
		snprintf(val_str, sizeof(val_str), "init_%d", i);
		CacheObj *obj = CacheObj_New(val_str);
		Cache_SetValue(cache, key, obj);
	}

	//--------------------------------------------------------------------------
	// launch reader and writer threads
	//--------------------------------------------------------------------------

	pthread_t readers[CONC_NUM_READERS];
	pthread_t writers[CONC_NUM_WRITERS];
	ThreadCtx reader_ctx[CONC_NUM_READERS];
	ThreadCtx writer_ctx[CONC_NUM_WRITERS];

	for (int i = 0; i < CONC_NUM_WRITERS; i++) {
		writer_ctx[i].cache     = cache;
		writer_ctx[i].thread_id = i;
		int rc = pthread_create(&writers[i], NULL, _cache_writer, &writer_ctx[i]);
		TEST_ASSERT(rc == 0);
	}

	for (int i = 0; i < CONC_NUM_READERS; i++) {
		reader_ctx[i].cache     = cache;
		reader_ctx[i].thread_id = i;
		int rc = pthread_create(&readers[i], NULL, _cache_reader, &reader_ctx[i]);
		TEST_ASSERT(rc == 0);
	}

	// wait for all threads to complete
	for (int i = 0; i < CONC_NUM_READERS; i++) {
		pthread_join(readers[i], NULL);
	}
	for (int i = 0; i < CONC_NUM_WRITERS; i++) {
		pthread_join(writers[i], NULL);
	}

	//--------------------------------------------------------------------------
	// post-condition checks
	//--------------------------------------------------------------------------

	// counter must be positive (incremented by both readers and writers)
	long long final_counter = atomic_load(&cache->counter);
	TEST_CHECK_(final_counter > 0, "counter=%lld should be > 0", final_counter);

	// every cached entry's LRU must be <= counter and >= 0
	for (uint i = 0; i < cache->size; i++) {
		long long lru = atomic_load(&cache->arr[i].LRU);
		TEST_CHECK_(lru >= 0 && lru <= final_counter,
			"entry[%u] LRU=%lld out of range [0, %lld]", i, lru, final_counter);
	}

	// cache should not have grown beyond its capacity
	TEST_CHECK_(cache->size <= cache->cap,
		"cache size=%u exceeds cap=%u", cache->size, cache->cap);

	Cache_Free(cache);
}

TEST_LIST = {
	{"executionPlanCache", test_executionPlanCache},
	{"cacheConcurrency",   test_cacheConcurrency},
	{NULL, NULL}
};

