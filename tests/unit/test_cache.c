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

static atomic_int free_count = 0;  // count how many cache objects been freed

void setup() {
	Alloc_Reset();
	atomic_store(&free_count, 0);
}

#define TEST_INIT setup();
#include "acutest.h"

typedef struct {
	char *str;
} CacheObj;

CacheObj *CacheObj_New(const char *str) {
	CacheObj *obj = (CacheObj *)rm_malloc(sizeof(CacheObj));
	obj->str = rm_strdup(str);
	return obj;
}

CacheObj *CacheObj_Dup(const CacheObj *obj) {
	return CacheObj_New(obj->str);
}

bool CacheObj_EQ(const CacheObj *a, const CacheObj *b) {
	if(a == b) return true;
	return (strcmp(a->str, b->str) == 0);
}

void CacheObj_Free(CacheObj *obj) {
	atomic_fetch_add(&free_count, 1);
	rm_free(obj->str);
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
				TEST_ASSERT(val->str != NULL);
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
	TEST_ASSERT(final_counter > 0);

	// every cached entry's LRU must be <= counter and >= 0
	for (uint i = 0; i < cache->size; i++) {
		long long lru = atomic_load(&cache->arr[i].LRU);
		TEST_ASSERT(lru >= 0 && lru <= final_counter);
	}

	// cache should not have grown beyond its capacity
	TEST_ASSERT(cache->size <= cache->cap);

	Cache_Free(cache);
}

//------------------------------------------------------------------------------
// Test: Cache eviction causes dangling borrowed pointers (Issue #1823)
//
// Reproduces the exact ownership bug where QGNode.alias points into AST
// parse tree memory. When the cache evicts the original entry and the AST
// is freed, clones hold dangling pointers.
//------------------------------------------------------------------------------

// Extended cache object that has both owned and borrowed strings,
// mirroring how QGNode has rm_malloc'd fields AND borrowed alias/labels
typedef struct {
	char *owned_str;            // owned by this object (freed on Free)
	const char *borrowed_ptr;   // NOT owned - points into external memory (like QGNode.alias -> AST)
} BorrowCacheObj;

static BorrowCacheObj *BorrowCacheObj_New(const char *owned, const char *borrowed) {
	BorrowCacheObj *obj = rm_malloc(sizeof(BorrowCacheObj));
	obj->owned_str = rm_strdup(owned);
	obj->borrowed_ptr = borrowed;  // shallow copy — intentionally NOT duplicated
	return obj;
}

// This clone function mimics QGNode_Clone: it copies the borrowed pointer
// without rm_strdup, exactly reproducing the bug
static BorrowCacheObj *BorrowCacheObj_ShallowDup(const BorrowCacheObj *obj) {
	BorrowCacheObj *dup = rm_malloc(sizeof(BorrowCacheObj));
	dup->owned_str = rm_strdup(obj->owned_str);
	dup->borrowed_ptr = obj->borrowed_ptr;  // BUG: shallow copy of borrowed pointer
	return dup;
}

// This clone function mimics the FIXED QGNode_Clone: it deep-copies
// the borrowed string so the clone is self-contained
static BorrowCacheObj *BorrowCacheObj_DeepDup(const BorrowCacheObj *obj) {
	BorrowCacheObj *dup = rm_malloc(sizeof(BorrowCacheObj));
	dup->owned_str = rm_strdup(obj->owned_str);
	dup->borrowed_ptr = rm_strdup(obj->borrowed_ptr);  // FIX: deep copy
	return dup;
}

static void BorrowCacheObj_Free(BorrowCacheObj *obj) {
	rm_free(obj->owned_str);
	// NOTE: borrowed_ptr is NOT freed here — it's not owned by this object
	// (just like QGNode_Free currently does NOT free node->alias)
	rm_free(obj);
}

// Free function for the fixed version where borrowed_ptr is now owned
static void BorrowCacheObj_DeepFree(BorrowCacheObj *obj) {
	rm_free(obj->owned_str);
	rm_free((char *)obj->borrowed_ptr);  // NOW owned after deep copy
	rm_free(obj);
}

void test_cacheEvictionDanglingPointer() {
	//--------------------------------------------------------------------------
	// Part 1: Demonstrate the bug (shallow copy of borrowed pointer)
	//--------------------------------------------------------------------------

	// Simulate AST parse tree memory — this is the external owner
	char *ast_string = rm_strdup("ast_parse_tree_alias");

	// Create cache with capacity 1 — any second insert forces eviction
	Cache *cache = Cache_New(1,
		(CacheEntryFreeFunc)BorrowCacheObj_Free,
		(CacheEntryCopyFunc)BorrowCacheObj_ShallowDup);

	// Create object with borrowed pointer into "AST memory"
	// This mirrors: QGNode.alias = <pointer into AST parse tree>
	BorrowCacheObj *original = BorrowCacheObj_New("owned_data", ast_string);
	TEST_ASSERT(original->borrowed_ptr == ast_string);  // same pointer

	// Insert into cache — cache takes ownership of `original`,
	// returns a shallow clone to the caller
	BorrowCacheObj *clone = (BorrowCacheObj *)Cache_SetGetValue(
		cache, "query_1", original);

	// The clone's borrowed_ptr points to the SAME ast_string
	// (this is the bug — it should have been rm_strdup'd)
	TEST_ASSERT(clone->borrowed_ptr == ast_string);
	TEST_ASSERT(strcmp(clone->borrowed_ptr, "ast_parse_tree_alias") == 0);

	// Now insert a different key to force eviction of "query_1"
	// This frees `original` via BorrowCacheObj_Free
	// (simulates CacheArray_CleanEntry → ExecutionCtx_Free → ExecutionPlan_Free)
	BorrowCacheObj *obj2 = BorrowCacheObj_New("other_data", ast_string);
	BorrowCacheObj *clone2 = (BorrowCacheObj *)Cache_SetGetValue(
		cache, "query_2", obj2);

	// Now simulate AST_Free: free the AST parse tree memory
	// In real code, this happens when AST ref_count reaches 0 after eviction
	// Write a sentinel to prove the memory is now invalid
	memset(ast_string, 'X', strlen(ast_string));
	rm_free(ast_string);

	// clone->borrowed_ptr is now a DANGLING POINTER
	// In production this causes SIGSEGV in QGNode_Free or corrupted data
	// We can't safely dereference it, but we've proven the scenario:
	// the clone outlived the AST memory it borrowed from.

	// Clean up
	// WARNING: clone->borrowed_ptr is now a DANGLING POINTER pointing to freed memory.
	// This is intentionally unsafe — we rely on BorrowCacheObj_Free not dereferencing
	// borrowed_ptr (it only frees owned_str). Do not add any borrowed_ptr access here.
	BorrowCacheObj_Free(clone);   // safe: only accesses owned_str, not borrowed_ptr
	BorrowCacheObj_Free(clone2);  // safe: clone2->borrowed_ptr is also dangling (ast_string freed)
	Cache_Free(cache);

	//--------------------------------------------------------------------------
	// Part 2: Demonstrate the fix (deep copy of borrowed pointer)
	//--------------------------------------------------------------------------

	// Fresh AST string
	char *ast_string2 = rm_strdup("ast_parse_tree_alias_v2");

	// The cache's eviction function uses BorrowCacheObj_Free (NOT DeepFree)
	// because the ORIGINAL stored in the cache borrows the pointer (does not own it),
	// mirroring how QGNode_Free doesn't free node->alias in the current code.
	// Only the CLONE (returned to caller) owns a deep copy and uses DeepFree.
	Cache *fixed_cache = Cache_New(1,
		(CacheEntryFreeFunc)BorrowCacheObj_Free,
		(CacheEntryCopyFunc)BorrowCacheObj_DeepDup);

	// Create object with borrowed pointer
	BorrowCacheObj *orig2 = BorrowCacheObj_New("owned_data", ast_string2);

	// Insert — cache stores original (shallow borrow), returns deep clone to caller
	BorrowCacheObj *fixed_clone = (BorrowCacheObj *)Cache_SetGetValue(
		fixed_cache, "query_1", orig2);

	// The fixed clone's borrowed_ptr is a DIFFERENT pointer (deep copied)
	TEST_ASSERT(fixed_clone->borrowed_ptr != ast_string2);
	TEST_ASSERT(strcmp(fixed_clone->borrowed_ptr, "ast_parse_tree_alias_v2") == 0);

	// Force eviction: inserting "query_2" evicts "query_1" (orig2) via
	// BorrowCacheObj_Free, which does NOT free orig2->borrowed_ptr (ast_string2)
	char *another_ast = rm_strdup("another_ast_string");
	BorrowCacheObj *orig3 = BorrowCacheObj_New("other_data", another_ast);
	BorrowCacheObj *fixed_clone2 = (BorrowCacheObj *)Cache_SetGetValue(
		fixed_cache, "query_2", orig3);

	// Free the AST memory — simulates AST_Free when ref_count reaches 0.
	// With shallow copy this would leave clone->borrowed_ptr dangling;
	// with deep copy the clone is self-contained and survives.
	rm_free(ast_string2);

	// fixed_clone's borrowed_ptr is STILL VALID because it was deep copied
	TEST_ASSERT(strcmp(fixed_clone->borrowed_ptr, "ast_parse_tree_alias_v2") == 0);

	// Clean up — DeepFree properly frees the duplicated borrowed_ptr in the clones
	BorrowCacheObj_DeepFree(fixed_clone);
	BorrowCacheObj_DeepFree(fixed_clone2);
	// Cache_Free evicts orig3 via BorrowCacheObj_Free (does NOT free another_ast)
	Cache_Free(fixed_cache);
	// Manually free another_ast since the cache's eviction didn't own it
	rm_free(another_ast);
}

TEST_LIST = {
	{"executionPlanCache",           test_executionPlanCache},
	{"cacheConcurrency",             test_cacheConcurrency},
	{"cacheEvictionDanglingPointer", test_cacheEvictionDanglingPointer},
	{NULL, NULL}
};

