/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "src/util/rmalloc.h"
#include "tests/utils/mock_log.h"
#include "src/util/cache/cache.h"
#include "src/execution_plan/execution_plan.h"

void setup() {
	Alloc_Reset();
	Logging_Reset();
}

#define TEST_INIT setup();
#include "acutest.h"

static int free_count = 0;  // count how many cache objects been freed

typedef struct {
	const char *str;
} CacheObj;

CacheObj *CacheObj_New(const char *str) {
	CacheObj *obj = (CacheObj *)rm_malloc(sizeof(CacheObj));
	obj->str = str;
	return obj;
}

CacheObj *CacheObj_Dup(const CacheObj *obj) {
	CacheObj *dup = (CacheObj *)rm_malloc(sizeof(CacheObj));
	memcpy(dup, obj, sizeof(CacheObj));
	return dup;
}

bool CacheObj_EQ(const CacheObj *a, const CacheObj *b) {
	if(a == b) return true;
	return (strcmp(a->str, b->str) == 0);
}

void CacheObj_Free(CacheObj *obj) {
	free_count++;
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

TEST_LIST = {
	{"executionPlanCache", test_executionPlanCache},
	{NULL, NULL}
};

