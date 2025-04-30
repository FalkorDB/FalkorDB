/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "string_pool.h"
#include "util/rmalloc.h"
#include "deps/xxHash/xxhash.h"

#include <string.h>
#include <pthread.h>

// an entry within the string pool
// composed of the actual string and a reference counter
typedef struct {
	char *key;       // string key
	uint32_t count;  // reference count
} InternString;

pthread_key_t _tlsStringPool; // thread local storage string-pool access flag

// key compare function
int keyCompare
(
	const void *key1,
	const void *key2,
	void *udata
) {
	return strcmp(((InternString *)key1)->key, ((InternString *)key2)->key);
}

// key free function
static void keyDestructor
(
	void *obj
) {
	rm_free(((InternString *)obj)->key);
}

// grant access to string-pool via TLS key
// if a thread has this key set, access to the string pool is granted
// otherwise the TLS key is NULL and access is denied
void StringPool_grantAccessViaTLS(void* unused) {
	ASSERT(unused == NULL);

	int res = pthread_setspecific(_tlsStringPool, (const void*)1);
	ASSERT(res == 0);
}

// create a new StringPool
StringPool StringPool_create(void) {
	// create thread local storage string-pool access flag
	int res = pthread_key_create(&_tlsStringPool, NULL);
	ASSERT(res == 0);

	return hashmap_new_with_redis_allocator(sizeof(InternString), 0, 0, 0, NULL,
			keyCompare, keyDestructor, NULL);
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
	
	uint64_t hash = XXH3_64bits(str, strlen(str));
	InternString istr = {.key = (char *)str, .count = 1};

	InternString *existing =
		(InternString *)hashmap_get_with_hash(pool, &istr, hash);

	if(existing == NULL) {
		istr.key = rm_strdup(str);
		hashmap_set_with_hash(pool, (void*)&istr, hash);
		return istr.key;
	}

	// increase string reference count
	existing->count += 1;

	return existing->key;
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

	uint64_t hash = XXH3_64bits(str, strlen(str));
	InternString istr = {.key = str, .count = 1};
	InternString *existing;  // existing dict entry

	existing = (InternString *)hashmap_set_with_hash(pool, (void*)&istr, hash);
	if(existing == NULL) return str;

	// increase string reference count
	existing->count += 1;

	return existing->key;
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

	uint64_t hash = XXH3_64bits(str, strlen(str));
	InternString istr = {.key = str, .count = 1};
	InternString *existing =
		(InternString *)hashmap_get_with_hash(pool, &istr, hash);

	if(unlikely(existing == NULL)) {
		// str is missing from pool
		return;
	}

	// decrease reference count
	existing->count -= 1;

	// free entry if reference count reached 0
	if(unlikely(existing->count == 0)) {
		InternString *res =
			(InternString *)hashmap_delete_with_hash(pool, &istr, hash);

		rm_free(res->key);
		ASSERT(res != NULL);
	}
}

// free pool
void StringPool_free
(
	StringPool *pool  // string pool
) {
	ASSERT(pool != NULL && *pool != NULL);

	StringPool p = *pool;

	// free hashtable
	hashmap_free(p);

	*pool = NULL;
}

