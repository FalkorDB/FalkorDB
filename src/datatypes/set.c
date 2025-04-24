/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "set.h"
#include "../util/rmalloc.h"

static uint64_t _id_hash
(
	const void *item,
	uint64_t seed0,
	uint64_t seed1
) {
	return *(uint64_t *)item;
}

set *Set_New(void) {
	return hashmap_new_with_allocator(rm_malloc, rm_realloc, rm_free, sizeof(uint64_t), 0, 0, 0, _id_hash, NULL, NULL, NULL);
}

bool Set_Contains
(
	set *s,
	SIValue v
) {
	unsigned long long const hash = SIValue_HashCode(v);
	return hashmap_get(s, (void*)&hash) != NULL;
}

// adds v to set
bool Set_Add
(
	set *s,
	SIValue v
) {
	unsigned long long const hash = SIValue_HashCode(v);
	return hashmap_set(s, (void*)&hash) == NULL;
}

// removes v from set
void Set_Remove
(
	set *s,
	SIValue v
) {
	unsigned long long const hash = SIValue_HashCode(v);
	hashmap_delete(s, (void*)&hash);
}

// Return number of elements in set
uint64_t Set_Size
(
	set *s
) {
	return hashmap_count(s);
}

// free set
void Set_Free
(
	set *s
) {
	hashmap_free(s);
}

