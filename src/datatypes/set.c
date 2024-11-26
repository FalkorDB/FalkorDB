/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "set.h"

// fake hash function
// hash of key is simply key
static uint64_t _id_hash
(
	const void *key
) {
	return (uint64_t)key;
}

static dictType _dt = {_id_hash, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};

set *Set_New(void) {
	return HashTableCreate(&_dt);
}

bool Set_Contains
(
	set *s,
	SIValue v
) {
	unsigned long long const hash = SIValue_HashCode(v);
	return HashTableFind(s, (void*)hash) != NULL;
}

// adds v to set
bool Set_Add
(
	set *s,
	SIValue v
) {
	unsigned long long const hash = SIValue_HashCode(v);
	return HashTableAdd(s, (void*)hash, NULL) == DICT_OK;
}

// removes v from set
void Set_Remove
(
	set *s,
	SIValue v
) {
	unsigned long long const hash = SIValue_HashCode(v);
	HashTableDelete(s, (void*)hash);
}

// Return number of elements in set
uint64_t Set_Size
(
	set *s
) {
	return HashTableElemCount(s);
}

// free set
void Set_Free
(
	set *s
) {
	HashTableRelease(s);
}

