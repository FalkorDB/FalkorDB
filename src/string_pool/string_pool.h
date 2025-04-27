/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../util/hashmap.h"

// StringPool is a dict of strings
// used primarily to reduce memory consumption by removing string duplication
// the StringPool uses hash(s) for its keys
// while the actual value is a referenced count string

// define StringPool as a dict pointer
typedef hashmap StringPool;

// grant access to string-pool via TLS key
// if a thread has this key set, access to the string pool is granted
// otherwise the TLS key is NULL and access is denied
void StringPool_grantAccessViaTLS(void* unused);

// create a new StringPool
StringPool StringPool_create(void);

// add a string to the pool
// incase the string is already stored in the pool
// its reference count is increased
// returns a pointer to the stored string
char *StringPool_rent
(
	StringPool pool,  // string pool
	const char *str   // string to add
);

// add string to pool in case it doesn't already exists
// the string isn't cloned
char *StringPool_rentNoClone
(
	StringPool pool,  // string pool
	char *str         // string to add
);

// remove string from pool
// the string will be free only when its reference count drops to 0
void StringPool_return
(
	StringPool pool,  // string pool
	char *str         // string to remove
);

// free pool
void StringPool_free
(
	StringPool *pool  // string pool
);

