/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../util/dict.h"

// StringPool is a dict of strings
// used primarly to reduce memory consumption by removing string duplication
//
// the StringPool uses hash(s) for its keys
// while the actual value is a referenced count string

typedef dict* StringPool;

// create a new StringPool
StringPool StringPool_create(void);

// add a string to the pool
// incase the string is already stored in the pool
// its reference count is increased
// returns a pointer to the stored string
char *StringPool_add
(
	StringPool pool,  // string pool
	const char *str   // string to add
);

// add string to pool in case it doesn't already exists
// the string isn't cloned
char *StringPoll_addNoClone
(
	StringPool pool,  // string pool
	char *str         // string to add
);

// remove string from pool
// the string will be free only when its reference count drops to 0
void StringPool_remove
(
	StringPool pool,  // string pool
	char *str         // string to remove
);

// free pool
void StringPool_free
(
	StringPool *pool  // string pool
);

