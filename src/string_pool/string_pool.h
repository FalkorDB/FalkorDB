/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../util/dict.h"

#ifdef DEBUG_STRINGPOOL
    #define STRINGPOOL_RENT(str)   (StringPool_rent((str), __FILE__, __LINE__))
    #define STRINGPOOL_RETURN(str) (StringPool_return((str), __FILE__, __LINE__))
#else
    #define STRINGPOOL_RENT(str)   (StringPool_rent((str)))
    #define STRINGPOOL_RETURN(str) (StringPool_return((str)))
#endif

// StringPool is a dict of strings
// used primarily to reduce memory consumption by removing string duplication
// the StringPool uses hash(s) for its keys
// while the actual value is a referenced count string

// define StringPool as a dict pointer
typedef struct OpaqueStringPool* StringPool;

// string pool statistics
typedef struct {
	uint64_t n_entries;    // number of entries in pool
	double avg_ref_count;  // average reference count
} StringPoolStats;

// create a new StringPool
StringPool StringPool_create(void);

// add a string to the pool
// incase the string is already stored in the pool
// its reference count is increased
// returns a pointer to the stored string
char *StringPool_rent
(
#ifdef DEBUG_STRINGPOOL
	const char *str,   // string to add
	const char *file,  // source file calling this function
	int line           // source file line calling this function
#else
	const char *str          // string to remove
#endif
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
#ifdef DEBUG_STRINGPOOL
	char *str,         // string to remove
	const char *file,  // source file calling this function
	int line           // source file line calling this function
#else
	char *str          // string to remove
#endif
);

// get string pool statistics
StringPoolStats StringPool_stats
(
	const StringPool pool  // string pool
);

// free pool
void StringPool_free
(
	StringPool *pool  // string pool
);

