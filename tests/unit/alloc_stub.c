/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

/* Minimal stub for test_indexer_regression which does not link falkordb.
 * Alloc_Reset() redirects the rm_* allocator shims to stdlib — required
 * by arr.h / rmalloc.h when running outside a Redis module context. */
#include <stdlib.h>
#include <string.h>

/* These are the function-pointer globals declared in rmalloc.h. */
void *(*RedisModule_Alloc)  (size_t n)              = malloc;
void *(*RedisModule_Realloc)(void *p, size_t n)     = realloc;
void *(*RedisModule_Calloc) (size_t nm, size_t sz)  = calloc;
void  (*RedisModule_Free)   (void *p)               = free;
char *(*RedisModule_Strdup) (const char *s)         = strdup;

void Alloc_Reset(void) {
	RedisModule_Alloc   = malloc;
	RedisModule_Realloc = realloc;
	RedisModule_Calloc  = calloc;
	RedisModule_Free    = free;
	RedisModule_Strdup  = strdup;
}
