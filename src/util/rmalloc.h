#ifndef __REDISEARCH_ALLOC__
#define __REDISEARCH_ALLOC__

#include <stdlib.h>
#include <string.h>
#include "../redismodule.h"

#ifdef REDIS_MODULE_TARGET /* Set this when compiling your code as a module */

// called when mem_capacity configuration changes
// note that this function might be called during query execution
//
// depending on the current and new memory-limit value (limited vs unlimited)
// and the currently used allocator (capped vs none capped)
// the allocator function pointers might be updated
void rm_set_mem_capacity(int64_t cap);

// reset thread memory consumption counter to 0 (no memory consumed)
void rm_reset_n_alloced();

static inline void *rm_malloc(size_t n) {
	return RedisModule_Alloc(n);
}
static inline void *rm_calloc(size_t nelem, size_t elemsz) {
	return RedisModule_Calloc(nelem, elemsz);
}
static inline void *rm_realloc(void *p, size_t n) {
	return RedisModule_Realloc(p, n);
}
static inline void *rm_aligned_malloc(size_t alignment, size_t size) {
	return rm_malloc(size);
}
static inline void rm_free(void *p) {
	RedisModule_Free(p);
}
static inline char *rm_strdup(const char *s) {
	return RedisModule_Strdup(s);
}
static inline char *rm_strndup(const char *s, size_t n) {
	char *ret = (char *)rm_malloc(n + 1);

	if(ret) {
		ret[n] = '\0';
		memcpy(ret, s, n);
	}
	return ret;
}

#endif
#ifndef REDIS_MODULE_TARGET
/* for non redis module targets */
#define rm_malloc malloc
#define rm_free free
#define rm_calloc calloc
#define rm_realloc realloc
#define rm_strdup strdup
#define rm_strndup strndup
#endif

#define rm_new(x) rm_malloc(sizeof(x))

/* Revert the allocator patches so that
 * the stdlib malloc functions will be used
 * for use when executing code from non-Redis
 * contexts like unit tests. */
void Alloc_Reset(void);

#endif

