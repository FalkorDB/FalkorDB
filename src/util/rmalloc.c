/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "rmalloc.h"

#include "../errors/errors.h"

#ifdef REDIS_MODULE_TARGET /* Set this when compiling your code as a module */

// amount of memory allocated for currently executed query thread_local counter
// it is possible to get into a situation where 'n_alloced' is negative
// this is because we're wrongly assuming that the number of bytes requested for
// an allocation is the actual number of bytes allocated
// it is likely that the allocator allocated more space then required
// in which case when the allocation is freed we will deduct
// actual allocated size from 'n_alloced' which can lead to negative values if
// bytes requested < bytes allocated
static __thread int64_t n_alloced;
static int64_t mem_capacity;  // maximum memory consumption for thread

// function pointers which hold the original address of RedisModule_Alloc*
static void (*RedisModule_Free_Orig)(void *ptr);
static void * (*RedisModule_Alloc_Orig)(size_t bytes);
static char * (*RedisModule_Strdup_Orig)(const char *str);
static void * (*RedisModule_Realloc_Orig)(void *ptr, size_t bytes);
static void * (*RedisModule_Calloc_Orig)(size_t nmemb, size_t size);

void rm_reset_n_alloced() {
	n_alloced = 0;
}

// returns true if a capacity is configured and allocating 'n_bytes' more on
// this thread would exceed it
bool rm_would_exceed_mem_capacity(size_t n_bytes) {
	return mem_capacity > 0 && (n_alloced + (int64_t)n_bytes) > mem_capacity;
}

// removes n_bytes from thread memory consumption
static inline void _nmalloc_decrement(int64_t n_bytes) {
	n_alloced -= n_bytes;
}

// adds nbytes to thread memory consumption
static inline void _nmalloc_increment(int64_t n_bytes) {
	n_alloced += n_bytes;
}

// guards an allocation of 'delta' additional bytes against the capacity
// BEFORE the memory is actually requested from the system allocator
// this is the key difference from a post-allocation check: a single huge
// allocation (e.g. UNWIND range(1, 50000000)) would otherwise be handed to
// the system allocator and OOM-kill the entire process long before any
// post-hoc check could fire
// if the allocation would exceed the capacity we raise a runtime exception,
// which longjmps back to the query's exception handler (set in
// ExecutionPlan_Execute) and unwinds without ever performing the allocation
// if no exception handler is set (outside query execution) the raise is a
// no-op and the caller proceeds with the allocation, preserving prior
// behaviour for non-query contexts and never returning NULL to callers that
// do not expect it
static inline void _nmalloc_guard(int64_t delta) {
	if(n_alloced + delta > mem_capacity) {
		// set n_alloced to MIN to avoid further out of memory exceptions
		// TODO: consider switching to double -inf
		n_alloced = INT32_MIN;

		// abort the query before the oversized allocation is performed
		ErrorCtx_RaiseRuntimeException(EMSG_QUERY_MEM_CONSUMPTION);
	}
}

void *rm_alloc_with_capacity(size_t n_bytes) {
	_nmalloc_guard(n_bytes);
	void *p = RedisModule_Alloc_Orig(n_bytes);
	_nmalloc_increment(n_bytes);
	return p;
}

void *rm_realloc_with_capacity(void *ptr, size_t n_bytes) {
	int64_t old_bytes = RedisModule_MallocSize(ptr);
	// guard on the net growth of the reallocation
	_nmalloc_guard((int64_t)n_bytes - old_bytes);
	// remove bytes of original allocation
	_nmalloc_decrement(old_bytes);
	// track new allocation size
	_nmalloc_increment(n_bytes);
	return RedisModule_Realloc_Orig(ptr, n_bytes);
}

void *rm_calloc_with_capacity(size_t n_elem, size_t size) {
	_nmalloc_guard((int64_t)n_elem * size);
	void *p = RedisModule_Calloc_Orig(n_elem, size);
	_nmalloc_increment(n_elem * size);
	return p;
}

char *rm_strdup_with_capacity(const char *str) {
	char *str_copy = RedisModule_Strdup_Orig(str);
	// use 'RedisModule_MallocSize' instead of strlen as it should be faster
	// in determining allocation size
	_nmalloc_increment(RedisModule_MallocSize(str_copy));
	return str_copy;
}

void rm_free_with_capacity(void *ptr) {
	_nmalloc_decrement(RedisModule_MallocSize(ptr));
	RedisModule_Free_Orig(ptr);
}

void rm_set_mem_capacity(int64_t cap) {
	bool is_capped = (mem_capacity > 0); // current allocator applies memory cap
	bool should_cap = (cap > 0); // should we use a memory capped allocator

	// The local enforced capacity should be set
	// before resetting function pointers
	// for instance if we're switching to capped allocator
	// we want the memory cap to be set
	mem_capacity = cap;
	if(should_cap && !is_capped) {
		// store the function pointer original values and change them
		// to the capped version
		RedisModule_Free_Orig     =  RedisModule_Free;
		RedisModule_Alloc_Orig    =  RedisModule_Alloc;
		RedisModule_Calloc_Orig   =  RedisModule_Calloc;
		RedisModule_Strdup_Orig   =  RedisModule_Strdup;
		RedisModule_Realloc_Orig  =  RedisModule_Realloc;
		RedisModule_Free          =  rm_free_with_capacity;
		RedisModule_Alloc         =  rm_alloc_with_capacity;
		RedisModule_Calloc        =  rm_calloc_with_capacity;
		RedisModule_Strdup        =  rm_strdup_with_capacity;
		RedisModule_Realloc       =  rm_realloc_with_capacity;
	} else if(!should_cap && is_capped) {
		// restore all function pointers to their original values
		RedisModule_Free     =  RedisModule_Free_Orig;
		RedisModule_Alloc    =  RedisModule_Alloc_Orig;
		RedisModule_Calloc   =  RedisModule_Calloc_Orig;
		RedisModule_Strdup   =  RedisModule_Strdup_Orig;
		RedisModule_Realloc  =  RedisModule_Realloc_Orig;
	}
}

#else

void rm_reset_n_alloced() {
}

bool rm_would_exceed_mem_capacity(size_t n_bytes) {
	return false;
}

void rm_set_mem_capacity(int64_t cap) {
}

#endif // REDIS_MODULE_TARGET

/* Redefine the allocator functions to use the malloc family.
 * Only to be used when running module code from a non-Redis
 * context, such as unit tests. */
void Alloc_Reset() {
	RedisModule_Alloc   = malloc;
	RedisModule_Realloc = realloc;
	RedisModule_Calloc  = calloc;
	RedisModule_Free    = free;
	RedisModule_Strdup  = strdup;
}
