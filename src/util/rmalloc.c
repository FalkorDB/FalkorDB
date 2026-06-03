/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "rmalloc.h"

#include "../errors/errors.h"
#include <stdatomic.h>

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
static bool track_alloced_bytes;
static bool allocator_patched;
static atomic_llong current_alloced_bytes;

// function pointers which hold the original address of RedisModule_Alloc*
static void (*RedisModule_Free_Orig)(void *ptr);
static void * (*RedisModule_Alloc_Orig)(size_t bytes);
static char * (*RedisModule_Strdup_Orig)(const char *str);
static void * (*RedisModule_Realloc_Orig)(void *ptr, size_t bytes);
static void * (*RedisModule_Calloc_Orig)(size_t nmemb, size_t size);

void rm_reset_n_alloced() {
	n_alloced = 0;
}

static inline void _nmalloc_apply_delta(int64_t delta) {
	n_alloced += delta;

	if(track_alloced_bytes) {
		atomic_fetch_add_explicit(&current_alloced_bytes, delta,
				memory_order_relaxed);
	}

	// check if capacity exceeded
	if(delta > 0 && mem_capacity > 0 && n_alloced > mem_capacity) {
		// set n_alloced to MIN to avoid further out of memory exceptions
		// TODO: consider switching to double -inf
		n_alloced = INT32_MIN;

		// throw exception cause memory limit exceeded
		ErrorCtx_SetError(EMSG_QUERY_MEM_CONSUMPTION);
	}
}

void *rm_alloc_with_capacity(size_t n_bytes) {
	void *p = RedisModule_Alloc_Orig(n_bytes);
	if(p != NULL) {
		_nmalloc_apply_delta((int64_t)RedisModule_MallocSize(p));
	}
	return p;
}

void *rm_realloc_with_capacity(void *ptr, size_t n_bytes) {
	const int64_t old_size = (ptr != NULL) ? (int64_t)RedisModule_MallocSize(ptr) : 0;
	void *new_ptr = RedisModule_Realloc_Orig(ptr, n_bytes);
	if(new_ptr == NULL) {
		return NULL;
	}

	const int64_t new_size = (int64_t)RedisModule_MallocSize(new_ptr);
	_nmalloc_apply_delta(new_size - old_size);
	return new_ptr;
}

void *rm_calloc_with_capacity(size_t n_elem, size_t size) {
	void *p = RedisModule_Calloc_Orig(n_elem, size);
	if(p != NULL) {
		_nmalloc_apply_delta((int64_t)RedisModule_MallocSize(p));
	}
	return p;
}

char *rm_strdup_with_capacity(const char *str) {
	char *str_copy = RedisModule_Strdup_Orig(str);
	if(str_copy != NULL) {
		// use 'RedisModule_MallocSize' instead of strlen as it should be faster
		// in determining allocation size
		_nmalloc_apply_delta((int64_t)RedisModule_MallocSize(str_copy));
	}
	return str_copy;
}

void rm_free_with_capacity(void *ptr) {
	if(ptr != NULL) {
		_nmalloc_apply_delta(-(int64_t)RedisModule_MallocSize(ptr));
	}
	RedisModule_Free_Orig(ptr);
}

static inline void _rm_update_allocator_hooks(void) {
	bool should_patch = (mem_capacity > 0 || track_alloced_bytes);

	if(should_patch && !allocator_patched) {
		// store the function pointer original values and change them
		// to the wrapped version
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
		allocator_patched = true;
	} else if(!should_patch && allocator_patched) {
		// restore all function pointers to their original values
		RedisModule_Free     =  RedisModule_Free_Orig;
		RedisModule_Alloc    =  RedisModule_Alloc_Orig;
		RedisModule_Calloc   =  RedisModule_Calloc_Orig;
		RedisModule_Strdup   =  RedisModule_Strdup_Orig;
		RedisModule_Realloc  =  RedisModule_Realloc_Orig;
		allocator_patched = false;
	}
}

void rm_set_mem_capacity(int64_t cap) {
	// The local enforced capacity should be set
	// before resetting function pointers
	// for instance if we're switching to wrapped allocator
	// we want the memory cap to be set
	mem_capacity = cap;
	_rm_update_allocator_hooks();
}

void rm_tracking_init(void) {
	track_alloced_bytes = true;
	atomic_store_explicit(&current_alloced_bytes, 0, memory_order_relaxed);
	_rm_update_allocator_hooks();
}

int64_t rm_get_current_alloced_bytes(void) {
	return atomic_load_explicit(&current_alloced_bytes, memory_order_relaxed);
}

#else

void rm_reset_n_alloced() {
}

void rm_set_mem_capacity(int64_t cap) {
}

void rm_tracking_init(void) {
}

int64_t rm_get_current_alloced_bytes(void) {
	return 0;
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
