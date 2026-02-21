/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#ifndef __MMAP_ALLOC__
#define __MMAP_ALLOC__

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// Default chunk size for mmap allocator (100MB)
#define MMAP_DEFAULT_CHUNK_SIZE (100 * 1024 * 1024)

// Initialize the mmap allocator
// chunk_size: size of each memory chunk to allocate (0 uses default)
// Returns true on success, false on failure
bool mmap_alloc_init(size_t chunk_size);

// Cleanup and free all mmap allocator resources
void mmap_alloc_cleanup(void);

// Allocate memory using mmap-backed allocator
// For allocations < chunk_size: allocates from current chunk
// For allocations >= chunk_size: uses dedicated mmap
void *mmap_alloc(size_t size);

// Reallocate memory
void *mmap_realloc(void *ptr, size_t size);

// Free memory allocated by mmap_alloc
void mmap_free(void *ptr);

// Get the actual allocated size of a pointer
size_t mmap_malloc_size(void *ptr);

// Get statistics about mmap allocator usage
typedef struct {
	size_t chunk_size;           // configured chunk size
	size_t chunks_allocated;     // number of chunks allocated
	size_t large_allocs;         // number of large (>chunk_size) allocations
	size_t total_memory;         // total memory allocated
	size_t memory_in_use;        // memory currently in use
} mmap_stats_t;

void mmap_get_stats(mmap_stats_t *stats);

#endif // __MMAP_ALLOC__
