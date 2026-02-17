/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "mmap_alloc.h"
#include <sys/mman.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdatomic.h>

// Allocation metadata stored before each allocation
typedef struct {
	size_t size;           // requested size
	size_t actual_size;    // actual allocated size (including metadata)
	bool is_large;         // true if this is a large (dedicated mmap) allocation
	void *chunk;           // pointer to the chunk this allocation belongs to (NULL for large allocs)
} alloc_header_t;

// Memory chunk structure
typedef struct chunk {
	void *memory;          // mmap'd memory region
	size_t size;           // total size of chunk
	size_t used;           // bytes used in this chunk
	struct chunk *next;    // next chunk in the list
} chunk_t;

// Global allocator state
static struct {
	size_t chunk_size;
	chunk_t *chunks;
	pthread_mutex_t lock;
	atomic_size_t chunks_allocated;
	atomic_size_t large_allocs;
	atomic_size_t total_memory;
	atomic_size_t memory_in_use;
	bool initialized;
} g_allocator = {0};

// Align size to 16 bytes for proper alignment
static inline size_t align_size(size_t size) {
	return (size + 15) & ~15;
}

// Allocate a new chunk
static chunk_t *allocate_chunk(size_t size) {
	chunk_t *chunk = (chunk_t *)malloc(sizeof(chunk_t));
	if (!chunk) {
		return NULL;
	}

	void *memory = mmap(NULL, size, PROT_READ | PROT_WRITE,
	                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (memory == MAP_FAILED) {
		free(chunk);
		return NULL;
	}

	chunk->memory = memory;
	chunk->size = size;
	chunk->used = 0;
	chunk->next = NULL;

	atomic_fetch_add(&g_allocator.chunks_allocated, 1);
	atomic_fetch_add(&g_allocator.total_memory, size);

	return chunk;
}

// Free a chunk
static void free_chunk(chunk_t *chunk) {
	if (chunk->memory) {
		munmap(chunk->memory, chunk->size);
		atomic_fetch_sub(&g_allocator.total_memory, chunk->size);
	}
	free(chunk);
}

bool mmap_alloc_init(size_t chunk_size) {
	if (g_allocator.initialized) {
		return true;
	}

	if (chunk_size == 0) {
		chunk_size = MMAP_DEFAULT_CHUNK_SIZE;
	}

	g_allocator.chunk_size = chunk_size;
	g_allocator.chunks = NULL;
	pthread_mutex_init(&g_allocator.lock, NULL);
	atomic_store(&g_allocator.chunks_allocated, 0);
	atomic_store(&g_allocator.large_allocs, 0);
	atomic_store(&g_allocator.total_memory, 0);
	atomic_store(&g_allocator.memory_in_use, 0);
	g_allocator.initialized = true;

	return true;
}

void mmap_alloc_cleanup(void) {
	if (!g_allocator.initialized) {
		return;
	}

	pthread_mutex_lock(&g_allocator.lock);

	chunk_t *chunk = g_allocator.chunks;
	while (chunk) {
		chunk_t *next = chunk->next;
		free_chunk(chunk);
		chunk = next;
	}

	g_allocator.chunks = NULL;
	g_allocator.initialized = false;

	pthread_mutex_unlock(&g_allocator.lock);
	pthread_mutex_destroy(&g_allocator.lock);
}

void *mmap_alloc(size_t size) {
	if (!g_allocator.initialized) {
		return NULL;
	}

	if (size == 0) {
		return NULL;
	}

	size_t aligned_size = align_size(size);
	size_t total_size = aligned_size + sizeof(alloc_header_t);

	// For large allocations (>= chunk_size), use dedicated mmap
	if (total_size >= g_allocator.chunk_size) {
		void *memory = mmap(NULL, total_size, PROT_READ | PROT_WRITE,
		                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
		if (memory == MAP_FAILED) {
			return NULL;
		}

		alloc_header_t *header = (alloc_header_t *)memory;
		header->size = size;
		header->actual_size = total_size;
		header->is_large = true;
		header->chunk = NULL;

		atomic_fetch_add(&g_allocator.large_allocs, 1);
		atomic_fetch_add(&g_allocator.total_memory, total_size);
		atomic_fetch_add(&g_allocator.memory_in_use, total_size);

		return (char *)memory + sizeof(alloc_header_t);
	}

	// For small allocations, allocate from chunks
	pthread_mutex_lock(&g_allocator.lock);

	chunk_t *chunk = g_allocator.chunks;

	// Find a chunk with enough space
	while (chunk) {
		if (chunk->size - chunk->used >= total_size) {
			break;
		}
		chunk = chunk->next;
	}

	// If no suitable chunk found, allocate a new one
	if (!chunk) {
		chunk = allocate_chunk(g_allocator.chunk_size);
		if (!chunk) {
			pthread_mutex_unlock(&g_allocator.lock);
			return NULL;
		}
		// Add to front of list
		chunk->next = g_allocator.chunks;
		g_allocator.chunks = chunk;
	}

	// Allocate from chunk
	void *memory = (char *)chunk->memory + chunk->used;
	chunk->used += total_size;

	pthread_mutex_unlock(&g_allocator.lock);

	alloc_header_t *header = (alloc_header_t *)memory;
	header->size = size;
	header->actual_size = total_size;
	header->is_large = false;
	header->chunk = chunk;

	atomic_fetch_add(&g_allocator.memory_in_use, total_size);

	return (char *)memory + sizeof(alloc_header_t);
}

void *mmap_realloc(void *ptr, size_t size) {
	if (!ptr) {
		return mmap_alloc(size);
	}

	if (size == 0) {
		mmap_free(ptr);
		return NULL;
	}

	// Get the old allocation header
	alloc_header_t *old_header = (alloc_header_t *)((char *)ptr - sizeof(alloc_header_t));

	// If the new size fits in the existing allocation, just update the header
	size_t aligned_size = align_size(size);
	size_t total_size = aligned_size + sizeof(alloc_header_t);

	if (total_size <= old_header->actual_size) {
		old_header->size = size;
		return ptr;
	}

	// Otherwise, allocate new memory and copy
	void *new_ptr = mmap_alloc(size);
	if (!new_ptr) {
		return NULL;
	}

	// Copy old data
	size_t copy_size = (old_header->size < size) ? old_header->size : size;
	memcpy(new_ptr, ptr, copy_size);

	// Free old memory
	mmap_free(ptr);

	return new_ptr;
}

void mmap_free(void *ptr) {
	if (!ptr || !g_allocator.initialized) {
		return;
	}

	alloc_header_t *header = (alloc_header_t *)((char *)ptr - sizeof(alloc_header_t));

	atomic_fetch_sub(&g_allocator.memory_in_use, header->actual_size);

	if (header->is_large) {
		// Large allocation - unmap directly
		munmap(header, header->actual_size);
		atomic_fetch_sub(&g_allocator.total_memory, header->actual_size);
		atomic_fetch_sub(&g_allocator.large_allocs, 1);
	}
	// For chunk allocations, we don't actually free the memory
	// The chunk will be reused for future allocations
	// This is a simple bump allocator - no per-allocation freeing
	// To support freeing, we would need a more complex allocator like a free list
}

size_t mmap_malloc_size(void *ptr) {
	if (!ptr) {
		return 0;
	}

	alloc_header_t *header = (alloc_header_t *)((char *)ptr - sizeof(alloc_header_t));
	return header->size;
}

void mmap_get_stats(mmap_stats_t *stats) {
	if (!stats) {
		return;
	}

	stats->chunk_size = g_allocator.chunk_size;
	stats->chunks_allocated = atomic_load(&g_allocator.chunks_allocated);
	stats->large_allocs = atomic_load(&g_allocator.large_allocs);
	stats->total_memory = atomic_load(&g_allocator.total_memory);
	stats->memory_in_use = atomic_load(&g_allocator.memory_in_use);
}
