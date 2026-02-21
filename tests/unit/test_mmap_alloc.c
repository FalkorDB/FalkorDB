/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "src/util/mmap_alloc.h"
#include "acutest.h"
#include <string.h>

// Test basic initialization and cleanup
void test_mmap_init(void) {
	// Test initialization with default chunk size
	TEST_ASSERT(mmap_alloc_init(0) == true);

	mmap_stats_t stats;
	mmap_get_stats(&stats);
	TEST_ASSERT(stats.chunk_size == MMAP_DEFAULT_CHUNK_SIZE);
	TEST_ASSERT(stats.chunks_allocated == 0);
	TEST_ASSERT(stats.large_allocs == 0);
	TEST_ASSERT(stats.total_memory == 0);
	TEST_ASSERT(stats.memory_in_use == 0);

	mmap_alloc_cleanup();

	// Test initialization with custom chunk size
	TEST_ASSERT(mmap_alloc_init(50 * 1024 * 1024) == true);
	mmap_get_stats(&stats);
	TEST_ASSERT(stats.chunk_size == 50 * 1024 * 1024);

	mmap_alloc_cleanup();
}

// Test small allocations from chunks
void test_small_alloc(void) {
	TEST_ASSERT(mmap_alloc_init(0) == true);

	// Allocate small blocks
	void *p1 = mmap_alloc(1024);
	TEST_ASSERT(p1 != NULL);

	void *p2 = mmap_alloc(2048);
	TEST_ASSERT(p2 != NULL);

	void *p3 = mmap_alloc(4096);
	TEST_ASSERT(p3 != NULL);

	// Verify memory is usable
	memset(p1, 0xAA, 1024);
	memset(p2, 0xBB, 2048);
	memset(p3, 0xCC, 4096);

	// Check malloc_size
	TEST_ASSERT(mmap_malloc_size(p1) == 1024);
	TEST_ASSERT(mmap_malloc_size(p2) == 2048);
	TEST_ASSERT(mmap_malloc_size(p3) == 4096);

	// Check stats
	mmap_stats_t stats;
	mmap_get_stats(&stats);
	TEST_ASSERT(stats.chunks_allocated == 1);
	TEST_ASSERT(stats.large_allocs == 0);
	TEST_ASSERT(stats.memory_in_use > 0);

	// Free allocations
	mmap_free(p1);
	mmap_free(p2);
	mmap_free(p3);

	mmap_alloc_cleanup();
}

// Test large allocations (>= chunk_size)
void test_large_alloc(void) {
	size_t chunk_size = 10 * 1024 * 1024; // 10MB
	TEST_ASSERT(mmap_alloc_init(chunk_size) == true);

	// Allocate a large block (>= chunk_size)
	size_t large_size = 15 * 1024 * 1024; // 15MB
	void *p1 = mmap_alloc(large_size);
	TEST_ASSERT(p1 != NULL);

	// Verify memory is usable
	memset(p1, 0xDD, large_size);

	// Check malloc_size
	TEST_ASSERT(mmap_malloc_size(p1) == large_size);

	// Check stats - should use dedicated mmap, not chunk
	mmap_stats_t stats;
	mmap_get_stats(&stats);
	TEST_ASSERT(stats.large_allocs == 1);
	TEST_ASSERT(stats.chunks_allocated == 0);

	// Free allocation
	mmap_free(p1);

	// Check stats after free
	mmap_get_stats(&stats);
	TEST_ASSERT(stats.large_allocs == 0);

	mmap_alloc_cleanup();
}

// Test multiple chunks
void test_multiple_chunks(void) {
	size_t chunk_size = 1024 * 1024; // 1MB
	TEST_ASSERT(mmap_alloc_init(chunk_size) == true);

	// Allocate enough to fill multiple chunks
	void *ptrs[10];
	size_t alloc_size = 200 * 1024; // 200KB each

	for (int i = 0; i < 10; i++) {
		ptrs[i] = mmap_alloc(alloc_size);
		TEST_ASSERT(ptrs[i] != NULL);
		memset(ptrs[i], i, alloc_size);
	}

	// Should have allocated multiple chunks
	mmap_stats_t stats;
	mmap_get_stats(&stats);
	TEST_ASSERT(stats.chunks_allocated >= 2);

	// Free all
	for (int i = 0; i < 10; i++) {
		mmap_free(ptrs[i]);
	}

	mmap_alloc_cleanup();
}

// Test realloc
void test_realloc(void) {
	TEST_ASSERT(mmap_alloc_init(0) == true);

	// Allocate and fill with data
	void *p = mmap_alloc(1024);
	TEST_ASSERT(p != NULL);
	memset(p, 0xEE, 1024);

	// Realloc to larger size
	void *p2 = mmap_realloc(p, 2048);
	TEST_ASSERT(p2 != NULL);

	// Check first byte is preserved
	TEST_ASSERT(((unsigned char *)p2)[0] == 0xEE);

	// Check size
	TEST_ASSERT(mmap_malloc_size(p2) == 2048);

	// Realloc to smaller size (should keep same pointer)
	void *p3 = mmap_realloc(p2, 512);
	TEST_ASSERT(p3 != NULL);
	TEST_ASSERT(mmap_malloc_size(p3) == 512);

	// Free
	mmap_free(p3);

	mmap_alloc_cleanup();
}

// Test NULL and zero size edge cases
void test_edge_cases(void) {
	TEST_ASSERT(mmap_alloc_init(0) == true);

	// Test zero size allocation
	void *p1 = mmap_alloc(0);
	TEST_ASSERT(p1 == NULL);

	// Test realloc with NULL pointer (should act like alloc)
	void *p2 = mmap_realloc(NULL, 1024);
	TEST_ASSERT(p2 != NULL);
	TEST_ASSERT(mmap_malloc_size(p2) == 1024);

	// Test realloc to zero size (should act like free)
	void *p3 = mmap_realloc(p2, 0);
	TEST_ASSERT(p3 == NULL);

	// Test free with NULL pointer (should not crash)
	mmap_free(NULL);

	mmap_alloc_cleanup();
}

// Test alignment
void test_alignment(void) {
	TEST_ASSERT(mmap_alloc_init(0) == true);

	// Allocate various sizes and check alignment
	for (size_t size = 1; size <= 1024; size *= 2) {
		void *p = mmap_alloc(size);
		TEST_ASSERT(p != NULL);

		// Check 16-byte alignment
		TEST_ASSERT(((uintptr_t)p & 15) == 0);

		mmap_free(p);
	}

	mmap_alloc_cleanup();
}

TEST_LIST = {
	{"mmap_init", test_mmap_init},
	{"small_alloc", test_small_alloc},
	{"large_alloc", test_large_alloc},
	{"multiple_chunks", test_multiple_chunks},
	{"realloc", test_realloc},
	{"edge_cases", test_edge_cases},
	{"alignment", test_alignment},
	{NULL, NULL}
};
