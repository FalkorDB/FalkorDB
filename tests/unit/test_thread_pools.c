/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "src/util/rmalloc.h"
#include "src/util/thpool/pool.h"
#include "src/configuration/config.h"

#include <assert.h>

#define WORKERS_COUNT 5

void setup() {
	Alloc_Reset();
}

#define TEST_INIT setup();
#include "acutest.h"

static void get_thread_friendly_id(void *arg) {
	int *threadID = (int*)arg;
	*threadID = ThreadPool_GetThreadID();	
}

void test_threadPool_threadID() {
	ThreadPool_CreatePool(WORKERS_COUNT, UINT64_MAX);

	// verify thread count equals to the number of worker threads
	TEST_ASSERT(WORKERS_COUNT == ThreadPool_ThreadCount());

	volatile int thread_ids[WORKERS_COUNT + 1];
	for(int i = 0; i < WORKERS_COUNT + 1; i++) {
		thread_ids[i] = -1;
	}

	// get main thread friendly id
	thread_ids[0] = ThreadPool_GetThreadID();

	// get worker threads friendly ids
	for(int i = 0; i < WORKERS_COUNT; i++) {
		int offset = i + 1;
		TEST_ASSERT(0 == 
				ThreadPool_AddWork(get_thread_friendly_id,
					(int*)(thread_ids + offset), false));
	}

	// wait for all threads
	for(int i = 0; i < WORKERS_COUNT + 1; i++) {
		while(thread_ids[i] == -1) { i = i; }
	}

	// main thread
	int main_thread_id = 0;
	TEST_ASSERT(thread_ids[0] == main_thread_id);

	// worker thread ids should be > main thread id
	for(int i = 0; i < WORKERS_COUNT; i++) {
		TEST_ASSERT(thread_ids[i+1] > main_thread_id);
	}
}

TEST_LIST = {
	{"threadPool_threadID", test_threadPool_threadID},
	{NULL, NULL}
};

