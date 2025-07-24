/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "thpool.h"
#include <sys/types.h>

#define THPOOL_QUEUE_FULL -2

// initialize pools
int ThreadPools_Init
(
	void
);

// create both readers and writers thread pools
int ThreadPools_CreatePools
(
	uint reader_count,
	uint writer_count,
	uint64_t max_pending_work
);

// return number of threads in both the readers and writers pools
uint ThreadPools_ThreadCount(void);

// return size of READERS thread-pool
uint ThreadPools_ReadersCount(void);

// retrieve current thread id
// 0         redis-main
// 1..N + 1  readers
// N + 2..   writers
int ThreadPools_GetThreadID(void);

// adds a read task
int ThreadPools_AddWorkReader
(
	void (*function_p)(void *),  // function to run
	void *arg_p,                 // function arguments
	int force                    // true will add task even if internal queue is full
);

// add a write task
int ThreadPools_AddWorkWriter
(
	void (*function_p)(void *),  // function to run
	void *arg_p,                 // function arguments
	int force                    // true will add task even if internal queue is full
);

// sets the limit on max queued queries in each thread pool
void ThreadPools_SetMaxPendingWork
(
	uint64_t val
);

// returns a list of queued tasks that match the given handler
// caller must free the returned list
void **ThreadPools_GetTasksByHandler
(
	void (*handler)(void *),  // task handler to match
	void (*match)(void *),    // [optional] function to invoke on each match
	uint32_t *n               // number of tasks returned
);

// destroies all threadpools, allows threads to exit gracefully
void ThreadPools_Destroy
(
	void
);

