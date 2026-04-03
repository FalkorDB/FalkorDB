/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "thpool.h"
#include <sys/types.h>

#define THPOOL_QUEUE_FULL -2

// initialize pool
int ThreadPool_Init(void);

// set up thread pool
// returns 1 if thread pool initialized, 0 otherwise
int ThreadPool_CreatePool
(
	uint count,
	uint64_t max_pending_work
);

// return number of threads in the pool
uint ThreadPool_ThreadCount(void);

// retrieve current thread id
// 0         redis-main
// 1..N + 1  workers
int ThreadPool_GetThreadID(void);

// adds a task
int ThreadPool_AddWork
(
	void (*function_p)(void *),  // function to run
	void *arg_p,                 // function arguments
	int force                    // true will add task even if internal queue is full
);

// sets the limit on max queued tasks in pool
void ThreadPool_SetMaxPendingWork
(
	uint64_t cap  // pool's queue capacity
);

// returns a list of queued tasks that match the given handler
// caller must free the returned list
void **ThreadPool_GetTasksByHandler
(
	void (*handler)(void *),  // task handler to match
	void (*match)(void *),    // [optional] function to invoke on each match
	uint32_t *n               // number of tasks returned
);

// destroies threadpool, allows threads to exit gracefully
void ThreadPool_Destroy(void);

