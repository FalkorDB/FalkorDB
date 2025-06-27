/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "pool.h"
#include "../../configuration/config.h"

#include <pthread.h>

//------------------------------------------------------------------------------
// Thread pool
//------------------------------------------------------------------------------

static threadpool _thpool = NULL;

pthread_t MAIN_THREAD_ID;  // redis main thread ID

// initialize pool
int ThreadPool_Init(void) {
	bool      config_read    = true;
	uint64_t  count          = 1;
	uint64_t  max_queue_size = UINT64_MAX;

	UNUSED(config_read);

	// get thread pool size and thread pool internal queue length from config
	config_read = Config_Option_get(Config_THREAD_POOL_SIZE, &count);
	ASSERT(config_read == true);

	config_read = Config_Option_get(Config_MAX_QUEUED_QUERIES, &max_queue_size);
	ASSERT(config_read == true);

	int res = ThreadPool_CreatePool(count, max_queue_size);

	//--------------------------------------------------------------------------
	// set main thread id
	//--------------------------------------------------------------------------

	MAIN_THREAD_ID = pthread_self();  // it is the main thread who's running

	return res;
}

// set up thread pool
// returns 1 if thread pool initialized, 0 otherwise
int ThreadPool_CreatePool
(
	uint count,
	uint64_t max_pending_work
) {
	ASSERT(_thpool == NULL);

	_thpool = thpool_init(count, "thread-pool");
	if(_thpool == NULL) return 0;

	ThreadPool_SetMaxPendingWork(max_pending_work);

	return 1;
}

// return number of threads in the pool
uint ThreadPool_ThreadCount(void) {
	ASSERT(_thpool != NULL);

	return thpool_num_threads(_thpool);
}

// retrieve current thread id
// 0    redis-main
// 1..N workers
int ThreadPool_GetThreadID(void) {
	ASSERT(_thpool != NULL);

	// thpool_get_thread_id returns -1 if pthread_self isn't in the thread pool
	// most likely Redis main thread
	int thread_id;
	pthread_t pthread = pthread_self();
	int count = thpool_num_threads(_thpool);

	// search in pool
	thread_id = thpool_get_thread_id(_thpool, pthread);

	// compensate for Redis main thread
	if(thread_id != -1) return thread_id + 1;

	return 0; // assuming Redis main thread
}

// pause all threads
void ThreadPool_Pause(void) {
	ASSERT(_thpool != NULL);

	thpool_pause(_thpool);
}

// resume all threads
void ThreadPool_Resume(void) {
	ASSERT(_thpool != NULL);

	thpool_resume(_thpool);
}

// adds a task
int ThreadPool_AddWork
(
	void (*function_p)(void *),  // function to run
	void *arg_p,                 // function arguments
	int force                    // true will add task even if internal queue is full
) {
	ASSERT(_thpool != NULL);

	// make sure there's enough room in thread pool queue
	if(!force && thpool_queue_full(_thpool)) return THPOOL_QUEUE_FULL;

	return thpool_add_work(_thpool, function_p, arg_p);
}

// sets the limit on max queued tasks in pool
void ThreadPool_SetMaxPendingWork
(
	uint64_t cap  // pool's queue capacity
) {
	if(_thpool != NULL) thpool_set_jobqueue_cap(_thpool, val);
}

// returns a list of queued tasks that match the given handler
// caller must free the returned list
void **ThreadPool_GetTasksByHandler
(
	void (*handler)(void *),  // task handler to match
	void (*match)(void *),    // [optional] function to invoke on each match
	uint32_t *n               // number of tasks returned
) {
	// validations
	ASSERT(handler != NULL);
	ASSERT(_thpool != NULL);

	// cap number of read tasks
	uint32_t _task_count = (thpool_get_jobqueue_len(_thpool) > 1000)
		? 1000
		: thpool_get_jobqueue_len(_thpool);

	void **tasks = malloc(sizeof(void *) * _task_count);

	// collect tasks
	thpool_get_tasks(_thpool, tasks, &_task_count, handler, match);

	// update number of tasks
	*n = _task_count;

	return tasks;
}

// destroies threadpool, allows threads to exit gracefully
void ThreadPool_Destroy
(
	void
) {
	ASSERT(_thpool != NULL);

	thpool_destroy(_thpool);
}

