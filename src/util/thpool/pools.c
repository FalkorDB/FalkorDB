/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include <pthread.h>
#include "RG.h"
#include "pools.h"
#include "../../configuration/config.h"

//------------------------------------------------------------------------------
// Thread pools
//------------------------------------------------------------------------------

static threadpool _readers_thpool = NULL;  // readers
static threadpool _writers_thpool = NULL;  // writers

pthread_t MAIN_THREAD_ID;  // redis main thread ID

int ThreadPools_Init
(
) {
	bool      config_read    = true;
	uint64_t  reader_count   = 1;
	int       writer_count   = 1;
	uint64_t  max_queue_size = UINT64_MAX;

	UNUSED(config_read);

	// get thread pool size and thread pool internal queue length from config
	config_read = Config_Option_get(Config_THREAD_POOL_SIZE, &reader_count);
	ASSERT(config_read == true);

	config_read = Config_Option_get(Config_MAX_QUEUED_QUERIES, &max_queue_size);
	ASSERT(config_read == true);

	int res = ThreadPools_CreatePools(reader_count, writer_count,
			max_queue_size);

	//--------------------------------------------------------------------------
	// set main thread and writer thread IDs
	//--------------------------------------------------------------------------

	MAIN_THREAD_ID = pthread_self();  // it is the main thread who's running

	return res;
}

// set up thread pools  (readers and writers)
// returns 1 if thread pools initialized, 0 otherwise
int ThreadPools_CreatePools
(
	uint reader_count,
	uint writer_count,
	uint64_t max_pending_work
) {
	ASSERT(writer_count    == 1);  // we only allow for a single writer
	ASSERT(_readers_thpool == NULL);
	ASSERT(_writers_thpool == NULL);

	_readers_thpool = thpool_init(reader_count, "reader");
	if(_readers_thpool == NULL) return 0;

	_writers_thpool = thpool_init(writer_count, "writer");
	if(_writers_thpool == NULL) return 0;

	ThreadPools_SetMaxPendingWork(max_pending_work);

	return 1;
}

// return number of threads in both the readers and writers pools
uint ThreadPools_ThreadCount
(
	void
) {
	ASSERT(_readers_thpool != NULL);
	ASSERT(_writers_thpool != NULL);

	uint count = 0;
	count += thpool_num_threads(_readers_thpool);
	count += thpool_num_threads(_writers_thpool);

	return count;
}

uint ThreadPools_ReadersCount
(
	void
) {
	ASSERT(_readers_thpool != NULL);
	return thpool_num_threads(_readers_thpool);
}

// retrieve current thread id
// 0         redis-main
// 1..N + 1  readers
// N + 2..   writers
int ThreadPools_GetThreadID
(
	void
) {
	ASSERT(_readers_thpool != NULL);
	ASSERT(_writers_thpool != NULL);

	// thpool_get_thread_id returns -1 if pthread_self isn't in the thread pool
	// most likely Redis main thread
	int thread_id;
	pthread_t pthread = pthread_self();
	int readers_count = thpool_num_threads(_readers_thpool);

	// search in writers
	thread_id = thpool_get_thread_id(_writers_thpool, pthread);
	// compensate for Redis main thread
	if(thread_id != -1) return readers_count + thread_id + 1;

	// search in readers pool
	thread_id = thpool_get_thread_id(_readers_thpool, pthread);
	// compensate for Redis main thread
	if(thread_id != -1) return thread_id + 1;

	return 0; // assuming Redis main thread
}

// pause all thread pools
void ThreadPools_Pause
(
	void
) {
	ASSERT(_readers_thpool != NULL);
	ASSERT(_writers_thpool != NULL);

	thpool_pause(_readers_thpool);
	thpool_pause(_writers_thpool);
}

void ThreadPools_Resume
(
	void
) {

	ASSERT(_readers_thpool != NULL);
	ASSERT(_writers_thpool != NULL);

	thpool_resume(_readers_thpool);
	thpool_resume(_writers_thpool);
}

// adds a read task
int ThreadPools_AddWorkReader
(
	void (*function_p)(void *),  // function to run
	void *arg_p,                 // function arguments
	int force                    // true will add task even if internal queue is full
) {
	ASSERT(_readers_thpool != NULL);

	// make sure there's enough room in thread pool queue
	if(!force && thpool_queue_full(_readers_thpool)) return THPOOL_QUEUE_FULL;

	return thpool_add_work(_readers_thpool, function_p, arg_p);
}

// add task for writer thread
int ThreadPools_AddWorkWriter
(
	void (*function_p)(void *),
	void *arg_p,
	int force
) {
	ASSERT(_writers_thpool != NULL);

	// make sure there's enough room in thread pool queue
	if(thpool_queue_full(_writers_thpool) && !force) return THPOOL_QUEUE_FULL;

	return thpool_add_work(_writers_thpool, function_p, arg_p);
}

void ThreadPools_SetMaxPendingWork(uint64_t val) {
	if(_readers_thpool != NULL) thpool_set_jobqueue_cap(_readers_thpool, val);
	if(_writers_thpool != NULL) thpool_set_jobqueue_cap(_writers_thpool, val);
}

// returns a list of queued tasks that match the given handler
// caller must free the returned list
void **ThreadPools_GetTasksByHandler
(
	void (*handler)(void *),  // task handler to match
	void (*match)(void *),    // [optional] function to invoke on each match
	uint32_t *n               // number of tasks returned
) {
	// validations
	ASSERT(handler         != NULL);
	ASSERT(_readers_thpool != NULL);
	ASSERT(_writers_thpool != NULL);

	// cap number of read tasks
	uint32_t r_task_count = (thpool_get_jobqueue_len(_readers_thpool) > 1000)
		? 1000
		: thpool_get_jobqueue_len(_readers_thpool);

	// cap number of write tasks
	uint32_t w_task_count = (thpool_get_jobqueue_len(_writers_thpool) > 1000)
		? 1000
		: thpool_get_jobqueue_len(_writers_thpool);

	void **tasks = malloc(sizeof(void *) * (r_task_count + w_task_count));

	// collect tasks from readers and writers
	thpool_get_tasks(_readers_thpool, tasks, &r_task_count, handler, match);
	thpool_get_tasks(_writers_thpool, tasks + r_task_count, &w_task_count,
			handler, match);

	// update number of tasks
	*n = r_task_count + w_task_count;

	return tasks;
}

void ThreadPools_Destroy
(
	void
) {
	ASSERT(_readers_thpool != NULL);
	ASSERT(_writers_thpool != NULL);

	thpool_destroy(_readers_thpool);
	thpool_destroy(_writers_thpool);
}

