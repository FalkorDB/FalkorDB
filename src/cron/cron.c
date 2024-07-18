/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "cron.h"
#include "util/heap.h"
#include "util/rmalloc.h"

#include <time.h>
#include <errno.h>
#include <pthread.h>
#include <stdbool.h>

#define MAX(a,b) ((a) >= (b)) ? (a) : (b)

//------------------------------------------------------------------------------
// Data structures
//------------------------------------------------------------------------------

// CRON task
typedef struct {
	struct timespec due;    // absolute time for when task should run
	CronTaskCB cb;          // callback to call when task is due
	CronTaskFree free;      // [optional] private data free function
	void *pdata;            // [optional] private data passed to callback
} CRON_TASK;

// CRON object
typedef struct {
	bool alive;                        // indicates cron is active
	heap_t *tasks;                     // min heap of cron tasks
	CRON_TASK* volatile current_task;  // current task being executed
	pthread_mutex_t mutex;             // mutex control access to tasks
	pthread_mutex_t condv_mutex;       // mutex control access to condv
	pthread_cond_t condv;              // conditional variable
	pthread_t thread;                  // thread running cron main loop
} CRON;

// single static CRON instance, initialized at CRON_Start
static CRON *cron = NULL;

// compares two time objects
static int cmp_timespec
(
	struct timespec a,
	struct timespec b
) {
	if(a.tv_sec == b.tv_sec) {
		return a.tv_nsec - b.tv_nsec;
	} else {
		return a.tv_sec - b.tv_sec;
	}
}

// minimum heap sort function
static int CRON_JobCmp
(
	const void *a,
	const void *b,
	void *udata
) {
	CRON_TASK *_a = (CRON_TASK*)a;
	CRON_TASK *_b = (CRON_TASK*)b;
	return cmp_timespec(_b->due, _a->due);
}

//------------------------------------------------------------------------------
// Utility functions
//------------------------------------------------------------------------------

// compute now + ms
static struct timespec due_in_ms
(
	uint ms
) {
	struct timespec due;
	clock_gettime(CLOCK_REALTIME, &due);

	due.tv_sec += ms / 1000;
	due.tv_nsec += (ms % 1000) * 1000000;
	// add the overflow seconds otherwise the time will be invalid
	// and the thread will wake up immediately which lead to busy loop
	due.tv_sec += due.tv_nsec / 1000000000;
	due.tv_nsec %= 1000000000;

	return due;
}

static void CRON_WakeUp(void) {
	// Set conditional variable to wake up CRON thread
	pthread_mutex_lock(&cron->condv_mutex);
	pthread_cond_signal(&cron->condv);
	pthread_mutex_unlock(&cron->condv_mutex);
}

// determines if task is due
static bool CRON_TaskDue
(
	const CRON_TASK *t
) {
	ASSERT(t != NULL);

	struct timespec now;
	clock_gettime(CLOCK_REALTIME, &now);
	return cmp_timespec(now, t->due) >= 0;
}

static CRON_TASK *CRON_Peek() {
	pthread_mutex_lock(&cron->mutex);
	CRON_TASK *task = Heap_peek(cron->tasks);
	pthread_mutex_unlock(&cron->mutex);
	return task;
}

static bool CRON_RemoveTask
(
	const CRON_TASK *t
) {
	ASSERT(t != NULL);

	pthread_mutex_lock(&cron->mutex);
	void *res = Heap_remove_item(cron->tasks, t);
	pthread_mutex_unlock(&cron->mutex);

	return res != NULL;
}

static bool CRON_RemoveCurrentTask
(
	const CRON_TASK *t  // task to remove
) {
	ASSERT(t != NULL);

	pthread_mutex_lock(&cron->mutex);
	cron->current_task = Heap_remove_item(cron->tasks, t);
	pthread_mutex_unlock(&cron->mutex);

	return cron->current_task != NULL;
}

static void CRON_InsertTask
(
	CRON_TASK *t
) {
	ASSERT(t != NULL);
	pthread_mutex_lock(&cron->mutex);
	Heap_offer(&cron->tasks, t);
	pthread_mutex_unlock(&cron->mutex);

	CRON_WakeUp();
}

static void CRON_PerformTask
(
	CRON_TASK *t
) {
	ASSERT(t);
	t->cb(t->pdata);
}

static void CRON_FreeTask
(
	CRON_TASK *t
) {
	ASSERT(t != NULL);

	// free task private data
	if(t->pdata != NULL && t->free != NULL) {
		t->free(t->pdata);
	}

	rm_free(t);
}

static void clear_tasks() {
	CRON_TASK *task = NULL;
	while((task = Heap_poll(cron->tasks))) {
		CRON_FreeTask(task);
	}
}

//------------------------------------------------------------------------------
// CRON main loop
//------------------------------------------------------------------------------

static void *Cron_Run
(
	void *arg
) {
	while(cron->alive) {
		// execute due tasks
		CRON_TASK *task = NULL;
		while((task = CRON_Peek()) && CRON_TaskDue(task)) {
			if(!CRON_RemoveCurrentTask(task)) {
				// task is aborted
				continue;
			}

			// perform and free task
			CRON_PerformTask(task);
			cron->current_task = NULL;
			CRON_FreeTask(task);
		}

		// sleep
		struct timespec timeout = (task) ? task->due : due_in_ms(1000);
		pthread_mutex_lock(&cron->condv_mutex);
		int res = pthread_cond_timedwait(&cron->condv, &cron->condv_mutex, &timeout);
		ASSERT(res == 0 || res == ETIMEDOUT);
		pthread_mutex_unlock(&cron->condv_mutex);
	}

	return NULL;
}

//------------------------------------------------------------------------------
// User facing API
//------------------------------------------------------------------------------

bool Cron_Start(void) {
	ASSERT(cron == NULL);

	cron = rm_malloc(sizeof(CRON));

	cron->alive = true;
	cron->tasks = Heap_new(CRON_JobCmp, NULL);

	bool res = true;
	res &= pthread_cond_init(&cron->condv, NULL)               == 0;
	res &= pthread_mutex_init(&cron->mutex, NULL)              == 0;
	res &= pthread_mutex_init(&cron->condv_mutex, NULL)        == 0;
	res &= pthread_create(&cron->thread, NULL, Cron_Run, NULL) == 0;

	return res;
}

// stops CRON
// clears all tasks and waits for thread to terminate
void Cron_Stop(void) {
	ASSERT(cron != NULL);

	// stop cron main loop
	cron->alive = false;
	CRON_WakeUp();

	// wait for thread to terminate
	pthread_join(cron->thread, NULL);

	clear_tasks();

	// free resources
	Heap_free(cron->tasks);
	pthread_mutex_destroy(&cron->mutex);
	pthread_mutex_destroy(&cron->condv_mutex);
	pthread_cond_destroy(&cron->condv);
	rm_free(cron);
	cron = NULL;
}

// create a new CRON task
CronTaskHandle Cron_AddTask
(
	uint when,          // number of miliseconds until task invocation
	CronTaskCB work,    // callback to call when task is due
	CronTaskFree free,  // [optional] task private data free function
	void *pdata         // [optional] private data to pass to callback
) {
	ASSERT(work   != NULL);
	ASSERT(cron   != NULL);
	ASSERT(!(free != NULL && pdata == NULL));

	CRON_TASK *task = rm_malloc(sizeof(CRON_TASK));

	task->cb    = work;
	task->due   = due_in_ms(when);
	task->free  = free;
	task->pdata = pdata;

	CRON_InsertTask(task);

	return (uintptr_t)task;
}

// tries to abort given task
// in case task is currently being executed, it will wait for it to finish
bool Cron_AbortTask
(
	CronTaskHandle t  // task to abort
) {
	ASSERT(cron != NULL);

	CRON_TASK *task = (CRON_TASK *)t;

	// try remove the task
	if(!CRON_RemoveTask(task)) {
		// in case task is currently being performed, wait for it to finish
		while (cron->current_task == task) { }

		// task wan't aborted
		return false;
	}
	
	// free task
	CRON_FreeTask(task);

	// managed to abort task
	return true;
}

