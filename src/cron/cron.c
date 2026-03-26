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

#define CRON_LOCK()                                            \
do {                                                           \
	ASSERT (cron != NULL) ;                                    \
	int res = pthread_mutex_lock (&cron->task_queue_lock) ;    \
	ASSERT (res == 0) ;                                        \
}                                                              \
while (false)                                                  \

#define CRON_UNLOCK()                                          \
do {                                                           \
	ASSERT (cron != NULL) ;                                    \
	int res = pthread_mutex_unlock (&cron->task_queue_lock) ;  \
	ASSERT (res == 0) ;                                        \
}                                                              \
while (false)                                                  \

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
	pthread_mutex_t task_queue_lock;   // guard heap + condvar
	pthread_cond_t task_enqueued;      // signaled when a task is inserted
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
	if (a.tv_sec == b.tv_sec) {
		return a.tv_nsec - b.tv_nsec ;
	} else {
		return a.tv_sec - b.tv_sec ;
	}
}

// minimum heap sort function
static int CRON_JobCmp
(
	const void *a,
	const void *b,
	void *udata
) {
	CRON_TASK *_a = (CRON_TASK*)a ;
	CRON_TASK *_b = (CRON_TASK*)b ;
	return cmp_timespec (_b->due, _a->due) ;
}

//------------------------------------------------------------------------------
// Utility functions
//------------------------------------------------------------------------------

// compute now + ms
static struct timespec due_in_ms
(
	uint ms
) {
	struct timespec due ;
	clock_gettime (CLOCK_REALTIME, &due) ;

	due.tv_sec += ms / 1000 ;
	due.tv_nsec += (ms % 1000) * 1000000 ;
	// add the overflow seconds otherwise the time will be invalid
	// and the thread will wake up immediately which lead to busy loop
	due.tv_sec += due.tv_nsec / 1000000000 ;
	due.tv_nsec %= 1000000000 ;

	return due ;
}

// peak at the next task and find out when it is due
// returns true if there is a task due.
// the method must be called under cron lock
static bool _CRON_PeekDue
(
	CRON_TASK **task,
	struct timespec *due
) {
	ASSERT (due  != NULL) ;
	ASSERT (task != NULL) ;

	struct timespec now ;

	*task = Heap_peek (cron->tasks) ;
	if (*task == NULL) {
		return false ;
	}

	clock_gettime (CLOCK_REALTIME, &now) ;
	*due = (*task)->due ;
	return cmp_timespec (now, *due) >= 0 ;
}

static bool CRON_RemoveTask
(
	const CRON_TASK *t
) {
	ASSERT (t != NULL) ;

	CRON_LOCK () ;

	void *removed = Heap_remove_item (cron->tasks, t) ;

	CRON_UNLOCK () ;

	return removed != NULL ;
}

// the method must be called under cron lock
static bool _CRON_RemoveCurrentTask
(
	const CRON_TASK *t  // task to remove
) {
	ASSERT (t != NULL) ;

	cron->current_task = Heap_remove_item (cron->tasks, t) ;
	return cron->current_task != NULL ;
}

static void CRON_InsertTask
(
	CRON_TASK *t
) {
	ASSERT (t != NULL) ;

	CRON_LOCK () ;

	Heap_offer (&cron->tasks, t) ;

	int res = pthread_cond_signal (&cron->task_enqueued) ;
	ASSERT (res == 0) ;

	CRON_UNLOCK () ;
}

static void CRON_PerformTask
(
	CRON_TASK *t
) {
	ASSERT (t) ;
	t->cb (t->pdata) ;
}

static void CRON_FreeTask
(
	CRON_TASK *t
) {
	ASSERT (t != NULL) ;

	// free task private data
	if (t->pdata != NULL && t->free != NULL) {
		t->free (t->pdata) ;
	}

	rm_free (t) ;
}

static void clear_tasks() {
	CRON_TASK *task = NULL ;
	while ((task = Heap_poll (cron->tasks))) {
		CRON_FreeTask (task) ;
	}
}

//------------------------------------------------------------------------------
// CRON main loop
//------------------------------------------------------------------------------

static void *Cron_Run
(
	void *arg
) {
	// hold the lock for the lifetime of the loop
    // pthread_cond_timedwait releases it atomically while sleeping
    // and reacquires it atomically on wake — so the lock is always
    // held at the top of each iteration without a redundant lock call

	CRON_LOCK () ;

	while (cron->alive) {
		CRON_TASK *task = NULL ;
		struct timespec due_time ;

		// drain all tasks that are due right now
        // the lock is released around the actual callback so that
        // inserters are never blocked while a task executes
		while (_CRON_PeekDue (&task, &due_time)) {
			if (!_CRON_RemoveCurrentTask (task)) {
				// task is aborted
				continue ;
			}

			// release the lock while the callback runs so that
            // CRON_AddTask / CRON_AbortTask do not stall
			CRON_UNLOCK () ;

			//------------------------------------------------------------------
			// perform and free task
			//------------------------------------------------------------------

			CRON_PerformTask (task) ;
			cron->current_task = NULL ;
			CRON_FreeTask (task) ;

			// lock before accessing the heap
			CRON_LOCK () ;
		}

		// no tasks are due right now
        // sleep until the nearest pending task is due, or at most 1 second
        // if no tasks exist (task == NULL) default to 1 second so we wake
        // up periodically to check cron->alive
		struct timespec sleep_until = (task) ? due_time : due_in_ms (1000) ;

		// atomically releases task_queue_lock and blocks
        // reacquires task_queue_lock before returning — whether woken by
        // task_enqueued or by timeout — so the next iteration is safe
		int res = pthread_cond_timedwait (&cron->task_enqueued,
				&cron->task_queue_lock, &sleep_until) ;
		ASSERT (res == 0 || res == ETIMEDOUT) ;
	}

	CRON_UNLOCK () ;

	return NULL ;
}

//------------------------------------------------------------------------------
// User facing API
//------------------------------------------------------------------------------

bool Cron_Start (void) {
	ASSERT (cron == NULL) ;

	cron = rm_calloc (1, sizeof (CRON)) ;

	cron->alive        = true ;
	cron->tasks        = Heap_new (CRON_JobCmp, NULL) ;
	cron->current_task = NULL ;

	int res = 0 ;
	res |= pthread_cond_init (&cron->task_enqueued, NULL) ;
	ASSERT (res == 0) ;

	res |= pthread_mutex_init (&cron->task_queue_lock, NULL) ;
	ASSERT (res == 0) ;

	res |= pthread_create (&cron->thread, NULL, Cron_Run, NULL) ;
	ASSERT (res == 0) ;

	return res == 0 ;
}

// stops CRON
// clears all tasks and waits for thread to terminate
void Cron_Stop (void) {
	ASSERT (cron != NULL) ;

	// set alive=false and signal under the lock so Cron_Run cannot miss
    // the state change between checking alive and entering timedwait

	CRON_LOCK () ;

	// stop cron main loop
	cron->alive = false ;

	int res = pthread_cond_signal (&cron->task_enqueued) ;
	ASSERT (res == 0) ;

	CRON_UNLOCK () ;

	// wait for thread to terminate
	pthread_join (cron->thread, NULL) ;

	clear_tasks () ;

	// free resources
	Heap_free (cron->tasks) ;

	res = pthread_cond_destroy (&cron->task_enqueued) ;
	ASSERT (res == 0) ;

	res = pthread_mutex_destroy (&cron->task_queue_lock) ;
	ASSERT (res == 0) ;

	rm_free (cron) ;

	cron = NULL ;
}

// create a new CRON task
CronTaskHandle Cron_AddTask
(
	uint when,          // number of milliseconds until task invocation
	CronTaskCB work,    // callback to call when task is due
	CronTaskFree free,  // [optional] task private data free function
	void *pdata         // [optional] private data to pass to callback
) {
	ASSERT (work   != NULL) ;
	ASSERT (cron   != NULL) ;
	ASSERT (!(free != NULL && pdata == NULL)) ;

	CRON_TASK *task = rm_malloc (sizeof(CRON_TASK)) ;

	task->cb    = work ;
	task->due   = due_in_ms (when) ;
	task->free  = free ;
	task->pdata = pdata ;

	CRON_InsertTask (task) ;

	return (uintptr_t)task ;
}

// tries to abort given task
// in case task is currently being executed, it will wait for it to finish
bool Cron_AbortTask
(
	CronTaskHandle t  // task to abort
) {
	ASSERT (cron != NULL) ;

	CRON_TASK *task = (CRON_TASK *)t ;

	// try remove the task
	if (!CRON_RemoveTask (task)) {
		// in case task is currently being performed, wait for it to finish
		while (cron->current_task == task) { }

		// task wasn't aborted
		return false ;
	}
	
	// free task
	CRON_FreeTask (task) ;

	// managed to abort task
	return true ;
}

