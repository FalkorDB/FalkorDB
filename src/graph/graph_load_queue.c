/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "graph_load_queue.h"
#include "../util/arr.h"
#include "../util/dict.h"
#include "../util/rmalloc.h"
#include "../util/thpool/pool.h"
#include "../configuration/config.h"

#include <pthread.h>

// a single parked waiter
typedef struct {
	void (*handler) (void *) ;  // resubmitted to the thread pool on drain
	void  *arg ;                // passed to `handler` on drain
} Waiter ;

// graph name (owned copy) -> arr_t of Waiter, parked behind the owner
static dict            *_entries = NULL ;
static pthread_mutex_t  _lock     = PTHREAD_MUTEX_INITIALIZER ;

GraphLoadQueueStatus GraphLoadQueue_AcquireOrWait
(
	const char *graph_name,
	void      (*handler) (void *),
	void       *arg
) {
	ASSERT (graph_name != NULL) ;
	ASSERT (handler     != NULL) ;
	ASSERT (arg         != NULL) ;

	GraphLoadQueueStatus status ;

	pthread_mutex_lock (&_lock) ;

	if (_entries == NULL) {
		_entries = HashTableCreate (&string_dt) ;
	}

	dictEntry *de = HashTableFind (_entries, graph_name) ;

	if (de == NULL) {
		// no load currently in flight for this graph - caller becomes owner
		Waiter *waiters = arr_new (Waiter, 0) ;
		HashTableAdd (_entries, rm_strdup (graph_name), waiters) ;
		status = GraphLoadQueue_OWNER ;
	} else {
		Waiter *waiters = HashTableGetVal (de) ;

		uint64_t cap = 0 ;
		bool res = Config_Option_get (Config_MAX_QUEUED_QUERIES, &cap) ;
		ASSERT (res) ;

		if ((uint64_t) arr_len (waiters) >= cap) {
			status = GraphLoadQueue_FULL ;
		} else {
			Waiter w = { .handler = handler, .arg = arg } ;
			arr_append (waiters, w) ;  // may reallocate - write the pointer back
			HashTableSetVal (_entries, de, waiters) ;
			status = GraphLoadQueue_PARKED ;
		}
	}

	pthread_mutex_unlock (&_lock) ;

	return status ;
}

void GraphLoadQueue_Drain
(
	const char *graph_name
) {
	ASSERT (graph_name != NULL) ;

	if (_entries == NULL) {
		return ;
	}

	pthread_mutex_lock (&_lock) ;

	dictEntry *de = HashTableUnlink (_entries, graph_name) ;
	ASSERT (de != NULL) ;  // Drain is only called by a thread that owns this entry

	Waiter *waiters = HashTableGetVal (de) ;
	rm_free (HashTableGetKey (de)) ;
	HashTableFreeUnlinkedEntry (_entries, de) ;

	pthread_mutex_unlock (&_lock) ;

	uint32_t n = arr_len (waiters) ;
	for (uint32_t i = 0 ; i < n ; i++) {
		// force=true - these waiters were already accepted into the
		// per-graph wait list (a separate capacity from the thread pool's
		// own work queue) and must not be silently dropped
		ThreadPool_AddWork (waiters[i].handler, waiters[i].arg, true) ;
	}

	arr_free (waiters) ;
}

void GraphLoadQueue_Free (void) {
	if (_entries == NULL) {
		return ;
	}

	pthread_mutex_lock (&_lock) ;

	dictIterator *iter = HashTableGetIterator (_entries) ;
	dictEntry *de ;

	while ((de = HashTableNext (iter)) != NULL) {
		rm_free (HashTableGetKey (de)) ;

		Waiter *waiters = HashTableGetVal (de) ;
		uint32_t n = arr_len (waiters) ;
		for (uint32_t i = 0 ; i < n ; i++) {
			// same as a normal drain - resubmit, force=true
			ThreadPool_AddWork (waiters[i].handler, waiters[i].arg, true) ;
		}

		arr_free (waiters) ;
	}

	HashTableReleaseIterator (iter) ;

	HashTableRelease (_entries) ;
	_entries = NULL ;

	pthread_mutex_unlock  (&_lock) ;
}

