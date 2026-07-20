/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "graph_load_queue.h"
#include "../util/arr.h"
#include "../util/dict.h"
#include "../util/rmalloc.h"
#include "../configuration/config.h"

#include <pthread.h>

// a single parked waiter
typedef struct {
	void               *waiter ;  // opaque handle, owned by the caller
	GraphLoadWaiterCB   cb ;      // invoked once, on drain
} Waiter ;

// graph name (owned copy) -> arr_t of Waiter, parked behind the owner
static dict            *_entries = NULL ;
static pthread_mutex_t  _lock     = PTHREAD_MUTEX_INITIALIZER ;

GraphLoadQueueStatus GraphLoadQueue_AcquireOrWait
(
	const char        *graph_name,
	void              *waiter,
	GraphLoadWaiterCB  cb
) {
	ASSERT (graph_name != NULL) ;
	ASSERT (waiter     != NULL) ;
	ASSERT (cb         != NULL) ;

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
			Waiter w = { .waiter = waiter, .cb = cb } ;
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
	const char *graph_name,
	bool        success
) {
	ASSERT (graph_name != NULL) ;
	ASSERT (_entries    != NULL) ;

	pthread_mutex_lock (&_lock) ;

	dictEntry *de = HashTableUnlink (_entries, graph_name) ;
	ASSERT (de != NULL) ;  // Drain is only called by a thread that owns this entry

	Waiter *waiters = HashTableGetVal (de) ;
	rm_free (HashTableGetKey (de)) ;
	HashTableFreeUnlinkedEntry (_entries, de) ;

	pthread_mutex_unlock (&_lock) ;

	uint32_t n = arr_len (waiters) ;
	for (uint32_t i = 0 ; i < n ; i++) {
		waiters[i].cb (waiters[i].waiter, success) ;
	}

	arr_free (waiters) ;
}
