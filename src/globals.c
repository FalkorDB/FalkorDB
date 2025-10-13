/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "globals.h"
#include "util/arr.h"
#include "util/thpool/pool.h"
#include "configuration/config.h"
#include "string_pool/string_pool.h"

struct Globals {
	pthread_rwlock_t lock;              // READ/WRITE lock
	bool process_is_child;              // running process is a child process
	CommandCtx **command_ctxs;          // list of CommandCtxs
	GraphContext **graphs_in_keyspace;  // list of graphs in keyspace
	StringPool string_pool;             // pool of reusable strings
	pthread_t main_thread_id;           // process main thread id
	struct GrB_ops ops;                 // global GraphBLAS objects
};

struct Globals _globals = {0};

// initialize global variables
void Globals_Init(void) {
	// expecting Global_Init to be called only once
	ASSERT(_globals.graphs_in_keyspace == NULL);

	// initialize
	_globals.process_is_child   = false;
	_globals.string_pool        = StringPool_create();
	_globals.graphs_in_keyspace = array_new(GraphContext*, 1);
	_globals.command_ctxs       = rm_calloc(ThreadPool_ThreadCount() + 1,
			sizeof(CommandCtx *));

	int res = pthread_rwlock_init(&_globals.lock, NULL);
	ASSERT(res == 0);

	// initialize GraphBLAS operators
	Global_GrB_Ops_Init();
}

StringPool Globals_Get_StringPool(void) {
	return _globals.string_pool;
}

// read global variable 'process_is_child'
bool Globals_Get_ProcessIsChild(void) {
	bool process_is_child = false;

	pthread_rwlock_rdlock(&_globals.lock);

	process_is_child = _globals.process_is_child;

	pthread_rwlock_unlock(&_globals.lock);

	return process_is_child;
}

// set global variable 'process_is_child'
void Globals_Set_ProcessIsChild
(
	bool process_is_child
) {
	pthread_rwlock_wrlock(&_globals.lock);

	_globals.process_is_child =	process_is_child;

	pthread_rwlock_unlock(&_globals.lock);
}

// get main thread id
pthread_t Globals_Get_MainThreadId(void) {
	return _globals.main_thread_id;
}

// get direct access to 'graphs_in_keyspace'
GraphContext **Globals_Get_GraphsInKeyspace(void) {
	return _globals.graphs_in_keyspace;
}

// add graph to global tracker
void Globals_AddGraph
(
	GraphContext *gc  // graph to add
) {
	ASSERT(gc != NULL);

	// increase ref count regardless if 'gc' is already tracked
	GraphContext_IncreaseRefCount(gc);

	// acuire write lock
	pthread_rwlock_wrlock(&_globals.lock);

	bool registered = false;
	uint n = array_len(_globals.graphs_in_keyspace);
	for(uint i = 0; i < n; i++) {
		if(_globals.graphs_in_keyspace[i] == gc) {
			registered = true;
			break;
		}
	}

	if(registered == false) {
		// append graph
		array_append(_globals.graphs_in_keyspace, gc);
	}

	// release lock
	pthread_rwlock_unlock(&_globals.lock);
}

// remove graph from global tracker
void Globals_RemoveGraph
(
	GraphContext *gc  // graph to remove
) {
	ASSERT(gc != NULL);

	uint64_t i = 0;
	uint64_t n = array_len(_globals.graphs_in_keyspace);
	if(n == 0) return;

	// acuire write lock
	pthread_rwlock_wrlock(&_globals.lock);

	// search for graph
	for(; i < n; i++) {
		if(_globals.graphs_in_keyspace[i] == gc) {
			break;
		}
	}

	// graph must be found
	ASSERT(i != n);

	// graph located, remove it
	array_del_fast(_globals.graphs_in_keyspace, i);

	// release lock
	pthread_rwlock_unlock(&_globals.lock);
}

// remove a graph by its name
void Globals_RemoveGraphByName
(
	const char *name  // graph name to remove
) {
	ASSERT(name != NULL);

	// acuire write lock
	pthread_rwlock_wrlock(&_globals.lock);

	// search for graph
	uint64_t i = 0;
	uint64_t n = array_len(_globals.graphs_in_keyspace);
	for(; i < n; i++) {
		GraphContext *gc = _globals.graphs_in_keyspace[i];
		if(strcmp(name, GraphContext_GetName(gc)) == 0) {
			break;
		}
	}

	if(i != n) {
		// graph located, remove it
		array_del_fast(_globals.graphs_in_keyspace, i);
	}

	// release lock
	pthread_rwlock_unlock(&_globals.lock);
}

// clear all tracked graphs
void Globals_ClearGraphs
(
	RedisModuleCtx *ctx
) {
	// acquire write lock
	pthread_rwlock_wrlock(&_globals.lock);
	
	for(uint i = 0; i < array_len(_globals.graphs_in_keyspace); i++) {
		GraphContext *gc = _globals.graphs_in_keyspace[i];
		if(gc->telemetry_stream != NULL) {
			RedisModule_FreeString(ctx, gc->telemetry_stream);
			gc->telemetry_stream = NULL;
		}
	}

	// clear graph tracking
	array_clear(_globals.graphs_in_keyspace);

	// release lock
	pthread_rwlock_unlock(&_globals.lock);
}

//------------------------------------------------------------------------------
// Command context tracking
//------------------------------------------------------------------------------

// track CommandCtx
void Globals_TrackCommandCtx
(
	CommandCtx *ctx  // CommandCtx to track
) {
	ASSERT(ctx != NULL);
	ASSERT(_globals.command_ctxs != NULL);

	int tid = ThreadPool_GetThreadID();

	// acuire read lock
	pthread_rwlock_rdlock(&_globals.lock);

	// expecting slot to be empty
	ASSERT(_globals.command_ctxs[tid] == NULL);

	// set ctx at the current thread entry
	_globals.command_ctxs[tid] = ctx;

	// release lock
	pthread_rwlock_unlock(&_globals.lock);

	// reset thread memory consumption to 0 (no memory consumed)
	rm_reset_n_alloced();
}

// untrack CommandCtx
void Globals_UntrackCommandCtx
(
	const CommandCtx *ctx  // CommandCtx to untrack
) {
	ASSERT(ctx != NULL);
	ASSERT(_globals.command_ctxs != NULL);

	int tid = ThreadPool_GetThreadID();

	// acuire read lock
	pthread_rwlock_rdlock(&_globals.lock);

	ASSERT(_globals.command_ctxs[tid] == ctx);

	// clear ctx at the current thread entry
	// CommandCtx_Free will free it eventually
	_globals.command_ctxs[tid] = NULL;

	// release lock
	pthread_rwlock_unlock(&_globals.lock);
}

// get all currently tracked CommandCtxs
void Globals_GetCommandCtxs
(
	CommandCtx **commands,  // array to populate
	uint32_t *count         // [input/output] number of entries in 'commands'
) {
	ASSERT(count != NULL);
	ASSERT(commands != NULL);
	ASSERT(_globals.command_ctxs != NULL);

	// make a copy of the command contexts
	uint32_t nthreads = ThreadPool_ThreadCount() + 1;
	uint32_t cap = *count;  // capacity of 'commands'
	uint32_t found = 0;     // number of command contexts found

	// acquire write lock
	pthread_rwlock_wrlock(&_globals.lock);

	// increase ref count of each CommandCtx
	for(uint32_t i = 0; i < nthreads && found < cap; i++) {
		CommandCtx *cmd = _globals.command_ctxs[i];
		if(cmd != NULL) {
			 CommandCtx_Incref(cmd);
			 commands[found++] = cmd;
		}
	}

	// release lock
	pthread_rwlock_unlock(&_globals.lock);

	// update number of command contexts found
	*count = found;
}

//------------------------------------------------------------------------------
// graphs in keyspace iterator
//------------------------------------------------------------------------------

// initialize iterator over graphs in keyspace
void Globals_ScanGraphs
(
	KeySpaceGraphIterator *it
) {
	ASSERT(it != NULL);
	it->idx = 0;
}

// seek iterator to index
void GraphIterator_Seek
(
	KeySpaceGraphIterator *it,  // iterator
	uint64_t idx                // index to seek to
) {
	ASSERT(it != NULL);
	it->idx = idx;
}

// advance iterator
// returns graph object in case iterator isn't depleted, otherwise returns NULL
GraphContext *GraphIterator_Next
(
	KeySpaceGraphIterator *it  // iterator to advance
) {
	ASSERT(it != NULL);

	GraphContext *gc = NULL;

	pthread_rwlock_rdlock(&_globals.lock);

	if(it->idx < array_len(_globals.graphs_in_keyspace)) {
		// prepare next call
		gc = _globals.graphs_in_keyspace[it->idx++];
		GraphContext_IncreaseRefCount(gc);
	}

	pthread_rwlock_unlock(&_globals.lock);

	return gc;
}

// free globals
void Globals_Free(void) {
	rm_free(_globals.command_ctxs);
	array_free(_globals.graphs_in_keyspace);
	StringPool_free(&_globals.string_pool);
	pthread_rwlock_destroy(&_globals.lock);
	Global_GrB_Ops_Free();
}

#include "graph/tensor/tensor_grb_functions.h"

void Global_GrB_Ops_Init(void) {
	struct GrB_ops *ops = &_globals.ops;
	// initialize global GraphBLAS objects

	//--------------------------------------------------------------------------
	// initialize operators for zombie handling
	//--------------------------------------------------------------------------
	GrB_OK (GxB_BinaryOp_new(
		&ops->not_zombie, (GxB_binary_function) _entry_present, 
		GrB_BOOL, GrB_BOOL, GrB_UINT64, "_entry_present", _ENTRY_PRESENT_JIT_STR
	));
	
	GrB_OK (GrB_BinaryOp_new( &ops->push_id, (GxB_binary_function) _push_element, 
		GrB_UINT64, GrB_UINT64, GrB_UINT64));
	
	GrB_OK (GrB_Semiring_new (&ops->any_alive, GrB_LOR_MONOID_BOOL, 
		ops->not_zombie));
	
	//--------------------------------------------------------------------------
	// initialize operators for freeing
	//--------------------------------------------------------------------------
	GrB_OK (GrB_UnaryOp_new(
		&ops->free_tensors, (GxB_unary_function) _free_vectors, 
		GrB_UINT64, GrB_UINT64));

	//--------------------------------------------------------------------------
	// initialize scalars
	//--------------------------------------------------------------------------
	GrB_OK (GrB_Scalar_new(&ops->empty, GrB_BOOL));
	GrB_OK (GrB_Scalar_new(&ops->bool_zombie, GrB_BOOL));
	GrB_OK (GrB_Scalar_new(&ops->u64_zombie, GrB_BOOL));

	GrB_OK (GrB_Scalar_setElement_BOOL(ops->bool_zombie, BOOL_ZOMBIE));
	GrB_OK (GrB_Scalar_setElement_UINT64(ops->u64_zombie, U64_ZOMBIE));
}

void Global_GrB_Ops_Free(void) {
	struct GrB_ops *ops = &_globals.ops;

	//--------------------------------------------------------------------------
	// free everything
	//--------------------------------------------------------------------------
	GrB_free(&ops->not_zombie);
	GrB_free(&ops->any_alive);
	GrB_free(&ops->free_tensors);
	GrB_free(&ops->empty);
	GrB_free(&ops->bool_zombie);
	GrB_free(&ops->u64_zombie);
}

const struct GrB_ops *Global_GrB_Ops_Get(void) {
	return &_globals.ops;
}