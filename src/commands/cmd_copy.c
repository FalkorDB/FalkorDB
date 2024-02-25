/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../redismodule.h"
#include "../util/thpool/pools.h"
#include "../graph/graphcontext.h"
#include "../serializers/serializer_io.h"
#include "../serializers/encoder/v14/encode_v14.h"
#include "../serializers/decoders/current/v14/decode_v14.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// encode graph into pipe
// this function should run on a child process, giving us the guarantee that
// 1. the cloned graph wouldn't change
// 2. due to memory seperation we do not need to hold any locks
// 3. we're allowed to make modification to the graph e.g. rename
static void _encode_graph
(
	RedisModuleCtx *ctx,         // redis module context
	int pipe_write_end,          // pipe write end
	GraphContext *src_graph,     // graph to clone
	const char *dest_graph_name  // name of cloned graph
) {
	// validations
	ASSERT(ctx             != NULL);
	ASSERT(src_graph       != NULL);
	ASSERT(dest_graph_name != NULL);

	// rename graph, needed by the decoding procedure
	// when the graph is decoded it is already holds the target name
	GraphContext_Rename(ctx, src_graph, dest_graph_name);

	//--------------------------------------------------------------------------
	// encode graph to stream
	//--------------------------------------------------------------------------

	SerializerIO writer = SerializerIO_FromPipe(pipe_write_end);

	// note: update this line when a new encoder is introduced
	RdbSaveGraph_v14(writer, src_graph);

	// clean up
	SerializerIO_Free(&writer);
}

// decode graph from pipe
// this function should be run on the parent process
// the parent process consumes data as it being produced by the child process
static void _decode_graph
(
	RedisModuleCtx *ctx,                // redis module context
	int pipe_read_end,                  // pipe read end
	RedisModuleString *dest_graph_name  // name of cloned graph
) {
	// create an empty dest graph
	// required by the decoding logic, which will populate an empty graph
	RedisModule_ThreadSafeContextLock(ctx); // lock GIL

	GraphContext *clone = GraphContext_Retrieve(ctx, dest_graph_name, true,
			true);
	GraphContext_DecreaseRefCount(clone);

	RedisModule_ThreadSafeContextUnlock(ctx);  // release GIL

	//--------------------------------------------------------------------------
	// decode graph from stream
	//--------------------------------------------------------------------------

	// create read serializer
	SerializerIO reader = SerializerIO_FromPipe(pipe_read_end);

	// decode graph
	RdbLoadGraphContext_v14(reader, dest_graph_name);

	// clean up
	SerializerIO_Free(&reader);
}

// GRAPH.COPY command context
typedef struct {
	char *src;                     // src graph id
	char *dest;                    // dest graph id
	RedisModuleBlockedClient *bc;  // blocked client
} GraphCopyContext;

// implements GRAPH.COPY logic
// this function is expected to run on a worker thread
// avoiding blocking redis main thread while the graph is being copied
static void _Graph_Copy
(
	void *context
) {
	GraphCopyContext *_context = (GraphCopyContext*)context;

	bool error = false;
	int pipefd[2] = {-1, -1};
	GraphContext *src_graph = NULL;

	char *src                    = _context->src;
	char *dest                   = _context->dest;
    RedisModuleBlockedClient *bc = _context->bc;
	RedisModuleCtx *ctx          = RedisModule_GetThreadSafeContext(bc);

	RedisModuleString * rm_src  =
		RedisModule_CreateString(ctx, src, strlen(src));
	RedisModuleString * rm_dest =
		RedisModule_CreateString(ctx, dest, strlen(dest));

	// lock GIL
	RedisModule_ThreadSafeContextLock(ctx);

	// make sure dest key does not exists
	RedisModuleKey *dest_key =
		RedisModule_OpenKey(ctx, rm_dest, REDISMODULE_READ);
	int dest_key_type = RedisModule_KeyType(dest_key);
	RedisModule_CloseKey(dest_key);

	// make sure src key is a graph
	src_graph = GraphContext_Retrieve(ctx, rm_src, true, false);

	// release GIL
	RedisModule_ThreadSafeContextUnlock(ctx);

	// validations
	if(dest_key_type != REDISMODULE_KEYTYPE_EMPTY) {
		// destination key already exists, abort
		error = true;
		RedisModule_ReplyWithError(ctx, "destination key already exists");
		goto cleanup;
	}

	if(src_graph == NULL) {
		// src graph is missing, abort
		error = true;
		// error alreay omitted by 'GraphContext_Retrieve'
		goto cleanup;
	}

	//--------------------------------------------------------------------------
	// create pipe
	//--------------------------------------------------------------------------

	if(pipe(pipefd) == -1) {
		// failed to create pipe, abort
		error = true;
		RedisModule_ReplyWithError(ctx,
				"Graph copy failed, could not create pipe");
		goto cleanup;
	}

	int pipe_read_end  = pipefd[0];
	int pipe_write_end = pipefd[1];

	//--------------------------------------------------------------------------
	// fork process
	//--------------------------------------------------------------------------

	// child process will encode src graph to stream
	// parent process will decode cloned graph from stream

	int pid = -1;
	while(pid == -1) {
		pid = RedisModule_Fork(NULL, NULL);
		if(pid < 0) {
			// failed to fork! retry
			// go to sleep for 1.0ms
			struct timespec sleep_time;
			sleep_time.tv_sec = 0;
			sleep_time.tv_nsec = 1000000;
			nanosleep(&sleep_time, NULL);
		} else if(pid == 0) {
			// child process
			_encode_graph(ctx, pipe_write_end, src_graph, dest);

			// all done, Redis require us to call 'RedisModule_ExitFromChild'
			RedisModule_ExitFromChild(0);
			return;
		} else {
			// parent process
			_decode_graph(ctx, pipe_read_end, rm_dest);
		}
	}

	// replicate command
	RedisModule_ThreadSafeContextLock(ctx);
	RedisModule_ReplicateVerbatim(ctx);
	RedisModule_ThreadSafeContextUnlock(ctx);

	// clean up
cleanup:

	// free redis module strings
	RedisModule_FreeString(ctx, rm_src);
	RedisModule_FreeString(ctx, rm_dest);

	// free command context
	rm_free(_context->src);
	rm_free(_context->dest);
	rm_free(_context);

	// decrease src graph ref-count
	if(src_graph != NULL) {
		GraphContext_DecreaseRefCount(src_graph);
	}

	// close pipe
	if(pipefd[0] != -1) {
		close(pipefd[0]);
		close(pipefd[1]);
	}

	// reply "OK" if no error was encountered
	if(!error) {
		RedisModule_ReplyWithCString(ctx, "OK");
	}

	// unblock the client
    RedisModule_UnblockClient(bc, NULL);
}

// clone a graph
// function executes on Redis main thread
//
// usage:
// GRAPH.COPY <src_graph> <dest_graph>
int Graph_Copy
(
	RedisModuleCtx *ctx,      // redis module context
	RedisModuleString **argv, // command argument
	int argc                  // number of argument
) {
	// validations
	ASSERT(ctx  != NULL);
	ASSERT(argv != NULL);

	// expecting exactly 3 arguments:
	// argv[0] command name
	// argv[1] src_graph_id
	// argv[2] dest_graph_id
	if(argc != 3) {
		return RedisModule_WrongArity(ctx);
	}

	// make a copy of arguments
	char *src  = rm_strdup(RedisModule_StringPtrLen(argv[1], NULL));
	char *dest = rm_strdup(RedisModule_StringPtrLen(argv[2], NULL));

	// block the client
    RedisModuleBlockedClient *bc = RedisModule_BlockClient(ctx, NULL, NULL,
			NULL, 0);

	// create command context
	GraphCopyContext *context = rm_malloc(sizeof(GraphCopyContext));

	context->bc   = bc;
	context->src  = src;
	context->dest = dest;

	// add copy command to thread-pool
	// to avoid hogging our only write thread, we'll run the GRAPH.COPY command
	// on a READ thread, usually there are multiple READ threads available
	if(ThreadPools_AddWorkReader(_Graph_Copy, context, false)
			== THPOOL_QUEUE_FULL) {
		// thread-pool queue is full, back off
		RedisModule_ReplyWithError(ctx, "Max pending queries exceeded");
	}

	return REDISMODULE_OK;
}

