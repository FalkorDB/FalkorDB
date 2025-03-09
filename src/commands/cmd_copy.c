/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// copying a graph is performed in a number of steps:
//
// 1. a cron task is created with the responsibility of creating a fork
//
// 2. the forked child process encodes the graph into a pipe
//
// 3. the parent process on the cron thread is decoding the graph as it being
//    encoded
//
// ┌──────────────────────┐
// │                      │                          ┌─────────────────────┐
// │    Parent process    │                          │                     │
// │    CRON thread       │                          │    Child process    │
// │                      │        ┌──────────┐      │                     │
// │        decode graph <├────────┼  Pipe    ├──────┼< encode graph       │
// │                      │        └──────────┘      │                     │
// └──────────────────────┘                          └─────────────────────┘


#include "RG.h"
#include "../cron/cron.h"
#include "../util/uuid.h"
#include "../redismodule.h"
#include "../graph/graphcontext.h"
#include "../serializers/serializer_io.h"
#include "../serializers/encoder/v16/encode_v16.h"
#include "../serializers/decoders/current/v16/decode_v16.h"

#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

extern RedisModuleType *GraphContextRedisModuleType;

// GRAPH.COPY command context
typedef struct {
	const char *src;               // src graph id
	const char *dest;              // dest graph id
	RedisModuleString *rm_src;     // redismodule string src
	RedisModuleString *rm_dest;    // redismodule string dest
	RedisModuleBlockedClient *bc;  // blocked client
	int pipe_fd[2];                // array holding pipe read and write ends
} GraphCopyContext;

// create a new graph copy context
static GraphCopyContext *GraphCopyContext_New
(
	RedisModuleBlockedClient *bc,  // blocked client
	RedisModuleString *src,        // src graph key name
	RedisModuleString *dest        // destination graph key name
) {
	ASSERT(bc   != NULL);
	ASSERT(src  != NULL);
	ASSERT(dest != NULL);

	GraphCopyContext *ctx = rm_malloc(sizeof(GraphCopyContext));

	// create a pipe through which the graph will be encoded (write end)
	// and decoded (read end)
	if(pipe(ctx->pipe_fd) == -1) {
		rm_free(ctx);
		return NULL;
	}

	ctx->bc      = bc;
	ctx->src     = RedisModule_StringPtrLen(src,  NULL);
	ctx->dest    = RedisModule_StringPtrLen(dest, NULL);
	ctx->rm_src  = src;
	ctx->rm_dest = dest;

	return ctx;
}

// free graph copy context
static void GraphCopyContext_Free
(
	GraphCopyContext *copy_ctx  // context to free
) {
	ASSERT(copy_ctx != NULL);

	//--------------------------------------------------------------------------
	// free pipe
	//--------------------------------------------------------------------------

	// close the read end
	if(copy_ctx->pipe_fd[0] >= 0) {
		close(copy_ctx->pipe_fd[0]);
	}

	// close the write end
	if(copy_ctx->pipe_fd[1] >= 0) {
		close(copy_ctx->pipe_fd[1]);
	}

	RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(copy_ctx->bc);
	RedisModule_FreeString(ctx, copy_ctx->rm_src);
	RedisModule_FreeString(ctx, copy_ctx->rm_dest);
	RedisModule_UnblockClient(copy_ctx->bc, NULL);
	RedisModule_FreeThreadSafeContext(ctx);

	rm_free(copy_ctx);
}

// encode graph to pipe
// this function should run on a child process, giving us the guarantees:
// 1. the cloned graph wouldn't change
// 2. due to memory seperation we do not need to hold any locks
// 3. we're allowed to make modification to the graph e.g. rename
static int encode_graph
(
	RedisModuleCtx *ctx,         // redis module context
	GraphContext *gc,            // graph to clone
	GraphCopyContext *copy_ctx,  // graph copy context
	FILE *f                      // pipe write end
) {
	// validations
	ASSERT(f        != NULL);
	ASSERT(gc       != NULL);
	ASSERT(ctx      != NULL);
	ASSERT(copy_ctx != NULL);

	int          res = 0;  // 0 indicates success
	SerializerIO io  = NULL;

	// rename graph, needed by the decoding procedure
	// when the graph is decoded it is already holds the target name
	GraphContext_Rename(ctx, gc, copy_ctx->dest);

	//--------------------------------------------------------------------------
	// serialize graph to stream
	//--------------------------------------------------------------------------

	// create serializer
	io = SerializerIO_FromStream(f);
	ASSERT(io != NULL);

	RdbSaveGraph_latest(io, gc);

	// free serializer
	SerializerIO_Free(&io);

	// all done, no errors
	return res;
}

// load graph from pipe
static void LoadGraphFromFile
(
	void *pdata,  // graph copy context
	FILE *f       // pipe read end
) {
	ASSERT(f     != NULL);
	ASSERT(pdata != NULL);

	GraphCopyContext *copy_ctx = (GraphCopyContext*)pdata;

	SerializerIO   io   = NULL;  // graph decode stream
	RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(copy_ctx->bc);

	// create serializer ontop of file descriptor
	io = SerializerIO_FromStream(f);
	ASSERT(io != NULL);

	// make sure each byte read off the stream is saved into buffer
	size_t size  = 0;     // buffer size
	char *buffer = NULL;  // read data off stream
	SerializerIO_SaveDataToBuffer(io, &buffer, &size);

	// decode graph from stream
	GraphContext *gc = RdbLoadGraphContext_latest(io, copy_ctx->rm_dest);
	ASSERT(gc != NULL);

	// free serializer, flush buffer
	SerializerIO_Free(&io);
	ASSERT(size   >  0);
	ASSERT(buffer != NULL);

	//--------------------------------------------------------------------------
	// add cloned graph to keyspace
	//--------------------------------------------------------------------------

	RedisModule_ThreadSafeContextLock(ctx); // lock GIL

	// make sure dest key does not exists
	RedisModuleKey *key =
		RedisModule_OpenKey(ctx, copy_ctx->rm_dest, REDISMODULE_READ);
	int key_type = RedisModule_KeyType(key);

	RedisModule_CloseKey(key);

	if(key_type != REDISMODULE_KEYTYPE_EMPTY) {
		// error!
		RedisModule_ThreadSafeContextUnlock(ctx);  // release GIL

		// free graph
		GraphContext_DecreaseRefCount(gc);

		RedisModule_ReplyWithError(ctx, "copy failed");
	} else {
		// create key
		key = RedisModule_OpenKey(ctx, copy_ctx->rm_dest, REDISMODULE_WRITE);

		// set value in key
		RedisModule_ModuleTypeSetValue(key, GraphContextRedisModuleType, gc);

		RedisModule_CloseKey(key);

		// replicate graph
		// GRAPH.RESTORE dest <payload>
		RedisModule_Replicate(ctx, "GRAPH.RESTORE", "cb", copy_ctx->dest,
				buffer, size);

		RedisModule_ThreadSafeContextUnlock(ctx);  // release GIL

		// register graph context for BGSave
		GraphContext_RegisterWithModule(gc);

		RedisModule_ReplyWithCString(ctx, "OK");
	}

	RedisModule_FreeThreadSafeContext(ctx);

	free(buffer);
}

// implements GRAPH.COPY logic
// this function is expected to run on a cron thread
// avoiding blocking redis main thread while trying to create a fork
static void _Graph_Copy
(
	void *context  // graph copy context
) {
	ASSERT(context != NULL);

	GraphCopyContext *copy_ctx = (GraphCopyContext*)context;

	GraphContext             *gc      = NULL;
	RedisModuleBlockedClient *bc      = copy_ctx->bc;
	RedisModuleString        *rm_src  = copy_ctx->rm_src;
	RedisModuleString        *rm_dest = copy_ctx->rm_dest;
	RedisModuleCtx           *ctx     = RedisModule_GetThreadSafeContext(bc);

	//--------------------------------------------------------------------------
	// validations
	//--------------------------------------------------------------------------

	// lock GIL
	RedisModule_ThreadSafeContextLock(ctx);

	// make sure dest key does not exists
	RedisModuleKey *dest_key =
		RedisModule_OpenKey(ctx, rm_dest, REDISMODULE_READ);
	int dest_key_type = RedisModule_KeyType(dest_key);
	RedisModule_CloseKey(dest_key);

	// make sure src key is a graph
	gc = GraphContext_Retrieve(ctx, rm_src, true, false);

	// release GIL
	RedisModule_ThreadSafeContextUnlock(ctx);

	// dest key shouldn't exists
	if(dest_key_type != REDISMODULE_KEYTYPE_EMPTY) {
		// destination key already exists, abort
		RedisModule_ReplyWithError(ctx, "destination key already exists");
		goto cleanup;
	}

	// src key should be a graph
	if(gc == NULL) {
		// src graph is missing, abort
		// error alreay omitted by 'GraphContext_Retrieve'
		goto cleanup;
	}

	//--------------------------------------------------------------------------
	// fork process
	//--------------------------------------------------------------------------

	// child process will encode src graph to a file
	// parent process will decode cloned graph from file

	uint backoff_ms  = 5;  // backoff time in ms
	int fork_retries = 5;  // number of retries

	int pid = -1;
	while(pid == -1 && fork_retries-- > 0) {
		// try to fork
		RedisModule_ThreadSafeContextLock(ctx); // lock GIL
		Graph_AcquireWriteLock(gc->g);          // lock graph for write

		pid = RedisModule_Fork(NULL, copy_ctx);

		if(pid < 0) {
			// failed to fork
			// release locks
			Graph_ReleaseLock(gc->g);
			RedisModule_ThreadSafeContextUnlock(ctx); // release GIL

			// failed to fork! retry in a bit
			// backoff time is doubled on each retry
			backoff_ms *= 2;
			struct timespec sleep_time;
			sleep_time.tv_sec  = backoff_ms / 1000;              // seconds
			sleep_time.tv_nsec = (backoff_ms % 1000) * 1000000;  // nano-seconds
			nanosleep(&sleep_time, NULL);

			// log warning
			RedisModule_Log(ctx, "warning",
					"GRAPH.COPY failed to fork, retrying in %d ms", backoff_ms);
		} else if(pid == 0) {
			//------------------------------------------------------------------
			// child process
			//------------------------------------------------------------------

			close(copy_ctx->pipe_fd[0]);  // close unused read-end
			copy_ctx->pipe_fd[0] = -1;

			// convert the write-end of the pipe to a FILE* stream
			FILE *write_fp = fdopen(copy_ctx->pipe_fd[1], "wb");
			ASSERT(write_fp != NULL);

			// encode graph
			int res = encode_graph(ctx, gc, copy_ctx, write_fp);

			// close file handle which in turn closes PIPE write end
			fclose(write_fp);
			copy_ctx->pipe_fd[1] = -1;


			// all done, Redis require us to call 'RedisModule_ExitFromChild'
			RedisModule_ExitFromChild(res);
		} else {
			//------------------------------------------------------------------
			// parent process
			//------------------------------------------------------------------

			// release locks
			Graph_ReleaseLock(gc->g);
			RedisModule_ThreadSafeContextUnlock(ctx); // release GIL

			close(copy_ctx->pipe_fd[1]); // close unused write-end
			copy_ctx->pipe_fd[1] = -1;

			// convert the read-end of the pipe to a FILE* stream
			FILE *read_fp = fdopen(copy_ctx->pipe_fd[0], "r");
			ASSERT(read_fp != NULL);

			LoadGraphFromFile(context, read_fp);

			// close file handle which in turn closes PIPE read end
			fclose(read_fp);
			copy_ctx->pipe_fd[0] = -1;
		}
	}

	if(fork_retries < 0) {
		// failed to fork
		RedisModule_ReplyWithError(ctx, "GRAPH.COPY aborted, failed to fork");
	}

	// clean up
cleanup:

	// decrease src graph ref-count
	if(gc != NULL) {
		GraphContext_DecreaseRefCount(gc);
	}

	// free command context
	GraphCopyContext_Free(copy_ctx);

	RedisModule_FreeThreadSafeContext(ctx);
}

// clone a graph
// this function executes on Redis main thread
//
// usage:
// GRAPH.COPY <src_graph> <dest_graph>
int Graph_Copy
(
	RedisModuleCtx *ctx,       // redis module context
	RedisModuleString **argv,  // command argument
	int argc                   // number of argument
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

	// block the client
	// TODO: determine if replica should use blocked client
	RedisModuleBlockedClient *bc = RedisModule_BlockClient(ctx, NULL, NULL,
			NULL, 0);

	// retain arguments
	RedisModule_RetainString(ctx, argv[1]);
	RedisModule_RetainString(ctx, argv[2]);

	// create command context
	GraphCopyContext *context = GraphCopyContext_New(bc, argv[1], argv[2]);

	// add GRAPH.COPY as a cron task to run as soon as possible
	Cron_AddTask(0, _Graph_Copy, NULL, context);


	return REDISMODULE_OK;
}

