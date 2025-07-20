/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// copying a graph is performed in a number of steps:
//
// 1. a cron task is created with the responsibility of creating a fork
//
// 2. the forked child process encodes the graph into a temporary file
//    once done the child exists and a callback is invoked on Redis main thread
//
// 3. a second cron task is created with the responsibility of decoding the
//    dumped file and creating a new graph key
//
//
//
// ┌────────────────┐                  ┌────────────────┐
// │                │                  │                │
// │   Cron Task    │                  │     Child      │
// │                │                  │                │
// │                │                  │                │       ┌────────┐
// │                │                  │                │       │        │
// │                │                  │                │       │ Dump   │
// │      Fork      ├─────────────────►│                ├──────►│ Graph  │
// │                │                  │                │       │        │
// │                │                  │                │       │        │
// └────────────────┘                  └───────┬────────┘       │        │
//                                             │                │        │
//                                             │                └────────┘
//                                             │                    ▲
// ┌────────────────┐                          │                    │
// │                │                          │                    │
// │  Main thread   │      Done callback       │                    │
// │                │◄─────────────────────────┘                    │
// │                │                                               │
// │                │                                               │
// │                │                                               │
// └────────┬───────┘                                               │
//          │                                                       │
//          │                                                       │
//          ▼                                                       │
// ┌────────────────┐                                               │
// │                │                                               │
// │   Cron Task    │                                               │
// │                │                                               │
// │                │                                               │
// │  Decode Graph  ├───────────────────────────────────────────────┘
// │                │
// │                │
// │                │
// │                │
// └────────────────┘


#include "RG.h"
#include "../cron/cron.h"
#include "../util/uuid.h"
#include "../redismodule.h"
#include "../graph/graphcontext.h"
#include "../serializers/serializer_io.h"
#include "../serializers/encoder/encode_graph.h"
#include "../serializers/decoders/decode_graph.h"

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
	RedisModuleCtx *rm_ctx;        // redis module context
	RedisModuleBlockedClient *bc;  // blocked client
	int pipefd[2];                 // pipe
} GraphCopyContext;

// create a new graph copy context
static GraphCopyContext *GraphCopyContext_New
(
	RedisModuleCtx *rm_ctx,        // redis module context
	RedisModuleBlockedClient *bc,  // blocked client
	RedisModuleString *src,        // src graph key name
	RedisModuleString *dest        // destination graph key name
) {
	ASSERT(src  != NULL);
	ASSERT(dest != NULL);
	ASSERT((rm_ctx != NULL && bc == NULL) || (rm_ctx == NULL && bc != NULL));

	GraphCopyContext *ctx = rm_calloc(1, sizeof(GraphCopyContext));

	ctx->bc      = bc;
	ctx->src     = RedisModule_StringPtrLen(src,  NULL);
	ctx->dest    = RedisModule_StringPtrLen(dest, NULL);
	ctx->rm_ctx  = rm_ctx;
	ctx->rm_src  = src;
	ctx->rm_dest = dest;

	// create pipe
	pipe(ctx->pipefd);

	return ctx;
}

// free graph copy context
static void GraphCopyContext_Free
(
	GraphCopyContext *copy_ctx  // context to free
) {
	ASSERT(copy_ctx != NULL);

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
static int EncodeGraphToPipe
(
	RedisModuleCtx *ctx,         // redis module context
	GraphContext *gc,            // graph to clone
	GraphCopyContext *copy_ctx   // graph copy context
) {
	// validations
	ASSERT(gc       != NULL);
	ASSERT(ctx      != NULL);
	ASSERT(copy_ctx != NULL);

	int res = 1;  // 1 indicates failure
	SerializerIO io  = NULL;

	//--------------------------------------------------------------------------
	// serialize graph to pipe
	//--------------------------------------------------------------------------

	// create serializer
	FILE *pipe_write = fdopen(copy_ctx->pipefd[1], "wb");
	if(pipe_write == NULL) {
		goto cleanup;
	}

	io = SerializerIO_FromStream(pipe_write, true);

	// encode graph to pipe
	RedisModule_Log(NULL, REDISMODULE_LOGLEVEL_NOTICE, "dump graph: %s",
			copy_ctx->src);

	// encode graph
	RdbSaveGraph_latest(io, gc);

	res = 0;  // 0 indicates success

cleanup:

	// free serializer
	if(io != NULL) SerializerIO_Free(&io);

	// close & flush FILE
	if(pipe_write != NULL) {
		fflush(pipe_write);
		fclose(pipe_write);
	}

	// close write end of the pipe
	close(copy_ctx->pipefd[1]);

	// all done, no errors
	return res;
}

// load graph from file
static void DecodeGraphFromPipe
(
	GraphCopyContext *copy_ctx  // graph copy context
) {
	ASSERT(copy_ctx != NULL);

	RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(copy_ctx->bc);

	FILE *fp = fdopen(copy_ctx->pipefd[0], "rb");
	ASSERT(fp);

	SerializerIO io = SerializerIO_FromStream(fp, false);
	ASSERT(io != NULL);

	// decode graph from pipe
	RedisModule_Log(ctx, REDISMODULE_LOGLEVEL_NOTICE, "Decoding graph: %s",
			copy_ctx->dest);

	// TODO: better to create an empty gc here and pass it to SerializerLoadGraph
	GraphContext *gc = SerializerLoadGraph(io, copy_ctx->dest,
			GRAPH_ENCODING_LATEST_V);

	SerializerIO_Free(&io);

	// close FILE*
	fclose(fp);

	// close read end of pipe
	close(copy_ctx->pipefd[0]);

	if(gc == NULL) {
		// failed to load graph
		RedisModule_ReplyWithError(ctx, "copy failed");
		return;
	}

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
		RedisModule_ReplyWithError(ctx, "copy failed");
	} else {
		// create key
		key = RedisModule_OpenKey(ctx, copy_ctx->rm_dest, REDISMODULE_WRITE);

		// set value in key
		RedisModule_ModuleTypeSetValue(key, GraphContextRedisModuleType, gc);

		RedisModule_CloseKey(key);

		// replicate graph copy command

		int replicated = RedisModule_Replicate(ctx, "GRAPH.COPY", "ss",
				copy_ctx->rm_src, copy_ctx->rm_dest);
		ASSERT(replicated == REDISMODULE_OK);

		// register graph context for BGSave
		GraphContext_RegisterWithModule(gc);

		RedisModule_ReplyWithCString(ctx, "OK");
	}

	RedisModule_ThreadSafeContextUnlock(ctx);  // release GIL

	RedisModule_FreeThreadSafeContext(ctx);  // free thread safe context

	// decrease copy ref count
	// copy will be freed only if it wasn't registered
	GraphContext_DecreaseRefCount(gc);
}

// implements GRAPH.COPY logic
// this function is expected to run on a cron thread
static void _Graph_Copy
(
	void *context  // graph copy context
) {
	ASSERT(context != NULL);

	GraphCopyContext *copy_ctx = (GraphCopyContext*)context;

	bool error = false;
	GraphContext *gc = NULL;

	RedisModuleBlockedClient *bc = copy_ctx->bc;
	RedisModuleString *rm_src  = copy_ctx->rm_src;
	RedisModuleString *rm_dest = copy_ctx->rm_dest;

	RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(bc);

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
		error = true;
		RedisModule_ReplyWithError(ctx, "destination key already exists");
		goto cleanup;
	}

	// src key should be a graph
	if(gc == NULL) {
		// src graph is missing, abort
		error = true;
		// error alreay omitted by 'GraphContext_Retrieve'
		goto cleanup;
	}

	//--------------------------------------------------------------------------
	// fork process
	//--------------------------------------------------------------------------

	// as part of fork preparation we don't want any pending changes
	// acquire READ lock on gc
	Graph_AcquireReadLock(gc->g);

	// make sure all matrices do not contain pending changes
	// we do not want to risk forking when some matrix is in mid flush
	Graph_ApplyAllPending(gc->g, false);

	// release graph READ lock
	Graph_ReleaseLock(gc->g);

	// child process will encode src graph to a pipe
	// parent process will decode cloned graph from pipe

	int pid   = -1;  // fork pid
	int tries = 5;   // number of tries

	while(pid == -1 && tries-- > 0) {
		// try to fork
		RedisModule_ThreadSafeContextLock(ctx); // lock GIL

		// acquire READ lock on gc
		// we do not want to fork while the graph is modified
		// might be redundant, see: GraphContext_LockForCommit
		Graph_AcquireReadLock(gc->g);

		// make sure all matrices do not contain pending changes
		// we do not want to risk forking when some matrix is in mid flush
		Graph_ApplyAllPending(gc->g, false);

		pid = RedisModule_Fork(NULL, copy_ctx);

		// release graph READ lock
		Graph_ReleaseLock(gc->g);

		// release GIL
		RedisModule_ThreadSafeContextUnlock(ctx);

		if(pid < 0) {
			// failed to fork! retry in a bit
			// go to sleep for 5.0ms
			struct timespec sleep_time;
			sleep_time.tv_sec = 0;
			sleep_time.tv_nsec = 5000000;
			nanosleep(&sleep_time, NULL);
		} else if(pid == 0) {
			// managed to fork, in child process

			// close read end of the pipe
			close(copy_ctx->pipefd[0]);

			// encode graph to disk
			int res = EncodeGraphToPipe(ctx, gc, copy_ctx);

			// all done, Redis require us to call 'RedisModule_ExitFromChild'
			RedisModule_ExitFromChild(res);
			return;
		} else {
			// close write end of pipe
			close(copy_ctx->pipefd[1]);

			// decode graph from pipe
			DecodeGraphFromPipe(copy_ctx);
		}
	}

	// clean up
cleanup:

	// decrease src graph ref-count
	if(gc != NULL) {
		GraphContext_DecreaseRefCount(gc);
	}

	RedisModule_FreeThreadSafeContext(ctx);

	// free copy context
	GraphCopyContext_Free(copy_ctx);
}

// implements GRAPH.COPY logic
// this function is expected to run on either Redis main thread
// in the case of AOF loading otherwise we should be on a cron thread
//static void __Graph_Copy
//(
//	void *context  // graph copy context
//) {
//	ASSERT(context != NULL);
//
//	GraphCopyContext *copy_ctx = (GraphCopyContext*)context;
//
//	bool error = false;
//
//	GraphContext      *gc      = NULL;
//	RedisModuleCtx    *ctx     = copy_ctx->rm_ctx;
//	RedisModuleString *rm_src  = copy_ctx->rm_src;
//	RedisModuleString *rm_dest = copy_ctx->rm_dest;
//
//	//--------------------------------------------------------------------------
//	// validations
//	//--------------------------------------------------------------------------
//
//	// make sure dest key does not exists
//	RedisModuleKey *dest_key =
//		RedisModule_OpenKey(ctx, rm_dest, REDISMODULE_READ);
//	int dest_key_type = RedisModule_KeyType(dest_key);
//	RedisModule_CloseKey(dest_key);
//
//	// make sure src key is a graph
//	gc = GraphContext_Retrieve(ctx, rm_src, true, false);
//
//	// dest key shouldn't exists
//	if(dest_key_type != REDISMODULE_KEYTYPE_EMPTY) {
//		// destination key already exists, abort
//		error = true;
//		RedisModule_ReplyWithError(ctx, "destination key already exists");
//		goto cleanup;
//	}
//
//	// src key should be a graph
//	if(gc == NULL) {
//		// src graph is missing, abort
//		error = true;
//		// error alreay omitted by 'GraphContext_Retrieve'
//		goto cleanup;
//	}
//
//	// encode graph to pipe on a worker thread
//	int res = encode_graph(ctx, gc, copy_ctx);
//
//	// perform decoding
//	LoadGraphFromFile(copy_ctx);
//
//	// clean up
//cleanup:
//
//	// decrease src graph ref-count
//	if(gc != NULL) {
//		GraphContext_DecreaseRefCount(gc);
//	}
//
//	if(error) {
//		// free command context only in the case of an error
//		// otherwise the fork callback is responsible for freeing this context
//		GraphCopyContext_Free(copy_ctx);
//	}
//}

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

	RedisModuleString *src_graph_id = argv[1];
	RedisModuleString *dst_graph_id = argv[2];

	// retain arguments
	RedisModule_RetainString(ctx, src_graph_id);
	RedisModule_RetainString(ctx, dst_graph_id);

	int cmd_flags = RedisModule_GetContextFlags(ctx);

	// block the client
	GraphCopyContext *cpy_ctx;
	if(cmd_flags & REDISMODULE_CTX_FLAGS_LOADING) {
		cpy_ctx = GraphCopyContext_New(ctx, NULL, src_graph_id, dst_graph_id);
		//__Graph_Copy(cpy_ctx);
	} else {
		RedisModuleBlockedClient *bc = RedisModule_BlockClient(ctx, NULL, NULL,
				NULL, 0);
		cpy_ctx = GraphCopyContext_New(NULL, bc, src_graph_id, dst_graph_id);

		// add GRAPH.COPY as a cron task to run as soon as possible
		Cron_AddTask(0, _Graph_Copy, NULL, cpy_ctx);
	}

	return REDISMODULE_OK;
}

