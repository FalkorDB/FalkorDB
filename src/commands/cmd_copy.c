/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../cron/cron.h"
#include "../util/uuid.h"
#include "../redismodule.h"
#include "../graph/graphcontext.h"
#include "../serializers/serializer_io.h"
#include "../serializers/encoder/v14/encode_v14.h"
#include "../serializers/decoders/current/v14/decode_v14.h"

#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

extern RedisModuleType *GraphContextRedisModuleType;

// GRAPH.COPY command context
typedef struct {
	const char *src;             // src graph id
	const char *dest;            // dest graph id
	char *path;                  // path to dumped graph on disk
	RedisModuleString *rm_src;   // redismodule string src
	RedisModuleString *rm_dest;  // redismodule string dest

	RedisModuleBlockedClient *bc;  // blocked client
} GraphCopyContext;

static char *_temp_file(void) {
	// /tmp/<UUID>.dump
	char *uuid = UUID_New();
	char *path;
	asprintf(&path, "/tmp/%s.dump", uuid);
	rm_free(uuid);

	return path;
}

GraphCopyContext *GraphCopyContext_New
(
	RedisModuleBlockedClient *bc,
	RedisModuleString *src,
	RedisModuleString *dest
) {
	ASSERT(bc   != NULL);
	ASSERT(src  != NULL);
	ASSERT(dest != NULL);

	GraphCopyContext *ctx = rm_malloc(sizeof(GraphCopyContext));

	ctx->bc      = bc;
	ctx->path    = _temp_file();
	ctx->rm_src  = src;
	ctx->rm_dest = dest;
	ctx->src     = RedisModule_StringPtrLen(src, NULL);
	ctx->dest    = RedisModule_StringPtrLen(dest, NULL);

	return ctx;
}

void GraphCopyContext_Free
(
	GraphCopyContext *copy_ctx
) {
	ASSERT(copy_ctx != NULL);

	// delete file in case it exists, no harm if file is missing
	remove(copy_ctx->path);

	free(copy_ctx->path);

	RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(copy_ctx->bc);
	RedisModule_FreeString(ctx, copy_ctx->rm_src);
	RedisModule_FreeString(ctx, copy_ctx->rm_dest);
	RedisModule_UnblockClient(copy_ctx->bc, NULL);
	RedisModule_FreeThreadSafeContext(ctx);

	rm_free(copy_ctx);
}

// encode graph to disk
// this function should run on a child process, giving us the guarantees:
// 1. the cloned graph wouldn't change
// 2. due to memory seperation we do not need to hold any locks
// 3. we're allowed to make modification to the graph e.g. rename
static int _encode_graph
(
	RedisModuleCtx *ctx,         // redis module context
	GraphContext *gc,            // graph to clone
	GraphCopyContext *copy_ctx   // graph copy context
) {
	// validations
	ASSERT(gc       != NULL);
	ASSERT(ctx      != NULL);
	ASSERT(copy_ctx != NULL);

	int res = 0;  // 0 indicates success

	// rename graph, needed by the decoding procedure
	// when the graph is decoded it is already holds the target name
	GraphContext_Rename(ctx, gc, copy_ctx->dest);

	//--------------------------------------------------------------------------
	// serialize graph to file
	//--------------------------------------------------------------------------

	// open dump file
	// write only, create if missing, truncate if exists
	// grant READ access to group (0644)
	int fd = open(copy_ctx->path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
	if(fd == -1) {
		// failed to open file
		res = 1;  // indicate error
		goto cleanup;
	}

	// create serializer
	SerializerIO io = SerializerIO_FromPipe(fd);
	ASSERT(io != NULL);

	// encode graph to disk
	RdbSaveGraph_v14(io, gc);

cleanup:

	// free serializer
	if(io != NULL) SerializerIO_Free(&io);

	// close file
	if(fd != -1) close(fd);

	// all done, no errors
	return res;
}

// load graph from file
static void _LoadGraphFromFile
(
	void *pdata  // graph copy context
) {
	ASSERT(pdata != NULL);

	GraphCopyContext *copy_ctx = (GraphCopyContext*)pdata;

	SerializerIO   io   = NULL;
	RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(copy_ctx->bc);

	//--------------------------------------------------------------------------
	// decode graph from disk
	//--------------------------------------------------------------------------

	// open file
	int fd = open(copy_ctx->path, O_RDONLY);
	if(fd == -1) {
		RedisModule_ReplyWithError(ctx, "copy failed");
		goto cleanup;
	}

	// create serializer ontop of file descriptor
	io = SerializerIO_FromPipe(fd);
	ASSERT(io != NULL);

	// decode graph from io
	GraphContext *gc = RdbLoadGraphContext_v14(io, copy_ctx->rm_dest);
	ASSERT(gc != NULL);

	// TODO: should we decrase gc ref count?

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

		// release GIL
		RedisModule_ThreadSafeContextUnlock(ctx);

		// register graph context for BGSave
		GraphContext_RegisterWithModule(gc);

		RedisModule_ReplyWithCString(ctx, "OK");
	}

cleanup:

	// free serializer
	if(io != NULL) SerializerIO_Free(&io);

	// close file descriptor
	if(fd != -1) close(fd);

	// free copy context
	GraphCopyContext_Free(copy_ctx);

	RedisModule_FreeThreadSafeContext(ctx);
}

// fork done handler
// this function runs on Redis main thread
static void _ForkDoneHandler
(
	int exitcode,    // fork return code
	int bysignal,    // how did fork terminated
	void *user_data  // private data (GraphCopyContext*)
) {
	ASSERT(user_data != NULL);

	if(exitcode == 1 || bysignal != 0) {
		GraphCopyContext *copy_ctx = (GraphCopyContext*)user_data;
		RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(copy_ctx->bc);
		RedisModule_ReplyWithError(ctx, "copy failed");
		// fork failed
		GraphCopyContext_Free(copy_ctx);
		RedisModule_FreeThreadSafeContext(ctx);
		return;
	}

	// perform decoding on a different thread to avoid blocking Redis
	Cron_AddTask(0, _LoadGraphFromFile, NULL, user_data);
}

// implements GRAPH.COPY logic
// this function is expected to run on a cron thread
// avoiding blocking redis main thread while the graph is being copied
static void _Graph_Copy
(
	void *context  // graph copy context
) {
	ASSERT(context != NULL);

	GraphCopyContext *copy_ctx = (GraphCopyContext*)context;

	bool error = false;
	GraphContext *gc = NULL;

	RedisModuleString *rm_src    = copy_ctx->rm_src;
	RedisModuleString *rm_dest   = copy_ctx->rm_dest;
    RedisModuleBlockedClient *bc = copy_ctx->bc;
	RedisModuleCtx *ctx          = RedisModule_GetThreadSafeContext(bc);

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

	// child process will encode src graph to a file
	// parent process will decode cloned graph from file

	int pid = -1;
	while(pid == -1) {
		// acquire READ lock on gc
		// we do not want to fork while the graph is modified
		Graph_AcquireReadLock(gc->g);

		// try to fork
		pid = RedisModule_Fork(_ForkDoneHandler, copy_ctx);
		if(pid < 0) {
			// failed to fork! retry in a bit

			// release graph READ lock
			Graph_ReleaseLock(gc->g);

			// go to sleep for 1.0ms
			struct timespec sleep_time;
			sleep_time.tv_sec = 0;
			sleep_time.tv_nsec = 1000000;
			nanosleep(&sleep_time, NULL);
		} else if(pid == 0) {
			// managed to fork, in child process
			// encode graph to disk
			int res = _encode_graph(ctx, gc, copy_ctx);
			// all done, Redis require us to call 'RedisModule_ExitFromChild'
			RedisModule_ExitFromChild(res);
			return;
		}
	}

	// release READ lock
	Graph_ReleaseLock(gc->g);

	// replicate command
	// TODO: handle cases where GRAPH.COPY fails on the master
	RedisModule_ThreadSafeContextLock(ctx);
	RedisModule_ReplicateVerbatim(ctx);
	RedisModule_ThreadSafeContextUnlock(ctx);

	// clean up
cleanup:

	// decrease src graph ref-count
	if(gc != NULL) {
		GraphContext_DecreaseRefCount(gc);
	}

	// reply "OK" if no error was encountered
	if(error) {
		// free command context only in the case of an error
		// otherwise the fork callback is responsible for freeing this context
		GraphCopyContext_Free(copy_ctx);
	}

	RedisModule_FreeThreadSafeContext(ctx);
}

// clone a graph
// function executes on Redis main thread
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

