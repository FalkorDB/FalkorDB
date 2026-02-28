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
#include "../util/path_utils.h"
#include "../redismodule.h"
#include "../graph/graphcontext.h"
#include "../serializers/serializer_io.h"
#include "../serializers/encoder/v18/encode_v18.h"
#include "../serializers/decoders/current/v18/decode_v18.h"

#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>

extern RedisModuleType *GraphContextRedisModuleType;

// GRAPH.COPY command context
typedef struct {
	const char *src;               // src graph id
	const char *dest;              // dest graph id
	char *path;                    // path to dumped graph on disk
	RedisModuleString *rm_src;     // redismodule string src
	RedisModuleString *rm_dest;    // redismodule string dest
	RedisModuleBlockedClient *bc;  // blocked client
} GraphCopyContext;

// return a full path to a temporary dump file
// e.g. /tmp/<UUID>.dump
static char *_temp_file(void) {
	char *uuid = UUID_New();
	char *path;
	char full_path[PATH_MAX];
	const char *temp_folder = NULL;
	Config_Option_get(Config_TEMP_FOLDER, &temp_folder);

	// construct the full path
	snprintf(full_path, sizeof(full_path), "%s/%s.dump", temp_folder, uuid);

	if(!is_safe_path(temp_folder, full_path)) {
		// log file access
		RedisModule_Log(NULL, REDISMODULE_LOGLEVEL_WARNING,
				"attempt to access unauthorized path %s", full_path);
		rm_free(uuid);
		return NULL;
	}

	// allocate and copy the path
	path = rm_strdup(full_path);

	rm_free(uuid);

	return path;
}

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

	ctx->bc      = bc;
	ctx->path    = _temp_file();
	ctx->rm_src  = src;
	ctx->rm_dest = dest;
	ctx->src     = RedisModule_StringPtrLen(src, NULL);
	ctx->dest    = RedisModule_StringPtrLen(dest, NULL);

	if(ctx->path == NULL) {
		rm_free(ctx);
		return NULL;
	}

	return ctx;
}

// free graph copy context
static void GraphCopyContext_Free
(
	GraphCopyContext *copy_ctx  // context to free
) {
	ASSERT(copy_ctx != NULL);

	// delete file in case it exists, no harm if file is missing
	RedisModule_Log(NULL, REDISMODULE_LOGLEVEL_NOTICE,
			"deleting dumped graph file: %s", copy_ctx->path);
	remove(copy_ctx->path);

	rm_free(copy_ctx->path);

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
static int encode_graph
(
	RedisModuleCtx *ctx,         // redis module context
	GraphContext *gc,            // graph to clone
	GraphCopyContext *copy_ctx   // graph copy context
) {
	// validations
	ASSERT (gc       != NULL) ;
	ASSERT (ctx      != NULL) ;
	ASSERT (copy_ctx != NULL) ;

	int          res = 0 ;     // 0 indicates success
	FILE         *f  = NULL ;  // dump file handler
	SerializerIO io  = NULL ;  // graph IO serializer

	// rename graph, needed by the decoding procedure
	// when the graph is decoded it is already holds the target name
	GraphContext_Rename (ctx, gc, copy_ctx->dest) ;

	//--------------------------------------------------------------------------
	// serialize graph to file
	//--------------------------------------------------------------------------

	// open dump file
	// write only, create if missing, truncate if exists
	// grant READ access to group (0644)
	f = fopen (copy_ctx->path, "wb") ;
	if(f == NULL) {
		// failed to open file
		RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_WARNING,
				"GRAPH.COPY failed to open file: %s for writing",
				copy_ctx->path) ;

		// indicate error
		res = 1 ;
		goto cleanup ;
	}

	// create serializer
	io = SerializerIO_FromStream(f, true);
	ASSERT(io != NULL);

	// encode graph to disk
	RedisModule_Log(NULL, REDISMODULE_LOGLEVEL_NOTICE, "dump graph: %s to: %s",
			copy_ctx->src, copy_ctx->path);

	RdbSaveGraph_latest(io, gc);

cleanup:

	// free serializer
	if (io != NULL) {
		SerializerIO_Free (&io) ;
	}

	// close file
	if (f != NULL) {
		fclose (f) ;
	}

	// all done, no errors
	return res ;
}

// load graph from file
static void LoadGraphFromFile
(
	void *pdata  // graph copy context
) {
	ASSERT(pdata != NULL);

	GraphCopyContext *copy_ctx = (GraphCopyContext*)pdata;

	SerializerIO   io      = NULL;  // graph decode stream
	char           *buffer = NULL;  // dumped graph
	FILE           *stream = NULL;  // memory stream over buffer
	RedisModuleCtx *ctx    = RedisModule_GetThreadSafeContext(copy_ctx->bc);

	//--------------------------------------------------------------------------
	// decode graph from disk
	//--------------------------------------------------------------------------

	// open file
	FILE *f = fopen (copy_ctx->path, "rb") ;
	if (f == NULL) {
		RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_WARNING,
				"GRAPH.COPY failed to open graph file: %s for reading",
				copy_ctx->path) ;

		RedisModule_ReplyWithError (ctx, "copy failed") ;
		goto cleanup ;
	}

	//--------------------------------------------------------------------------
	// load dumped file to memory
	//--------------------------------------------------------------------------

	// seek to the end of the file
	fseek (f, 0, SEEK_END) ;

	// get current position, which is the size of the file
	long fileLength = ftell (f) ;

	// seek to the beginning of the file
	rewind (f) ;

	// allocate buffer to hold entire dumped graph
	buffer = rm_malloc (sizeof(char) * fileLength) ;

	// read file content into buffer
	size_t read = fread (buffer, fileLength, 1, f) ;
	assert (read == 1) ;

	fclose (f) ;  // close file

	//--------------------------------------------------------------------------
	// create memory stream
	//--------------------------------------------------------------------------

	stream = fmemopen (buffer, fileLength, "r") ;
	if (stream == NULL) {
		RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_WARNING,
				"GRAPH.COPY failed to open memory stream") ;

		RedisModule_ReplyWithError (ctx, "copy failed") ;
		goto cleanup ;
	}

	// create serializer ontop of file descriptor
	io = SerializerIO_FromStream(stream, false);
	ASSERT(io != NULL);

	// decode graph from io
	RedisModule_Log(NULL, REDISMODULE_LOGLEVEL_NOTICE,
			"Decoding graph: %s from: %s", copy_ctx->dest, copy_ctx->path);

	GraphContext *gc = RdbLoadGraphContext_latest(io, copy_ctx->rm_dest);
	ASSERT(gc != NULL);

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

		RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_WARNING,
				"GRAPH.COPY failed copied graph key: %s is already set",
				copy_ctx->dest) ;

		RedisModule_ReplyWithError(ctx, "copy failed");
	} else {
		// create key
		key = RedisModule_OpenKey(ctx, copy_ctx->rm_dest, REDISMODULE_WRITE);

		// set value in key
		RedisModule_ModuleTypeSetValue(key, GraphContextRedisModuleType, gc);

		RedisModule_CloseKey(key);

		// replicate graph
		// GRAPH.RESTORE dest <payload>
		RedisModule_Replicate (ctx, "GRAPH.RESTORE", "cb", copy_ctx->dest,
				buffer, fileLength) ;

		RedisModule_ThreadSafeContextUnlock (ctx) ;  // release GIL

		// register graph context for BGSave
		GraphContext_RegisterWithModule (gc) ;

		RedisModule_ReplyWithCString (ctx, "OK") ;
	}

cleanup:

	// free serializer
	if (io != NULL) {
		SerializerIO_Free (&io) ;
	}

	// close file descriptor
	if(stream != NULL) {
		fclose (stream) ;
	}

	// free buffer
	if (buffer != NULL) {
		rm_free (buffer) ;
	}

	// free copy context
	GraphCopyContext_Free (copy_ctx);

	RedisModule_FreeThreadSafeContext (ctx) ;
}

// fork done handler
// this function runs on Redis main thread
static void ForkDoneHandler
(
	int exitcode,    // fork return code
	int bysignal,    // how did fork terminated
	void *user_data  // private data (GraphCopyContext*)
) {
	ASSERT (user_data != NULL) ;

	// check fork exit code
	if (exitcode != 0 || bysignal != 0) {
		RedisModule_Log (NULL, REDISMODULE_LOGLEVEL_WARNING,
				"GRAPH.COPY fork indicate failure") ;

		GraphCopyContext *copy_ctx = (GraphCopyContext*)user_data ;
		RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext (copy_ctx->bc) ;
		RedisModule_ReplyWithError (ctx, "copy failed") ;

		// fork failed
		GraphCopyContext_Free (copy_ctx) ;
		RedisModule_FreeThreadSafeContext (ctx) ;
		return ;
	}

	// perform decoding on a different thread to avoid blocking Redis
	Cron_AddTask (0, LoadGraphFromFile, NULL, user_data) ;
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

	// acquire READ lock on gc
	// we do not want to fork while the graph is modified
	// might be redundant, see: GraphContext_LockForCommit
	Graph_AcquireReadLock (gc->g) ;
	Graph_ApplyAllPending (gc->g, false) ;  // flush all pending changes

	int pid = -1 ;
	while (pid == -1) {
		// try to fork
		pid = RedisModule_Fork (ForkDoneHandler, copy_ctx) ;
		if (pid < 0) {
			// failed to fork! retry in a bit
			// go to sleep for 5.0ms
			struct timespec sleep_time ;
			sleep_time.tv_sec = 0 ;
			sleep_time.tv_nsec = 5000000 ;
			nanosleep (&sleep_time, NULL) ;
		} else if (pid == 0) {
			// managed to fork, in child process
			// encode graph to disk
			int res = encode_graph (ctx, gc, copy_ctx) ;
			// all done, Redis require us to call 'RedisModule_ExitFromChild'
			RedisModule_ExitFromChild (res) ;
			return ;
		} else {
			// release graph READ lock
			Graph_ReleaseLock (gc->g) ;
		}
	}

	// clean up
cleanup:

	// decrease src graph ref-count
	if (gc != NULL) {
		GraphContext_DecreaseRefCount (gc);
	}

	if (error) {
		// free command context only in the case of an error
		// otherwise the fork callback is responsible for freeing this context
		GraphCopyContext_Free (copy_ctx) ;
	}

	RedisModule_FreeThreadSafeContext (ctx) ;
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
	RedisModuleBlockedClient *bc = RedisModule_BlockClient(ctx, NULL, NULL,
			NULL, 0);

	// retain arguments
	RedisModule_RetainString(ctx, argv[1]);
	RedisModule_RetainString(ctx, argv[2]);

	// create command context
	GraphCopyContext *context = GraphCopyContext_New(bc, argv[1], argv[2]);

	if(context == NULL) {
		RedisModule_FreeString(ctx, argv[1]);
		RedisModule_FreeString(ctx, argv[2]);
		RedisModule_UnblockClient(bc, NULL);
		return RedisModule_ReplyWithError(ctx, "Failed to create copy context");
	}

	// add GRAPH.COPY as a cron task to run as soon as possible
	Cron_AddTask(0, _Graph_Copy, NULL, context);

	return REDISMODULE_OK;
}

