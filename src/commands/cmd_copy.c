/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../redismodule.h"
#include "../graph/graphcontext.h"
#include "../serializers/serializer_io.h"
#include "../serializers/encoder/v14/encode_v14.h"
#include "../serializers/decoders/current/v14/decode_v14.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static void child
(
	RedisModuleCtx *ctx,
	int pipe_write_end,
	GraphContext *src_graph,
	const char *dest_graph_name
) {
	ASSERT(ctx             != NULL);
	ASSERT(src_graph       != NULL);
	ASSERT(dest_graph_name != NULL);

	// rename src graph
	GraphContext_Rename(ctx, src_graph, dest_graph_name);

	//--------------------------------------------------------------------------
	// encode graph to stream
	//--------------------------------------------------------------------------

	SerializerIO writer = SerializerIO_FromPipe(pipe_write_end);

	RdbSaveGraph_v14(writer, src_graph);

	SerializerIO_Free(&writer);
}

static void parent
(
	RedisModuleCtx *ctx,
	int pipe_read_end,
	RedisModuleString *dest_graph_name
) {
	//--------------------------------------------------------------------------
	// read src graph from stream
	//--------------------------------------------------------------------------

	// create read serializer
	SerializerIO reader = SerializerIO_FromPipe(pipe_read_end);

	// decode graph
	GraphContext *clone = RdbLoadGraphContext_v14(reader, dest_graph_name);

	// free reader
	SerializerIO_Free(&reader);

	// save graph to keyspace
}

int Graph_Copy
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc
) {
	ASSERT(ctx != NULL);

	if(argc != 3) {
		return RedisModule_WrongArity(ctx);
	}

	const char *src  = RedisModule_StringPtrLen(argv[1], NULL);
	const char *dest = RedisModule_StringPtrLen(argv[2], NULL);

	printf("src: %s\n", src);
	printf("dest: %s\n", dest);

	// TODO: lock GIL

	// make sure dest graph does not exists
	GraphContext *dest_graph = GraphContext_Retrieve(ctx, argv[2], true, false);
	if(dest_graph != NULL) {
		GraphContext_DecreaseRefCount(dest_graph);
		//RedisModule_ReplyWithError(EMSG_GRAPH_EXISTS, dest);
		printf(EMSG_GRAPH_EXISTS, dest);
		return REDISMODULE_OK;
	}

	// make sure src graph exists
	GraphContext *src_graph = GraphContext_Retrieve(ctx, argv[1], true, false);
	if(src_graph == NULL) {
		//RedisModule_ReplyWithError(EMSG_NON_GRAPH_KEY, src);
		printf(EMSG_NON_GRAPH_KEY, src);
		return REDISMODULE_OK;
	}

	// create an empty dest graph
	GraphContext *clone = GraphContext_Retrieve(ctx, argv[2], true, true);
	GraphContext_DecreaseRefCount(clone);

	// create pipe
	// pipe[0] read end
	// pipe[1] write end
	int pipefd[2];
	if(pipe(pipefd) == -1) {
		// failed to create pipe
		goto cleanup;
	}

	//--------------------------------------------------------------------------
	// fork process
	//--------------------------------------------------------------------------

	// child process will encode src graph to stream
	// parent process will decode cloned graph from stream

	//int pid = RedisModule_Fork(RedisModuleForkDoneHandler cb, void *user_data);
	int pid = RedisModule_Fork(NULL, NULL);
	if(pid == 0) {
		// child process
		child(ctx, pipefd[1], src_graph, dest);

		// all done, Redis require us to call 'RedisModule_ExitFromChild'
		RedisModule_ExitFromChild(0);
		return REDISMODULE_OK;
	} else {
		// parent process
		parent(ctx, pipefd[0], argv[2]);
	}

	// close pipe
	close(pipefd[0]);
	close(pipefd[1]);

	// replicate command
	RedisModule_ReplicateVerbatim(ctx);

	// clean up
cleanup:
	GraphContext_DecreaseRefCount(src_graph);

	RedisModule_ReplyWithCString(ctx, "OK");

	return REDISMODULE_OK;
}

