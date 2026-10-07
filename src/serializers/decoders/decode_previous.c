/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "decode_graph.h"
#include "decode_previous.h"
#include "prev/decoders.h"

GraphContext *Decode_Previous
(
	RedisModuleIO *rdb,
	int encver
) {
	SerializerIO io   = NULL;
	GraphContext *ctx = NULL;

	switch(encver) {
		case 10:
			ctx = RdbLoadGraphContext_v10(rdb);
			break;

		case 11:
			ctx = RdbLoadGraphContext_v11(rdb);
			break;

		case 12:
			ctx = RdbLoadGraphContext_v12(rdb);
			break;

		case 13:
			ctx = RdbLoadGraphContext_v13(rdb);
			break;

		case 14: {
			io = SerializerIO_FromRedisModuleIO(rdb, false);
			const RedisModuleString *rm_key_name =
				RedisModule_GetKeyNameFromIO(rdb);
			ctx = RdbLoadGraphContext_v14(io, rm_key_name);
			break;
		}

		case 15: {
			io = SerializerIO_FromRedisModuleIO(rdb, false);
			const RedisModuleString *rm_key_name =
				RedisModule_GetKeyNameFromIO(rdb);
			ctx = RdbLoadGraphContext_v15(io, rm_key_name);
			break;
		}

		case 16: {
			io = SerializerIO_FromRedisModuleIO(rdb, false);
			const RedisModuleString *rm_key_name =
				RedisModule_GetKeyNameFromIO(rdb);
			ctx = RdbLoadGraphContext_v16(io, rm_key_name);
			break;
		}

		case 17: {
			io = SerializerIO_FromBufferedRedisModuleIO(rdb, false);
			const RedisModuleString *rm_key_name =
				RedisModule_GetKeyNameFromIO(rdb);
			ctx = RdbLoadGraphContext_v17(io, rm_key_name);
			break;
		}

		case 18: {
			io = SerializerIO_FromBufferedRedisModuleIO(rdb, false);
			const RedisModuleString *rm_key_name =
				RedisModule_GetKeyNameFromIO(rdb);
			ctx = RdbLoadGraphContext_v18(io, rm_key_name);
			break;
		}

		case 19: {
			io = SerializerIOv2_FromBufferedRedisModuleIO(rdb, false);
			const RedisModuleString *rm_key_name =
				RedisModule_GetKeyNameFromIO(rdb);
			ctx = RdbLoadGraphContext_v19(io, rm_key_name, false);
			break;
		}

		default:
			ASSERT(false && "attempted to read unsupported RedisGraph version from RDB file.");
			break;
	}

	// for SerializerIO-based decoders (v14+), detect a short read / IO error
	// before the serializer is torn down; v10-v13 read RedisModuleIO directly
	// and are best-effort (no graceful short-read handling)
	if(io != NULL) {
		bool io_error = SerializerIO_Error(io);
		if(io_error) {
			RdbLoadGraph_LogFailure(rdb, io);
		}
		SerializerIO_Free(&io);

		if(io_error) {
			// short read - abort the load; free the partial graph if it isn't
			// registered yet (created by this virtual key), otherwise leave it
			// for Redis to reconcile via the earlier keys' free callbacks
			if(ctx != NULL && GraphContext_RefCount(ctx) == 0) {
				GraphContext_Free(ctx);
			}
			return NULL;
		}
	}

	return ctx;
}

