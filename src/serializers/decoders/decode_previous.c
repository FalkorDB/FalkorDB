/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

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
			SerializerIO_Free(&io);
			break;
		}

		case 15: {
			io = SerializerIO_FromRedisModuleIO(rdb, false);
			const RedisModuleString *rm_key_name =
				RedisModule_GetKeyNameFromIO(rdb);
			ctx = RdbLoadGraphContext_v15(io, rm_key_name);
			SerializerIO_Free(&io);
			break;
		}

		case 16: {
			io = SerializerIO_FromRedisModuleIO(rdb, false);
			const RedisModuleString *rm_key_name =
				RedisModule_GetKeyNameFromIO(rdb);
			ctx = RdbLoadGraphContext_v16(io, rm_key_name);
			SerializerIO_Free(&io);
			break;
		}

		default:
			ASSERT(false && "attempted to read unsupported RedisGraph version from RDB file.");
			break;
	}

	return ctx;
}

