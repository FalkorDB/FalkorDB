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
	switch(encver) {
		case 9:
			return RdbLoadGraphContext_v9(rdb);
		case 10:
			return RdbLoadGraphContext_v10(rdb);
		case 11:
			return RdbLoadGraphContext_v11(rdb);
		case 12:
			return RdbLoadGraphContext_v12(rdb);
		case 13:
			return RdbLoadGraphContext_v13(rdb);
		case 14: {
			SerializerIO io = SerializerIO_FromRedisModuleIO(rdb);
			const RedisModuleString *rm_key_name = RedisModule_GetKeyNameFromIO(rdb);
			return RdbLoadGraphContext_v14(io, rm_key_name);
		}
		default:
			ASSERT(false && "attempted to read unsupported RedisGraph version from RDB file.");
			return NULL;
	}
}

