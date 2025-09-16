/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "decoders.h"
#include "decode_graph.h"
#include "../encoding_version.h"

GraphContext *RdbLoadGraph
(
	RedisModuleIO *rdb
) {
	const RedisModuleString *rm_key_name = RedisModule_GetKeyNameFromIO(rdb);

	SerializerIO io = SerializerIO_FromBufferedRedisModuleIO(rdb, false);
	GraphContext *gc = RdbLoadGraphContext_latest(io, rm_key_name);
	SerializerIO_Free(&io);

	return gc;
}

// get decoder for specified version
DecoderFP Decoder_GetDecoder
(
	uint64_t v  // version
) {
	if (v > GRAPH_ENCODING_LATEST_V) {
		printf ("unknown decoder version (%llu).\n", v) ;
		return NULL ;
		// not backward compatible
	} else if(v < 14) {
		printf ("decoder version (%llu) is too old.\n", v) ;
		return NULL;
	}

	switch (v) {
		case 14:
			return RdbLoadGraphContext_v14 ;

		case 15:
			return RdbLoadGraphContext_v15 ;

		case 16:
			return RdbLoadGraphContext_v16 ;

		case 17:
			return RdbLoadGraphContext_v17 ;

		case 18:
			return RdbLoadGraphContext_latest ;

		default:
			ASSERT (false && "attempted to get unsupported decoder version.") ;
			break ;
	}

	return NULL ;
}

