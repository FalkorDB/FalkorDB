/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "decode_graph.h"
#include "current/v19/decode_v19.h"
#include "../encoding_version.h"

GraphContext *RdbLoadGraph
(
	RedisModuleIO *rdb
) {
	const RedisModuleString *rm_key_name = RedisModule_GetKeyNameFromIO (rdb) ;

	SerializerIO io = SerializerIOv2_FromBufferedRedisModuleIO (rdb, false) ;
	GraphContext *gc = RdbLoadGraphContext_latest (io, rm_key_name, false) ;
	SerializerIO_Free (&io) ;

	return gc ;
}

RdbLoadGraphContext_t Graph_GetDecoder
(
	uint32_t version
) {
	// expose only SerializerIO-based decoders that match the canonical
	// RdbLoadGraphContext_latest signature. offload dumps are never older than
	// the version offloading shipped in, so the latest decoder covers all
	// current dumps. when a newer encoding version becomes latest, map each
	// aging version to its decoder here — wrapping any prev-version decoder that
	// omits the `detached` parameter in a small adapter with this signature.
	switch(version) {
		case GRAPH_ENCODING_LATEST_V:
			return RdbLoadGraphContext_latest ;
		default:
			return NULL ;  // no SerializerIO decoder for this version
	}
}

