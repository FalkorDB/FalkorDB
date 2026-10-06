/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "decode_graph.h"
#include "current/v20/decode_v20.h"
#include "prev/v19/decode_v19.h"
#include "../encoding_version.h"

// log why a graph key failed to load, see decode_graph.h
void RdbLoadGraph_LogFailure
(
	RedisModuleIO *rdb,  // redis IO the key was read from
	SerializerIO io      // serializer that failed
) {
	const char *reason = SerializerIO_ErrorReason (io) ;
	if (reason == NULL) {
		return ;
	}

	const RedisModuleString *key = RedisModule_GetKeyNameFromIO (rdb) ;
	RedisModule_Log (NULL, "warning", "Failed loading graph key '%s': %s",
			key != NULL ? RedisModule_StringPtrLen (key, NULL) : "",
			reason) ;
}

GraphContext *RdbLoadGraph
(
	RedisModuleIO *rdb
) {
	const RedisModuleString *rm_key_name = RedisModule_GetKeyNameFromIO (rdb) ;

	SerializerIO io = SerializerIOv2_FromBufferedRedisModuleIO (rdb, false) ;
	GraphContext *gc = RdbLoadGraphContext_latest (io, rm_key_name, false) ;

	// detect a short read / IO error before the serializer is torn down
	bool io_error = SerializerIO_Error (io) ;
	if (io_error) {
		RdbLoadGraph_LogFailure (rdb, io) ;
	}
	SerializerIO_Free (&io) ;

	if(io_error) {
		// short read - abort the load and return NULL so Redis fails cleanly
		// (a truncated RESTORE errors, a truncated replication stream retries)
		// a graph that isn't registered yet (ref count 0) was created by this
		// virtual key and must be freed here - nothing else will. an already
		// registered graph belongs to earlier virtual keys and is reconciled by
		// Redis via their key free callbacks when it aborts the load
		if(gc != NULL && GraphContext_RefCount(gc) == 0) {
			GraphContext_Free(gc);
		}
		return NULL;
	}

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
		case 19:
			// v19 shares the latest decoder signature (it carries the same
			// `detached` parameter), so no adapter is needed
			return RdbLoadGraphContext_v19 ;
		default:
			return NULL ;  // no SerializerIO decoder for this version
	}
}

