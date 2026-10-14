/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "decode_graph.h"
#include "current/v20/decode_v20.h"
#include "prev/decoders.h"
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

// create the serializer backend matching an encoding version; the reader
// changed over time, so older versions decode through an older serializer
static SerializerIO _SerializerForVersion
(
	RedisModuleIO *rdb,  // redis IO
	int encver           // encoding version
) {
	switch(encver) {
		case 14:
		case 15:
		case 16:
			return SerializerIO_FromRedisModuleIO (rdb, false) ;
		case 17:
		case 18:
			return SerializerIO_FromBufferedRedisModuleIO (rdb, false) ;
		case 19:
		case GRAPH_ENCODING_LATEST_V:  // 20
			return SerializerIOv2_FromBufferedRedisModuleIO (rdb, false) ;
		default:
			// unreachable: versions below GRAPH_DECODE_MIN_V are rejected before
			// we get here, and newer-than-latest are rejected too
			return NULL ;
	}
}

// dispatch to the decoder for an encoding version
static GraphContext *_DecodeVersion
(
	SerializerIO io,               // serializer (abort handler armed)
	const RedisModuleString *key,  // key name
	int encver                     // encoding version
) {
	switch(encver) {
		case 14: return RdbLoadGraphContext_v14 (io, key) ;
		case 15: return RdbLoadGraphContext_v15 (io, key) ;
		case 16: return RdbLoadGraphContext_v16 (io, key) ;
		case 17: return RdbLoadGraphContext_v17 (io, key) ;
		case 18: return RdbLoadGraphContext_v18 (io, key) ;
		case 19: return RdbLoadGraphContext_v19 (io, key, false) ;
		case GRAPH_ENCODING_LATEST_V:
			return RdbLoadGraphContext_latest (io, key, false) ;
		default:
			return NULL ;
	}
}

// load a graph key encoded with 'encver', see decode_graph.h.
//
// this is the single decode dispatcher: it installs one abort handler (landing
// pad) for the whole decode. every decoder (v14 .. latest) reads through the
// TryRead API, which longjmps straight here on a short read - or on malformed
// data via SerializerIO_SetError - so no decoder installs its own handler and
// no read in the call tree tests for an error by hand. a decoder stashes the
// in-flight graph via SerializerIO_SetContext so this landing pad can tear it
// down after a jump unwound the decoder's own locals
GraphContext *RdbLoadGraph
(
	RedisModuleIO *rdb,  // redis IO
	int encver           // encoding version
) {
	const RedisModuleString *key = RedisModule_GetKeyNameFromIO (rdb) ;

	SerializerIO io = _SerializerForVersion (rdb, encver) ;
	ASSERT (io != NULL) ;

	GraphContext *gc = NULL ;

	if (SerializerIO_Try (io) == 0) {
		gc = _DecodeVersion (io, key, encver) ;
	} else {
		// short read / malformed data: the decode longjmped here; recover the
		// partially-built graph the decoder stashed (NULL if it failed before
		// creating it)
		gc = (GraphContext*) SerializerIO_GetContext (io) ;
	}
	SerializerIO_Disarm (io) ;

	// detect a short read / IO error before the serializer is torn down
	bool io_error = SerializerIO_Error (io) ;
	if (io_error) {
		RdbLoadGraph_LogFailure (rdb, io) ;
	}
	SerializerIO_Free (&io) ;

	if (io_error) {
		// abort the load and return NULL so Redis fails cleanly (a truncated
		// RESTORE errors, a truncated replication stream retries). a graph that
		// isn't registered yet (ref count 0) was created by this virtual key and
		// must be freed here - nothing else will. an already registered graph
		// belongs to earlier virtual keys and is reconciled by Redis via their
		// key free callbacks when it aborts the load
		if (gc != NULL && GraphContext_RefCount (gc) == 0) {
			GraphContext_Free (gc) ;
		}
		return NULL ;
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

