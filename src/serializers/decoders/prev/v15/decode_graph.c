/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "decode_v15.h"
#include "../../../../index/indexer.h"

static GraphContext *_GetOrCreateGraphContext
(
	char *graph_name
) {
	GraphContext *gc = GraphContext_UnsafeGetGraphContext (graph_name) ;
	if (!gc) {
		// new graph is being decoded
		// inform the module and create new graph context
		gc = GraphContext_New (graph_name) ;
		Graph *g = GraphContext_GetGraph (gc) ;

		// while loading the graph
		// minimize matrix realloc and synchronization calls
		GraphContext_AcquireWriteLock (gc) ;
		Graph_SetMatrixPolicy (g, SYNC_POLICY_RESIZE) ;
	}

	// free the name string, as it either not in used or copied
	RedisModule_Free (graph_name) ;

	return gc ;
}

// the first initialization of the graph data structure guarantees that
// there will be no further re-allocation of data blocks and matrices
// since they are all in the appropriate size
static void _InitGraphDataStructure
(
	Graph *g,
	uint64_t node_count,
	uint64_t edge_count,
	uint64_t deleted_node_count,
	uint64_t deleted_edge_count
) {
	Graph_AllocateNodes (g, node_count + deleted_node_count) ;
	Graph_AllocateEdges (g, edge_count + deleted_edge_count) ;
}

static GraphContext *_DecodeHeader
(
	SerializerIO rdb
) {
	// Header format:
	// Graph name
	// Node count
	// Edge count
	// Deleted node count
	// Deleted edge count
	// Label matrix count
	// Relation matrix count - N
	// Does relationship matrix Ri holds mutiple edges under a single entry X N
	// Number of graph keys (graph context key + meta keys)
	// Schema

	// graph name
	char *graph_name = SerializerIO_TryReadCString(rdb);

	// _GetOrCreateGraphContext frees graph_name
	GraphContext *gc = _GetOrCreateGraphContext(graph_name);

	// stash the in-flight graph so the dispatcher's landing pad can tear it
	// down after a short read unwinds this decoder's locals
	SerializerIO_SetContext(rdb, gc);

	// each key header contains the following:
	// #nodes, #edges, #deleted nodes, #deleted edges, #labels matrices, #relation matrices
	uint64_t node_count         = SerializerIO_TryReadUnsigned(rdb);
	uint64_t edge_count         = SerializerIO_TryReadUnsigned(rdb);
	uint64_t deleted_node_count = SerializerIO_TryReadUnsigned(rdb);
	uint64_t deleted_edge_count = SerializerIO_TryReadUnsigned(rdb);
	uint64_t label_count        = SerializerIO_TryReadUnsigned(rdb);
	uint64_t relation_count     = SerializerIO_TryReadUnsigned(rdb);

	// guard the label and relation counts before they size allocations / drive
	// loops. graph_name is already freed and gc is stashed, so a SetError jump
	// tears the partial graph down via the dispatcher's landing pad
	if(label_count > GRAPH_MAX_LABELS) {
		SerializerIO_SetError(rdb, "label count %llu exceeds limit",
				(unsigned long long) label_count);
		return NULL;  // SetError longjmps (when armed) to the landing pad
	}
	if(relation_count > GRAPH_MAX_RELATIONS) {
		SerializerIO_SetError(rdb, "relation count %llu exceeds limit",
				(unsigned long long) relation_count);
		return NULL;  // SetError longjmps (when armed) to the landing pad
	}

	Graph *g = GraphContext_GetGraph (gc) ;
	GraphDecodeContext *decoding_context = GraphContext_GetDecodingCtx (gc) ;

	// if it is the first key of this graph,
	// allocate all the data structures, with the appropriate dimensions
	bool first_vkey =
		GraphDecodeContext_GetProcessedKeyCount(decoding_context) == 0;

	if(first_vkey == true) {
		_InitGraphDataStructure(g, node_count, edge_count, deleted_node_count,
				deleted_edge_count) ;
		decoding_context->multi_edge = arr_new(uint64_t, relation_count);
	}

	// enable/disable support for multi-edge, per relationship matrix; append to
	// the (gc-owned) array on the first key, discard on later keys. support is
	// actually enabled on all relationship matrices once the graph is loaded.
	// a short read mid-loop jumps out - the partial array belongs to gc and is
	// freed with it, so there is nothing to clean up here
	for(uint i = 0; i < relation_count; i++) {
		uint64_t me = SerializerIO_TryReadUnsigned(rdb);
		if(first_vkey == true) {
			arr_append(decoding_context->multi_edge, me);
		}
	}

	// total keys representing the graph; a short read would have jumped already
	uint64_t key_number = SerializerIO_TryReadUnsigned(rdb);
	if(first_vkey == true) {
		GraphDecodeContext_SetKeyCount(decoding_context, key_number);
	}

	// decode graph schemas
	RdbLoadGraphSchema_v15(rdb, gc, !first_vkey);

	// save decode statistics for later progess reporting
	// e.g. "Decoded 20000/4500000 nodes"
	decoding_context->node_count         = node_count;
	decoding_context->edge_count         = edge_count;
	decoding_context->deleted_node_count = deleted_node_count;
	decoding_context->deleted_edge_count = deleted_edge_count;

	return gc;
}

static PayloadInfo *_RdbLoadKeySchema
(
	SerializerIO rdb
) {
	// Format:
	// #Number of payloads info - N
	// N * Payload info:
	//     Encode state
	//     Number of entities encoded in this state.

	uint64_t payloads_count = SerializerIO_TryReadUnsigned(rdb);

	// guard the payload count before it sizes the array below
	if(payloads_count > GRAPH_MAX_KEY_PAYLOADS) {
		SerializerIO_SetError(rdb, "payload count %llu exceeds limit",
				(unsigned long long) payloads_count);
		return NULL;  // SetError longjmps to the landing pad
	}

	PayloadInfo *payloads = arr_new(PayloadInfo, payloads_count);

	for(uint i = 0; i < payloads_count; i++) {
		// for each payload
		// load its type and the number of entities it contains
		PayloadInfo payload_info;
		payload_info.state =  SerializerIO_TryReadUnsigned(rdb);
		payload_info.entities_count =  SerializerIO_TryReadUnsigned(rdb);
		arr_append(payloads, payload_info);
	}
	return payloads;
}

GraphContext *RdbLoadGraphContext_v15
(
	SerializerIO rdb,
	const RedisModuleString *rm_key_name
) {

	// Key format:
	//  Header
	//  Payload(s) count: N
	//  Key content X N:
	//      Payload type (Nodes / Edges / Deleted nodes/ Deleted edges/ Graph schema)
	//      Entities in payload
	//  Payload(s) X N

	// the decode abort handler is installed once by the dispatcher
	// (RdbLoadGraph): a short read or a SerializerIO_SetError longjmps straight
	// back there. _DecodeHeader stashes the in-flight graph via
	// SerializerIO_SetContext so the dispatcher's landing pad can tear it down
	GraphContext *gc = _DecodeHeader(rdb);

	// armed: _DecodeHeader either returns a graph or jumps, so gc is non-NULL
	// here; guard defensively so a malformed header can't be dereferenced
	if (gc == NULL) return NULL ;

	Graph *g = GraphContext_GetGraph (gc) ;
	GraphDecodeContext *decoding_context = GraphContext_GetDecodingCtx (gc) ;

	// log progress
	RedisModule_Log(NULL, "notice",
			"Graph '%s' processing virtual key: %" PRIu64 "/%" PRIu64,
			GraphContext_GetName(gc), decoding_context->keys_processed + 1,
			decoding_context->graph_keys_count);

	// id ceilings: datablock ids run below the header's live + deleted counts
	uint64_t node_id_limit =
		decoding_context->node_count + decoding_context->deleted_node_count;
	uint64_t edge_id_limit =
		decoding_context->edge_count + decoding_context->deleted_edge_count;

	// load the key schema
	PayloadInfo *key_schema = _RdbLoadKeySchema(rdb);

	// a malformed payload count jumps to the landing pad; guard defensively
	if (key_schema == NULL) return gc ;

	// The decode process contains the decode operation of many meta keys, representing independent parts of the graph
	// Each key contains data on one or more of the following:
	// 1. Nodes - The nodes that are currently valid in the graph
	// 2. Deleted nodes - Nodes that were deleted and there ids can be re-used. Used for exact replication of data block state
	// 4. Edges - The edges that are currently valid in the graph
	// 4. Deleted edges - Edges that were deleted and there ids can be re-used. Used for exact replication of data block state
	// 5. Graph schema - Properties, indices
	// The following switch checks which part of the graph the current key holds, and decodes it accordingly
	uint payloads_count = arr_len(key_schema);
	for(uint i = 0; i < payloads_count; i++) {
		PayloadInfo payload = key_schema[i];
		switch(payload.state) {
			case ENCODE_STATE_NODES:
				Graph_SetMatrixPolicy(g, SYNC_POLICY_NOP);
				RdbLoadNodes_v15(rdb, gc, payload.entities_count, node_id_limit);

				// log progress
				RedisModule_Log(NULL, "notice",
						"Graph '%s' processed %zu/%" PRIu64 " nodes",
						GraphContext_GetName(gc),
						Graph_UncompactedNodeCount(g),
						decoding_context->node_count);

				break;
			case ENCODE_STATE_DELETED_NODES:
				RdbLoadDeletedNodes_v15(rdb, gc, payload.entities_count,
						node_id_limit);

				// log progress
				RedisModule_Log(NULL, "notice",
						"Graph '%s' processed %u/%" PRId64 " deleted nodes",
						GraphContext_GetName(gc),
						Graph_DeletedNodeCount(g),
						decoding_context->deleted_node_count);

				break;
			case ENCODE_STATE_EDGES:
				Graph_SetMatrixPolicy(g, SYNC_POLICY_NOP);
				RdbLoadEdges_v15(rdb, gc, payload.entities_count, node_id_limit,
						edge_id_limit);

				// log progress
				RedisModule_Log(NULL, "notice",
						"Graph '%s' processed %" PRId64 "/%" PRId64 " edges",
						GraphContext_GetName(gc), Graph_EdgeCount(g),
						decoding_context->edge_count);

				break;
			case ENCODE_STATE_DELETED_EDGES:
				RdbLoadDeletedEdges_v15(rdb, gc, payload.entities_count,
						edge_id_limit);

				// log progress
				RedisModule_Log(NULL, "notice",
						"Graph '%s' processed %u/%" PRId64 " deleted edges",
						GraphContext_GetName(gc),
						Graph_DeletedEdgeCount(g),
						decoding_context->deleted_edge_count);

				break;
			case ENCODE_STATE_GRAPH_SCHEMA:
				// skip, handled in _DecodeHeader
				break;
			default:
				SerializerIO_SetError(rdb, EMSG_DECODER_UNKNOWN_PAYLOAD);
				break;
		}
	}

	arr_free(key_schema);

	// update decode context
	GraphDecodeContext_IncreaseProcessedKeyCount(decoding_context);

	// before finalizing keep encountered meta keys names, for future deletion
	const char *key_name = RedisModule_StringPtrLen(rm_key_name, NULL);

	// the virtual key name is not equal the graph name
	if(strcmp(key_name, GraphContext_GetName (gc)) != 0) {
		GraphDecodeContext_AddMetaKey(decoding_context, key_name);
	}

	if (GraphDecodeContext_Finished (decoding_context)) {
		// release graph write lock
		GraphContext_ReleaseLock (gc) ;

		// set the node label matrix
		Serializer_Graph_SetNodeLabels(g);

		// flush graph matrices
		Graph_ApplyAllPending(g, true);

		uint rel_count   = Graph_RelationTypeCount(g);
		uint label_count = Graph_LabelTypeCount(g);

		// get delay indexing configuration
		bool delay_indexing;
		Config_Option_get(Config_DELAY_INDEXING, &delay_indexing);

		// update the node statistics, enable node indices
		for(uint i = 0; i < label_count; i++) {
			GrB_Index nvals;
			Delta_Matrix L = Graph_GetLabelMatrix(g, i);
			Delta_Matrix_nvals(&nvals, L);
			GraphStatistics_IncNodeCount(&g->stats, i, nvals);

			Index idx;
			Schema *s = GraphContext_GetSchemaByID(gc, i, SCHEMA_NODE);
			idx = PENDING_IDX(s);

			if(idx != NULL) {
				if(delay_indexing) {
					// start async indexing
					Indexer_PopulateIndex(gc, s, idx);
				} else {
					// index populated enable it
					Index_Enable(idx);
					Schema_ActivateIndex(s);
				}
			}
		}

		// enable all edge indices
		for(uint i = 0; i < rel_count; i++) {
			Index idx;
			Schema *s = GraphContext_GetSchemaByID(gc, i, SCHEMA_EDGE);
			idx = PENDING_IDX(s);

			if(idx != NULL) {
				if(delay_indexing) {
					// start async indexing
					Indexer_PopulateIndex(gc, s, idx);
				} else {
					// index populated enable it
					Index_Enable(idx);
					Schema_ActivateIndex(s);
				}
			}
		}

		// make sure graph doesn't contains may pending changes
		ASSERT(Graph_Pending(g) == false);

		GraphDecodeContext_Reset (decoding_context) ;

		RedisModule_Log(NULL, "notice", "Done decoding graph %s", GraphContext_GetName(gc));
	}

	return gc;
}

