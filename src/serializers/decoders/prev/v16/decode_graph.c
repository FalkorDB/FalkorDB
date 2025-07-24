/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "decode_v16.h"
#include "../../../../index/indexer.h"

static GraphContext *_GetOrCreateGraphContext
(
	char *graph_name
) {
	GraphContext *gc = GraphContext_UnsafeGetGraphContext(graph_name);
	if(gc == NULL) {
		// new graph is being decoded
		// inform the module and create new graph context
		gc = GraphContext_New(graph_name);
		// while loading the graph
		// minimize matrix realloc and synchronization calls
		Graph_SetMatrixPolicy(gc->g, SYNC_POLICY_RESIZE);
	}

	// free the name string, as it either not in used or copied
	RedisModule_Free(graph_name);

	return gc;
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
	uint64_t deleted_edge_count,
	uint64_t label_count,
	uint64_t relation_count
) {
	Graph_AllocateNodes(g, node_count + deleted_node_count);
	Graph_AllocateEdges(g, edge_count + deleted_edge_count);
	for(uint64_t i = 0; i < label_count; i++) Graph_AddLabel(g);
	for(uint64_t i = 0; i < relation_count; i++) Graph_AddRelationType(g);
	// flush all matrices
	// guarantee matrix dimensions matches graph's nodes count
	Graph_ApplyAllPending(g, true);
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
	char *graph_name = SerializerIO_ReadBuffer(rdb, NULL);

	// each key header contains the following:
	// #nodes, #edges, #deleted nodes, #deleted edges, #labels matrices, #relation matrices
	uint64_t node_count         = SerializerIO_ReadUnsigned(rdb);
	uint64_t edge_count         = SerializerIO_ReadUnsigned(rdb);
	uint64_t deleted_node_count = SerializerIO_ReadUnsigned(rdb);
	uint64_t deleted_edge_count = SerializerIO_ReadUnsigned(rdb);
	uint64_t label_count        = SerializerIO_ReadUnsigned(rdb);
	uint64_t relation_count     = SerializerIO_ReadUnsigned(rdb);
	uint64_t multi_edge[relation_count];

	for(uint i = 0; i < relation_count; i++) {
		multi_edge[i] = SerializerIO_ReadUnsigned(rdb);
	}

	// total keys representing the graph
	uint64_t key_number = SerializerIO_ReadUnsigned(rdb);

	GraphContext *gc = _GetOrCreateGraphContext(graph_name);
	Graph *g = gc->g;

	// if it is the first key of this graph,
	// allocate all the data structures, with the appropriate dimensions
	bool first_vkey =
		GraphDecodeContext_GetProcessedKeyCount(gc->decoding_context) == 0;

	if(first_vkey == true) {
		_InitGraphDataStructure(gc->g, node_count, edge_count,
			deleted_node_count, deleted_edge_count, label_count, relation_count);

		gc->decoding_context->multi_edge = array_new(uint64_t, relation_count);
		for(uint i = 0; i < relation_count; i++) {
			// enable/Disable support for multi-edge
			// we will enable support for multi-edge on all relationship
			// matrices once we finish loading the graph
			array_append(gc->decoding_context->multi_edge,  multi_edge[i]);
		}

		GraphDecodeContext_SetKeyCount(gc->decoding_context, key_number);
	}

	// decode graph schemas
	RdbLoadGraphSchema_v16(rdb, gc, !first_vkey);

	// save decode statistics for later progess reporting
	// e.g. "Decoded 20000/4500000 nodes"
	gc->decoding_context->node_count         = node_count;
	gc->decoding_context->edge_count         = edge_count;
	gc->decoding_context->deleted_node_count = deleted_node_count;
	gc->decoding_context->deleted_edge_count = deleted_edge_count;

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

	uint64_t payloads_count = SerializerIO_ReadUnsigned(rdb);
	PayloadInfo *payloads = array_new(PayloadInfo, payloads_count);

	for(uint i = 0; i < payloads_count; i++) {
		// for each payload
		// load its type and the number of entities it contains
		PayloadInfo payload_info;

		payload_info.state          = SerializerIO_ReadUnsigned(rdb);
		payload_info.entities_count = SerializerIO_ReadUnsigned(rdb);

		array_append(payloads, payload_info);
	}

	return payloads;
}

GraphContext *RdbLoadGraphContext_v16
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

	GraphContext *gc = _DecodeHeader(rdb);

	// log progress
	RedisModule_Log(NULL, "notice",
			"Graph '%s' processing virtual key: %" PRId64 "/%" PRId64,
			GraphContext_GetName(gc), gc->decoding_context->keys_processed + 1,
			gc->decoding_context->graph_keys_count);

	// load the key schema
	PayloadInfo *payloads = _RdbLoadKeySchema(rdb);

	// The decode process contains the decode operation of many meta keys, representing independent parts of the graph
	// Each key contains data on one or more of the following:
	// 1. Nodes - The nodes that are currently valid in the graph
	// 2. Deleted nodes - Nodes that were deleted and there ids can be re-used. Used for exact replication of data block state
	// 4. Edges - The edges that are currently valid in the graph
	// 4. Deleted edges - Edges that were deleted and there ids can be re-used. Used for exact replication of data block state
	// The following switch checks which part of the graph the current key holds, and decodes it accordingly
	uint payloads_count = array_len(payloads);
	for(uint i = 0; i < payloads_count; i++) {
		PayloadInfo payload = payloads[i];
		switch(payload.state) {
			case ENCODE_STATE_NODES:
				Graph_SetMatrixPolicy(gc->g, SYNC_POLICY_NOP);
				RdbLoadNodes_v16(rdb, gc, payload.entities_count);

				// log progress
				RedisModule_Log(NULL, "notice",
						"Graph '%s' processed %zu/%" PRIu64 " nodes",
						GraphContext_GetName(gc),
						Graph_UncompactedNodeCount(gc->g),
						gc->decoding_context->node_count);

				break;
			case ENCODE_STATE_DELETED_NODES:
				RdbLoadDeletedNodes_v16(rdb, gc, payload.entities_count);

				// log progress
				RedisModule_Log(NULL, "notice",
						"Graph '%s' processed %u/%" PRId64 " deleted nodes",
						GraphContext_GetName(gc),
						Graph_DeletedNodeCount(gc->g),
						gc->decoding_context->deleted_node_count);

				break;
			case ENCODE_STATE_EDGES:
				Graph_SetMatrixPolicy(gc->g, SYNC_POLICY_NOP);
				RdbLoadEdges_v16(rdb, gc, payload.entities_count);

				// log progress
				RedisModule_Log(NULL, "notice",
						"Graph '%s' processed %" PRId64 "/%" PRId64 " edges",
						GraphContext_GetName(gc), Graph_EdgeCount(gc->g),
						gc->decoding_context->edge_count);

				break;
			case ENCODE_STATE_DELETED_EDGES:
				RdbLoadDeletedEdges_v16(rdb, gc, payload.entities_count);

				// log progress
				RedisModule_Log(NULL, "notice",
						"Graph '%s' processed %u/%" PRId64 " deleted edges",
						GraphContext_GetName(gc),
						Graph_DeletedEdgeCount(gc->g),
						gc->decoding_context->deleted_edge_count);

				break;
			default:
				ASSERT(false && "Unknown encoding");
				break;
		}
	}

	array_free(payloads);

	// update decode context
	GraphDecodeContext_IncreaseProcessedKeyCount(gc->decoding_context);

	// before finalizing keep encountered meta keys names, for future deletion
	const char *key_name = RedisModule_StringPtrLen(rm_key_name, NULL);

	// the virtual key name is not equal the graph name
	if(strcmp(key_name, gc->graph_name) != 0) {
		GraphDecodeContext_AddMetaKey(gc->decoding_context, key_name);
	}

	if(GraphDecodeContext_Finished(gc->decoding_context)) {
		Graph *g = gc->g;

		// set the node label matrix
		Serializer_Graph_SetNodeLabels(g);

		// flush graph matrices
		Graph_ApplyAllPending(g, true);

		// revert to default synchronization behavior
		Graph_SetMatrixPolicy(g, SYNC_POLICY_FLUSH_RESIZE);

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

		GraphDecodeContext_Reset(gc->decoding_context);

		RedisModule_Log(NULL, "notice", "Done decoding graph %s", GraphContext_GetName(gc));
	}

	return gc;
}

