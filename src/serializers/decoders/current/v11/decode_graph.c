/*
* Copyright 2018-2022 Redis Labs Ltd. and Contributors
*
* This file is available under the Redis Labs Source Available License Agreement
*/

#include "decode_v11.h"

static GraphContext *_GetOrCreateGraphContext
(
	char *graph_name
) {
	GraphContext *gc = GraphContext_GetRegisteredGraphContext(graph_name);
	if(!gc) {
		// New graph is being decoded. Inform the module and create new graph context.
		gc = GraphContext_New(graph_name);
		// While loading the graph, minimize matrix realloc and synchronization calls.
		Graph_SetMatrixPolicy(gc->g, SYNC_POLICY_RESIZE);
	}
	// Free the name string, as it either not in used or copied.
	rm_free(graph_name);

	// Set the GraphCtx in thread-local storage.
	QueryCtx_SetGraphCtx(gc);

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
	uint64_t label_count,
	uint64_t relation_count
) {
	Graph_AllocateNodes(g, node_count);
	Graph_AllocateEdges(g, edge_count);
	for(uint64_t i = 0; i < label_count; i++) Graph_AddLabel(g);
	for(uint64_t i = 0; i < relation_count; i++) Graph_AddRelationType(g);
	// flush all matrices
	// guarantee matrix dimensions matches graph's nodes count
	Graph_ApplyAllPending(g, true);
}

static GraphContext *_DecodeHeader
(
	IODecoder *io
) {
	// Header format:
	// Graph name
	// Node count
	// Edge count
	// Label matrix count
	// Relation matrix count - N
	// Does relationship matrix Ri holds mutiple edges under a single entry X N
	// Number of graph keys (graph context key + meta keys)
	// Schema

	// graph name
	char *graph_name = IODecoder_LoadStringBuffer(io, NULL);

	// each key header contains the following:
	// #nodes, #edges, #labels matrices, #relation matrices
	uint64_t  node_count      =  IODecoder_LoadUnsigned(io);
	uint64_t  edge_count      =  IODecoder_LoadUnsigned(io);
	uint64_t  label_count     =  IODecoder_LoadUnsigned(io);
	uint64_t  relation_count  =  IODecoder_LoadUnsigned(io);
	uint64_t  multi_edge[relation_count];

	for(uint i = 0; i < relation_count; i++) {
		multi_edge[i] = IODecoder_LoadUnsigned(io);
	}

	// total keys representing the graph
	uint64_t key_number = IODecoder_LoadUnsigned(io);

	GraphContext *gc = _GetOrCreateGraphContext(graph_name);
	Graph *g = gc->g;

	// if it is the first key of this graph,
	// allocate all the data structures, with the appropriate dimensions
	if(GraphDecodeContext_GetProcessedKeyCount(gc->decoding_context) == 0) {
		_InitGraphDataStructure(gc->g, node_count, edge_count, label_count, relation_count);

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
	RdbLoadGraphSchema_v11(io, gc);

	return gc;
}

static PayloadInfo *_RdbLoadKeySchema
(
	IODecoder *io
) {
	// Format:
	// #Number of payloads info - N
	// N * Payload info:
	//     Encode state
	//     Number of entities encoded in this state.

	uint64_t payloads_count = IODecoder_LoadUnsigned(io);
	PayloadInfo *payloads = array_new(PayloadInfo, payloads_count);

	for(uint i = 0; i < payloads_count; i++) {
		// for each payload
		// load its type and the number of entities it contains
		PayloadInfo payload_info;
		payload_info.state =  IODecoder_LoadUnsigned(io);
		payload_info.entities_count =  IODecoder_LoadUnsigned(io);
		array_append(payloads, payload_info);
	}
	return payloads;
}

GraphContext *RdbLoadGraphContext_v11
(
	IODecoder *io
) {

	// Key format:
	//  Header
	//  Payload(s) count: N
	//  Key content X N:
	//      Payload type (Nodes / Edges / Deleted nodes/ Deleted edges/ Graph schema)
	//      Entities in payload
	//  Payload(s) X N

	GraphContext *gc = _DecodeHeader(io);

	// load the key schema
	PayloadInfo *key_schema = _RdbLoadKeySchema(io);

	// The decode process contains the decode operation of many meta keys, representing independent parts of the graph
	// Each key contains data on one or more of the following:
	// 1. Nodes - The nodes that are currently valid in the graph
	// 2. Deleted nodes - Nodes that were deleted and there ids can be re-used. Used for exact replication of data block state
	// 3. Edges - The edges that are currently valid in the graph
	// 4. Deleted edges - Edges that were deleted and there ids can be re-used. Used for exact replication of data block state
	// 5. Graph schema - Properties, indices
	// The following switch checks which part of the graph the current key holds, and decodes it accordingly
	uint payloads_count = array_len(key_schema);
	for(uint i = 0; i < payloads_count; i++) {
		PayloadInfo payload = key_schema[i];
		switch(payload.state) {
			case ENCODE_STATE_NODES:
				Graph_SetMatrixPolicy(gc->g, SYNC_POLICY_NOP);
				RdbLoadNodes_v11(io, gc, payload.entities_count);
				break;
			case ENCODE_STATE_DELETED_NODES:
				RdbLoadDeletedNodes_v11(io, gc, payload.entities_count);
				break;
			case ENCODE_STATE_EDGES:
				Graph_SetMatrixPolicy(gc->g, SYNC_POLICY_NOP);
				RdbLoadEdges_v11(io, gc, payload.entities_count);
				break;
			case ENCODE_STATE_DELETED_EDGES:
				RdbLoadDeletedEdges_v11(io, gc, payload.entities_count);
				break;
			case ENCODE_STATE_GRAPH_SCHEMA:
				// skip, handled in _DecodeHeader
				break;
			default:
				ASSERT(false && "Unknown encoding");
				break;
		}
	}
	array_free(key_schema);

	// update decode context
	GraphDecodeContext_IncreaseProcessedKeyCount(gc->decoding_context);

	// before finalizing keep encountered meta keys names, for future deletion
	if(io->t == IODecoderType_RDB) {
		RedisModuleIO *rdb_io = (RedisModuleIO*)io->io;
		const RedisModuleString *rm_key_name =
			RedisModule_GetKeyNameFromIO(rdb_io);
		const char *key_name =
			RedisModule_StringPtrLen(rm_key_name, NULL);

		// the virtual key name is not equal the graph name
		if(strcmp(key_name, gc->graph_name) != 0) {
			GraphDecodeContext_AddMetaKey(gc->decoding_context, key_name);
		}
	}

	if(GraphDecodeContext_Finished(gc->decoding_context)) {
		Graph *g = gc->g;

		// set the node label matrix
		Serializer_Graph_SetNodeLabels(g);

		// revert to default synchronization behavior
		Graph_SetMatrixPolicy(g, SYNC_POLICY_FLUSH_RESIZE);
		Graph_ApplyAllPending(g, true);

		uint label_count = Graph_LabelTypeCount(g);
		// update the node statistics
		for(uint i = 0; i < label_count; i++) {
			GrB_Index nvals;
			RG_Matrix L = Graph_GetLabelMatrix(g, i);
			RG_Matrix_nvals(&nvals, L);
			GraphStatistics_IncNodeCount(&g->stats, i, nvals);
		}

		// make sure graph doesn't contains may pending changes
		ASSERT(Graph_Pending(g) == false);

		GraphDecodeContext_Reset(gc->decoding_context);

		RedisModule_Log(NULL, "notice", "Done decoding graph %s", gc->graph_name);
	}

	// release thread-local variables
	QueryCtx_Free();

	return gc;
}

