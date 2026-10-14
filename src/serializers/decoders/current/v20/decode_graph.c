/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "decode_v20.h"
#include "../../../../index/indexer.h"
#include "../../../../index/cch_index.h"
#include "../../../../globals.h"

// find the graph a multi-key load of 'graph_name' is building, NULL if none
//
// a graph split across keys (main key + virtual keys) is assembled from all of
// them, so later keys must extend the graph the first key created. only such a
// part-way graph qualifies: a live graph of the same name (e.g. the value a
// RESTORE ... REPLACE is about to replace, or the source of a RESTORE under
// another key) is never extended, the payload is decoded into a new graph
static GraphContext *_GetGraphBeingDecoded
(
	const char *graph_name
) {
	KeySpaceGraphIterator it ;
	Globals_ScanGraphs (&it) ;

	GraphContext *found = NULL ;
	GraphContext *gc ;
	while ((gc = GraphIterator_Next (&it)) != NULL) {
		bool match = strcmp (GraphContext_GetName (gc), graph_name) == 0 &&
			GraphDecodeContext_GetProcessedKeyCount (
					GraphContext_GetDecodingCtx (gc)) ;
		GraphContext_DecreaseRefCount (gc) ;
		if (match) {
			found = gc ;
			break ;
		}
	}

	return found ;
}

static GraphContext *_GetOrCreateGraphContext
(
	char *graph_name
) {
	GraphContext *gc = _GetGraphBeingDecoded (graph_name) ;
	if (!gc) {
		// new graph is being decoded
		// inform the module and create new graph context
		gc = GraphContext_New (graph_name) ;
		GraphContext_AcquireWriteLock (gc) ;
	}

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
	Graph_AllocateNodes(g, node_count + deleted_node_count);
	Graph_AllocateEdges(g, edge_count + deleted_edge_count);
}

static GraphContext *_DecodeHeader
(
	SerializerIO rdb,
	bool detached
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

	GraphContext *gc = NULL ;
	if (detached) {
		gc = GraphContext_New (graph_name) ;
		GraphContext_AcquireWriteLock (gc) ;
	} else {
		gc = _GetOrCreateGraphContext (graph_name) ;
	}
	RedisModule_Free (graph_name) ;
	SerializerIO_SetContext (rdb, gc) ;

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
		SerializerIO_SetError(rdb, EMSG_DECODE_LIMIT, "label count",
				(unsigned long long) label_count);
		return NULL;  // SetError longjmps (when armed) to the landing pad
	}
	if(relation_count > GRAPH_MAX_RELATIONS) {
		SerializerIO_SetError(rdb, EMSG_DECODE_LIMIT, "relation count",
				(unsigned long long) relation_count);
		return NULL;  // SetError longjmps (when armed) to the landing pad
	}

	Graph *g = GraphContext_GetGraph (gc) ;
	GraphDecodeContext *decoding_context = GraphContext_GetDecodingCtx (gc) ;

	// if it is the first key of this graph,
	// allocate all the data structures, with the appropriate dimensions
	bool first_vkey =
		GraphDecodeContext_GetProcessedKeyCount(decoding_context) == 0;

	if (first_vkey == true) {
		_InitGraphDataStructure (g, node_count, edge_count,
			deleted_node_count, deleted_edge_count) ;
		decoding_context->multi_edge = arr_new (uint64_t, relation_count) ;
	}

	// enable/disable support for multi-edge, per relationship matrix; append to
	// the (gc-owned) array on the first key, discard on later keys. support is
	// actually enabled on all relationship matrices once the graph is loaded.
	// a short read mid-loop jumps out - the partial array belongs to gc and is
	// freed with it, so there is nothing to clean up here
	for (uint i = 0 ; i < relation_count ; i++) {
		uint64_t me = SerializerIO_TryReadUnsigned (rdb) ;
		if (first_vkey == true) {
			arr_append (decoding_context->multi_edge, me) ;
		}
	}

	// total keys representing the graph; a short read would have jumped already,
	// so reaching here means key_number is a value we actually read
	uint64_t key_number = SerializerIO_TryReadUnsigned(rdb);
	if (first_vkey == true) {
		GraphDecodeContext_SetKeyCount (decoding_context, key_number) ;
	}

	// decode graph schemas
	RdbLoadGraphSchema_v20 (rdb, gc, !first_vkey) ;

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
		SerializerIO_SetError(rdb, EMSG_DECODE_LIMIT, "payload count",
				(unsigned long long) payloads_count);
		return NULL;  // SetError longjmps to the landing pad
	}

	PayloadInfo *payloads = arr_new(PayloadInfo, payloads_count);

	for(uint i = 0; i < payloads_count; i++) {
		// for each payload
		// load its type and the number of entities it contains
		PayloadInfo payload_info;

		payload_info.state          = SerializerIO_TryReadUnsigned(rdb);
		payload_info.entities_count = SerializerIO_TryReadUnsigned(rdb);

		arr_append(payloads, payload_info);
	}

	return payloads;
}

GraphContext *RdbLoadGraphContext_latest
(
	SerializerIO rdb,
	const RedisModuleString *rm_key_name,
	bool detached
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
	// back there, so the reads in the call tree below never test for an error by
	// hand. _DecodeHeader stashes the in-flight graph via SerializerIO_SetContext
	// so the dispatcher's landing pad can tear it down. when this decoder is
	// driven unarmed - the offload module calls it directly - the same reads
	// latch the error flag and return instead of jumping
	GraphContext *gc = _DecodeHeader (rdb, detached) ;

	// armed: _DecodeHeader either returns a graph or jumps, so gc is non-NULL
	// here. unarmed (offload): a malformed header latches the error and returns
	// NULL instead of jumping - the dispatcher's landing pad won't run, so free
	// the partial graph it stashed (only when this virtual key created it)
	if (gc == NULL) {
		gc = (GraphContext *) SerializerIO_GetContext (rdb) ;
		if (gc != NULL && GraphContext_RefCount (gc) == 0) {
			GraphContext_Free (gc) ;
		}
		return NULL ;
	}

	Graph        *g  = GraphContext_GetGraph (gc) ;
	GraphDecodeContext *decoding_context = GraphContext_GetDecodingCtx (gc) ;

	// log progress
	RedisModule_Log(NULL, "notice",
			"Graph '%s' processing virtual key: %" PRId64 "/%" PRId64,
			GraphContext_GetName(gc), decoding_context->keys_processed + 1,
			decoding_context->graph_keys_count);

	// load the key schema
	PayloadInfo *payloads = _RdbLoadKeySchema(rdb);

	// armed: a malformed payload count jumps; unarmed (offload): it latches the
	// error and returns NULL - return the partial graph rather than arr_len(NULL)
	if (payloads == NULL) return gc ;

	// The decode process contains the decode operation of many meta keys, representing independent parts of the graph
	// Each key contains data on one or more of the following:
	// 1. Nodes - The nodes that are currently valid in the graph
	// 2. Deleted nodes - Nodes that were deleted and there ids can be re-used. Used for exact replication of data block state
	// 4. Edges - The edges that are currently valid in the graph
	// 4. Deleted edges - Edges that were deleted and there ids can be re-used. Used for exact replication of data block state
	// The following switch checks which part of the graph the current key holds, and decodes it accordingly
	// a short read or malformed data inside any loader aborts to the landing
	// pad above, so the loop body never has to test for an error itself
	uint payloads_count = arr_len(payloads);
	for(uint i = 0; i < payloads_count; i++) {
		PayloadInfo payload = payloads[i];
		switch(payload.state) {
			case ENCODE_STATE_NODES:
				RdbLoadNodes_v20(rdb, g, payload.entities_count,
						decoding_context->node_count +
						decoding_context->deleted_node_count);

				// log progress
				RedisModule_Log(NULL, "notice",
						"Graph '%s' processed %zu/%" PRIu64 " nodes",
						GraphContext_GetName(gc),
						Graph_UncompactedNodeCount(g),
						decoding_context->node_count);

				break;

			case ENCODE_STATE_DELETED_NODES:
				RdbLoadDeletedNodes_v20(rdb, g, payload.entities_count,
						decoding_context->node_count +
						decoding_context->deleted_node_count);

				// log progress
				RedisModule_Log(NULL, "notice",
						"Graph '%s' processed %u/%" PRIu64 " deleted nodes",
						GraphContext_GetName(gc),
						Graph_DeletedNodeCount(g),
						decoding_context->deleted_node_count);

				break;

			case ENCODE_STATE_EDGES:
				RdbLoadEdges_v20(rdb, g, payload.entities_count,
						decoding_context->edge_count +
						decoding_context->deleted_edge_count);

				// log progress
				RedisModule_Log(NULL, "notice",
						"Graph '%s' processed %" PRId64 "/%" PRId64 " edges",
						GraphContext_GetName(gc), Graph_EdgeCount(g),
						decoding_context->edge_count);

				break;
			case ENCODE_STATE_DELETED_EDGES:
				RdbLoadDeletedEdges_v20(rdb, g, payload.entities_count,
						decoding_context->edge_count +
						decoding_context->deleted_edge_count);

				// log progress
				RedisModule_Log(NULL, "notice",
						"Graph '%s' processed %u/%" PRId64 " deleted edges",
						GraphContext_GetName(gc),
						Graph_DeletedEdgeCount(g),
						decoding_context->deleted_edge_count);

				break;

			case ENCODE_STATE_LABELS_MATRICES:
				RedisModule_Log(NULL, "notice",
						"Graph '%s' loading label matrices",
						GraphContext_GetName(gc));

				RdbLoadLabelMatrices_v20(rdb, g);
				break;

			case ENCODE_STATE_RELATION_MATRICES:
				RedisModule_Log(NULL, "notice",
						"Graph '%s' loading relation matrices",
						GraphContext_GetName(gc));

				RdbLoadRelationMatrices_v20(rdb, g);
				break;

			case ENCODE_STATE_ADJ_MATRIX:
				RedisModule_Log(NULL, "notice",
						"Graph '%s' loading Adjacency matrix",
						GraphContext_GetName(gc));

				RdbLoadAdjMatrix_v20(rdb, g);
				break;

			case ENCODE_STATE_LBLS_MATRIX:
				RedisModule_Log(NULL, "notice",
						"Graph '%s' loading Labels matrix",
						GraphContext_GetName(gc));

				RdbLoadLblsMatrix_v20(rdb, g);
				break;

			case ENCODE_STATE_CCH_INDICES:
				RdbLoadCCH_v20(rdb, gc);
				break;

			default:
				SerializerIO_SetError(rdb, EMSG_DECODER_UNKNOWN_PAYLOAD);
				break;
		}
	}

	arr_free(payloads);

	// all reads for this key are done; the post-processing below (index
	// population, CCH build) issues no reads, and the dispatcher disarms the
	// abort handler once this decoder returns
	GraphDecodeContext_IncreaseProcessedKeyCount(decoding_context);

	// before finalizing keep encountered meta keys names, for future deletion
	const char *key_name = RedisModule_StringPtrLen(rm_key_name, NULL);

	// the virtual key name is not equal the graph name
	if(strcmp(key_name, GraphContext_GetName (gc)) != 0) {
		GraphDecodeContext_AddMetaKey(decoding_context, key_name);
	}

	if (GraphDecodeContext_Finished (decoding_context)) {
		Graph_ApplyAllPending (g, false) ;
		GraphContext_ReleaseLock (gc) ;

		uint rel_count   = Graph_RelationTypeCount(g);
		uint label_count = Graph_LabelTypeCount(g);

		// get delay indexing configuration
		bool delay_indexing;
		Config_Option_get(Config_DELAY_INDEXING, &delay_indexing);

		// report index construction method
		if(delay_indexing) {
			RedisModule_Log(NULL, "notice",
					"Graph '%s' Indexes are constructed in the background.",
					GraphContext_GetName(gc));
		} else {
			RedisModule_Log(NULL, "notice", "Graph '%s' Constructing indexes.",
					GraphContext_GetName(gc));
		}

		// update the node statistics, enable node indices
		for(LabelID i = 0; i < label_count; i++) {
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
					// populate index
					Index_Populate (idx, gc) ;
					Index_Enable (idx) ;
					Schema_ActivateIndex (s) ;
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
					// populate index
					Index_Populate (idx, gc) ;
					Index_Enable(idx);
					Schema_ActivateIndex(s);
				}
			}
		}

		// graph-level CCH path indices are decoded as definitions only (their
		// identity) during ENCODE_STATE_CCH_INDICES; rebuild each hierarchy now
		// that the whole graph is materialized, one after another
		uint cch_count = GraphContext_CCHIndexCount(gc);
		for(uint i = 0; i < cch_count; i++) {
			CCHIndex_Build(GraphContext_GetCCHIndexAt(gc, i), g);
		}

		// make sure graph doesn't contains may pending changes
		ASSERT(Graph_Pending(g) == false);

		GraphDecodeContext_Reset(decoding_context);
		RedisModule_Log(NULL, "notice", "Done decoding graph %s",
				GraphContext_GetName(gc));
	}

	return gc;
}

