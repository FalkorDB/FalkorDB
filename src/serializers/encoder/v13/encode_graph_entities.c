/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "encode_v13.h"
#include "../../../datatypes/datatypes.h"

// forword decleration
static void _RdbSaveSIValue
(
	RedisModuleIO *rdb,
	const SIValue *v
);

static void _RdbSaveSIArray
(
	RedisModuleIO *rdb,
	const SIValue list
) {
	/* saves array as
	   unsigned : array legnth
	   array[0]
	   .
	   .
	   .
	   array[array length -1]
	 */
	uint arrayLen = SIArray_Length(list);
	RedisModule_SaveUnsigned(rdb, arrayLen);
	for(uint i = 0; i < arrayLen; i ++) {
		SIValue value = SIArray_Get(list, i);
		_RdbSaveSIValue(rdb, &value);
	}
}

static void _RdbSaveSIValue
(
	RedisModuleIO *rdb,
	const SIValue *v
) {
	// Format:
	// SIType
	// Value
	RedisModule_SaveUnsigned(rdb, v->type);
	switch(v->type) {
		case T_BOOL:
		case T_INT64:
			RedisModule_SaveSigned(rdb, v->longval);
			return;
		case T_DOUBLE:
			RedisModule_SaveDouble(rdb, v->doubleval);
			return;
		case T_STRING:
			RedisModule_SaveStringBuffer(rdb, v->stringval, strlen(v->stringval) + 1);
			return;
		case T_ARRAY:
			_RdbSaveSIArray(rdb, *v);
			return;
		case T_POINT:
			RedisModule_SaveDouble(rdb, Point_lat(*v));
			RedisModule_SaveDouble(rdb, Point_lon(*v));
		case T_NULL:
			return; // No data beyond the type needs to be encoded for a NULL value.
		default:
			ASSERT(0 && "Attempted to serialize value of invalid type.");
	}
}

static void _RdbSaveEntity
(
	RedisModuleIO *rdb,
	const GraphEntity *e
) {
	// Format:
	// #attributes N
	// (name, value type, value) X N 

	const AttributeSet set = GraphEntity_GetAttributes(e);
	uint16_t attr_count = AttributeSet_Count(set);

	RedisModule_SaveUnsigned(rdb, attr_count);

	for(int i = 0; i < attr_count; i++) {
		Attribute_ID attr_id;
		SIValue value = AttributeSet_GetIdx(set, i, &attr_id);
		RedisModule_SaveUnsigned(rdb, attr_id);
		_RdbSaveSIValue(rdb, &value);
	}
}

static void _RdbSaveEdge
(
	RedisModuleIO *rdb,
	const Graph *g,
	const Edge *e,
	int r
) {

	// Format:
	//  edge ID
	//  source node ID
	//  destination node ID
	//  relation type
	//  edge properties

	RedisModule_SaveUnsigned(rdb, ENTITY_GET_ID(e));

	// source node ID
	RedisModule_SaveUnsigned(rdb, Edge_GetSrcNodeID(e));

	// destination node ID
	RedisModule_SaveUnsigned(rdb, Edge_GetDestNodeID(e));

	// relation type
	RedisModule_SaveUnsigned(rdb, r);

	// edge properties
	_RdbSaveEntity(rdb, (GraphEntity *)e);
}

static void _RdbSaveNode_v13
(
	RedisModuleIO *rdb,
	GraphContext *gc,
	GraphEntity *n
) {
	// Format:
	//     ID
	//     #labels M
	//     (labels) X M
	//     #properties N
	//     (name, value type, value) X N */

	// save ID
	EntityID id = ENTITY_GET_ID(n);
	RedisModule_SaveUnsigned(rdb, id);

	// retrieve node labels
	uint l_count;
	NODE_GET_LABELS(gc->g, (Node *)n, l_count);
	RedisModule_SaveUnsigned(rdb, l_count);

	// save labels
	for(uint i = 0; i < l_count; i++) RedisModule_SaveUnsigned(rdb, labels[i]);

	// properties N
	// (name, value type, value) X N
	_RdbSaveEntity(rdb, (GraphEntity *)n);
}

static void _RdbSaveDeletedEntities_v13
(
	RedisModuleIO *rdb,
	GraphContext *gc,
	uint64_t deleted_entities_to_encode,
	uint64_t *deleted_id_list
) {
	// Get the number of deleted entities already encoded.
	uint64_t offset = GraphEncodeContext_GetProcessedEntitiesOffset(gc->encoding_context);

	// Iterated over the required range in the datablock deleted items.
	for(uint64_t i = offset; i < offset + deleted_entities_to_encode; i++) {
		RedisModule_SaveUnsigned(rdb, deleted_id_list[i]);
	}
}

void RdbSaveDeletedNodes_v13
(
	RedisModuleIO *rdb,
	GraphContext *gc,
	uint64_t deleted_nodes_to_encode
) {
	// Format:
	// node id X N

	if(deleted_nodes_to_encode == 0) return;
	// get deleted nodes list
	uint64_t *deleted_nodes_list = Serializer_Graph_GetDeletedNodesList(gc->g);
	_RdbSaveDeletedEntities_v13(rdb, gc, deleted_nodes_to_encode, deleted_nodes_list);
}

void RdbSaveDeletedEdges_v13
(
	RedisModuleIO *rdb,
	GraphContext *gc,
	uint64_t deleted_edges_to_encode
) {
	// Format:
	// edge id X N

	if(deleted_edges_to_encode == 0) return;

	// get deleted edges list
	uint64_t *deleted_edges_list = Serializer_Graph_GetDeletedEdgesList(gc->g);
	_RdbSaveDeletedEntities_v13(rdb, gc, deleted_edges_to_encode, deleted_edges_list);
}

void RdbSaveNodes_v13
(
	RedisModuleIO *rdb,
	GraphContext *gc,
	uint64_t nodes_to_encode
) {
	// Format:
	// Node Format * nodes_to_encode:
	//  ID
	//  #labels M
	//  (labels) X M
	//  #properties N
	//  (name, value type, value) X N

	if(nodes_to_encode == 0) return;
	// get graph's node count
	uint64_t graph_nodes = Graph_NodeCount(gc->g);
	// get the number of nodes already encoded
	uint64_t offset = GraphEncodeContext_GetProcessedEntitiesOffset(gc->encoding_context);

	// get datablock iterator from context,
	// already set to offset by a previous encodeing of nodes, or create new one
	DataBlockIterator *iter = GraphEncodeContext_GetDatablockIterator(gc->encoding_context);
	if(!iter) {
		iter = Graph_ScanNodes(gc->g);
		GraphEncodeContext_SetDatablockIterator(gc->encoding_context, iter);
	}

	for(uint64_t i = 0; i < nodes_to_encode; i++) {
		GraphEntity e;
		e.attributes = (AttributeSet *)DataBlockIterator_Next(iter, &e.id);
		_RdbSaveNode_v13(rdb, gc, &e);
	}

	// check if done encodeing nodes
	if(offset + nodes_to_encode == graph_nodes) {
		DataBlockIterator_Free(iter);
		iter = NULL;
		GraphEncodeContext_SetDatablockIterator(gc->encoding_context, iter);
	}
}

// Auxilary function to encode a multiple edges array,
// while consdirating the allowed number of edges to encode
// returns true if the number of encoded edges has reached the capacity
static void _RdbSaveMultipleEdges
(
	RedisModuleIO *rdb,                  // RDB IO.
	GraphContext *gc,                    // Graph context.
	uint r,                              // Edges relation id.
	EdgeID *multiple_edges_array,        // Multiple edges array (passed by ref).
	uint *multiple_edges_current_index,  // Current index of the array to start encoding from (passed by ref).
	uint64_t *encoded_edges,             // Number of encoded edges in this phase (passed by ref).
	uint64_t edges_to_encode,            // Allowed capacity for encoding edges.
	NodeID src,                          // Edges source node id.
	NodeID dest                          // Edges destination node id.
) {
	uint edgeCount = array_len(multiple_edges_array);

	// define function local variables from passed-by-reference parameters.
	uint i = *multiple_edges_current_index;
	uint encoded_edges_count = *encoded_edges;

	// add edges as long the number of encoded edges is in the allowed range
	// and the array is not depleted
	while(i < edgeCount && encoded_edges_count < edges_to_encode) {
		Edge e;
		EdgeID edgeID = multiple_edges_array[i++];
		e.src_id  = src;
		e.dest_id = dest;
		Graph_GetEdge(gc->g, edgeID, &e);
		_RdbSaveEdge(rdb, gc->g, &e, r);
		encoded_edges_count++;
	}

	// update passed-by-reference parameters
	*encoded_edges = encoded_edges_count;
	*multiple_edges_current_index = i;
}

void RdbSaveEdges_v13
(
	RedisModuleIO *rdb,
	GraphContext *gc,
	uint64_t edges_to_encode
) {
	// Format:
	// Edge format * edges_to_encode:
	//  edge ID
	//  source node ID
	//  destination node ID
	//  relation type
	//  edge properties

	GrB_Info info;
	UNUSED(info);

	if(edges_to_encode == 0) return;

	// get graph's edge count
	uint64_t graph_edges = Graph_EdgeCount(gc->g);

	// get the number of edges already encoded
	uint64_t offset = GraphEncodeContext_GetProcessedEntitiesOffset(gc->encoding_context);

	// count the edges that will be encoded in this phase
	uint64_t encoded_edges = 0;

	// get current relation matrix
	uint r = GraphEncodeContext_GetCurrentRelationID(gc->encoding_context);

	RG_Matrix M = Graph_GetRelationMatrix(gc->g, r, false);

	// get matrix tuple iterator from context
	// already set to the next entry to fetch
	// for previous edge encide or create new one
	RG_MatrixTupleIter *iter = GraphEncodeContext_GetMatrixTupleIterator(gc->encoding_context);
	if(!RG_MatrixTupleIter_is_attached(iter, M)) {
		info = RG_MatrixTupleIter_attach(iter, M);
		ASSERT(info == GrB_SUCCESS);
	}

	// first, see if the last edges encoding stopped at multiple edges array
	EdgeID *multiple_edges_array = GraphEncodeContext_GetMultipleEdgesArray(gc->encoding_context);
	NodeID src = GraphEncodeContext_GetMultipleEdgesSourceNode(gc->encoding_context);
	NodeID dest = GraphEncodeContext_GetMultipleEdgesDestinationNode(gc->encoding_context);
	uint multiple_edges_current_index = GraphEncodeContext_GetMultipleEdgesCurrentIndex(
											gc->encoding_context);
	if(multiple_edges_array) {
		_RdbSaveMultipleEdges(rdb, gc, r, multiple_edges_array,
							  &multiple_edges_current_index,
							  &encoded_edges, edges_to_encode, src, dest);
		// if the multiple edges array filled the capacity of entities allowed
		// to be encoded, finish encoding
		if(encoded_edges == edges_to_encode) {
			goto finish;
		} else {
			// reset the multiple edges context for re-use
			multiple_edges_array = NULL;
			multiple_edges_current_index = 0;
		}
	}

	uint relation_count = Graph_RelationTypeCount(gc->g);
	// write the required number of edges
	while(encoded_edges < edges_to_encode) {
		Edge e;
		EdgeID edgeID;

		// try to get next tuple
		info = RG_MatrixTupleIter_next_UINT64(iter, &src, &dest, &edgeID);

		// if iterator is depleted
		// get new tuple from different matrix or finish encode
		while(info == GxB_EXHAUSTED) {
			// proceed to next relation matrix
			r++;

			// if done iterating over all the matrices, jump to finish
			if(r == relation_count) goto finish;

			// get matrix and set iterator
			M = Graph_GetRelationMatrix(gc->g, r, false);
			info = RG_MatrixTupleIter_attach(iter, M);
			ASSERT(info == GrB_SUCCESS);
			info = RG_MatrixTupleIter_next_UINT64(iter, &src, &dest, &edgeID);
		}
		
		ASSERT(info == GrB_SUCCESS);

		e.src_id = src;
		e.dest_id = dest;
		if(SINGLE_EDGE(edgeID)) {
			Graph_GetEdge(gc->g, edgeID, &e);
			_RdbSaveEdge(rdb, gc->g, &e, r);
			encoded_edges++;
		} else {
			multiple_edges_array = (EdgeID *)(CLEAR_MSB(edgeID));
			_RdbSaveMultipleEdges(rdb, gc, r, multiple_edges_array,
								  &multiple_edges_current_index, &encoded_edges, edges_to_encode, src, dest);
			// if the multiple edges array filled the capacity of entities
			// allowed to be encoded, finish encoding
			if(encoded_edges == edges_to_encode) {
				goto finish;
			} else {
				// reset the multiple edges context for re-use
				multiple_edges_array = NULL;
				multiple_edges_current_index = 0;
			}
		}
	}

finish:
	// check if done encoding edges
	if(offset + edges_to_encode == graph_edges) {
		RG_MatrixTupleIter_detach(iter);
	}

	// update context
	GraphEncodeContext_SetCurrentRelationID(gc->encoding_context, r);
	GraphEncodeContext_SetMutipleEdgesArray(gc->encoding_context, multiple_edges_array,
											multiple_edges_current_index, src, dest);
}
