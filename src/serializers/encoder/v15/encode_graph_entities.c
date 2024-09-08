/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "encode_v15.h"
#include "../../../datatypes/datatypes.h"

// forword decleration
static void _RdbSaveSIValue
(
	SerializerIO rdb,
	const SIValue *v
);

static void _RdbSaveSIArray
(
	SerializerIO rdb,
	const SIValue list
) {
	// saves array as
	// unsigned : array legnth
	// array[0]
	// .
	// .
	// .
	// array[array length -1]

	uint arrayLen = SIArray_Length(list);
	SerializerIO_WriteUnsigned(rdb, arrayLen);
	for(uint i = 0; i < arrayLen; i ++) {
		SIValue value = SIArray_Get(list, i);
		_RdbSaveSIValue(rdb, &value);
	}
}

static void _RdbSaveSIVector
(
	SerializerIO rdb, 
	SIValue v
) {
	// saves a vector
	// unsigned : vector dimension
	// vector[0]
	// .
	// .
	// .
	// vector[vector dimension -1]

	uint32_t dim = SIVector_Dim(v);
	SerializerIO_WriteUnsigned(rdb, dim);

	// get vector elements
	void *elements = SIVector_Elements(v);

	// save individual elements
	float *values = (float*)elements;
	for(uint32_t i = 0; i < dim; i ++) {
		SerializerIO_WriteFloat(rdb, values[i]);
	}
}

static void _RdbSaveSIValue
(
	SerializerIO rdb,
	const SIValue *v
) {
	// Format:
	// SIType
	// Value

	SerializerIO_WriteUnsigned(rdb, v->type);

	switch(v->type) {
		case T_BOOL:
		case T_INT64:
			SerializerIO_WriteSigned(rdb, v->longval);
			break;
		case T_DOUBLE:
			SerializerIO_WriteDouble(rdb, v->doubleval);
			break;
		case T_STRING:
			SerializerIO_WriteBuffer(rdb, v->stringval,
					strlen(v->stringval) + 1);
			break;
		case T_ARRAY:
			_RdbSaveSIArray(rdb, *v);
			break;
		case T_POINT:
			SerializerIO_WriteDouble(rdb, Point_lat(*v));
			SerializerIO_WriteDouble(rdb, Point_lon(*v));
			break;
		case T_VECTOR_F32:
			_RdbSaveSIVector(rdb, *v);
			break;
		case T_NULL:
			break;  // no data beyond type needs to be encoded for NULL
		default:
			ASSERT(0 && "Attempted to serialize value of invalid type.");
	}
}

static void _RdbSaveEntity
(
	SerializerIO rdb,
	const GraphEntity *e
) {
	// Format:
	// #attributes N
	// (name, value type, value) X N 

	const AttributeSet set = GraphEntity_GetAttributes(e);
	uint16_t attr_count = AttributeSet_Count(set);

	SerializerIO_WriteUnsigned(rdb, attr_count);

	for(int i = 0; i < attr_count; i++) {
		AttributeID attr_id;
		SIValue value = AttributeSet_GetIdx(set, i, &attr_id);
		SerializerIO_WriteUnsigned(rdb, attr_id);
		_RdbSaveSIValue(rdb, &value);
	}
}

static void _RdbSaveEdge
(
	SerializerIO rdb,
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

	SerializerIO_WriteUnsigned(rdb, ENTITY_GET_ID(e));

	// source node ID
	SerializerIO_WriteUnsigned(rdb, Edge_GetSrcNodeID(e));

	// destination node ID
	SerializerIO_WriteUnsigned(rdb, Edge_GetDestNodeID(e));

	// relation type
	SerializerIO_WriteUnsigned(rdb, r);

	// edge properties
	_RdbSaveEntity(rdb, (GraphEntity *)e);
}

static void _RdbSaveNode_v15
(
	SerializerIO rdb,
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
	SerializerIO_WriteUnsigned(rdb, id);

	// retrieve node labels
	uint l_count;
	NODE_GET_LABELS(gc->g, (Node *)n, l_count);
	SerializerIO_WriteUnsigned(rdb, l_count);

	// save labels
	for(uint i = 0; i < l_count; i++) SerializerIO_WriteUnsigned(rdb, labels[i]);

	// properties N
	// (name, value type, value) X N
	_RdbSaveEntity(rdb, (GraphEntity *)n);
}

static void _RdbSaveDeletedEntities_v15
(
	SerializerIO rdb,
	GraphContext *gc,
	uint64_t deleted_entities_to_encode,
	uint64_t *deleted_id_list
) {
	// Get the number of deleted entities already encoded.
	uint64_t offset = GraphEncodeContext_GetProcessedEntitiesOffset(gc->encoding_context);

	// Iterated over the required range in the datablock deleted items.
	for(uint64_t i = offset; i < offset + deleted_entities_to_encode; i++) {
		SerializerIO_WriteUnsigned(rdb, deleted_id_list[i]);
	}
}

void RdbSaveDeletedNodes_v15
(
	SerializerIO rdb,
	GraphContext *gc,
	uint64_t deleted_nodes_to_encode
) {
	// Format:
	// node id X N

	if(deleted_nodes_to_encode == 0) return;
	// get deleted nodes list
	uint64_t *deleted_nodes_list = Serializer_Graph_GetDeletedNodesList(gc->g);
	_RdbSaveDeletedEntities_v15(rdb, gc, deleted_nodes_to_encode, deleted_nodes_list);
}

void RdbSaveDeletedEdges_v15
(
	SerializerIO rdb,
	GraphContext *gc,
	uint64_t deleted_edges_to_encode
) {
	// Format:
	// edge id X N

	if(deleted_edges_to_encode == 0) return;

	// get deleted edges list
	uint64_t *deleted_edges_list = Serializer_Graph_GetDeletedEdgesList(gc->g);
	_RdbSaveDeletedEntities_v15(rdb, gc, deleted_edges_to_encode, deleted_edges_list);
}

void RdbSaveNodes_v15
(
	SerializerIO rdb,
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
		_RdbSaveNode_v15(rdb, gc, &e);
	}

	// check if done encodeing nodes
	if(offset + nodes_to_encode == graph_nodes) {
		DataBlockIterator_Free(iter);
		iter = NULL;
		GraphEncodeContext_SetDatablockIterator(gc->encoding_context, iter);
	}
}

void RdbSaveEdges_v15
(
	SerializerIO rdb,
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

	bool depleted;

	if(edges_to_encode == 0) return;

	// get graph's edge count
	uint64_t graph_edges = Graph_EdgeCount(gc->g);

	// get the number of edges already encoded
	uint64_t offset = GraphEncodeContext_GetProcessedEntitiesOffset(gc->encoding_context);

	// count the edges that will be encoded in this phase
	uint64_t encoded_edges = 0;

	// get current relation matrix
	uint r = GraphEncodeContext_GetCurrentRelationID(gc->encoding_context);

	NodeID src;
	NodeID dest;

	// get matrix tuple iterator from context
	// already set to the next entry to fetch
	// for previous edge encide or create new one
	uint relation_count = Graph_RelationTypeCount(gc->g);
	RelationIterator *iter = GraphEncodeContext_GetMatrixTupleIterator(gc->encoding_context);
	if(r < relation_count) {
		Graph_GetRelationMatrix(gc->g, r, false);
		Graph_GetMultiEdgeRelationMatrix(gc->g, r);

		if(!RelationIterator_is_attached(iter, gc->g->relations[r])) {
			RelationIterator_AttachSourceRange(iter, gc->g->relations[r], 0, UINT64_MAX, false);
		}
	}

	// write the required number of edges
	while(encoded_edges < edges_to_encode) {
		Edge e;
		EdgeID edgeID;

		// try to get next tuple
		depleted = !RelationIterator_next(iter, &src, &dest, &edgeID);

		// if iterator is depleted
		// get new tuple from different matrix or finish encode
		while(depleted) {
			// proceed to next relation matrix
			r++;

			// if done iterating over all the matrices, jump to finish
			if(r == relation_count) goto finish;

			// get matrix and set iterator
			Graph_GetRelationMatrix(gc->g, r, false);
			Graph_GetMultiEdgeRelationMatrix(gc->g, r);
			RelationIterator_AttachSourceRange(iter, gc->g->relations[r], 0, UINT64_MAX, false);
			depleted = !RelationIterator_next(iter, &src, &dest, &edgeID);
		}
		
		ASSERT(!depleted);

		e.src_id = src;
		e.dest_id = dest;
		Graph_GetEdge(gc->g, edgeID, &e);
		_RdbSaveEdge(rdb, gc->g, &e, r);
		encoded_edges++;
	}

finish:
	// check if done encoding edges
	if(offset + edges_to_encode == graph_edges) {
		*iter = (RelationIterator){0};
	}

	// update context
	GraphEncodeContext_SetCurrentRelationID(gc->encoding_context, r);
}