/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "encode_v16.h"
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

// encode a single node
static void _RdbSaveNode_v16
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
	for(uint i = 0; i < l_count; i++) {
		SerializerIO_WriteUnsigned(rdb, labels[i]);
	}

	// properties N
	// (name, value type, value) X N
	_RdbSaveEntity(rdb, (GraphEntity *)n);
}

// encode deleted entities IDs
static void _RdbSaveDeletedEntities_v16
(
	SerializerIO rdb,
	GraphContext *gc,
	uint64_t n,
	uint64_t offset,
	uint64_t *deleted_id_list
) {
	ASSERT(n > 0);

	// iterated over the required range in the datablock deleted items
	for(uint64_t i = offset; i < offset + n; i++) {
		SerializerIO_WriteUnsigned(rdb, deleted_id_list[i]);
	}
}

// encode deleted node IDs
// return number of elements encoded
uint64_t RdbSaveDeletedNodes_v16
(
	SerializerIO rdb,  // RDB
	GraphContext *gc,  // graph context
	uint64_t offset,   // offset
	uint64_t n         // number of deleted nodes to encode
) {
	// Format:
	// node id X N

	ASSERT(n > 0);

	// get deleted nodes list
	uint64_t *deleted_nodes_list = Serializer_Graph_GetDeletedNodesList(gc->g);
	_RdbSaveDeletedEntities_v16(rdb, gc, n, offset, deleted_nodes_list);

	return n;
}

// encode deleted edges IDs
// return number of elements encoded
uint64_t RdbSaveDeletedEdges_v16
(
	SerializerIO rdb,  // RDB
	GraphContext *gc,  // graph context
	uint64_t offset,   // offset
	uint64_t n         // number of deleted edges to encode
) {
	// Format:
	// edge id X N

	ASSERT(n > 0);

	// get deleted edges list
	uint64_t *deleted_edges_list = Serializer_Graph_GetDeletedEdgesList(gc->g);
	_RdbSaveDeletedEntities_v16(rdb, gc, n, offset, deleted_edges_list);

	return n;
}

// encode nodes
// returns number of nodes encoded
uint64_t RdbSaveNodes_v16
(
	SerializerIO rdb,  // RDB
	GraphContext *gc,  // graph context
	uint64_t offset,   // iterator offset
	uint64_t n         // number of nodes to encode
) {
	// Format:
	// Node Format * nodes_to_encode:
	//  ID
	//  #labels M
	//  (labels) X M
	//  #properties N
	//  (name, value type, value) X N

	ASSERT(n != 0);

	// get graph's node count
	uint64_t graph_nodes = Graph_NodeCount(gc->g);

	// get datablock iterator from context,
	// already set to offset by a previous encodeing of nodes, or create new one
	DataBlockIterator *iter =
		GraphEncodeContext_GetDatablockIterator(gc->encoding_context);
	if(!iter) {
		iter = Graph_ScanNodes(gc->g);
		GraphEncodeContext_SetDatablockIterator(gc->encoding_context, iter);
	}

	for(uint64_t i = 0; i < n; i++) {
		GraphEntity e;
		e.attributes = (AttributeSet *)DataBlockIterator_Next(iter, &e.id);
		_RdbSaveNode_v16(rdb, gc, &e);
	}

	// check if done encodeing nodes
	if(offset + n == graph_nodes) {
		DataBlockIterator_Free(iter);
		iter = NULL;
		GraphEncodeContext_SetDatablockIterator(gc->encoding_context, iter);
	}

	return n;
}

// encode edges
// returns number of encoded edges.
uint64_t RdbSaveEdges_v16
(
	SerializerIO rdb,  // RDB
	GraphContext *gc,  // graph context
	uint64_t offset,   // offset
	uint64_t n         // number of edges to encode
) {
	// Format:
	// Edge format:
	//  edge ID
	//  source node ID
	//  destination node ID
	//  relation type
	//  edge properties
	
	ASSERT(n > 0);

	// number of relationship matrices in the graph
	int relations_count = Graph_RelationTypeCount(gc->g);

	// count the edges that will be encoded in this phase
	uint64_t encoded_edges = 0;

	Delta_Matrix R;       // current relation matrix
	Edge         e;       // current edge
	NodeID       src;     // edge source node id
	NodeID       dest;    // edge destination node id
	EdgeID       edgeID;  // edge id

	// get matrix tuple iterator from context
	// already set to the next entry to fetch
	// for previous edge encide or create new one
	TensorIterator *iter =
		GraphEncodeContext_GetMatrixTupleIterator(gc->encoding_context);

	// get current relation matrix
	uint r = GraphEncodeContext_GetCurrentRelationID(gc->encoding_context);

	// first relationship matrix
	if(r == 0) {
		R = Graph_GetRelationMatrix(gc->g, r, false);

		// attach iterator if not already attached
		if(!TensorIterator_is_attached(iter, R)) {
			TensorIterator_ScanRange(iter, R, 0, UINT64_MAX, false);
		}
	}

	//--------------------------------------------------------------------------
	// encode edges
	//--------------------------------------------------------------------------

	while(encoded_edges < n) {
		// try to get next tuple
		bool depleted = !TensorIterator_next(iter, &src, &dest, &edgeID);

		// if iterator is depleted
		// get new tuple from different matrix or finish encoding
		while(depleted) {
			// if done iterating over all the matrices, jump to finish
			if(++r == relations_count) goto finish;

			// get matrix and set iterator
			R = Graph_GetRelationMatrix(gc->g, r, false);

			TensorIterator_ScanRange(iter, R, 0, UINT64_MAX, false);
			depleted = !TensorIterator_next(iter, &src, &dest, &edgeID);
		}
		
		ASSERT(!depleted);

		// set edge endpoints
		e.src_id  = src;
		e.dest_id = dest;

		// get edge attribute set
		bool edge_found = Graph_GetEdge(gc->g, edgeID, &e);
		ASSERT(edge_found == true);

		// encode edge
		_RdbSaveEdge(rdb, gc->g, &e, r);

		encoded_edges++;
	}

	// we want to stop encoding right at the begining of a new row
	// and so continue encoding edges until a new row is encountered
	NodeID prev_src = e.src_id;
	while(TensorIterator_next(iter, &src, &dest, &edgeID)) {
		// same row
		if(src == prev_src) {
			// set edge endpoints
			e.src_id  = src;
			e.dest_id = dest;

			// get edge attribute set
			bool edge_found = Graph_GetEdge(gc->g, edgeID, &e);
			ASSERT(edge_found == true);

			// encode edge
			_RdbSaveEdge(rdb, gc->g, &e, r);

			encoded_edges++;
		} else {
			// a new row encountered
			// reset iterator to the begining of the row
			TensorIterator_ScanRange(iter, R, src, UINT64_MAX, false);
			break;
		}
	}

finish:
	// check if done encoding edges
	if(offset + encoded_edges == Graph_EdgeCount(gc->g)) {
		*iter = (TensorIterator){0};
	}

	// update context
	GraphEncodeContext_SetCurrentRelationID(gc->encoding_context, r);

	return encoded_edges;
}

